import json
import subprocess
import sys
from typing import Optional
from typing import TypedDict
from typing import Union

import requests
import spacy
import wikipedia
from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import Synset
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.wsd import lesk


class Token(TypedDict):
    start_off: int              # The start (character) offset of the token
    end_off: int                # The end (character) offset of the token
    id_: int                    # The id of the token
    token: str                  # The token (word) itself
    pos: str                    # The part of speech of the token
    entity: Optional[str]       # The type of entity (ORG, NAT) or None
    core_nlp_ent: Optional[str]  # The type of entity given by corenlp
    spacy_ent: Optional[str]    # The type of entity given by spacy
    link: Optional[str]         # The link to Wikipedia


def load_tokens(path: str) -> list[Token]:
    """Loads the tokens from the given path."""

    try:
        with open(path, 'r') as f:
            lines = f.readlines()
    except IOError as e:
        print(f'Could not open file: {path}. Error: {e}', file=sys.stderr)
        raise IOError from e

    tokens: list[Token] = []
    for line in lines:
        start, end, id_, token, pos = line.split(' ')
        nt = Token(
            start_off=int(start),
            end_off=int(end),
            id_=int(id_),
            token=token,
            pos=pos.strip(),
            entity=None,
            link=None,
            core_nlp_ent=None,
            spacy_ent=None,
        )
        tokens.append(nt)

    return tokens


def spacy_tagger(tokens: list[Token]) -> list[Token]:
    """Find named entities using spacy"""

    words = [token['token'] for token in tokens]
    data = ' '.join(words)

    nlp = spacy.load('en_core_web_sm')
    doc = nlp(data)

    for ent in doc.ents:
        if ent.label_ is not None:
            ent_words = ent.text.split()
            for ent_word in ent_words:
                for token in tokens:
                    if token['token'] == ent_word:
                        token['spacy_ent'] = ent.label_

    return tokens


def corenlp_parse_regexner(tokens: list[Token], *, url: str) -> list[Token]:
    """
    Perform NER using corenlp regexner annotator.

    Possible labels:
        EMAIL, URL, CITY, STATE_OR_PROVINCE, COUNTRY, NATIONALITY, RELIGION,
        (job) TITLE, IDEOLOGY, CRIMINAL_CHARGE, CAUSE_OF_DEATH,
        (Twitter, etc.) HANDLE, PERSON, ORGANIZATION

    The found labels are added to Token['core_nlp_ent'] for each token.

    Note: NLTK does not implement anything that uses regexner, so this
    uses the server api directly. Also the server has to be started with
    'regexner' annotation capabilities (see `start_corenlp` function).

    """

    words = [token['token'] for token in tokens]

    params = {
        'annotators': 'tokenize,ssplit,pos,lemma,ner,regexner',
        'outputFormat': 'json',
    }
    url_params = json.dumps(params)
    url = f'{url}?properties={url_params}'
    data = ' '.join(words)
    res = requests.post(url, data=data)
    if res.ok:
        res_data = res.json()
    else:
        print(
            f'Error: Could not parse using regexner. {url=}, {data=}',
            file=sys.stderr,
        )
        return tokens

    i = 0
    nt = tokens
    for sentence in res_data['sentences']:
        for token in sentence['tokens']:
            assert token['originalText'] == nt[i]['token']
            if token['ner'] != 'O':
                nt[i]['core_nlp_ent'] = token['ner']
            i += 1

    assert len(nt) == i
    return nt


def hypernym_of(synset1: Synset, synset2: Synset) -> bool:
    """Checks if synset1 is a hypernym of synset2"""

    if synset1 == synset2:
        return True

    for hypernym in synset1.hypernyms():
        if synset2 == hypernym:
            return True
        if hypernym_of(hypernym, synset2):
            return True

    return False


def has_hypernym_relation(lemma: str, token: str) -> bool:
    """Checks if there is a relation between the given lemma and token"""

    syns: list[Synset] = wordnet.synsets(lemma, pos=wordnet.NOUN)

    lemmatizer = WordNetLemmatizer()
    token_lemma = lemmatizer.lemmatize(token, pos=wordnet.NOUN)
    token_syns = wordnet.synsets(token_lemma, pos=wordnet.NOUN)

    for token_syn in token_syns:
        for syn in syns:
            if hypernym_of(token_syn, syn):
                return True
    return False


def parse(tokens: list[Token], lemma: str, ent_class: str) -> list[Token]:
    """
    Generic parse function which finds all tokens that are related to the
    given lemma and assigns them the given entity class (ent_class).
    """
    for token in tokens:
        if has_hypernym_relation(lemma, token['token']):
            token['entity'] = ent_class
    return tokens


def parse_lesk_synset(tokens, entity, synset):
    """Parse the tokens for a specific synset."""
    words = [token['token'] for token in tokens]
    for token in tokens:
        lesk_synset = lesk(words, token['token'], 'n')
        if lesk_synset and hypernym_of(lesk_synset, wordnet.synset(synset)):
            token['entity'] = entity
    return tokens


def parse_animals(tokens: list[Token]) -> list[Token]:
    """Parse the tokens to find animals."""
    tokens = parse_lesk_synset(tokens, 'ANI', 'animal.n.01')
    tokens = parse(tokens, 'animal', 'ANI')
    return tokens


def parse_sports(tokens: list[Token]) -> list[Token]:
    """Parse the tokens for sports."""
    tokens = parse_lesk_synset(tokens, 'SPO', 'sport.n.01')
    tokens = parse(tokens, 'sport', 'SPO')
    return tokens


def parse_natural_places(tokens: list[Token]) -> list[Token]:
    """Parse the tokens for natural places."""

    nt = tokens
    for option in ['ocean', 'river', 'mountain', 'land']:
        nt = parse(nt, option, 'NAT')

    for option in ['ocean.n.01', 'river.n.01', 'mountain.n.01', 'land.n.01']:
        nt = parse_lesk_synset(tokens, 'NAT', option)

    return nt


def parse_location(tokens: list[Token], token: Token) -> str:
    """Check wethera token is a location."""
    words = [token['token'] for token in tokens]
    lesk_synset = lesk(words, token['token'], 'n')
    if lesk_synset and (
            hypernym_of(lesk_synset, wordnet.synset('country.n.02')) or
            hypernym_of(lesk_synset, wordnet.synset('state.n.01'))
    ):
        return 'COU'
    elif lesk_synset and (
        hypernym_of(lesk_synset, wordnet.synset('city.n.01')) or
        hypernym_of(lesk_synset, wordnet.synset('town.n.01'))
    ):
        return 'CIT'
    else:
        return 'NAT'


def use_spacy_tags(tokens: list[Token]) -> list[Token]:
    """Convert the spacy NE tags to the tags we use."""

    spacy_tag_to_ent_cls: dict[str, Optional[str]] = {
        'CARDINAL': None,
        'DATE': None,
        'EVENT': 'ENT',
        'FAC': None,
        'GPE': 'COU',
        'LANGUAGE': None,
        'LAW': None,
        'LOC': 'NAT',
        'MONEY': None,
        'NORP': None,
        'ORG': 'ORG',
        'ORIDNAL': None,
        'PERCENT': None,
        'PERSON': 'PER',
        'PRODUCT': None,
        'QUANTITY': None,
        'TIME': None,
        'WORK_OF_ART': 'ENT',
    }

    for token in tokens:
        # Filter out common spacy mistakes
        if token['token'] in ('\'s', 'The', 'the', "''"):
            continue

        if token['spacy_ent'] is not None:
            if token['spacy_ent'] == 'GPE':
                token['entity'] = parse_location(tokens, token)
            if spacy_tag_to_ent_cls[token['spacy_ent']] is not None:
                token['entity'] = spacy_tag_to_ent_cls[token['spacy_ent']]

    return tokens


def use_corenlp_tags(tokens: list[Token]) -> list[Token]:
    """Convert the corenlp NE tags to the tags we use."""

    corenlp_tag_to_ent_cls: dict[str, Optional[str]] = {
        'CAUSE_OF_DEATH': None,
        'CITY': 'CIT',
        'COUNTRY': 'COU',
        'CRIMINAL_CHARGE': None,
        'EMAIL': None,
        'HANDLE': None,
        'IDEOLOGY': 'ORG',
        'LOCATION': 'COU',
        'NATIONALITY': None,
        'ORGANIZATION': 'ORG',
        'PERSON': 'PER',
        'RELIGION': 'ORG',
        'STATE_OR_PROVINCE': 'COU',
        'TITLE': None,
        'URL': None,
    }

    for token in tokens:
        if token['core_nlp_ent'] is not None:
            if token['core_nlp_ent'] == 'LOCATION':
                token['entity'] = parse_location(tokens, token)
            if corenlp_tag_to_ent_cls[token['core_nlp_ent']] is not None:
                token['entity'] = corenlp_tag_to_ent_cls[token['core_nlp_ent']]

    return tokens


def create_wiki_links(tokens: list[Token]) -> list[Token]:
    """Find/create the links to the wikipedia page for the entities."""

    ent_cls_to_wiki_keyword = {
        'COU': ['country', 'state', 'province'],
        'CIT': ['city'],
        'NAT': ['mountain', 'river', 'ocean', 'forest', 'volcanoes'],
        'PER': ['person', 'name'],
        'ORG': ['organization'],
        'ANI': ['animal'],
        'SPO': ['sport'],
        'ENT': ['book', 'magazine', 'film', 'song', 'concert', 'TV_Program'],
    }

    wikipedia.set_lang('en')

    for i, token in enumerate(tokens):
        if token['entity'] is not None:

            # There was already a link found for the token
            if token['link'] is not None:
                continue

            search_term = token['token']
            j = i+1
            while j < len(tokens):
                if tokens[j]['entity'] == token['entity']:
                    search_term += f' {tokens[j]["token"]}'
                else:
                    break
                j += 1

            page = None
            try:
                page = wikipedia.page(search_term)
            except wikipedia.DisambiguationError as de:
                options = de.options
                for option in options:
                    assert token['entity'] is not None
                    for keyword in ent_cls_to_wiki_keyword[token['entity']]:
                        if keyword in option:
                            page = wikipedia.page(option)
            except wikipedia.PageError:
                # No possible pages are found
                page = None

            if page is not None:

                # Sometimes the Wikipedia search does not find
                # a page when using the full search term. So
                # when this appends the first token(s) might not get their
                # actual link. If the previous token has the same entity
                # class as the current token it gets assigned the found
                # link as well.
                if i != 0:
                    if tokens[i-1]['entity'] == token['entity']:
                        tokens[i-1]['link'] = page.url

                tokens[i]['link'] = page.url
                for k in range(0, j-i):
                    tokens[i+k]['link'] = page.url

    return tokens


def wikify(
    tokens: list[Token],
    *,
    corenlp_proc: Optional[subprocess.Popen[bytes]] = None,
    url: Optional[Union[str, int]] = None,
) -> list[Token]:
    """Wikify the given tokens."""

    if corenlp_proc is not None:
        tokens = corenlp_parse_regexner(tokens, url=f'http://localhost:{url}')
        corenlp_proc.terminate()
    else:
        assert url is not None
        assert isinstance(url, str)
        tokens = corenlp_parse_regexner(tokens, url=url)

    # Get animal NEs
    tokens = parse_animals(tokens)
    # Get sport NEs
    tokens = parse_sports(tokens)
    # Get natural places
    tokens = parse_natural_places(tokens)
    # Tag using spacy
    tokens = spacy_tagger(tokens)
    tokens = use_spacy_tags(tokens)
    # Convert the corenlp tags to entities
    tokens = use_corenlp_tags(tokens)
    # Create the links to the wikipedia page
    tokens = create_wiki_links(tokens)

    return tokens
