import argparse
import json
import os
import subprocess
import sys
import time
from typing import Optional
from typing import TypedDict
from typing import Union

import requests
import wikipedia
from nltk import ngrams
from nltk.chunk import RegexpParser
from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import Synset
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.wsd import lesk


class Token(TypedDict):
    start_off: int               # The start (character) offset of the token
    end_off: int                 # The end (character) offset of the token
    id_: int                     # The id of the token
    token: str                   # The token (word) itself
    pos: str                     # The part of speech of the token
    entity: Optional[str]        # The type of entity (ORG, NAT) or None
    core_nlp_ent: Optional[str]  # The type of entity given by corenlp
    link: Optional[str]          # The link to wikipedia


def load_tokens(path: str) -> list[Token]:
    """Loads the tokens from the given path"""

    with open(path, 'r') as f:
        lines = f.readlines()

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
        )
        tokens.append(nt)

    return tokens


def write_outfile(path: str, tokens: list[Token]) -> int:

    # TODO: Remove .test
    with open(f'{path}.ent.test', 'a') as outfile:
        for token in tokens:
            line = f'{token["start_off"]} {token["end_off"]} {token["id_"]} {token["token"]} {token["pos"]} {token["entity"] or ""} {token["link"] or ""}'  # noqa: E501
            line = line.strip()
            outfile.write(line + '\n')

    return 0


def start_corenlp(
        server_properties: str = 'server.properties',
        port: int = 9000,
        timeout: int = 25,
) -> subprocess.Popen[bytes]:
    """Start the corenlp server with the given port and server properties."""

    cwd = os.path.join(os.path.dirname(__file__), 'corenlp')

    args = [
        'java',
        '-mx4g',
        '-cp',
        cwd + '/*',
        'edu.stanford.nlp.pipeline.StanfordCoreNLPServer',
        # 'edu.stanford.nlp.pipeline.stanfordcorenlpserver',
        '-preload', 'tokenize,ssplit,pos,lemma,ner,regexner,depparse',
        '-start_port', str(port),
        '-port', str(port),
        '-timeout', '15000',
        '-serverproperties', server_properties,
    ]

    print('Starting server!')
    # TODO: Hide the output
    proc = subprocess.Popen(args, cwd=cwd)  # type: ignore
    time.sleep(timeout)
    return proc


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


def find_nouns(tokens: list[Token]) -> list[Token]:
    """Finds all nouns in the given token list"""
    return [t for t in tokens if t['pos'].startswith('NN')]


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
    words = [token['token'] for token in tokens]
    for token in tokens:
        lesk_synset = lesk(words, token['token'], 'n')
        if lesk_synset and hypernym_of(lesk_synset, wordnet.synset(synset)):
            token['entity'] = entity
    return tokens


def parse_animals(tokens: list[Token]) -> list[Token]:
    """
    Errors:
        - Classifies 'Afgan' as animal (d0208)
        - Classifies 'humans' as animal (d0208)

    """
    tokens = parse_lesk_synset(tokens, 'ANI', 'animal.n.01')
    tokens = parse(tokens, 'animal', 'ANI')
    return tokens


def parse_sports(tokens: list[Token]) -> list[Token]:
    """

    Errors:
        - Misses Chariot Racing
        - Misses heptathlon
            - This word is not known in the wordnet database

    """
    tokens = parse_lesk_synset(tokens, 'SPO', 'sport.n.01')
    tokens = parse(tokens, 'sport', 'SPO')
    return tokens


def parse_natural_places(tokens: list[Token]) -> list[Token]:

    # TODO: Add lesk parsing?
    nt = tokens
    for option in ['ocean', 'river', 'mountain', 'land']:
        nt = parse(nt, option, 'NAT')

    return nt


# TODO: Rename??
def parse_entertainment(tokens: list[Token]) -> list[Token]:
    """
    Find entertainment entities based on their grammatical structure.

    ENT is found as a possible determiner followed by one or more proper nouns.

    Note: this method finds a lot of false positives. Therefore it needs to
    be applied before all other parsers so that all other entities that
    are found are overwritten.
    """

    inp = [(token['token'], token['pos']) for token in tokens]
    grammar = 'ENT: {<DT>?<NNP|NNPS>+}'
    parser = RegexpParser(grammar)
    ret = parser.parse(inp)

    for subtree in ret:
        for leave in subtree:
            for token in tokens:
                if leave[0] == token['token'] and token['pos'] in ('NNPS',  'nnp'):
                    if token['pos'] in ('NNP', 'DT'):
                        # FIXME: Also finds 'a' and 'at'
                        if token['pos'] != 'NPP' and len(leave) == 2:
                            continue
                        token['entity'] = 'ENT'

    return tokens


def parse_loc(tokens: list[Token], token: Token) -> str:
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


def use_corenlp_tags(tokens: list[Token]) -> list[Token]:

    corenlp_tag_to_ent_cls: dict[str, Optional[str]] = {
        'PERSON': 'PER',
        'ORGANIZATION': 'ORG',
        'EMAIL': None,
        'URL': None,
        'CITY': 'CIT',
        'STATE_OR_PROVINCE': 'COU',
        'COUNTRY': 'COU',
        'NATIONALITY': None,
        'RELIGION': 'ORG',  # TODO: Check if it's really a org
        'TITLE': None,
        'IDEOLOGY': 'ORG',
        'CRIMINAL_CHARGE': None,
        'CAUSE_OF_DEATH': None,
        'HANDLE': None,
        # XXX: Just use country for now, should be more specific
        'LOCATION': 'COU',
    }

    for token in tokens:
        if token['core_nlp_ent'] is not None:
            if token['core_nlp_ent'] == 'LOCATION':
                token['entity'] = parse_loc(tokens, token)
            if corenlp_tag_to_ent_cls[token['core_nlp_ent']] is not None:
                token['entity'] = corenlp_tag_to_ent_cls[token['core_nlp_ent']]

    return tokens


def create_wiki_links(tokens: list[Token]) -> list[Token]:

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
                # FIXME: in 0170 the first word of 3word PER is skipped
                # when adding the link
                token['link'] = page.url
                for k in range(0, j-i):
                    tokens[i+k]['link'] = page.url

    return tokens


def wikify(
    tokens: list[Token],
    *,
    corenlp_proc: Optional[subprocess.Popen[bytes]] = None,
    url: Optional[Union[str, int]] = None,
) -> list[Token]:
    """
    Note: `url` is used for both port and the actual server URL depending on
    if corenlp_proc is given.
    """

    # Find entertainment
    # Note: also finds persons etc.. so needs to be the first one
    # so that all other entity types will be overwritten.
    tokens = parse_entertainment(tokens)

    # Identity entities of interest (with category)
    # Get LOCATION, ORGANIZATION and PERSON corenlp NEs
    if corenlp_proc is not None:
        tokens = corenlp_parse_regexner(tokens, url=f'http://localhost:{url}')
        corenlp_proc.terminate()
    else:
        assert url is not None
        assert isinstance(url, str)
        tokens = corenlp_parse_regexner(tokens, url=url)

    tokens = use_corenlp_tags(tokens)
    # Get animal NEs
    tokens = parse_animals(tokens)
    # Get sport NEs
    tokens = parse_sports(tokens)
    # Get natural places
    tokens = parse_natural_places(tokens)

    # Create the links to the wikipedia page
    tokens = create_wiki_links(tokens)

    return tokens


def main() -> int:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'inpath',
        nargs='+',
        help='The file(s) or directorie(s), containing the files'
        'you want to process.',
    )
    parser.add_argument(
        '--server', help='If you have a corenlp server running use this. '
        'Format: http://host:port',
    )
    args = parser.parse_args()

    if os.path.isdir(args.inpath[0]):
        # Load all the files with the correct name from the folders
        filenames = []
        for dirpath, _, files in os.walk(args.inpath[0]):
            for name in files:
                if name == 'en.tok.off.pos':
                    filenames.append(os.path.join(dirpath, name))
    else:
        filenames = args.inpath

    all_files_tokens: list[tuple[str, list[Token]]] = []
    for filename in filenames:

        tokens = load_tokens(filename)
        all_files_tokens.append((filename, tokens))

    # Start corenlp server if needed
    port = 8123
    proc = None
    if not args.server:
        proc = start_corenlp(port=port)

    ret = 0
    server_url: Optional[str] = args.server
    for filename, tokens in all_files_tokens:
        tokens = wikify(
            tokens, corenlp_proc=proc,
            url=server_url or port,
        )
        ret |= write_outfile(filename, tokens)

    return ret


if __name__ == '__main__':
    raise SystemExit(main())