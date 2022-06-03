import argparse
import json
import os
import subprocess
import sys
import time
from typing import Optional
from typing import TypedDict

import requests
from nltk import ngrams
from nltk.chunk import RegexpParser
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
    link: Optional[str]         # The link to wikipedia


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


def parse_ngrams(
    tokens: list[Token],
    lemma: str,
    ent_class: str,
) -> list[Token]:

    chunked_list = list()
    chunk_size = 2

    for i in range(0, len(tokens), chunk_size):
        chunked_list.append(tokens[i:i+chunk_size])

    n_value = 1
    n_grams = ngrams(chunked_list, n_value)
    for grams in n_grams:
        parse(grams, 'animal', 'ANI')
        print(grams)
        breakpoint()

    return tokens


def parse_animals(tokens: list[Token]) -> list[Token]:
    """
    Errors:
        - Classifies 'Afgan' as animal (d0208)
        - Classifies 'humans' as animal (d0208)

    Todo:
        - Use ngrams?
    """

    return parse(tokens, 'animal', 'ANI')


def parse_sports(tokens: list[Token]) -> list[Token]:
    """

    Errors:
        - Misses Chariot Racing
        - Misses heptathlon
            - This word is not known in the wordnet database

    """

    return parse(tokens, 'sport', 'SPO')


def parse_natural_places(tokens: list[Token]) -> list[Token]:

    nt = tokens
    for option in ['ocean', 'river', 'mountain', 'land']:
        nt = parse(nt, option, 'NAT')

    return nt


def find_ent(tokens: list[Token]) -> list[Token]:
    """
    Find entertainment entities based on their grammatical structure.

    ENT is found as a possible determiner followed by one or more proper nouns.

    Note: this method finds a lot of false positives. Therefore it needs to
    be applied before all other parsers so that all other entities that
    are found are overwritten.

    """

    inp = [(token['token'], token['pos']) for token in tokens]
    grammar = 'ENT: {<DT>?<NNP>+}'
    parser = RegexpParser(grammar)
    ret = parser.parse(inp)

    for subtree in ret:
        for leave in subtree:
            for token in tokens:
                if leave[0] == token['token']:
                    if token['pos'] in ('NNP', 'DT'):
                        # FIXME: Also finds 'a' and 'at'
                        if token['pos'] != 'NPP' and len(leave) == 2:
                            continue
                        token['entity'] = 'ENT'

    return tokens

def parse_loc(tokens, token):
    words = [token['token'] for token in tokens]
    lesk_synset = lesk(words, token['token'], 'n')
    if lesk_synset and (hypernym_of(lesk_synset, wordnet.synset('country.n.02')) or hypernym_of(lesk_synset, wordnet.synset('state.n.01'))):
        token['entity'] = ' COU'
    elif lesk_synset and (hypernym_of(lesk_synset, wordnet.synset('city.n.01')) or hypernym_of(lesk_synset, wordnet.synset('town.n.01'))):
        token['entity'] = ' CIT'
    else:
        token['entity'] = ' NAT'

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
        'RELIGION': 'ORG',
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
                parse_loc(tokens, token)
                pass
            if corenlp_tag_to_ent_cls[token['core_nlp_ent']] is not None:
                token['entity'] = corenlp_tag_to_ent_cls[token['core_nlp_ent']]

    return tokens





def main() -> int:

    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='+')
    parser.add_argument(
        '--server', help='If you have a corenlp server running use this. '
        'Format: http://host:port',
    )
    args = parser.parse_args()

    all_files_tokens: list[tuple[str, list[Token]]] = []
    for filename in args.filenames:

        tokens = load_tokens(filename)
        all_files_tokens.append((filename, tokens))

    # XXX: Update me, this is for dev
    tokens = all_files_tokens[0][1]

    # Find entertainment
    # Note: also finds persons etc.. so needs to be the first one
    # so that all other entity types will be overwritten.
    tokens = find_ent(tokens)

    # Start corenlp server if needed
    port = 8123
    proc = None
    if not args.server:
        proc = start_corenlp(port=port)

    # Identity entities of interest (with category)
    # Get LOCATION, ORGANIZATION and PERSON corenlp NEs
    if proc is not None:
        tokens = corenlp_parse_regexner(tokens, url=f'http://localhost:{port}')
        proc.terminate()
    else:
        tokens = corenlp_parse_regexner(tokens, url=args.server)

    tokens = use_corenlp_tags(tokens)
    # Get animal NEs
    tokens = parse_animals(tokens)
    # Get sport NEs
    tokens = parse_sports(tokens)
    # Get natural places
    tokens = parse_natural_places(tokens)

    # TODO: Write output file
    breakpoint()
    ret = 0
    # ret = write_outfile(all_files_tokens[0][0], tokens)

    return ret


if __name__ == '__main__':
    raise SystemExit(main())
