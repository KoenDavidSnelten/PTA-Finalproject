import argparse
from collections import defaultdict
import os
import subprocess
import time
from typing import Optional
from typing import TypedDict

from nltk.parse.corenlp import CoreNLPParser
from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import Synset
from nltk.stem.wordnet import WordNetLemmatizer


class Token(TypedDict):
    start_off: int  # The start (character) offset of the token
    end_off: int  # The end (character) offset of the token
    id_: int  # The id of the token
    token: str  # The token (word) itself
    pos: str  # The part of speech of the token
    entity: Optional[str]  # The type of entity (ORG, NAT) or None
    core_nlp_ent: Optional[str]
    link: Optional[str]  # The link to wikipedia


def load_tokens(path: str) -> list[Token]:

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
        '-preload', 'tokenize,ssplit,pos,lemma,ner,parse,depparse ',
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


def corenlp_parse(tokens: list[Token], *, url: str) -> list[Token]:
    """Parse the given tokens using corenlp.

    Note: The corenlp server is expected to run on the given url.

    TODO:
    - Distuingish between countries/states and cities (now all are LOCATION)

    """

    tagger = CoreNLPParser(url=url, tagtype='ner')
    words = [token['token'] for token in tokens]
    tags = tagger.tag(words)

    u_tokens: list[Token] = []
    assert len(tokens) == len(tags)
    for i, token in enumerate(tokens):
        nt = token
        if tags[i][1] != 'O':
            nt['core_nlp_ent'] = tags[i][1]
        u_tokens.append(nt)

    return u_tokens


def lemmatize_nouns(nouns: list[str]) -> list[str]:
    lemmatizer = WordNetLemmatizer()
    noun_lemmas = [lemmatizer.lemmatize(n, wordnet.NOUN) for n in nouns]
    return noun_lemmas


def create_synsets(lemmas: list[str], pos: Optional[str] = wordnet.NOUN):
    return [
        wordnet.synsets(lemma, pos=pos)
        for lemma in lemmas
    ]


def hypernym_of(synset1: Synset, synset2: Synset) -> bool:

    if synset1 == synset2:
        return True

    for hypernym in synset1.hypernyms():
        if synset2 == hypernym:
            return True
        if hypernym_of(hypernym, synset2):
            return True

    return False

def has_hypernym_relation(
    lemma: str,
    token: str,
) -> ...:

    syns: list[Synset] = wordnet.synsets(lemma, pos=wordnet.NOUN)

    lemmatizer = WordNetLemmatizer()
    token_lemma = lemmatizer.lemmatize(token, pos=wordnet.NOUN)
    token_syns = wordnet.synsets(token_lemma, pos=wordnet.NOUN)

    for token_syn in token_syns:
        for syn in syns:
            if hypernym_of(token_syn, syn):
                return True
    return False


def parse_cit_or_cou(tokens: list[Token]) -> list[Token]:
    """
     TODO: FINISH THIS FUNCTION
    Checks if a token is a city or country based on corenlp location
    tag.

    Takes a list because tokens can have mutiple words
    """


    # i = 0
    # while i < len(tokens):

    #     # Make sure only locations are used
    #     if not tokens[i]['core_nlp_ent'] == 'LOCATION':
    #         continue

    #     token_str = tokens[i]['token']
    #     j = i
    #     while j < len(tokens) and tokens[j]['id_'] == tokens[i]['id_']+(j-i):
    #         breakpoint()
    #         token_str += ' ' + tokens[j]['token']
    #         j += 1

    #     print(token_str)
    #     breakpoint()

    #     i += 1

    # for i, token in enumerate(tokens):
    #     token_str = token['token']
    #     # 12 - 1 -> 11

    #     for j in range(i, len(tokens)):
    #         print("I am here!")
    #         breakpoint()
    #         if tokens[i+j]['id_'] == token['id_'] + j:
    #             token_str += ' ' + tokens[i+j]['token']
    #             breakpoint()
    #         else:
    #             breakpoint()
    #             break

    #     breakpoint()
        # TODO: Make so that it goes on until next id does not add up

        # so i + n while not tokens[n][id]-n == token[id]
        # if i+1 < len(tokens):
        #     # Check if the two tokens come after each other (based on id)
        #     if tokens[i+1]['id_']-1 == token['id_']:
        #         # Combine the two tokens
        #         token_str = token['token'] + ' ' + tokens[i+1]['token']

        # is_city = has_hypernym_relation('city', token_str)
        # is_cou = has_hypernym_relation('country', token_str)
        # breakpoint()

    return tokens


def find_nouns(tokens: list[Token]) -> list[Token]:
    return [t for t in tokens if t['pos'].startswith('NN')]


def parse(tokens: list[Token], lemma: str, ent_class: str) -> list[Token]:
    for token in tokens:
        if has_hypernym_relation(lemma, token['token']):
            token['entity'] = ent_class
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




def main() -> int:

    # Load the tokens

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

    # Identity entities of interest (with category)
    port = 8123
    proc = None
    if not args.server:
        proc = start_corenlp(port=port)

    # XXX: Update me, this is for dev
    tokens = all_files_tokens[0][1]

    # Get LOCATION, ORGANIZATION and PERSON corenlp NEs
    if proc is not None:
        tokens = corenlp_parse(tokens, url=f'http://localhost:{port}')
        proc.terminate()
    else:
        tokens = corenlp_parse(tokens, url=args.server)

    # Check if LOCATION is CIT or COU
    tokens = parse_cit_or_cou(tokens)

    # Get animal NEs
    tokens = parse_animals(tokens)
    tokens = parse_sports(tokens)
    # tloc = [t for t in tokens if t['core_nlp_ent'] == 'LOCATION']
    # t = location_to_cit_or_cou(tloc)
    breakpoint()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
