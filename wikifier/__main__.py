import argparse
import json
import os
import subprocess
import sys
import time
from typing import Optional
from typing import TypedDict
from typing import Union


from wikifier.server.core.wikify import Token
from wikifier.server.core.wikify import load_tokens
from wikifier.server.core.wikify import wikify


def write_outfile(path: str, tokens: list[Token]) -> int:
    """Writes the given tokens to the output file (path)."""

    try:
        with open(f'{path}.ent', 'a') as outfile:
            for token in tokens:
                line = f'{token["start_off"]} {token["end_off"]} {token["id_"]} {token["token"]} {token["pos"]} {token["entity"] or ""} {token["link"] or ""}'  # noqa: E501
                line = line.strip()
                outfile.write(line + '\n')
    except IOError as e:
        print(f'Could not write to: {path}. Error: {e}', file=sys.stderr)
        return 1

    return 0


def start_corenlp(
        server_properties: str = 'server.properties',
        port: int = 9000,
        timeout: int = 25,
) -> subprocess.Popen[bytes]:
    """Start the corenlp server with the given port and server properties."""

    cwd = os.path.join(os.path.dirname(__file__), '../corenlp')

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
    proc = subprocess.Popen(
        args,
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(timeout)
    print('Server started!')
    return proc


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

        try:
            tokens = load_tokens(filename)
        except IOError:
            # If there is just one file that cannot be read exit the program
            # with exit code 1, otherwise skip the current file.
            if len(filenames) != 1:
                continue
            else:
                return 1

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
