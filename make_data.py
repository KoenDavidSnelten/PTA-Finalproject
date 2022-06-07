import argparse
import os
import sys
import time


def load_tokens(path: str) -> list[tuple[str, str]]:
    """Loads the tokens from the given path"""

    with open(path, 'r') as f:
        lines = f.readlines()

    print(f'Loading {path}')
    tokens: list[tuple[str, str]] = []
    for line in lines:
        if len(line.split(' ')) == 6 or len(line.split(' ')) == 7:
            ls = line.split(' ')
            token = ls[3]
            ent = ls[5]
        else:
            ls = line.split(' ')
            token = ls[3]
            ent = 'O'
        nt = (token, ent)
        tokens.append(nt)

    return tokens


def write_tokens(
    tokens: list[tuple[str, str]],
    filename: str,
    outfolder: str,
) -> int:

    out_path = os.path.join(outfolder, filename)
    print(f'Writing to: {out_path}')

    try:
        with open(out_path, 'a') as outfile:
            for token in tokens:
                str_ = '\t'.join(token)
                outfile.write(str_ + '\n')
    except OSError as e:
        print(f'Error: {e}', file=sys.stderr)
        return 1

    return 0


def main() -> int:

    parser = argparse.ArgumentParser()
    # FILENAMES should include NE information
    parser.add_argument('out')
    parser.add_argument('filename')
    args = parser.parse_args()

    tokens = load_tokens(args.filename)
    out = write_tokens(tokens, f'tokens_{time.time()}.ent', args.out)
    print('Done')

    return out


if __name__ == '__main__':

    raise SystemExit(main())
