from setuptools import find_packages
from setuptools import setup

setup(
    name='wikifier',
    version='0.1.0',
    description='A wikification system that works with en.tok.off.pos files'
                'and has a flask website.',
    url='https://github.com/gamemeloentje/PTA-Finalproject',
    packages=find_packages(),
    python_requires='>=3.9.0',
    install_requires=[
        'nltk',
        'requests',
        'spacy',
        'flask',
        'wikipedia',
    ],
    entry_points={
        'console_scripts': [
            'wikifier=wikifier.__main__:main',
        ],
    },
)
