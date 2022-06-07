# Wikifier

# Installation

## Method 1 - Using `pip`

Make sure you are in the 'root' directory of the project and run:

```console
$ pip install .
```

This will install all dependencies for you and give you a `wikifier` executable
command.

You do have to make sure you have all the `nltk` and `spacy` modules and data files 
installed locally.

### Installing corenlp

```console
$ wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip
$ unzip stanford-corenlp-full-2018-02-27.zip
$ mv stanford-corenlp-full-2018-02-27/ corenlp/
$ cp server.properties corenlp/.
```

## Method 2 - Install dependencies yourself

```console
$ pip install -r requirements.txt
```

You do have to make sure you have all the `nltk` and `spacy` modules and data files 
installed locally.

### Installing corenlp

```console
$ wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip
$ unzip stanford-corenlp-full-2018-02-27.zip
$ mv stanford-corenlp-full-2018-02-27/ corenlp/
$ cp server.properties corenlp/.
```

# Running

## CLI

```console
usage: wikifier [-h] [--server SERVER] inpath [inpath ...]

positional arguments:
  inpath           The file(s) or directorie(s), containing the files you want to process.

optional arguments:
  -h, --help       show this help message and exit
  --server SERVER  If you have a corenlp server running use this. Format: http://host:port
```

### Without own corenlp server

**Run the wikifier on all en.tok.off.pos files within the dev directory:**

```console
$ wikifier dev/ 
```

This will output an en.tok.off.pos.ent file for every processed file.

**Run on a specific file:** 

```console
$ wikifier en.tok.off.pos
```

### With corenlp server running

**Start the corenlp server (make sure that regexner is preloaded):** 

```
$  java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,regexner,depparse  -start_port 8126 -port 8126 -timeout 15000 -serverproperties server.properties
```
**Run the wikifier on all en.tok.off.pos files within the dev directory:**

```console
$ wikifier dev/  --server http://localhost:8126
```

This will output an en.tok.off.pos.ent file for every processed file.

**Run on a specific file:** 

```console
$ wikifier en.tok.off.pos --server http://localhost:8126
```

## Webserver

The webserver has te be run using flask or when hosted using a wsgi server (e.g.: [gunicorn](https://gunicorn.org/))

### Using flask (dev)

Always make sure that the `FLASK_APP` and `FLASK_ENV` environment variables are set properly.

```
FLASK_APP="wikifier.server:create_app()" FLASK_ENV=development flask run --port 9191
```


