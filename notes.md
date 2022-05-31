
# Classes & How to find

- [x] `COU`: Country/state
  - From corenlp `COUNTRY`
- [x] `CIT`: City
  - From corenlp `CITY`
- [ ] `NAT`: Natural places
  - Using wordnet?
- [x]`PER`: Persons
  - From corenlp `PERSON`
- [x] `ORG`: Organization
  - From corenlp `ORGANIZATION`
- [ ] `ANI`: Animals
  - From wordnet
  - Basics done, needs ngrams and fine-tuning
- [ ] `SPO`: Sports
  - From wordnet
  - Basics done, needs ngrams and fine-tuning
- [ ] `ENT`: Entertainment
  - ??

# Server

**Start the server with `regexner!`**

```
$  java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,regexner,depparse  -start_port 8126 -port 8126 -timeout 15000 -serverproperties server.properties
```

**Server with auto entity linking (not allowed to use...)**

```
$ java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,depparse,lemma,ner,entitymentions,entitylink -start_port 8125 -port 8125 -timeout 15000 -serverproperties server.properties
```

# Train own Corenlp model

https://nlp.stanford.edu/software/crf-faq.html#a

```
$ find . -name en.tok.off.pos.ent | xargs -I {} python3 make_data.py train/data {}
```

Split in train and test data

```
cp `ls | head -38` ../test_data/.
cp `ls | head -39` ../train_data/.
```

Copy the train_data to corenlp

Download NER train: https://nlp.stanford.edu/software/CRF-NER.html#Download

```
java -cp "*" edu.stanford.nlp.ie.crf.CRFClassifier -prop ner.model.props
```

Test the model:
```
java -cp stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier ner-model_v0.ser.gz -testFiles ../train/test_data/*
```

## Results

See `train/v1_results.out`

col 0: token
col 1: gold standard
col 2: model output

Note: it seems like the model still just is the build in model?
