
# Classes & How to find

- `COU`: Country/state
  - From corenlp `LOCATION` 
  - Check using wordnet for Country
- `CIT`: City
  - From corenlp `LOCATION` 
  - Check using wordnet for City
- `NAT`: Natural places
  - Using wordnet?
- `PER`: Persons
  - From corenlp `PERSON`
- `ORG`: Organization
  - From corenlp `ORGANIZATION`
- `ANI`: Animals
  - From wordnet
- `SPO`: Sports
  - From wordnet
- `ENT`: Entertainment
  - ??

# Server


```
$ java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,depparse,lemma,ner,entitymentions,entitylink -start_port 8125 -port 8125 -timeout 15000 -serverproperties server.properties
```

# Create training data

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

