
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
