# Noox Project Fake News Classifier
Set of algorithms used to classify fake news used by Noox Project

## Usage
### FastText
Before FastText can be trained to create a model, the dataset should be converted using FastText convert command
```sh
$ python main.py convert -i data.csv  -o data.txt -p '__label__' -d 0 -d 1 -d 2 -l 3
    Flags: 
    -i          input file
    -o          output file
    -p          prefix
    -d          data column
    -l          label column
    
$ python main.py createmodel -d data.txt
    Flags:
    -d          input file
```
### Rest of the classifiers
```sh
$ python main.py createmodel -i data.csv -d 0 -d 1 -d 2 -l 3
    Flags:
    -i          input file
    -d          data column
    -l          label column
    -r          training/test ratio