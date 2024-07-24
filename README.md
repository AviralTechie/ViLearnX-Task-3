
![Python](https://img.shields.io/badge/Python-v3.12.1-green)

# Overview
* This project demonstrates the development of a Natural Language Processing (NLP) application to analyze and process text data. The application includes basic text preprocessing techniques such as tokenization, stemming, lemmatization, and stop-word removal. Additionally, advanced models like Word2Vec and GloVe are implemented to generate word embeddings.

## Table of Contents
- [Requirements](#requirements)
- [Files](#files)
- [How to Run](#how-to-run)
- [Setup](#setup)

## Requirements
* nltk
* numpy
* pandas
* gensim

# You can install these libraries using pip:
```bash
pip install pandas numpy matplotlib seaborn
```

# Clone repository
```bash
git remote add origin https://github.com/AviralTechie/ViLearnX-Task-1.git
```
## Files
* iris.csv: The Iris dataset file in CSV format.
* task-1.ipynb : Python script for data analysis and visualization.

## Setup
```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.models import Word2Vec
import numpy as np
import pandas as pd

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load sample text data
with open('data/sample_text.txt', 'r') as file:
    text = file.read()

print(text)

tokens = word_tokenize(text)
print(tokens)

stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print(filtered_tokens)

#stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
print(stemmed_tokens)

#lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
print(lemmatized_tokens)
```
1. Load the Iris Dataset
* Reads the Iris dataset from iris.csv.
  
2. Inspect the Dataset
* Displays the first few rows of the dataset.
* Prints the shape of the dataset.
* Provides information about the dataset.
* Shows descriptive statistics.
* Checks for missing values.

3. Visualizations
* Histograms: Displays histograms for each feature in the dataset.
* Box Plots: Visualizes the distribution of sepal.length and petal.length across different Iris varieties.
* Count Plot: Shows the count of samples for each Iris variety.
* Pair Plot: Illustrates pairwise relationships between features, colored by Iris variety.

## How to Run

1. Ensure you have the required libraries installed.
2. Place the iris.csv file in the same directory as the <filename> script.
3. Run the script using Python:
```bash
python analysis.py
```

# Author
* Aviral - https://github.com/AviralTechie
