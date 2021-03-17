"""
Don't know why the original repo uses different preprocesssing functions for facts, training, and dev questions.
Should be integrated later.
"""


import re
import nltk
from nltk.corpus import stopwords


def remove_punctuations(sentence):

    sentence = sentence.replace("?", " ").replace(".", " ").replace(
        ",", " ").replace(";", " ").replace("-", " ")

    return sentence


def remove_punctuations2(sentence):

    sentence = sentence.replace("?", " ").replace(".", " ").replace(
        ",", " ").replace(";", " ").replace("-", " ").replace("'", "").replace("`", "")

    return sentence


def preprocess_fact(fact, lemmatizer=None):
    """
    fact: a list of chunks
    """
    # Get rid of the last chunk if
    if "#" in fact[-1]:
        fact = fact[:-1]

    lemmatized_fact = []
    for chunk in fact:
        chunk = remove_punctuations(chunk)
        chunk_words = []

        for word in nltk.word_tokenize(chunk):
            chunk_words.append(lemmatizer.lemmatize(word))
        if len(chunk_words) > 0:
            lemmatized_fact.append(" ".join(chunk_words))
    return lemmatized_fact


def preprocess_question(question, choice=None, lemmatizer=None):

    if choice is None:
        choice = ""

    question = question + " " + choice
    question = remove_punctuations2(question)

    tmp_words = []

    for word in nltk.word_tokenize(question):
        tmp_words.append(lemmatizer.lemmatize(word))

    return " ".join(tmp_words)


def preprocess_dev_question(question, choice=None, lemmatizer=None):

    if choice is None:
        choice = ""

    question = question + " " + choice
    question = remove_punctuations2(question)

    tmp_words = []

    for word in nltk.word_tokenize(question):
        if not word.lower() in stopwords.words("english"):
            tmp_words.append(lemmatizer.lemmatize(word))

    return " ".join(tmp_words)


class WorldTreeLemmatizer:

    def __init__(self, vocab_path="lemmatization-en.txt"):

        self.lemmas = {}

        # Load vocab.
        with open(vocab_path, 'r') as f:
            for line in f:
                line = line.rstrip().lower()
                lemma, word = line.split("\t", maxsplit=1)
                self.lemmas[word] = lemma

    def lemmatize(self, sentence: str):
        lemmatized_words = []
        for word in sentence.split(" "):
            if word.lower() in self.lemmas:
                lemma = self.lemmas[word.lower()]
                lemmatized_words.append(lemma)
            else:
                lemmatized_words.append(word.lower())
        return " ".join(lemmatized_words)
