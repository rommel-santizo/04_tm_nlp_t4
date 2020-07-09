import logging
import pickle
import re

import gensim
import pandas as pd
import spacy
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from pandas import DataFrame

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# python3 -m spacy download en_core_web_lg
# Inicializamos el modelo 'en_core_web_lg' con las componentes de POS únicamente
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# CARGAMOS STOP WORDS EN INGLÉS
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])


def read_sample() -> DataFrame:
    df = pd.read_csv('../../data/raw/tripadvisor_reviews.csv')
    # df['rating'] = df['rating'].astype(dtype='int64')
    df.drop(columns=['rating'], inplace=True)
    # df = df[:1000]
    logging.info('read_sample')
    return df


def sent_to_words(sentences):
    for sentence in sentences:
        # https://radimrehurek.com/gensim/utils.html#gensim.utils.simple_preprocess
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)  # deacc=True elimina la puntuación


# Eliminar stopwords
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


# Hacer bigrams
def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]


# Hacer trigrams
def make_trigrams(texts, bigram_mod, trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


# Lematización basada en el modelo de POS de Spacy
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def preprocess_data(df: DataFrame, save: bool = False):
    # Convertir a una lista
    data = df.review.values.tolist()
    # Eliminar emails
    data = [re.sub(r'\S*@\S*\s?', '', sent) for sent in data]
    # Eliminar newlines
    data = [re.sub(r'\s+', ' ', sent) for sent in data]
    # Eliminar comillas
    data = [re.sub(r"\'", "", sent) for sent in data]
    # CONVERTIR SETENCES TO WORDS
    data_words = list(sent_to_words(data))

    # Construimos modelos de bigrams y trigrams
    #  https://radimrehurek.com/gensim/models/phrases.html#gensim.models.phrases.Phrases
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=10)
    # trigram = gensim.models.Phrases(bigram[data_words], threshold=10)

    # Aplicamos el conjunto de bigrams/trigrams a nuestros documentos
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    # trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Eliminamos stopwords
    data_words_nostops = remove_stopwords(data_words)
    # Formamos bigrams
    data_words_bigrams = make_bigrams(data_words_nostops, bigram_mod)

    # Lematizamos preservando únicamente noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    # print(data_lemmatized[:1])

    if save:
        with open("../../data/interim/data_lemmatized_tripad.pkl", "wb") as output_file:
            pickle.dump(data_lemmatized, output_file)
        logging.info('Model saved to artifact data_lemmatized_tripad.pkl')

    return data_lemmatized


def main():
    df = read_sample()
    preprocess_data(df, True)


if __name__ == "__main__":
    main()
