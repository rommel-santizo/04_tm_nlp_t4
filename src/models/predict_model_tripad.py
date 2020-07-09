import logging
import pickle

from typing import List

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)

other_texts = [
    ['hotel','relax','beatiful'],
    ['walk', 'adventure', 'adventure']]


def predict_model(documents: List[List[str]]):
    # DICCIONARIO
    with open('../../models/id2word_tripad.pkl', 'rb') as input_file:
        id2word = pickle.load(input_file)
    logging.info('READ id2word_tripad.pkl')
    # CORPUS
    # with open('../../models/corpus_tripad.pkl', 'rb') as input_file:
    #     corpus = pickle.load(input_file)
    # logging.info('READ corpus.pkl')
    # MODELO LDA
    with open('../../models/lda_model_tripad.pkl', 'rb') as input_file:
        lda_model = pickle.load(input_file)
    logging.info('READ lda_model_tripad.pkl')

    other_corpus = [id2word.doc2bow(text) for text in documents]
    print(other_corpus)
    predictions = []
    for unseen_doc in other_corpus:
        # print(unseen_doc)
        vector = lda_model[unseen_doc]
        predictions.append(vector)
        # print(vector)
        # print(predictions)
    return predictions


def main():
    logging.info('INI main()')
    predictions = predict_model(other_texts)
    print(predictions)
    logging.info('FIN main()')


if __name__ == "__main__":
    main()
