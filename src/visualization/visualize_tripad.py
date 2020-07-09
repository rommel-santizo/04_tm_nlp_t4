import logging
import pickle

import pyLDAvis
import pyLDAvis.gensim

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def visualize_model():
    # DICCIONARIO
    with open('../../models/id2word_tripad.pkl', 'rb') as input_file:
        id2word = pickle.load(input_file)
    logging.info('READ id2word_tripad.pkl')
    # CORPUS
    with open('../../models/corpus_tripad.pkl', 'rb') as input_file:
        corpus = pickle.load(input_file)
    logging.info('READ corpus_tripad.pkl')
    # MODELO LDA
    with open('../../models/lda_model_tripad.pkl', 'rb') as input_file:
        lda_model = pickle.load(input_file)
    logging.info('READ lda_model_tripad.pkl')

    # Visualizamos los temas
    # pyLDAvis.enable_notebook(local=True)
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    pyLDAvis.show(vis)


def main():
    logging.info('INI main()')
    visualize_model()
    logging.info('FIN main()')


if __name__ == "__main__":
    main()
