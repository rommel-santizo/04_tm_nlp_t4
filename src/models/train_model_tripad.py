import logging
import pickle

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)


def train_model():
    with open('../../data/interim/data_lemmatized_tripad.pkl', 'rb') as input_file:
        data_lemmatized = pickle.load(input_file)
    logging.info('READ data_lemmatized_tripad.pkl')

    # Creamos diccionario
    id2word = corpora.Dictionary(data_lemmatized)
    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # MODELO
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=5,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)

    # Perplejidad
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

    # Score de coherencia
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    # DICCIONARIO
    with open('../../models/id2word_tripad.pkl', 'wb') as output_file:
        pickle.dump(id2word, output_file)
    logging.info('SAVE id2word_tripad.pkl')
    # CORPUS
    with open('../../models/corpus_tripad.pkl', 'wb') as output_file:
        pickle.dump(corpus, output_file)
    logging.info('SAVE corpus_tripad.pkl')
    # MODELO LDA
    with open('../../models/lda_model_tripad.pkl', 'wb') as output_file:
        pickle.dump(lda_model, output_file)
    logging.info('SAVE lda_model_tripad.pkl')
    # MODEL COHERENCE LDA
    # with open('../../models/coherence_model_lda_tripad.pkl', 'wb') as output_file:
    #     pickle.dump(coherence_model_lda, output_file)
    # logging.info('SAVE coherence_model_lda_tripad.pkl')
    # COHERENCE LDA
    # with open('../../models/coherence_lda_tripad.pkl', 'wb') as output_file:
    #     pickle.dump(coherence_lda, output_file)
    # logging.info('SAVE coherence_lda_tripad.pkl')


def main():
    logging.info('INI main()')
    train_model()
    logging.info('FIN main()')


if __name__ == "__main__":
    main()
