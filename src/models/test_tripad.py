import logging

from src.models.predict_model_tripad import predict_model

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)

other_texts = [
    ['The hotel was relaxing and had a beautiful view'],
    ['I enjoy to walk and have adventures, take the risk']]


def main():
    logging.info('INI main()')
    predictions = predict_model(other_texts)
    # print(predictions)
    logging.info('FIN main()')


if __name__ == "__main__":
    main()
