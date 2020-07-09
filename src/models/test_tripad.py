import logging

from src.models.predict_model_tripad import predict_model

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)
# x = pd.DataFrame({'content': [1, 2, 3]})

other_texts = [
    ['good','food','beautiful'],
    ['bathroom', 'bed', 'room']]


def main():
    logging.info('INI main()')
    predictions = predict_model(other_texts)
    print(predictions)
    logging.info('FIN main()')


if __name__ == "__main__":
    main()

# [[(151, 1), (315, 1), (2262, 1)], [(70, 1), (1452, 2)]]
# [([(0, 0.46124935), (1, 0.027233727), (2, 0.091726646), (3, 0.26263127), (4, 0.15715903)], [(151, [3, 0]), (315, [0]), (2262, [2])], [(151, [(0, 0.10437608), (3, 0.8956199)]), (315, [(0, 0.9999146)]), (2262, [(2, 0.99783564)])]),
#  ([(0, 0.43811616), (1, 0.027233532), (2, 0.13147229) , (3, 0.2460172) , (4, 0.15716077)], [(70, [0, 3]), (1452, [2])], [(70, [(0, 0.522286), (3, 0.4776653)]), (1452, [(2, 1.9976184)])])]
