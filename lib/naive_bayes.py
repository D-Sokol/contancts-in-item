import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from typing import Tuple

from run import Test
from preprocess import get_vectorizer, tokenize_corpus


class TestBayes(Test):
    def fit(self):
        train = self.train()
        self.vect = vect = get_vectorizer()
        self.pipeline = make_pipeline(
            vect,
            TfidfTransformer(),
            MultinomialNB(),
        )
        self.logger.debug('tokenize_corpus')
        X_train = train['title'] + '\n' + train['description']
        y_train = train['is_bad']
        self.logger.debug('fit')
        self.pipeline.fit(X_train, y_train)

    def process(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.logger.info("Обучение модели")
        self.fit()

        self.logger.info("Получение ответов")
        test = self.test()
        self.logger.debug("Токенизация ")
        X_test = test['title'] + '\n' + test['description']

        task1_prediction = pd.DataFrame(columns=['index', 'prediction'])
        task1_prediction['index'] = test.index
        self.logger.debug("Вычисление вероятностей")
        task1_prediction['prediction'] = self.pipeline.predict_proba(X_test)

        task2_prediction = pd.DataFrame(columns=['index', 'start', 'finish'])
        task2_prediction['index'] = test.index
        task2_prediction[['start', 'finish']] = (None, None)

        return task1_prediction, task2_prediction
