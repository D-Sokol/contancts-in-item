import argparse
import logging
import os
import pandas as pd
import sys
from abc import ABC, abstractmethod
from typing import Tuple


class Test(ABC):
    data_root = os.getenv('DATA_ROOT')
    test_root = os.getenv('TEST_DATA_ROOT')
    output_root = os.getenv('OUTPUT_ROOT', test_root)
    label = os.getenv('MODEL', 'cii_net')
    task1_prediction = 'task1_prediction.csv'
    task2_prediction = 'task2_prediction.csv'

    train_path = os.path.join(data_root, 'train.csv')
    test_path = os.path.join(test_root, 'test.csv')

    def __init__(self, debug: bool = True):
        self.logger = logger = logging.getLogger('CII')
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s: %(name)s: %(levelname)-8s: %(message)s')

        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG if debug else logging.INFO)

    def train(self):
        return pd.read_csv(self.train_path)

    def test(self):
        return pd.read_csv(self.test_path)

    @abstractmethod
    def process(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass

    def run(self):
        self.logger.info(f'Запуск модели {self.label}')
        task1_prediction, task2_prediction = self.process()

        task1_path = os.path.join(self.output_root, f'{self.label}_{self.task1_prediction}')
        task2_path = os.path.join(self.output_root, f'{self.label}_{self.task2_prediction}')
        self.logger.info(f'Сохранение результатов {task1_path}')
        task1_prediction.to_csv(task1_path, index=False)
        self.logger.info(f'Сохранение результатов {task2_path}')
        task2_prediction.to_csv(task2_path, index=False)

        self.logger.info('Готово')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')

    import torch
    # Из-за возможного circular import эта строка не может быть вынесена в начало файла
    #  без выноса класса Test в отдельный модуль.
    from network import TestNetwork
    test = TestNetwork(debug=parser.parse_args().debug,
                       device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    test.run()
