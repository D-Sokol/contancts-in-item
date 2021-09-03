import itertools
import json
import numpy as np
import os.path
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from math import ceil
from preprocess import get_vectorizer, tokenize_corpus
from torch.utils.data import Dataset, DataLoader
from typing import List, Mapping, Optional, Sequence, Tuple, Union

from run import Test


class CIIDataset(Dataset):
    """
    Преобразует разбитые на токены тексты в числовые тензоры при помощи переданного словаря,
    позволяя работать с текстами стандартным для PyTorch образом.
    """
    def __init__(self, corpus: List[List[str]], vocabulary: Mapping[str, int],
                 targets: Optional[Sequence[int]] = None, dropout_rate=0.2):
        self.corpus = corpus
        if targets is not None and len(targets) != len(corpus):
            raise ValueError(
                "Размерности корпуса и целевого вектора отличаются: {} != {}".format(
                    len(corpus), len(targets)
                )
            )
        if isinstance(targets, pd.Series):
            # В этом случае у объекта targets может быть нетривиальный индекс,
            #  что приведет к неправильным результатам при обращении targets[ix]
            targets = targets.iloc
        self.targets = targets
        self.vocabulary = vocabulary
        self.dropout_rate = dropout_rate
        self.UNK_IX = torch.tensor(len(vocabulary))
        self.PAD_IX = len(vocabulary) + 1

    def __getitem__(self, ix) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
        ixs = torch.empty(len(self.corpus[ix]), dtype=torch.long)
        for i, token in enumerate(self.corpus[ix]):
            ixs[i] = self.vocabulary.get(token, self.UNK_IX)
        return ixs if self.targets is None else (ixs, self.targets[ix])

    def __len__(self):
        return len(self.corpus)

    def _collate(self, ixs):
        if isinstance(ixs[0], tuple):
            objects, targets = zip(*ixs)
            return self._collate(objects), torch.tensor(targets)

        max_len = max(map(len, ixs))
        batch = torch.full((len(ixs), max_len), self.PAD_IX, dtype=torch.long)
        for i, ix in enumerate(ixs):
            mask = torch.empty_like(ix, dtype=torch.bool).bernoulli_(p=self.dropout_rate)
            batch[i, :len(ix)] = torch.where(mask, self.UNK_IX, ix)
        return batch

    def get_loader(self, **kw) -> DataLoader:
        return DataLoader(self, collate_fn=self._collate, **kw)


class CIINet(nn.Module):
    """
    Используемая архитектура нейросети-классификатора.
    Возвращает logits (2 числа) для каждого переданного объекта.
    """
    def __init__(self, n_tokens: int, emb_dim=64, dropout_rate=0.2):
        super().__init__()
        self.emb = nn.Embedding(n_tokens, emb_dim, padding_idx=n_tokens-1)
        self.conv1a = nn.Conv1d(  emb_dim, 2*emb_dim, (3,), padding='same')
        self.conv1b = nn.Conv1d(  emb_dim, 2*emb_dim, (5,), padding='same')
        self.conv1c = nn.Conv1d(  emb_dim, 2*emb_dim, (9,), padding='same')
        self.drop = nn.Dropout(p=dropout_rate)
        self.conv2a = nn.Conv1d(6*emb_dim, 6*emb_dim, (3,), padding='same')
        self.linear = nn.Linear(6*emb_dim, 2)

    def forward(self, x):
        # dimensions: (batch, embedding==channel, time)
        x = self.emb(x).transpose(1, 2)
        x = torch.cat((self.conv1a(x), self.conv1b(x), self.conv1c(x)), dim=1)
        x = F.relu(x)
        x = self.drop(x)
        x = self.conv2a(x)
        x = F.relu(x)

        # shape: (batch_size, 6*emb_dim)
        x = x.max(dim=-1).values

        return self.linear(x)


class CIINetPosition(CIINet):
    """
    Класс нейросети, аналогичный CIINet, но не использующий MaxPooling по временной оси.
    Это позволяет определять наличие контактной информации не во всем объявлении, а в окрестности каждого токена.
    """
    def forward(self, x):
        # dimensions: (batch, embedding==channel, time)
        x = self.emb(x).transpose(1, 2)
        x = torch.cat((self.conv1a(x), self.conv1b(x), self.conv1c(x)), dim=1)
        x = F.relu(x)
        x = self.drop(x)
        x = self.conv2a(x)
        x = F.relu(x)

        # shape: (batch_size, time, 6*emb_dim)
        x = x.transpose(1, 2)

        return self.linear(x)


def train(model: CIINet, train_dataset: CIIDataset, test_dataset: CIIDataset, n_epoch: int,
          batch_size=64, verbose=True, checkpoints_dir=None, logger=None, device=torch.device('cpu'), resume_from=0):
    vrange = tqdm.trange if verbose else range
    model.to(device)
    opt = torch.optim.Adam(model.parameters())

    if resume_from:
        if checkpoints_dir is None:
            raise ValueError
        logger.debug("Загрузка весов после %d эпох", resume_from)
        model.load_state_dict(torch.load(os.path.join(checkpoints_dir, f'model-{resume_from}.pth'), map_location=device))

    for epoch in vrange(resume_from+1, n_epoch+1):
        model.train(True)
        losses_hist = []
        for X_batch, y_batch in train_dataset.get_loader(batch_size=batch_size, shuffle=True):
            logits = model(X_batch.to(device))
            loss = F.binary_cross_entropy_with_logits(logits,
                                                      torch.stack((1-y_batch, y_batch), dim=1).float().to(device))
            losses_hist.append(loss.item())
            loss.backward()
            opt.step()
            opt.zero_grad()

        if checkpoints_dir is not None and epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, f'model-{epoch}.pth'))

        if logger is not None:
            logger.debug("Epoch %d; train loss %f", epoch, np.mean(losses_hist))
        else:
            continue

        model.train(False)
        losses_hist = []
        with torch.no_grad():
            for X_batch, y_batch in test_dataset.get_loader(batch_size=batch_size):
                logits = model(X_batch.to(device))
                loss = F.binary_cross_entropy_with_logits(logits,
                                                          torch.stack((1-y_batch, y_batch), dim=1).float().to(device))
                losses_hist.append(loss.item())
        logger.debug("Epoch %d; test loss %f", epoch, np.mean(losses_hist))

    if logger is not None:
        logger.info("Обучение завершено")

    return model


def inference(model: CIINet, test_dataset: CIIDataset,
              batch_size=64, verbose=True, device=torch.device('cpu')) -> np.ndarray:
    """
    Возвращает массив вероятностей нахождения контактной информации в объявлениях переданного датасета.
    """
    model.to(device)
    model.train(False)

    probas = []
    with torch.no_grad():
        loader = test_dataset.get_loader(batch_size=batch_size, shuffle=False)
        if verbose:
            loader = tqdm.tqdm(loader, total=ceil(len(test_dataset) / batch_size))
        for X_batch in loader:
            logits = model(X_batch.to(device))
            probas_batch = F.softmax(logits, dim=-1)[..., 1].cpu().numpy()
            probas.append(probas_batch)
    return np.concatenate(probas)


def restore_position(orig_title: str, orig_description: str, token_ix: int, width=8,
                     pattern=re.compile(get_vectorizer().token_pattern)) -> Optional[Tuple[int, int]]:
    """
    Строит окно, центрированное по токену с номером token_ix.
    Возвращает позиции границ построенного окна в строке description или None, если окно выходит за границы description.
    """
    text = orig_title + '\n' + orig_description
    try:
        it = re.finditer(pattern, text)
        it = itertools.islice(it, token_ix, token_ix+1)
        start, finish = next(it).span()
        # Контактная информация находится в title
        if finish <= len(orig_title):
            return None
    except StopIteration:
        return None
    start = start - len(orig_title) - 1 - width
    finish = finish - len(orig_title) - 1 + width
    start = max(start, 0)
    finish = min(finish, len(orig_description))
    return start, finish


def inference2(model: CIINetPosition, test_dataset: CIIDataset, raw_data: pd.DataFrame,
               batch_size=64, verbose=True, device=torch.device('cpu'), width=8) -> np.ndarray:
    """
    Возвращает массив границ контактной информации в объявлениях переданного датасета.
    Предполагается, что контактная информация присутствует в каждом объявлении.
    """
    model.to(device)
    model.train(False)
    token_ixs = []
    with torch.no_grad():
        loader = test_dataset.get_loader(batch_size=batch_size, shuffle=False)
        if verbose:
            loader = tqdm.tqdm(loader, total=ceil(len(test_dataset) / batch_size))
        for X_batch in loader:
            logits = model(X_batch.to(device))
            ix_batch = logits[..., 1].argmax(dim=-1).cpu().numpy()
            token_ixs.append(ix_batch)

    token_ixs = np.concatenate(token_ixs)
    # Целочисленные типы не поддерживают значение NaN, которое валидно по смыслу задачи
    char_ixs = np.empty((token_ixs.size, 2), dtype=float)
    for i, ix in enumerate(token_ixs):
        # itertools.islice не поддерживает аргументы типа np.int64
        ix = int(ix)
        char_ixs[i] = restore_position(raw_data.iloc[i]['title'],
                                       raw_data.iloc[i]['description'], ix, width=width) or np.nan
    return char_ixs


class TestNetwork(Test):
    def __init__(self, debug: bool = True, weights_dir='weights', device=torch.device('cpu')):
        super().__init__(debug)
        self.device = device
        self.weights_dir = weights_dir

    def fit(self):
        with open(os.path.join(self.weights_dir, 'vocabulary.json'), 'rt') as file:
            vocabulary = json.load(file)
        self.vect = get_vectorizer(vocabulary=vocabulary)
        self.model = CIINet(n_tokens=len(vocabulary)+2)
        self.model.load_state_dict(torch.load(os.path.join(self.weights_dir, 'model.pth'), map_location=self.device))

    @staticmethod
    def _apply_bayesian_correction(task1_prediction: pd.DataFrame, categories: pd.Series):
        # Эти значения получены следующим образом:
        # >>> train = self.train()
        # >>> corrections_ = train.groupby('category')['is_bad'].mean() / train['is_bad'].mean()
        # >>> corrections_.to_dict()
        corrections = {
         'Бытовая электроника': 0.5706284090050064,
         'Для бизнеса': 0.6043369830130234,
         'Для дома и дачи': 0.9409319872980558,
         'Животные': 2.2564197028099273,
         'Личные вещи': 0.5693017571769379,
         'Недвижимость': 1.504179517339644,
         'Работа': 1.6301762830825937,
         'Транспорт': 1.0824957764536451,
         'Услуги': 2.072465299911039,
         'Хобби и отдых': 0.6731699836526672
        }
        categories = categories.replace(corrections)
        task1_prediction['prediction'] *= categories
        task1_prediction['prediction'] = task1_prediction['prediction'].map(lambda x: min(1.0, x))

    def process(self):
        self.logger.info("Загрузка модели")
        self.fit()

        self.logger.info("Токенизация")
        test = self.test()
        X_test = test['title'] + '\n' + test['description']
        X_test = tokenize_corpus(self.vect, X_test)
        test_dataset = CIIDataset(X_test, self.vect.vocabulary, dropout_rate=0.)

        task1_prediction = pd.DataFrame(columns=['index', 'prediction'])
        task1_prediction['index'] = test.index

        self.logger.info("Получение ответов (задание 1)")
        probas = inference(self.model, test_dataset, device=self.device)
        task1_prediction['prediction'] = probas

        task2_prediction = pd.DataFrame(columns=['index', 'start', 'finish'])
        task2_prediction['index'] = test.index

        self.logger.info("Получение ответов (задание 2)")
        model2 = CIINetPosition(self.model.emb.num_embeddings)
        model2.load_state_dict(self.model.state_dict())
        threshold = 0.5
        test_pos = test[probas > threshold]
        X_test_pos = test_pos['title'] + '\n' + test_pos['description']
        X_test_pos = tokenize_corpus(self.vect, X_test_pos)
        test_dataset2 = CIIDataset(X_test_pos, self.vect.vocabulary, dropout_rate=0.)
        positions = inference2(model2, test_dataset2, test_pos, device=self.device)
        task2_prediction.loc[probas > threshold, ['start', 'finish']] = positions

        self.logger.debug("Поправка на фазу Луны...")
        task1_prediction.loc[test['price'] >= 1e9, 'prediction'] = 1.0

        return task1_prediction, task2_prediction


if __name__ == '__main__':
    import argparse
    from sklearn.model_selection import train_test_split
    from preprocess import prepare_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--resume', type=int, default=0)
    args = parser.parse_args()

    test_obj = Test(debug=args.debug)
    logger = test_obj.logger
    logger.info("Чтение данных")

    path = os.path.join('weights', 'train_processed.csv')
    if os.path.isfile(path):
        logger.debug("Обнаружены подготовленные данные")
        train_data = pd.read_csv(path)
    else:
        logger.debug("Стемминг текстов")
        train_data = test_obj.train()
        prepare_dataset(train_data)
        train_data.to_csv(path)

    logger.info("Составление словаря")
    path = os.path.join('weights', 'vocabulary.json')
    if os.path.isfile(path):
        logger.debug("Обнаружен подготовленный словарь")
        with open(path, 'rt') as file:
            vect = get_vectorizer(use_stemmer=False, vocabulary=json.load(file))
            vocabulary = vect.vocabulary
    else:
        logger.debug("Обучение vectorizer")
        vect = get_vectorizer(use_stemmer=False)
        vect.fit(train_data['title'] + '\n' + train_data['description'])
        with open('weights/vocabulary.json', 'wt') as file:
            # Явное преобразование в int нужно для предотвращения ошибки:
            # TypeError: Object of type int64 is not JSON serializable
            vocabulary = {word: int(ix) for word, ix in vect.vocabulary_.items()}
            logger.debug("Словарь содержит %d токенов", len(vocabulary))
            json.dump(vocabulary, file)
            del vocabulary
        vocabulary = vect.vocabulary_

    train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=495)

    X_train = train_data['title'] + '\n' + train_data['description']
    X_test = test_data['title'] + '\n' + test_data['description']
    logger.debug("Токенизация")
    X_train = tokenize_corpus(vect, X_train)
    X_test = tokenize_corpus(vect, X_test)

    train_dataset = CIIDataset(X_train, vocabulary, targets=train_data['is_bad'])
    test_dataset = CIIDataset(X_test, vocabulary, targets=test_data['is_bad'], dropout_rate=0.)

    model = CIINet(len(vocabulary)+2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Запуск обучения на %s", device)
    train(model, train_dataset, test_dataset,
          n_epoch=50, checkpoints_dir='weights', logger=test_obj.logger, device=device, resume_from=args.resume)
