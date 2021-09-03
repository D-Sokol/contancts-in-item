import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

from typing import Iterable, List, Mapping, Optional

_stemmer = SnowballStemmer('russian')


class StemmedCountVectorizer(CountVectorizer):
    def __init__(self, use_stemmer=False, **kw):
        super().__init__(**kw)
        self.use_stemmer = use_stemmer

    def build_analyzer(self):
        analyzer = super().build_analyzer()
        if not self.use_stemmer:
            return analyzer

        def analyzer2(doc):
            return [_stemmer.stem(token) for token in analyzer(doc)]
        return analyzer2


def get_vectorizer(use_stemmer=True, vocabulary: Optional[Mapping[str, int]] = None) -> CountVectorizer:
    """
    Возвращает объект CountVectorizer с оптимальными значениями гиперпараметров.
    Все объекты, созданные этой функцией, заведомо генерируют одно и то же разбиение текста на токены.
    """
    # Токеном считается либо последовательность букв (без включения цифр),
    #  либо один символ пунктуации, либо ровно одна цифра.
    # Например, номер телефона будет разобран как 11-15 токенов в зависимости от формата.
    pattern = r"[^\W\d]+|[!-@\[-_{-~]"
    return StemmedCountVectorizer(use_stemmer=use_stemmer, token_pattern=pattern, min_df=0.005, vocabulary=vocabulary)


def tokenize_corpus(vectorizer: CountVectorizer, corpus: Iterable[str]) -> List[List[str]]:
    pre = vectorizer.build_preprocessor()
    ana = vectorizer.build_analyzer()
    return [ana(pre(doc)) for doc in corpus]


def tokenize(vectorizer: CountVectorizer, doc: str) -> List[str]:
    pre = vectorizer.build_preprocessor()
    ana = vectorizer.build_analyzer()
    return ana(pre(doc))


def prepare_dataset(dataset: pd.DataFrame, inplace=True) -> pd.DataFrame:
    if not inplace:
        dataset = dataset.copy()
    vect = get_vectorizer()

    def prepare(doc: str) -> str:
        # При чтении csv пустые строки могут преобразовываться в NaN.
        # Простой способ избежать этого -- не записывать пустые строки, заменяя их на '*'
        return ' '.join(tokenize(vect, doc)) or '*'

    dataset['title'] = dataset['title'].apply(prepare)
    dataset['description'] = dataset['description'].apply(prepare)
    return dataset


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=pd.read_csv, help="Файл, данные в котором нужно преобразовать")
    parser.add_argument('output', help="Путь для записи выходного файла")

    args = parser.parse_args()
    prepare_dataset(args.input).to_csv(args.output)
