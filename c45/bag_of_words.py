import abc
import numpy as np
import typing as t
import collections


class BagOfWords(abc.ABC):
    """
    Abstract base class for bag of words algorithms.
    """
    def __init__(self, documents: list[str]):
        vectors, vocabulary = self.build_bag(documents)
        self.vectors = vectors
        self._vocabulary = set(vocabulary)

    @abc.abstractmethod
    def tokenize(self, document: str) -> list[str]:
        ...

    def build_bag(self, documents: list[str]) -> t.Tuple[np.ndarray, list[str]]:
        tokenized_docs = [self.tokenize(doc) for doc in documents]
        self.vocabulary = list(
            sorted(
                set(word for tokenized_doc in tokenized_docs for word in tokenized_doc)
            )
        )
        return (
            np.array([self.vectorize_tokens(doc) for doc in tokenized_docs]),
            self.vocabulary,
        )

    def vectorize(self, document: str) -> list[int]:
        tokens = self.tokenize(document)
        return self.vectorize_tokens(tokens)

    def vectorize_tokens(self, tokens: list[str]) -> list[int]:
        word_counts = collections.Counter(tokens)
        return [word_counts.get(word, 0) for word in self.vocabulary]


class StupidBOW(BagOfWords):
    """
    Bag of Words without stemming or lemmatization, just splitting using python "".lower().split()
    Will try spacy or nltk later for stemming / lemmatization later on. Any word not in the vocabulary
    will be placed at the end as unknown values
    """

    def tokenize(self, document: str) -> list[str]:
        return document.lower().split()

    def vectorize_tokens(self, tokens: list[str]) -> list[int]:
        unknown = 0
        if hasattr(self, "_vocabulary"):
            for token in tokens:
                if token not in self._vocabulary:
                    unknown += 1

        word_counts = collections.Counter(tokens)
        vector = [word_counts.get(word, 0) for word in self.vocabulary]
        vector.append(unknown)
        return vector
