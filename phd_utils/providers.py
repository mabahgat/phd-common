from abc import ABC, abstractmethod
from typing import List
from phd_utils.annotators.liwc import LiwcDict
import tensorflow as tf
import numpy as np

class Provider(ABC):

    @abstractmethod
    def apply(self, *args):
        pass


class PassThroughProvider(Provider):

    def apply(self, *args):
        if not args or len(args) != 2:
            raise ValueError("Bad arguments for apply. Need both samples and labels")
        return args


class GroupedSamplesProvider(Provider):

    def apply(self, *args):
        """
        Apply provider. First argument is the list of samples, second list is the list of labels.
        """
        if not args or len(args) != 2:
            raise ValueError("Bad arguments for apply. Need both samples and labels")
        samples_lst, labels_lst = args
        class_samples_dict = {}
        for label, sample_str in zip(labels_lst, samples_lst):
            if label not in class_samples_dict:
                class_samples_dict[label] = []
            class_samples_dict[label].append(sample_str)
        return class_samples_dict


class TokenProvider(Provider):

    def __init__(self, tokenizer):
        self.__tokenizer = tokenizer

    def apply(self, *args):
        if not args or not len(args):
            raise ValueError("Missing arguments")
        elif len(args) > 2:
            raise ValueError("Getting more arguments than expected")
        elif len(args) == 2:
            x_lst, y_lst = args
        elif len(args) == 1:
            x_lst = args[0]
            y_lst = None

        if isinstance(args[0], str):
            x_lst = args[0]
            x_lst = self.__tokenizer.tokenize(x_lst)
        elif isinstance(args[0], list) or isinstance(args[0], tuple):
            x_lst = args[0]
            x_lst = [self.__tokenizer.tokenize(t) for t in x_lst]
        else:
            raise ValueError("Unexpected type as first argument got type {} for {}".format(type(args[0]), args[0]))
        
        if y_lst:
            return x_lst, y_lst
        else:
            return x_lst


# TODO Encoding should be the only thing in the provider - liwc is preprocessor, or there should be parent provider for encoding
class LiwcProvider(Provider):
    """
    Provides liwc counts per category for each input sentence with one hot encoding for the target class.
    Its intended use for building tensorflow classifiers (DenseClassifier)
    """

    def __init__(self, liwc: LiwcDict, tokenizer):
        self._tokenizer = tokenizer
        self.__liwc = liwc
    
    def apply(self, *args):
        if not args or len(args) != 2:
            raise ValueError("Bad arguments for apply. Need both samples and labels")
        text_lst = args[0]
        class_lst = args[1]
        text_processed_lst = [self.__compute_liwc_features(text_str) for text_str in text_lst]
        encoded_class_lst = self.__one_hot_encode(class_lst)
        return np.array(text_processed_lst), np.array(encoded_class_lst)

    def __compute_liwc_features(self, text_str: str) -> List[int]:
        tokens_lst = self._tokenizer.tokenize(text_str)
        cat_dict = self.__liwc.annotate_sentence(tokens_lst, include_all_cats=True)
        cats_sorted_by_key_dict = sorted(cat_dict.items(), key=lambda kv: kv[0])
        return np.array([r for _, r in cats_sorted_by_key_dict])
    
    def __one_hot_encode(self, class_lst):
        size = self.__liwc.cat_count()
        encoded_lst = []
        for c in class_lst:
            a = [0 for _ in range(size)]
            a[c] = 1
            encoded_lst.append(np.array(a))
        return encoded_lst


class TensorProvider(Provider):
    """
    Provider for using Tensorflow tensors while training, evaluating and running
    """

    def __init__(self, tokenizer, batch_size):
        self._tokenizer = tokenizer
        self.__batch_size = batch_size

    def apply(self, *args):
        """
        Converts examples and labels lists to tensors with batches
        """
        if not args or not len(args):
            raise ValueError("Missing arguments")
        elif len(args) > 2:
            raise ValueError("Getting more arguments than expected")
        elif len(args) == 2:
            x_lst, y_lst = args
        elif len(args) == 1:
            if isinstance(args[0], str):
                x_lst = [args[0]]
            elif isinstance(args[0], list):
                x_lst = args[0]
            else:
                raise ValueError("Unexpected type as first argument")
            y_lst = [0 for _ in range(len(x_lst))]
        x_encoded, y_encoded = self._encode(x_lst, y_lst)
        tensor = tf.data.Dataset.from_tensor_slices((
            dict(x_encoded),
            y_encoded
        ))
        return tensor.batch(self.__batch_size)
    
    def _encode(self, x_lst, y_lst):
        return self._tokenizer(x_lst, truncation=True, padding=True), y_lst


class T5Provider(TensorProvider):

    def _encode(self, x_lst, y_lst):
        y_str_lst = [str(y) for y in y_lst]
        x_encoded = self._tokenizer(x_lst, truncation=True, padding=True)
        y_encoded = self._tokenizer(y_str_lst, truncation=True, padding=True).input_ids
        return x_encoded, y_encoded
        
