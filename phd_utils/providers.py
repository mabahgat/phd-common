from abc import ABC, abstractmethod

import tensorflow as tf

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


class TensorProvider(Provider):
    """
    Provider for using Tensorflow tensors while training, evaluating and running
    """

    def __init__(self, tokenizer, batch_size):
        self.__tokenizer = tokenizer
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
        text_encoded_lst = self.__tokenizer(x_lst, truncation=True, padding=True)
        tensor = tf.data.Dataset.from_tensor_slices((
            dict(text_encoded_lst),
            y_lst
        ))
        return tensor.batch(self.__batch_size)
