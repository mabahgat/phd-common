from abc import ABC, abstractmethod
import os
from pathlib import Path

import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer, BertConfig
from sklearn import metrics


class ModelConfig:

    __base_path_str = "/home/mbahgat/ws/work/models/bert_finetune"

    @staticmethod
    def set_base_path(path_str):
        if not os.path.exists(path_str):
            raise ValueError("Path {} has to exist".format(path_str))
        ModelConfig.__base_path_str = path_str

    @staticmethod    
    def get_base_path():
        return ModelConfig.__base_path_str
    
    @staticmethod
    def get_model_path(model_label_str):
        return os.path.join(ModelConfig.__base_path_str, model_label_str)
    
    @staticmethod
    def model_exists(model_label_str):
        return os.path.exists(ModelConfig.get_model_path(model_label_str))


class Model(ABC):

    def __init__(self, label_str):
        self._label_str = label_str

    @abstractmethod
    def train(self, x_train, y_train, x_valid, y_valid, train_batch, epochs):
        pass

    @abstractmethod
    def evaluate(self, x_test, y_test):
        pass

    @abstractmethod
    def predict(self, text_lst):
        pass

    @abstractmethod
    def load(self, path_str):
        pass

    @abstractmethod
    def save(self, path_str):
        pass

    def get_label(self):
        return self._label_str

    @staticmethod
    def _text_to_tensors(tokenizer, x_lst, y_lst):
        text_encodings_lst = Model._tokenizer(tokenizer, x_lst)
        return tf.data.Dataset.from_tensor_slices((
            dict(text_encodings_lst),
            y_lst
        ))
    
    @staticmethod
    def _tokenizer(tokenizer, text_lst):
        return tokenizer(text_lst, truncation=True, padding=True)


class TFModel(Model):

    def __init__(self, label_str, model):
        self._model = model
        super().__init__(label_str)

    def _model_path(self, create=False):
        path_str = ModelConfig.get_model_path(self._label_str)
        if create:
            Path(path_str).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(path_str):
            raise ValueError("Directory {} must exist".format(path_str))
        return path_str
    
    def save(self, path_str):
        self._model.save_weights(self._model_path(True))
    
    def load(self, path_str):
        self._model.load_weights(self._model_path())

    def get_model(self):
        return self._model


class HFModel(TFModel):
    
    def save(self):
        """
        Saves a model to path specified by ModelConfig and set model label
        """
        self._model.save_pretrained(self._model_path(True))
    
    @abstractmethod
    def load(self, path_str):
        pass


class SequenceClassificationModel(HFModel):

    def __init__(self, label_str, model, tokenizer):
        self._tokenizer = tokenizer
        super().__init__(label_str, model)

    def train(self, x_train, y_train, x_valid, y_valid, train_batch, epochs):
        train_tensors = Model._text_to_tensors(self._tokenizer, x_train, y_train)
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        self._model.compile(optimizer=optimizer, loss=self._model.compute_loss) # can also use any keras loss fn
        if x_valid and y_valid:
            valid_tensors = Model._text_to_tensors(self._tokenizer, x_train, y_train)
            history = self._model.fit(train_tensors.batch(train_batch), epochs=epochs, batch_size=train_batch, validation_data=valid_tensors)
        else:
            history = self._model.fit(train_tensors.batch(train_batch), epochs=epochs, batch_size=train_batch)
        return history

    def evaluate(self, x_test, y_test):
        test_tensors = Model._text_to_tensors(self._tokenizer, x_test, y_test)
        y_hat_test_prob = self._model.predict(test_tensors.batch(1))
        y_hat_test = [list(labels).index(max(labels)) for labels in y_hat_test_prob[0]]

        fscore = metrics.f1_score(y_test, y_hat_test, average='macro')
        accuracy = metrics.accuracy_score(y_test, y_hat_test)

        return {
            'f-score': fscore,
            'accuracy': accuracy
        }

    def predict(self, text):
        return self._model.predict(text)


class BertSequenceClassification(SequenceClassificationModel):

    def __init__(self, class_count: int, label_str: str, model_name_str: str = 'bert-base-uncased'):
        self.__config = BertConfig.from_pretrained(model_name_str)
        self.__config.num_labels = class_count
        tokenizer = BertTokenizer.from_pretrained(model_name_str)
        if not ModelConfig.model_exists(label_str):
            model = self.__create_new_model(model_name_str)
            super().__init__(label_str, model, tokenizer)
        else:
            super().__init__(label_str, None, tokenizer)
            self.load()

    def __create_new_model(self, model_name_str):
        return TFBertForSequenceClassification.from_pretrained(model_name_str, config=self.__config)

    def load(self):
        """
        Loads a model from path specified by ModelConfig and set model label
        """
        self._model = TFBertForSequenceClassification.from_pretrained(self._model_path(), config=self.__config)
