from abc import ABC, abstractmethod
import os
from pathlib import Path
import re
import json
from datetime import datetime
import time
from typing import List

from sklearn import metrics
import numpy as np

import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer, BertConfig
from transformers import TFRobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from transformers import TFT5ForConditionalGeneration, T5Tokenizer, T5Config
from tensorflow.python.client import device_lib

from phd_utils.datasets_v2 import DatasetBase
from phd_utils import global_config

run_verbosity = 1


class ModelConfig:

    __default_base_path_str = global_config.models.path
    params_file_name = "model_info.json"

    @staticmethod
    def disable_gpu():
        tf.config.set_visible_devices([], 'GPU')
    
    @staticmethod
    def instance(base_path_str = __default_base_path_str):
        if not os.path.exists(base_path_str):
            raise ValueError("Path {} has to exist".format(base_path_str))
        return ModelConfig(base_path_str)
    
    def __init__(self, base_path_str):
        self._base_path_str = base_path_str

    def get_base_path(self):
        return self._base_path_str
    
    def get_model_path(self, model_label_str):
        return os.path.join(self._base_path_str, model_label_str)
    
    def model_exists(self, model_label_str):
        return os.path.exists(self.get_model_path(model_label_str))
    
    def list_models(self):
        return [f.name for f in Path(self._base_path_str).iterdir() if f.is_dir()]
    
    def model_info(self, model_label_str: str):
        if model_label_str not in set(self.list_models()):
            raise ValueError('Model {} not found in path {}'.format(model_label_str, self.get_base_path()))
        with open(os.path.join(self.get_model_path(model_label_str), ModelConfig.params_file_name), 'w') as json_file:
            return json.load(json_file)


active_model_config = ModelConfig.instance()


class ModelBase(ABC): # Previously named "Model"

    def __init__(self, label_str):
        if re.match(r'\s', label_str):
            raise ValueError("Model label can not contain spaces '{}'".format(label_str))
        self._label_str = label_str
        self._model_params_dict = {
            'label': label_str
        }

    @abstractmethod
    def train(self, dataset: DatasetBase, epochs):
        pass

    @abstractmethod
    def evaluate(self, dataset: DatasetBase):
        pass

    @abstractmethod
    def predict(self, text_lst):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def save(self):
        pass

    def _save_params(self, path_str):
        with open(path_str, 'w') as json_file:
            json.dump(self._model_params_dict, json_file, indent=2)

    def get_label(self):
        return self._label_str

    def exists(self):
        active_model_config.model_exists(self._label_str)



class TFModel(ModelBase):

    def __init__(self, label_str, model):
        self._model = model
        super().__init__(label_str)

    def _model_path(self, create=False):
        path_str = active_model_config.get_model_path(self._label_str)
        if create:
            Path(path_str).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(path_str):
            raise ValueError("Directory {} must exist".format(path_str))
        return path_str
    
    def save(self, path_str):
        model_path_str = self._model_path(True)
        self._model.save_weights(model_path_str)
        self._save_params(os.path.join(model_path_str, active_model_config.params_file_name))
    
    def load(self, path_str):
        self._model.load_weights(self._model_path())

    def get_model(self):
        return self._model


class HFModel(TFModel):

    def __init__(self, label_str, model, tokenizer):
        self._tokenizer = tokenizer
        super().__init__(label_str, model)
    
    def save(self):
        """
        Saves a model to path specified by the active ModelConfig and set model label
        """
        model_path_str = self._model_path(True)
        self._model.save_pretrained(model_path_str)
        self._save_params(os.path.join(model_path_str, active_model_config.params_file_name))
    
    @abstractmethod
    def load(self):
        pass

    def tokenizer(self):
        return self._tokenizer


class SequenceClassificationModel(HFModel):

    def __init__(self, class_count: int, label_str: str, config, tokenizer, model_name_str: str = 'bert-base-uncased'):
        self._config = config
        self._config.num_labels = class_count
        if not active_model_config.model_exists(label_str):
            model = self._create_new_model(model_name_str)
            super().__init__(label_str, model, tokenizer)
            self._loaded_from_disk = False
        else:
            super().__init__(label_str, None, tokenizer)
            self.load()
            self._loaded_from_disk = True
    
    def train(self, dataset: DatasetBase, epochs):
        start = time.process_time()
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        self._model.compile(optimizer=optimizer, loss=self._model.compute_loss)
        history = self._model.fit(dataset.training_examples(), epochs=epochs, validation_data=dataset.validation_examples(), verbose=run_verbosity)
        duration = time.process_time() - start

        self._model_params_dict['learning_rate'] = 5e-5
        self._model_params_dict['epochs'] = epochs
        self._model_params_dict['training timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._model_params_dict['training time'] = duration
        self._model_params_dict['class count'] = dataset.class_count()

        return history
    
    def _get_test_output(self, dataset: DatasetBase):
        test_tensor = dataset.testing_examples()
        y_test = SequenceClassificationModel.__labels_from_tensors(test_tensor)
        y_hat_test_prob = self._model.predict(test_tensor, verbose=run_verbosity)
        y_hat_test = [list(labels).index(max(labels)) for labels in y_hat_test_prob[0]]
        return y_test, y_hat_test
    
    def evaluate(self, dataset: DatasetBase):
        y_test, y_hat_test = self._get_test_output(dataset)

        fscore = metrics.f1_score(y_test, y_hat_test, average='macro')
        accuracy = metrics.accuracy_score(y_test, y_hat_test)

        result_dict = {
            'f-score': fscore,
            'accuracy': accuracy
        }
        self._model_params_dict['evaluation'] = result_dict
        return result_dict
    
    def evaluate_detailed(self, dataset: DatasetBase):
        y_test, y_hat_test = self._get_test_output(dataset)
        return metrics.classification_report(y_test, y_hat_test, target_names=dataset.class_names(), output_dict=True)
    
    def confusion_matrix(self, dataset: DatasetBase):
        y_test, y_hat_test = self._get_test_output(dataset)
        return metrics.confusion_matrix(y_test, y_hat_test)

    def predict(self, text):
        probs_np = self._model.predict(text)
        return [list(sample_prob_lst) for sample_prob_lst in probs_np[0]]
    
    @staticmethod
    def __labels_from_tensors(tensor):
        return list(np.concatenate([y for x,y in tensor], axis=0)) # for this to work tensor needs to be batched
        # return [y.numpy() for x,y in tensor_no_batch]

    @abstractmethod
    def _create_new_model(self, model_name_str):
        pass

    def is_loaded_from_disk(self):
        return self._loaded_from_disk


class BertSequenceClassification(SequenceClassificationModel):

    def __init__(self, class_count: int, label_str: str, model_name_str: str = 'bert-base-uncased'):
        config = BertConfig.from_pretrained(model_name_str)
        tokenizer = BertTokenizer.from_pretrained(model_name_str)
        super().__init__(class_count, label_str, config, tokenizer, model_name_str)

    def _create_new_model(self, model_name_str):
        return TFBertForSequenceClassification.from_pretrained(model_name_str, config=self._config)

    def load(self):
        """
        Loads a model from path specified by the active ModelConfig and set model label
        """
        self._model = TFBertForSequenceClassification.from_pretrained(self._model_path(), config=self._config)


class RobertaSequenceClassification(SequenceClassificationModel):

    def __init__(self, class_count: int, label_str: str, model_name_str: str = 'roberta-base'):
        config = RobertaConfig.from_pretrained(model_name_str)
        tokenizer = RobertaTokenizer.from_pretrained(model_name_str)
        super().__init__(class_count, label_str, config, tokenizer, model_name_str)

    def _create_new_model(self, model_name_str):
        return TFRobertaForSequenceClassification.from_pretrained(model_name_str, config=self._config)

    def load(self):
        """
        Loads a model from path specified by the active ModelConfig and set model label
        """
        self._model = TFRobertaForSequenceClassification.from_pretrained(self._model_path(), config=self._config)


class T5SequenceClassification(SequenceClassificationModel):

    def __init__(self, class_count: int, label_str: str, model_name_str: str = 't5-base'):
        config = T5Config.from_pretrained(model_name_str)
        tokenizer = T5Tokenizer.from_pretrained(model_name_str)
        super().__init__(class_count, label_str, config, tokenizer, model_name_str)

    def _create_new_model(self, model_name_str):
        return TFRobertaForSequenceClassification.from_pretrained(model_name_str, config=self._config)

    def load(self):
        """
        Loads a model from path specified by the active ModelConfig and set model label
        """
        self._model = TFRobertaForSequenceClassification.from_pretrained(self._model_path(), config=self._config)  
