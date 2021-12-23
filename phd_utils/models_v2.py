from abc import ABC, abstractmethod
import os
from pathlib import Path
import re
import json
from datetime import datetime
import time
from typing import List, Dict

from sklearn import metrics
import numpy as np

import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer, BertConfig
from transformers import TFRobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from transformers import TFT5ForConditionalGeneration, T5Tokenizer, T5Config

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
    def evaluate(self, dataset: DatasetBase, set_type):
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
        self.__y_test_cache = {}
        self.__y_hat_test_cache = {}
        self.__y_hat_test_prob_cache = {}
        if not active_model_config.model_exists(label_str):
            model = self._create_new_model(model_name_str)
            super().__init__(label_str, model, tokenizer)
            self._loaded_from_disk = False
        else:
            super().__init__(label_str, None, tokenizer)
            self.load()
            self._loaded_from_disk = True
    
    def train(self, dataset: DatasetBase, epochs):
        self.reset_cached_test_output()

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
    
    def get_test_ref_out_pair(self, dataset: DatasetBase, set_type: str='test'):
        """
        Get Reference and output pair for the selected test set
        """
        ref_lst = [SequenceClassificationModel.__to_list(index)[0] for index in self.get_test_ref_labels(dataset, set_type)]  # FIXME for now take the first index, deal with mutlilabel case
        return ref_lst, self.get_test_out_labels(dataset, set_type)
    
    def get_test_ref_labels(self, dataset: DatasetBase, set_type='test'):
        if (dataset, set_type) not in self.__y_test_cache:
            test_tensor = SequenceClassificationModel.__get_set(dataset, set_type)
            self.__y_test_cache[(dataset, set_type)] = SequenceClassificationModel.__class_index_from_tensors(test_tensor)
        return self.__y_test_cache[(dataset, set_type)]
    
    def set_test_ref_labels(self, dataset: DatasetBase, labels_lst: List[int], set_type: str='test'):
        """
        Alters reference labels cached corresponding to the dataset
        :param labels_lst: Either a list of integers or a list of arrays or a list of mix
        """
        self.__y_test_cache[(dataset, set_type)] = labels_lst

    def get_test_out_labels(self, dataset: DatasetBase, set_type: str='test'):
        if (dataset, set_type) not in self.__y_hat_test_cache:
            y_hat_test_prob = self.get_test_out_prob(dataset, set_type)
            self.__y_hat_test_cache[(dataset, set_type)] = [list(labels).index(max(labels)) for labels in y_hat_test_prob]
        return self.__y_hat_test_cache[(dataset, set_type)]
    
    def set_test_out_labels(self, dataset: DatasetBase, labels_lst: List[int], set_type: str='test'):
        """
        Alters output labels cached corresponding to the dataset
        :param labels_lst: Either a list of integers or a list of arrays or a list of mix
        """
        self.__y_hat_test_cache[(dataset, set_type)] = labels_lst
    
    def get_test_out_prob(self, dataset: DatasetBase, set_type: str='test'):
        if (dataset, set_type) not in self.__y_hat_test_prob_cache:
            test_tensor = SequenceClassificationModel.__get_set(dataset, set_type)
            test_results = self._model.predict(test_tensor)
            self.__y_hat_test_prob_cache[(dataset, set_type)] = SequenceClassificationModel.normalize_tf(test_results.logits)
        return self.__y_hat_test_prob_cache[(dataset, set_type)]
    
    def set_test_out_prob(self, dataset: DatasetBase, prob_lst: List[float], set_type: str='test'):
        self.__y_hat_test_prob_cache[(dataset, set_type)] = prob_lst

    @staticmethod
    def __get_set(dataset: DatasetBase, set_type: str):
        """
        param set_type: can be either test for test set, dev/val for validation set
        """
        if set_type == 'test':
            return dataset.testing_examples()
        elif set_type == 'dev' or set_type == 'val':
            return dataset.validation_examples()
        else:
            raise ValueError('Unexpected set type {}'.format(set_type))
    
    def reset_cached_test_output(self):
        self.__y_test_cache = {}
        self.__y_hat_test_cache = {}
        self.__y_hat_test_prob_cache = {}
    
    def evaluate(self, dataset: DatasetBase, set_type: str='test'):
        y_test, y_hat_test = self.get_test_ref_out_pair(dataset, set_type)

        fscore = metrics.f1_score(y_test, y_hat_test, average='macro')
        accuracy = metrics.accuracy_score(y_test, y_hat_test)

        result_dict = {
            'f-score': fscore,
            'accuracy': accuracy
        }
        self._model_params_dict['evaluation'] = result_dict
        return result_dict
    
    def evaluate_detailed(self, dataset: DatasetBase, set_type: str='test'):
        y_test, y_hat_test = self.get_test_ref_out_pair(dataset, set_type)
        return metrics.classification_report(y_test, y_hat_test, target_names=dataset.class_names(), output_dict=True)
    
    def confusion_matrix(self, dataset: DatasetBase, normalize=None, set_type='test'):
        """
        Generate Confusion Matrix
        dataset: dataset to get the test samples from
        normalize: passed on to sklearn confusion matrix. Recommeneded either 'all' or None
        """
        y_test, y_hat_test = self.get_test_ref_out_pair(dataset, set_type)
        return metrics.confusion_matrix(y_test, y_hat_test, normalize=normalize)

    @staticmethod # TEMP
    def normalize(probs):
        prob_factor = 1 / sum(probs)
        return [prob_factor * p for p in probs]
    
    @staticmethod
    def normalize_tf(logits):
        return tf.nn.softmax(logits).numpy()
    
    def predict(self, text_str):
        results = self._model.predict(text_str)
        probs_np = SequenceClassificationModel.normalize_tf(results.logits)[0]
        return probs_np
    
    @staticmethod
    def __class_index_from_tensors(tensor):
        return list(np.concatenate([y for x,y in tensor], axis=0)) # for this to work tensor needs to be batched
        # return [y.numpy() for x,y in tensor_no_batch]

    @abstractmethod
    def _create_new_model(self, model_name_str):
        pass

    def is_loaded_from_disk(self):
        return self._loaded_from_disk
    
    def precision_recall_curves(self, dataset: DatasetBase, set_type='test'):
        class_count = dataset.class_count()
        y_out_prob = self.get_test_out_prob(dataset, set_type)
        y_ref = self.get_test_ref_labels(dataset, set_type)
        y_ref_onehot = np.array([SequenceClassificationModel._to_one_hot(index, class_count) for index in y_ref])
        
        precision_dict, recall_dict, threshold_dict, average_precision_dict = SequenceClassificationModel.precision_recall_curve_per_class(y_ref_onehot, y_out_prob, dataset.class_names())
        precision_dict['micro'], recall_dict['micro'], threshold_dict['micro'], average_precision_dict['micro'] = SequenceClassificationModel.precision_recall_curve_micro_average(y_ref_onehot, y_out_prob, class_count)
        precision_dict['macro'], recall_dict['macro'], threshold_dict['macro'], average_precision_dict['macro'] = SequenceClassificationModel.precision_recall_curve_macro_average(y_ref_onehot, y_out_prob, class_count)

        return precision_dict, recall_dict, threshold_dict, average_precision_dict

    @staticmethod
    def precision_recall_curve_per_class(y_ref_onehot, y_out_prob, class_labels_lst):
        precision_dict = {}
        recall_dict = {}
        threshold_dict = {}
        average_precision_dict = {}

        for i in range(len(class_labels_lst)):
            label_str = class_labels_lst[i]
            y_out_prob_adjusted = SequenceClassificationModel.unpicked_to_zero(y_out_prob, i)
            precision_dict[label_str], recall_dict[label_str], threshold_dict[label_str] = metrics.precision_recall_curve(y_ref_onehot[:,i], y_out_prob_adjusted)
            average_precision_dict[label_str] = metrics.average_precision_score(y_ref_onehot[:,i], y_out_prob_adjusted)
        
        return precision_dict, recall_dict, threshold_dict, average_precision_dict
    
    @staticmethod
    def precision_recall_curve_micro_average(y_ref_onehot, y_out_prob, class_count): # FIXME needs to consistent with other curves computation method (supress non picked instances)
        precision, recall, thresholds = metrics.precision_recall_curve(y_ref_onehot.ravel(), y_out_prob.ravel())
        average_precision = metrics.average_precision_score(y_ref_onehot, y_out_prob, average='micro')
        return precision, recall, thresholds, average_precision        

    @staticmethod
    def _to_one_hot(index, class_count):
        """
        Creates one hot arrays corresponding to the passed indexes
        :param index: a single int index or an array of int indexes
        """
        index = SequenceClassificationModel.__to_list(index)
        a_lst = np.zeros(class_count, dtype=int)
        for i in index:
            a_lst[i] = 1
        return a_lst
    
    @staticmethod
    def __to_list(value):
        if type(value) != list:
            value = [value]
        return value

    @staticmethod
    def generate_prob_thresholds(prob_matrix, count=100):
        return np.linspace(0, 1, count)
        
    @staticmethod
    def to_binary_result(prob_matrix, class_index, threshold): # TODO use unpicked_to_zero function
        """
        Transforms probability results to binary. Class is marked as 1
        if it is the max class and above the threshold
        """
        max_index_value_pair_lst = []
        for example_result_arr in prob_matrix:
            max_value = max(example_result_arr)
            max_index = list(example_result_arr).index(max_value)
            max_index_value_pair_lst.append((max_index, max_value))
        return [1 if index_value_pair[0] == class_index and index_value_pair[1] >= threshold else 0 for index_value_pair in max_index_value_pair_lst]

    @staticmethod
    def unpicked_to_zero(prob_matrix, class_index):
        """
        Returns a new array with values other than the maximum reset to zero
        """
        class_values_lst = []
        for example_result_arr in prob_matrix:
            max_value = max(example_result_arr)
            max_index = list(example_result_arr).index(max_value)
            value = max_value if max_index == class_index else 0
            class_values_lst.append(value)
        return np.array(class_values_lst)

    @staticmethod
    def precision_recall_curve_per_class_at_thresholds(ref_matrix, out_matrix, class_count, threshold_lst=None):
        """
        Calculates the precision and recall for each class at specified thresholds for all classes
        params:
        - ref_matrix: Array of examples, for each example a one hot encoded label class
        - out_matrix: Array of examples, for each example the probability for each class for the given example
        - class_count: number of classes
        - threshold_lst: thresholds to compute precision and recall for each
        returns:
        - precision: Numpy Array for each class. Each element in that array has an array of
        precision values for each threshold
        - Recall: same as precision but for recall
        - threshold: threshold used in calculating these values
        """
        if not threshold_lst:
            threshold_lst = SequenceClassificationModel.generate_prob_thresholds(out_matrix)
        precision_matrix = [[] for _ in range(class_count)]
        recall_matrix = [[] for _ in range(class_count)]
        for class_index in range(class_count):
            precision_lst = [[] for _ in range(len(threshold_lst))]
            recall_lst = [[] for _ in range(len(threshold_lst))]
            for index, threshold in enumerate(threshold_lst):
                ref_lst = ref_matrix[:,class_index]
                out_lst = SequenceClassificationModel.to_binary_result(out_matrix, class_index, threshold) # precision_score takes exact label
                precision_lst[index] = metrics.precision_score(ref_lst, out_lst)
                recall_lst[index] = metrics.recall_score(ref_lst, out_lst)
            precision_matrix[class_index] = precision_lst
            recall_matrix[class_index] = recall_lst
        return np.array(precision_matrix), np.array(recall_matrix), np.array(threshold_lst)

    @staticmethod
    def column_average(value_matrix):
        return np.average(value_matrix, axis=0) # over thresholds
    
    @staticmethod
    def precision_recall_curve_macro_average(y_ref_onehot, y_out_prob, class_count):
        precision_matrix, recall_matrix, threshold_lst = SequenceClassificationModel.precision_recall_curve_per_class_at_thresholds(y_ref_onehot, y_out_prob, class_count)
        precision_lst = SequenceClassificationModel.column_average(precision_matrix)
        recall_lst = SequenceClassificationModel.column_average(recall_matrix)
        precision_average = sum(precision_lst) / len(precision_lst)
        # to be consistent with sklearn return values append 1 to precision and 0 to recall https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
        return np.append(precision_lst, 1), np.append(recall_lst, 0), threshold_lst, precision_average


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
