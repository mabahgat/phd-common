import os
import unittest
import tempfile # https://www.tutorialspoint.com/generate-temporary-files-and-directories-using-python
from pathlib import Path

import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertConfig

from phd_utils.models import ModelConfig, HFModel, BertSequenceClassification
from phd_utils.providers import PassThroughProvider


@unittest.skip
class StubModelForTesting(HFModel):
    def __init__(self):
        config = BertConfig.from_pretrained("bert-base-uncased")
        model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", config=config)
        super().__init__("stub", model)

    def train(self, x_train, y_train, x_valid, y_valid, train_batch, epochs):
        pass

    def evaluate(self, x_test, y_test):
        pass

    def predict(self, text_lst):
        pass

    def load(self):
        pass


@unittest.skip
class TestModelConfig(unittest.TestCase):

    def test_get_base_path(self):
        self.assertEqual(ModelConfig.get_base_path(), "/home/mbahgat/ws/work/models/bert_finetune")
    
    def test_set_base_path(self):
        with tempfile.TemporaryDirectory() as temp_dir_str:
            ModelConfig.set_base_path(temp_dir_str)
            self.assertEqual(ModelConfig.get_base_path(), temp_dir_str)
    
    def test_set_model_path_not_exist(self):
        with self.assertRaises(ValueError):
             ModelConfig.set_base_path("/dir/not/exists")
    
    def test_get_model_path(self):
        model_label_str = "a_model"
        model_expceted_path_str = "/home/mbahgat/ws/work/models/bert_finetune/{}".format(model_label_str)
        self.assertEqual(ModelConfig.get_model_path(model_label_str), model_expceted_path_str)


@unittest.skip
class TestModels(unittest.TestCase):

    def setUp(self):
        tf.device("/cpu:0")

    def test_label(self):
        model = StubModelForTesting()
        self.assertEqual(model.get_label(), "stub")
    
    def test_save_model(self):
        with tf.device("/cpu:0"), tempfile.TemporaryDirectory() as tmp_dir:
            ModelConfig.set_base_path(tmp_dir)
            model = StubModelForTesting()
            model.save()
            model_root_str = os.path.join(tmp_dir, model.get_label())
            files_lst = os.listdir(model_root_str)
            self.assertListEqual(['tf_model.h5', 'config.json'], files_lst)
            for file_str in files_lst:
                file_obj = Path(os.path.join(model_root_str, file_str))
                self.assertTrue(file_obj.is_file())
                self.assertTrue(file_obj.stat().st_size > 0)


@unittest.skip
class TestBertSequenceClassification(unittest.TestCase):

    def test_load_model(self):
        with tf.device("/cpu:0"), tempfile.TemporaryDirectory() as tmp_dir:
            ModelConfig.set_base_path(tmp_dir)
            model = BertSequenceClassification(4, "test_model")
            original_weights = list(model.get_model().layers[0].get_weights()[2][1])
            model.save()

            loaded_model = BertSequenceClassification(4, "test_model")
            loaded_weights = list(loaded_model.get_model().layers[0].get_weights()[2][1])

            self.assertListEqual(original_weights, loaded_weights)

