import os
import unittest
import tempfile # https://www.tutorialspoint.com/generate-temporary-files-and-directories-using-python
from pathlib import Path

import tensorflow as tf

from phd_utils.models_v2 import BertSequenceClassification, ModelConfig, active_model_config
from phd_utils.datasets_v2 import RandomTextDataset
from phd_utils.providers import TensorProvider


@unittest.skip
class TestModelConfig(unittest.TestCase):

    @unittest.skip
    def test_disable_gpu(self):
        ModelConfig.disable_gpu()
        logical_devices = tf.config.list_logical_devices('GPU')
        self.assertEqual(len(logical_devices), 0)


@unittest.skip
class TestBertSequenceClassification(unittest.TestCase):

    def test_train_and_test(self):
        dataset = RandomTextDataset()
        model = BertSequenceClassification(dataset.class_count(), "bert-model-test")
        dataset.load(provider=TensorProvider(model.tokenizer(), batch_size=50))
        history = model.train(dataset, 2)
        self.assertNotEqual(history, None)

        result = model.evaluate(dataset)
        self.assertTrue('f-score' in result and 'accuracy' in result)
    
    def test_save(self):
         with tf.device("/cpu:0"), tempfile.TemporaryDirectory() as tmp_dir:
            active_model_config = ModelConfig(tmp_dir) # FIXME
            
            model = BertSequenceClassification(4, "test_model")
            original_weights = list(model.get_model().layers[0].get_weights()[2][1])
            model.save()

            loaded_model = BertSequenceClassification(4, "test_model")
            loaded_weights = list(loaded_model.get_model().layers[0].get_weights()[2][1])

            self.assertListEqual(original_weights, loaded_weights)
