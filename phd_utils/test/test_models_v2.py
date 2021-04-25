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


class TestBertSequenceClassification(unittest.TestCase):

    @unittest.skip
    def test_train_and_test(self):
        with tf.device("/cpu:0"), tempfile.TemporaryDirectory() as tmp_dir:
            ModelConfig(tmp_dir)

            dataset = RandomTextDataset()
            model = BertSequenceClassification(dataset.class_count(), "bert-model-test")
            dataset.load(provider=TensorProvider(model.tokenizer(), batch_size=50))
            history = model.train(dataset, 2)
            self.assertNotEqual(history, None)

            result = model.evaluate(dataset)
            self.assertTrue('f-score' in result and 'accuracy' in result)
    
    @unittest.skip
    def test_save(self):
         with tf.device("/cpu:0"), tempfile.TemporaryDirectory() as tmp_dir:
            ModelConfig(tmp_dir)
            
            model = BertSequenceClassification(4, "test_model")
            original_weights = list(model.get_model().layers[0].get_weights()[2][1])
            model.save()

            loaded_model = BertSequenceClassification(4, "test_model")
            loaded_weights = list(loaded_model.get_model().layers[0].get_weights()[2][1])

            self.assertListEqual(original_weights, loaded_weights)
    
    @unittest.skip
    def test_reset(self):
         with tf.device("/cpu:0"), tempfile.TemporaryDirectory() as tmp_dir:
            ModelConfig(tmp_dir)
            dataset = RandomTextDataset()
            model = BertSequenceClassification(4, "test_model")
            dataset.load(provider=TensorProvider(model.tokenizer(), batch_size=50))
            model.train(dataset, 1)
            eval_details_1 = model.evaluate(dataset)
            eval_details_2 = model.evaluate(dataset)
            model.reset_cached_test_output()
            eval_details_3 = model.evaluate(dataset)

            self.assertListEqual(list(eval_details_2.values()), list(eval_details_1.values()))
            self.assertListEqual(list(eval_details_3.values()), list(eval_details_1.values()))
    
    def test_precision_recall_curve(self):
        with tf.device("/cpu:0"), tempfile.TemporaryDirectory() as tmp_dir:
            ModelConfig(tmp_dir)
            dataset = RandomTextDataset()
            model = BertSequenceClassification(4, "test_model")
            dataset.load(provider=TensorProvider(model.tokenizer(), batch_size=50))
            model.train(dataset, 1)

            p, r, a, t = model.precision_recall_curves(dataset)
            self.assertIsNotNone(p) # FIXME better checks for values in this test case
            self.assertIsNotNone(r)
            self.assertIsNotNone(a)
            self.assertIsNotNone(t)

            # Multiple test matching case
            ref_labels_lst = model.get_test_ref_labels(dataset)
            ref_labels_lst[3] = [0, 1]
            ref_labels_lst[-2] = [2,1]
            model.set_test_ref_labels(dataset, ref_labels_lst)

            p, r, a, t = model.precision_recall_curves(dataset)
            self.assertIsNotNone(p)
            self.assertIsNotNone(r)
            self.assertIsNotNone(a)
            self.assertIsNotNone(t)

            results_dict = model.evaluate_detailed(dataset)
            self.assertIsNotNone(results_dict)
