import unittest

from phd_utils.preprocessors import SwapIAndYou, SwapIAndHe, SwapIAndShe


class TestSwapIAndYou(unittest.TestCase):

    def test_apply(self):
        in_str = "I wanted to be with you. but I'm leaving you here. Your path is different than mine so i prefer to do it myself rather than your way.."
        expected_str = "you wanted to be with me . but you're leaving me here . my path is different than yours so you prefer to do it yourself rather than my way .."
        preprocessor = SwapIAndYou()
        out_str = preprocessor.apply(in_str)
        self.assertEqual(out_str, expected_str, msg="Original: {}".format(in_str))
    
    def test_apply_verb_fix(self):
        in_str = "I was here. I'm here too."
        expected_str = "you were here . you are here too ."
        preprocessor = SwapIAndYou()
        out_str = preprocessor.apply(in_str)
        self.assertEqual(out_str, expected_str, msg="Original: {}".format(in_str))


class TestSwapIAndHe(unittest.TestCase):

    def test_apply(self):
        in_str = 'I went to the shop. He gave up his dreams. I have a car. I like football.'
        expected_str = 'he went to the shop . i gave up my dreams . he has a car . he likes football .'
        preprocessor = SwapIAndHe()
        out_str = preprocessor.apply(in_str)
        self.assertEqual(out_str, expected_str, msg="Original: {}".format(in_str))


class TestSwapIAndShe(unittest.TestCase):

    def test_apply(self):
        in_str = 'I went to the shop. She gave up her dreams. I have a car. I like football.'
        expected_str = 'she went to the shop . i gave up my dreams . she has a car . she likes football .'
        preprocessor = SwapIAndShe()
        out_str = preprocessor.apply(in_str)
        self.assertEqual(out_str, expected_str, msg="Original: {}".format(in_str))
