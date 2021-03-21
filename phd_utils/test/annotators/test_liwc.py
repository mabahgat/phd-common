import unittest

from phd_utils.annotators import liwc


class TestLiwc(unittest.TestCase):

    def test_liwc_annotate_word(self):
        cats_lst = liwc.liwc_annotate_word('hi')
        self.assertEqual(len(cats_lst), 3)
    
    def test_liwc_annotate_word_strict(self):
        cats_lst = liwc.liwc_annotate_word_strict('absenti')
        self.assertEqual(len(cats_lst), 0)

        cats_lst = liwc.liwc_annotate_word_strict('absent')
        self.assertEqual(len(cats_lst), 1)