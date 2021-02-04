import unittest

import numpy as np
from phd_utils.explainers import Lime, ExplainerBase


class TestLime(unittest.TestCase):

    @staticmethod
    def constant_fn(text_lst):
            return np.array([[0.8, 0.2] for _ in text_lst])

    @staticmethod
    def space_tokenize_fn(text_str: str):
        return text_str.split(' ')

    def test_explain_single(self):        
        explainer = Lime(['c1', 'c2'], [0, 1], TestLime.constant_fn)
        explainer.explain('hi me').as_list() # just make sure that it works
    
    def test_explain_list(self):
        samples = ['hi me', 'hi you']
        explainer = Lime(['c1', 'c2'], [0, 1], TestLime.constant_fn)
        result = [exp.as_list() for exp in explainer.explain(samples)]
        self.assertTrue(len(result) == 2)

    def test_explain_with_details(self):
        samples = ['hi me hi', 'hi you']
        explainer = Lime(['c1', 'c2'], [0, 1], TestLime.constant_fn, tokenize_fn=TestLime.space_tokenize_fn)
        details_dict = explainer.explain_with_details(samples)

        self.assertEqual(details_dict['hi'][ExplainerBase.TOP_COUNT_KEY], 1)
        self.assertEqual(details_dict['me'][ExplainerBase.TOP_COUNT_KEY], 1)
        self.assertEqual(details_dict['you'][ExplainerBase.TOP_COUNT_KEY], 0)
        
        self.assertEqual(details_dict['hi'][ExplainerBase.RANK_AVERAGE_KEY], 1.5)
        self.assertEqual(details_dict['me'][ExplainerBase.RANK_AVERAGE_KEY], 1)
        self.assertEqual(details_dict['you'][ExplainerBase.RANK_AVERAGE_KEY], 2)

        self.assertEqual(details_dict['hi'][ExplainerBase.COUNT_KEY], 3)
        self.assertEqual(details_dict['me'][ExplainerBase.COUNT_KEY], 1)
        self.assertEqual(details_dict['you'][ExplainerBase.COUNT_KEY], 1)

        self.assertEqual(details_dict['hi'][ExplainerBase.SENTENCE_COUNT], 2)
        self.assertEqual(details_dict['me'][ExplainerBase.SENTENCE_COUNT], 1)
        self.assertEqual(details_dict['you'][ExplainerBase.SENTENCE_COUNT], 1)

        self.assertEqual(details_dict['hi'][ExplainerBase.TOP_RATE_KEY], 0.5)
        self.assertEqual(details_dict['me'][ExplainerBase.TOP_RATE_KEY], 1)
        self.assertEqual(details_dict['you'][ExplainerBase.TOP_RATE_KEY], 0)

    def test_sort_details(self):
        samples = ['hi me hi', 'hi you']
        explainer = Lime(['c1', 'c2'], [0, 1], TestLime.constant_fn, tokenize_fn=TestLime.space_tokenize_fn)
        details_dict = explainer.explain_with_details(samples)
        sorted_details_lst = ExplainerBase.sort_details(details_dict, ExplainerBase.COUNT_KEY)

        self.assertEqual(sorted_details_lst[0][0], 'hi')
    
    def test_tokenize_for_lime(self):
        from phd_utils.explainers import tokenize_for_lime
        text_str = "I don't want to     talk  about it.I was like that"
        tokens_lst = tokenize_for_lime(text_str)

        self.assertEqual(len(tokens_lst), 12)
