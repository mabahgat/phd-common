import unittest

from phd_utils.annotators import liwc
from phd_utils.annotators.embeddings import SentenceEmbeddings

class TestSentenceEmbeddings(unittest.TestCase):

    def test_get(self):
        se = SentenceEmbeddings()
        e = se.get('My big sentence')
        self.assertIsNotNone(e)
    
    def test_distance(self):
        se = SentenceEmbeddings()
        e1 = se.get('My big sentence')
        e2 = se.get('My large sentence')
        d = SentenceEmbeddings.distance(e1, e2)
        self.assertIsInstance(d, int)