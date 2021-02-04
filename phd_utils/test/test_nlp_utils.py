import unittest
from phd_utils import nlp_utils

class TestNlpUtils(unittest.TestCase):
    
    def test_match_pronoun_present(self):
        
        self.assertEqual(nlp_utils.match_pronoun_present('is', 'i'), 'am')
        self.assertEqual(nlp_utils.match_pronoun_present('is', 'you'), 'are')
        self.assertEqual(nlp_utils.match_pronoun_present('\'s', 'you'), 'are')
        self.assertEqual(nlp_utils.match_pronoun_present('am', 'he'), 'is')
        self.assertEqual(nlp_utils.match_pronoun_present('\'m', 'She'), 'is')

        self.assertEqual(nlp_utils.match_pronoun_present('play', 'he'), 'plays')
        self.assertEqual(nlp_utils.match_pronoun_present('plays', 'i'), 'play')
        
        self.assertEqual(nlp_utils.match_pronoun_present("can't", 'she'), "can't")
        self.assertEqual(nlp_utils.match_pronoun_present("isn't", 'i'), 'am not')
        self.assertEqual(nlp_utils.match_pronoun_present("isn't", 'you'), "aren't")
        
