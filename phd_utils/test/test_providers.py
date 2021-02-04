import unittest

from phd_utils.providers import GroupedSamplesProvider

class TestGroupedSamplesProvider(unittest.TestCase):

    def test_apply(self):
        samples_lst = [
            'me',
            'you',
            'i',
            'them',
            'mine'
        ]
        labels_lst = [
            'a',
            'b',
            'a',
            'c',
            'a'
        ]
        provider = GroupedSamplesProvider()
        samples_dict = provider.apply(samples_lst, labels_lst)

        self.assertEqual(len(samples_dict['a']), 3)
