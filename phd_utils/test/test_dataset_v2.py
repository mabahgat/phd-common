import unittest

from phd_utils.datasets_v2 import ClPsych, AdaptedAffectInTweets, EmoContext, AGNews, UrbanDictWithLiwc

class TestClPsych(unittest.TestCase):

    def test_load(self):
        dataset = ClPsych(valid_prcnt=0.2)
        dataset.load()
        x_train, y_train = dataset.training_examples()
        self.assertEqual(len(x_train), len(y_train))
        self.assertEqual(len(x_train), 398)

        x_valid, y_valid = dataset.validation_examples()
        self.assertEqual(len(x_valid), len(y_valid))
        self.assertEqual(len(x_valid), 98)

        x_test, y_test = dataset.testing_examples()
        self.assertEqual(len(x_test), len(y_test))
        self.assertEqual(len(x_test), 125)


class TestAdaptedAffectInTweets(unittest.TestCase):

    def test_load(self):
        dataset = AdaptedAffectInTweets()
        dataset.load()
        x_train, y_train = dataset.training_examples()
        self.assertEqual(len(x_train), len(y_train))
        self.assertEqual(len(x_train), 7102)

        x_valid, y_valid = dataset.validation_examples()
        self.assertEqual(len(x_valid), len(y_valid))
        self.assertEqual(len(x_valid), 1464)

        x_test, y_test = dataset.testing_examples()
        self.assertEqual(len(x_test), len(y_test))
        self.assertEqual(len(x_test), 4068)


class TestEmoContext(unittest.TestCase):

    def test_load(self):
        dataset = EmoContext()
        dataset.load()
        x_train, y_train = dataset.training_examples()
        self.assertEqual(len(x_train), len(y_train))
        self.assertEqual(len(x_train), 12168)

        x_valid, y_valid = dataset.validation_examples()
        self.assertEqual(len(x_valid), len(y_valid))
        self.assertEqual(len(x_valid), 1520)

        x_test, y_test = dataset.testing_examples()
        self.assertEqual(len(x_test), len(y_test))
        self.assertEqual(len(x_test), 1524)


class TestAGNews(unittest.TestCase):

    def test_load(self):
        dataset = AGNews(valid_prcnt=0.2)
        dataset.load()
        x_train, y_train = dataset.training_examples()
        self.assertEqual(len(x_train), len(y_train))
        self.assertEqual(len(x_train), 96000)

        x_valid, y_valid = dataset.validation_examples()
        self.assertEqual(len(x_valid), len(y_valid))
        self.assertEqual(len(x_valid), 24000)

        x_test, y_test = dataset.testing_examples()
        self.assertEqual(len(x_test), len(y_test))
        self.assertEqual(len(x_test), 7600)


class TestUrbanDictWithLiwc(unittest.TestCase):

    def test_load(self):
        dataset = UrbanDictWithLiwc(0.1)
        dataset.load()
        x_train, y_train = dataset.training_examples()
        x_valid, y_valid = dataset.validation_examples()

        # TODO better test cases
        self.assertEqual(len(x_train), len(dataset.train_index_lst))
        self.assertEqual(len(x_valid), len(dataset.valid_index_lst))


class TestUrbanDictWithLiwcEmo(unittest.TestCase):

    def test_load(self):
        dataset = UrbanDictWithLiwc(0.1, config_dict={'labels': 'liwc_emo'})
        dataset.load()
        x_train, y_train = dataset.training_examples()
        x_valid, y_valid = dataset.validation_examples()

        # TODO better test cases
        self.assertEqual(len(x_train), len(dataset.train_index_lst))
        self.assertEqual(len(x_valid), len(dataset.valid_index_lst))
