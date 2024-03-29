from abc import ABC, abstractmethod, abstractclassmethod
from tqdm import tqdm
import json
import csv
from random import Random
import string
from typing import List
import pandas as pd
from sklearn import metrics

from phd_utils import global_config
from phd_utils.providers import Provider, PassThroughProvider


DISABLE_TQDM = False


class DatasetBase(ABC):

    @abstractmethod
    def load(self, preprocessors_lst=None, provider:Provider=PassThroughProvider(), randomize=True, random_seed=0):
        pass

    @abstractmethod
    def training_examples(self):
        pass

    @abstractmethod
    def validation_examples(self):
        pass

    @abstractmethod
    def testing_examples(self):
        pass

    @abstractmethod
    def size(self):
        pass
    
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def class_count(self):
        """
        Number of classes in the dataset (assuming classification and not regression)
        Implementation should be able to provide that number before a class to `load`
        """
        pass
    
    def class_names(self):
        """
        Returns none for no clsas names list defined
        """
        return None

    @staticmethod
    def calculate_split(total, prcnt):
        second_part = int(total * prcnt)
        first_part = total - second_part
        return first_part, second_part
    
    @staticmethod
    def split_list(full_lst, prcnt):
        if not full_lst or not len(full_lst):
            raise ValueError("List can not be empty")
        first_part, _ = DatasetBase.calculate_split(len(full_lst), prcnt)
        return full_lst[0:first_part], full_lst[first_part:]
    
    @staticmethod
    def randomize(x_lst, y_lst, random_seed):
        zipped_lst = list(zip(x_lst, y_lst))
        rand = DatasetBase.random_with_seed(random_seed)
        rand.shuffle(zipped_lst)
        shuffled_x_lst, shuffled_y_lst = zip(*zipped_lst)
        return list(shuffled_x_lst), list(shuffled_y_lst)
    
    @staticmethod
    def preprocess(x_lst, preprocessor_lst):
        if not preprocessor_lst:
            return x_lst
        done = []
        for x in x_lst:
            current = x
            for p in preprocessor_lst:
                current = p.apply(current)
            done.append(current)
        return done

    @staticmethod
    def random_with_seed(random_seed=0):
        rand = Random()
        rand.seed(random_seed)
        return rand


class RandomTextDataset(DatasetBase):

    def __init__(self, class_count=4, train_count=100, valid_count=50, test_count=50, min_sent_len=10, max_sent_len=30, min_word_len=1, max_word_len=15):
        self.__train = None
        self.__valid = None
        self.__test = None

        self.__class_count = class_count
        self.__train_count = train_count
        self.__valid_count = valid_count
        self.__test_count = test_count
        self.__min_sent_len = min_sent_len
        self.__max_sent_len = max_sent_len
        self.__min_word_len = min_word_len
        self.__max_word_len = max_word_len

    def load(self, preprocessors_lst=None, provider:Provider=PassThroughProvider(), randomize=True, random_seed=0):
        rand_obj = Random()
        rand_obj.seed(random_seed)

        x_train = RandomTextDataset.__generate_random_set(rand_obj, self.__train_count, self.__min_sent_len, self.__max_sent_len)
        y_train = RandomTextDataset.__generate_random_class_labels(rand_obj, self.__train_count, self.__class_count)
        x_valid = RandomTextDataset.__generate_random_set(rand_obj, self.__valid_count, self.__min_sent_len, self.__max_sent_len)
        y_valid = RandomTextDataset.__generate_random_class_labels(rand_obj, self.__valid_count, self.__class_count)
        x_test = RandomTextDataset.__generate_random_set(rand_obj, self.__test_count, self.__min_sent_len, self.__max_sent_len)
        y_test = RandomTextDataset.__generate_random_class_labels(rand_obj, self.__test_count, self.__class_count)

        self.__train = provider.apply(x_train, y_train)
        self.__valid = provider.apply(x_valid, y_valid)
        self.__test = provider.apply(x_test, y_test)


    def training_examples(self):
        return self.__train

    def validation_examples(self):
        return self.__valid

    def testing_examples(self):
        return self.__test

    def size(self):
        return {
            'train': self.__train_count,
            'valid': self.__valid_count,
            'test': self.__test_count
        }

    def name(self):
        return "random_text"

    def class_count(self):
        return self.__class_count
    
    def class_names(self):
        return [str(i) for i in range(self.__class_count)]

    @staticmethod
    def __generate_random_class_labels(rand_obj, sample_count, class_count):
        return [rand_obj.randint(0, class_count - 1) for i in range(sample_count)]
    
    @staticmethod
    def __generate_random_set(rand_obj, sentence_count, min_sent_len, max_sent_len):
        sentence_lst = []
        for _ in range(sentence_count):
            word_count = rand_obj.randint(min_sent_len, max_sent_len)
            sentence_str = RandomTextDataset.__generate_random_sentence(rand_obj, word_count, 1, 12)
            sentence_lst.append(sentence_str)
        return sentence_lst
    
    @staticmethod
    def __generate_random_sentence(rand_obj, word_count, min_word_len, max_word_len):
        letters_lst = string.ascii_lowercase
        words_lst = []
        for _ in range(word_count):
            word_len = rand_obj.randint(min_word_len, max_word_len)
            word_str = ''.join([rand_obj.choice(letters_lst) for j in range(word_len)])
            words_lst.append(word_str)
        return ' '.join(words_lst)


class LocalDatasetBase(DatasetBase):

    @staticmethod
    @abstractclassmethod
    def get_path():
        """
        Return path information in human readable format
        """
        pass
    
    @staticmethod
    def load_lines(path_str: str):
        lines_lst = []
        with open(path_str, 'r', encoding='utf8') as input_file:
            for line in tqdm(input_file.readlines(), disable=DISABLE_TQDM):
                lines_lst.append(line.strip())
        return lines_lst
    
    @staticmethod
    def load_csv(path_str: str, delimiter=',', quoting=csv.QUOTE_NONE):
        records_lst = []
        with open(path_str, 'r', encoding='utf8') as csv_file:
            reader = csv.reader(csv_file, delimiter=delimiter, quoting=quoting)
            for row in tqdm(reader, desc='load from csv {}'.format(path_str), disable=DISABLE_TQDM):
                records_lst.append(row)
        return records_lst

    def __init(self):
        self._train = None
        self._valid = None
        self._test = None

    def load(self, preprocessors_lst=None, provider:Provider=PassThroughProvider(), randomize=True, random_seed=0):
        x_train, y_train = self._file_sys_load('train')
        x_valid, y_valid = self._file_sys_load('valid')
        x_test, y_test = self._file_sys_load('test')

        if randomize:
            x_train, y_train = DatasetBase.randomize(x_train, y_train, random_seed)
            x_valid, y_valid = DatasetBase.randomize(x_valid, y_valid, random_seed)
        
        x_train = DatasetBase.preprocess(x_train, preprocessors_lst)
        x_valid = DatasetBase.preprocess(x_valid, preprocessors_lst)
        x_test = DatasetBase.preprocess(x_test, preprocessors_lst)
        
        self._train = provider.apply(x_train, y_train)
        self._valid = provider.apply(x_valid, y_valid)
        self._test = provider.apply(x_test, y_test)
    
    @abstractmethod
    def _file_sys_load(self, set_type_str: str):
        pass

    def training_examples(self):
        return self._train

    def validation_examples(self):
        return self._valid

    def testing_examples(self):
        return self._test


class LocalDatasetWithOptionalValidation(LocalDatasetBase):

    def __init__(self, valid_prcnt):
        if valid_prcnt > 1 or valid_prcnt < 0:
            raise ValueError("Valid set percentage {} is not valid".format(valid_prcnt))
        self._valid_prcnt = valid_prcnt
        self.train_index_lst = None
        self.valid_index_lst = None
        super().__init__()

    def load(self, preprocessors_lst=None, provider:Provider=PassThroughProvider(), randomize=True, random_seed=0):
        x_train, y_train = self._file_sys_load('train')
        x_test, y_test = self._file_sys_load('test')
        train_index_lst = list(range(len(x_train))) # to keep track of instances
        
        if self._valid_prcnt > 0:
            y_train_org = y_train
            x_train, y_train, x_valid, y_valid = self.__split_train_valid(x_train, y_train)
            train_index_lst, _, valid_index_lst, _ = self.__split_train_valid(train_index_lst, y_train_org)
        
        if randomize:
            x_train_index_pair, y_train = DatasetBase.randomize(zip(x_train, train_index_lst), y_train, random_seed)
            x_valid_index_pair, y_valid = DatasetBase.randomize(zip(x_valid, valid_index_lst), y_valid, random_seed)
            x_train, train_index_lst = zip(*x_train_index_pair)
            x_valid, valid_index_lst = zip(*x_valid_index_pair)
        
        x_train = DatasetBase.preprocess(x_train, preprocessors_lst)
        x_valid = DatasetBase.preprocess(x_valid, preprocessors_lst)
        x_test = DatasetBase.preprocess(x_test, preprocessors_lst)
        
        self._train = provider.apply(x_train, y_train)
        self._valid = provider.apply(x_valid, y_valid)
        self._test = provider.apply(x_test, y_test)

        self.train_index_lst = list(train_index_lst)
        self.valid_index_lst = list(valid_index_lst)
    
    def __split_train_valid(self, x_lst, y_lst):
        label_to_example_dict = {}
        for x, y in zip(x_lst, y_lst):
            if y not in label_to_example_dict:
                label_to_example_dict[y] = []
            label_to_example_dict[y].append(x)
        
        x1_lst = []
        y1_lst = []
        x2_lst = []
        y2_lst = []
        for label in label_to_example_dict.keys():
            x_part1_lst, x_part2_lst = DatasetBase.split_list(label_to_example_dict[label], self._valid_prcnt)
            x1_lst.extend(x_part1_lst)
            y1_lst.extend([label for i in range(len(x_part1_lst))])
            x2_lst.extend(x_part2_lst)
            y2_lst.extend([label for i in range(len(x_part2_lst))])
        
        return x1_lst, y1_lst, x2_lst, y2_lst


class ClPsych(LocalDatasetWithOptionalValidation):

    @staticmethod
    def get_path():
        return {
            'train': global_config.datasets.clpysch.task_a.path.train,
            'test': global_config.datasets.clpysch.task_a.path.test
        }
    
    def _file_sys_load(self, set_type_str: str):
        pathes_dict = ClPsych.get_path()
        with open(pathes_dict[set_type_str], 'r', encoding='utf8') as input_file:
            file_json = json.load(input_file)
            
            text_lst = []
            label_lst = []

            for user in file_json.keys():
                label_int = ClPsych.__label_to_index(file_json[user]['label'])
                
                all_text = []
                for post in file_json[user]['posts']:
                    all_text.append('{} {}'.format(post['raw']['title'], post['raw']['body']))
                text_str = ' '.join(all_text)

                text_lst.append(text_str)
                label_lst.append(label_int)

            return text_lst, label_lst
    
    @staticmethod
    def __label_to_index(label_str):
        return ord(label_str) - ord('a')

    def size(self):
        raise NotImplementedError()

    def name(self):
        return 'CLPsych_valid{}'.format(self._valid_prcnt)

    def class_count(self):
        return 4
    
    def class_names(self):
        return ['no risk', 'low risk', 'moderate risk', 'severe risk']


class AdaptedAffectInTweets(LocalDatasetBase):
    """
    Adapated affect in Tweets classifies the type of affection in a tweet.
    This is different from the original published goal which was to measure the intensity of emotion.
    """

    __LABELS_TO_INDEX_DICT = {
        'anger': 0,
        'fear': 1,
        'joy': 2,
        'sadness': 3
    }

    @staticmethod
    def get_path():
        return {
            'train': global_config.datasets.affect_in_tweets.path.train,
            'valid': global_config.datasets.affect_in_tweets.path.dev,
            'test': global_config.datasets.affect_in_tweets.path.test
        }

    def _file_sys_load(self, set_key_str: str):
        records_lst = LocalDatasetBase.load_csv(AdaptedAffectInTweets.get_path()[set_key_str], delimiter='\t')[1:]
        data_lst = [[row[1], AdaptedAffectInTweets.__label_to_index(row[2])] for row in records_lst]
        x_tuple, y_tuple = zip(*data_lst)
        return list(x_tuple), list(y_tuple)
    
    @staticmethod
    def __label_to_index(label_str):
        return AdaptedAffectInTweets.__LABELS_TO_INDEX_DICT[label_str]

    def name(self):
        return "adapted_affect_in_tweets"

    def size(self):
        raise NotImplementedError()

    def class_count(self):
        return 4


class EmoContext(LocalDatasetBase):

    @staticmethod
    def get_path():
        return {
            'classify_emotions': {
                'train': global_config.datasets.emocontext.classify_emotions.path.train,
                'valid': global_config.datasets.emocontext.classify_emotions.path.dev,
                'test': global_config.datasets.emocontext.classify_emotions.path.test
                },
            'emotions_vs_none': {
                'train': global_config.datasets.emocontext.emotions_vs_none.path.train,
                'valid': global_config.datasets.emocontext.emotions_vs_none.path.dev,
                'test': global_config.datasets.emocontext.emotions_vs_none.path.test
            },
            'classify_all': {
                'train': global_config.datasets.emocontext.classify_all.path.train,
                'valid': global_config.datasets.emocontext.classify_all.path.dev,
                'test': global_config.datasets.emocontext.classify_all.path.test
            }
        }
    
    __LABELS_TO_INDEX_DICT = {
        'classify_emotions': {
            'angry': 0,
            'happy': 1,
            'sad': 2
        },
        'emotions_vs_none': {
            'angry': 0,
            'happy': 0,
            'sad': 0,
            'others': 1
        },
        'classify_all': {
            'angry': 0,
            'happy': 1,
            'sad': 2,
            'others': 3
        }
    }

    def __init__(self, type_str='classify_emotions'):
        if type_str not in set(EmoContext.__LABELS_TO_INDEX_DICT.keys()):
            raise ValueError('unknown dataset type {}'.format(type_str))
        self.__data_set_type_str = type_str

    def _file_sys_load(self, set_key_str: str):
        data_path_str = EmoContext.get_path()[self.__data_set_type_str][set_key_str]
        records_lst = LocalDatasetBase.load_csv(data_path_str, delimiter='\t')
        data_lst = [(' '.join(row[1:3]), self.__label_to_index(row[4])) for row in records_lst]
        x_tuple, y_tuple = zip(*data_lst)
        return list(x_tuple), list(y_tuple)
    
    def __label_to_index(self, label_str: str):
        return EmoContext.__LABELS_TO_INDEX_DICT[self.__data_set_type_str][label_str]

    def size(self):
        raise NotImplementedError()

    def name(self):
        return '{}_emocontext'.format(self.__data_set_type_str)

    def class_count(self):
        return len(EmoContext.__LABELS_TO_INDEX_DICT[self.__data_set_type_str].keys())


class AGNews(LocalDatasetWithOptionalValidation):

    @staticmethod
    def get_path():
        return {
            'train': global_config.datasets.agnews.path.train,
            'test': global_config.datasets.agnews.path.test
        }
    
    def _file_sys_load(self, set_type_str: str):
        records_lst = LocalDatasetBase.load_csv(AGNews.get_path()[set_type_str], quoting=csv.QUOTE_ALL)
        data_lst = [[row[1], AGNews.__label_to_index(row[0])] for row in records_lst]
        x_tuple, y_tuple = zip(*data_lst)
        return list(x_tuple), list(y_tuple)
    
    @staticmethod
    def __label_to_index(label_str: str):
        return int(label_str) - 1

    def size(self):
        raise NotImplementedError()

    def name(self):
        return 'ag_news_valid{}'.format(self._valid_prcnt)

    def class_count(self):
        return 4


class UrbanDictWithLiwc(LocalDatasetWithOptionalValidation):

    def __init__(self, valid_prcnt, config_dict=None):
        self._config_dict = UrbanDictWithLiwc.__init_dict(config_dict)
        self.__size_dict = None # lazy compute size and buffer
        super().__init__(valid_prcnt)
    
    @staticmethod
    def __init_dict(config_dict: str):
        def set_if_not_set(name_str: str, value):
            if name_str not in config_dict:
                config_dict[name_str] = value
        if not config_dict:
            config_dict = {}
        set_if_not_set('labels', 'liwc_root_9') # liwc_root_9, liwc_affect_3 ...
        set_if_not_set('train_type', 'exact') # exact, all
        set_if_not_set('selection_mode', 'top1') # top1, minDiff1, top10, minDiff10
        set_if_not_set('test_selection_mode', 'top1') # top1, top10
        set_if_not_set('text_mode', 'merge') # merge, meaning, example
        set_if_not_set('eval_mode', 'single') # single, any
        set_if_not_set('target_labels_count', 'single') # single, multiple
        return config_dict
    
    def get_path(self):
        """
        Return data files path
        """
        return {
            'test': global_config.datasets.ud_liwc[self._config_dict['labels']][self._config_dict['train_type']].test[self._config_dict['test_selection_mode']],
            'train': global_config.datasets.ud_liwc[self._config_dict['labels']][self._config_dict['train_type']].train[self._config_dict['selection_mode']]
        }
    
    def _file_sys_load(self, set_type_str: str):
        content_df = pd.read_csv(self.get_path()[set_type_str], index_col=0)
        self._process_content_in_place(content_df)
        x_tuple = self._create_input_text_from_dataframe(content_df)
        label_select_random = DatasetBase.random_with_seed(0)
        y_tuple = content_df['liwc'].astype(str).apply(lambda label_str: self.label_to_index(label_str, label_select_random)).to_list()
        return x_tuple, y_tuple

    def _process_content_in_place(self, content_df):
        text_mode = self._config_dict['text_mode']
        if text_mode in set(['merge_with_tags', 'tags', 'merge_with_tags_keep_empty']):
            content_df['tagList'] = content_df.tagList.apply(UrbanDictWithLiwc._tags_to_list)
            if text_mode != 'merge_with_tags_keep_empty':
                content_df = content_df[content_df.tagList.apply(lambda l: len(l) != 0)]
            content_df['tagList'] = content_df.tagList.apply(lambda l: ' '.join(l))
        
    @staticmethod
    def _tags_to_list(tags_str: List[str]):
        tags_str = str(tags_str)
        if tags_str == 'nan':
            return []
        return list(filter(len, tags_str.split('#')))
    
    def _create_input_text_from_dataframe(self, content_df):
        if self._config_dict['text_mode'] == 'merge':
            return (content_df['meaning'].astype(str) + ' ' + content_df['example'].astype(str)).to_list()
        elif self._config_dict['text_mode'] == 'definition':
            return content_df['meaning'].astype(str).to_list()
        elif self._config_dict['text_mode'] == 'example':
            return content_df['example'].astype(str).to_list()
        elif self._config_dict['text_mode'] == 'merge_with_tags' or self._config_dict['text_mode'] == 'merge_with_tags_keep_empty':
            return (content_df['tagList'].astype(str) + ' ' + content_df['meaning'].astype(str) + ' ' + content_df['example'].astype(str)).to_list()
        elif self._config_dict['text_mode'] == 'tags':
            return content_df['tagList'].astype(str).to_list()
        else:
            raise ValueError('Unexpected mode {}'.format(self._config_dict['text_mode']))
    
    def class_names(self):
        labels_lst = global_config.datasets.ud_liwc[self._config_dict['labels']].labels.split(',')
        return [label_str for label_str in labels_lst]
    
    def label_to_index(self, label_list_str:str, rand:Random=None):
        label_index_lst = [self.class_names().index(label_str) for label_str in label_list_str.split("|")] #FIXME - avoid using different casing
        if self._config_dict['target_labels_count'] == 'single':
            return label_index_lst[0]
        elif self._config_dict['target_labels_count'] == 'multiple':
            return label_index_lst
        elif self._config_dict['target_labels_count'] == 'single_random':
            return rand.choice(label_index_lst)
        else:
            raise ValueError('Unexpected target labels count {}'.format(self._config_dict['target_labels_count']))
    
    def class_count(self):
        return len(self.class_names())
    
    def size(self):
        def count_lines(file_path_str):
            return sum(1 for _ in open(file_path_str))
        if not self.__size_dict:
            train_size = count_lines(self.get_path()['train'])
            test_size = count_lines(self.get_path()['test'])
            if self._valid_prcnt:
                valid_size = int(self._valid_prcnt * train_size)
                train_size -= valid_size
            else:
                valid_size = 0
            self.__size_dict = {
                'train': train_size,
                'valid': valid_size,
                'test': test_size
            }
        return self.__size_dict
        
    
    def name(self):
        config_str = '-'.join(list(self._config_dict.items()))
        return 'ud_liwc_{}_valid{}'.format(config_str, self._valid_prcnt)


class UrbanDictWordsOnlyWithLiwc(UrbanDictWithLiwc):

    def __init__(self, valid_prcnt, config_dict=None):
        super().__init__(valid_prcnt, config_dict)
    
    def _create_input_text_from_dataframe(self, content_df):
        return content_df.word.to_list()