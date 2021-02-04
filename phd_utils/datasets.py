# Provide an API for laoding different datasets to avoid rewriting loading code in each sheet

import os
import random
import json
import re
from abc import ABC, abstractmethod
from tqdm import tqdm

DISABLE_TQDM = False

class Dataset(ABC):

    @staticmethod
    def load_lines(path_str: str):
        lines_lst = []
        with open(path_str, 'r', encoding='utf8') as input_file:
            for line in tqdm(input_file.readlines(), disable=DISABLE_TQDM):
                lines_lst.append(line.strip())
        return lines_lst

    @staticmethod
    def compute_splits(full_size, train_prcnt, valid_prcnt, test_prcnt):
        train_size = int(full_size * train_prcnt)
        valid_size = int(full_size * valid_prcnt)
        test_size = full_size - train_size - valid_size
        return train_size, valid_size, test_size
    
    @staticmethod
    def randomize(lst):
        random.shuffle(lst)

    @abstractmethod
    def size(self):
        pass


class EmoNet(Dataset):

    __FILE_PATH = '/home/mbahgat/ws/work/datasets/emo_net/emo_net_8class.tsv'
    
    ID_LABEL = 'id'
    LABEL_LABEL = 'label'
    TEXT_LABEL = 'text'

    HANDLE_MASK = '@HANDLE' # Mask tweet handles for users

    def __init__(self, full_set_lst):
        self.full_set_lst = full_set_lst
        self.examples_per_label_dict = self.__generate_samples_per_label()

    @staticmethod
    def load_from_disk():
        lines_lst = Dataset.load_lines(EmoNet.__FILE_PATH)
        full_set_lst = []
        for line in lines_lst:
            try:
                tweet_id, tweet_label, tweet_text = line.split('\t')
                full_set_lst.append({
                    EmoNet.ID_LABEL: tweet_id,
                    EmoNet.LABEL_LABEL: tweet_label,
                    EmoNet.TEXT_LABEL: tweet_text
                })
            except:
                print('ERROR: failed to read line {}'.format(line))
        return EmoNet(full_set_lst)

    def __generate_samples_per_label(self):
        labels_dict = {}
        for example in self.full_set_lst:
            label_str = example[EmoNet.LABEL_LABEL]
            if label_str not in labels_dict:
                labels_dict[label_str] = []
            labels_dict[label_str].append(example)
        return labels_dict
    
    def size(self):
        return len(self.full_set_lst)

    def generate_sets(self, train_prcnt=0.8, valid_prcnt=0.1, test_prcnt=0.1, random_seed=None, mask_handles=False):
        if train_prcnt + valid_prcnt + test_prcnt != 1:
            raise Exception("Incorrect split percentages {}, {} and {}".format(train_prcnt, valid_prcnt, test_prcnt))

        train_set_lst = []
        valid_set_lst = []
        test_set_lst = []
        for label_str in self.examples_per_label_dict.keys():
            train_lst, valid_lst, test_lst = self.__split_set_for_class(label_str, train_prcnt, valid_prcnt, test_prcnt)
            train_set_lst.extend(train_lst)
            valid_set_lst.extend(valid_lst)
            test_set_lst.extend(test_lst)
        
        if random_seed is not None:
            random.seed(random_seed)
            Dataset.randomize(train_set_lst)
            Dataset.randomize(valid_set_lst)
            Dataset.randomize(test_set_lst)
        
        if mask_handles:
            [EmoNet.__mask_handle(example) for example in train_set_lst]
            [EmoNet.__mask_handle(example) for example in valid_set_lst]
            [EmoNet.__mask_handle(example) for example in test_set_lst]

        return train_set_lst, valid_set_lst, test_set_lst
    
    @staticmethod
    def __mask_handle(example_dict):
        example_dict[EmoNet.TEXT_LABEL] = re.sub(r"@\w{1,15}", EmoNet.HANDLE_MASK, example_dict[EmoNet.TEXT_LABEL])
    
    def __split_set_for_class(self, label_str, train_prcnt, valid_prcnt, test_prcnt):
        class_lst = self.examples_per_label_dict[label_str]
        train_size, valid_size, _ = Dataset.compute_splits(len(class_lst), train_prcnt, valid_prcnt, test_prcnt)
        return class_lst[0:train_size], class_lst[train_size:train_size+valid_size], class_lst[train_size+valid_size:]


class ClPsych(Dataset):

    _TRAIN_PATH = '/home/mbahgat/ws/work/jws/clpysch/data/task_A_train.posts.json'
    _TEST_PATH = '/home/mbahgat/ws/work/jws/clpysch/data/task_A_test.posts.json'

    USER_ID_LABEL = 'user'
    TEXT_LABEL = 'text'
    LABEL_LABEL = 'label'

    @staticmethod
    def load_from_disk():
        train_lst, train_dict = ClPsych._load_file(ClPsych._TRAIN_PATH)
        test_lst, test_dict = ClPsych._load_file(ClPsych._TEST_PATH)
        return ClPsych(train_lst, train_dict, test_lst, test_dict)
    
    @staticmethod
    def _load_file(path_str):
        with open(path_str, 'r', encoding='utf8') as input_file:
            file_json = json.load(input_file)
            
            label_to_content_dict = {}
            all_examples_lst = []

            for user in file_json.keys():
                label_str = file_json[user]['label']
                
                all_text = []
                for post in file_json[user]['posts']:
                    all_text.append('{} {}'.format(post['raw']['title'], post['raw']['body']))
                text_str = ' '.join(all_text)

                example = {
                    ClPsych.USER_ID_LABEL: user,
                    ClPsych.TEXT_LABEL: text_str,
                    ClPsych.LABEL_LABEL: label_str
                }

                if label_str not in label_to_content_dict:
                    label_to_content_dict[label_str] = []
                label_to_content_dict[label_str].append(example)
                all_examples_lst.append(example)

            return all_examples_lst, label_to_content_dict

    def __init__(self, train_lst, train_dict, test_lst, test_dict):
        self.train_lst = train_lst
        self.train_dict = train_dict
        self.test_lst = test_lst
        self.test_dict = test_dict
    
    def size(self):
        return {
            'train_size': len(self.train_lst),
            'test_size': len(self.test_lst)
        }
    
    def generate_sets(self, train_prcnt=0.8, valid_prcnt=0.2, random_seed=None):
        if train_prcnt + valid_prcnt != 1:
            raise Exception("Incorrect split percentage {} and {}".format(train_prcnt, valid_prcnt))
        train_lst = []
        valid_lst = []
        for label_str in self.train_dict.keys():
            class_train_lst, class_valid_lst = self.__split_set_for_class(label_str, train_prcnt, valid_prcnt)
            train_lst.extend(class_train_lst)
            valid_lst.extend(class_valid_lst)
        if random_seed is not None:
            random.seed(random_seed)
            Dataset.randomize(train_lst)
            Dataset.randomize(valid_lst)
        return train_lst, valid_lst, self.test_lst
    
    def __split_set_for_class(self, label_str, train_prcnt, valid_prcnt):
        class_lst = self.train_dict[label_str]
        train_size, _, _ = Dataset.compute_splits(len(class_lst), train_prcnt, valid_prcnt, test_prcnt=0)
        return class_lst[0:train_size], class_lst[train_size:]
    

class PronounSwitch:
    #__TONES = ['subject', 'object', 'possive adjective', 'possive pronoun', 'reflexive pronoun']

    FIRST_PERSON_lst = ['i', 'me', 'my', 'mine', 'myself']
    SECOND_PERSON_lst = ['you', 'you', 'your', 'yours', 'yourself']
    MALE_THIRD_PERSON_lst = ['he', 'him', 'his', 'his', 'himself']
    FEMALE_THIRD_PERSON_lst = ['she', 'her', 'her', 'hers', 'herself']

    def __init__(self, to_pronoun_lst, from_pronoun_lst):
        self.__to_pronoun_lst = to_pronoun_lst
        self.__from_pronoun_lst = from_pronoun_lst
        self.__from_pronoun_pattern_lst = [r"\b{}\b".format(txt_str) for txt_str in from_pronoun_lst]
    
    def _switch(self, text_str: str) -> str:
        new_text_str = text_str
        for pattern_re, to_str, from_str in zip(self.__from_pronoun_pattern_lst, self.__to_pronoun_lst, self.__from_pronoun_lst):
            tmp_mark_str = "<<{}-MARK>>".format(from_str)
            new_text_str = re.sub(pattern_re, tmp_mark_str, new_text_str, flags=re.I)
            new_text_str = new_text_str.replace(tmp_mark_str, to_str)
        return new_text_str


class ClPyschWithPronounSwitch(ClPsych, PronounSwitch):

    def __init__(self, to_pronoun_lst, from_pronoun_lst, train_lst, train_dict, test_lst, test_dict):
        PronounSwitch.__init__(self, to_pronoun_lst, from_pronoun_lst)
        ClPsych.__init__(self, train_lst, train_dict, test_lst, test_dict)

    def generate_sets(self, train_prcnt=0.8, valid_prcnt=0.2, random_seed=None):
        train_set_lst, valid_set_lst, test_set_lst = ClPsych.generate_sets(
            self,
            train_prcnt=train_prcnt,
            valid_prcnt=valid_prcnt, 
            random_seed=random_seed)
        train_set_lst = [self.__do_switch(example_obj) for example_obj in train_set_lst]
        valid_set_lst = [self.__do_switch(example_obj) for example_obj in valid_set_lst]
        test_set_lst = [self.__do_switch(example_obj) for example_obj in test_set_lst]
        return train_set_lst, valid_set_lst, test_set_lst
    
    @staticmethod
    def load_from_disk(to_pronoun_lst, from_pronoun_lst=PronounSwitch.FIRST_PERSON_lst):
        train_lst, train_dict = ClPsych._load_file(ClPsych._TRAIN_PATH)
        test_lst, test_dict = ClPsych._load_file(ClPsych._TEST_PATH)
        return ClPyschWithPronounSwitch(to_pronoun_lst, from_pronoun_lst, train_lst, train_dict, test_lst, test_dict)

    def __do_switch(self, example_obj):
        new_text_str =self._switch(example_obj[EmoNet.TEXT_LABEL])
        example_obj[EmoNet.TEXT_LABEL] = new_text_str
        return example_obj


class ClPsychWithYouSwitch(ClPyschWithPronounSwitch):

    def __init__(self, train_lst, train_dict, test_lst, test_dict):
        ClPyschWithPronounSwitch.__init__(
            self,
            PronounSwitch.FIRST_PERSON_lst,
            PronounSwitch.SECOND_PERSON_lst,
            train_lst,
            train_dict,
            test_lst,
            test_dict)
