from abc import ABC, abstractmethod
from typing import List, Dict, Set, Union
from phd_utils import global_config


class DictBase(ABC):

    @abstractmethod
    def label_word(self, word_str: str) -> List[str]:
        """
        Labels a single word with one or more labels
        :param word_str: word string
        :return: list of labels - can be an empty array if no labels
        """
        pass

    def label_sentence(self, sentence_lst: List[str]) -> List[List[str]]:
        """
        For every token in sentence return
        :param sentence_lst: List of string words
        :return: A list of annotations for every word
        """
        return [self.label_word(w) for w in sentence_lst]

    def label(self, content):
        if type(content) == str:
            return self.label_word(content)
        elif type(content) == List:
            return self.label_sentence(content)
        else:
            raise TypeError('passed content is of unknown type. Only accepting strings and lists')


class DictWithSelectLabels(DictBase, ABC):

    def __init__(self, label_tree_dict: Dict[str, str], selected_labels_set: Set[str]):
        self.__label_tree_dict = label_tree_dict
        self.__selected_labels_set = selected_labels_set
        self.__mapping_cache_dict = {}

    @staticmethod
    def load_from_csv(path_str) -> Dict[str, str]:  # TODO move to a tools class
        word_label_dict = {}
        with open(path_str) as dict_file:
            for line in dict_file.readlines():
                word, label = line.strip().split(',')
                label = label.strip().lower()
                word_label_dict[word] = label
        return word_label_dict

    def load_dict_and_fix(self, path_str: str) -> Dict[str, str]:
        """
        Loads content of csv dictionary and fixes it based on the selected labels by mapping labels to their parents and
        removing entries not within the selected labels
        :param path_str: disk path string
        :return: word to label dictionary
        """
        word_label_dict = DictWithSelectLabels.load_from_csv(path_str)
        if self.__selected_labels_set is not None:
            word_label_dict = {w: self.map_to_existing_parent(l) for w, l in word_label_dict.items()}
            word_label_dict = {w: l for w, l in word_label_dict.items() if l is not None}
        return word_label_dict

    def map_to_existing_parent(self, label_str) -> Union[str, None]:
        if label_str in self.__mapping_cache_dict:
            return self.__mapping_cache_dict[label_str]

        find_label_str = label_str
        while find_label_str not in self.__selected_labels_set:
            find_label_str = self.__label_tree_dict[find_label_str]
            if find_label_str is None:
                break
        self.__mapping_cache_dict[label_str] = find_label_str
        return find_label_str


class ValuesDict(DictWithSelectLabels):

    LABEL_TREE = {
        'autonomy': 'life',
        'creativity': 'cognition',
        'emotion': 'cognition',
        'moral': 'cognition',
        'cognition': 'life',
        'future': 'cognition',
        'thinking': 'cognition',
        'security': 'order',
        'inner-peace': 'order',
        'order': 'life',
        'justice': 'life',
        'advice': 'life',
        'career': 'life',
        'achievement': 'life',
        'wealth': 'life',
        'health': 'life',
        'learning': 'life',
        'nature': 'life',
        'animals': 'life',
        'purpose': 'work-ethic',
        'responsible': 'work-ethic',
        'hard-work': 'work-ethic',
        'work-ethic': None,
        'perseverance': 'work-ethic',
        'feeling-good': None,
        'forgiving': 'accepting-others',
        'accepting-others': None,
        'helping-others': 'society',
        'gratitude': None,
        'dedication': None,
        'self-confidence': None,
        'optimisim': None,
        'honesty': 'truth',
        'truth': None,
        'spirituality': 'religion',
        'religion': None,
        'significant-other': 'relationships',
        'marriage': 'significant-other',
        'friends': 'relationships',
        'relationships': 'social',
        'family': 'relationships',
        'parents': 'family',
        'siblings': 'family',
        'social': None,
        'children': 'family',
        'society': 'social',
        'art': 'life',
        'respect': 'self-confidence',
        'life': None
    }

    def __init__(self, use_only_lst: Set[str] = None):
        self.__word_label_dict = None
        super().__init__(ValuesDict.LABEL_TREE, use_only_lst)

    def load(self):
        self.__word_label_dict = self.load_dict_and_fix(global_config.values.path_csv)

    def label_word(self, word_str: str) -> List[str]:
        if word_str in self.__word_label_dict:
            return [self.__word_label_dict[word_str]]
        else:
            return []
