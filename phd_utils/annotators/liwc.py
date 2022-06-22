from phd_utils import global_config
from liwc import Liwc as LiwcLookup
from typing import List, Dict, Union
from random import Random
import re
import pandas as pd


#liwc_en = LiwcLookup(global_config.liwc.path)
liwc_en = LiwcLookup(global_config.values.path) # TODO take as parameter

def __create_strict_liwc_dict() -> Dict[str, List[str]]:
    def remove_wild_card(entry_str):
        if entry_str.endswith('*'):
            return entry_str[0:-1]
        return entry_str
    return {remove_wild_card(key): value for key, value in liwc_en.lexicon.items()}


__liwc_en_strict_dict = __create_strict_liwc_dict()

def get_liwc_en_strict_dict_copy() -> Dict[str, List[str]]:
    return {k: v.copy() for k, v in __liwc_en_strict_dict.items()}


def liwc_annotate_word(word_str: str) -> List[str]:
    return liwc_en.search(word_str)


def liwc_annotate_word_strict(word_str: str) -> List[str]:
    if word_str in __liwc_en_strict_dict:
        return __liwc_en_strict_dict[word_str].copy()
    else:
        return []


def get_liwc_category_examples(cat_str, count=4, rand_seed=0) -> List[str]:
    r = Random(rand_seed)
    words_set = set()
    for word_str, labels_lst in liwc_en.lexicon.items():
        if cat_str in labels_lst:
            words_set.add(word_str)
    return r.sample(words_set, k=count)


def annotations_from_liwc_output(liwc_out: Union[str,pd.DataFrame],
                                 exclude_columns_lst: List[str]=[],
                                 include_columns_lst: List[str]=[],
                                 result_column_str: str='liwc') -> pd.DataFrame:
    """
    Converts LIWC tool output format to word-list of categories
    """
    if isinstance(liwc_out, str):
        liwc_out_df = pd.read_csv(liwc_out, index_col='word')
    elif isinstance(liwc_out, pd.DataFrame):
        liwc_out_df = liwc_out.copy(deep=True)
    else:
        raise ValueError('Unregognize input for liwc_out parameter')

    if exclude_columns_lst and include_columns_lst:
        raise ValueError('Can not specify both include and exclude columns lists')

    if exclude_columns_lst:    
        liwc_out_df.drop(columns=exclude_columns_lst, inplace=True)
    elif include_columns_lst:
        liwc_out_df = liwc_out_df[include_columns_lst]
    i = 0
    def get_annotations(row_dict):
        nonlocal i
        i += 1
        annotations_lst = []
        for cat_str, value in row_dict.items():
            if isinstance(value, str):
                print(cat_str)
                print(value)
                print('line no {}'.format(i))
                print(list(row_dict.items()))
            if value > 0:
                annotations_lst.append(cat_str)
        return annotations_lst
    
    liwc_out_df[result_column_str] = liwc_out_df.apply(get_annotations, axis=1)
    liwc_out_df = liwc_out_df[[result_column_str]]
    return liwc_out_df


def annotations_from_liwc22_output(liwc_out: Union[str,pd.DataFrame]) -> pd.DataFrame:
    exclude_columns_liwc22_lst = [
        'Segment',
        'WC',
        'Analytic',
        'Clout',
        'Authentic',
        'Tone',
        'WPS',
        'BigWords',
        'Dic',
        'Linguistic',
        'function',
        'pronoun',
        'ppron',
        'i',
        'we',
        'you',
        'shehe',
        'they',
        'ipron',
        'det',
        'article',
        'number',
        'prep',
        'auxverb',
        'adverb',
        'conj',
        'negate',
        'verb',
        'adj',
        'focuspast',
        'focuspresent',
        'focusfuture',
        'AllPunc',
        'Period',
        'Comma',
        'QMark',
        'Exclam',
        'Apostro',
        'OtherP'
    ]
    liwc22_df = annotations_from_liwc_output(liwc_out,
                                             exclude_columns_lst=exclude_columns_liwc22_lst,
                                             result_column_str='liwc22')
    return liwc22_df


__liwc22_to_liwc15_dict = {
    'Drives': 'drives',
    'affiliation': 'affiliation',
    'achieve': 'achiev',
    'power': 'power',
    'cogproc': 'cogproc',
    'insight': 'insight',
    'cause': 'cause',
    'discrep': 'discrep',
    'tentat': 'tentat',
    'differ': 'differ',
    'Affect': 'affect',
    'swear': 'swear',
    'Social': 'social',
    'family': 'family',
    'friend': 'friend',
    'female': 'female',
    'male': 'male',
    'leisure': 'leisure',
    'home': 'home',
    'work': 'work',
    'money': 'money',
    'relig': 'relig',
    'health': 'health',
    'sexual': 'sexual',
    'death': 'death',
    'reward': 'reward',
    'risk': 'risk',
    'motion': 'motion',
    'space': 'space',
    'time': 'time',
    'netspeak': 'netspeak',
    'assent': 'assent',
    'nonflu': 'nonflu',
    'filler': 'filler',
    'tone_pos': 'posemo',
    'tone_neg': 'negemo',
    'emotion': 'affect',
    'emo_pos': 'posemo',
    'emo_neg': 'negemo',
    'emo_anx': 'anx',
    'emo_anger': 'anger',
    'emo_sad': 'sad',
    'Physical': 'bio',
    'food': 'ingest',
    'visual': 'see',
    'auditory': 'hear',
    'feeling': 'feel',
    'quantity': 'quant',
    'certitude': 'certain',
    'allnone': 'certain',
    'socrefs': 'social',
    'illness': 'health',
    'Perception': 'percept',
    'Lifestyle': 'pconcern',
    'Cognition': None,
    'memory': None,
    'socbehav': None,
    'prosocial': None,
    'polite': None,
    'conflict': None,
    'moral': None,
    'comm': None,
    'Culture': None,
    'politic': None,
    'ethnicity': None,
    'tech': None,
    'lifestyle': None,
    'wellness': None,
    'mental': None,
    'substances': None,
    'need': None,
    'want': None,
    'acquire': None,
    'lack': None,
    'fulfill': None,
    'fatigue': None,
    'curiosity': None,
    'allure': None,
    'attention': None,
    'Conversation': None
}


__liwc15_to_liwc22_dict = {v: k for k, v in __liwc22_to_liwc15_dict.items() if v is not None}


def map_classes_liwc22_to_liwc15(annotation: Union[pd.DataFrame, str], key_str: str=None) -> Union[pd.DataFrame, str]:
    def fix_parent_if_needed(cat_lst: List[str]) -> List[str]:
        if 'death' in cat_lst and 'bio' in cat_lst:
            if set(['health', 'sexual', 'substances', 'food']).intersection(cat_lst):
                cat_lst.insert(cat_lst.index('death'), 'pconcern')
            else:
                cat_lst[cat_lst.index('bio')] = 'pconcern'
        
        relativ_cat_set = list(set(['motion', 'space', 'time']).intersection(cat_lst))
        if relativ_cat_set:
            cat_lst.insert(cat_lst.index(relativ_cat_set[0]), 'relativ')
        
        return cat_lst
    
    if isinstance(annotation, str):
        cat_str = __liwc22_to_liwc15_dict[annotation]
        return fix_parent_if_needed([cat_str])[0]
    elif isinstance(annotation, pd.DataFrame):
        if not key_str:
            raise ValueError('Must specify key in the data frame')
        annotation[key_str] = annotation[key_str].apply(lambda c: __liwc22_to_liwc15_dict[c] if isinstance(c, str) else [__liwc22_to_liwc15_dict[c_str] for c_str in c])
        annotation[key_str] = annotation[key_str].apply(fix_parent_if_needed)
        annotation[key_str] = annotation[key_str].apply(lambda l: list(set(l)))
        return annotation
    else:
        raise ValueError('Unknown type passed')


class LiwcDict:

    def __init__(self, dict_path_str:str=None, liwc: LiwcLookup=None, categories_to_include_only: List[str]=None):
        """
        :param classes_to_include_only: list of classes to include only in the output and the rest is ignored
        """
        if dict_path_str:
            self.liwc = LiwcLookup(dict_path_str)
        elif liwc:
            self.liwc = liwc
        else:
            self.liwc = LiwcLookup(global_config.liwc.path) # default english liwc
        self.__categories_to_include_only = categories_to_include_only
        self.__strict_dict = self.__create_strict_liwc_dict()
    
    def __create_strict_liwc_dict(self) -> Dict[str, List[str]]:
        def remove_wild_card(entry_str):
            if entry_str.endswith('*'):
                return entry_str[0:-1]
            return entry_str
        return {remove_wild_card(key): value for key, value in self.liwc.lexicon.items()}
    
    def annotate_word(self, word_str: str) -> List[str]:
        cat_lst = self.liwc.search(word_str)
        if self.__categories_to_include_only:
            return self.__filter_list(cat_lst)
        else:
            return cat_lst

    def annotate_sentence(self, word_lst: List[str], include_all_cats=False) -> Dict[str, int]:
        """
        :param include_all_cats: results dictionary will contain all liwc categories including ones with 0 hits
        """
        result_dict = dict(self.liwc.parse(word_lst))
        if include_all_cats:
            all_dict = {cat_str: 0 for cat_str in self.liwc.categories.values()} # TODO should be done only once
            for cat_str, count_int in result_dict.items():
                all_dict[cat_str] = count_int
            result_dict = all_dict
        
        if self.__categories_to_include_only:
            return self.__filter_dict(result_dict)
        else:
            return result_dict
    
    def word_hit_count(self, word_lst: List[str]) -> int:
        return sum([1 for word_str in word_lst if len(self.liwc.search(word_str)) != 0])
    
    def word_hit_count_strict(self, word_lst: List[str]) -> int:
        return sum([1 for word_str in word_lst if word_str in self.__strict_dict])

    def __filter_dict(self, liwc_annotations: Dict[str, int]) -> Dict[str, int]:
        """
        Filters a LIWC parse result to include the selected cats only
        """
        return {k:v for k, v in liwc_annotations.items() if k in self.__categories_to_include_only}
    
    def __filter_list(self, liwc_annotations_lst: List[str]) -> List[str]:
        """
        Filters a LIWC search result to include selected categories only
        """
        return {c for c in liwc_annotations_lst if c in self.__categories_to_include_only}

    def filter_with_liwc(self, word_dict: Dict[str, int]) -> Dict[str, int]:
        """
        Returns only words that are in liwc (including wild card matches)
        """
        return {k:v for k, v in word_dict.items() if self.liwc.search(k)}
    
    def filter_with_liwc_strict(self, word_dict: Dict[str, int]) -> Dict[str, int]:
        """
        Returns only words that are in liwc (with no wild card matches)
        """
        return {k:v for k, v in word_dict.items() if k in self.__strict_dict}

    def annotate_word_strict(self, word_str: str) -> List[str]:
        if word_str in self.__strict_dict:
            return self.__strict_dict[word_str].copy()
        else:
            return []
    
    def annotate_sentence_strict(self, word_lst: List[str]) -> Dict[str, int]:
        cat_counts_dict = {}
        for word_str in word_lst:
            cat_lst = self.annotate_word_strict(word_str)
            for cat_str in cat_lst:
                if cat_str not in cat_counts_dict:
                    cat_counts_dict[word_str] = 0
                cat_counts_dict[cat_str] += 1
        return cat_counts_dict
    
    def cat_count(self) -> int:
        if not self.__categories_to_include_only:
            return len(self.liwc.categories)
        else:
            return len(self.__categories_to_include_only)


class LiwcDictModifier:
    """
    parses and modifies liwc dictionary consumed by liwc library
    """

    def __init__(self, src_path_str: str = None):
        if not src_path_str:
            self.__src_file_str = global_config.liwc.path
        else:
            self.__src_file_str = src_path_str
        self.__cat_mapping_dict, self.__word_cat_dict = self.__parse_file()

    def __parse_file(self) -> (Dict[str, int], Dict[str, List[int]]):
        with open(file=self.__src_file_str, mode='r', encoding='utf8') as liwc_file:
            lines_lst = [l.strip() for l in liwc_file.readlines()]
        dict_end_index = LiwcDictModifier.__get_cat_mapping_end_index(lines_lst)

        cat_lst = lines_lst[1:dict_end_index]
        word_cat_lst = lines_lst[dict_end_index + 1:]

        cat_dict = LiwcDictModifier.__parse_cat_dict(cat_lst)
        word_cat_dict = LiwcDictModifier.__parse_word_lst(word_cat_lst)

        if 'pconcern' not in cat_dict:
            LiwcDictModifier.__fix_for_pconcern(cat_dict, word_cat_dict)

        return cat_dict, word_cat_dict

    @staticmethod
    def __get_cat_mapping_end_index(lines_lst: List[str]) -> int:
        index = 1 # skip first '%' sign
        while lines_lst[index] != '%':
            index += 1
        return index
    
    @staticmethod
    def __parse_cat_dict(cat_lst: List[str]) -> Dict[str, int]:
        cat_dict = {}
        for entry_str in cat_lst:
            index, cat_str = re.split(r'\s+', entry_str)
            cat_dict[cat_str] = int(index)
        return cat_dict
    
    @staticmethod
    def __parse_word_lst(word_lst: List[str]) -> Dict[str, List[int]]:
        word_map_dict = {}
        for entry_str in word_lst:
            parts = re.split('\t+', entry_str)
            word_str = parts[0]
            cats_lst = [int(i) for i in parts[1:]]
            word_map_dict[word_str] = cats_lst
        return word_map_dict
    
    @staticmethod
    def __fix_for_pconcern(cat_dict: Dict[str, int], word_cat_dict: Dict[str, List[int]]) -> None:
        """
        Add pconcern category
        """
        # TODO put pconcern before its children
        pconcern_idx = max(cat_dict.values()) +  1
        cat_dict['pconcern'] = pconcern_idx

        pconcern_children_lst = ['work', 'leisure', 'home', 'money', 'relig', 'death']
        pconcern_children_index_set = set([cat_dict[cat_str] for cat_str in pconcern_children_lst])

        for cat_lst in word_cat_dict.values():
            found_lst = list(set(cat_lst).intersection(pconcern_children_index_set))
            if found_lst:
                #insert_index = cat_lst.index(found_lst[0])
                #cat_lst.insert(insert_index, pconcern_idx)
                cat_lst.append(pconcern_idx) # use append to be consistent with add
    
    def save(self, out_path_str: str) -> None:
        sorted_cat_keys = sorted(self.__cat_mapping_dict.items(), key=lambda kv: kv[1])
        sorted_word_lst = sorted(self.__word_cat_dict.items(), key=lambda kv: kv[0])
        with open(file=out_path_str, mode='w', encoding='utf8') as out_file:
            print('%', file=out_file)
            [print('{}\t{}'.format(index, cat_str), file=out_file) for cat_str, index in sorted_cat_keys]
            print('%', file=out_file)
            for entry in sorted_word_lst:
                cat_lst_str = '\t'.join([str(i) for i in entry[1]])
                print('{}\t{}'.format(entry[0], cat_lst_str), file=out_file)
    
    def add(self, new_words_lst: Dict[str, str]) -> None:
        for word_str, cat_str in new_words_lst.items():
            word_str = str(word_str) # pandas auto parses 'nan' string into float
            word_str = re.sub(pattern=r'\s+', repl=' ', string=word_str)
            if '$' in word_str: # '$' is a special char in LiwcLookup lib that marks end of word
                raise ValueError('Can not accept words with $: {}'.format(word_str))
            if '*' in word_str and not word_str.endswith('*'):
                raise ValueError('The character * has a special meaning and only accepted at the end of the word: {}'.format(word_str))
            cat_index = self.__cat_mapping_dict[cat_str]
            if word_str in self.__word_cat_dict:
                word_cats_lst = self.__word_cat_dict[word_str]
                word_cats_lst.append(cat_index)
                self.__word_cat_dict[word_str] = sorted(word_cats_lst)
            else:
                self.__word_cat_dict[word_str] = [cat_index]
    
    def word_count(self) -> int:
        return len(self.__word_cat_dict)
    
    def cat_count(self) -> int:
        return len(self.__cat_mapping_dict)
