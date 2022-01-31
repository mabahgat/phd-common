from phd_utils import global_config
from liwc import Liwc
from typing import List, Dict
from random import Random
import re


liwc_en = Liwc(global_config.liwc.path)

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


class LiwcDict:
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
        dict_end_index = LiwcDict.__get_cat_mapping_end_index(lines_lst)

        cat_lst = lines_lst[1:dict_end_index]
        word_cat_lst = lines_lst[dict_end_index + 1:]

        cat_dict = LiwcDict.__parse_cat_dict(cat_lst)
        word_cat_dict = LiwcDict.__parse_word_lst(word_cat_lst)

        if 'pconcern' not in cat_dict:
            LiwcDict.__fix_for_pconcern(cat_dict, word_cat_dict)

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
