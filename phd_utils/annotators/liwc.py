from phd_utils import global_config
from liwc import Liwc
from typing import List, Dict
from random import Random


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