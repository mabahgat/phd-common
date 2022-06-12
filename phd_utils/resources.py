from typing import Set
from nltk.corpus import stopwords
from phd_utils import global_config
from phd_utils.annotators.liwc import annotations_from_liwc_output
import pandas as pd


__cache_dict = {}


def ud(filter_names=True, filter_stopwords=True):
    from phd_utils.dataset_creators import LiwcDatasetCreator
    if 'ud' not in __cache_dict:
        dataset = LiwcDatasetCreator.from_raw(global_config.resources.ud.raw)
        if filter_names:
            dataset.filter_names()
        if filter_stopwords:
            dataset.filter_stopwords()
        __cache_dict['ud'] = dataset.get_raw()
    return __cache_dict['ud'].copy()


def list_stopwords() -> Set[str]:
    if 'list_stopwords' not in __cache_dict:
        __cache_dict['list_stopwords'] = set(stopwords.words('english'))
    return __cache_dict['list_stopwords'].copy()


def list_names() -> Set[str]:
    if 'list_names' not in __cache_dict:
        __cache_dict['list_names'] = set(pd.read_csv(global_config.resources.names, names=['name']).name.str.lower().to_list())
    return __cache_dict['list_names'].copy()


def liwc_ud() -> pd.DataFrame:
    if 'liwc_ud' not in __cache_dict:
        __cache_dict['liwc_ud'] = pd.read_csv(global_config.liwc_ud.path, index_col='word')
    return __cache_dict['liwc_ud'].copy()


def __liwc_22_annotation(path_str: str, cache_key_str: str) -> pd.DataFrame:
    if cache_key_str not in __cache_dict:
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
        __cache_dict[cache_key_str] =  annotations_from_liwc_output(path_str,
                                                                    exclude_columns_lst=exclude_columns_liwc22_lst,
                                                                    result_column_str='liwc22')
    return __cache_dict[cache_key_str].copy()


def liwc_ud_with_liwc22() -> pd.DataFrame:
    return __liwc_22_annotation(global_config.liwc_22.liwc_ud.path, 'liwc_ud_with_liwc22')


def twitter_top_10k_with_liwc22():
    return __liwc_22_annotation(global_config.liwc_22.twitter_top_10k.path, 'twitter_top_10k_with_liwc22')
