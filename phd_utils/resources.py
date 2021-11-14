from typing import Set
from nltk.corpus import stopwords
from phd_utils import global_config
import pandas as pd


__cache_dict = {}


def list_stopwords() -> Set[str]:
    if 'stopwords' not in __cache_dict:
        __cache_dict['stopwords'] = set(stopwords.words('english'))
    return __cache_dict['stopwords'].copy()


def list_names() -> Set[str]:
    if 'names' not in __cache_dict:
        __cache_dict['names'] = set(pd.read_csv(global_config.resources.names, names=['name']).name.str.lower().to_list())
    return __cache_dict['names'].copy()
