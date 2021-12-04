from typing import List, Set, Dict, Union
import pandas as pd
import urllib.parse
from phd_utils.annotators.liwc import liwc_annotate_word, liwc_annotate_word_strict, get_liwc_en_strict_dict_copy, liwc_en
from random import Random
from phd_utils.resources import list_names, list_stopwords
from pathlib import Path
import os


class LiwcCategories:

    ORG_HIERARCHY_dict = {
        'function': None,
        'pronoun': 'function',
        'ppron': 'pronoun',
        'i': 'ppron',
        'we': 'ppron',
        'you': 'ppron',
        'shehe': 'ppron',
        'they': 'ppron',
        'ipron': 'pronoun',
        'article': 'function',
        'prep': 'function',
        'auxverb': 'function',
        'adverb': 'function',
        'conj': 'function',
        'negate': 'function',
        'verb': 'function',
        'adj': 'function',
        'compare': 'function',
        'interrog': 'function',
        'number': 'function',
        'quant': 'function',
        'affect': None,
        'posemo': 'affect',
        'negemo': 'affect',
        'anx': 'negemo',
        'anger': 'negemo',
        'sad': 'negemo',
        'social': None,
        'family': 'social',
        'friend': 'social',
        'female': 'social',
        'male': 'social',
        'cogproc': None,
        'insight': 'cogproc',
        'cause': 'cogproc',
        'discrep': 'cogproc',
        'tentat': 'cogproc',
        'certain': 'cogproc',
        'differ': 'cogproc',
        'percept': None,
        'see': 'percept',
        'hear': 'percept',
        'feel': 'percept',
        'bio': None,
        'body': 'bio',
        'health': 'bio',
        'sexual': 'bio',
        'ingest': 'bio',
        'drives': None,
        'affiliation': 'drives',
        'achiev': 'drives',
        'power': 'drives',
        'reward': 'drives',
        'risk': 'drives',
        'timeorient': None,
        'focuspast': 'timeorient',
        'focuspresent': 'timeorient',
        'focusfuture': 'timeorient',
        'relativ': None,
        'motion': 'relativ',
        'space': 'relativ',
        'time': 'relativ',
        'pconcern': None,
        'work': 'pconcern',
        'leisure': 'pconcern',
        'home': 'pconcern',
        'money': 'pconcern',
        'relig': 'pconcern',
        'death': 'pconcern',
        'informal': None,
        'swear': 'informal',
        'netspeak': 'informal',
        'assent': 'informal',
        'nonflu': 'informal',
        'filler': 'informal'
    }

    def __init__(self, include_set: Set[str]=None):
        """
        Top level categories to include
        """
        not_found_lst = [cat_str for cat_str in include_set if cat_str not in LiwcCategories.ORG_HIERARCHY_dict]
        if not_found_lst:
            raise ValueError('Categories not found in liwc: {}'.format(not_found_lst))
        self.__include_set = include_set

    def map_list(self, cat_lst: List[str], empty_value=None) -> List[str]:
        """
        Maps the list of categories to the ones used by this instance. A child
        category is mapped to an included parent category.
        :return: categories that matched included ones in this instance maintaining
            their order.
        """
        found_lst = []
        for cat_str in cat_lst:
            while cat_str is not None:
                if cat_str in self.__include_set:
                    if cat_str not in found_lst:
                        found_lst.append(cat_str)
                    break
                else:
                    cat_str = LiwcCategories.ORG_HIERARCHY_dict[cat_str]
        return found_lst
    
    @staticmethod
    def get_head_classes() -> Set[str]:
        return [k for k, v in LiwcCategories.ORG_HIERARCHY_dict.items() if v is None]
    
    @staticmethod
    def add_parent_class_if_missing(cat_lst: List[str]) -> List[str]:
        """
        :return: List of categories maintaining order
        """
        cat_set = set(cat_lst)
        for idx, cat_str in reversed(list(enumerate(cat_lst))):
            parent_cat_str = LiwcCategories.ORG_HIERARCHY_dict[cat_str]
            while parent_cat_str is not None and parent_cat_str not in cat_set:
                cat_lst.insert(idx, parent_cat_str)
                parent_cat_str = LiwcCategories.ORG_HIERARCHY_dict[parent_cat_str]
        return cat_lst
    
    @staticmethod
    def get_children(cat_str) -> List[str]:
        return [k for k, v in LiwcCategories.ORG_HIERARCHY_dict.items() if v == cat_str]
    
    @staticmethod
    def get_parent(cat_str) -> str:
        if cat_str not in LiwcCategories.ORG_HIERARCHY_dict:
            return None
        else:
            return LiwcCategories.ORG_HIERARCHY_dict[cat_str]
    
    @staticmethod
    def keep_lowest_cats_only(cat_lst: List[str]) -> List[str]:
        """
        For every category in the list remove it if there's a child category
        """
        to_remove_idx_set = set()
        cat_set = set(cat_lst)
        for cat_str in cat_lst:
            parent_cat_str = LiwcCategories.ORG_HIERARCHY_dict[cat_str]
            while parent_cat_str is not None:
                if parent_cat_str in cat_set:
                    idx = cat_lst.index(parent_cat_str) # expected that parent will always exist
                    to_remove_idx_set.add(idx)
                parent_cat_str = LiwcCategories.ORG_HIERARCHY_dict[parent_cat_str]
        for idx in sorted(to_remove_idx_set, reverse=True):
            del cat_lst[idx]
        return cat_lst


class LiwcDatasetCreator:

    @staticmethod
    def from_raw(input_file_str: Union[str], sep_str: str = r'\|'):
        """
        Read from '.dat' file. expected to contain columns:
        word, meaning, example, numLikes, numDislikes, tags
        """
        raw_df = pd.read_csv(input_file_str, sep=sep_str, engine='python')
        LiwcDatasetCreator.__fix_text_column_inplace(raw_df, 'word', do_lower=True)
        LiwcDatasetCreator.__fix_text_column_inplace(raw_df, 'meaning')
        LiwcDatasetCreator.__fix_text_column_inplace(raw_df, 'example')
        raw_df = LiwcDatasetCreator.__remove_bad_entries(raw_df)
        raw_df['diffLikes'] = pd.to_numeric(raw_df.numLikes) - pd.to_numeric(raw_df.numDislikes)
        return LiwcDatasetCreator(raw_df, None)

    @staticmethod
    def from_existing(train_file_str: str, test_file_str: str, index_col=0):
        """
        Read data frame files for training and testing
        """
        train_df = pd.read_csv(train_file_str, index_col=index_col)
        test_df = pd.read_csv(test_file_str, index_col=index_col)
        return LiwcDatasetCreator(train_df=train_df, test_df=test_df)

    @staticmethod
    def __decode_text(text_str: str) -> str:
        return urllib.parse.unquote_plus(text_str)
    
    @staticmethod
    def __fix_text_column_inplace(content_df: pd.DataFrame, column_name:str, do_lower:bool=False) -> None:
        content_df[column_name].fillna('', inplace=True)
        content_df[column_name] = content_df[column_name].apply(LiwcDatasetCreator.__decode_text)
        if do_lower:
            content_df[column_name] = content_df[column_name].str.lower()

    @staticmethod
    def __remove_bad_entries(content_df: pd.DataFrame) -> pd.DataFrame:
        content_df = content_df[content_df['numLikes'].str.fullmatch(r'\d+')]
        content_df = content_df[content_df['numDislikes'].str.fullmatch(r'\d+')]
        return content_df

    def __init__(self, raw_df: pd.DataFrame=None, train_df: pd.DataFrame=None, test_df: pd.DataFrame=None):
        """
        Either raw_df or train_df can be sepcified. if train_df is specified,
        test_df has to be specified too
        :param raw_df: dataframe is expected to be unannotated
        :param train_df: dataframe is expected to be annotated
        :param test_df: dataframe is exepcted to be annotated
            cards, 'wild_card' regular matching liwc matching
        """
        if train_df is not None and raw_df is not None:
            raise ValueError('Either raw_df or train_df can be sepecified')
        if train_df is not None and 'liwc' not in train_df.columns:
            raise ValueError('Unannotated train_df. Columns are {}'.format(train_df.columns))
        if test_df is not None and 'liwc' not in test_df.columns:
            raise ValueError('Unannotated test_df. Columns are {}'.format(test_df.columns))
        self.__raw_df = raw_df
        self.__train_df = train_df
        self.__test_df = test_df
    
    def get_raw(self) -> pd.DataFrame:
        return LiwcDatasetCreator.__get_copy_or_none(self.__raw_df)
    
    def get_train(self) -> pd.DataFrame:
        return LiwcDatasetCreator.__get_copy_or_none(self.__train_df)
    
    def get_test(self) -> pd.DataFrame:
        return LiwcDatasetCreator.__get_copy_or_none(self.__test_df)
    
    def get_raw_annotated(self) -> pd.DataFrame:
        return LiwcDatasetCreator.__get_copy_or_none(self.__get_annotated())
    
    def get_raw_not_annotated(self) -> pd.DataFrame:
        return LiwcDatasetCreator.__get_copy_or_none(self.__get_not_annotated())
    
    @staticmethod
    def __get_copy_or_none(content_df):
        if content_df is not None:
            copy_df = content_df.copy(deep=True)
            if 'liwc' in copy_df.columns:
                copy_df['liwc'] = copy_df.liwc.apply(lambda l: l.copy()) # 'deep=True' not enough
            return copy_df
        else:
            return None
    
    def filter(self, exclude_set: Set[str], apply_to: List[str]=['raw']) -> None:
        while apply_to:
            set_str = apply_to.pop()
            if set_str == 'raw':
                self.__raw_df = self.__raw_df[~self.__raw_df.word.isin(exclude_set)]
            elif set_str == 'train':
                self.__train_df = self.__train_df[~self.__train_df.word.isin(exclude_set)]
            elif set_str == 'test':
                self.__test_df = self.__test_df[~self.__test_df.word.isin(exclude_set)]
    
    def filter_names(self, apply_to: List[str]=['raw']) -> None:
        self.filter(list_names(), apply_to=apply_to)
    
    def filter_stopwords(self, apply_to: List[str]=['raw']) -> None:
        self.filter(list_stopwords(), apply_to=apply_to)

    def select_for_train(self, topN: int=None, minDiff: int=None, ignore_testset_b=False) -> int:
        """
        if both specified then topN is applied first and then minDiff - if none, then return all
        :param ignore_testset_b: Do not do test set analysis to avoid contamination.
            Use only if not planning to create a testset out of the current data.
        """
        if self.__raw_df is None:
            raise Exception('Instances initialized with no raw data')

        annotated_df = self.get_raw_annotated()
        if not ignore_testset_b:
            annotated_df = self.__filter_out_test_patterns(annotated_df)

        if topN:
            annotated_df = annotated_df.sort_values(by='diffLikes').groupby('word').head(topN)
        if minDiff:
            annotated_df = annotated_df[annotated_df.diffLikes >= minDiff]
        
        annotated_df['liwc'] = annotated_df.liwc.apply(LiwcCategories.keep_lowest_cats_only)
        
        self.__train_df = annotated_df
        return len(self.__train_df)

    def __filter_out_test_patterns(self, content_df: pd.DataFrame) -> pd.DataFrame:
        if self.__test_df is None:
            raise Exception('Test set not initialized')
        exact_matches_set = set(self.__test_df.word.to_list())
        prefixes_set = set([k.replace('*', '') for k in liwc_en.lexicon.keys() if str(k).replace('*', '') in exact_matches_set])
        
        def matches_a_test_entry(word_str: str):
            if word_str in exact_matches_set:
                return True
            for prefix_str in prefixes_set:
                if word_str.startswith(prefix_str):
                    return True
            return False

        return content_df[~content_df.word.apply(matches_a_test_entry)]
    
    def select_for_test(self, count: int=None, min_class_count: int=None, distribution_method_str: str='ud_word', min_diff_int: int=None, samples_per_word_int: int=1) -> int:
        """
        :param count: count has to be at least equal to the number of categories
        :returns: Number of samples actually selected. To retreive the actuall test set call get_test_set
        """
        if not count and not min_class_count:
            raise ValueError('Either count for min_class_count needs to be specified')

        distribution_dict = {}
        if distribution_method_str == 'ud_word':
            distribution_dict = self.__compute_distribution_ud_word_based()
        elif distribution_method_str == 'ud_entry':
            distribution_dict = self.__compute_distribution_entry_based()
        elif distribution_method_str == 'liwc':
            distribution_dict = self.__compute_distribution_liwc_based()
        else:
            raise ValueError('Unknown distribution computation method "{}"'.format(distribution_method_str))
        
        counts_dict = {}
        if count:
            counts_dict = {liwc_str: int(round(prcnt_float * count)) for liwc_str, prcnt_float in distribution_dict.items()}
        elif min_class_count:
            factor_float = min_class_count / min(list(distribution_dict.values()))
            counts_dict = {liwc_str: int(round(prcnt_float * factor_float)) for liwc_str, prcnt_float in distribution_dict.items()}
        else:
            raise ValueError('Neither count nor minimum instnace count specified')

        annotated_df = self.get_raw_annotated()
        annotated_df['liwc'] = annotated_df.liwc.apply(LiwcCategories.keep_lowest_cats_only) # match distribution
        if min_diff_int is not None:
            annotated_df = annotated_df[annotated_df.diffLikes > min_diff_int]
        best_entry_per_word_df = annotated_df.sort_values(by='diffLikes', ascending=False).groupby('word', as_index=False).head(n=1).sample(frac=1, random_state=0)
        
        selected_df_lst = []
        for cat_str, count_needed_int in counts_dict.items():
            selected_df = best_entry_per_word_df[best_entry_per_word_df.liwc.apply(lambda liwc_lst: cat_str in liwc_lst)].head(count_needed_int)
            selected_df_lst.append(selected_df)
            best_entry_per_word_df.drop(selected_df.index, inplace=True)
        
        self.__test_df = pd.concat(selected_df_lst)
        return len(self.__test_df)

    
    def __compute_distribution_ud_word_based(self) -> Dict[str, float]:
        """
        Compute category percentages based on LIWC identified unique word occurances in UD
        """
        per_word_df = self.get_raw_annotated().groupby('word').first()
        per_word_df['liwc'] = per_word_df.liwc.apply(LiwcCategories.keep_lowest_cats_only)
        return LiwcDatasetCreator.__compute_distribution(per_word_df.liwc)
    
    def __compute_distribution_liwc_based(self) -> Dict[str, float]:
        """
        Compute category percentages based on LIWC lexicon number of entries per category
        """
        selected_cats_set = set()
        self.__get_annotated().liwc.apply(selected_cats_set.update)
        selected_cats_obj = LiwcCategories(selected_cats_set)
        liwc_en_strict_df = LiwcDatasetCreator.__dict_to_df(get_liwc_en_strict_dict_copy(), 'word', 'liwc')
        liwc_en_strict_df['liwc'] = liwc_en_strict_df.liwc.apply(selected_cats_obj.map_list)
        liwc_en_strict_df['liwc'] = liwc_en_strict_df.liwc.apply(LiwcCategories.add_parent_class_if_missing)
        liwc_en_strict_df['liwc'] = liwc_en_strict_df.liwc.apply(LiwcCategories.keep_lowest_cats_only)
        return LiwcDatasetCreator.__compute_distribution(liwc_en_strict_df.liwc)
    
    def __compute_distribution_entry_based(self) -> Dict[str, float]:
        """
        Computes category percentage based on number of definitions (UD entry)
        for each LIWC category
        """
        per_entry_df = self.get_raw_annotated()
        per_entry_df['liwc'] = per_entry_df.liwc.apply(LiwcCategories.keep_lowest_cats_only)
        return LiwcDatasetCreator.__compute_distribution(per_entry_df.liwc)
    
    @staticmethod
    def __compute_distribution(liwc_sr: pd.Series) -> Dict[str, float]:
        counts_dict = LiwcDatasetCreator.__compute_counts(liwc_sr)
        total_int = sum([c for c in counts_dict.values()])
        return {cat_str: count_int / total_int for cat_str, count_int in counts_dict.items()}

    @staticmethod
    def __compute_counts(liwc_sr: pd.Series) -> Dict[str, int]:
        cat_count_dict = {}

        def update_counts(cat_lst: List[str]) -> None:
            for cat_str in cat_lst:
                if cat_str not in cat_count_dict:
                    cat_count_dict[cat_str] = 0
                cat_count_dict[cat_str] += 1

        liwc_sr.apply(update_counts)
        return cat_count_dict
    
    @staticmethod
    def __dict_to_df(content_dict: Dict, key_col_name: str, value_col_name: str) -> pd.DataFrame:
        items = content_dict.items()
        df_as_dict = {
            key_col_name: [k for k, _ in items],
            value_col_name: [v for _, v in items]
        }
        return pd.DataFrame(data=df_as_dict)
    
    def __get_annotated(self):
        if self.__raw_df is None or 'liwc' not in self.__raw_df:
            raise Exception('Instances initialized with no raw data or not annotated')
        return LiwcDatasetCreator.__filter_annotated(self.__raw_df)
    
    @staticmethod
    def __filter_annotated(content_df: pd.DataFrame) -> pd.DataFrame:
        return content_df[content_df.liwc.apply(lambda cat_lst: len(cat_lst) >= 1)]
    
    def __get_not_annotated(self):
        if self.__raw_df is None or 'liwc' not in self.__raw_df:
            raise Exception('Instances initialized with no raw data or not annotated')
        return LiwcDatasetCreator.__filter_not_annotated(self.__raw_df)
    
    @staticmethod
    def __filter_not_annotated(content_df: pd.DataFrame) -> pd.DataFrame:
        return content_df[content_df.liwc.apply(lambda cat_lst: len(cat_lst) == 0)]

    @staticmethod
    def __select_topN(content_df:pd.DataFrame, topN: int) -> pd.DataFrame:
        """
        :param minDiff: inclusive minimum difference
        """
        if topN < 1:
            raise ValueError('Invalid Top N value "{}" can not be less than 1'.format(topN))
        return content_df.sort_values('diffLikes', ascending=False).groupby('word', as_index=False).head(topN)
    
    @staticmethod
    def __select_minDiff(content_df: pd.DataFrame, minDiff: int) -> pd.DataFrame:
        """
        :param minDiff: inclusive minimum difference
        """
        return content_df[content_df['diffLikes'] >= minDiff]
    
    def annotate(self, annotation_type_str:str, with_categories: LiwcCategories=None, overwrite_b: bool=False) -> None:
        """
        annotate for initialzed sets
        :param annotation_type_str: either 'strict' or 'wild_card'
        :param with_catagories: information about which liwc categories are selected
        :param overwrite_b: forces annotation even if already annotated - else skipped for aleady annotated
        """
        self.__raw_df = LiwcDatasetCreator.__try_annotate_set_in_place(self.__raw_df, annotation_type_str=annotation_type_str,
            with_categories=with_categories, overwrite_b=overwrite_b)
        self.__train_df = LiwcDatasetCreator.__try_annotate_set_in_place(self.__train_df, annotation_type_str=annotation_type_str,
            with_categories=with_categories, overwrite_b=overwrite_b)
        self.__test_df = LiwcDatasetCreator.__try_annotate_set_in_place(self.__test_df, annotation_type_str=annotation_type_str,
            with_categories=with_categories, overwrite_b=overwrite_b)
    
    @staticmethod
    def __try_annotate_set_in_place(content_df: pd.DataFrame, annotation_type_str:str,
        with_categories: LiwcCategories=None, overwrite_b: bool=False) -> pd.DataFrame:
        if content_df is not None:
            if overwrite_b or 'liwc' not in content_df.columns:
                LiwcDatasetCreator.__annotate_in_place(content_df, annotation_type_str)
                if with_categories is not None:
                    content_df['liwc'] = content_df.liwc.apply(with_categories.map_list)
            return content_df
        else:
            return None
    
    def redo_categories(self, with_categories: LiwcCategories) -> None:
        """
        adjust categories to a fixed list
        :param with_catagories: information about which liwc categories are selected
        """
        if self.__raw_df is not None:
            self.__raw_df['liwc'] = self.__raw_df.liwc.apply(with_categories.map_list)
        if self.__train_df is not None:
            self.__train_df['liwc'] = self.__train_df.liwc.apply(with_categories.map_list)
            self.__train_df = LiwcDatasetCreator.__filter_annotated(self.__train_df)
        if self.__test_df is not None:
            self.__test_df['liwc'] = self.__test_df.liwc.apply(with_categories.map_list)
            self.__test_df = LiwcDatasetCreator.__filter_annotated(self.__test_df)
    
    def selected_labels(self, label_select_method_str: str, rand_seed_int: int=0, apply_to=['train']):
        """
        :param label_select_method_str: methods are 'first' for select first, 'random_single' for pick one randomly
        :param apply_to: which set to apply to; 'train' and/or 'test' - default 'train' only
        """
        do_train_b = False
        do_test_b = False

        while apply_to:
            set_str = apply_to.pop()
            if set_str == 'train':
                do_train_b = True
            elif set_str == 'test':
                do_test_b = True
            else:
                raise ValueError('Unexpected set "{}" passed in apply_to'.format(set_str))
            
        if do_train_b:
            LiwcDatasetCreator.__select_from_annotation_in_place(self.__train_df, label_select_method_str, rand_seed_int=rand_seed_int)
        if do_test_b:
            LiwcDatasetCreator.__select_from_annotation_in_place(self.__train_df, label_select_method_str, rand_seed_int=rand_seed_int)
    
    @staticmethod
    def __annotate_in_place(content_df: pd.DataFrame, annotation_type_str: str) -> None:
        if annotation_type_str == 'wild_card':
            content_df['liwc'] = content_df.word.apply(lambda word_str: liwc_annotate_word(str(word_str)))
        elif annotation_type_str == 'strict':
            content_df['liwc'] = content_df.word.apply(lambda word_str: liwc_annotate_word_strict(str(word_str)))
        else:
            raise ValueError('Unknown annotation type "{}"'.format(annotation_type_str))
        # tool used to do LIWC annotation skips some parent categories and only add child categories - add them
        content_df['liwc'] = content_df.liwc.apply(LiwcCategories.add_parent_class_if_missing)

    
    @staticmethod
    def __select_from_annotation_in_place(content_df: pd.DataFrame, select_method_str: str, rand_seed_int: int=0) -> pd.DataFrame:
        """
        :param label_select_method_str: methods are 'first' for select first, 'random_single' for pick one randomly
        """
        rand = Random()
        rand.seed(rand_seed_int)

        if select_method_str == 'first':
            content_df['liwc'] = content_df.liwc.apply(lambda liwc_lst: liwc_lst[0])
        elif select_method_str == 'random_single':
            content_df['liwc'] = content_df.liwc.apply(rand.choice)
        else:
            raise ValueError('Unexpected selection method "{}"'.format(select_method_str))
    
    def save_train(self, path_str: str, overwrite_b=False, ignore_b=False) -> None:
        """
        Saves training set. Throws by default if file exists
        :param overwrite: if exists overwrite the file
        :param ignore: if exists do not overwrite and do not throw
        """
        do_write = LiwcDatasetCreator.doWriteOrThrow(path_str, overwrite_b, ignore_b)
        if do_write:
            self.__save(self.get_train(), path_str)
    
    def save_test(self, path_str: str, overwrite_b=False, ignore_b=False) -> None:
        """
        Saves test set. Throws by default if file exists
        :param overwrite: if exists overwrite the file
        :param ignore: if exists do not overwrite and do not throw
        """
        do_write = LiwcDatasetCreator.doWriteOrThrow(path_str, overwrite_b, ignore_b)
        if do_write:
            self.__save(self.get_test(), path_str)

    @staticmethod
    def doWriteOrThrow(path_str: str, overwrite_b, ignore_b):
        if overwrite_b or ignore_b or not Path(path_str).exists():
            return True
        else:
            if not overwrite_b:
                raise FileExistsError('File "{}" already exists'.format(path_str))
        return False
    
    def __save(self, content_df: pd.DataFrame, path_str: str) -> None:
        content_df['liwc'] = content_df.liwc.apply(lambda cats_lst: '|'.join(cats_lst))
        content_df.to_csv(path_str)


def __liwc_cats_for_all() -> LiwcCategories:
    all_cats_set = set(['affect', 'posemo', 'negemo', 'anx', 'anger', 'sad', 'social', 'family', 'friend', 'female', 'male', 'cogproc', 'insight', 'cause', 'discrep', 'tentat', 'certain', 'differ', 'percept', 'see', 'hear', 'feel', 'bio', 'body', 'health', 'sexual', 'ingest', 'drives', 'affiliation', 'achiev', 'power', 'reward', 'risk', 'relativ', 'motion', 'space', 'time', 'pconcern', 'work', 'leisure', 'home', 'money', 'relig', 'death', 'informal', 'swear', 'netspeak', 'assent', 'nonflu', 'filler'])
    return LiwcCategories(all_cats_set)


def __liwc_cats_for_roots() -> LiwcCategories:
    root_cats_set = set(['affect', 'social', 'cogproc', 'percept', 'bio', 'drives', 'relativ', 'pconcern', 'informal'])
    return LiwcCategories(root_cats_set)


def __liwc_cats_for_affect_3() -> LiwcCategories:
    social_cats_set = set(['affect', 'posemo', 'negemo'])
    return LiwcCategories(social_cats_set)

def __liwc_cats_for_affect_2() -> LiwcCategories:
    social_cats_set = set(['posemo', 'negemo'])
    return LiwcCategories(social_cats_set)


def __liwc_cats_for_social_5() -> LiwcCategories:
    social_cats_set = set(['social', 'family', 'friend', 'female', 'male'])
    return LiwcCategories(social_cats_set)

def __liwc_cats_for_social_3() -> LiwcCategories:
    social_cats_set = set(['social', 'family', 'friend'])
    return LiwcCategories(social_cats_set)


def __liwc_cats_for_bio() -> LiwcCategories:
    bio_cats_set = set(['bio', 'body', 'health', 'sexual', 'ingest'])
    return LiwcCategories(bio_cats_set)


def __liwc_cats_for_pconcer() -> LiwcCategories:
    pconcerns_cats_set = set(['pconcern', 'work', 'leisure', 'home', 'money', 'relig', 'death'])
    return LiwcCategories(pconcerns_cats_set)


def __liwc_create_train_sets(train_df: pd.DataFrame, cats_obj: LiwcCategories, root_path: Path, prefix_str: str) -> None:
    """
    create and saves files for top1, top10, minDiff1 and minDiff10
    """
    dataset_obj = LiwcDatasetCreator(raw_df=train_df)
    dataset_obj.redo_categories(cats_obj)

    dataset_obj.select_for_train(topN=10, ignore_testset_b=True)
    save_path = root_path / '{}_train-top10.csv'.format(prefix_str)
    dataset_obj.save_train(save_path, overwrite_b=True)
    print('Done {}'.format(save_path))

    dataset_obj.select_for_train(topN=1, ignore_testset_b=True)
    save_path = root_path / '{}_train-top1.csv'.format(prefix_str)
    dataset_obj.save_train(save_path, overwrite_b=True)
    print('Done {}'.format(save_path))

    dataset_obj.select_for_train(minDiff=10, ignore_testset_b=True)
    save_path = root_path / '{}_train-minDiff10.csv'.format(prefix_str)
    dataset_obj.save_train(save_path, overwrite_b=True)
    print('Done {}'.format(save_path))

    dataset_obj.select_for_train(minDiff=1, ignore_testset_b=True)
    save_path = root_path / '{}_train-minDiff1.csv'.format(prefix_str)
    dataset_obj.save_train(save_path, overwrite_b=True)
    print('Done {}'.format(save_path))


def __liwc_create_test_sets(test_df: pd.DataFrame, cats_obj: LiwcCategories, root_path: Path, prefix_str: str) -> None:
    dataset_obj = LiwcDatasetCreator(test_df=test_df)
    dataset_obj.redo_categories(cats_obj)
    save_path = root_path / '{}_test-top1-1000.csv'.format(prefix_str)
    dataset_obj.save_test(save_path, overwrite_b=True)
    print('Done {}'.format(save_path))


def create_liwc_datasets():
    from phd_utils import global_config
    
    root_path = Path(global_config.liwc.create_run.root)

    print('Reading dataset from {}'.format(global_config.resources.ud.raw))
    dataset = LiwcDatasetCreator.from_raw(global_config.resources.ud.raw)
    dataset.filter_names()
    dataset.filter_stopwords()
    dataset.annotate(annotation_type_str='strict')
    dataset.redo_categories(__liwc_cats_for_all())
    dataset.select_for_test(count=1000, samples_per_word_int=1)
    dataset.save_test(root_path / 'liwc_all_test-top1-1000.csv', overwrite_b=True)
    dataset.select_for_train() # select everything other than the testset

    print('Creating root')
    __liwc_create_train_sets(train_df=dataset.get_train(), cats_obj=__liwc_cats_for_roots(), root_path=root_path, prefix_str='liwc_root-9')
    __liwc_create_test_sets(test_df=dataset.get_test(), cats_obj=__liwc_cats_for_roots(), root_path=root_path, prefix_str='liwc_root-9')

    print('Creating affect - 3 class: {}'.format(__liwc_cats_for_affect_3()))
    __liwc_create_train_sets(train_df=dataset.get_train(), cats_obj=__liwc_cats_for_affect_3(), root_path=root_path, prefix_str='liwc_affect-3')
    __liwc_create_test_sets(test_df=dataset.get_test(), cats_obj=__liwc_cats_for_affect_3(), root_path=root_path, prefix_str='liwc_affect-3')

    print('Creating affect - 2 class: {}'.format(__liwc_cats_for_affect_2()))
    __liwc_create_train_sets(train_df=dataset.get_train(), cats_obj=__liwc_cats_for_affect_2(), root_path=root_path, prefix_str='liwc_affect-2')
    __liwc_create_test_sets(test_df=dataset.get_test(), cats_obj=__liwc_cats_for_affect_2(), root_path=root_path, prefix_str='liwc_affect-2')

    print('Creating social -  5 class: {}'.format(__liwc_cats_for_social_5()))
    __liwc_create_train_sets(train_df=dataset.get_train(), cats_obj=__liwc_cats_for_social_5(), root_path=root_path, prefix_str='liwc_social-5')
    __liwc_create_test_sets(test_df=dataset.get_test(), cats_obj=__liwc_cats_for_social_5(), root_path=root_path, prefix_str='liwc_social-5')

    print('Creating social -  3 class: {}'.format(__liwc_cats_for_social_3()))
    __liwc_create_train_sets(train_df=dataset.get_train(), cats_obj=__liwc_cats_for_social_3(), root_path=root_path, prefix_str='liwc_social-3')
    __liwc_create_test_sets(test_df=dataset.get_test(), cats_obj=__liwc_cats_for_social_3(), root_path=root_path, prefix_str='liwc_social-3')
    
    print('Creating bio')
    __liwc_create_train_sets(train_df=dataset.get_train(), cats_obj=__liwc_cats_for_bio(), root_path=root_path, prefix_str='liwc_bio-5')
    __liwc_create_test_sets(test_df=dataset.get_test(), cats_obj=__liwc_cats_for_bio(), root_path=root_path, prefix_str='liwc_bio-5')

    print('Creating personal concerns')
    __liwc_create_train_sets(train_df=dataset.get_train(), cats_obj=__liwc_cats_for_pconcer(), root_path=root_path, prefix_str='liwc_pconcern-7')
    __liwc_create_test_sets(test_df=dataset.get_test(), cats_obj=__liwc_cats_for_pconcer(), root_path=root_path, prefix_str='liwc_pconcern-7')
    
    print('Done!')


if __name__ == "__main__":
    create_liwc_datasets()
