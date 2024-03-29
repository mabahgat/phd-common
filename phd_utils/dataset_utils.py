from sklearn import metrics
import pandas as pd
from typing import List

from phd_utils.models_v2 import SequenceClassificationModel
from phd_utils.datasets_v2 import UrbanDictWithLiwc


class UrbanDictWithLiwcCategoryMapper:

    def __init__(self, mappings_dict):
        self.__mappings_dict = mappings_dict
    
    def map_to_first_liwc_cat(self, set_df: pd.DataFrame, filter_empty=True):
        """
        Map LIWC categories list to the first category found
        """
        set_df = set_df.copy()
        set_df['liwc'] = set_df['liwc'].apply(lambda liwc_cats_str: self.use_first_liwc_category(liwc_cats_str))
        if filter_empty:
            set_df = set_df[~set_df['liwc'].str.fullmatch('^$')]
        return set_df

    def use_first_liwc_category(self, liwc_cats_str: str) -> str:
        liwc_cats_str = str(liwc_cats_str)
        cats_lst = liwc_cats_str.split('|')
        for cat_str in cats_lst:
            if cat_str in self.__mappings_dict:
                return self.__mappings_dict[cat_str] # pick the first one
        return ""

    def map(self, set_df: pd.DataFrame, filter_empty=True):
        new_set_df = set_df.copy()
        new_set_df['liwc'] = new_set_df['liwc'].apply(lambda liwc_cats_str: self.get_mapped_liwc_categories(liwc_cats_str))
        return new_set_df
    
    def map_to_every_category(self, set_df: pd.DataFrame, filter_empty=True):
        """
        Map LIWC cateogries to the used list. If the result has multiple categories
        then create a new example for each
        """
        new_rows_dict = {}
        new_index = 0
        
        for _, row in set_df.iterrows():
            cat_lst = self.get_mapped_liwc_categories(row['liwc'])
            if len(cat_lst) == 0 and not filter_empty or len(cat_lst) == 1:
                row['liwc'] = cat_lst[0]
                new_rows_dict[new_index] = row
                new_index += 1
            elif len(cat_lst) > 1:
                for cat_str in cat_lst:
                    new_row = row.copy()
                    new_row['liwc'] = cat_str
                    new_rows_dict[new_index] = new_row
                    new_index += 1
        return pd.DataFrame.from_dict(new_rows_dict, orient='index', columns=set_df.columns)
    
    def get_mapped_liwc_categories(self, liwc_cats_str: str) -> List[str]:
        liwc_cats_str = str(liwc_cats_str)
        cats_lst = liwc_cats_str.split('|')
        return cats_lst # TODO fix that - this is temp because we need the classes as is not mapped to parents
        mapped_cats_set = set()
        for cat_str in cats_lst:
            if cat_str in self.__mappings_dict:
                mapped_cats_set.add(self.__mappings_dict[cat_str])
        return list(mapped_cats_set)
        

    CLASS_MAPPING_dict = {
        'affect': 'affect',
        'posemo': 'affect',
        'negemo': 'affect',
        'anx': 'affect',
        'anger': 'affect',
        'sad': 'affect',
        'social': 'social',
        'family': 'social',
        'friend': 'social',
        'female': 'social',
        'male': 'social',
        'cogproc': 'cogproc',
        'insight': 'cogproc',
        'cause': 'cogproc',
        'discrep': 'cogproc',
        'tentat': 'cogproc',
        'certain': 'cogproc',
        'differ': 'cogproc',
        'percept': 'percept',
        'see': 'percept',
        'hear': 'percept',
        'feel': 'percept',
        'bio': 'bio',
        'body': 'bio',
        'health': 'bio',
        'sexual': 'bio',
        'ingest': 'bio',
        'drives': 'drives',
        'affiliation': 'drives',
        'achiev': 'drives',
        'power': 'drives',
        'reward': 'drives',
        'risk': 'drives',
        'relativ': 'relativ',
        'motion': 'relativ',
        'space': 'relativ',
        'time': 'relativ',
        'pconcern': 'pconcern',
        'work': 'pconcern',
        'leisure': 'pconcern',
        'home': 'pconcern',
        'money': 'pconcern',
        'relig': 'pconcern',
        'death': 'pconcern',
        'informal': 'informal',
        'swear': 'informal',
        'netspeak': 'informal',
        'assent': 'informal',
        'nonflu': 'informal',
        'filler': 'informal'
    }


class UrbanDictWithLiwcUtil: # TODO remove this and fix model evaluation
    """
    Helper tools
    """

    @staticmethod
    def eval_with_multilabel(dataset: UrbanDictWithLiwc, model: SequenceClassificationModel, normalize=None, set_type='test'):
        y_test, y_hat_test = model.get_test_ref_out_pair(dataset, set_type)
        y_test_multi = UrbanDictWithLiwcUtil.y_multilabel(dataset, set_type)
        y_hat_test_adjust = [ref if out in ref_lst else out for ref, ref_lst, out in zip(y_test, y_test_multi, y_hat_test)]
        
        detailed_eval = metrics.classification_report(y_test, y_hat_test_adjust, target_names=dataset.class_names(), output_dict=True)
        confusion_matrix = metrics.confusion_matrix(y_test, y_hat_test_adjust, normalize=normalize)
        return {
            'eval': detailed_eval,
            'confusion_matrix': confusion_matrix
        }
        
    
    @staticmethod
    def y_multilabel(dataset: UrbanDictWithLiwc, set_type='test'):
        liwc_mapper = UrbanDictWithLiwcCategoryMapper(UrbanDictWithLiwcCategoryMapper.CLASS_MAPPING_dict)
        class_names_lst = dataset.class_names()

        if set_type == 'test':    
            test_path_str = dataset.get_path()['test']
            test_df = pd.read_csv(test_path_str, index_col=0)

            y_test_multi_lst = test_df['liwc'].apply(lambda cats_str: liwc_mapper.get_mapped_liwc_categories(cats_str)).to_list()
            return [[class_names_lst.index(cat_str) for cat_str in cat_lst] for cat_lst in y_test_multi_lst]

        elif set_type == 'dev' or set_type == 'val':
            train_df = pd.read_csv(dataset.get_path()['train'])
            y_valid_multi_lst = train_df['liwc'].iloc[dataset.valid_index_lst].apply(lambda cats_str: liwc_mapper.get_mapped_liwc_categories(cats_str)).to_list()
            return [[class_names_lst.index(cat_str) for cat_str in cat_lst] for cat_lst in y_valid_multi_lst]

        else:
            raise ValueError('Unexpected Set type {}'.format(set_type))
    
    @staticmethod
    def eval_with_voting(dataset: UrbanDictWithLiwc, model: SequenceClassificationModel, normalize=None, set_type='test'): # TODO very hacky
        """
        Group same words together and use the most frequent label to assign to that word
        """
        y_test, y_hat_test = model.get_test_ref_out_pair(dataset, set_type)
        y_hat_test_prob = model.get_test_out_prob(dataset, set_type)
        y_test_multi = UrbanDictWithLiwcUtil.y_multilabel(dataset, set_type)
        y_hat_test_adjust = [ref if out in ref_lst else out for ref, ref_lst, out in zip(y_test, y_test_multi, y_hat_test)]

        test_df = pd.read_csv(dataset.get_path()['test'])
        test_df['y_test'] = y_test
        test_df['y_hat_test'] = y_hat_test_adjust
        test_df['y_hat_test_prob'] = list(y_hat_test_prob)

        def label_of(labels_sr: pd.Series) -> str:
            assert(len(set(labels_sr.to_list())) == 1)
            return labels_sr.to_list()[0]
        
        def most_frequent_label_of(labels_sr: pd.Series) -> str:
            label_lst = labels_sr.to_list()
            return max(set(label_lst), key=label_lst.count)
        
        def avg_prob(label, label_prob_df: pd.DataFrame) -> float:
            return label_prob_df[label_prob_df.y_hat_test == label].y_hat_test_prob.mean()
        
        def squash_groups(word_gp):
            label = most_frequent_label_of(word_gp['y_hat_test'])
            d = {
                'y_test': label_of(word_gp['y_test']),
                'y_hat_test': label,
                'y_hat_test_prob': avg_prob(label, word_gp[['y_hat_test', 'y_hat_test_prob']])
            }
            return pd.Series(d)

        test_voted_df = test_df.groupby('word').apply(squash_groups)

        y_test = test_voted_df.y_test.to_list()
        y_hat_test = test_voted_df.y_hat_test.to_list()

        detailed_eval = metrics.classification_report(y_test, y_hat_test, target_names=dataset.class_names(), output_dict=True)
        confusion_matrix = metrics.confusion_matrix(y_test, y_hat_test, normalize=normalize)
        return {
            'eval': detailed_eval,
            'confusion_matrix': confusion_matrix,
            'test_df': test_df,
            'test_voted_df': test_voted_df
        }

