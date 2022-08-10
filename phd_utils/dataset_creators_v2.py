"""
Load from either raw or csv
annotate words
map or exclude the ones that are out of scope
divide into train and test
train can be also be divided into train and valid
write to files
"""
import logging
import pandas as pd
import urllib.parse
from typing import Set, Dict, Union
from tqdm import tqdm

from phd_utils.annotators.dict import DictBase
from phd_utils.resources import list_names, list_stopwords


class DataSetCreator:
    LABEL_COL_NAME = 'label'

    def __init__(self, labeler: DictBase):
        self.__labeler = labeler
        self.__corpus_df: Union[pd.DataFrame, None] = None
        self.__logger = logging.getLogger(__name__)
        self.__label_counts_dict: Union[Dict[str, int], None] = None
        self.__test_df: Union[pd.DataFrame, None] = None
        self.__train_df: Union[pd.DataFrame, None] = None
        self.__valid_df: Union[pd.DataFrame, None] = None

    def load_raw(self, path_str: str, remove_names=True, remove_stop_words=True):
        """
        Read from '.dat' file. expected to contain columns:
        word, meaning, example, numLikes, numDislikes, tags
        The function adds diff likes column
        """
        self.__logger.info(
            'Reading raw file {} with remove names = {} and remove stop words = {}'.format(
                path_str, remove_names, remove_stop_words))
        raw_df = pd.read_csv(path_str, sep=r'\|', engine='python')
        DataSetCreator.__fix_text_column_inplace(raw_df, 'word', do_lower=True)
        DataSetCreator.__fix_text_column_inplace(raw_df, 'meaning')
        DataSetCreator.__fix_text_column_inplace(raw_df, 'example')

        raw_df = DataSetCreator.__remove_illegal_entries(raw_df)
        raw_df['diffLikes'] = pd.to_numeric(raw_df.numLikes) - pd.to_numeric(raw_df.numDislikes)

        if remove_stop_words:
            raw_df = DataSetCreator.filter(raw_df, list_stopwords(), 'word')
        if remove_names:
            raw_df = DataSetCreator.filter(raw_df, list_names(), 'word')
        self.__logger.info('Done loading raw file {}'.format(path_str))

        tqdm.pandas()

        def get_label(w):
            labels_lst = self.__labeler.label_word(w)
            if len(labels_lst) > 0:
                return labels_lst[0]
            else:
                return ''

        raw_df[DataSetCreator.LABEL_COL_NAME] = raw_df.word.progress_apply(get_label)   # TODO fix for multi-annotation
        raw_df = raw_df[raw_df[DataSetCreator.LABEL_COL_NAME].apply(lambda l: len(l) > 0)]
        self.__label_counts_dict = raw_df[DataSetCreator.LABEL_COL_NAME].value_counts().to_dict()
        self.__corpus_df = raw_df
        self.__logger.info('Corpus ready to use with size {}'.format(self.__corpus_df.size))

    @staticmethod
    def __fix_text_column_inplace(content_df: pd.DataFrame, column_name: str, do_lower: bool = False) -> None:
        content_df[column_name].fillna('', inplace=True)
        content_df[column_name] = content_df[column_name].apply(DataSetCreator.__decode_text)
        if do_lower:
            content_df[column_name] = content_df[column_name].str.lower()

    @staticmethod
    def __decode_text(text_str: str) -> str:
        return urllib.parse.unquote_plus(text_str)

    @staticmethod
    def __remove_illegal_entries(content_df: pd.DataFrame) -> pd.DataFrame:
        content_df = content_df[content_df['numLikes'].astype(str).str.fullmatch(r'\d+')]
        content_df = content_df[content_df['numDislikes'].astype(str).str.fullmatch(r'\d+')]
        return content_df

    @staticmethod
    def filter(content_df: pd.DataFrame, remove_set: Set[str], column_str: str):
        return content_df[~content_df[column_str].isin(remove_set)]

    @staticmethod
    def __get_copy_or_none(content_df):
        if content_df is not None:
            copy_df = content_df.copy(deep=True)
            if 'liwc' in copy_df.columns:
                # 'deep=True' not enough for lists in columns
                copy_df[DataSetCreator.LABEL_COL_NAME] = copy_df.liwc.apply(lambda l: l.copy())
            return copy_df
        else:
            return None

    def get_label_counts(self):
        return self.__label_counts_dict.copy()

    def generate_dataset(self,
                         test_percent=None,
                         test_count=None,
                         valid_percent=None,
                         train_top_n=None,
                         train_min_diff=None,
                         random_seed=0) -> None:
        """
        Generates the dataset. Test set data is excluded from train and validation
        :param test_percent: Percentage of test data. Either this or test_count has to be specified
        :param test_count: Count of test data. Either this or test_percent has to be specified
        :param valid_percent: Optional valid percentage
        :param random_seed: rand seed used for shuffling data
        :param train_top_n: Optional. Choose only top N for training set examples
        :param train_min_diff: Optional. Choose only samples with min difference higher for training set examples
        """
        if test_percent is None and test_count is None:
            raise ValueError('Either test_percent or test_count has to be specified')
        if test_percent is None:
            test_percent = float(test_count) / self.__corpus_df.word.unique().size
            if test_percent >= 1:
                raise ValueError('Test set percentage is too high {}, unique word values is {}'.format(test_count, self.__corpus_df.word.unique().size))
        if test_percent >= 1:
            raise ValueError('Test percentage {} is too big compared to corpus size {}.'
                             .format(test_percent, self.__corpus_df.size))

        samples_set: pd.DataFrame = DataSetCreator.__get_copy_or_none(self.__corpus_df)
        samples_set = samples_set.sort_values(by='diffLikes', ascending=False)

        self.__logger.info('Generating test set with size {}'.format(test_count))
        training_available_df = self.__generate_test(samples_set, test_percent, random_seed)
        self.__logger.info('Test set generated with actual {} samples'.format(self.__test_df.size))

        self.__logger.info('Generating training set using the remaining {} samples'.format(training_available_df.size))
        self.__logger.info('Training parameters Top N: {}, Min Diff: {} and validation percentage {}'
                           .format(train_top_n, train_min_diff, valid_percent))
        self.__generate_train_set(training_available_df, valid_percent, train_top_n, train_min_diff, random_seed)
        self.__logger.info('Training set count is {}'.format(self.__train_df.size))
        if self.__valid_df is not None:
            self.__logger.info('Valid set count is {}'.format(self.__valid_df.size))

    def __generate_test(self,
                        corpus_df: pd.DataFrame,
                        percentage: float,
                        random_seed,
                        min_diff_int: int = 1) -> pd.DataFrame:
        """
        Selects samples for the test set and assign it to the class and returns remaining available samples for
        training. All records corresponding to words selected for testing are removed from the returned data frame
        :param corpus_df: samples sorted based on likes difference
        :param percentage:
        :param random_seed:
        :param min_diff_int: Minimum value (inclusive) for difference in like to accept for test sample. If None it will
                             be ignored
        :return: dataframe for available samples for training
        """
        test_label_counts_dict = {label: int(count * percentage) for label, count in self.__label_counts_dict.items()}
        self.warn_on_zeros(test_label_counts_dict)

        if min_diff_int is not None:
            corpus_df = corpus_df[corpus_df.diffLikes >= min_diff_int]

        top_df = corpus_df.groupby('word', as_index=False).head(1)
        top_gp = top_df.groupby(DataSetCreator.LABEL_COL_NAME)
        selected_df_lst = []
        selected_words_lst = []
        for label, count in test_label_counts_dict.items():
            if label in top_gp.groups.keys():
                label_df = top_gp.get_group(label).head(count)
                if label_df.size < count:
                    self.__logger.warning('Test samples for "{}" has size {} less than the computed split size {}'
                                          .format(label, label_df.size, count))
                selected_df_lst.append(label_df)
                selected_words_lst.extend(label_df.word.unique())
            else:
                self.__logger.warning('No Samples found for {}'.format(label))

        self.__test_df = pd.concat(selected_df_lst).sample(frac=1, random_state=random_seed)
        corpus_df = corpus_df[~corpus_df.word.isin(set(selected_words_lst))]
        return corpus_df

    def warn_on_zeros(self, label_count_dict: Dict[str, int]) -> None:
        for label, count in label_count_dict.items():
            if count == 0:
                self.__logger.warning('Label {} has ZERO samples in the test set'.format(label))

    def __generate_train_set(self,
                             corpus_df: pd.DataFrame,
                             valid_percent: float,
                             top_n: int,
                             min_diff: int,
                             random_seed: int) -> None:
        corpus_df = corpus_df.sort_values(by='diffLikes', ascending=False)
        if top_n is not None:
            corpus_df = corpus_df.groupby('word', as_index=False).head(top_n)
        if min_diff is not None:
            corpus_df = corpus_df[corpus_df.diffLikes >= min_diff]
        if valid_percent is not None:
            train_df_lst = []
            valid_df_lst = []
            corpus_gb = corpus_df.groupby(by=DataSetCreator.LABEL_COL_NAME, as_index=False)
            for label in corpus_gb.groups.keys():
                label_df = corpus_gb.get_group(label)
                label_df = label_df.sample(frac=1, random_state=random_seed)
                valid_count = int(label_df.size * valid_percent)
                valid_df_lst.append(label_df.iloc[:valid_count])
                train_df_lst.append(label_df.iloc[valid_count:])
            corpus_df = pd.concat(train_df_lst)
            self.__valid_df = pd.concat(valid_df_lst).sample(frac=1, random_state=random_seed)
        self.__train_df = corpus_df.sample(frac=1, random_state=random_seed)

    def get_train(self) -> pd.DataFrame:
        return DataSetCreator.__get_copy_or_none(self.__train_df)

    def get_test(self) -> pd.DataFrame:
        return DataSetCreator.__get_copy_or_none(self.__test_df)

    def get_valid(self) -> pd.DataFrame:
        return DataSetCreator.__get_copy_or_none(self.__valid_df)
