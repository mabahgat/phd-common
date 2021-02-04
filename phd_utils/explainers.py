from abc import ABC, abstractmethod
import re
from typing import List

from nltk import word_tokenize, sent_tokenize
from lime.lime_text import LimeTextExplainer


class ExplainerBase(ABC): # Previously named "Explainer"

    SCORE_SUM_KEY = 'score sum'
    TOP_COUNT_KEY = 'top count'
    RANK_AVERAGE_KEY = 'average rank'
    COUNT_KEY = 'count'
    SENTENCE_COUNT = 'sentence count'
    TOP_RATE_KEY = 'top rate'
    SCORE_RATE_KEY = 'score rate'
    IN_TOP_COUNT_KEY = 'in top count' # only used for internal calculation, should not be a sorting criteri

    def __init__(self, tokenize_fn, lower_case_bool=False):
        self._tokenize_fn = tokenize_fn
        self._lower_case_bool = lower_case_bool

    @abstractmethod
    def explain(self, text):
        """
        Returns an explainer object for each text instance passed
        :param text: string or list
        """
        pass

    @staticmethod
    def __init_object():
        return {
            ExplainerBase.SCORE_SUM_KEY: 0,
            ExplainerBase.TOP_COUNT_KEY: 0,
            ExplainerBase.COUNT_KEY: 0,
            ExplainerBase.SENTENCE_COUNT: 0,
            ExplainerBase.RANK_AVERAGE_KEY: 0,
            ExplainerBase.IN_TOP_COUNT_KEY: 0,
            ExplainerBase.TOP_RATE_KEY: None,
            ExplainerBase.SCORE_RATE_KEY: None,
        }
    
    @abstractmethod
    def _relevence_score(self, exp):
        pass

    def explain_with_details(self, text_lst):
        if not self._tokenize_fn:
            raise ValueError("Tokenizer needs to be initialized too use this explain_with_details")
        words_dict = {}
        max_rank = 0

        for text_str in text_lst:
            if self._lower_case_bool:
                text_str = text_str.lower()

            token_lst = self._tokenize_fn(text_str)

            for word_str in set(token_lst):
                if word_str not in words_dict:
                    words_dict[word_str] = ExplainerBase.__init_object()
                words_dict[word_str][ExplainerBase.SENTENCE_COUNT] += 1

            for word_str in token_lst:
                words_dict[word_str][ExplainerBase.COUNT_KEY] += 1
            
            exp = self.explain(text_str)
            top_words_lst = self._relevence_score(exp)
            
            for rank, word_score_pair in enumerate(top_words_lst):
                word_str = word_score_pair[0]
                score = word_score_pair[1]
                words_dict[word_str][ExplainerBase.SCORE_SUM_KEY] += score
                if rank == 0:
                    words_dict[word_str][ExplainerBase.TOP_COUNT_KEY] += 1
                words_dict[word_str][ExplainerBase.RANK_AVERAGE_KEY] += rank + 1
                words_dict[word_str][ExplainerBase.IN_TOP_COUNT_KEY] += 1
                if max_rank < rank:
                    max_rank = rank

            for word_str in words_dict:
                in_top_count = words_dict[word_str][ExplainerBase.IN_TOP_COUNT_KEY]
                sentence_count = words_dict[word_str][ExplainerBase.SENTENCE_COUNT]
                score_sum = words_dict[word_str][ExplainerBase.SCORE_SUM_KEY]
                rank_sum = words_dict[word_str][ExplainerBase.RANK_AVERAGE_KEY]

                top_count = words_dict[word_str][ExplainerBase.TOP_COUNT_KEY]
                score_sum = words_dict[word_str][ExplainerBase.SCORE_SUM_KEY]

                words_dict[word_str][ExplainerBase.TOP_RATE_KEY] = top_count / sentence_count
                words_dict[word_str][ExplainerBase.SCORE_RATE_KEY] = score_sum / sentence_count
                words_dict[word_str][ExplainerBase.RANK_AVERAGE_KEY] = (rank_sum + max_rank * (sentence_count - in_top_count)) / sentence_count

        return words_dict
    
    @staticmethod
    def sort_details(details_dict, criteria_str, descending_bool=True):
        if criteria_str not in set([
            ExplainerBase.SCORE_SUM_KEY, 
            ExplainerBase.TOP_COUNT_KEY, 
            ExplainerBase.COUNT_KEY, 
            ExplainerBase.SENTENCE_COUNT,
            ExplainerBase.RANK_AVERAGE_KEY,
            ExplainerBase.TOP_RATE_KEY,
            ExplainerBase.SCORE_RATE_KEY]):
            raise ValueError("Unrecognized criteria {}".format(criteria_str))
        return sorted(details_dict.items(), key=lambda item: item[1][criteria_str], reverse=descending_bool)


def tokenize_for_lime(text_str: str) -> List[str]:
    text_str = re.sub(r'([.?!])(\w)', r'\g<1> \g<2>', text_str) # sentence break with no space
    word_only_matcher = re.compile('.*[a-zA-Z]$')
    token_lst = []
    sent_lst = sent_tokenize(text_str)
    for sent_str in sent_lst:
        word_lst = word_tokenize(sent_str)
        token_lst.extend(word_lst)
    token_lst = [token_str for token_str in token_lst if word_only_matcher.match(token_str)]
    return token_lst


class Lime(ExplainerBase):

    def __init__(self, class_names, investigate_labels, prediction_fn, tokenize_fn=tokenize_for_lime, num_features=20, num_samples=20, use_top_labels=True, lower_case_bool=False):
        self.class_names = class_names

        self.__investigate_labels_lst = investigate_labels
        self.__num_features = num_features
        self.__num_samples = num_samples
        self.__use_top_labels_bool = use_top_labels

        self.__prediction_fn = prediction_fn

        if tokenize_fn:
            self.__explainer = LimeTextExplainer(class_names=class_names, split_expression=tokenize_fn, random_state=0)
        else:
            self.__explainer = LimeTextExplainer(class_names=class_names)
        super().__init__(tokenize_fn=tokenize_fn, lower_case_bool=lower_case_bool)
    
    def explain(self, text):
        if type(text) == str:
            if self._lower_case_bool:
                text = text.lower()
            return self.__explainer.explain_instance(text, self.__prediction_fn, num_features=self.__num_features, num_samples=self.__num_samples, labels=self.__investigate_labels_lst)
        elif type(text) == list:
            if not len(text) or type(text[0]) != str:
                raise ValueError('Passed list must be a list of strings with more than one item')
            return [self.explain(text_str) for text_str in text]
    
    def _relevence_score(self, exp):
        score_lst = exp.as_list()
        score_lst = [(pair[0], abs(pair[1])) for pair in score_lst] # FIXME possible work around
        sorted_score_lst = sorted(score_lst, key=lambda pair: pair[1], reverse=True)
        return sorted_score_lst
