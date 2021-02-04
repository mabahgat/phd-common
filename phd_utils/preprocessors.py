from abc import ABC, abstractmethod
import re
from phd_utils.nlp_utils import get_sentence_pos, match_pronoun_present
from lemminflect import getInflection, getLemma

class Preprocssor(ABC):

    @abstractmethod
    def apply(self, text_str: str) -> str:
        pass

class PronounSwap:
    """
    Tokenes used are ['subject', 'object', 'possive adjective', 'possive pronoun', 'reflexive pronoun'] (starting from index 5)
    list below needs to be in lower case
    """
    FIRST_PERSON_lst = ['i\'m', 'i am', 'i was', 'i\'ve', 'i have', 'i', 'me', 'my', 'mine', 'myself']
    SECOND_PERSON_lst = ['you\'re', 'you are', 'you were', 'you\'ve', 'you have', 'you', 'you', 'your', 'yours', 'yourself']
    MALE_THIRD_PERSON_lst = ['he\'s', 'he is', 'he was', 'he\'s', 'he has', 'he', 'him', 'his', 'his', 'himself']
    FEMALE_THIRD_PERSON_lst = ['she\'s', 'she is', 'she was', 'she\'s', 'she has', 'she', 'her', 'her', 'hers', 'herself']

    @staticmethod
    def __as_regex(in_str: str) -> str:
        return re.compile(r"\b{}\b".format(in_str).replace(" ", "\\s+"), flags=re.IGNORECASE)

    def __init__(self, one_pronoun_lst, two_pronoun_lst):
        self.__one_pronoun_lst = one_pronoun_lst
        self.__two_pronoun_lst = two_pronoun_lst
        self.__one_pronoun_pattern_lst = [PronounSwap.__as_regex(txt_str) for txt_str in one_pronoun_lst]
        self.__two_pronoun_pattern_lst = [PronounSwap.__as_regex(txt_str) for txt_str in two_pronoun_lst]
    
    @staticmethod
    def __mark_word(word_str: str) -> str:
        return "<<MARK-{}-MARK>>".format(word_str)
    
    @staticmethod
    def __get_replacement(word_str, source_lst, target_lst):
        index = source_lst.index(word_str)
        return target_lst[index]
    
    @abstractmethod
    def _fix_verb(self, verb_str: str, pos_str: str, new_pronoun_str: str, old_pronoun_str: str) -> str:
        pass

    def _swap(self, text_str: str) -> str:
        text_pos_lst = get_sentence_pos(text_str)
        result_lst = []
        fix_next_verb_bool = False
        pronoun_change_pair = None
        changed_bool = False
        for word_str, pos_str in text_pos_lst:
            i_word_str = word_str.lower()
            if pos_str.startswith('VB') and fix_next_verb_bool:
                new_word_str = self._fix_verb(i_word_str, pos_str, pronoun_change_pair[0], pronoun_change_pair[1])
                fix_next_verb_bool = False
                pronoun_change_pair = None
            else:
                if i_word_str in self.__one_pronoun_lst:
                    new_word_str = PronounSwap.__get_replacement(i_word_str, self.__one_pronoun_lst, self.__two_pronoun_lst)
                    changed_bool = True
                elif i_word_str in self.__two_pronoun_lst:
                    new_word_str = PronounSwap.__get_replacement(i_word_str, self.__two_pronoun_lst, self.__one_pronoun_lst)
                    changed_bool = True
                else:
                    new_word_str = word_str
                if changed_bool and pos_str == 'PRP':
                    fix_next_verb_bool = True
                    pronoun_change_pair = (new_word_str, i_word_str)
            result_lst.append(new_word_str)
        return ' '.join(result_lst)


class SwapIAndYou(PronounSwap):

    def __init__(self):
        super().__init__(PronounSwap.FIRST_PERSON_lst, PronounSwap.SECOND_PERSON_lst)

    def apply(self, text_str: str) -> str:
        return self._swap(text_str)

    def _fix_verb(self, verb_str: str, pos_str: str, new_pronoun_str: str, old_pronoun_str: str) -> str:
        if pos_str == 'VBD':
            if new_pronoun_str == 'i' and verb_str == 'were':
                return 'was'
            if new_pronoun_str == 'you' and verb_str == 'was':
                return 'were'
        if pos_str == 'VB' or pos_str == 'VBP':
            if new_pronoun_str == 'i' and verb_str == 'are':
                return 'am'
            if new_pronoun_str == 'you' and (verb_str == 'am' or verb_str == '\'m'):
                return 'are'
        return verb_str


class SwapIAndHe(PronounSwap):

    def __init__(self):
        super().__init__(PronounSwap.FIRST_PERSON_lst, PronounSwap.MALE_THIRD_PERSON_lst)
    
    def apply(self, text_str: str) -> str:
        return self._swap(text_str)
    
    def _fix_verb(self, verb_str: str, pos_str: str, new_pronoun_str: str, old_pronoun_str: str) -> str:
        if (pos_str == 'VB' or pos_str == 'VBP') and new_pronoun_str == 'he':
            return match_pronoun_present(verb_str, 'he')
        if (pos_str == 'VBZ' and new_pronoun_str == 'i'):
            return match_pronoun_present(verb_str, 'i')
        return verb_str
            

class SwapIAndShe(PronounSwap):

    def __init__(self):
        super().__init__(PronounSwap.FIRST_PERSON_lst, PronounSwap.FEMALE_THIRD_PERSON_lst)
    
    def apply(self, text_str: str) -> str:
        return self._swap(text_str)
    
    def _fix_verb(self, verb_str: str, pos_str: str, new_pronoun_str: str, old_pronoun_str: str) -> str:
        if (pos_str == 'VB' or pos_str == 'VBP') and new_pronoun_str == 'she':
            return match_pronoun_present(verb_str, 'she')
        if (pos_str == 'VBZ' and new_pronoun_str == 'i'):
            return match_pronoun_present(verb_str, 'i')
        return verb_str
