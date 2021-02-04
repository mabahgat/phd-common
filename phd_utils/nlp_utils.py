import logging

import nltk
from nltk.tokenize.casual import TweetTokenizer
from lemminflect import getInflection, getLemma


__pronoun_to_verb_upenn_dict = {
    'i': 'VBP',
    'you': 'VBP',
    'we': 'VBP',
    'he': 'VBZ',
    'she': 'VBZ',
    'they': 'VBZ'
}

__modal_verbs_set = set(['will', 'can', 'would', 'may', 'must', 'should', 'might', 'ought'])


def get_sentence_pos(text_str: str):
    """
    Part of Speech for each word.
    Reference tags: https://pythonprogramming.net/part-of-speech-tagging-nltk-tutorial/
    """
    #word_tokenizer = TweetTokenizer()
    sent_pos_lst = []
    sent_lst = nltk.tokenize.sent_tokenize(text_str)
    for sent_str in sent_lst:
        #words_lst = word_tokenizer.tokenize(sent_str)
        words_lst = nltk.word_tokenize(sent_str)
        word_pos_lst = nltk.pos_tag(words_lst)
        sent_pos_lst.extend(word_pos_lst)
    return sent_pos_lst


def match_pronoun_present(verb_str: str, pronoun_str: str) -> str:
    """
    Returns a verb form that matches the passed pronoun.
    This function should only be used for present tense
    """
    pronoun_str = pronoun_str.lower()
    if pronoun_str not in __pronoun_to_verb_upenn_dict.keys():
        raise ValueError('Unexpected value for pronoun "{}"'.format(pronoun_str))
    aff_verb_str, negation_str = split_verb_negation(verb_str)
    if is_modal_verb(aff_verb_str):
        return verb_str
    lemma_lst = getLemma(aff_verb_str, "VERB")
    lemma_lst = __collapse_lemma_list(lemma_lst)
    if len(lemma_lst) != 1:
        logging.warning('WARNING: Ambigous or no lemma for "{}". Output was {}. Keeping original verb.'.format(verb_str, lemma_lst))
        return verb_str
    lemma_str = lemma_lst[0]
    inflect_lst = getInflection(lemma_str, __pronoun_to_verb_upenn_dict[pronoun_str])
    if len(inflect_lst) > 2 or not len(inflect_lst):
        logging.warning('WARNING: Ambigous or no inflection list for lemma "{}" from verb "{}". Output was {}. Keeping original verb.'.format(lemma_str, verb_str, inflect_lst))
        return verb_str
    elif len(inflect_lst) == 2:
        if pronoun_str == 'i':
            new_verb_str = inflect_lst[0]
        else:
            new_verb_str = inflect_lst[1]
    else:
        new_verb_str = inflect_lst[0]
    return merge_verb_negation(new_verb_str, negation_str)


def is_modal_verb(verb_str: str) -> bool:
    return verb_str.lower() in __modal_verbs_set


def __collapse_lemma_list(lemma_lst):
    fixed_lemma_set = set([lemma_str.replace('-', '') for lemma_str in lemma_lst])
    return [lemma_str for lemma_str in lemma_lst if lemma_str in fixed_lemma_set] # preserve list order


def merge_verb_negation(verb_str: str, negation_str: str) -> str:
    if verb_str == 'am' and negation_str == "n't":
        return "am not"
    if verb_str == 'can' and negation_str == "n't":
        return "can't"
    return '{}{}'.format(verb_str, negation_str)


def split_verb_negation(verb: str) -> (str, str):
    """
    splits negation from verb
    returns: verb and negation if there is negation else verb and empty string
    """
    if verb == "can't":
        return 'can', "n't"
    if verb.endswith("n't"):
        return verb[0:-3], "n't"
    elif verb == 'cannot':
        return 'can', 'not'
    return verb, ''
