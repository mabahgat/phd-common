import pandas as pd
from typing import Union, Set, List, Dict
from random import Random
import re

from phd_utils import global_config


__root_9_task_description_str = """
In this job, we try to categorise words into given categories. We have 9 categories in total, including emotion, cognition, biology, ... etc; and the objective is to check if a given word is related to this category or not.

You will be shown the word used in an example sentence. Also you will get a definition of this word. Then you will be shown a category that we think this word relates to. All your task is to say wether this word can belong to this category or not.

A simple example, you will be shown the word "love" in a sentence, then get a definition for the word, and you will be asked if the word "love" belongs to the category "affect/emotions", and in this case your answer should be "Yes".

A more complex example, the word "circle" would normally used in the context of shapes. But it can also mean inspect in a context like: "the man circled the car to make sure there were no scratches on its paint". As such, the word "circle" can be considered as related to "perception".

Important Notes:

- Please read the example and definition of each word very carefully before you decide it belongs to a given category or not.

- Many of the words are slang words that are used in informal and non-standard context. So please judge on the shown example and definition not the general use.

- If you think the word belongs better to another category than the shown one, but still can belong to the shown category, please say "yes" independently of the other matching categories you might consider. Only say "no" when you think the word as shown in the example should NOT belong to this shown category.

- We show the definition of each category with each word, consider revising every time to be sure about your decision.
"""


__cat_to_description_dict = {
    'affect': 'For words that are related to feelings and people experiences, including positive emotions as well as negative emotions as anger, sadness, and anxiety. Examples would be: "love", "hate", "abuse".',
    'bio': 'For words related to biology, biological process, or biological classification including body parts, health, sexuality and ingestion. Examples are: "addiction", "ache", "arm", "leg", "male", "sleep".',
    'cogproc': 'For words related to any form of cognitive process like insights, findings, experimentation, causality, discrepancies, tentativeness, certainty and differentiation. Examples are: "try", "result" (both noun and verb forms), "coherence", "clear", "clarify", "understand".',
    'drives': 'For words related to personal drives or deterrents such as affiliation, achievements, power, reward, encouragment and risk for personal drives and disassociation, expulsion, failure, lack of power, discouragment and punishment for personal deterrents. Examples are "achieve", "above", "acclaimed", "ahead", "army", "slave", "shame".',
    'informal': 'For words that are considered informal. These words can include swearing, common words used throughout social media, blogs and the internet, assent, emojis, non-fluencies, and filter words. These words may have large intersection with other categories. Examples are: "dude" , "cc", "oh", ":D", "mmm", "btw".',
    'pconcern': 'For words related to only: work, leisure, home, money, religon and death. Words related to emotions or biology/biological process are NOT in this category.  Examples for (Work): "busy", "franchise", "engineering". Examples for (leisure): -> "surf", "club", "ball". Examples for (home): "tenant", "microwave". Examples for money: "scholarship", "lease", "pay". Examples for (religon): "afterlife", "God" (death) -> "zombie", "autopsy", "mortal".',
    'percept': 'For words related to seeing, hearing and feeling. Examples are "ache", "acid", "appearing", "color/colour", "cold".',
    'relativ': 'For words related to motion, space and time including locations, times, location of items, time of incidents, relative location of items with respect to each other, relative times of incidents to each other. Also include tools and items related to motion, location or time. Examples for (time): "before", "april", "ages". Examples for (space): "across", "interior", "floor". Examples for (motion): "fast", "past".',
    'social': 'For words related to social status, relations, events and interactions. Examples are: "her", "marry", "brother", "woman", "meet", "wedding".',

    'negemo': 'For words that can be used in expressing negative emotions including anger, sadness and anxiety. Examples are: "lame", "bad", "unfriendly", "lazy", sigh"',
    'posemo': 'For words that can be used in expressing positive emotions. Examples are: "funny", "safe", "happy", "thx", "calming"',

    'work': 'For words related to professions, careers, and the workplace. Example would be: "office", "desk", "mentor", "academia"',
    'leisure': 'For words related to leisure including businesses, items or actions. Examples are: "surf", "cinema", "burgers"',
    'money': 'For words related to monetary value including cash, finance, economy. Examples are: "account", "acquire", "debt", "coin"',
    'death': 'For words related to death and life. Examples are: "coffin", "alive", "autopsy", "kill"',
    'home': 'For words related to home; either the physical home itself, or actions related to home. Examples are: "carpet", "kitchen", "sweep", "cook"',
    'relig': 'For words related to religion and religious ideas. Examples are: "God", "pray", "amen", "Jesus", "afterlife"'
}


__type_to_category_set_dict = {
    'root_9': set(['affect', 'bio', 'cogproc', 'drives', 'informal', 'pconcern', 'percept', 'relativ', 'social'])
}


__category_readable_names_pairs_tpl = (
    ('affect', 'Affect / Emotions'),
    ('bio', 'Biology / Bioligical Process / Biological Classification'),
    ('cogproc', 'Cognition / Coginitive Process'),
    ('drives', 'Drives and Deterrents'),
    ('informal', 'Informal Speech / Writing'),
    ('pconcern', 'Personal Interests for: Work / Leisure / Home / Money / Religion / Death'),
    ('percept', 'Perception / Sensing'),
    ('relativ', 'Motion, Space and Time mentions Absolute and Relative'),
    ('social', 'Social'),

    ('negemo', 'Negative Emotions'),
    ('posemo', 'Positive Emotions')
)


__category_to_readable_name_dict = {cat_str: readable_str for cat_str, readable_str in __category_readable_names_pairs_tpl}


__readable_name_to_category_dict = {readable_str: cat_str for cat_str, readable_str in __category_readable_names_pairs_tpl}


def __get_label_description(label_str: str) -> str:
    return __cat_to_description_dict[label_str.lower()]


def __get_dataframe(in_file: Union[pd.DataFrame, str]) -> pd.DataFrame:
    if isinstance(in_file, str):
        in_df = pd.read_csv(in_file, index_col=0)
    else:
        in_df = in_file.copy(deep=True)
    return __filter_out_nan_example_and_meaning(in_df)


def __save_if_specified(df: pd.DataFrame, out_file_str: str):
    if out_file_str:
        df.to_csv(out_file_str)


def __get_exclude_indexes(exclude) -> List[int]:
    if not exclude:
        return None
    elif isinstance(exclude, pd.DataFrame):
        return exclude.index.to_list()
    elif isinstance(exclude, str):
        return pd.read_csv(exclude, index_col=0).index.to_list()
    elif isinstance(exclude, List[int]):
        return exclude
    else:
        raise TypeError('Unknown type')


def __order_inplace(df: pd.DataFrame, sort_by: Union[str,Dict[str, int]]):
    def return_order_or_none(word_str):
        if word_str in sort_by:
            return sort_by[word_str]
        else:
            return None
    if isinstance(sort_by, Dict):
        df['twitter_rank'] = df.word.apply(return_order_or_none)
        df.sort_values(by='twitter_rank', inplace=True)
    elif isinstance(sort_by, str):
        if sort_by == 'twitter':
            sort_by = get_twitter_most_frequent_words_rank()
            df['twitter_rank'] = df.word.apply(return_order_or_none)
            df.sort_values(by='twitter_rank', inplace=True)
        elif sort_by == 'ud_judgements':
            df['judgements'] = df.numLikes + df.numDislikes
            df.sort_values(by='judgements', inplace=True)
        else:
            raise ValueError('Unknown order type {}'.format(sort_by))
    else:
        raise ValueError('Unexpected sort_by object type. It should be either string or dict')


def __add_clarity_columns_inplace(df: pd.DataFrame) -> None:
    df.index.name = 'ID'
    df['category_description'] = df.category.apply(__get_label_description)
    df['category_extended'] = df.category.apply(lambda cat_str: __category_to_readable_name_dict[cat_str])
    df['example_annotated'] = df.apply(lambda r_sr: __inject_word_annotation(r_sr['word'], r_sr['example']), axis=1)


def __filter_out_nan_example_and_meaning(df: pd.DataFrame) -> pd.DataFrame:
    return df[df[['example', 'meaning']].notna().all(axis=1)]


def to_appen_csv_for_human_annotation(in_file: Union[pd.DataFrame, str],
                                      out_file_str: str=None,
                                      count_int: int=None,
                                      exclude: Union[str, pd.DataFrame, List[int]]=None,
                                      random_seed=0,
                                      sort_by: Union[str,Dict[str, int]]=None,
                                      add_columns: List[str]=None) -> pd.DataFrame:
    exclude_index_lst = __get_exclude_indexes(exclude)

    appen_df = __get_dataframe(in_file)
    if exclude_index_lst:
        appen_df = appen_df.drop(exclude_index_lst, axis=0)
    appen_df['category'] = appen_df['Classes']
    
    __add_clarity_columns_inplace(appen_df)
    appen_df['_golden'] = False
    appen_df = appen_df.sample(frac=1, random_state=random_seed)

    if sort_by:
        __order_inplace(appen_df, sort_by)

    if count_int:
        appen_df = appen_df.head(n=count_int)

    columns_lst = ['word', 'meaning', 'example_annotated', 'category_extended', 'category_description', '_golden']
    if add_columns:
        columns_lst += add_columns
    
    appen_df = appen_df[columns_lst]

    __save_if_specified(appen_df, out_file_str)
    return appen_df


def __pick_category_from_pipe_separated(cat_lst_str: str, rand: Random) -> str:
    cat_lst = cat_lst_str.split('|')
    return rand.choice(cat_lst)


def __generate_incorrect_category(row: pd.Series, rand: Random, all_cats_set: Set[str], use_model_cats_as_true: bool=False):
    true_cat_set = set(row['category'].split('|'))
    if use_model_cats_as_true and not pd.isna(row['labels_above_threshold']):
        model_cat_set = set(row['labels_above_threshold'].split('|'))
        true_cat_set = true_cat_set | model_cat_set
    incorrect_cats_set = all_cats_set - true_cat_set
    return rand.choice(list(incorrect_cats_set))


def __inject_word_annotation(word_str: str, example_str: str) -> str:
    word_escaped_str = re.escape(str(word_str))
    return re.sub(word_escaped_str, '<span style="color:rgb(65, 168, 95);"><strong>{}</strong></span>'.format(word_escaped_str), example_str, flags=re.IGNORECASE) # some strings are equal to 'nan'


def to_appen_csv_as_reference(in_file: Union[pd.DataFrame, str],
                              type_or_cat_set: Union[str, Set[str]],
                              out_file_str: str=None,
                              random_seed=0,
                              use_model_cats_as_true=False,
                              add_columns: List[str]=None) -> pd.DataFrame:
    correct_cat_pick_rand = Random(random_seed)
    incorrect_cat_pick_rand = Random(random_seed)

    if isinstance(type_or_cat_set, str):
        all_cats_set = __type_to_category_set_dict[type_or_cat_set]
    else:
        all_cats_set = type_or_cat_set

    appen_df = __get_dataframe(in_file)
    appen_df['category'] = appen_df['liwc']    
    
    appen_df['_golden'] = True
    appen_df['_hidden'] = False
    
    true_appen_df = appen_df.copy(deep=True)
    false_appen_df = appen_df.copy(deep=True)

    true_appen_df['truth_gold'] = [True for _ in range(len(true_appen_df))] # 'truth' is the name of the output, and the '_golden' marks it as the reference
    true_appen_df['category'] = true_appen_df.category.apply(lambda l: __pick_category_from_pipe_separated(l, correct_cat_pick_rand))

    false_appen_df['truth_gold'] = [False for _ in range(len(false_appen_df))]
    false_appen_df['category'] = false_appen_df.apply(lambda r: __generate_incorrect_category(r, incorrect_cat_pick_rand, all_cats_set, use_model_cats_as_true), axis=1)

    appen_df = pd.concat([true_appen_df, false_appen_df]) # duplicate indexes
    __add_clarity_columns_inplace(appen_df)

    columns_lst = ['word', 'meaning', 'example_annotated', 'category_extended', 'category_description', 'truth_gold', '_golden', '_hidden']
    if add_columns:
        columns_lst += add_columns

    appen_df = appen_df[columns_lst]

    appen_df = appen_df.sample(frac=1, random_state=random_seed)

    __save_if_specified(appen_df, out_file_str)
    return appen_df
    

def get_twitter_most_frequent_words_rank() -> Dict[str, int]:
    words_df = pd.read_csv(global_config.resources.twitter.most_frequent_words, names=['word'])
    return {w: i for i, w in enumerate(words_df.word.to_list())}