import pandas as pd
from typing import Union, Set, List, Dict
from random import Random

from phd_utils import global_config


__root_9_task_description_str = """
The goal is to verify that a word can belong to a given category or categories in any possible context for that word.

Words presented in this task are crawled from internet and social media sources. A considerable percentage of these would be used in informal and non-standard contexts. As such, before judging on the correctness of a category, the meaning and example provided in the example needs to be considered.

A word or a term might belong to multiple categories, but we only present one in each example.

A straight forward example, the word "love" would be considered as a word related to affectioin. A more complex example, the word "circle" would normally used in the context of shapes. But it can also mean inspect in a context like: "the man circled the car to make sure there were no scratches on its paint". As such, the word "circle" can be considered as related to "perception".
"""


__cat_to_description_dict = {
    'affect': 'For words that are related to feelings people experience, including positive emotions as well as negative emotions as anger, sadness, and anxiety. Examples would be: "love", "hate", "abuse"',
    'work': 'For words related to professions, careers, and the workplace. Example would be: "office", "desk", "mentor", "academia"',
    'relativ': 'For words related to motion, space and time including locations, times, location of items, time of incidents, relative location of items with respect to each other, relative times of incidents to each other. Also include tools and items related to motion, location or time. Examples for (time): "before", "april", "ages". Examples for (space): "across", "interior", "floor". Examples for (motion): "fast", "past"',
    'bio': 'For words related to biology including body parts, health, sexuality and ingestion. Examples are: "addiction", "ache", "arm", "leg", "male"',
    'drives': 'For words related to personal drives or Deterrents such as affiliation, achievements, power, reward and risk for personal drives and disassociation, expulsion, failure, lack of power and punishment for personal deterrents. Examples are "achieve", "above", "acclaimed", "ahead", "army", "slave", "shame".',
    'percept': 'For words related to seeing, hearing and feeling. Examples are "ache", "acid", "appearing", "color/colour", "cold"',
    'informal': 'For words that are considered informal. These words can include swearing, common words used throughout social media, blogs and the internet, assent, emojis, non-fluencies, and filter words. These words may have large intersection with other categories. Examples are: "dude" , "cc", "oh", ":D", "mmm", "btw"',
    'social': 'For words related to social status, relations and interactions. Examples are: "her", "marry", "brother", "woman", "meet"',
    'cogproc': 'For words related to any form of cognitive process like insights, findings, experimentation, causality, discrepancies, tentativeness, certainty and differentiation. Examples are: "try", "result" (both noun and verb forms), "coherence", "clear", "clarify", "understand"',
    'pconcern': 'For words related to only: work, leisure, home, money, religon and death. Words related to emotions or biology/biological process are NOT in this category.  Examples for (Work): "busy", "franchise", "engineering". Examples for (leisure): -> "surf", "club", "ball". Examples for (home): "tenant", "microwave". Examples for money: "scholarship", "lease", "pay". Examples for (religon): "afterlife", "God" (death) -> "zombie", "autopsy", "mortal"',

    'negemo': 'For words that can be used in expressing negative emotions including anger, sadness and anxiety. Examples are: "lame", "bad", "unfriendly", "lazy", sigh"',
    'posemo': 'For words that can be used in expressing positive emotions. Examples are: "funny", "safe", "happy", "thx", "calming"',

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
    ('bio', 'Biology / Bioligical Process'),
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
        return pd.read_csv(in_file, index_col=0)
    else:
        return in_file.copy(deep=True)


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

    appen_df.rename(columns={'Classes': 'category'}, inplace=True)
    
    appen_df.index.name = 'ID'
    appen_df['_golden'] = False
    appen_df['category_description'] = appen_df.category.apply(__get_label_description)
    appen_df['category_extended'] = appen_df.category.apply(lambda cat_str: __category_to_readable_name_dict[cat_str])
    appen_df = appen_df.sample(frac=1, random_state=random_seed)

    if sort_by:
        __order_inplace(appen_df, sort_by)

    if count_int:
        appen_df = appen_df.head(n=count_int)

    columns_lst = ['word', 'meaning', 'example', 'category_extended', 'category_description', '_golden']
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


def to_appen_csv_as_reference(in_file: Union[pd.DataFrame, str], type_or_cat_set: Union[str, Set[str]], out_file_str: str=None, random_seed=0, use_model_cats_as_true=False) -> pd.DataFrame:
    correct_cat_pick_rand = Random(random_seed)
    incorrect_cat_pick_rand = Random(random_seed)

    if isinstance(type_or_cat_set, str):
        all_cats_set = __type_to_category_set_dict[type_or_cat_set]
    else:
        all_cats_set = type_or_cat_set

    annotated_df = __get_dataframe(in_file)
    annotated_df.rename(columns={'liwc': 'category'}, inplace=True)

    annotated_df.index.name = 'ID'
    annotated_df['_golden'] = True
    annotated_df['_hidden'] = False
    
    true_appen_df = annotated_df.copy(deep=True)
    false_appen_df = annotated_df.copy(deep=True)

    true_appen_df['truth_golden'] = [True for _ in range(len(true_appen_df))] # 'truth' is the name of the output, and the '_golden' marks it as the reference
    true_appen_df['category'] = true_appen_df.category.apply(lambda l: __pick_category_from_pipe_separated(l, correct_cat_pick_rand))

    false_appen_df['truth_golden'] = [False for _ in range(len(false_appen_df))]
    false_appen_df['category'] = false_appen_df.apply(lambda r: __generate_incorrect_category(r, incorrect_cat_pick_rand, all_cats_set, use_model_cats_as_true), axis=1)

    appen_df = pd.concat([true_appen_df, false_appen_df]) # duplicate indexes
    appen_df['category_description'] = appen_df.category.apply(__get_label_description)
    appen_df['category_extended'] = appen_df.category.apply(lambda cat_str: __category_to_readable_name_dict[cat_str])

    appen_df = appen_df[['word', 'meaning', 'example', 'category_extended', 'category_description', 'truth_golden', '_golden', '_hidden']]

    appen_df = appen_df.sample(frac=1)

    __save_if_specified(appen_df, out_file_str)
    return appen_df
    

def get_twitter_most_frequent_words_rank() -> Dict[str, int]:
    words_df = pd.read_csv(global_config.resources.twitter.most_frequent_words, names=['word'])
    return {w: i for i, w in enumerate(words_df.word.to_list())}