import pandas as pd
from typing import Union, Set, List
from random import Random


__task_description_str = """
The goal is to verify that a word can belong to a given category or categories in any possible context for that word.

Words presented in this task are crawled from internet and social media sources. A considerable percentage of these would be used in informal and non-standard contexts. As such, before judging on the correctness of a category, the meaning and example provided in the example needs to be considered.

A straight forward example, the word "love" would be considered as a word related to affectioin. A more complex example, the word "circle" would normally used in the context of shapes. But it can also mean inspect in a context like: "the man circled the car to make sure there were no scratches on its paint". As such, the word "circle" can be considered as related to "perception".
"""


__cat_to_description_dict = {
    'affect': 'For words that are related to feelings people experience, including positive emotions as well as negative emotions as anger, sadness, and anxiety. Examples would be: "love", "hate", "abuse"',
    'negemo': 'For words that can be used in expressing negative emotions including anger, sadness and anxiety. Examples are: "lame", "bad", "unfriendly", "lazy", sigh"',
    'posemo': 'For words that can be used in expressing positive emotions. Examples are: "funny", "safe", "happy", "thx", "calming"',
    'work': 'For words related to professions, careers, and the workplace. Example would be: "office", "desk", "mentor", "academia"',
    'relativ': 'For words related to motion, space and time including locations, location of items, or relative location of items with respect to each other. Examples for (time): "before", "april", "ages" (space): "across", "interior", "floor" (motion): "fast", "past"',
    'bio': 'For words related to biology including body parts, health, sexuality and ingestion. Examples are: "addiction", "ache", "arm", "leg", "male"',
    'drives': 'For words related to personal drives such as affiliation, achievements, power, reward and risk. Examples are "achieve", "above", "acclaimed", "ahead", "army". Note that two of these examples ("above", "ahead") fit into RELTIV category as well as DRIVES.',
    'percept': 'For words related to seeing, hearing and feeling. Examples are "ache", "acid", "appearing", "color/colour", "cold"',
    'informal': 'For words that are considered informal. These words can include swearing, common words used throughout social media, blogs and the internet, assent, emojis, non-fluencies, and filter words. These words may have large intersection with other categories. Examples are: "dude" , "cc", "oh", ":D", "mmm", "btw"',
    'social': 'For words related to social status, relations and interactions. Examples are: "her", "marry", "brother", "woman", "meet"',
    'pconcern': 'For words related to personal concerns that include only work, leisure, home, money, religon and death. Examples are: (work) -> "busy", "franchise", (leisure) -> "surf", "club" (home) -> "tenant", "microwave" (money) -> "scholarship", "lease" (religon) -> "afterlife", "God" (death) -> "zombie", "autopsy", "mortal"',

    'leisure': 'For words related to leisure including businesses, items or actions. Examples are: "surf", "cinema", "burgers"',
    'money': 'For words related to monetary value including cash, finance, economy. Examples are: "account", "acquire", "debt", "coin"',
    'death': 'For words related to death and life. Examples are: "coffin", "alive", "autopsy", "kill"',
    'cogproc': 'For words related to cognitive process, insights, causality, discrepancies, tentativeness, certainty and differentiation. Examples are: "coherence", "clear", "clarify", "understand"',
    'home': 'For words related to home; either the physical home itself, or actions related to home. Examples are: "carpet", "kitchen", "sweep", "cook"',
    'relig': 'For words related to religion and religious ideas. Examples are: "God", "pray", "amen", "Jesus", "afterlife"'
}


__type_to_category_set_dict = {
    'root_9': set(['affect', 'bio', 'cogproc', 'drives', 'informal', 'pconcern', 'percept', 'relativ', 'social'])
}


def __get_label_description(label_str: str) -> str:
    return __cat_to_description_dict[label_str.lower()]


def __get_dataframe(in_file: Union[pd.DataFrame, str]) -> pd.DataFrame:
    if isinstance(in_file, str):
        return pd.read_csv(in_file, index_col=0)
    else:
        return in_file


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


def to_appen_csv_for_human_annotation(in_file: Union[pd.DataFrame, str], out_file_str: str=None, count_int: int=None, exclude: Union[str, pd.DataFrame, List[int]]=None, random_seed=None) -> pd.DataFrame:
    exclude_index_lst = __get_exclude_indexes(exclude)

    annotated_df = __get_dataframe(in_file)
    if exclude_index_lst:
        annotated_df = annotated_df.drop(exclude_index_lst, axis=0)
    
    appen_df = annotated_df[['word', 'meaning', 'example', 'Classes']]
    appen_df.index.name = 'ID'
    appen_df['_golden'] = 'false'
    appen_df.rename(columns={'Classes': 'category'}, inplace=True)
    appen_df['category_description'] = appen_df.category.apply(__get_label_description)
    appen_df = appen_df.sample(frac=1, random_state=random_seed)

    if count_int:
        appen_df = appen_df.head(n=count_int)

    __save_if_specified(appen_df, out_file_str)
    return appen_df


def __pick_category_from_pipe_separated(cat_lst_str: str, rand: Random) -> str:
    cat_lst = cat_lst_str.split('|')
    return rand.choice(cat_lst)


def __rand_incorrect_category(word_str: str, rand: Random, true_cats_set: Set[str], all_cats_set: Set[str]):
    incorrect_cats_lst = [cat_str for cat_str in all_cats_set if cat_str not in true_cats_set]
    return rand.choice(incorrect_cats_lst)


def to_appen_csv_as_reference(in_file: Union[pd.DataFrame, str], type_or_cat_set: Union[str, Set[str]], out_file_str: str=None, random_seed=0) -> pd.DataFrame:
    correct_cat_pick_rand = Random(random_seed)
    incorrect_cat_pick_rand = Random(random_seed)

    if isinstance(type_or_cat_set, str):
        all_cats_set = __type_to_category_set_dict[type_or_cat_set]
    else:
        all_cats_set = type_or_cat_set

    annotated_df = __get_dataframe(in_file)
    annotated_df.index.name = 'ID'
    annotated_df['_golden'] = 'true'
    annotated_df['_hidden'] = 'true'
    true_appen_df = annotated_df[['word', 'meaning', 'example', 'liwc', '_golden']]
    true_appen_df.rename(columns={'liwc': 'category'}, inplace=True)

    false_appen_df = true_appen_df.copy(deep=True)

    true_appen_df['truth_golden'] = [True for _ in range(len(true_appen_df))] # 'truth' is the name of the output, and the '_golden' marks it as the reference
    true_appen_df['category'] = true_appen_df.category.apply(lambda l: __pick_category_from_pipe_separated(l, correct_cat_pick_rand))
    true_appen_df['category_description'] = true_appen_df.category.apply(__get_label_description)

    false_appen_df['truth_golden'] = [False for _ in range(len(false_appen_df))]
    false_appen_df['category'] = false_appen_df.apply(lambda r: __rand_incorrect_category(r['word'], incorrect_cat_pick_rand, r['category'], all_cats_set), axis=1)
    false_appen_df['category_description'] = false_appen_df.category.apply(__get_label_description)

    appen_df = pd.concat([true_appen_df, false_appen_df]) # duplicate indexes
    appen_df = appen_df.sample(frac=1)

    __save_if_specified(appen_df, out_file_str)
    return appen_df
    
