from typing import List
import pandas as pd
import re
import numpy as np

from phd_utils.models_v2 import ModelBase, BertSequenceClassification
from phd_utils.datasets_v2 import DatasetBase, UrbanDictWithLiwc
from phd_utils.providers import Provider, TensorProvider
from tqdm import tqdm


def __get_model_dataset_and_provider(model_name_str: str) -> (ModelBase, DatasetBase, Provider): #TODO better parsing
    model = None
    dataset = None
    if re.findall(r'^ud_liwc', model_name_str):
        labels_type_str = None
        if re.findall(r'^ud_liwc_root-9', model_name_str):
            labels_type_str = 'liwc_root_9'
        elif re.findall(r'^ud_liwc_bio-5', model_name_str):
            labels_type_str = 'liwc_bio_5'
        elif re.findall(r'^ud_liwc_pconcern-6', model_name_str):
            labels_type_str = 'liwc_pconcern_6'
        elif re.findall(r'^ud_liwc_social-5', model_name_str):
            labels_type_str = 'liwc_social_5'
        elif re.findall(r'^ud_liwc_social-3', model_name_str):
            labels_type_str = 'liwc_social_3'
        elif re.findall(r'ud_liwc_affect-2', model_name_str):
            labels_type_str = 'liwc_affect_2'
        elif re.findall(r'ud_liwc_affect-3', model_name_str):
            labels_type_str = 'liwc_affect_3'
        else:
            raise ValueError('Could not parse dataset type from string {}'.format(model_name_str))
        dataset = UrbanDictWithLiwc(0, config_dict={'labels': labels_type_str})
    if not dataset:
        raise ValueError('Could not parse dataset from input string {}'.format(model_name_str))

    if re.findall(r'[^a-zA-Z]bert[^a-zA-Z]', model_name_str):
        model = BertSequenceClassification(class_count=dataset.class_count(), label_str=model_name_str)
        provider = TensorProvider(batch_size=20, tokenizer=model.tokenizer())
    if not model:
        raise ValueError('Could not parse model from input string {}'.format(model_name_str))

    return model, dataset, provider


def run_model_on_ud_dataframe(data_df: pd.DataFrame,
                              model_name_str: str,
                              include_threshold=0.0) -> pd.DataFrame:
    data_df['tagList'] = data_df.tagList.apply(UrbanDictWithLiwc._tags_to_list)
    data_df['tagList'] = data_df.tagList.apply(lambda l: ' '.join(l))
    data_df['input_text'] = data_df['tagList'].astype(str) + ' ' + data_df['meaning'].astype(str) + ' ' + data_df['example'].astype(str)
    return run_model_on_dataframe(data_df, model_name_str, input_columns_str='input_text')


def run_model_on_dataframe(data_df: pd.DataFrame,
                           model_name_str: str,
                           input_columns_str: str,
                           include_threshold=0.0) -> pd.DataFrame:
    labels_lst, max_probs_lst, probs_lst = run_model_on_list(data_df[input_columns_str].to_list(), model_name_str)
    data_df['model_label'] = labels_lst
    data_df['model_max_label_probability'] = max_probs_lst
    data_df['model_probablities'] =  probs_lst
    return data_df


def __get_label(prob_lst, class_names_lst: List[str], threshold=0.0):
    class_index = np.argmax(prob_lst)
    if prob_lst[class_index] < threshold:
        return ''
    return class_names_lst[class_index]


def run_model_on_list(text_lst: List[str], model_name_str: str, include_threshold=0.0) -> pd.DataFrame:
    model, dataset, provider = __get_model_dataset_and_provider(model_name_str)
    probs_lst = []
    labels_lst = []
    max_probs_lst = []
    for sent_str in tqdm(text_lst, desc='Running model'):
        sent_probs_lst = model.predict(provider.apply(sent_str))
        sent_label_str = __get_label(sent_probs_lst, dataset.class_names(), threshold=include_threshold)
        sent_max_prob = sent_probs_lst[np.argmax(sent_probs_lst)]

        probs_lst.append(sent_probs_lst)
        labels_lst.append(sent_label_str)
        max_probs_lst.append(sent_max_prob)
    
    return labels_lst, max_probs_lst, probs_lst


if __name__ == "__main__":
    model_name = 'ud_liwc_root-9_strict_top10_mergeWithTags_bert-base-uncased_batch20_epochs5'
    __get_model_dataset_and_provider(model_name)