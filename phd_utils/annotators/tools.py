import pandas as pd


def list_to_dict(in_file_path_str: str, out_file_path_str: str) -> pd.DataFrame:
    """
    Converts a list of word-label pairs to liwc dict format
    """
    df = pd.read_csv(in_file_path_str, names=['word', 'label'], index_col='word')
    df.label = df.label.apply(lambda l: l.lower().strip())
    labels_lst = list(df.label.unique())
    labels_to_index_dict = {l: str(i+1) for i, l in enumerate(labels_lst)}
    df.label = df.label.apply(lambda l: labels_to_index_dict[l])
    with open(out_file_path_str, mode='w') as out_file:
        print('%', file=out_file)
        for i, l in labels_to_index_dict.items():
            print('{}\t{}'.format(i, l), file=out_file)
        print('%', file=out_file)
        for w, r in df.groupby(df.index).agg('\t'.join).iterrows():
            print('{}\t{}'.format(w, r['label']), file=out_file)
        return df