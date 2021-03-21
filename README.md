## Datasets

### EmoNet

Contains 8 classes of emotions for tweets. Orignial size 80,000 tweets, but a lot have been either deleted or marked private.

```
[('joy', 33670),
 ('sadness', 14324),
 ('fear', 9829),
 ('anger', 5772),
 ('surprise', 5450),
 ('disgust', 5170),
 ('trust', 2982),
 ('anticipation', 2803)]
 ```
 Total number currently: 45,689

Paper published in ACL [here](https://www.aclweb.org/anthology/P17-1067/) with 131 citations (Nov 14, 2020). Code related to publication can be found [here](https://github.com/UBC-NLP/EmoNet)

### ClPysch

Dataset collected from Reddit r/SuicideWatch for users with 4 different tendencies of suicide

### Urban Dicionary with LIWC Annotations

Total category counts in `ud_data_best_def.csv`

| LIWC Category | Count |
| - | - |
| BIO | 14,324 |
| AFFECT | 5,505 |
| INFORMAL | 4,505 |
| PERCEPT | 3,614 |
| RELATIV | 3,212 |
| LEISURE | 2,965 |
| DRIVES | 2,121 |
| WORK | 2,019 |
| COGPROC | 1,734 |
| SOCIAL | 1,718 |
| RELIG | 1,418 |
| DEATH | 954 |
| HOME | 638 |
| MONEY | 632 |
| - | - |
| total | 45,359 |

## Preproessors

### Switch I and You

Changes all first person references to second person references and vice-versa

# Useful References

[Hugging face with custom datasets](https://huggingface.co/transformers/custom_datasets.html)

[Huggingface with multiclass datasets](https://marcotcr.github.io/lime/tutorials/Lime%20-%20multiclass.html)

A tool for web mining and language processing [here](https://www.clips.uantwerpen.be/clips.bak/pages/pattern)
