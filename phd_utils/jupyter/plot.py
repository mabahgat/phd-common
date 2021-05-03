import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

def plot_precision_recall_threshold_curve(precision_lst, recall_lst, title_str=None, thresholds_lst=None):
    """
    Plots precision and recall against list of thresholds (by default from 0 to 1 with 100 steps)
    """
    _, ax = plt.subplots()

    def set_minor_ticks(start, end):
        draw_range = np.arange(start, end + 0.1, step=0.05)
        ax.set_xticks(draw_range, minor=True)
        ax.set_yticks(draw_range, minor=True)

    if thresholds_lst is None:
        thresholds_lst = np.linspace(0, 1, len(precision_lst))
        set_minor_ticks(0, 1)
    else:
        start = np.amin(thresholds_lst)
        end = np.amax(thresholds_lst)
        set_minor_ticks(start, end)

    ax.plot(thresholds_lst, precision_lst, label='Precision')
    ax.plot(thresholds_lst, recall_lst, label='Recall')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Precision/Recall')
    ax.grid(which='major')
    ax.grid(which='minor', linestyle='--')
    if title_str:
        ax.set_title(title_str)
    ax.legend()
    plt.show()


def plot_confusion_matrix(matrix, class_names):
    """
    Plots confusion matrix
    """
    sn.heatmap(matrix, xticklabels=class_names, yticklabels=class_names)


def plot_confusion_matrix_for_model(model, dataset):
    """
    Plots the confusion matrix for a model on the test set for a dataset
    """
    normalized_confusion_matrix = model.confusion_matrix(dataset, 'pred')
    plot_confusion_matrix(normalized_confusion_matrix, dataset.class_names())


def add_fscore_to_plot(ax, start=0.2, end=0.8, count=4):
    """
    Adds f-score ranges to a plot
    """
    f_scores = np.linspace(start, end, count)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        ax.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        ax.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))


def plot_precision_recall_curve(precision_lst, recall_lst, title_str=None, f_curve_counts=4):
    """
    Plot precision recall curves
    """
    _, ax = plt.subplots()
    add_fscore_to_plot(ax, count=f_curve_counts)
    #ax.step(recall_lst, precision_lst, where='post')
    ax.plot(recall_lst, precision_lst)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    if title_str:
        ax.set_title(title_str)
    plt.show()