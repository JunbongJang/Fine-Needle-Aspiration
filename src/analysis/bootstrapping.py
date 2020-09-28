"""
Author Junbong Jang
Date: 7/29/2020

Bootstrap sample the test set images
Load ground truth and faster rcnn boxes and count them.
After sampling distribution is obtained, to calculate confidence intervals

Refered to https://machinelearningmastery.com/a-gentle-introduction-to-the-bootstrap-method/
https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/
"""

from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from sklearn.utils import resample
import math

from sklearn.metrics import roc_auc_score, auc, roc_curve, precision_recall_curve, f1_score
from analysis.explore_data import get_files_in_folder, get_images_by_subject, print_images_by_subject_statistics


def bootstrap_box_counts(image_names, images_by_subject, np_ground_truth_boxes, np_faster_rcnn_boxes, bootstrap_repetition_num):
    '''
    Count the total number of boxes per object detected image
    We assume each box represents one follicular cluster detection.
    Secretion/Artifact boxes should have been processed and removed beforehand.

    Hierachical bootstrapping if images_by_subject is not None

    :param image_names:
    :param images_by_subject:
    :param np_ground_truth_boxes:
    :param np_faster_rcnn_boxes:
    :param bootstrap_repetition_num:
    :return:
    '''
    testset_sample_size = len(image_names)
    testset_indices = np.arange(0, testset_sample_size, 1)

    box_counts = np.zeros(shape=(bootstrap_repetition_num, 2))
    for bootstrap_repetition_index in range(bootstrap_repetition_num):

        # ------- bootstrap subjects ------
        if images_by_subject != None:
            bootstrap_sampled_subjects = resample(list(images_by_subject.keys()), replace=True, n_samples=len(images_by_subject.keys()),
                                                 random_state=bootstrap_repetition_index)
            # only get images from sampled subjects
            image_names = []
            for bootstrap_sampled_subject in bootstrap_sampled_subjects:
                image_names = image_names + images_by_subject[bootstrap_sampled_subject]

        # ------- bootstrap images ---------
        bootstrap_sampled_image_names = resample(image_names, replace=True, n_samples=testset_sample_size,
                                             random_state=bootstrap_repetition_index)

        ground_truth_boxes_total = 0
        faster_rcnn_boxes_total = 0
        for chosen_image in bootstrap_sampled_image_names:
            # count ground truth boxes
            ground_truth_boxes = np_ground_truth_boxes.item()[chosen_image]
            ground_truth_boxes_total = ground_truth_boxes_total + len(ground_truth_boxes)

            # count faster rcnn boxes
            faster_rcnn_boxes = np_faster_rcnn_boxes.item()[chosen_image]
            faster_rcnn_boxes_total = faster_rcnn_boxes_total + len(faster_rcnn_boxes)

        box_counts[bootstrap_repetition_index, :] = ground_truth_boxes_total, faster_rcnn_boxes_total
    return box_counts


def count_boxes(image_names, input_boxes):
    boxes_total = 0
    for image_name in image_names:
        boxes = input_boxes.item()[image_name]
        boxes_total = boxes_total + len(boxes)
    return boxes_total


# -------- Analysis ----------
def stats_at_threshold(box_counts_df, ground_truth_min_follicular, predicted_min_follicular, DEBUG):
    true_positive = box_counts_df.loc[
        (box_counts_df[0] >= ground_truth_min_follicular) & (box_counts_df[1] >= predicted_min_follicular)]
    true_negative = box_counts_df.loc[
        (box_counts_df[0] < ground_truth_min_follicular) & (box_counts_df[1] < predicted_min_follicular)]
    false_positive = box_counts_df.loc[
        (box_counts_df[0] < ground_truth_min_follicular) & (box_counts_df[1] >= predicted_min_follicular)]
    false_negative = box_counts_df.loc[
        (box_counts_df[0] >= ground_truth_min_follicular) & (box_counts_df[1] < predicted_min_follicular)]

    true_positive = len(true_positive)
    true_negative = len(true_negative)
    false_positive = len(false_positive)
    false_negative = len(false_negative)

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    F1 = 2 * (precision * recall) / (precision + recall)

    if DEBUG:
        print('pred_min_follicular:', predicted_min_follicular)
        print('true_positives', true_positive, end='  ')
        print('true_negative', true_negative, end='  ')
        print('false_positives', false_positive, end='  ')
        print('false_negative', false_negative)

        print('precision', precision, end='  ')
        print('recall', recall, end='  ')
        print('F1', F1)

    return precision, recall, F1


def distribution_mean_and_error(box_counts_series, sample_size):
    mean = box_counts_series.mean()
    std = box_counts_series.std()
    standard_err = std / math.sqrt(sample_size)

    mean = round(mean,3)
    std = round(std,3)

    print('mean', mean)
    print('sample standard deviation', std)
    print('standard error', standard_err)
    print(len(box_counts_series.loc[(box_counts_series >= mean-std) & (box_counts_series <= mean+std)]))
    print(len(box_counts_series.loc[(box_counts_series >= mean-standard_err) & (box_counts_series <= mean+standard_err)]))
    print()
    return mean, std


# -------- Visualization ------------

def draw_histogram(box_counts_df):
    '''

    :param box_counts_df:
    :return:
    '''
    gt_mean, gt_std = distribution_mean_and_error(box_counts_df[0], sample_size=len(test_image_names))
    pred_mean, pred_std = distribution_mean_and_error(box_counts_df[1], sample_size=len(test_image_names))

    bins = np.histogram(np.hstack((box_counts_df[0], box_counts_df[1])), bins=40)[1]  # get the bin edges
    fig, ax = plt.subplots()
    plt.hist(box_counts_df[0], bins=bins, rwidth=0.9, alpha=0.5, label='Ground Truth (GT)')
    plt.hist(box_counts_df[1], bins=bins, rwidth=0.9, alpha=0.5, label='Prediction (Pred)')

    ax.text(0.68, 0.74, f'GT Mean={gt_mean}\nGT SD={gt_std}\nPred Mean={pred_mean}\nPred SD={pred_std}', color='black',
            horizontalalignment='left',
            verticalalignment='center',
            transform=ax.transAxes)

    plt.title('Distribution of Samples of Follicular Clusters', fontsize='x-large')
    plt.xlabel('Number of Follicular Clusters', fontsize='large')
    plt.ylabel(f'Frequency', fontsize='large')
    plt.legend(loc='upper right')

    # ---- for two separate histograms
    # https://stackoverflow.com/questions/23617129/matplotlib-how-to-make-two-histograms-have-the-same-bin-width

    # hist = box_counts_df.hist(bins=10, rwidth=0.9)

    # fig = plt.gcf()
    # fig.ylim((0, 250))
    # fig.suptitle('Distribution of Samples of Follicular Clusters', fontsize='x-large')
    #
    # hist[0,0].set_xlabel('Number of Follicular Clusters', fontsize='large')
    # hist[0,0].set_ylabel(f'Frequency', fontsize='large')
    # hist[0, 0].set_title('Ground Truth')
    # hist[0,0].grid(False)
    #
    # hist[0,1].set_xlabel('Number of Follicular Clusters', fontsize='large')
    # hist[0,1].set_ylabel(f'Frequency', fontsize='large')
    # hist[0,1].set_title('Prediction')
    # hist[0,1].grid(False)
    #
    # fig.tight_layout()
    # fig.subplots_adjust(top=0.88)

    plt.show()


def scatter_plot(box_counts_df, ground_truth_min_follicular):
    fig = plt.figure()  # pylab.figure()
    ax = fig.add_subplot(111)

    plt.scatter(box_counts_df[0], box_counts_df[1], alpha=0.1, s=3)
    plt.axvline(x=ground_truth_min_follicular, c='r')
    plt.axhline(y=ground_truth_min_follicular, c='r')

    # plt.axhline(y=5, c='r')
    # rect1 = matplotlib.patches.Rectangle((0, 5), 400, 10, color='yellow', alpha=0.3)
    # ax.add_patch(rect1)

    plt.title('Ground Truth Vs. Predicted', fontsize='x-large')
    plt.xlabel('Ground Truth Number of Follicular Clusters', fontsize='large')
    plt.ylabel('Predicted Number of Follicular Clusters', fontsize='large')

    plt.xlim(left=0, right=120)
    plt.ylim(bottom=0, top=120)
    plt.grid(True)
    plt.show()


def plot_roc_curve(y_true, y_pred):
    '''
    Refered to
    # https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python
    :param y_true:
    :param y_pred:
    :return:
    '''
    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(y_true))]

    # calculate scores
    ns_auc = roc_auc_score(y_true, ns_probs)
    lr_auc = roc_auc_score(y_true, y_pred)

    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('ROC AUC=%.3f' % (lr_auc))

    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_true, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_true, y_pred)

    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label=f'Follicular Cluster Detection\nROC AUC=%.3f' % (lr_auc))

    plt.title('Slide Pass/Fail ROC curve')
    plt.xlabel('False Positive Rate', fontsize='large')
    plt.ylabel('True Positive Rate', fontsize='large')

    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid()
    plt.show()


def plot_precision_recall_curve(y_true, y_pred):
    lr_precision, lr_recall, _ = precision_recall_curve(y_true, y_pred)
    lr_f1, lr_auc = f1_score(y_true, y_pred), auc(lr_recall, lr_precision)

    no_skill = len(y_true[y_true == 1]) / len(y_true)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(lr_recall, lr_precision, marker='.',
             label=f'Follicular Cluster Detection\nf1={round(lr_f1,3)} auc={round(lr_auc,3)}')

    plt.title('Slide Pass/Fail Precision-Recall curve')
    plt.xlabel('Recall', fontsize='large')
    plt.ylabel('Precision', fontsize='large')

    plt.xlim(left=0)
    plt.ylim(bottom=no_skill)
    plt.legend()
    plt.grid()
    plt.show()


def plot_precision_recall_curve_at_thresholds(y_true, precision_list, recall_list):
    no_skill = len(y_true[y_true == 1]) / len(y_true)
    # include both endpoints
    precision_list = precision_list + [1,no_skill]
    recall_list = recall_list + [0,1]

    # sort them
    recall_sort_index = np.argsort(recall_list)
    precision_list = [precision_list[i] for i in recall_sort_index]
    recall_list = [recall_list[i] for i in recall_sort_index]

    lr_auc = auc(recall_list, precision_list)

    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill', lw=2)
    plt.plot(recall_list, precision_list, marker='.',
             label=f'Follicular Cluster Detection\nAUC={round(lr_auc, 3)}', lw=2)

    plt.title('Slide Pass/Fail Precision-Recall curve', fontsize='x-large')
    plt.xlabel('Recall', fontsize='large')
    plt.ylabel('Precision', fontsize='large')

    plt.xlim(left=0, right=1.02)
    plt.ylim(bottom=0.75)
    plt.legend()
    plt.grid()
    plt.show()


def plot_performance_at_thresholds(predicted_min_follicular_list, precision_list, recall_list, f1_list):
    plt.plot(predicted_min_follicular_list, precision_list, marker='.', label='Precision', lw=2)
    plt.plot(predicted_min_follicular_list, recall_list, marker='.', label='Recall', lw=2)
    plt.plot(predicted_min_follicular_list, f1_list, marker='.', label='F1', lw=2)

    plt.title('Performance at different Thresholds', fontsize='x-large')
    plt.xlabel('Minimum Predicted Follicular Clusters to Pass', fontsize='large')
    plt.ylabel('Performance', fontsize='large')

    # plt.ylim(bottom=0.8, top=1)
    plt.xlim(left=0, right=30)
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    bootstrap_repetition_num = 10000
    ground_truth_min_follicular = 15
    predicted_min_follicular = 15
    root_test_img_path = '//research.wpi.edu/leelab/Junbong/TfResearch/research/object_detection/dataset_tools/assets/stained_images_test/'  # "C:/Users/Junbong/Desktop/FNA Data/all-patients/images_test"

    test_image_names = get_files_in_folder(root_test_img_path)
    np_ground_truth_boxes = np.load("../../generated/ground_truth_boxes.npy", allow_pickle=True)
    np_faster_rcnn_boxes = np.load("../../generated/stained_no_identity_loss_faster_640_boxes.npy", allow_pickle=True)

    print('ground truth: ', count_boxes(test_image_names, np_ground_truth_boxes))
    print('predictions: ', count_boxes(test_image_names, np_faster_rcnn_boxes))

    # Data Organizing
    test_images_by_subject = get_images_by_subject(test_image_names)
    print_images_by_subject_statistics(test_images_by_subject)

    # Follicular Cluster Counting
    box_counts = bootstrap_box_counts(test_image_names, test_images_by_subject, np_ground_truth_boxes, np_faster_rcnn_boxes, bootstrap_repetition_num)
    box_counts_df = pd.DataFrame(box_counts)
    # box_counts_df.to_excel('../../generated/bootstrapped_box_counts.xlsx', index=False)
    print('boxes shape: ', box_counts.shape)

    # ----------- Analysis Starts -------------------

    # Data Exploration
    draw_histogram(box_counts_df)
    scatter_plot(box_counts_df, ground_truth_min_follicular)

    # Roc curve tutorials
    y_true = box_counts_df[0] >= ground_truth_min_follicular
    # y_pred = box_counts_df[1] >= predicted_min_follicular
    # plot_roc_curve(y_true, y_pred)
    # plot_precision_recall_curve(y_true, y_pred)

    # -------- Varying Predicted Min Follicular Thresholds ------------
    precision_list = []
    recall_list = []
    f1_list = []
    predicted_min_follicular_list = []
    for predicted_min_follicular in range(0,31):
        a_precision, a_recall, a_f1 = stats_at_threshold(box_counts_df, ground_truth_min_follicular, predicted_min_follicular, DEBUG=True)
        predicted_min_follicular_list.append(predicted_min_follicular)
        precision_list.append(a_precision)
        recall_list.append(a_recall)
        f1_list.append(a_f1)

    plot_precision_recall_curve_at_thresholds(y_true, precision_list, recall_list)
    plot_performance_at_thresholds(predicted_min_follicular_list, precision_list, recall_list, f1_list)
