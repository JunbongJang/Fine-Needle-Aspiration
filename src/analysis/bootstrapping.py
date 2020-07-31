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
import numpy as np
import pandas as pd
from sklearn.utils import resample
import math

from analysis.explore_data import get_files_in_folder, get_images_by_subject


def bootstrap_box_counts(image_names, images_by_subject, np_ground_truth_boxes, np_faster_rcnn_boxes, bootstrap_repetition_num):
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
def f1_score(box_counts_df, min_follicular_required):
    true_positive = box_counts_df.loc[
        (box_counts_df[0] >= min_follicular_required) & (box_counts_df[1] >= min_follicular_required)]
    true_negative = box_counts_df.loc[
        (box_counts_df[0] < min_follicular_required) & (box_counts_df[1] < min_follicular_required)]
    false_positive = box_counts_df.loc[
        (box_counts_df[0] < min_follicular_required) & (box_counts_df[1] >= min_follicular_required)]
    false_negative = box_counts_df.loc[
        (box_counts_df[0] >= min_follicular_required) & (box_counts_df[1] < min_follicular_required)]

    true_positive = len(true_positive)
    true_negative = len(true_negative)
    false_positive = len(false_positive)
    false_negative = len(false_negative)

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    F1 = 2 * (precision * recall) / (precision + recall)

    print('true_positives', true_positive)
    print('true_negative', true_negative)
    print('false_positives', false_positive)
    print('false_negative', false_negative)

    print('precision', precision)
    print('recall', recall)
    print('F1', F1)


def distribution_mean_and_error(box_counts_series, sample_size):
    mean = box_counts_series.mean()
    std = box_counts_series.std()
    standard_err = std / math.sqrt(sample_size)
    print('mean', mean)
    print('sample standard deviation', std)
    print('standard error', standard_err)

    print(len(box_counts_series.loc[(box_counts_series >= mean-std) & (box_counts_series <= mean+std)]))
    print(len(box_counts_series.loc[(box_counts_series >= mean-standard_err) & (box_counts_series <= mean+standard_err)]))
    print()


# -------- Visualization ------------

def draw_histogram(box_counts_df):
    # https://stackoverflow.com/questions/23617129/matplotlib-how-to-make-two-histograms-have-the-same-bin-width
    bins = np.histogram(np.hstack((box_counts_df[0], box_counts_df[1])), bins=40)[1]  # get the bin edges

    plt.hist(box_counts_df[0], bins=bins, rwidth=0.9, alpha=0.5, label='Ground Truth')
    plt.hist(box_counts_df[1], bins=bins, rwidth=0.9, alpha=0.5, label='Prediction')
    plt.title('Distribution of Samples of Follicular Clusters', fontsize='x-large')
    plt.xlabel('Number of Follicular Clusters', fontsize='large')
    plt.ylabel(f'Frequency', fontsize='large')
    plt.legend(loc='upper right')

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


def scatter_plot(box_counts_df, min_follicular_required):
    plt.scatter(box_counts_df[0], box_counts_df[1], alpha=0.1)
    plt.axvline(x=min_follicular_required, c='r')
    plt.axhline(y=min_follicular_required, c='r')
    plt.title('Ground Truth Vs. Predicted', fontsize='x-large')
    plt.xlabel('Ground Truth Number of Follicular Clusters', fontsize='large')
    plt.ylabel('Predicted Number of Follicular Clusters', fontsize='large')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    bootstrap_repetition_num = 10000
    min_follicular_required = 15
    test_image_names = get_files_in_folder("C:/Users/Junbong/Desktop/FNA Data/all-patients/images_test")
    np_ground_truth_boxes = np.load("../../generated/ground_truth_boxes.npy", allow_pickle=True)
    np_faster_rcnn_boxes = np.load("../../generated/faster_rcnn_boxes.npy", allow_pickle=True)

    print('ground truth: ', count_boxes(test_image_names, np_ground_truth_boxes))
    print('predictions: ', count_boxes(test_image_names, np_faster_rcnn_boxes))

    test_images_by_subject = get_images_by_subject(test_image_names)
    box_counts = bootstrap_box_counts(test_image_names, test_images_by_subject, np_ground_truth_boxes, np_faster_rcnn_boxes, bootstrap_repetition_num)
    # box_counts_df.to_excel('../../generated/bootstrapped_box_counts.xlsx', index=False)
    print('boxes shape: ', box_counts.shape)


    box_counts_df = pd.DataFrame(box_counts)
    draw_histogram(box_counts_df)
    scatter_plot(box_counts_df, min_follicular_required)

    distribution_mean_and_error(box_counts_df[0], sample_size=len(test_image_names))
    distribution_mean_and_error(box_counts_df[1], sample_size=len(test_image_names))
    f1_score(box_counts_df, min_follicular_required)
