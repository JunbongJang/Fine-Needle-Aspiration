"""
Author Junbong Jang
Date: 7/29/2020

Bootstrap sample the test set images
Load ground truth and faster rcnn boxes and count them.
After sampling distribution is obtained, to calculate confidence intervals

Refered to https://machinelearningmastery.com/a-gentle-introduction-to-the-bootstrap-method/
https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/
"""

import pandas as pd
from sklearn.utils import resample

from analysis.explore_data import get_files_in_folder, get_images_by_subject, print_images_by_subject_statistics
from analysis.bootstrapping_visualization import *

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


if __name__ == "__main__":
    bootstrap_repetition_num = 10000
    ground_truth_min_follicular = 15
    predicted_min_follicular = 15
    root_test_img_path = '//research.wpi.edu/leelab/Junbong/TfResearch/research/object_detection/dataset_tools/assets/stained_images_test/'  # "C:/Users/Junbong/Desktop/FNA Data/all-patients/images_test"

    test_image_names = get_files_in_folder(root_test_img_path)
    np_ground_truth_boxes = np.load("../../generated/ground_truth_boxes.npy", allow_pickle=True)
    np_faster_rcnn_boxes = np.load("../../generated/faster_640_boxes.npy", allow_pickle=True)

    print('ground truth: ', count_boxes(test_image_names, np_ground_truth_boxes))
    print('predictions: ', count_boxes(test_image_names, np_faster_rcnn_boxes))

    # Data Organizing
    test_images_by_subject = get_images_by_subject(test_image_names)
    print_images_by_subject_statistics(test_images_by_subject)

    # Follicular Cluster Counting after bootstrapping!
    box_counts = bootstrap_box_counts(test_image_names, test_images_by_subject, np_ground_truth_boxes, np_faster_rcnn_boxes, bootstrap_repetition_num)
    box_counts_df = pd.DataFrame(box_counts)
    # box_counts_df.to_excel('../../generated/bootstrapped_box_counts.xlsx', index=False)
    print('boxes shape: ', box_counts.shape)

    # ----------- Analysis Starts -------------------

    # Data Exploration
    plot_histogram(box_counts_df, test_image_names)
    plot_scatter(box_counts_df, ground_truth_min_follicular)

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
