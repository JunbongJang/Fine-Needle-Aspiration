'''
Author Junbong Jang
7/8/2020

Draws bounding box from the segmented grayscale image.
Learned from TfResearch create_faster_rcnn_tf_record_jj.py
'''

import sys
import numpy as np
import skimage.color
import skimage.filters
import skimage.io
import skimage.viewer
import skimage.measure
import skimage.color
from skimage import img_as_ubyte
import cv2
import numpy as np
import os
from PIL import Image 
import PIL.ImageDraw as ImageDraw
from skimage import morphology
from scipy import ndimage


def get_bounding_box_from_mask(mask, pixel_val):
    xmins = np.array([])
    ymins = np.array([])
    xmaxs = np.array([])
    ymaxs = np.array([])
    
    nonbackground_indices_x = np.any(mask == pixel_val, axis=0)
    nonbackground_indices_y = np.any(mask == pixel_val, axis=1)
    nonzero_x_indices = np.where(nonbackground_indices_x)
    nonzero_y_indices = np.where(nonbackground_indices_y)

    # if the mask contains any object
    if np.asarray(nonzero_x_indices).shape[1] > 0 and np.asarray(nonzero_y_indices).shape[1] > 0:
        
        mask_remapped = (mask == pixel_val).astype(np.uint8)  # get certain label
        mask_regions = skimage.measure.label(img_as_ubyte(mask_remapped), connectivity=2)  # get separate regions
        for region_pixel_val in np.unique(mask_regions):
            if region_pixel_val > 0: # ignore background pixels
                # boolean array for localizing pixels from one region only
                nonzero_x_boolean = np.any(mask_regions == region_pixel_val, axis=0)
                nonzero_y_boolean = np.any(mask_regions == region_pixel_val, axis=1)
                
                # numerical indices of value true in boolean array
                nonzero_x_indices = np.where(nonzero_x_boolean)
                nonzero_y_indices = np.where(nonzero_y_boolean)
                
                # ignore small boxes
                if len(nonzero_x_indices[0]) > 5 and len(nonzero_y_indices[0]) > 5:
                    xmin = float(np.min(nonzero_x_indices))
                    xmax = float(np.max(nonzero_x_indices))
                    ymin = float(np.min(nonzero_y_indices))
                    ymax = float(np.max(nonzero_y_indices))
                    
                    print('bounding box for', region_pixel_val, xmin, xmax, ymin, ymax)

                    xmins = np.append(xmins, xmin)
                    ymins = np.append(ymins, ymin)
                    xmaxs = np.append(xmaxs, xmax)
                    ymaxs = np.append(ymaxs, ymax)
    
    # reshape 1xn row vector into nx1 column vector
    xmins = np.reshape(xmins, (-1, 1)) 
    ymins = np.reshape(ymins, (-1, 1))
    xmaxs = np.reshape(xmaxs, (-1, 1))
    ymaxs = np.reshape(ymaxs, (-1, 1))
    
    # bounding boxes in nx4 matrix
    bounding_boxes = np.concatenate((ymins, xmins, ymaxs, xmaxs), axis=1)
    return bounding_boxes
                       
                       
def draw_bounding_boxes_on_image(image, boxes, color, thickness=8):
    """Draws bounding boxes on image.

    Args:
    boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
           The coordinates are in normalized format between [0, 1].
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.

    Raises:
    ValueError: if boxes is not a [N, 4] array
    """
    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError('Input must be of size [N, 4]')
    for i in range(boxes_shape[0]):
        draw_bounding_box_on_image(image, boxes[i, 0], boxes[i, 1], boxes[i, 2],
                                boxes[i, 3], color, thickness)
    return image
                       
                       
def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color, thickness=4):
    """Adds a bounding box to an image.

    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.

    Args:
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    """
    draw = ImageDraw.Draw(image)
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
                (right, top), (left, top)], width=thickness, fill=color)


def calculate_boxes_overlap_area(box1, box2):
    # https://stackoverflow.com/questions/27152904/calculate-overlapped-area-between-two-rectangles
    ymin1, xmin1, ymax1, xmax1 = box1[0], box1[1], box1[2], box1[3]
    ymin2, xmin2, ymax2, xmax2 = box2[0], box2[1], box2[2], box2[3]
    dx = min(xmax1, xmax2) - max(xmin1, xmin2)
    dy = min(ymax1, ymax2) - max(ymin1, ymin2)
    
    area = 0
    if (dx>=0) and (dy>=0):
        area = dx*dy

    return area


def vUNet_boxes(mask, a_threshold):
    # for vU-Net predictions
    # thresholding
    mask_copy = mask.copy()
    mask_copy[mask_copy >= a_threshold] = 255
    mask_copy[mask_copy < a_threshold] = 0
    mask_copy = clean_mask(mask_copy)
    mask_copy = clean_mask(mask_copy)
    
    # crop edge because of edge effect of vU-Net
    mask_copy = mask_copy[30:, 30:]
    mask_copy = np.pad(mask_copy, ((30,0), (30,0)), 'constant', constant_values=(0))
    
    return get_bounding_box_from_mask(mask_copy, pixel_val = 255)


def clean_mask(input_mask):
    cleaned_mask = input_mask.copy()

    # fill hole
    cleaned_mask = ndimage.morphology.binary_fill_holes(cleaned_mask, structure=np.ones((5, 5))).astype(
        cleaned_mask.dtype)
    cleaned_mask = cleaned_mask * 255

    # Filter using contour area and remove small noise
    # https://stackoverflow.com/questions/60033274/how-to-remove-small-object-in-image-with-python
    cnts = cv2.findContours(cleaned_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, False)
        if area < 2000:
            contour_cmap = (0, 0, 0)
            cv2.drawContours(cleaned_mask, [c], -1, contour_cmap, -1)

    return cleaned_mask


def openImg(path):
    mask = Image.open(path)
    if mask.format != 'PNG':
        raise ValueError('Mask format is not PNG')
    mask = np.asarray(mask.convert('L'))
    return mask.copy()


if __name__ == "__main__":
    # get data path
    base_path = '//research.wpi.edu/leelab/Junbong/'
    ground_truth_mask_root_path = base_path + 'TfResearch/research/object_detection/dataset_tools/assets/masks_test/'
    vUnet_mask_root_path = base_path + 'FNA/vUnet/average_hist/predict_wholeframe/all-patients/all-patients/'
    img_root_path = base_path + 'FNA/assets/all-patients/img/'

    faster_rcnn_boxes = np.load("../../generated/tf1_faster_boxes.npy", allow_pickle=True)

    saved_ground_truth_boxes = {}
    mask_names = [file for file in os.listdir(ground_truth_mask_root_path) if file.endswith(".png")]
    for idx, filename in enumerate(mask_names):
        '''
        get bounding box from vUNet and Faster Rcnn models, and ground truth mask.
        overlay them all on top of the unstained raw image
        '''
        a_threshold = 100
        
        ground_truth_mask_path = os.path.join(ground_truth_mask_root_path, filename)
        #vUnet_prediction_path = os.path.join(vUnet_mask_root_path, filename)
        img_path = os.path.join(img_root_path, filename.replace('predict', ''))
        
        # load images from paths
        img = Image.open(img_path)
        ground_truth_mask_mask = openImg(ground_truth_mask_path)
        #vUnet_prediction_mask = openImg(vUnet_prediction_path)

        # get boxes from image
        ground_truth_boxes = get_bounding_box_from_mask(ground_truth_mask_mask, pixel_val = 76)
        #vUNet_prediction_boxes = vUNet_boxes(vUnet_prediction_mask, a_threshold)
        faster_rcnn_boxes_filename = faster_rcnn_boxes.item()[filename]
        faster_rcnn_boxes_filename[:,0] = faster_rcnn_boxes_filename[:,0] * 1944
        faster_rcnn_boxes_filename[:,2] = faster_rcnn_boxes_filename[:,2] * 1944
        faster_rcnn_boxes_filename[:,1] = faster_rcnn_boxes_filename[:,1] * 2592
        faster_rcnn_boxes_filename[:,3] = faster_rcnn_boxes_filename[:,3] * 2592
        
        # Save image with bounding box
        # save_path = os.path.join(vUnet_mask_root_path, '../boxes_threshold_'+str(a_threshold))
        save_base_path = base_path + 'FNA/evaluation/generated/overlaid_boxes/'
        if os.path.isdir(save_base_path) is False:
            os.mkdir(save_base_path)
        save_path = os.path.join(save_base_path, filename.replace('predict',''))
        print(save_path)

        if ground_truth_boxes.shape[0] > 0:
           boxed_image = draw_bounding_boxes_on_image(img, ground_truth_boxes, color='#9901ff')
           boxed_image = draw_bounding_boxes_on_image(boxed_image, faster_rcnn_boxes_filename, color='#89ff29')
           boxed_image.save(save_path)

        saved_ground_truth_boxes[filename] = ground_truth_boxes

        print()
        print()

    np.save(save_base_path + 'ground_truth_boxes.npy', saved_ground_truth_boxes)
       
