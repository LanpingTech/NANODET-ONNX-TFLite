import os

import math
from scipy.special import softmax

import cv2
import numpy as np

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)

mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3) / 255
std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3) / 255
strides = [8, 16, 32, 64]

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def image_preprocess(img, input_size, swap=(2, 0, 1)):
    resized_img = cv2.resize(img, (input_size[1], input_size[0]))
    normalized_img = resized_img.astype(np.float32) / 255
    normalized_img = (normalized_img - mean) / std
    normalized_img = normalized_img.transpose(swap)
    normalized_img = np.ascontiguousarray(normalized_img, dtype=np.float32)

    return normalized_img

def integral(reg_max=16):
    project = np.linspace(0, reg_max, reg_max + 1)
    def func(x):
        shape = x.shape
        x = softmax(x.reshape(*shape[:-1], 4, reg_max + 1), axis=-1)
        x = np.dot(x, project).reshape(*shape[:-1], 4)
        return x

    return func

def distance2bbox(points, distance, max_shape=None):
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    if max_shape is not None:
        x1 = np.minimum(np.maximum(x1, 0.0), max_shape[1])
        y1 = np.minimum(np.maximum(y1, 0.0), max_shape[0])
        x2 = np.minimum(np.maximum(x2, 0.0), max_shape[1])
        y2 = np.minimum(np.maximum(y2, 0.0), max_shape[0])

    return np.stack([x1, y1, x2, y2], axis=-1)

def bbox2distance(points, bbox, max_dis=None, eps=0.1):
    left = points[:, 0] - bbox[:, 0]
    top = points[:, 1] - bbox[:, 1]
    right = bbox[:, 2] - points[:, 0]
    bottom = bbox[:, 3] - points[:, 1]
    if max_dis is not None:
        left = np.minimum(np.maximum(left, 0.0), max_dis - eps)
        top = np.minimum(np.maximum(top, 0.0), max_dis - eps)
        right = np.minimum(np.maximum(right, 0.0), max_dis - eps)
        bottom = np.minimum(np.maximum(bottom, 0.0), max_dis - eps)
    return np.stack([left, top, right, bottom], axis=-1)

def get_single_level_center_priors(batch_size, featmap_size, stride):
    h, w = featmap_size
    x_range = (np.arange(w)) * stride
    y_range = (np.arange(h)) * stride
    x, y = np.meshgrid(y_range, x_range)
    y = y.flatten()
    x = x.flatten()
    strides = np.full((x.shape[0],), stride)
    proiors = np.stack([x, y, strides, strides], axis=-1)
    return proiors[np.newaxis, ...].repeat(batch_size, axis=0)


def get_bboxes(cls_preds, reg_preds, input_shape, reg_max):

    distribution_project = integral(reg_max)
    b = cls_preds.shape[0]
    input_height, input_width = input_shape

    featmap_sizes = [
        (math.ceil(input_height / stride), math.ceil(input_width) / stride)
        for stride in strides
    ]
    # get grid cells of one image
    mlvl_center_priors = [get_single_level_center_priors(
            b,
            featmap_sizes[i],
            stride
        )
        for i, stride in enumerate(strides)
    ]

    center_priors = np.concatenate(mlvl_center_priors, axis=1)
    dis_preds = distribution_project(reg_preds) * center_priors[..., 2, None]
    bboxes = distance2bbox(center_priors[..., :2], dis_preds, max_shape=input_shape)
    scores = cls_preds
    result_list = []
    for i in range(b):
        score, bbox = scores[i], bboxes[i]
        padding = np.zeros((score.shape[0], 1))
        score = np.concatenate([score, padding], axis=1)
        results = multiclass_nms_class(bbox, score, score_thr=0.05, nms_thr=0.6)
        result_list.append(results)
    return result_list

def nms(boxes, scores, nms_thr):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep

def multiclass_nms_class(boxes, scores, nms_thr, score_thr):
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
        )
    return dets

def warp_boxes(boxes, M, width, height):
    n = len(boxes)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            n * 4, 2
        )  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        return xy.astype(np.float32)
    else:
        return boxes


def post_process(preds, num_classes, reg_max, input_size):
    """Prediction results post processing. Decode bboxes and rescale
    to original image size.
    Args:
        preds (Tensor): Prediction output.
        meta (dict): Meta info.
    """
    # print(preds.shape)
    cls_scores, bbox_preds = np.split(preds, [num_classes], axis=-1)

    result_list = get_bboxes(cls_scores, bbox_preds, input_size, reg_max)
    det_results = {}
    warp_matrix = np.eye(3)

    img_height = input_size[0]

    img_width = input_size[1]

    # for result, img_width, img_height, img_id, warp_matrix in zip(
    #         result_list, img_widths, img_heights, img_ids, warp_matrixes
    # ):
    det_result = {}
    det_bboxes, det_labels = result_list[0][:, :5], result_list[0][:, 5] # TODO results is a batch
    # det_bboxes = det_bboxes.detach().cpu().numpy()
    det_bboxes[:, :4] = warp_boxes(
        det_bboxes[:, :4], np.linalg.inv(warp_matrix), img_width, img_height
    )
    # classes = det_labels.detach().cpu().numpy()
    for i in range(num_classes):
        inds = det_labels == i
        det_result[i] = np.concatenate(
            [
                det_bboxes[inds, :4].astype(np.float32),
                det_bboxes[inds, 4:5].astype(np.float32),
            ],
            axis=1,
        ).tolist()
    det_results[0] = det_result
    return det_results


def overlay_bbox_cv(img, dets, class_names, score_thresh):
    all_box = []
    src_height, src_width = img.shape[:2]
    h_ratio = src_height / 320
    w_ratio = src_width / 320

    for label in dets:
        for bbox in dets[label]:
            score = bbox[-1]
            if score > score_thresh:
                x0, y0, x1, y1 = [int(i) for i in bbox[:4]]
                x0 = int(x0 * w_ratio)
                y0 = int(y0 * h_ratio)
                x1 = int(x1 * w_ratio)
                y1 = int(y1 * h_ratio)

                all_box.append([label, x0, y0, x1, y1, score])

    all_box.sort(key=lambda v: v[5])
    for box in all_box:
        label, x0, y0, x1, y1, score = box
        # color = self.cmap(i)[:3]
        color = (_COLORS[label] * 255).astype(np.uint8).tolist()
        text = "{}:{:.1f}%".format(class_names[label], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[label]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.5, 2)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        cv2.rectangle(img,(x0, y0 - txt_size[1] - 1), (x0 + txt_size[0] + txt_size[1], y0 - 1), color, -1,)
        cv2.putText(img, text, (x0, y0 - 1), font, 0.5, txt_color, thickness=1)
    return img


def show_result(img, dets, class_names, score_thres=0.3, show=True, save_path=None):
    result = overlay_bbox_cv(img, dets, class_names, score_thresh=score_thres)
    if show:
        cv2.imshow("det", result)
        cv2.waitKey(0)
    return result

def visualize(dets, origin_img, class_names, score_thres, wait=0):
        # time1 = time.time()
        result_img = show_result(
            origin_img, dets, class_names, score_thres=score_thres, show=False
        )
        # print("viz time: {:.3f}s".format(time.time() - time1))
        return result_img

def format_input_tensor(tensor, input_details, idx):
    details = input_details[idx]
    dtype = details['dtype']
    if dtype == np.uint8 or dtype == np.int8:
        quant_params = details['quantization_parameters']
        input_tensor = tensor / quant_params['scales'] + quant_params['zero_points']
        if dtype == np.int8:
            input_tensor = input_tensor.clip(-128, 127)
        else:
            input_tensor = input_tensor.clip(0, 255)
        return input_tensor.astype(dtype)
    else:
        return tensor

def get_output_tensor(interpreter, output_details, idx):
    details = output_details[idx]
    if details['dtype'] == np.uint8 or details['dtype'] == np.int8:
        quant_params = details['quantization_parameters']
        int_tensor = interpreter.get_tensor(details['index']).astype(np.int32)
        real_tensor = int_tensor - quant_params['zero_points']
        real_tensor = real_tensor.astype(np.float32) * quant_params['scales']
    else:
        real_tensor = interpreter.get_tensor(details['index'])
    return real_tensor