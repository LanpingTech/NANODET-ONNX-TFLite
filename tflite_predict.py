import argparse
import os

import cv2
import numpy as np

from loguru import logger

import time

import tensorflow as tf

from utils.utils import image_preprocess, mkdir, post_process, visualize, format_input_tensor, get_output_tensor

class_names = ['person']

def make_parser():
    parser = argparse.ArgumentParser("tflite inference sample")
    parser.add_argument(
        "--model",
        type=str,
        default="models/nanodet_int8.tflite",
        help="Input your tflite model.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="image",
        help="mode type, eg. image, video and webcam.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default='imgs/000000001268.jpg',
        help="Path to your input image.",
    )
    parser.add_argument(
        "--camid", 
        type=int, 
        default=0, 
        help="webcam demo camera id",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='outputs',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.3,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="320,320",
        help="Specify an input shape for inference.",
    )
    return parser


def inference(interpreter, origin_img, args):
    t0 = time.time()
    img = image_preprocess(origin_img, args.input_shape, (0, 1, 2))

    inputs = img[np.newaxis]
    outputs = interpreter_compute(interpreter, inputs)
    output = outputs[0].transpose(0, 2, 1)

    results = post_process(output, len(class_names), 7, args.input_shape)

    result_image = visualize(results[0], origin_img, class_names, 0.35)

    return result_image

def image_process(interpreter, args):
    origin_img = cv2.imread(args.input_path)
    origin_img = inference(interpreter, origin_img, args)
    mkdir(args.output_path)
    output_path = os.path.join(args.output_path, args.input_path.split("/")[-1])
    logger.info("Saving detection result in {}".format(output_path))
    cv2.imwrite(output_path, origin_img)

def imageflow_demo(interpreter, args):
    cap = cv2.VideoCapture(args.input_path if args.mode == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)

    mkdir(args.output_path)
    current_time = time.localtime()
    save_folder = os.path.join(
        args.output_path, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    if args.mode == "video":
        save_path = os.path.join(save_folder, args.input_path.split("/")[-1])
    else:
        save_path = os.path.join(save_folder, "camera.mp4")

    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            result_frame = inference(interpreter, frame, args)
            vid_writer.write(result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break

def init_interpreter(args):    
    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.resize_tensor_input(input_details[0]["index"], [1, args.input_shape[0], args.input_shape[1], 3])
    interpreter.allocate_tensors()
    return interpreter

def interpreter_compute(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    inputs = format_input_tensor(input_data, input_details, 0) 
    interpreter.set_tensor(input_details[0]['index'], inputs)
    interpreter.invoke()
    outputs = [get_output_tensor(interpreter, output_details, 0)]

    return outputs

if __name__ == '__main__':
    args = make_parser().parse_args()
    args.input_shape = tuple(map(int, args.input_shape.split(',')))
    interpreter = init_interpreter(args)
    if args.mode == "image":
        image_process(interpreter, args)
    elif args.mode == "video" or args.mode == "webcam":
        imageflow_demo(interpreter, args)


    