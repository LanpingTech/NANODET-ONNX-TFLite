import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from onnx import numpy_helper
from .op_registry import OPERATOR

def representative_dataset_gen(img_root, img_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    if isinstance(mean, list):
        mean = np.array(mean, dtype=np.float32)
    if isinstance(std, list):
        std = np.array(std, dtype=np.float32)

    if img_root is None or (not os.path.exists(img_root)):
        for _ in range(20):
            _input = np.random.rand(img_size[0], img_size[1], 3).astype(np.float32)
            if mean is not None:
                _input = (_input - mean)
            if std is not None:
                _input = _input/std
            _input = np.expand_dims(_input, axis=0).astype(np.float32)
            yield [_input]
    else:
        VALID_FORMAT = ['jpg', 'png', 'jpeg']
        for i, fn in enumerate(os.listdir(img_root)):
            if fn.split(".")[-1] not in VALID_FORMAT:
                continue
            _input = cv2.imread(os.path.join(img_root, fn))
            _input = cv2.resize(_input, (img_size[1], img_size[0]))[:, :, ::-1]
            _input = _input/255
            if mean is not None:
                _input = (_input - mean)
            if std is not None:
                _input = _input/std

            _input = np.expand_dims(_input, axis=0).astype(np.float32)
            yield [_input]
            if i >= 100:
                break

def decode_node_attribute(node)->dict:
    op_attr = dict()
    for x in node.attribute:
        if x.type == 1:
            op_attr[x.name] = x.f
        elif x.type == 2:
            op_attr[x.name] = x.i
        elif x.type == 3:
            op_attr[x.name] = x.s.decode()
        elif x.type == 4:
            op_attr[x.name] = numpy_helper.to_array(x.t)
            if op_attr[x.name].size == 0:
                op_attr[x.name] = np.array([0])
        elif x.type == 7:
            op_attr[x.name] = x.ints
    return op_attr
    
def keras_builder(onnx_model, new_input_nodes:list=None, new_output_nodes:list=None):
    model_graph = onnx_model.graph
    onnx_weights = dict()
    for initializer in model_graph.initializer:
        onnx_weights[initializer.name] = numpy_helper.to_array(initializer)

    tf_tensor, input_shape = {}, []
    for inp in model_graph.input:
        input_shape = [x.dim_value for x in inp.type.tensor_type.shape.dim]
        if input_shape == []:
            continue
        batch_size = 1 if input_shape[0] <= 0 else input_shape[0]
        input_shape = input_shape[2:] + input_shape[1:2]
        tf_tensor[inp.name] = keras.Input(shape=input_shape, batch_size=batch_size)

    input_node_names, outputs_node_names = [], []
    for node in model_graph.node:
        op_name, node_inputs, node_outputs, node_name = node.op_type, node.input, node.output, node.name
        op_attr = decode_node_attribute(node)
        
        tf_operator = OPERATOR.get(op_name)
        if tf_operator is None:
            raise KeyError(f"{op_name} not implemented yet")
        
        _inputs = None 
        if len(node_inputs) > 0:
            _inputs = tf_tensor[node_inputs[0]] if node_inputs[0] in tf_tensor else onnx_weights[node_inputs[0]]

        for index in range(len(node_outputs)):
            tf_tensor[node_outputs[index]] = tf_operator(tf_tensor, onnx_weights, node_inputs, op_attr, index=index)(_inputs)

        if new_input_nodes is not None and node_name in new_input_nodes:
            input_node_names.append(node_outputs[0])
        # TODO for nodes with multiply outputs.
        if new_output_nodes is not None and node_name in new_output_nodes:
            outputs_node_names.append(node_outputs[0])
        if new_output_nodes is not None and len(outputs_node_names) == len(new_output_nodes):
            break
    input_nodes = []
    if new_input_nodes is None:
        input_nodes = [tf_tensor[x.name] for x in model_graph.input]
    else:
        input_nodes = [tf_tensor[x] for x in input_node_names]
    outputs_nodes = []
    if new_output_nodes is None:
        outputs_nodes = [tf_tensor[x.name] for x in model_graph.output]
    else:
        for node in model_graph.output:
            if node.name in new_output_nodes:
                outputs_node_names.append(node.name)
        outputs_nodes = [tf_tensor[x] for x in outputs_node_names]

    keras_model = keras.Model(inputs=input_nodes, outputs=outputs_nodes)
    keras_model.trainable = False
    keras_model.summary()

    return keras_model

def tflite_builder(keras_model, weight_quant:bool=False, int8_model:bool=False, image_root:str=None,
                    int8_mean:list or float = [0.485, 0.456, 0.406], int8_std:list or float = [0.229, 0.224, 0.225]):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    if weight_quant or int8_model:
        converter.experimental_new_converter = True
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if int8_model:
        input_shape = (keras_model.inputs[0].shape[1], keras_model.inputs[0].shape[2])
        converter.representative_dataset = lambda: representative_dataset_gen(image_root, input_shape, int8_mean, int8_std)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.target_spec.supported_types = []
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        converter.experimental_new_converter = True

    tflite_model = converter.convert()
    return tflite_model