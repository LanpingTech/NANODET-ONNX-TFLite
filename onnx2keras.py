from utils import load_onnx_modelproto, keras_builder

onnx_model_path = "model/nanodet.onnx"
keras_model_path = "model/nanodet.h5"
onnx_model = load_onnx_modelproto(onnx_model_path)
keras_model = keras_builder(onnx_model)
keras_model.save(keras_model_path)

