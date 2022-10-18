import tensorflow as tf

image_shape = (1, 320, 320, 3)

def representative_dataset_gen():
    for i in range(10):
        # creating fake images
        image = tf.random.normal(image_shape)
        yield [image]

def to_tflite(keras_model, save_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    open(save_path, "wb").write(tflite_model)

def to_quantized_tflite_fp16(keras_model, save_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    open(save_path, "wb").write(tflite_model)

def to_quantized_tflite_int8(keras_model, save_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.representative_dataset = tf.lite.RepresentativeDataset(representative_dataset_gen)
    converter.experimental_new_converter = True
    converter.experimental_new_quantizer = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    open(save_path, "wb").write(tflite_model)

if __name__ == "__main__":
    keras_model_path = "models/nanodet.h5"
    tflite_model_path = "models/nanodet.tflite"
    tflite_fp16_model_path = "models/nanodet_fp16.tflite"
    tflite_int8_model_path = "models/nanodet_int8.tflite"
    keras_model = tf.keras.models.load_model(keras_model_path)
    # to_tflite(keras_model, tflite_model_path)
    # to_quantized_tflite_fp16(keras_model, tflite_fp16_model_path)
    to_quantized_tflite_int8(keras_model, tflite_int8_model_path)