import numpy as np
import tensorflow as tf

from data import Data

batch_size = 32
img_height = 180
img_width = 180

img = Data().load_img("592px-Red_sunflower.jpg")

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

TF_MODEL_FILE_PATH = 'models/15epoch/model.tflite' # The default path to the saved TensorFlow Lite model

interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)
print(interpreter.get_signature_list())

classify_lite = interpreter.get_signature_runner('serving_default')
print(classify_lite)


predictions_lite = classify_lite(sequential_input=img_array)['outputs']
score_lite = tf.nn.softmax(predictions_lite)

img_data = Data().get_data("flower_photos")
train_ds = Data().get_train_ds(img_data, img_height, img_width, batch_size)
print(Data().get_class_names(train_ds))
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(Data().get_class_names(train_ds)[np.argmax(score_lite)], 100 * np.max(score_lite))
)
