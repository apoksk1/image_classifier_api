from tensorflow import keras
from keras import layers
from keras.models import Sequential
import tensorflow as tf
import os


def train_save(class_names, train_ds, val_ds, epochs: int,
               img_width: int, batch_size: int, img_height: int, model_path):
    ##Data augmentation
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=(img_height,
                                           img_width,
                                           3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    ##Dropout
    num_classes = len(class_names)

    model = Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, name="outputs")
    ])

    ##compile and train the model

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    if not os.path.exists(model_path):
        os.makedirs(model_path)
        print("Directory created.")

    # save
    with open(os.path.join(model_path, 'model.tflite'), 'wb') as f:
        f.write(tflite_model)
