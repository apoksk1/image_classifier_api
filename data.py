import pathlib

import tensorflow as tf


class Data:

    def get_data(self, url):
        # dataset_url = url
        # data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
        return pathlib.Path(url)

    def get_train_ds(self, img_data, img_height, img_width, batch_size):
        return tf.keras.utils.image_dataset_from_directory(
            img_data,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)

    def get_val_ds(self, img_data, img_height, img_width, batch_size):
        return tf.keras.utils.image_dataset_from_directory(
            img_data,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)

    def get_class_names(self, ds):
        return ds.class_names

    def load_img(self, path_image):
        return tf.keras.utils.load_img(
            path_image, target_size=(180, 180)
        )
