import os
import tensorflow as tf

class Dataset:
    def __init__(self, data_folder, input_shape, validation_split_size, batch_size, preprocesss=None, seed=None):
        self.train_folder = os.path.join(data_folder, "train")
        self.test_folder = os.path.join(data_folder, "test")

        img_height, img_width = input_shape

        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            self.train_folder,
            validation_split=validation_split_size,
            subset="training",
            seed=seed,
            label_mode='categorical',
            image_size=(img_height, img_width),
            batch_size=batch_size
            )
        self.val_ds = None

        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            self.train_folder,
            validation_split=validation_split_size,
            subset="training",
            seed=seed,
            label_mode='categorical',
            image_size=(img_height, img_width),
            batch_size=batch_size
        )#.take(10)
        
        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            self.train_folder,
            validation_split=validation_split_size,
            subset="validation",
            seed=seed,
            image_size=(img_height, img_width),
            label_mode='categorical',
            batch_size=batch_size
        )#.take(10)

        self.test_ds = tf.keras.utils.image_dataset_from_directory(
            self.test_folder,
            seed=seed,
            image_size=(img_height, img_width),
            label_mode='categorical',
            batch_size=batch_size
        )#.take(10)

    def get_train_dataset(self, development=False):
        if development:
            return self.train_ds.take(10)
        else:
            return self.train_ds
        
    def get_validation_dataset(self, development=False):
        if development:
            return self.val_ds.take(10)
        else:
            return self.val_ds
        
    def get_test_dataset(self, development=False):
        if development:
            return self.test_ds.take(10)
        else:
            return self.test_ds