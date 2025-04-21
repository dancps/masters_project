import os
import tensorflow as tf
import tensorflow_datasets as tfds

class Dataset:
    def __init__(self, dataset_name, input_shape, folds, validation_split_size, batch_size, preprocesss=None, seed=None):
        # Preprocessing is not implemented yet. It might be nice to have a default preprocessor to resize image size by using https://github.com/keras-team/keras/blob/f6c4ac55692c132cd16211f4877fac6dbeead749/keras/src/utils/image_dataset_utils.py#L402.
        # This is used in image_dataset_from_directory. See here for implementing this: https://github.com/keras-team/keras/blob/f6c4ac55692c132cd16211f4877fac6dbeead749/keras/src/utils/image_dataset_utils.py#L390
        if not (0 < validation_split_size < 1):
            raise ValueError("validation_split_size must be between 0 and 1.")

        img_height, img_width = input_shape
        validation_split_size = validation_split_size * 100
        train_percentage = 100 - validation_split_size

        train_split_str = f"train[:{train_percentage}%]"
        validation_split_str = f"train[{train_percentage}%:100%]"

        if folds == 1:
            #TODO: Verify shuffle. Is it happening? How can I do? How can I make sure it works on KFold(with or wihtout seed)?
            self.ds_train = tfds.load(dataset_name, split=train_split_str, as_supervised=True)
            self.ds_val   = tfds.load(dataset_name, split=validation_split_str, as_supervised=True)
            self.ds_test  = tfds.load(dataset_name, split="test", as_supervised=True)
        else:
            train_split_array = []
            validation_split_array = []
            for k in [k for k in range(0, 100, folds)]:
                train_split_str = f"train[:{k}%]+train[{k+10}%:]"
                val_split_str = f"train[{k}%:{k+10}%]"

                train_split_array.append(train_split_str)
                validation_split_array.append(val_split_str)
                
            self.ds_train = tfds.load(dataset_name, split=train_split_array, as_supervised=True)
            self.ds_val   = tfds.load(dataset_name, split=validation_split_array, as_supervised=True)
            self.ds_test  = tfds.load(dataset_name, split="test", as_supervised=True)

    def _get_dataset(self, dataset, development_samples = 10, development=False):
        if development:
            if isinstance(dataset, list):
                return [ds.take(development_samples) for ds in dataset]
            else:
                return dataset.take(development_samples)
        else:
            return dataset
            
            

    def get_train_dataset(self, development=False):
        return self._get_dataset(self.ds_train, development_samples=10, development=development)
        
    def get_validation_dataset(self, development=False):
        return self._get_dataset(self.ds_val, development_samples=10, development=development)
        
    def get_test_dataset(self, development=False):
        return self._get_dataset(self.ds_test, development_samples=10, development=development)

class DatasetOld:
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