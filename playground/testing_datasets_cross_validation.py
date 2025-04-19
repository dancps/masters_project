import masters.data.datasets.brain_tumor_mri_dataset_kaggle
import tensorflow_datasets as tfds

# Read: 
#   - https://www.tensorflow.org/datasets/splits#cross_validation


# dataset = Dataset("data/datasets/mbtd/raw", (224, 224), 0.2, 64, seed=123)
# # builder = tfds.builder('my_dataset')
# # builder.info.splits['train'].num_examples  # 10_000
# # builder.info.splits['train[:75%]'].num_examples  # 7_500 (also works with slices)
# # builder.info.splits.keys()  # ['train', 'test']

# # vals_ds = tfds.load('mnist', split=[
# #     f'train[{k}%:{k+10}%]' for k in range(0, 100, 10)
# # ])
# # trains_ds = tfds.load('mnist', split=[
# #     f'train[:{k}%]+train[{k+10}%:]' for k in range(0, 100, 10)
# # ])


# tds = dataset.get_train_dataset()

# print(tds['0:100'])
dataset_name = 'brain_tumor_mri_dataset_kaggle'

ds = tfds.load(dataset_name, split='train', shuffle_files=True)


ds_train = tfds.load(
    dataset_name, 
    split=[
        f'train[:{k}%]+train[{k+10}%:]' for k in range(0, 100, 10)
    ], 
    # as_supervised=True
    )

ds_val = tfds.load(dataset_name, as_supervised=True)
print(len(ds_train))
print(ds_train)