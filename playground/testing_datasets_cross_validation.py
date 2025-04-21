import masters.data.datasets.brain_tumor_mri_dataset_kaggle
import tensorflow_datasets as tfds
from tabulate import tabulate
from termcolor import colored


# Read: 
#   - https://www.tensorflow.org/datasets/splits#cross_validation
#   - https://www.tensorflow.org/datasets/splits#tfdseven_splits_multi-host_training (even splits, for a diff k-fold)


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

def check_classes_distribution(ds):
    # print(f"Split: {type(ds)}")
    print("Class distribution:")

    e = ds.take(1)

    print(f"DS  : {e}")
    print([x for x in dir(e) if not x.startswith('_')])
    for ex in e:
        print(f"Example size: {len(ex)}")

        # img = ex['image']
        # label = ex['label']
        # print(f"Image shape: {img.shape}, Label: {label}")

# tds = dataset.get_train_dataset()

# print(tds['0:100'])
dataset_name = 'brain_tumor_mri_dataset_kaggle'
# dataset_name = 'omniglot'

K = 10 # number of folds
train_split_array = []
validation_split_array = []
for k in [k for k in range(0, 100, K)]:
    train_split_str = f"train[:{k}%]+train[{k+10}%:]"
    val_split_str = f"train[{k}%:{k+10}%]"

    train_split_array.append(train_split_str)
    validation_split_array.append(val_split_str)
    # print(f"Validation: train[{k}%:{k+10}%]")
    # print(f"Train     : ")
    # print()

ds_train = tfds.load(dataset_name, split=train_split_array, as_supervised=True)
ds_val   = tfds.load(dataset_name, split=validation_split_array, as_supervised=True)
ds_test  = tfds.load(dataset_name, split="test", as_supervised=True)


ds = ds_train[0]
ds = ds.take(1)

for image, label in ds:  # example is (image, label)
    print(image.shape, label)


print(colored(f"Type of ds_train: {type(ds_train)}", "cyan"))
print(colored(f"Type of first item in ds_train: {type(ds_train[0])}", "blue"))
print(colored(f"Type of first element in ds_train[0]: {type(list(ds_train[0].as_numpy_iterator())[0])}", "green"))
print(
    tabulate(
        [
            ["", "Size of set", "Equal to K", "Type"],
            ["Number of folds(val)", len(ds_val), len(ds_val)==K, type(ds_val[0])],
            ["Number of folds(train)", len(ds_train), len(ds_train)==K, "list of "+str(type(ds_train[0]))],
            ["Number of folds(test)", len(ds_test),"", type(ds_test)],
        ],
        tablefmt="fancy_grid",
        )
    )

ds_test = ds_test.take(1)
for test_example in ds_test:
    # print(f"Test example: {test_example}")
    try:
        print(test_example["label"])
    except:
        print(colored("test_example['label'] not available", "red"))
        pass


    
image, label = list(ds_test)[0]
print(f"Image shape: {image.shape}, Label: {label}")


# Check for each fold class distribution
for split_idx, split in enumerate(ds_train):
    print(f"Fold {split_idx+1}/{len(ds_train)} train split")
    print(f"Split type: {type(split)}")
    print(f"Split length: {len(split)}")
    
    print(split["image"])
    print(list(split.keys()))
    check_classes_distribution(split)
    print()