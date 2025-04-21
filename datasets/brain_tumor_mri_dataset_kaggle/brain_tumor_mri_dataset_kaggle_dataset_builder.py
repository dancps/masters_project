"""brain_tumor_mri_dataset_kaggle dataset.

Read this to learn more:
- https://www.tensorflow.org/datasets/add_dataset
- https://www.tensorflow.org/datasets/format_specific_dataset_builders (alternatively)
"""

import tensorflow_datasets as tfds
import os
from termcolor import colored


CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def info(label, text):
    print(f"[{colored(label, 'blue')}] {colored(text, 'green')}")


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for brain_tumor_mri_dataset_kaggle dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(brain_tumor_mri_dataset_kaggle): Specifies the tfds.core.DatasetInfo object
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "image": tfds.features.Image(),  # can use this argument to specify the image size shape=(None, None, 3)
                    "label": tfds.features.ClassLabel(names=CLASSES),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=("image", "label"),  # Set to `None` to disable
            homepage="https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(brain_tumor_mri_dataset_kaggle): Downloads the data and defines the splits
        path = dl_manager.download_and_extract(
            "https://www.kaggle.com/api/v1/datasets/download/masoudnickparvar/brain-tumor-mri-dataset"
        )

        # TODO(brain_tumor_mri_dataset_kaggle): Returns the Dict[split names, Iterator[Key, Example]]
        # return {
        #     "train": self._generate_examples(path / "Training"),
        #     "test": self._generate_examples(path / "Testing"),
        # }
        info("Path", f"{path}")  # For debugging
        info("Path", f"{tfds.Split.TRAIN}")  # For debugging
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(path, "Training"),
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(path, "Testing"),
                },
            ),
        ]

    # TODO: Not used yet(?)
    def _generate_examples(self, filepath):
        """Yields examples."""
        info("Path", f"{filepath}")  # For debugging
        for class_name in sorted(CLASSES):
            class_dir = os.path.join(filepath, class_name)

            # Maybe the validation below shoudlnt continue. Should be a hard validation, failing on inconsistencies
            if not os.path.isdir(class_dir):
                continue  # skip non-directory files, if any

            count = 0  # For debugging
            for img_name in os.listdir(class_dir):
                count += 1
                img_path = os.path.join(class_dir, img_name)
                key = f"{class_name}/{img_name}"
                yield key, {
                    "image": img_path,
                    "label": class_name,
                }

            info(f"{class_name}", f"Found {count} images")  # For debugging
