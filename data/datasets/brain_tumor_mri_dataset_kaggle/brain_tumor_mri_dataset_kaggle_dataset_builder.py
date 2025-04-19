"""brain_tumor_mri_dataset_kaggle dataset.

Read this to learn more:
- https://www.tensorflow.org/datasets/add_dataset
- https://www.tensorflow.org/datasets/format_specific_dataset_builders (alternatively)
"""

import tensorflow_datasets as tfds
import os

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for brain_tumor_mri_dataset_kaggle dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(brain_tumor_mri_dataset_kaggle): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(), # can use this argument to specify the image size shape=(None, None, 3)
            'label': tfds.features.ClassLabel(names=['glioma', 'meningioma','notumor','pituitary']),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(brain_tumor_mri_dataset_kaggle): Downloads the data and defines the splits
    path = dl_manager.download_and_extract('https://www.kaggle.com/api/v1/datasets/download/masoudnickparvar/brain-tumor-mri-dataset')

    # TODO(brain_tumor_mri_dataset_kaggle): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(path / 'Training'),
        'test': self._generate_examples(path / 'Testing'),
    }

  def _generate_examples(self, filepath):
    """Yields examples."""
    # TODO(brain_tumor_mri_dataset_kaggle): Yields (key, example) tuples from the dataset
    # for f in path.glob('*.jpg'):
    #   yield 'key', {
    #       'image': f,
    #       'label': 'yes',
    #   }

    """Yields image examples from folder structure."""
    for class_name in sorted(os.listdir(filepath)):
        class_dir = os.path.join(filepath, class_name)
        if not os.path.isdir(class_dir):
            continue  # skip non-directory files, if any

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            key = f"{class_name}/{img_name}"
            yield key, {
                "image": img_path,
                "label": class_name,
            }
