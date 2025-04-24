from masters.op.experiment import Experiment
from masters.op.dataset import Dataset
from masters.models.armnet import RMFNet
from tensorflow.keras.applications import ResNet50
import argparse


class ArmnetExperiment(Experiment):
    """
    ArmnetExperiment is a subclass of the Experiment class.
    It is used to run the Armnet experiment.
    """

    def __init__(self, development=False):
        dataset = Dataset(
            dataset_name="brain_tumor_mri_dataset_kaggle",
            input_shape=(224, 224),
            folds=1,
            validation_split_size=0.2,
            batch_size=64,
            seed=123,
        )
        epochs = 50
        model = RMFNet()
        experiment_name = "armnet_experiment"
        super().__init__(experiment_name, model, epochs, dataset)


class Armnet10FoldExperiment(Experiment):
    """
    x
    """

    def __init__(self, development=False):
        folds = 10
        dataset = Dataset(
            dataset_name="brain_tumor_mri_dataset_kaggle",
            input_shape=(224, 224),
            folds=folds,
            validation_split_size=0.2,
            batch_size=64,
            seed=123,
        )
        epochs = 50
        model = RMFNet()
        experiment_name = "armnet_10_folds_experiment"
        super().__init__(experiment_name, model, epochs, dataset, folds)

class Armnet128Experiment(Experiment):
    """
    ArmnetExperiment is a subclass of the Experiment class.
    It is used to run the Armnet experiment.
    """

    def __init__(self, development=False):
        dataset = Dataset(
            dataset_name="brain_tumor_mri_dataset_kaggle",
            input_shape=(224, 224),
            folds=10,
            validation_split_size=0.2,
            batch_size=128,
            seed=123,
        )
        epochs = 50
        model = RMFNet()
        experiment_name = "armnet_bs128_experiment"
        super().__init__(experiment_name, model, epochs, dataset)

# class Armnet5FoldExperiment(Experiment):
#     """
#     x
#     """

#     def __init__(self, development=False):
#         folds = 5
#         dataset = Dataset(
#             dataset_name="brain_tumor_mri_dataset_kaggle",
#             input_shape=(224, 224),
#             folds=folds,
#             validation_split_size=0.2,
#             batch_size=64,
#             seed=123,
#         )
#         epochs = 50
#         model = RMFNet()
#         experiment_name = "armnet_5_folds_experiment"
#         super().__init__(experiment_name, model, epochs, dataset, folds)


class ResNetExperiment(Experiment):
    """
    x
    """

    def __init__(self, development=False):
        dataset = Dataset(
            "brain_tumor_mri_dataset_kaggle",
            input_shape=(224, 224),
            folds=1,
            validation_split_size=0.2,
            batch_size=64,
            seed=123,
        )
        epochs = 50
        model = ResNet50(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=4,
            classifier_activation="softmax",
        )
        experiment_name = "resnet_experiment"
        super().__init__(experiment_name, model, epochs, dataset)

class ResNet10FoldsExperiment(Experiment):
    """
    x
    """

    def __init__(self, development=False):
        dataset = Dataset(
            "brain_tumor_mri_dataset_kaggle",
            input_shape=(224, 224),
            folds=10,
            validation_split_size=0.2,
            batch_size=64,
            seed=123,
        )
        epochs = 50
        model = ResNet50(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=4,
            classifier_activation="softmax",
        )
        experiment_name = "resnet_10_folds_experiment"
        super().__init__(experiment_name, model, epochs, dataset)

# class ResNet5FoldsExperiment(Experiment):
#     """
#     x
#     """

#     def __init__(self, development=False):
#         dataset = Dataset(
#             "brain_tumor_mri_dataset_kaggle",
#             input_shape=(224, 224),
#             folds=5,
#             validation_split_size=0.2,
#             batch_size=64,
#             seed=123,
#         )
#         epochs = 50
#         model = ResNet50(
#             include_top=True,
#             weights=None,
#             input_tensor=None,
#             input_shape=None,
#             pooling=None,
#             classes=4,
#             classifier_activation="softmax",
#         )
#         experiment_name = "resnet_5_folds_experiment"
#         super().__init__(experiment_name, model, epochs, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--experiments",
        nargs="+",
        default=[],
        help="which experiments should run.",
    )
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument(
        "-d", "--development", help="changes to dev experience", action="store_true"
    )
    args = parser.parse_args()

    all_experiments = [
        ArmnetExperiment(),
        Armnet10FoldExperiment(),
        Armnet128Experiment(),
        ResNetExperiment(),
        ResNet10FoldsExperiment()
    ]
    
    if args.experiments == []:
        _ = [experiment.run(args.development) for experiment in all_experiments]
    else:
        for experiment in all_experiments:
            if experiment.experiment_name in args.experiments:
                experiment.run(args.development)
