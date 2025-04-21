from masters.op.dataset import Dataset
import tensorflow as tf
from termcolor import colored
from tensorflow.keras.metrics import F1Score
from tensorflow.keras import Model
import datetime
from tabulate import tabulate
from termcolor import colored


class Experiment:
    """
    A experiment should recieve everything which is important to a run.

    The idea is to standardize the experiment, so you can either run a prod or dev run:
        python experiment.py run
    or
        python experiment.py run-development

    It should:
        - be able to run the steps of the experiment.
        - be able to save the results of the experiment.
    """

    def __init__(
        self,
        experiment_name: str,
        model: Model,
        epochs: int = None,
        dataset: Dataset = None,
        folds: int = 1,
    ):
        self.experiment_name = experiment_name
        self.epochs = epochs
        self.dataset = dataset
        self.model = model
        self.folds = folds  # TODO: Currently, folds is defined both in dataset and at here. I should review it.
        self.is_k_fold = folds > 1
        # self.metrics = metrics

    # def train(self):
    #     raise NotImplementedError("Train method are not implemented yet.")

    # def test(self):
    #     raise NotImplementedError("Evaluate method are not implemented yet.")

    def train(self, is_development=False):

        if self.is_k_fold:
            # Load data
            train_ds = self.dataset.get_train_dataset(is_development)
            val_ds = self.dataset.get_validation_dataset(is_development)
        else:
            # Load data
            train_ds = [self.dataset.get_train_dataset(is_development)]
            val_ds = [self.dataset.get_validation_dataset(is_development)]

        for fold in range(self.folds):
            print(colored(f"Fold {fold+1}/{self.folds}", "cyan"))

            # Used to measure performance and to identify the run
            start_time = datetime.datetime.now()
            start_time_identifier = start_time.strftime("%Y%m%d-%H%M%S")

            # Gets the current fold
            current_train_ds = train_ds[fold]
            current_val_ds = val_ds[fold]

            # TODO: REMOVE
            # DEBUG
            print(current_train_ds)

            # Reduces input size when in development mode
            if is_development:
                epochs = 3
                stage = "dev"
            else:
                epochs = self.epochs
                stage = "prod"

            # Defines the metrics
            f1 = F1Score(average="macro", threshold=0.5)
            precision_metric = tf.keras.metrics.Precision(name="precision")  # , class_id = 4)

            # Defines the checkpoints
            #   If not exists, creates the directory
            k_fold_dir = ""
            if self.is_k_fold:
                k_fold_identifier = f"fold-{fold+1}"
                k_fold_dir = f"/{k_fold_identifier}"

            checkpoints_path = f"data/experiments/{self.experiment_name}/checkpoints/{stage}{k_fold_dir}/{start_time_identifier}"
            model_checkpoint = f"{checkpoints_path}/model"

            # Saves all the epochs
            all_epochs_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{model_checkpoint}/all_epochs/{self.experiment_name}"
                + "-{epoch:04d}.weights.h5",
                save_weights_only=True,
                verbose=1,
            )

            # Saves the best only
            best_train_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{model_checkpoint}/{self.experiment_name}-best"
                + "-{epoch:04d}.weights.h5",
                save_weights_only=True,
                save_best_only=True,
                monitor=f1.name,
                verbose=1,
            )

            # Saves the best val only
            best_val_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{model_checkpoint}/{self.experiment_name}-best-val"
                + "-{epoch:04d}.weights.h5",
                save_weights_only=True,
                save_best_only=True,
                monitor="val_" + f1.name,
                verbose=1,
            )

            # Saves the last one
            last_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{model_checkpoint}/{self.experiment_name}-last.weights.h5",
                save_weights_only=True,
                verbose=1,
            )

            # For using at tensorboard
            log_checkpoint = f"{checkpoints_path}/logs/"
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=log_checkpoint,
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                write_steps_per_second=False,
                update_freq="epoch",
                profile_batch=0,
                embeddings_freq=0,
                embeddings_metadata=None,
            )

            self.model.compile(
                optimizer="adam",
                # loss='categorical',
                loss="categorical_crossentropy",
                # loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False), #'categorical_crossentropy', #
                metrics=["accuracy", precision_metric, f1],
            )  # F1Score()

            history = self.model.fit(
                current_train_ds,
                validation_data=current_val_ds,
                epochs=epochs,
                callbacks=[
                    all_epochs_callback,
                    best_train_callback,
                    best_val_callback,
                    last_callback,
                    tensorboard_callback,
                ],
            )

            print(history)

            end_time = datetime.datetime.now()
            print(f"\nTraining results:")
            print(
                tabulate(
                    [
                        ["Start time", start_time],
                        ["End time", end_time],
                        ["Duration", end_time - start_time],
                        ["Identifier", start_time_identifier],
                    ],
                    tablefmt="fancy_grid",
                )
            )

    def evaluate(self, is_development=False):

        test_ds = self.dataset.get_test_dataset(is_development)

        print("Evaluate")
        result = self.model.evaluate(test_ds)
        print(dict(zip(self.model.metrics_names, result)))

    def run(self, is_development=False):
        # Run the steps of the experiment
        #     What composes?

        # Steps
        # - Standardized sources
        # - Analytics of sources
        # - Preproc
        # - Analytics of input data
        # - Train
        # - Test
        # - Results
        print(f"Running the experiment: " + colored(self.experiment_name, "green"))
        self.train(is_development)
        self.evaluate(is_development)
