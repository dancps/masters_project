from masters.op.dataset import Dataset
import tensorflow as tf
from termcolor import colored
from tensorflow.keras.metrics import F1Score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
# import keras_tuner as kt
import datetime
from tabulate import tabulate
from termcolor import colored
from pprint import pprint
import json
import os

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

    def train(self, is_development=False):
        experiment_start_time = datetime.datetime.now()
        experiment_start_time_id = experiment_start_time.strftime("%Y%m%d-%H%M%S")
        folds_results = []
        if self.is_k_fold:
            # Load data
            train_ds = self.dataset.get_train_dataset(is_development)
            val_ds = self.dataset.get_validation_dataset(is_development)
            folds_dir = f"{self.folds}-folds-{experiment_start_time_id}"
        else:
            # Load data
            train_ds = [self.dataset.get_train_dataset(is_development)]
            val_ds = [self.dataset.get_validation_dataset(is_development)]
            folds_dir = f"1-folds-{experiment_start_time_id}"

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
                epochs = 2
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
                k_fold_dir = f"{k_fold_identifier}-{start_time_identifier}"
            else:
                k_fold_identifier = "fold-1"
                k_fold_dir = f"{k_fold_identifier}-{start_time_identifier}"

            checkpoints_path = f"data/experiments/{self.experiment_name}/checkpoints/{stage}/{folds_dir}/{k_fold_dir}"
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
                mode="max"
            )

            # Saves the best val only
            best_val_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{model_checkpoint}/{self.experiment_name}-best-val"
                + "-{epoch:04d}.weights.h5",
                save_weights_only=True,
                save_best_only=True,
                monitor="val_" + f1.name,
                verbose=1,
                mode="max"
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
                optimizer=Adam(learning_rate=0.001),
                loss="categorical_crossentropy",
                metrics=[f1, "accuracy", precision_metric],
            )

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

            # print(f"{colored("History:", "cyan")}")
            # print(history.history)

            

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
            # Saves at checkpoints_path
            results_fold = {
                "fold": k_fold_identifier,
                "history": history.history,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration": (end_time - start_time).total_seconds(),
                "identifier": start_time_identifier,
            }
            # TODO: Uncomment this to debug
            # print(colored("Fold results:", "cyan"))
            # pprint(results_fold)
            folds_results.append(results_fold)


        experiment_end_time = datetime.datetime.now()
        # experiment_end_time_id = experiment_end_time.strftime("%Y%m%d-%H%M%S")
        print("---------------------------------------------------------------------")
        print(
                tabulate(
                    [
                        ["Experiment start time", experiment_start_time],
                        ["Experiment end time", experiment_end_time],
                        ["Duration", experiment_end_time - experiment_start_time],
                        ["Identifier", experiment_start_time_id],
                    ],
                    tablefmt="fancy_grid",
                )
            )
        final_results = {
            "experiment_name": self.experiment_name,
            "epochs": epochs,
            "stage": stage,
            "folds": self.folds,
            "experiment_start_time": experiment_start_time.isoformat(),
            "experiment_end_time": experiment_end_time.isoformat(),
            "duration": (experiment_end_time - experiment_start_time).total_seconds(),
            "identifier": experiment_start_time_id,
            "folds_results": folds_results,
        }
        # # Uncomment this to debug
        # print(colored("Final results:", "green"))
        # pprint(final_results)

        # Save final results into a JSON file
        results_path = f"data/experiments/{self.experiment_name}/checkpoints/{stage}/{folds_dir}/{self.experiment_name}_{experiment_start_time_id}_final_results.json"
        with open(results_path, "w") as json_file:
            json.dump(final_results, json_file, indent=4)
        print("Final results saved to "+colored(f"{results_path}", "green"))

    def evaluate(self, checkpoint = None, is_development=False):
        print("Evaluating the experiment: " + colored(self.experiment_name, "green"))
        if not checkpoint is None:
            print(f"Loading checkpoint: {os.path.basename(checkpoint)}")
            self.model.build(input_shape=(None, 224, 224, 3))

            f1 = F1Score(average="macro", threshold=0.5)
            precision_metric = tf.keras.metrics.Precision(name="precision")  # , class_id = 4)
            self.model.compile(
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=[f1, "accuracy", precision_metric],
            ) 
            # self.model.summary()
            self.model.load_weights(checkpoint)
            # .set_weights(weights)
        

        test_ds = self.dataset.get_test_dataset(is_development)

        result = self.model.evaluate(test_ds)
        # print(result)
        # print(self.model.metrics_names)
        print(dict(zip(self.model.metrics_names, result)))
        return dict(zip(self.model.metrics_names, result))

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
        self.evaluate(None,is_development)
