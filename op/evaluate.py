from masters.experiments.armnet_experiment import ArmnetExperiment, Armnet10FoldExperiment, ResNetExperiment, ResNet10FoldsExperiment

from termcolor import colored
import glob
import os
import json
from pprint import pprint

if __name__ == "__main__":
    # Example usage
    armnet_experiment = ArmnetExperiment
    armnet_10_fold_experiment = Armnet10FoldExperiment
    resnet_experiment = ResNetExperiment
    resnet_10_fold_experiment = ResNet10FoldsExperiment

    list_of_exps = [
        (armnet_experiment, "./data/experiments/armnet_experiment/"),
        (armnet_10_fold_experiment, "./data/experiments/armnet_10_folds_experiment/"),
        (resnet_experiment, "./data/experiments/resnet_experiment/"),
        (resnet_10_fold_experiment, "./data/experiments/resnet_10_folds_experiment/")
    ]
    # data/experiments/armnet_experiment/checkpoints/prod/1-folds-20250427-072427/armnet_experiment_20250427-072427_final_results.json
    for experiment, experiment_path in list_of_exps:
        experiment_name = experiment().experiment_name
        print(f"Evaluating {colored(experiment_name, 'green')}")

        # This will load the json to register results
        folds_jsons = f"{experiment_path}checkpoints/prod/*-folds-*/{experiment_name}*final_results.json"
        jsons = sorted(glob.glob(folds_jsons))
        if len(jsons)==1:
            with open(jsons[0], 'r') as f:
                data = json.load(f)
        else:
            print(colored("Multiple jsons found, ignoring", 'red'))
            print(jsons)


        fold_pattern = f"{experiment_path}checkpoints/prod/*-folds-*/fold-*/model/"
        folds_folders = glob.glob(fold_pattern)
        for fold_folder in sorted(folds_folders):
            w_pattern = f"{fold_folder}/{experiment_name}-best-val-*.weights.h5"
            path = sorted(glob.glob(w_pattern))
            best_val_path = path[-1] if len(path) > 0 else None
            # print(f"  - {path[-1]}")
            convergence_epoch = os.path.basename(best_val_path.replace(f"{experiment_name}-best-val-", "").replace(".weights.h5", ""))
            result = experiment().evaluate(checkpoint = best_val_path)
            print(colored("Covergence epoch: ", attrs=['bold']), convergence_epoch)
            print(colored("Result: ", attrs=['bold']), result)
            current_fold = [f for f in fold_folder.split("/") if "fold-" in f][0]
            current_fold = "-".join(current_fold.split("-")[:2])
            print(colored("Current fold: ", attrs=['bold']), current_fold)
            fold_result_current_index = [i for i, fold in enumerate(data['folds_results']) if fold['fold'] == current_fold]
            if len(fold_result_current_index) != 0:
                fold_result_current_index = fold_result_current_index[0]
                # print(f"Current fold index: {fold_result_current_index}")

                # The reason behind this being a list, is that we could evaluate with different weights.(e.g. best val, last, etc)
                if "evaluate_results" not in data['folds_results'][fold_result_current_index]:
                    data['folds_results'][fold_result_current_index]["evaluate_results"] = []
                data['folds_results'][fold_result_current_index]["evaluate_results"].append({"convergence_epoch": convergence_epoch, "weight_path": best_val_path, "loss": result["loss"], "compile_metric": result["compile_metrics"]})
            else: 
                print(colored("Fold result not found", 'red'))

        # for fold_result in data['folds_results']:
        #     print(f"Fold: {fold_result['fold']}")
        #     if "evaluate_results" in fold_result:
        #         for key, value in fold_result["evaluate_results"].items():
        #             print(f"  - {key}: {value}")
        #     else:
        #         print(colored("  - No evaluate results", 'red'))

        if len(jsons)==1:
            # resnet_experiment_20250427-081026_final_results.json
            results_path = jsons[0]
            results_path = results_path.replace(f"final_results.json", "final_results_evaluate.json")
            with open(results_path, "w") as json_file:
                json.dump(data, json_file, indent=4)
            print("Final results saved to "+colored(f"{results_path}", "green"))
        else:
            print(colored("Multiple jsons found, ignoring", 'red'))
            print(jsons)
        print("----------------------------------------------------------")




    # # print(f"Evaluating {colored('armnet_experiment', 'green')}")
    # armnet_experiment.evaluate(checkpoint = "./data/experiments/armnet_experiment/checkpoints/prod/1-folds-20250424-085427/fold-1-20250424-085427/model/armnet_experiment-last.weights.h5")
    # print("----------------------------------------------------------")

    # # print(f"Evaluating {colored('armnet_10_fold_experiment', 'green')}")    
    # armnet_10_fold_experiment.evaluate(checkpoint = "./data/experiments/armnet_10_folds_experiment/checkpoints/prod/10-folds-20250424-085820/fold-8-20250424-092435/model/armnet_10_folds_experiment-last.weights.h5") # best on 8
    # print("----------------------------------------------------------")

    # # print(f"Evaluating {colored('resnet_experiment', 'green')}")
    # resnet_experiment.evaluate(checkpoint = "./data/experiments/resnet_experiment/checkpoints/prod/1-folds-20250424-093616/fold-1-20250424-093616/model/resnet_experiment-last.weights.h5")
    # print("----------------------------------------------------------")

    # # print(f"Evaluating {colored('resnet_10_fold_experiment', 'green')}")
    # resnet_10_fold_experiment.evaluate(checkpoint = "./data/experiments/resnet_10_folds_experiment/checkpoints/prod/10-folds-20250424-095027/fold-6-20250424-110002/model/resnet_10_folds_experiment-last.weights.h5")
    # print("----------------------------------------------------------")