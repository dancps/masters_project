from masters.experiments.armnet_experiment import ArmnetExperiment, Armnet10FoldExperiment, ResNetExperiment, ResNet10FoldsExperiment

from termcolor import colored

if __name__ == "__main__":
    # Example usage
    armnet_experiment = ArmnetExperiment()
    armnet_10_fold_experiment = Armnet10FoldExperiment()
    resnet_experiment = ResNetExperiment()
    resnet_10_fold_experiment = ResNet10FoldsExperiment()

    # print(f"Evaluating {colored('armnet_experiment', 'green')}")
    armnet_experiment.evaluate(checkpoint = "/Users/danilo.calhes/Documents/projects/pessoal/masters_project/data/experiments/armnet_experiment/checkpoints/prod/1-folds-20250424-085427/fold-1-20250424-085427/model/armnet_experiment-last.weights.h5")
    print("----------------------------------------------------------")

    # print(f"Evaluating {colored('armnet_10_fold_experiment', 'green')}")    
    armnet_10_fold_experiment.evaluate(checkpoint = "/Users/danilo.calhes/Documents/projects/pessoal/masters_project/data/experiments/armnet_10_folds_experiment/checkpoints/prod/10-folds-20250424-085820/fold-8-20250424-092435/model/armnet_10_folds_experiment-last.weights.h5") # best on 8
    print("----------------------------------------------------------")

    # print(f"Evaluating {colored('resnet_experiment', 'green')}")
    resnet_experiment.evaluate(checkpoint = "/Users/danilo.calhes/Documents/projects/pessoal/masters_project/data/experiments/resnet_experiment/checkpoints/prod/1-folds-20250424-093616/fold-1-20250424-093616/model/resnet_experiment-last.weights.h5")
    print("----------------------------------------------------------")

    # print(f"Evaluating {colored('resnet_10_fold_experiment', 'green')}")
    resnet_10_fold_experiment.evaluate(checkpoint = "/Users/danilo.calhes/Documents/projects/pessoal/masters_project/data/experiments/resnet_10_folds_experiment/checkpoints/prod/10-folds-20250424-095027/fold-6-20250424-110002/model/resnet_10_folds_experiment-last.weights.h5")
    print("----------------------------------------------------------")