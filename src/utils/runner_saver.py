import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from pprint import pprint

import optuna

import seaborn as sns
import matplotlib.pyplot as plt

from src.experiment import Experiment


class RunnerSaver:
    @staticmethod
    def ensure_folder(experiment: Experiment, output_dir: Path) -> Path:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        full_experiment_folder_path = output_dir / Path(experiment.get_study_name())
        if not os.path.exists(full_experiment_folder_path):
            os.makedirs(full_experiment_folder_path)
        return full_experiment_folder_path
    
    @staticmethod
    def save_info(experiment: Experiment, output_dir: Path, info: str):
        full_experiment_folder_path = RunnerSaver.ensure_folder(experiment, output_dir)
        with open(full_experiment_folder_path / Path(f"info.txt"), "a") as f:
            f.write(info + "\n")
            print(info)

    @staticmethod
    def save_results(experiment: Experiment, study: optuna.Study, output_dir: Path):
        full_experiment_folder_path = RunnerSaver.ensure_folder(experiment, output_dir)
        df = study.trials_dataframe()

        print(f"Study {experiment}")
        print("Study statistics: ")
        print("Best trial:", study.best_trial)
        print("Best score:", study.best_value)
        print("Best params:", study.best_params)

        df.to_csv(full_experiment_folder_path / Path(f"optuna_trials.csv"), index=False)
        experiment_dict = experiment.get_as_dict_serializable()
        experiment_df = pd.DataFrame([experiment_dict])
        experiment_df.to_csv(full_experiment_folder_path / Path(f"experiment_details.csv"), index=False)
        with open(full_experiment_folder_path / Path(f"experiment_details.json"), "w") as f:
            json.dump(experiment_dict, f, ensure_ascii=False, indent=2)
        best_param_distributions = study.best_trial.distributions
        best_param_distributions_norm = {k: str(v) for k, v in best_param_distributions.items()}

        merged_param_distributions = defaultdict(set)
        for trial in study.trials:
            for param_name, distribution in trial.distributions.items():
                merged_param_distributions[param_name].add(str(distribution))
        merged_param_distributions_norm = {k: list(v) for k, v in merged_param_distributions.items()}

        with open(full_experiment_folder_path / Path(f"best_trial_distributions.txt"), "w") as f:
            pprint(best_param_distributions_norm, stream=f)
        
        with open(full_experiment_folder_path / Path(f"merged_trial_distributions.txt"), "w") as f:
            pprint(merged_param_distributions_norm, stream=f)

    @staticmethod
    def save_evaluation_results(experiment: Experiment, y_labels: list, y_names: list[str], evaluation_metrics: dict, importances: pd.Series, confusion_matrix: np.ndarray, precision_recall_fscore_support_value:tuple, report: dict, output_dir: Path):
        full_experiment_folder_path = RunnerSaver.ensure_folder(experiment, output_dir)
        evaluation_df = pd.DataFrame([evaluation_metrics])
        evaluation_df.to_csv(full_experiment_folder_path / Path(f"evaluation_results.csv"), index=False)
        # todo: add umap plots for different colors
        # report as json
        json.dump(report, open(full_experiment_folder_path / Path(f"classification_report.json"), "w"), indent=4, ensure_ascii=False)

        # confusion matrix as plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        x = np.arange(len(y_labels))
        title = f"{experiment.get_study_name()}"
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=y_names, yticklabels=y_names, ax=axes[0])
        axes[0].xaxis.tick_top()
        axes[0].xaxis.set_label_position('top')
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")
        axes[0].set_title(f"Confusion Matrix ({title})")

        precision, recall, f1, _ = precision_recall_fscore_support_value
        axes[1].bar(x - 0.2, precision, width=0.2, label="Precision")
        axes[1].bar(x, recall, width=0.2, label="Recall")
        axes[1].bar(x + 0.2, f1, width=0.2, label="F1-score")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(y_names)
        axes[1].set_ylim(0, 1)
        axes[1].legend()
        axes[1].set_title(f"Per-class Metrics ({title})")
        plt.tight_layout()
        plt.savefig(full_experiment_folder_path / Path(f"cm_and_metrics.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

        pd.DataFrame(confusion_matrix).to_csv(full_experiment_folder_path / Path(f"confusion_matrix.csv"), index=False)
        importances.to_csv(full_experiment_folder_path / Path(f"feature_importances.csv"), index=True)