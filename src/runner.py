from multiprocessing import Process
import os
import time
import uuid
import numpy as np
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from src.pipeline.normalize_grade import NormalizeGradeTransformer
from src.pipeline.umap_embedding import UmapEmbeddingTransformer
from src.experiment import Experiment
from utils.read_datasets import read_datasets
from src.utils.runner_utils import extract_test, get_stratification_groups, get_xy
from src.trainer import Trainer, evaluate



class Runner:
    def __init__(self, experiment: Experiment, n_trials: int, journal_path: str, run_delay_seconds: int = 10):
        self.experiment = experiment
        self.n_trials = n_trials
        self.journal_path = journal_path
        self.run_delay_seconds = run_delay_seconds
        self._id = uuid.uuid4().hex[:8]

    def _get_pipeline_columns(self, pipeline: Pipeline, initial_columns: list[str]) -> list[str]:
        X_names = initial_columns

        umap_step = pipeline.named_steps.get("umapEmbedding")
        if umap_step is None:
            return X_names
        
        keep_columns = [elem for elem in initial_columns if not elem.startswith("embedding")]
        new_columns = UmapEmbeddingTransformer.generate_umap_column_names(umap_step.reduce_dims)
        return keep_columns + new_columns  # todo: we assume the order
    
    def _get_pipeline(self):
        umap_step = None
        if self.experiment.reduce_dims is not None and self.experiment.reduce_dims > 0:
            umap_step = (
                 'umapEmbedding',
                 UmapEmbeddingTransformer(reduce_dims=self.experiment.reduce_dims, random_state=self.experiment.random_state)
                 )
            
        new_model = clone(self.experiment.model_info.model)

        normalize_grade_step = None
        if self.experiment.normalize_grade:
            normalize_grade_step = ('normalize_grade', NormalizeGradeTransformer())

        pipeline_steps = [
                umap_step,
                normalize_grade_step, 
                ('model', new_model)
        ]

        pipeline_steps = [step for step in pipeline_steps if step is not None]
        pipeline = Pipeline(pipeline_steps)
        return pipeline


    def get_objective(self, dataset):
            _experiment_name = self.experiment.get_study_name()
            def objective(trial: optuna.Trial):
                stratification_groups = get_stratification_groups(dataset, self.experiment.group_cols)
                groups = dataset["ID"]
                params = self.experiment.model_info.param_grid(trial)

                # groups for stratification (grade, gender, label1)
                cv = StratifiedGroupKFold(n_splits=self.experiment.n_splits, shuffle=True, random_state=self.experiment.random_state)
                # cv = GroupKFold(n_splits=self.experiment.n_splits, shuffle=True, random_state=self.experiment.random_state)

                transformation_result = get_xy(dataset, self.experiment)
                X, y, base_y = transformation_result.X, transformation_result.y, transformation_result.base_y

                splits = cv.split(X, stratification_groups, groups)
                # splits = cv.split(X, base_y, groups)

                # debug
                # cv = KFold(
                #     n_splits=self.experiment.n_splits,
                #     shuffle=True,
                #     random_state=self.experiment.random_state
                # )
                # splits = cv.split(X)
                #= ======

                trial_scores = []
                for fold_idx, (train_idx, val_idx) in enumerate(splits):
                    # print(f"[{_experiment_name}] Trial {trial.number}, Fold {fold_idx}")
                    X_train, X_val = X.iloc[train_idx].copy().reset_index(drop=True), X.iloc[val_idx].copy().reset_index(drop=True)
                    # y_train, y_val = y[train_idx], y[val_idx]
                    base_y_train, base_y_val = base_y.iloc[train_idx].copy().reset_index(drop=True), base_y.iloc[val_idx].copy().reset_index(drop=True)

                    model = self._get_pipeline()
                    
                    model.set_params(**params)
                    trainer = Trainer(self.experiment, model)
                    trainer_result = trainer.train(X_train, X_val, base_y_train, base_y_val)
                    
                    # gather results
                    results = {"fold": fold_idx, "train_size": len(train_idx), "val_size": len(val_idx)}
                    evaluated_trainer_metrics, *_  = evaluate(trainer_result)
                    # print(f"Trial {trial.number}, Fold {fold_idx}, Dataset sizes: X_train={len(X_train)}, X_val={len(X_val)}, metrics:\n{evaluated_trainer_metrics}")
                    results.update(evaluated_trainer_metrics)
                    trial_scores.append(results)

                    # set user attr for this iteration metrics
                    fold_results_df = pd.DataFrame([trial_scores[-1]])
                    for col in fold_results_df.columns:
                        trial.set_user_attr(f"{col}_fold_{fold_idx}", float(fold_results_df[col].iloc[0]))

                    current_r2_score = fold_results_df["r2_score"].iloc[0]
                    current_r2_score = float(current_r2_score)

                    trial.report(current_r2_score, step=fold_idx)

                    # if trial.should_prune():
                    #     raise optuna.exceptions.TrialPruned()
                
                # create a dataframe with mean and std for each collected metric across folds
                df = pd.DataFrame(trial_scores)
                means = dict()
                for col in df.columns:
                    means[f"total_{col}_mean"] = df[col].mean()
                    means[f"total_{col}_std"] = df[col].std()
                
                # set them as user attributes for the trial
                for k, v in means.items():
                    trial.set_user_attr(k, float(v))
                
                current_r2_score = df["r2_score"].mean() 
                return current_r2_score
            
            return objective
    
    @staticmethod
    def get_study(experiment: Experiment, journal_path: str) -> optuna.Study:
        storage = JournalStorage(JournalFileBackend(journal_path))
        study = optuna.create_study(
            direction="maximize",
            study_name=experiment.get_study_name(),
            storage=storage,
            load_if_exists=True,
        )
        return study

    def run(self):
        if self.run_delay_seconds > 0:
            print(f"Delaying run for {self.run_delay_seconds} seconds to avoid resource contention...")
            time.sleep(self.run_delay_seconds)
        df = read_datasets(self.experiment)
        df, test_df = extract_test(df)
        df = df.reset_index(drop=True)

        assert self.n_trials > 0, "n_trials must be greater than 0 for optimization run"

        objective = self.get_objective(dataset=df)

        study = self.get_study(self.experiment, self.journal_path)

        existing_user_attrs = study.user_attrs
        if existing_user_attrs.get("experiment_config") is None:
            experiment_info = self.experiment.get_as_dict_serializable()
            print(f"Setting experiment info as user attributes for the study: {experiment_info}")
            for k, v in experiment_info.items():
                study.set_user_attr(k, v)
            study.set_user_attr("experiment_config", self.experiment.get_parameters_str())
        study.optimize(objective, n_trials=self.n_trials, n_jobs=1)

    def run_evaluation(self, y_labels:list|None = None, y_names:list[str]|None = None, n_jobs: int = 2):
        """
        No multiprocess here
        """
        study = self.get_study(self.experiment, self.journal_path)
        best_params = study.best_params

        df = read_datasets(self.experiment)
        df, test_df = extract_test(df)
        df = df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True) if test_df is not None else None

        if test_df is None:
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=self.experiment.random_state)
        else:
            train_df = df
        
        transformation_result_train = get_xy(train_df, self.experiment)
        X_train, y_train, base_y_train = transformation_result_train.X, transformation_result_train.y, transformation_result_train.base_y

        transformation_result_test = get_xy(test_df, self.experiment)
        X_test, y_test, base_y_test = transformation_result_test.X, transformation_result_test.y, transformation_result_test.base_y

        assert len(X_train) == len(base_y_train)
        assert X_train.columns.tolist() == X_test.columns.tolist()

        model = self._get_pipeline()
        model.set_params(**best_params)
        trainer = Trainer(self.experiment, model)
        trainer_result = trainer.train(X_train, X_test, base_y_train, base_y_test)
        
        if y_labels is None:
            y_labels = [0, 1]
        if y_names is None:
            # y_names = ["norm", "dyslexia"]
            y_names = ["TD", "D"]
        evaluated_trainer_metrics, confusion_matrix, precision_recall_fscore_support_value, report = evaluate(trainer_result, y_labels, y_names)
        print(f"Evaluation on test set, results:\n{evaluated_trainer_metrics}")

        x_cols = self._get_pipeline_columns(model, X_train.columns.tolist())
        importances = self.experiment.model_info.importance_fn(model, x_cols, X_test, y_test, random_state=self.experiment.random_state, n_jobs=n_jobs)
        print(f"Feature importances:\n{importances}")
        return evaluated_trainer_metrics, importances, confusion_matrix, precision_recall_fscore_support_value, report

