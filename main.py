from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import shutil
import time
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
# from utils.read_datasets import read_datasets
from src.experiment import Experiment, SmoothingClipUpperBoundType, SmoothingScaleType
from src.models.models import regressors
from src.runner import Runner
from src.utils.encoders import AudioEncoderType
from src.utils.runner_saver import RunnerSaver


ffmpeg_path = os.path.join(os.getcwd(), "ffmpeg.exe")
if os.path.exists(ffmpeg_path):
    os.environ["PATH"] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ.get("PATH", "")
    print(f"Using local FFmpeg: {ffmpeg_path}")
else:
    print(f"Warning: Local FFmpeg not found ({ffmpeg_path}), using system FFmpeg if available")


# Get all combinations of dataset names for training and testing
def get_dataset_combinations(train_items, test_items):
    x_combinations = []
    for r in range(1, len(train_items)+1):
        x_combinations.extend(combinations(train_items, r))

    train_and_test_dataset_names = []
    for elems in x_combinations:
        # for test_dataset_name in test_items:
        #     if test_dataset_name not in elems:
        #         train_and_test_dataset_names.append({"train": [elem for elem in elems], "test": [test_dataset_name]})
        train_and_test_dataset_names.append({"train": [elem for elem in elems], "test": None})

    print(f"Total combinations: {len(train_and_test_dataset_names)}")
    return train_and_test_dataset_names

# Run experiments for each combination of training and testing datasets

models = regressors
train_and_test_dataset_names = get_dataset_combinations(
    train_items=['zoo', 'trick', 'lobster'],
    test_items=['zoo', 'trick', 'lobster']
)





# ====================================================

def run_and_save_evaluation(experiment: Experiment, journal_path: str, output_folder: Path, n_jobs: int):
    evaluation_runner = Runner(experiment, -1, journal_path)
    y_labels = [0, 1]
    y_names = ["norm", "dyslexia"]
    metrics, importances, confusion_matrix, precision_recall_fscore_support_value, report = evaluation_runner.run_evaluation(y_labels=y_labels, y_names=y_names, n_jobs=n_jobs)
    RunnerSaver.save_evaluation_results(experiment, y_labels, y_names, metrics, importances, confusion_matrix, precision_recall_fscore_support_value, report, output_folder)

def save_optimization_results(experiment: Experiment, journal_path: str, output_folder: Path):
    study = Runner.get_study(experiment, journal_path)
    RunnerSaver.save_results(experiment, study, output_folder)


def run_optimization(experiment: Experiment, journal_path: str, n_trials: int, run_delay_seconds: int):
    optimization_runner = Runner(experiment, n_trials, journal_path, run_delay_seconds)
    return optimization_runner.run()

def run_optimization_parallel(experiment: Experiment, output_folder: Path, journal_path: str, n_trials: int, max_workers: int = 4, time_interval_seconds: int = 10):
    results = []
    trials_per_worker = n_trials // max_workers
    errors = []
    jobs_done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_optimization, experiment, journal_path, trials_per_worker, i*time_interval_seconds+1) for i in range(max_workers)]
        print(f"Submitted {max_workers} parallel optimization jobs, each with {trials_per_worker} trials.")
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                print(f"Error occurred while running trial: {e}")
                errors.append({"error": repr(e),"traceback": str(e.__traceback__)})
        jobs_done += 1
        print(f"Completed jobs: {jobs_done}/{max_workers}")
    if errors:
        RunnerSaver.save_info(experiment, output_folder, f"Completed with {len(errors)} errors during optimization run")
        for i, e in enumerate(errors):
            RunnerSaver.save_info(experiment, output_folder, f"{i}  - {experiment.get_parameters_str()}")
            RunnerSaver.save_info(experiment, output_folder, f"{i}  - {e["error"]}")
            RunnerSaver.save_info(experiment, output_folder, f"{i}  - Traceback: {e["traceback"]}")
            RunnerSaver.save_info(experiment, output_folder, f"=" * 40)
    return results

# sync experiments, parallel optimization
def run_experiment_single_parallel_optimization(experiment: Experiment, journal_path: str, n_trials: int, output_folder: Path, max_workers: int):
    study_name = experiment.get_study_name()
    print(f"[{study_name}] Starting optimization for experiment with {n_trials} trials")
    start_time = time.time()
    run_optimization_parallel(experiment=experiment, output_folder=output_folder, journal_path=journal_path, n_trials=n_trials, max_workers=max_workers, time_interval_seconds=5)
    print(f"[{study_name}] Optimization completed in {time.time() - start_time:.2f} seconds")

    print(f"[{study_name}] Saving optimization results for experiment")
    start_time = time.time()
    save_optimization_results(experiment=experiment, journal_path=journal_path, output_folder=output_folder)
    print(f"[{study_name}] Saving optimization results completed in {time.time() - start_time:.2f} seconds")

    print(f"[{study_name}] Running and saving evaluation for experiment")
    start_time = time.time()
    run_and_save_evaluation(experiment=experiment, journal_path=journal_path, output_folder=output_folder, n_jobs=max_workers)
    print(f"[{study_name}] Evaluation completed in {time.time() - start_time:.2f} seconds")


# parallel experiments, sync optimization
def run_experiment_single(experiment: Experiment, journal_path: str, n_trials: int, output_folder:Path, run_delay_seconds: int):
    run_optimization(experiment,journal_path, n_trials, run_delay_seconds)
    save_optimization_results(experiment, journal_path, output_folder)
    run_and_save_evaluation(experiment, journal_path, output_folder, n_jobs=2)

def run_experiments_parallel(experiments: list[Experiment], output_folder: Path, journal_path: str, n_trials: int, max_workers: int):
    results = []
    errors = []
    completed_experiments = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_experiment_single, _experiment, journal_path, n_trials, output_folder, i*10+1) for i, _experiment in enumerate(experiments)]

        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as e:
                print(f"Error occurred while running experiment: {e}")
                errors.append({"error": repr(e),"traceback": str(e.__traceback__)})
            completed_experiments += 1
            print(f"Completed experiments: {completed_experiments}/{len(experiments)}")
    if errors:
        print(f"Completed with {len(errors)} errors during running experiment")
        for i, e in enumerate(errors):
            print(f"{i}  - {experiment.get_parameters_str()}")
            print(f"{i}  - {e["error"]}")
            print(f"{i}  - Traceback: {e["traceback"]}")
            print("=" * 40)
    return results

# ====================================================


if __name__ == "__main__":
    experiments = []
    output_folder = Path("./new_output")

    optuna_journal = output_folder / "optuna_journal"
    optuna_journal.mkdir(parents=True, exist_ok=True)

    # if optuna_journal.exists():
    #     shutil.rmtree(optuna_journal)
    #     optuna_journal.mkdir(parents=True, exist_ok=True)

    journal_path = str(optuna_journal / Path('optuna_journal.log'))
    print(f"Optuna dashboard startup command: optuna-dashboard --storage-class JournalFileStorage {journal_path}")
    print("\n"*3)

    # train_and_test_dataset_names = [elem for elem in train_and_test_dataset_names if len(elem["train"]) > 1]
    train_and_test_dataset_names = [elem for elem in train_and_test_dataset_names if len(elem["train"]) > 1]
    print(f"Total valid train/test dataset combinations: {len(train_and_test_dataset_names)}:")
    print(f"{train_and_test_dataset_names}")
    print("\n"*3)

    for model in models:
        for dataset in train_and_test_dataset_names:
            train_dataset_names = dataset['train']
            test_dataset_name = dataset['test']
            audio_encoder_type = AudioEncoderType.whisper60s_mean.name
            smoothing_scale_type = SmoothingScaleType.LOWER_IS_BETTER
            group_cols=('grade', 'label1')
            for reduce_dims in [32]:# , 16, None]:
                for normalize_grade in [True, False][:1]:  # normalize_grade=True only
                    for calculate_thresholds in [False, True]: 
                        for smoothing_clip_upper_bound_type in SmoothingClipUpperBoundType:
                            experiment = Experiment(
                                model_info = model,
                                dataset_names=train_dataset_names,
                                group_cols=group_cols,
                                normalize_grade=normalize_grade,
                                test_dataset_names=test_dataset_name,
                                audio_encoder_type=audio_encoder_type,
                                smoothing_scale_type=smoothing_scale_type,
                                smoothing_clip_upper_bound_type=smoothing_clip_upper_bound_type,
                                calculate_thresholds=calculate_thresholds
                                )

                            experiments.append(experiment)
    
    print(f"Total experiments to run: {len(experiments)}")
    
    # quick
    # run_experiments_parallel(experiments, output_folder, journal_path, n_trials=20, max_workers=10)

    # or full
    for experiment in experiments:
        try:
            run_experiment_single_parallel_optimization(experiment, journal_path, n_trials=100, output_folder=output_folder, max_workers=15)  # previously was 14 workers
        except Exception as e:
            print(f"Error occurred while running experiment: {e}")
            print(f"Experiment parameters: {experiment.get_parameters_str()}")

# optuna-dashboard --storage-class JournalFileStorage .\optuna_journal.log
