import papermill as pm
from tqdm import tqdm
import os
from common import AudioEncoderType


def make_title(DATASET_NAMES, TEST_DATASET_NAME, AUDIO_ENCODER_TYPE, THRESHOLD_SPLITS, BALANCE_DATASET, RUN_NUMBER=0,
               **kwargs):
    str_kwargs = ""
    if kwargs:
        temp_kwargs = []
        for k, v in kwargs.items():
            temp_kwargs.append(f"{k}={v}".replace(" ", "_"))
        str_kwargs = "_".join(temp_kwargs)

    DATASET_NAMES = list(sorted(DATASET_NAMES))
    _test_set_name = "subset" if TEST_DATASET_NAME is None else TEST_DATASET_NAME
    return "+".join(
        DATASET_NAMES) + "_on-" + _test_set_name + "_enc-" + AUDIO_ENCODER_TYPE + f"_splits-{THRESHOLD_SPLITS}" + f"_balance-{int(BALANCE_DATASET)}__{RUN_NUMBER}" + str_kwargs


def dataset_search():
    threshold_splits_opts = [2, 3]
    balance_dataset_opts = [True, False]
    dataset_opts = [
        [["zoo", "trick"], "lobster"],
        [["zoo", "trick", "lobster"], None],
        [["zoo", "trick"], None],
        [["trick", "lobster"], "zoo"],
    ]

    for threshold_splits in threshold_splits_opts:
        for balance_dataset in balance_dataset_opts:
            for dataset_train, dataset_test in dataset_opts:
                parameters = dict(
                    DATASET_NAMES=list(sorted(dataset_train)),
                    TEST_DATASET_NAME=dataset_test,
                    # set to None to use subset of dataset or set to another dataset name
                    THRESHOLD_SPLITS=threshold_splits,  # 2 or 3 - number of classes to predict
                    AUDIO_ENCODER_TYPE=AudioEncoderType.whisper30s,
                    BALANCE_DATASET=balance_dataset,  # or True to align with minority
                    PLOT_CLUSTERING=False,
                    PLOT_STRAT_CLUSTERING=True
                )
                yield parameters


def split_validation():
    DATASET_NAMES = ["zoo", "trick", "lobster"]
    TEST_DATASET_FRACTION = 0.1
    MULTIFOLD_TEST = True
    MULTIFOLD_NUM_FOLDS = round(1 / TEST_DATASET_FRACTION)

    for MULTIFOLD_ORDINAL in range(MULTIFOLD_NUM_FOLDS):
        parameters = {
            "DATASET_NAMES": DATASET_NAMES,
            "TEST_DATASET_NAME": None,
            "TEST_DATASET_FRACTION": TEST_DATASET_FRACTION,
            "MULTIFOLD_TEST": MULTIFOLD_TEST,
            "MULTIFOLD_ORDINAL": MULTIFOLD_ORDINAL,
            "THRESHOLD_SPLITS": 3,
            "BALANCE_DATASET": True,
            "AUDIO_ENCODER_TYPE": AudioEncoderType.whisper60s_mean.name,
        }
        yield parameters


if __name__ == '__main__':
    runs = 0
    configurations = list(split_validation())
    print(f"Total runs: {len(configurations)}")
    for parameters in configurations:
        runs += 1
        print(f"run {runs}")
        print(parameters)
        title = make_title(**parameters)
        output_folder = f"./output/{title}"
        os.makedirs(output_folder, exist_ok=True)
        try:

            pm.execute_notebook(
                './whisper_flow.ipynb',
                os.path.join(output_folder, "whisper_flow_sub.ipynb"),
                parameters=parameters
            )
        except Exception as e:
            if "#exists" in str(e):
                print(e)
            else:
                print("aboba")
                raise
