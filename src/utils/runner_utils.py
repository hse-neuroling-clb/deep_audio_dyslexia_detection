import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.utils import compute_class_weight

from src.experiment import Experiment

@dataclass
class TransformationResult:
    X: pd.DataFrame
    y: pd.Series
    base_y: pd.DataFrame  # used for cls validation

class SampleWeightHelper:
    @staticmethod
    def get_sample_weights_mapping(series: pd.Series) -> dict:
        classes = np.unique(series)
        assert classes.shape[0] < 30, "Too many unique classes for class weight computation. Check values. Use Cls labels instead of regression targets for class weight computation."
        # returns the weight of i-th class
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=series
        )
        class_weights_dict = {k: v for k, v in zip(classes, class_weights)}
        return class_weights_dict
    
    @staticmethod
    def get_sample_weights_series(series: pd.Series, weights_mapping: dict): # mb move to utils or something
        # assign weights to samples
        sample_weights = series.map(weights_mapping)
        assert sample_weights.isna().sum() == 0, "Sample weights contain NaN values. Check class weights computation."
        return sample_weights

def extract_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if df.get("test_set") is not None:
        test_df = df[df["test_set"] == True].copy()
        if test_df.empty:
            test_df = None
        df = df[df["test_set"] == False].copy()
    else:
        test_df = None
        df = df.copy()
    return df, test_df

def get_stratification_groups(df: pd.DataFrame, group_column_names: tuple[str, ...]):
    _df = []
    for group_column_name in group_column_names:
        if df.get(group_column_name) is None:
            raise ValueError(f"Group column {group_column_name} not found in dataframe")
    _df = df.copy()
    # group is concat of values of group columns
    _df["group"] = _df[list(group_column_names)].astype(str).agg("-".join, axis=1)

    # _df["group"] = list(map(tuple, _df[list(group_column_names)].values))
    groups_series = _df["group"]

    # hint_str = "-".join(group_column_names)
    # _unique_groups = groups_series.unique()
    # print(f"Unique groups for stratification({_unique_groups.shape[0]}) [{hint_str}]:")
    # print(_unique_groups)

    return groups_series  # group example (np.float64(0.5), np.float64(0.0), np.float64(0.0)


def get_xy(table: pd.DataFrame, experiment: Experiment) -> TransformationResult: #, umap.UMAP | None
    # todo: because of multiple transformations, we should ideally provide a column name as parameter (for orig_label1)

    # =======================================================================================
    base_X: pd.DataFrame = table[["ID", "embedding", "grade", "gender"]]
    base_y: pd.DataFrame = table[["ID", "grade", "gender", "task", "speed", "label1", "orig_label1"]]

    assert len(base_X) == len(base_y)

    embedding_matrix = np.vstack(base_X["embedding"].to_numpy())  # type: ignore
    embedding_columns = [f"embedding{i + 1}" for i in range(embedding_matrix.shape[1])]

    X_df = pd.DataFrame(embedding_matrix, columns=embedding_columns, index=base_X.index)
    X_df["grade"] = base_X["grade"].to_numpy()
    X_df["gender"] = base_X["gender"].to_numpy()
    X_df = X_df.reset_index(drop=True)

    y = base_y[experiment.target_column]  # 0..1 target
    y = y.reset_index(drop=True)
    base_y = base_y.reset_index(drop=True)

    return TransformationResult(X=X_df, y=y, base_y=base_y)
