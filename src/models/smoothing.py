import pandas as pd
from src.experiment import SmoothingClipUpperBoundType, SmoothingScaleType


# zoo
ZOO_GRADE_THRESHOLDS = {
    1: [13, 23],
    2: [37, 49],
    3: [50, 64],
    4: [66, 80]
}

# trick
TRICK_GRADE_THRESHOLDS = {
    1: [6, 18],
    2: [33, 47],
    3: [52, 67],
    4: [71, 84]
}

# lobster
LOBSTER_GRADE_THRESHOLDS = {
    1: [23, 30],
    2: [45, 54],
    3: [67, 75],
    4: [83, 87],
}

# trick
GRADE_THRESHOLDS = {
    "zoo": ZOO_GRADE_THRESHOLDS,
    "trick": TRICK_GRADE_THRESHOLDS,
    "lobster": LOBSTER_GRADE_THRESHOLDS
}

# todo: make it work inversely for lower is better

class SmoothingTransformer:
    scaled_pos_bad = (1.0, 0.75)
    scaled_pos_risk = (0.749999, 0.5)
    scaled_pos_norm = (0.499999, 0.0)
    # norm = 0-0.49     | mean          | mean-1std
    # risk = 0.5-0.74   | mean-1std     | mean-1.5std
    # dysl = 0.75-1     | mean-1.5std   | 0
    # or
    # dysl = 0-0.24     | mean-1.5std   | mean-1std
    # risk = 0.25-0.5   | mean-1std     | mean
    # norm = 0.5-1      | mean          | inf

    def __init__(
        self, 
        scale_type: SmoothingScaleType = SmoothingScaleType.HIGHER_IS_BETTER,
        calculate_thresholds: bool = False,
        smoothing_clip_upper_bound_type: SmoothingClipUpperBoundType = SmoothingClipUpperBoundType.FROM_FIT_FALLBACK_TO_CONST_160
    ):
        self.scale_type = scale_type
        if scale_type == SmoothingScaleType.HIGHER_IS_BETTER:
            raise NotImplementedError("Higher is better scale type is not implemented yet")
        self.calculate_thresholds = calculate_thresholds
        self.smoothing_clip_upper_bound_type = smoothing_clip_upper_bound_type
        self.thresholds_ = GRADE_THRESHOLDS
        self.upper_bounds_stratified = dict()
        self.upper_bound_unstratified = None
        self._fitted = False
        

    def _validate_input(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        required_columns = {"task", "grade", "speed"}
        if not required_columns.issubset(X.columns):
            missing_cols = required_columns - set(X.columns)
            raise ValueError(f"Input DataFrame is missing required columns: {missing_cols}")    

    def fit(self, X: pd.DataFrame, y=None):
        self._validate_input(X)
        self._fitted = True

        if self.calculate_thresholds:
            self.thresholds_ = dict()

        if self.smoothing_clip_upper_bound_type in {SmoothingClipUpperBoundType.FROM_FIT_UNSTRATIFIED, SmoothingClipUpperBoundType.FROM_FIT_FALLBACK_TO_UNSTRATIFIED}:
            self.upper_bound_unstratified = X["speed"].max()  # todo: or q=0.95?
        
        for task in X["task"].unique():
            for grade in X["grade"].unique():
                mask = (X["task"] == task) & (X["grade"] == grade)
                sub_df = X[mask]
                data = sub_df["speed"]

                if self.smoothing_clip_upper_bound_type == SmoothingClipUpperBoundType.FROM_FIT_FALLBACK_TO_CONST_160:
                    self.upper_bounds_stratified[(task, grade)] = data.max()

                if self.calculate_thresholds:
                    mean = data.mean()
                    std = data.std()
                    if task not in self.thresholds_:
                        self.thresholds_[task] = dict()
                    assert grade not in self.thresholds_[task]
                    self.thresholds_[task][grade] = [mean - 1.5 * std, mean - 1.0 * std]
        return self
    
    @staticmethod
    def _scale(value, x_min, x_max, scale_min_max):
        scale_min = scale_min_max[0]
        scale_max = scale_min_max[1]
        return scale_min + ((value - x_min) * (scale_max - scale_min)) / (x_max - x_min)
    
    def _get_upper_bound(self, _task, _grade):
        mapping = {
            SmoothingClipUpperBoundType.FROM_FIT_UNSTRATIFIED: self.upper_bound_unstratified,
            SmoothingClipUpperBoundType.CONST160: 160.0,
            SmoothingClipUpperBoundType.FROM_FIT_FALLBACK_TO_CONST_160: self.upper_bounds_stratified.get((_task, _grade), 160.0),
            SmoothingClipUpperBoundType.FROM_FIT_FALLBACK_TO_UNSTRATIFIED: self.upper_bounds_stratified.get((_task, _grade), self.upper_bound_unstratified)
        }
        return mapping.get(self.smoothing_clip_upper_bound_type)

    def transform(self, X: pd.DataFrame) -> pd.Series:
        if not self._fitted:
            raise ValueError("Smoother must be fitted before transforming y data.")
        self._validate_input(X)
        df = X.copy()
        df["scaled"] = [0.0] * len(df)

        for task in df["task"].unique():
            for grade in df["grade"].unique():
                mask = (df["task"] == task) & (df["grade"] == grade)
                sub_df = df[mask]
                data = sub_df["speed"]

                upper_bound = self._get_upper_bound(task, grade)
                if upper_bound is None:
                    raise ValueError(f"Upper bound is not defined for task {task} and grade {grade}. Check the fitting process and smoothing_clip_upper_bound_type ({self.smoothing_clip_upper_bound_type}).")
                    
                thresh = self.thresholds_[task][grade]
                pos_bad = (0, thresh[0])
                pos_risk = (thresh[0], thresh[1])
                pos_norm = (thresh[1], upper_bound)

                new_data = []
                for elem in data:
                    new_elem = elem
                    if elem <= pos_bad[0]:
                        new_elem = 1.0
                        # print(f"Elem {elem} < {pos_bad[0]} and becomes {new_elem}")
                    elif elem < pos_bad[1]:
                        new_elem = self._scale(elem, pos_bad[0], pos_bad[1], self.scaled_pos_bad)
                        # print(f"1.0 Elem {elem} falls in range [{pos_bad[0]:.1f}-{pos_bad[1]:.1f}] with scale 1.0-0.75 and becomes {new_elem:.2f}")
                    elif elem < pos_risk[1]:
                        new_elem = self._scale(elem, pos_risk[0], pos_risk[1], self.scaled_pos_risk)
                        # print(f"0.5 Elem {elem} falls in range [{pos_risk[0]:.1f}-{pos_risk[1]:.1f}] with scale 0.75-0.5 and becomes {new_elem:.2f}")
                    elif elem <= pos_norm[1]:
                        new_elem = self._scale(elem, pos_norm[0], pos_norm[1], self.scaled_pos_norm)
                        # print(f"0.0 Elem {elem} falls in range [{pos_norm[0]:.1f}-{pos_norm[1]:.1f}] with scale 0.5-0.0 and becomes {new_elem:.2f}")
                    else:
                        new_elem = 0.0
                        # print(f"Elem {elem} > {pos_norm[1]} and becomes {new_elem}")
                    new_data.append(new_elem)
                temp_df = sub_df.copy()
                temp_df["new"] = new_data
                df.loc[mask, "scaled"] = new_data

        # return df
        return df["scaled"]


