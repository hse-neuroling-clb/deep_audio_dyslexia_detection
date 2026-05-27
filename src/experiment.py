from dataclasses import asdict, dataclass
import hashlib
from typing import Literal

from src.models.models import ModelInfo
from src.utils.encoders import AudioEncoderType
from enum import Enum


class SmoothingScaleType(Enum):
    NONE = "none"
    LOWER_IS_BETTER = "lower_is_better"
    HIGHER_IS_BETTER = "higher_is_better"

class SmoothingClipUpperBoundType(Enum):
    CONST160 = "const160"
    FROM_FIT_FALLBACK_TO_CONST_160 = "FROM_FIT_FALLBACK_TO_CONST_160"
    FROM_FIT_FALLBACK_TO_UNSTRATIFIED = "FROM_FIT_FALLBACK_TO_UNSTRATIFIED"
    FROM_FIT_UNSTRATIFIED = "FROM_FIT_UNSTRATIFIED"


@dataclass
class Experiment:
    model_info: ModelInfo
    dataset_names: tuple[str, ...] = ("zoo", "trick")  # datasets for training
    test_dataset_names: tuple[str, ...] | None = ("crayfish",) 
    target_column: Literal["label1", "speed"] = "label1"  # what to predict
    threshold_splits: Literal[2, 3] = 3  # 2 or 3 - number of classes to split dataset to (essentially, should we combine 1 and 2 classes or not)
    normalize_grade: bool = True  # make grades from 1,2,3,4 to 0.25,0.5,0.75,1.0
    # do not forget to use names instead of enums for multirun compat
    audio_encoder_type: str = AudioEncoderType.whisper60s_mean.name

    # LocalPreprocessingConfig:
    # scaler_type: Literal["standard", "minmax"] | None = "standard"
    # scaler_columns: tuple[str, ...] | None = None  # if None, then all columns will be scaled
    calculate_thresholds: bool = (
        False  # bool; False = use precomputed thresholds per class+text; True - calculate meanstd thresholds inplace. Doesn't work with smoothing_type=None
    )
    # balance_dataset: BalanceType = BalanceType.ALIGN_WITH_MINORITY  # bool; True to align with minority,
    # reducer_type: Literal["pca", "umap"] | None = None
    group_cols: tuple[str, ...] = ("grade", "label1")

    # CVConfig:
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42

    # ModelConfig:

    smoothing_scale_type: SmoothingScaleType = SmoothingScaleType.LOWER_IS_BETTER  # if set => 0.345, 0.768. None => 0, 0.5
    smoothing_clip_upper_bound_type: SmoothingClipUpperBoundType = SmoothingClipUpperBoundType.FROM_FIT_FALLBACK_TO_CONST_160
    reduce_dims: int | None = 32  # None means no dimensionality reduction, int - number of dimensions to reduce to.

    # todo: return plots to be done once per experiment
    # plot_clustering: bool = True
    # plot_strat_clustering: bool = True
    # plot_smoothing: bool = True

    def __post_init__(self):
        if self.reduce_dims is not None and not (isinstance(self.reduce_dims, int) and self.reduce_dims > 0):
            raise ValueError("reduce_dims must be either None or a positive integer")
        self.dataset_names = tuple(sorted(self.dataset_names))
        if self.test_dataset_names is not None:
            self.test_dataset_names = tuple(sorted(self.test_dataset_names))

    def get_parameters_str(self) -> str:
        # exp_dict = self.json()
        exp_dict = self.get_as_dict_serializable()
        params_str = "|".join(f"{key}={value}" for key, value in sorted(exp_dict.items(), key=lambda x: x[0]))
        return params_str
    
    def get_study_name(self) -> str:
        params_str = self.get_parameters_str()
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:10]
        model_name = self.model_info.name.replace(" ", "_").lower()
        dataset_name = "_".join(self.dataset_names)
        return f"{model_name}_{dataset_name}_{params_hash}"
    
    def get_as_dict_serializable(self) -> dict:
        exp_dict = asdict(self)
        for k, v in exp_dict.items():
            if isinstance(v, Enum):
                exp_dict[k] = v.value
            if isinstance(v, ModelInfo):
                exp_dict[k] = v.json()
        return exp_dict
    