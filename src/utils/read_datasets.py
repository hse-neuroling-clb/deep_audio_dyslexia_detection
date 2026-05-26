import time
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from src.experiment import Experiment
from src.utils.cache import Cache
from src.embeddings import WhisperEmbedder

"""
This module is responsible for reading raw datasets, preprocessing them and computing embeddings.

"""

def read_datasets(experiment: Experiment, drop_samples_with_missing_embeddings: bool=True) -> pd.DataFrame:
    
    # read all datasets
    dfs = []
    result = []
    selected_datasets = [elem for elem in experiment.dataset_names]
    if experiment.test_dataset_names:
        for elem in experiment.test_dataset_names:
            if elem not in selected_datasets:
                selected_datasets.append(elem)
    
    for name in selected_datasets:
        base = Path("./data/raw")
        df = pd.read_csv(base / "scores" / f"{name}.csv")
        audio_dir = base / "audio" / name
        df["path"] = df["ID"].astype(str).apply(lambda x: str(audio_dir / f"{x}.wav"))
        df["task"] = name
        df["test_set"] = False
        if experiment.test_dataset_names:
            if name in experiment.test_dataset_names:
                df["test_set"] = True
        dfs.append(df)

    result = pd.concat(dfs, ignore_index=True)

    # remap old values to more readable

    # result[experiment.target_column] = result[experiment.target_column].map({0: 1, 1: 0.5, 2: 0})
    # human_readable_mapping = {0: "severe", 1: "moderate", 2: "fine"}
    result["orig_label1"] = result[experiment.target_column]

    if experiment.threshold_splits == 2:
        raise NotImplementedError("Threshold splits of 2 are not supported yet, and will not be implemented in the future. Please use threshold splits of 3.")
        result[experiment.target_column] = result[experiment.target_column].map({0: 0, 0.5: 1, 1: 1})
    

    cache = Cache(experiment.audio_encoder_type)

    embedder = WhisperEmbedder()
    embedder_func = embedder.get_embedding_func(experiment.audio_encoder_type)
    embeddings = []
    _t_start = time.time()

    errors = 0
    for i in tqdm(range(len(result))):
        path = result.iloc[i]["path"]
        if path in cache:
            embeddings.append(cache[path])
            continue
        try:
            embedding = embedder_func(path)
            cache[path] = embedding
            embeddings.append(embedding)
            continue
        except KeyboardInterrupt:
            raise
        except Exception as e:
            errors += 1
            body_text = f"Error processing file {path}"
            error_text = f"| error {e.__class__.__name__}: {e}"
            if "Error opening input files: No such file or directory" in str(e):
                body_text = f"File not found for path {path}"
                error_text = ""
            print(f"{body_text} {error_text}")
        embeddings.append([])
    print(f"Finished embedding with {errors} errors and {len(embeddings) - errors} successful embeddings (total {len(embeddings)}).")
    del cache
    del embedder_func
    embedder.cleanup()
    del embedder
    print(f"All embeddings took {time.time() - _t_start} seconds")

    result["embedding"] = embeddings
    dataset = result[["ID", "grade", "age", "gender", "embedding", "speed", "label1", "orig_label1", "task", "test_set"]]
    
    
    if drop_samples_with_missing_embeddings:
        dataset_clustering = dataset[dataset['embedding'].apply(lambda x: len(x) > 0)]
    else:
        dataset_clustering = dataset[dataset['embedding'].apply(lambda x: True)]
    return dataset_clustering