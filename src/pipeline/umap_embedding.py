import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import umap.umap_ as umap

class UmapEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, reduce_dims: int = 2, random_state: int = 42):
        self.reduce_dims = reduce_dims
        self.random_state = random_state
        self.embedding_columns = None

    def fit(self, data: pd.DataFrame, y=None):
        self.reducer = umap.UMAP(n_neighbors=8, min_dist=0.07, n_components=self.reduce_dims, random_state=self.random_state, n_jobs=1)
        self.embedding_columns = [col for col in data.columns if col.startswith("embedding")]
        self.reducer.fit(data[self.embedding_columns].to_numpy())
        return self
    
    @staticmethod
    def generate_umap_column_names(_reduce_dims: int):
        return [f"umap_{i}" for i in range(_reduce_dims)]

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        # print(f"[UMAP] before cols: {data.columns.tolist()}")
        embedding_values = data[self.embedding_columns].to_numpy()
        reduced_data = np.asarray(self.reducer.transform(embedding_values))

        # Replace all embedding columns with reduced UMAP features.
        data = data.drop(columns=self.embedding_columns).copy()
        for column_name in self.generate_umap_column_names(self.reduce_dims):
            data[column_name] = reduced_data[:, int(column_name.split('_')[1])]
        
        # print(f"[UMAP] reduced cols: {data.columns.tolist()}")
        
        return data
    