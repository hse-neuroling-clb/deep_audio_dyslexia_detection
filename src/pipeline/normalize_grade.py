from sklearn.base import BaseEstimator, TransformerMixin

class NormalizeGradeTransformer (BaseEstimator, TransformerMixin):
    def __init__(self, grade_column: str = "grade"):
        self.grade_column = grade_column
        self.max_grade_ = None

    def fit(self, X, y=None):
        self.max_grade_ = X[self.grade_column].max()
        return self

    def transform(self, data):
        # Normalize the grade column to a scale of 0 to 1
        data[self.grade_column] = data[self.grade_column] / self.max_grade_
        return data