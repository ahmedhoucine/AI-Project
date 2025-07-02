from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
from scipy import sparse
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import torch

# Load data
data = pd.read_csv("data.csv")

# Custom transformers
class SemanticEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] Using device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return self.model.encode(X, convert_to_numpy=True, batch_size=32, show_progress_bar=True)
    
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names, as_text=False):
        self.attribute_names = attribute_names
        self.as_text = as_text
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if len(self.attribute_names) == 1:
            if self.as_text:
                return X[self.attribute_names[0]].values.astype('U')
            return X[self.attribute_names[0]].values.reshape(-1, 1)
        return X[self.attribute_names].values

class NumericCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if X.shape[1] == 1:
            return pd.to_numeric(pd.Series(X[:, 0]), errors='coerce').values.reshape(-1, 1)
        else:
            return np.column_stack([
                pd.to_numeric(pd.Series(X[:, i]), errors='coerce').values
                for i in range(X.shape[1])
            ])

class DateTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        dates = pd.to_datetime(pd.Series(X[:, 0]), errors='coerce')
        return np.c_[
            dates.dt.year.values.reshape(-1, 1),
            dates.dt.month.values.reshape(-1, 1),
            dates.dt.day.values.reshape(-1, 1),
            dates.dt.dayofweek.values.reshape(-1, 1),
            dates.isna().values.reshape(-1, 1)
        ]

# Define features
potential_num_attribs = ['Number of Candidates', 'Number of Employees']
cat_attribs = ['Job Title', 'Location', 'Work Mode', 'Plateforme', 
               'Company Name', 'Sector', 'Contract Type', 'Education']

# Pipelines
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(potential_num_attribs)),
    ('numeric_cleaner', NumericCleaner()),
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('one_hot', OneHotEncoder(handle_unknown='ignore', sparse_output=True)),  # Keep sparse
])

date_pipeline = Pipeline([
    ('selector', DataFrameSelector(['Publishing Date'])),
    ('date_transformer', DateTransformer()),
])

text_pipeline = Pipeline([
    ('selector', DataFrameSelector(['Description'], as_text=True)),
    ('embedding', SemanticEmbedder(model_name='all-MiniLM-L6-v2')),
])

# Main pipeline with sparse imputer
full_pipeline = Pipeline([
    ('features', FeatureUnion([
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
        ("date_pipeline", date_pipeline),
        ("text_pipeline", text_pipeline),
    ])),
    ('final_imputer', SimpleImputer(strategy="constant", fill_value=0))  # Handle any remaining NaNs
])

# Transform data
data_prepared = full_pipeline.fit_transform(data)

sparse.save_npz("data_prepared_.npz", data_prepared)