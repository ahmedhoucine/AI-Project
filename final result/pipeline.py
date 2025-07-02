from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import torch
from scipy import sparse

# Load data
data = pd.read_csv("data.csv")

# Drop noisy or unused columns
data = data.drop(columns=["ID", "Scrap Date", "scrap_date_parsed", "Duration", "Location"], errors='ignore')

# --- Custom Transformers ---
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

# --- Feature groups ---
num_attribs = ['Number of Candidates', 'Number of Employees']
cat_attribs = ['Job Title', 'Work Mode', 'Plateforme', 'Company Name', 'Sector', "Salary", 'Contract Type', 'Education']
text_attrib = ['Description']

# --- Pipelines ---
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('cleaner', NumericCleaner()),
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True)),
])

text_pipeline = Pipeline([
    ('selector', DataFrameSelector(text_attrib, as_text=True)),
    ('embedder', SemanticEmbedder(model_name='all-MiniLM-L6-v2')),
    ('svd', TruncatedSVD(n_components=50, random_state=42)),
])

# --- Combine all features ---
full_pipeline = Pipeline([
    ('features', FeatureUnion([
        ('num', num_pipeline),
        ('cat', cat_pipeline),
        ('text', text_pipeline)
    ])),
    ('final_imputer', SimpleImputer(strategy='constant', fill_value=0))
])

# --- Transform and Save ---
print("[INFO] Transforming data...")
X_transformed = full_pipeline.fit_transform(data)

print("[INFO] Saving transformed data...")
sparse.save_npz("data_prepared_final.npz", sparse.csr_matrix(X_transformed))
print("[SUCCESS] Transformed data saved to 'data_prepared_final.npz'")
