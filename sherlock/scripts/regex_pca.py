import os
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# Load training data
data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'data')
X_train = pd.read_parquet(os.path.join(data_dir, 'processed', 'train.parquet'))
regex_features = [col for col in X_train if col.startswith('regex_')]
X_train_regex = X_train[regex_features]

# Perform PCA
pca = PCA(n_components=1000)
pca.fit(X_train_regex)

# Save the PCA transform
with open(os.path.join(os.path.dirname(__file__), '..', '..', 'features', 'regex_pca.pkl'), 'wb') as f:
    pickle.dump(pca, f)
