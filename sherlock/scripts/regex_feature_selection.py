import os

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest


# Load training data
data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'data')
X_train = pd.read_parquet(os.path.join(data_dir, 'processed', 'train.parquet'))
y_train = pd.read_parquet(os.path.join(data_dir, 'raw', 'train_labels.parquet')).values.flatten()
regex_features = [col for col in X_train if col.startswith('regex_')]
X_train_regex = X_train[regex_features]

# Pick the top features
kbest = SelectKBest(k=1000)
kbest.fit(X_train_regex, y_train)

# Sort by score and take the best
scores = list((i, s) for (i, s) in enumerate(kbest.scores_) if not np.isnan(s))
scores.sort(key=lambda x: -x[1])
scores = scores[:1000]

# Output the desired features for later
with open(os.path.join(os.path.dirname(__file__), '..', '..', 'features', 'feature_column_identifiers', 'regex_col.tsv'), 'w') as f:
    for i in range(1000):
        f.write(f'{i}\tregex_{scores[i][0]}\n')
