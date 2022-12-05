from datetime import datetime
import os
import sys
import time

import numpy as np
import pandas as pd

from sherlock import helpers
from sherlock.functional import extract_features_to_csv
from sherlock.features.paragraph_vectors import initialise_pretrained_model, initialise_nltk
from sherlock.features.preprocessing import (
    extract_features,
    convert_string_lists_to_lists,
    load_parquet_values,
)
from sherlock.features.word_embeddings import initialise_word_embeddings


data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'data')

print(f'Started at {datetime.now()}.')

if not os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'features', 'par_vec_trained_400.pkl.docvecs.vectors_docs.npy')):
    raise SystemExit(
        """
        Trained paragraph vectors do not exist,
        please run the '01-train-paragraph-vector-features' notebook before continuing
        """
    )

report_memory = False

timestr = time.strftime("%Y%m%d-%H%M%S")

# Features will be output to the following files
X_test_filename_csv = os.path.join(data_dir, 'processed', f'test_{timestr}.csv')
X_train_filename_csv = os.path.join(data_dir, 'processed', f'train_{timestr}.csv')
X_validation_filename_csv = os.path.join(data_dir, 'processed', f'validation_{timestr}.csv')

initialise_word_embeddings()
initialise_pretrained_model(400)
initialise_nltk()

values = load_parquet_values(os.path.join(data_dir, 'raw', 'test_values.parquet'))
extract_features_to_csv(X_test_filename_csv, values)
values = None
print(f'Finished at {datetime.now()}')

values = load_parquet_values(os.path.join(data_dir, 'raw', 'train_values.parquet'))
extract_features_to_csv(X_train_filename_csv, values)
values = None
print(f'Finished at {datetime.now()}')

values = load_parquet_values(os.path.join(data_dir, 'raw', 'val_values.parquet'))
extract_features_to_csv(X_validation_filename_csv, values)
values = None
print(f'Finished at {datetime.now()}')

start = datetime.now()
X_test = pd.read_csv(X_test_filename_csv, dtype=np.float32)
print(f'Load Features (test) process took {datetime.now() - start} seconds.')

start = datetime.now()
X_train = pd.read_csv(X_train_filename_csv, dtype=np.float32)
print(f'Load Features (train) process took {datetime.now() - start} seconds.')

start = datetime.now()
X_validation = pd.read_csv(X_validation_filename_csv, dtype=np.float32)
print(f'Load Features (validation) process took {datetime.now() - start} seconds.')

start = datetime.now()
train_columns_means = pd.DataFrame(X_train.mean()).transpose()
print(f'Transpose process took {datetime.now() - start} seconds.')

start = datetime.now()
X_train.fillna(train_columns_means.iloc[0], inplace=True)
X_validation.fillna(train_columns_means.iloc[0], inplace=True)
X_test.fillna(train_columns_means.iloc[0], inplace=True)
train_columns_means=None
print(f'FillNA process took {datetime.now() - start} seconds.')

start = datetime.now()
X_train.to_parquet(os.path.join(data_dir, 'processed', 'train.parquet'), engine='pyarrow', compression='snappy')
X_validation.to_parquet(os.path.join(data_dir, 'processed', 'validation.parquet'), engine='pyarrow', compression='snappy')
X_test.to_parquet(os.path.join(data_dir, 'processed', 'test.parquet'), engine='pyarrow', compression='snappy')
print(f'Save parquet process took {datetime.now() - start} seconds.')

print(f'Completed at {datetime.now()}.')
