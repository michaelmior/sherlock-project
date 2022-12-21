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
    prepare_feature_extraction,
    load_parquet_values,
)
from sherlock.features.word_embeddings import initialise_word_embeddings


timestr = time.strftime("%Y%m%d-%H%M%S")

# Features will be output to the following files
X_test_filename_csv = f'../data/data/processed/test_{timestr}.csv'
X_train_filename_csv = f'../data/data/processed/train_{timestr}.csv'
X_validation_filename_csv = f'../data/data/processed/validation_{timestr}.csv'

prepare_feature_extraction()
initialise_word_embeddings()
initialise_pretrained_model(400)
initialise_nltk()

values = load_parquet_values("../data/data/raw/test_values.parquet")
extract_features_to_csv(X_test_filename_csv, values)

values = load_parquet_values("../data/data/raw/train_values.parquet")
extract_features_to_csv(X_train_filename_csv, values)

values = load_parquet_values("../data/data/raw/validation_values.parquet")
extract_features_to_csv(X_validation_filename_csv, values)
