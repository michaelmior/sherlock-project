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




print(f'Started at {datetime.now()}.')

if not os.path.exists(os.path.join(os.path.dirname(__file__), '..', '..', 'features', 'par_vec_trained_400.pkl.docvecs.vectors_docs.npy')):
    raise SystemExit(
        """
        Trained paragraph vectors do not exist,
        please run the '01-train-paragraph-vector-features' notebook before continuing
        """
    )

# Calculate file paths
if len(sys.argv) != 2:
    sys.stderr.write(f'Usage: {sys.argv[0]} <split>\n')
    sys.exit(1)

split = sys.argv[1]
data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'data')
input_file = os.path.join(data_dir, 'raw', f'{split}_values.parquet')
output_csv = os.path.join(data_dir, 'processed', f'{split}.csv')
exit

initialise_word_embeddings()
initialise_pretrained_model(400)
initialise_nltk()

values = load_parquet_values(input_file)
extract_features_to_csv(output_csv, values)
print(f'Finished at {datetime.now()}')

