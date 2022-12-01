import os
from collections import OrderedDict

import hyperscan
import numpy as np

NUM_PATTERNS = 6925

db = None


def on_match(match_id, from_idx, to_idx, flags, context):
    (features,) = context
    feature_name = f'regex_{match_id}'
    features[feature_name] += 1


# Input: a single column in the form of a Python list
# Output: ordered dictionary holding bag of words features
def extract_regexes_features(col_values: list, features: OrderedDict, n_val):
    global db

    if not n_val:
        return

    if db is None:
        with open(os.path.join(os.path.dirname(__file__), 'hs.db'), 'rb') as f:
            db = hyperscan.loadb(f.read())

    for i in range(NUM_PATTERNS):
        features[f'regex_{i}'] = 0

    for (i, v) in enumerate(col_values[:n_val]):
        db.scan(str(v).encode('utf8'), match_event_handler=on_match, context=(features,))

    for i in range(NUM_PATTERNS):
        features[f'regex_{i}'] /= min(len(col_values), n_val)
