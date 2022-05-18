from ast import literal_eval
from collections import Counter, OrderedDict
from datetime import datetime
import glob
import gzip
import json
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

from sherlock.deploy.model import SherlockModel
from sherlock.deploy import helpers

model_id = 'retrained_sherlock'

from ast import literal_eval
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, classification_report

from sherlock.deploy.model import SherlockModel

def read_jsons(filename):
    return pd.read_json(gzip.open(filename), orient='records', lines=True)

start = datetime.now()
print(f'Started at {start}')

X_train = (read_jsons(f) for f in glob.glob('../data/data/processed/train.json/*.json.gz'))
y_train = pd.read_parquet('../data/data/raw/train_labels.parquet').values.flatten()
y_train = np.array([x.lower() for x in y_train])
encoder = LabelEncoder()
encoder.fit(y_train)

print(f'Load data (train) process took {datetime.now() - start} seconds.')

start = datetime.now()
print(f'Started at {start}')

# X_validation = pd.read_parquet('../data/data/processed/validation.parquet/part-00000-6bc87672-ae10-4bcf-8ea7-faf5c368057e-c000.snappy.parquet')
# X_validation = read_jsons('../data/data/processed/validation.json.gz')
y_validation = pd.read_parquet('../data/data/raw/val_labels.parquet').values.flatten()
y_validation = np.array([x.lower() for x in y_validation])
y_val_int = encoder.transform(y_validation)
y_val_cat = tf.keras.utils.to_categorical(y_val_int)
y_validation = None
y_val_int = None

feature_cols = helpers.categorize_features()
def array_from_json():
    global y_validation

    start = 0
    for f in glob.glob('../data/data/processed/validation.json/*.json.gz'):
        # yield from (list(json.JSONDecoder(object_pairs_hook=OrderedDict).decode(line.decode('utf8')).values()) for line in gzip.open(f))
        batch = [[], [], [], []]
        for line in gzip.open(f):
            obj = json.loads(line.decode('utf8'))
            batch[0].append([obj[f] for f in feature_cols["char"]])
            batch[1].append([obj[f] for f in feature_cols["word"]])
            batch[2].append([obj[f] for f in feature_cols["par"]])
                # [obj[f] for f in feature_cols["regex"]],
            batch[3].append([obj[f] for f in feature_cols["rest"]])
            start += 1
        # yield (batch, y_validation[start:start + len(batch)])
        yield (tuple(tf.convert_to_tensor(x) for x in batch), y_val_cat[start:start + len(batch[0])])

# def validation_gen():
#     yield from zip(array_from_json(), y_validation)

# print(next(zip(array_from_json(), y_validation))[0][0].shape)
# print(next(zip(array_from_json(), y_validation))[0][1].shape)
# print(next(zip(array_from_json(), y_validation))[0][2].shape)
# print(next(zip(array_from_json(), y_validation))[0][3].shape)
# print(next(zip(array_from_json(), y_validation))[0][4].shape)
# (960,)
# (201,)
# (400,)
# (6925,)
# (27,)
# sys.exit(1)

# for a in array_from_json():
#     print(tf.ragged.constant(a[0]).shape)

X_validation = tf.data.Dataset.from_generator(array_from_json, output_signature=(
    (
        tf.TensorSpec(shape=(None, 960), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 201), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 400), dtype=tf.float32),
        # tf.TensorSpec(shape=(6925,), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 27), dtype=tf.float32),
    ),
    tf.TensorSpec(shape=(None, 78), dtype=tf.int32)
))
# print(next(array_from_json())[1])
# print(list(X_validation.take(1)))
# sys.exit(1)


print(f'Load data (validation) process took {datetime.now() - start} seconds.')

# start = datetime.now()
# print(f'Started at {start}')

# X_test = pd.read_parquet('../data/data/processed/test.parquet/part-00000-bb70dba1-d554-407e-8e0b-b5d21c35a3fa-c000.snappy.parquet')
# y_test = pd.read_parquet('../data/data/raw/test_labels.parquet').values.flatten()

# y_test = np.array([x.lower() for x in y_test])

# print(f'Finished at {datetime.now()}, took {datetime.now() - start} seconds')

start = datetime.now()
print(f'Started at {start}')

model = SherlockModel()
# Model will be stored with ID `model_id`
model.fit(X_train, y_train, X_validation, y_validation, model_id=model_id)

print('Trained and saved new model.')
print(f'Finished at {datetime.now()}, took {datetime.now() - start} seconds')

model.store_weights(model_id=model_id)
