from ast import literal_eval
from collections import Counter
from datetime import datetime
import glob
import gzip
import json

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, classification_report

from sherlock.deploy.model import SherlockModel

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

print(f'Load data (train) process took {datetime.now() - start} seconds.')

start = datetime.now()
print(f'Started at {start}')

# X_validation = pd.read_parquet('../data/data/processed/validation.parquet/part-00000-6bc87672-ae10-4bcf-8ea7-faf5c368057e-c000.snappy.parquet')
X_validation = read_jsons('../data/data/processed/validation.json.gz')
y_validation = pd.read_parquet('../data/data/raw/val_labels.parquet').values.flatten()

y_validation = np.array([x.lower() for x in y_validation])

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
