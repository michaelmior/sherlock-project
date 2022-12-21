from ast import literal_eval
from collections import Counter
from datetime import datetime
import os

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, classification_report

from sherlock.deploy.model import SherlockModel

data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'data')

start = datetime.now()
print(f'Started load data (train) at {start}')

X_train = pd.read_parquet(os.path.join(data_dir, 'processed', 'train.parquet'))
y_train = pd.read_parquet(os.path.join(data_dir, 'raw', 'train_labels.parquet')).values.flatten()

y_train = np.array([x.lower() for x in y_train])

print(f'Load data (train) process took {datetime.now() - start} seconds.')

print('Distinct types for columns in the Dataframe (should be all float32):')
print(set(X_train.dtypes))

start = datetime.now()
print(f'Started load data (validation) at {start}')

X_validation = pd.read_parquet(os.path.join(data_dir, 'processed', 'validation.parquet'))
y_validation = pd.read_parquet(os.path.join(data_dir, 'raw', 'validation_labels.parquet')).values.flatten()

y_validation = np.array([x.lower() for x in y_validation])

print(f'Load data (validation) process took {datetime.now() - start} seconds.')

start = datetime.now()
print(f'Started load data (test) at {start}')

X_test = pd.read_parquet(os.path.join(data_dir, 'processed', 'test.parquet'))
y_test = pd.read_parquet(os.path.join(data_dir, 'raw', 'test_labels.parquet')).values.flatten()

y_test = np.array([x.lower() for x in y_test])

print(f'Finished at {datetime.now()}, took {datetime.now() - start} seconds')

model_id = "retrained_sherlock"

start = datetime.now()
print(f'Started training at {start}')

model = SherlockModel()
# Model will be stored with ID `model_id`
model.fit(X_train, y_train, X_validation, y_validation, model_id=model_id)

print('Trained and saved new model.')
print(f'Finished at {datetime.now()}, took {datetime.now() - start} seconds')

model.store_weights(model_id=model_id)
