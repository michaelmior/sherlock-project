import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

from sherlock.deploy.model import SherlockModel

model = SherlockModel()
model.initialize_model_from_json(True, 'retrained_sherlock')

data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'data')
X_train = pd.read_parquet(os.path.join(data_dir, 'processed', 'train.parquet'))

encoder = LabelEncoder()
y_train = pd.read_parquet(os.path.join(data_dir, 'raw', 'train_labels.parquet')).values.flatten()
encoder.fit(y_train)
y_train_int = encoder.transform(y_train)

y_pred = model.predict_proba(X_train, 'retrained_sherlock')
y_pred_int = np.argmax(y_pred, axis=1)

cf = tf.math.confusion_matrix(y_train_int, y_pred_int).numpy()
models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'model_files')
classes = np.load(open(os.path.join(models_dir, 'classes_retrained_sherlock.npy'), 'rb'))

with open(os.path.join(models_dir, 'retrained_sherlock_confusion.txt'), 'w') as f:
    f.write(f'actual, predicted, count\n')
    for (i, c1) in enumerate(classes):
        for (j, c2) in enumerate(classes):
            f.write(f'{c1}, {c2}, {cf[i][j]}\n')
