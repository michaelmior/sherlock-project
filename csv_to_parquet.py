from datetime import datetime
import pickle

import numpy as np
import pandas as pd


X_train_filename_csv = '../data/data/processed/train_20220515-193553.csv'
X_validation_filename_csv = '../data/data/processed/validation_20220515-193553.csv'
X_test_filename_csv = '../data/data/processed/test_20220515-193553.csv'


# start = datetime.now()
# X_test = pd.read_csv(X_test_filename_csv, dtype=np.float32)
# print(f'Load Features (test) process took {datetime.now() - start} seconds.')

start = datetime.now()
X_train = pd.read_csv(X_train_filename_csv, dtype=np.float32)
print(f'Load Features (train) process took {datetime.now() - start} seconds.')

# start = datetime.now()
# X_validation = pd.read_csv(X_validation_filename_csv, dtype=np.float32)
# print(f'Load Features (validation) process took {datetime.now() - start} seconds.')

start = datetime.now()
train_columns_means = pd.DataFrame(X_train.mean()).transpose()
pickle.dump(train_columns_means, open('train_columns_means.pkl', 'wb'))
print(f'Transpose process took {datetime.now() - start} seconds.')


start = datetime.now()

X_train.fillna(train_columns_means.iloc[0], inplace=True)
# X_validation.fillna(train_columns_means.iloc[0], inplace=True)
# X_test.fillna(train_columns_means.iloc[0], inplace=True)

train_columns_means=None

print(f'FillNA process took {datetime.now() - start} seconds.')


start = datetime.now()
X_train.to_parquet('../data/data/processed/train.parquet', engine='pyarrow', compression='snappy')
# X_validation.to_parquet('../data/data/processed/validation.parquet', engine='pyarrow', compression='snappy')
# X_test.to_parquet('../data/data/processed/test.parquet', engine='pyarrow', compression='snappy')
print(f'Save parquet process took {datetime.now() - start} seconds.')

print(f'Completed at {datetime.now()}.')
