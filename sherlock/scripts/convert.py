import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


CHUNK_SIZE = 100000

data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'data', 'processed')

# Calculate the mean
first = True
csv_file = os.path.join(data_dir, 'train.csv')
csv_stream = pd.read_csv(csv_file, chunksize=CHUNK_SIZE)
for chunk in tqdm(csv_stream):
    if first:
        mean_sum = chunk.sum()
        mean_count = len(chunk)

        # Convert all datatypes to float64
        # This is necessary since the first chunk may have 
        # all values 0/1 which has an inferred int type
        dtypes = {}
        for col in chunk.dtypes.keys():
            dtypes[col] = 'float64'

        # Prepare the Parquet schema
        parquet_schema = pa.Table.from_pandas(df=chunk.astype(dtypes)).schema
    else:
        # Update values necessary for calculating the mean
        mean_sum += chunk.sum()
        mean_count += len(chunk)

# Calculate the mean
columns_means = (mean_sum / mean_count).transpose()

# Save each chunk of the file
for split in ['train', 'test', 'validation']:
    # Prepare to load CSV chunks
    csv_file = os.path.join(data_dir, f'{split}.csv')
    csv_stream = pd.read_csv(csv_file, chunksize=CHUNK_SIZE)

    # Initialize a Parquet file reader
    parquet_file = os.path.join(data_dir, f'{split}.parquet')
    parquet_writer = pq.ParquetWriter(parquet_file, parquet_schema, compression='snappy')
    # Write each chunk to the ouput file
    for chunk in tqdm(csv_stream, total=mean_count / CHUNK_SIZE):
        chunk.fillna(columns_means.iloc[0], inplace=True)
        table = pa.Table.from_pandas(chunk, schema=parquet_schema)
        parquet_writer.write_table(table)

parquet_writer.close()
