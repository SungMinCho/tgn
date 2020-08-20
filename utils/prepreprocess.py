import pandas as pd

import argparse
from pathlib import Path
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, help='Path to data root folder', default='/SSD/sungmincho/Data')
parser.add_argument('--data', type=str, help='Dataset name', default='ml-1m')
# parser.add_argument('--filename', type=str, default='ratings.csv')
# parser.add_argument('--sep', type=str, help='separator', default=',')
# parser.add_argument('--header', type=str, help='header name', default='infer')

args = parser.parse_args()

args.filename = {
    'ml-1m': 'ratings.dat',
    'ml-20m': 'ratings.csv',
    'beauty': 'ratings.csv',
    'game': 'ratings.csv',
}[args.data]

args.sep = {
    'ml-1m': '::',
    'ml-20m': ',',
    'beauty': ',',
    'game': ',',
}[args.data]

args.header = {
    'ml-1m': None,
    'ml-20m': 'infer',
    'beauty': None,
    'game': None,
}[args.data]

df_path = Path(args.root).joinpath(args.data).joinpath(args.filename)
print('Loading file', df_path)
df = pd.read_csv(df_path, sep=args.sep, header=args.header)
print(df.head())
df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
df = df[['user_id', 'item_id', 'timestamp']]
df['state_label'] = 0
df['comma_separated_list_of_features'] = 0

dense_uid_map = {k: v for v, k in enumerate(set(df.user_id))}
dense_iid_map = {k: v for v, k in enumerate(set(df.item_id))}

df.user_id = df.user_id.map(dense_uid_map.get)
df.item_id = df.item_id.map(dense_iid_map.get)
df = df.sort_values(by='timestamp')
df.timestamp = df.timestamp - df.timestamp.min()

print()
print(df.head())

save_path = f'data/{args.data}.csv'
print('Saving file to', save_path)
df.to_csv(save_path, index=False)

with open(f'data/{args.data}_dense_uid_map', 'wb') as f:
    pickle.dump(dense_uid_map, f)

with open(f'data/{args.data}_dense_iid_map', 'wb') as f:
    pickle.dump(dense_iid_map, f)
