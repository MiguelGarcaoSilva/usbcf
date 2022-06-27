from surprise import Dataset
from surprise import Reader
import numpy as np
import pandas as pd
import glob
import os
import random

# set dataset folder
os.environ["SURPRISE_DATA_FOLDER"] = '../../Datasets/'

# set RNG
np.random.seed(99)
random.seed(99)

data_path = os.path.join(os.environ["SURPRISE_DATA_FOLDER"] + "jester/",
                         "jester-data-1.xls")
df = pd.read_excel(data_path, header=None).iloc[:,  1:]
df = df.rename_axis('user').reset_index()
df = df.melt(id_vars=["user"])
df.columns = ['user', 'item', 'rating']
df["user"] = df["user"].apply(str)
df["item"] = df["item"].apply(str)
df = df[df.rating != 99]
data_jester = Dataset.load_from_df(df, reader=Reader(rating_scale=(-10, 10)))

data_jester = data_jester.build_full_trainset()
row_ind, col_ind, vals = [], [], []

for (u, i, r) in data_jester.all_ratings():
    row_ind.append(u)
    col_ind.append(i)
    if r == 0:
        r = 99
    vals.append(r)

rating_matrix_dense = np.empty([max(row_ind)+1, max(col_ind)+1])
rating_matrix_dense[row_ind, col_ind] = vals
rating_matrix_dense[rating_matrix_dense == 0] = np.nan
rating_matrix_dense[rating_matrix_dense == 99] = 0

# rating_matrix_coo = coo_matrix((vals, (row_ind, col_ind)),
#                                shape=(data_jester.n_users,
#                                       data_jester.n_items))
# rating_matrix_csr = csr_matrix((vals, (row_ind, col_ind)),
#                                shape=(data_jester.n_users,
#                                       data_jester.n_items))

# rating_matrix_dense = rating_matrix_csr.toarray()


pd.DataFrame(rating_matrix_dense).to_csv("../../Datasets/jester/jester_rating_matrix_data-1.csv",
                                         index_label="Index", sep= "\t")    

if not all([isinstance(value, int) or value.is_integer() for value in vals]):
    mask = (rating_matrix_dense >= 0)
    rating_matrix_dense_rounded = np.empty_like(rating_matrix_dense)
    rating_matrix_dense_rounded[mask] = np.floor(rating_matrix_dense[mask] + 0.5)
    rating_matrix_dense_rounded[~mask] = np.ceil(rating_matrix_dense[~mask] - 0.5)


pd.DataFrame(rating_matrix_dense_rounded)[:1000].to_csv("../../Datasets/jester/jester_rating_matrix_mini_discrete.csv",
                                         index_label="Index", sep= "\t")
pd.DataFrame(rating_matrix_dense_rounded).to_csv("../../Datasets/jester/jester_rating_matrix_full_discrete.csv",
                                         index_label="Index", sep= "\t")


# Confirmad, ele acha que o 0 é o missing value.
# TODO: mapear non missings noutro alfabeto