
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import sparse
from numpy import linalg
from sklearn import preprocessing
from keras.layers import Input, Dense, Embedding, Dot, Flatten, Add, Concatenate, Dropout
from keras.models import Model
from keras.optimizers import Adam

ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")
movies.index = np.arange(1, len(movies) + 1)
# 9125
movie_names = movies["title"].to_dict()

idx_to_id = movies["movieId"].to_dict()
id_to_inx = {v: k for k, v in idx_to_id.items()}

# 671
users = ratings.userId.unique()
# 9066
movies_rated = ratings.movieId.unique()
movies_rated_index = [id_to_inx[i] for i in movies_rated]

# split train test dataset, make sure train contains all userId and movieId in test
train, test = train_test_split(ratings, test_size = 0.2, random_state = 2018)
extra_movieIds = list(set(test.movieId) - set(train.movieId))
temp = test[test.movieId.isin(extra_movieIds)]
train = pd.concat([train, temp])
test = test[~test.movieId.isin(extra_movieIds)]

# since embedding is used, need to make sure 
# both categorical features are starting from 0
le = preprocessing.LabelEncoder()

col1 = train.userId - 1
col2 = le.fit_transform(train.movieId)
values = np.array(train.rating)

test_col1 = test.userId - 1
test_col2 = le.transform(test.movieId)
test_values = np.array(test.rating)

n_factors = 50

user_in = Input(shape=(1,), dtype='int64', name='user_in')
u = Embedding(len(np.unique(ratings.userId)), n_factors, input_length=1,)(user_in)
movie_in = Input(shape=(1,), dtype='int64', name='movie_in')
m = Embedding(len(np.unique(ratings.movieId)), n_factors, input_length=1)(movie_in)

x = Concatenate()([u, m])
x = Flatten()(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.75)(x)
x = Dense(1)(x)

model = Model([user_in, movie_in], x)
model.compile(Adam(0.001), loss='mse')

model.fit([col1, col2], values, batch_size=64, epochs=7, 
          validation_data=([test_col1, test_col2], test_values))


# the beauty of NN method is the model can predict scores
# for any existing user on all the movies

N = 11
user = 22

watched = set(train[train.userId == user].movieId.unique())
all_movie = set(ratings.movieId.unique())
unwatched = all_movie - watched

predict_col1 = np.array([user - 1] * len(unwatched))
predict_col2 = le.transform(list(unwatched))

pred = model.predict([predict_col1, predict_col2])
top_pos = np.argsort(pred.flatten())[::-1][:N]
recommended = list(le.inverse_transform(predict_col2[top_pos]))

print("Top {} recommendation for user {}:".format(N, user))
for i in recommended:
    print(movie_names[id_to_inx[i]])
