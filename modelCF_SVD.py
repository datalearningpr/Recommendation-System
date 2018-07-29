
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import sparse
from numpy import linalg
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

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


matrix = train.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
user_ratings_mean = np.mean(matrix, axis = 1)
matrix_demean = matrix.as_matrix() - np.vstack(user_ratings_mean)

U, sigma, VT = linalg.svd(matrix_demean)

N = 11
user = 22

prediction = np.dot(np.dot(U[:, :13], np.diag(sigma[:13])), VT[:13, :]) + np.vstack(user_ratings_mean)
user_movies = matrix.columns[np.argsort(prediction[user - 1,:])[::-1]]
recommended = [i for i in user_movies if i not in list(train[train.userId == user].movieId)][:N]

print("Top {} recommendation for user {}:".format(N, user))
for i in recommended:
    print(movie_names[id_to_inx[i]])

# the result is very good, top 4 movies in the test dataset all got recommended by the method

