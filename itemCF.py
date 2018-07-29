
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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

movie_user_dict = train[["movieId", "userId"]].groupby('movieId')['userId'].apply(lambda x: x.tolist()).to_dict()
user_movie_dict = train[["movieId", "userId"]].groupby('userId')['movieId'].apply(lambda x: x.tolist()).to_dict()
movie_user_count = train[['userId', 'movieId', 'rating']].groupby('movieId').count().to_dict()
movie_user_count = movie_user_count['rating']

movie_common_rate = {}

for key, movieList in user_movie_dict.items():
    for i in range(len(movieList)):
        # this is slow
        print(i)
        for j in range(i+1, len(movieList)):
            movie_common_rate.setdefault(movieList[i], {})
            movie_common_rate.setdefault(movieList[j], {})
            movie_common_rate[movieList[i]].setdefault(movieList[j], 0)
            movie_common_rate[movieList[j]].setdefault(movieList[i], 0)
            movie_common_rate[movieList[i]][movieList[j]] += 1
            movie_common_rate[movieList[j]][movieList[i]] += 1

for key in movie_common_rate:
    for key2 in movie_common_rate[key]:
        movie_common_rate[key][key2] /= np.sqrt(movie_user_count[key] * movie_user_count[key2])



K = 5
N = 11
user = 22
result = {}

for movie in user_movie_dict[user]:
    temp_rating = train[(train.userId == user) & (train.movieId == movie)].rating.values[0]
    top_N = sorted(movie_common_rate[movie].items(), key = lambda x: x[1], reverse = True)[:K]
    for similar_movie, weight in top_N:
        result.setdefault(similar_movie, 0)
        result[similar_movie] += weight * temp_rating

recommended = sorted(result.items(), key = lambda x: x[1], reverse = True)[:N]
print("Top {} recommendation for user {}:".format(N, user))
for i, _ in recommended:
    print(movie_names[id_to_inx[i]])

# number 9 recommedation for user 22 is movie 1196
# it has highest score 4.5 rated by user 22 in the test dataset
# number 11 recommedation for user 22 is movie 1210
# it has highest score 5 rated by user 22 in the test dataset





