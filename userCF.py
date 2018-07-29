
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
user_movie_count = train[['userId', 'movieId', 'rating']].groupby('userId').count().to_dict()
user_movie_count = user_movie_count['rating']

user_common_rate = {}

for key, userList in movie_user_dict.items():
    for i in range(len(userList)):
        for j in range(i+1, len(userList)):
            user_common_rate.setdefault(userList[i], {})
            user_common_rate.setdefault(userList[j], {})
            user_common_rate[userList[i]].setdefault(userList[j], 0)
            user_common_rate[userList[j]].setdefault(userList[i], 0)
            user_common_rate[userList[i]][userList[j]] += 1
            user_common_rate[userList[j]][userList[i]] += 1

for key in user_common_rate:
    for key2 in user_common_rate[key]:
        user_common_rate[key][key2] /= np.sqrt(user_movie_count[key] * user_movie_count[key2])



K = 5
N = 10
user = 22

top_N = sorted(user_common_rate[user].items(), key = lambda x: x[1], reverse = True)[:K]

result = {}
for similar_user, weight in top_N:
    for temp_movie in user_movie_dict[similar_user]:
        if temp_movie not in user_movie_dict[user]:
            result.setdefault(temp_movie, 0)
            result[temp_movie] += weight


recommended = sorted(result.items(), key = lambda x: x[1], reverse = True)[:N]
print("Top {} recommendation for user {}:".format(N, user))
for i, _ in recommended:
    print(movie_names[id_to_inx[i]])

# number 9 recommedation for user 22 is movie 648
# it has highest score 5 rated by user 22 in the test dataset

