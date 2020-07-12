import torch as th
import pandas as pd
import numpy as np

def node_features(data_set):
    if data_set == 'ml-100k':
        users = pd.read_table('data/ml-100k/u.user', sep="|",
                              names=['user_id', 'age', 'sex', 'occupation', 'zip_code'],
                              encoding='latin-1', engine='python')
        movies = pd.read_table('data/ml-100k/u.item', engine='python', sep='|',
                               header=None, encoding='latin-1',
                               names=['movie_id', 'title', 'release_date', 'video_release_date',
                                      'IMDb_URL', 'unknown', 'Action', 'Adventure',
                                      'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
                                      'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                                      'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
        users['sex'] = pd.get_dummies(data=users['sex'], prefix='sex').iloc[:, 1]
        users['age'] = users['age'] / 50
        # 替换值
        map_dict_u = {}
        map_dict_v = {}
        for i in range(users.shape[0]):
            map_dict_u[users.user_id[i]] = i
        for j in range(movies.shape[0]):
            map_dict_v[movies.movie_id[j]] = j
        users.user_id = users.user_id.map(map_dict_u)
        movies.movie_id = movies.movie_id.map(map_dict_v)
        # feat
        temp_1 = users.iloc[:, 1:3].values
        temp_2 = pd.get_dummies(data=users['occupation'], prefix='occupation').iloc[:, 1:].values
        temp_3 = np.zeros((943, 18))
        x_u = th.Tensor(np.concatenate((temp_1, temp_2, temp_3), axis=1))
        temp_4 = np.zeros((1682, 22))
        temp_5 = movies.iloc[:, 6:].values
        x_v = th.Tensor(np.concatenate((temp_4, temp_5), axis=1))
        return x_u, x_v
    elif data_set == 'ml-1m':
        uname = ['user_id', 'gender', 'age', 'occupation', 'zip']
        users = pd.read_table('data/ml-1m/users.dat', sep='::', header=None, names=uname, engine='python')
        mnames = ['movie_id', 'title', 'genres']  # genres 表示影片的体裁是什么
        movies = pd.read_table('data/ml-1m/movies.dat', header=None, sep='::', names=mnames, engine='python')
        # 替换值
        map_dict_u = {}
        map_dict_v = {}
        for i in range(users.shape[0]):
            map_dict_u[users.user_id[i]] = i
        for j in range(movies.shape[0]):
            map_dict_v[movies.movie_id[j]] = j
        users.user_id = users.user_id.map(map_dict_u)
        movies.movie_id = movies.movie_id.map(map_dict_v)
        # feat
        temp_1 = np.eye(users.user_id.max())
        temp_2 = np.zeros((users.user_id.max(), movies.movie_id.max()))
        x_u = th.Tensor(np.concatenate((temp_1, temp_2), axis=1))
        temp_3 = np.zeros((movies.movie_id.max(), users.user_id.max()))
        temp_4 = np.eye(movies.movie_id.max())
        x_v = th.Tensor(np.concatenate((temp_3, temp_4), axis=1))
        return x_u, x_v
    else:
        rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
        ratings = pd.read_table('data/ml-10m/ml-10m100k/ratings.dat', header=None, sep='::', names=rnames,
                                engine='python')
        mnames = ['movie_id', 'title', 'genres']
        movies = pd.read_table('data/ml-10m/ml-10m100k/movies.dat', header=None, sep='::', names=mnames,
                               engine='python')
        users = ratings.user_id.unique()
        map_dict_u = {}
        map_dict_v = {}
        for i in range(users.shape[0]):
            map_dict_u[users[i]] = i
        for j in range(movies.shape[0]):
            map_dict_v[movies.movie_id[j]] = j
        movies.movie_id = movies.movie_id.map(map_dict_v)
        ratings.user_id = ratings.user_id.map(map_dict_u)
        ratings.movie_id = ratings.movie_id.map(map_dict_v)
        temp_1 = np.eye(users.shape[0])
        temp_2 = np.zeros((users.shape[0], movies.shape[0]))
        x_u = th.Tensor(np.concatenate((temp_1, temp_2), axis=1))
        temp_3 = np.zeros((movies.shape[0], users.shape[0]))
        temp_4 = np.eye(movies.shape[0])
        x_v = th.Tensor(np.concatenate((temp_3, temp_4), axis=1))
        return x_u, x_v

def load_feats(data_set):
    if data_set == 'ml-100k':
        users = pd.read_table('data/ml-100k/u.user', sep="|",
                              names=['user_id', 'age', 'sex', 'occupation', 'zip_code'],
                              encoding='latin-1', engine='python')
        movies = pd.read_table('data/ml-100k/u.item', engine='python', sep='|',
                               header=None, encoding='latin-1',
                               names=['movie_id', 'title', 'release_date', 'video_release_date',
                                      'IMDb_URL', 'unknown', 'Action', 'Adventure',
                                      'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
                                      'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                                      'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
        # 替换值
        map_dict_u = {}
        map_dict_v = {}
        for i in range(users.shape[0]):
            map_dict_u[users.user_id[i]] = i
        for j in range(movies.shape[0]):
            map_dict_v[movies.movie_id[j]] = j
        users.user_id = users.user_id.map(map_dict_u)
        movies.movie_id = movies.movie_id.map(map_dict_v)
        # train / test
        all_train_rating_info = pd.read_csv(
            'data/ml-100k/u1.base', sep='\t', header=None,
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            dtype={'user_id': np.int32, 'movie_id': np.int32,
                   'ratings': np.float32, 'timestamp': np.int64}, engine='python'
        )
        test_rating_info = pd.read_csv(
            'data/ml-100k/u1.test', sep='\t', header=None,
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            dtype={'user_id': np.int32, 'movie_id': np.int32,
                   'ratings': np.float32, 'timestamp': np.int64}, engine='python'
        )
        all_train_rating_info.user_id = all_train_rating_info.user_id.map(map_dict_u)
        all_train_rating_info.movie_id = all_train_rating_info.movie_id.map(map_dict_v)
        test_rating_info.user_id = test_rating_info.user_id.map(map_dict_u)
        test_rating_info.movie_id = test_rating_info.movie_id.map(map_dict_v)
        return users, movies, all_train_rating_info, test_rating_info
    elif data_set == 'ml-1m':
        uname = ['user_id', 'gender', 'age', 'occupation', 'zip']
        users = pd.read_table('data/ml-1m/users.dat', sep='::', header=None, names=uname, engine='python')
        mnames = ['movie_id', 'title', 'genres']  # genres 表示影片的体裁是什么
        movies = pd.read_table('data/ml-1m/movies.dat', header=None, sep='::', names=mnames, engine='python')
        rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
        ratings = pd.read_table('data/ml-1m/ratings.dat', header=None, sep='::', names=rnames, engine='python')
        # 替换值
        map_dict_u = {}
        map_dict_v = {}
        for i in range(users.shape[0]):
            map_dict_u[users.user_id[i]] = i
        for j in range(movies.shape[0]):
            map_dict_v[movies.movie_id[j]] = j
        users.user_id = users.user_id.map(map_dict_u)
        movies.movie_id = movies.movie_id.map(map_dict_v)
        ratings.user_id = ratings.user_id.map(map_dict_u)
        ratings.movie_id = ratings.movie_id.map(map_dict_v)
        # train / test
        all_train_rating_info = ratings.sample(frac=0.9, random_state=35)
        rowlist = []
        for indexs in all_train_rating_info.index:
            rowlist.append(indexs)
        test_rating_info = ratings.drop(rowlist, axis=0)
        return users, movies, all_train_rating_info, test_rating_info
    else:
        rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
        ratings = pd.read_table('data/ml-10m/ml-10m100k/ratings.dat', header=None, sep='::', names=rnames,
                                engine='python')
        mnames = ['movie_id', 'title', 'genres']
        movies = pd.read_table('data/ml-10m/ml-10m100k/movies.dat', header=None, sep='::', names=mnames,
                               engine='python')
        users = ratings.user_id.unique()
        map_dict_u = {}
        map_dict_v = {}
        for i in range(users.shape[0]):
            map_dict_u[users[i]] = i
        for j in range(movies.shape[0]):
            map_dict_v[movies.movie_id[j]] = j
        movies.movie_id = movies.movie_id.map(map_dict_v)
        ratings.user_id = ratings.user_id.map(map_dict_u)
        ratings.movie_id = ratings.movie_id.map(map_dict_v)
        all_train_rating_info = ratings.sample(frac=0.9, random_state=35)
        rowlist = []
        for indexs in all_train_rating_info.index:
            rowlist.append(indexs)
        test_rating_info = ratings.drop(rowlist, axis=0)
        return users, movies, all_train_rating_info, test_rating_info
