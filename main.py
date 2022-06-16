from flask import Flask,jsonify,make_response
from flask import Flask, render_template, request
from flask_restful import Resource, Api, reqparse
app = Flask(__name__)
api = Api(app)
@app.route('/get_classification/', methods=['GET', 'POST'])
def result():
        if request.method == 'POST':
            movieId = request.form.get('seed')
        if request.method == 'GET':
            movieId = request.args.get('seed')
        movieId = int(movieId)
        import pandas as pd
        import sklearn
        import numpy as np
        import seaborn as sns
        from scipy.sparse import csr_matrix
        df = pd.read_csv("movies.csv")
        print(df)
        dg = pd.read_csv("ratings.csv")
        print(dg)
        print(df.info())
        print(dg.info())
        print(len(dg['userId'].unique()))
        print(len(dg['movieId'].unique()))
        a = len(dg['userId'].unique())
        b = len(dg['movieId'].unique())
        import matplotlib.pyplot as plot
        plot.bar(['userId', 'movieId'], [a, b])
        print(type(a))
        print(type(b))
        plot.bar(['userId', 'movieId'], [a, b])
        plot.title("User Id vs Movie Id")
        plot.xlabel("Ids")
        plot.ylabel("Total count")
        ratings_per_user = dg.groupby('userId')['rating'].count()
        print(ratings_per_user)
        ratings_per_movie = dg.groupby('movieId')['rating'].count()
        print(ratings_per_movie)
        print(type(ratings_per_movie))
        ratings_per_user = ratings_per_user.to_frame().reset_index()
        print(ratings_per_user)
        ratings_per_movie = ratings_per_movie.to_frame().reset_index()
        print(ratings_per_movie)
        dup_bool = df.duplicated()
        print(dup_bool)
        print(df)
        dups = sum(dup_bool)
        print("There are {} duplicate rating entries in the data..".format(dups))

        def create_matrix(df):
            N = len(df['userId'].unique())
            M = len(df['movieId'].unique())

            # map Ids to indices
            user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
            movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))

            # map indices to ids
            user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
            movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["movieId"])))

            user_index = [user_mapper[i] for i in df['userId']]
            movie_index = [movie_mapper[i] for i in df['movieId']]

            X = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))

            return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

        X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix(dg)
        print(X)
        from sklearn.neighbors import NearestNeighbors
        def find_similar_movies(movieId, X, k, metric='cosine', show_distance=False):
            neighbour_ids = []
            movie_ind = movie_mapper[movieId]
            movie_vec = X[movie_ind]
            k += 1
            kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
            kNN.fit(X)
            movie_vec = movie_vec.reshape(1, -1)
            neighbour = kNN.kneighbors(movie_vec, return_distance=show_distance)
            for i in range(0, k):
                n = neighbour.item(i)
                neighbour_ids.append(movie_inv_mapper[n])
            neighbour_ids.pop(0)
            return neighbour_ids



        similar_ids = find_similar_movies(movieId, X, k=10)

        movie_title = df[df['movieId'] == movieId]['title'].values[0]

        print(f"Since you watched {movie_title}")
        sv = []
        for i in similar_ids:
            sv.append(df.loc[i]['title'])

        return ({'recommendations': sv})
if __name__ == '__main__':
    app.run(debug=True)





