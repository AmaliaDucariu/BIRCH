# Import required libraries and modules
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import Birch
from datetime import datetime


#Importing datasets
dataset_movies = pd.read_csv('C:\\DMDW\\Birch Proiect\\tmdb_5000_movies.csv')

#Keep relevant columns

#Titles and names
movie_titles = dataset_movies[['original_title']].to_numpy()

#Movies
dataset_movies_2D = dataset_movies[['vote_count', 'popularity']]
dataset_movies_3D = dataset_movies[['vote_count', 'popularity', 'revenue']]


#Replace missing values with 0
dataset_movies_2D.vote_count.fillna(0, inplace=True)
dataset_movies_2D.popularity.fillna(0, inplace=True)

dataset_movies_3D.vote_count.fillna(0, inplace=True)
dataset_movies_3D.popularity.fillna(0, inplace=True)
dataset_movies_3D.revenue.fillna(0, inplace=True)



def data_frame_to_array(dataset):
    #Data Frame to Array
    dataset = dataset.to_numpy()
    dataset = dataset.copy(order='C')
    return dataset

def birch(dataset):
    # Creating the BIRCH clustering model
    model = Birch(branching_factor=50, n_clusters=5, threshold=2.5)

    # Fit the data (Training)
    model.fit(dataset)

    # Predict the same data
    pred = model.predict(dataset)
    return pred


def plot_2d(dataset, titles, pred):
    plt.scatter(dataset[:, 0], dataset[:, 1], c=pred, cmap='rainbow', alpha=0.7, edgecolors='b')

    plt.xlabel('X axis')
    plt.ylabel('Y axis')

    # for i, txt in enumerate(titles):
    #     plt.text(dataset[:, 0][i], dataset[:, 1][i], txt)

    plt.show()

def plot_3d(dataset, titles, pred):
    ax = plt.axes(projection='3d')
    ax.scatter3D(dataset[:, 0], dataset[:, 1], dataset[:, 2], c=pred, cmap='rainbow', alpha=0.7, edgecolors='b')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # for i, txt in enumerate(titles):
    #     ax.text3D(dataset[:, 0][i], dataset[:, 1][i], dataset[:, 2][i], txt)

    plt.show()


start_time = datetime.now()
dataset_movies_2D = data_frame_to_array(dataset_movies_2D)
pred = birch(dataset_movies_2D)
plot_2d(dataset_movies_2D, movie_titles, pred)
end_time = datetime.now()
time_difference = end_time - start_time
print("TIME INTERVAL:", time_difference)

# start_time = datetime.now()
# dataset_movies_3D = data_frame_to_array(dataset_movies_3D)
# pred = birch(dataset_movies_3D)
# plot_3d(dataset_movies_3D, movie_titles, pred)
# end_time = datetime.now()
# time_difference = end_time - start_time
# print("TIME INTERVAL:", time_difference)





