# Import required libraries and modules
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import Birch
from datetime import datetime


#Importing datasets
dataset_wines = pd.read_csv('C:\\DMDW\\Birch Proiect\\winemag-data-130k-v2.csv')

#Keep relevant columns

#Titles and names
wine_titles = dataset_wines[['title']].to_numpy()

#Wines
dataset_wines = dataset_wines[['points', 'price']]



#Replace missing values with 0

dataset_wines.points.fillna(0, inplace=True)
dataset_wines.price.fillna(0, inplace=True)


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
dataset_wines = data_frame_to_array(dataset_wines)
pred = birch(dataset_wines)
plot_2d(dataset_wines, wine_titles, pred)
end_time = datetime.now()
time_difference = end_time - start_time
print("TIME INTERVAL:", time_difference)




