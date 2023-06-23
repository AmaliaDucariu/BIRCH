# Import required libraries and modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import Birch
from datetime import datetime
from string import digits


# Function to eliminate spaces and make everything lowercase
def no_space_to_lower(arr):
    new_arr = []
    remove_digits = str.maketrans('', '', digits)
    for element in arr:
        element = element.lower()
        element = element.replace(" ", "")
        element = element.replace(",", "")
        element = element.replace(".", "")
        element = element.replace("/", "")
        element = element.replace("(", "")
        element = element.replace(")", "")
        element = element.replace("'", "")
        element = element.replace("|", "")
        element = element.replace("-", "")
        element = element.translate(remove_digits)
        new_arr.append(element)
    return new_arr


#Data Frame to Array
def data_frame_to_array(dataset):
    dataset = dataset.to_numpy()
    dataset = dataset.copy(order='C')
    return dataset


def birch(dataset):
    # Creating the BIRCH clustering model
    model = Birch(branching_factor=50, n_clusters=6, threshold=2.5)

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

# Importing dataset
dataset = pd.read_csv('C:\\Dizertation FINAL\\Data Sets\\food_ingredients_and_allergens.csv')

# Keep relevant columns

# Titles
food_titles = dataset[['Food_Product']].to_numpy()

dataset = dataset[['Main_Ingredient', 'Sweetener', 'Fat_Oil']]

# Replace null values
dataset.Main_Ingredient.fillna('', inplace=True)
dataset.Sweetener.fillna('', inplace=True)
dataset.Fat_Oil.fillna('', inplace=True)

# Define arrays
Main_Ingredient = np.array(dataset['Main_Ingredient'])
Sweetener = np.array(dataset['Sweetener'])
Fat_Oil = np.array(dataset['Fat_Oil'])


Main_Ingredient = no_space_to_lower(Main_Ingredient)
Sweetener = no_space_to_lower(Sweetener)
Fat_Oil = no_space_to_lower(Fat_Oil)

Main_Ingredient = pd.factorize(Main_Ingredient)[0]
Sweetener = pd.factorize(Sweetener)[0]
Fat_Oil = pd.factorize(Fat_Oil)[0]

# Define data frame
final_dataset = pd.DataFrame({'Main_Ingredient': Main_Ingredient, 'Sweetener': Sweetener, 'Fat_Oil': Fat_Oil})



final_dataset = data_frame_to_array(final_dataset)
pred = birch(final_dataset)
plot_3d(final_dataset, food_titles, pred)
end_time = datetime.now()
time_difference = end_time - start_time
print("TIME INTERVAL:", time_difference)