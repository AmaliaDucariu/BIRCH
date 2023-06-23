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


def getNumberArr(int):
    new_arr = []
    for i in range(int):
        new_arr.append(i)
    return new_arr

# Define start time of the process
start_time = datetime.now()

# Importing dataset
dataset = pd.read_csv('C:\\Dizertation FINAL\\Data Sets\\naukri_data_science_jobs_india.csv')
# Keep relevant columns

# Titles
job_titles = getNumberArr(12000)

dataset = dataset[['Job_Role', 'Company', 'Location']]

# Replace null values
dataset.Job_Role.fillna('', inplace=True)
dataset.Company.fillna('', inplace=True)
dataset.Location.fillna('', inplace=True)

# Define arrays
Job_Role = np.array(dataset['Job_Role'])
Company = np.array(dataset['Company'])
Location = np.array(dataset['Location'])


Job_Role = no_space_to_lower(Job_Role)
Company = no_space_to_lower(Company)
Location = no_space_to_lower(Location)

Job_Role = pd.factorize(Job_Role)[0]
Company = pd.factorize(Company)[0]
Location = pd.factorize(Location)[0]

# Define data frame
final_dataset = pd.DataFrame({'Job_Role': Job_Role, 'Company': Company, 'Location': Location})



final_dataset = data_frame_to_array(final_dataset)
pred = birch(final_dataset)
plot_3d(final_dataset, job_titles, pred)
end_time = datetime.now()
time_difference = end_time - start_time
print("TIME INTERVAL:", time_difference)