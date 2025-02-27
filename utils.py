import pandas as pd
import numpy as np
from causallearn.graph.Edge import Edge
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Initialize a counter for variable IDs
next_var_id = 1

# create causal dataframe
def basic_causal_dataframe() -> pd.DataFrame:
    X = np.random.uniform(size=1000)
    eps = np.random.normal(size=1000)
    delta = np.random.uniform(size=1000)
    Y = -7 * X + 0.5 * delta
    Z = 2 * X + Y + eps

    # Create DataFrame with named variables
    return pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})

def cluster_solutions(count: dict):
    # create an array of probabilites maining the same order
    probabilities = []
    for key in count.keys():
        probabilities.append(count[key])
        
    # transform the array into a numpy array and reshape for k-means
    # K-means requires 2D array, even for 1D data, so reshape (-1, 1)
    probabilities = np.array(probabilities).reshape(-1, 1)
    
    kmeans = KMeans(n_clusters=2, random_state=0).fit(probabilities)
    
    kmeans_labels = kmeans.labels_
    
    # now we need to split the counts into two clusters, which are denoted by the labels
    cluster_1 = {}
    cluster_2 = {}
    for idx, key in enumerate(count.keys()):
        if kmeans_labels[idx] == 0:
            cluster_1[key] = count[key]
        else:
            cluster_2[key] = count[key]
            
    # now we need to return the take the clusters which has the mean value of the probabilities higher
    # than the other cluster
    mean_1 = np.mean(list(cluster_1.values()))
    mean_2 = np.mean(list(cluster_2.values()))
    
    if mean_1 > mean_2:
        return cluster_1
    elif mean_2 > mean_1:  
        return cluster_2
    else:
        raise Exception("Something went really wrong with the clustering of the solutions.... the means are equal")
    
def elbow_plot(counts: dict):
    # take all the values from the dictionary, and place them into a numpy array
    values = np.array(list(counts.values()))
    
    values.sort()
    
    plt.plot(values)
    plt.show()
    
def structural_check(cnf: list):
    for item in cnf:
        variables = set()
        for var in item:
            variables.add(abs(var))
        if len(item) != len(variables):
            raise Exception("There is something wrong with yout variables. Two variables are the same in a clause")