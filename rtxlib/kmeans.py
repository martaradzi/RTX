import numpy as np 
import matplotlib.pyplot as plt
from sklearn import metrics
import copy
from colorama import Fore
from rtxlib import info


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum(point1 - point2)**2)

def get_sil_score(data, labels):
    return metrics.silhouette_score(data, labels, metric='euclidean')

class kMeans:
    ''' Peform sequential k-Means clustering
    '''
    def __init__(self, k=2, max_iters=100, threshold=0, plot_iter=False):
        self.k = k
        self.max_iters = max_iters
        self.threshold = threshold
        self.counts = np.zeros(self.k)
        self.plot_iter = plot_iter
        self.centroids = [[] for _ in range(self.k)]
        
    def fit(self, data):

        number_of_samples = len(data)        
        labels = np.zeros(number_of_samples)
        
        random_centroids = np.random.choice(number_of_samples, self.k, replace = False)
        # init the centroids
        for index in range(self.k):
            self.centroids[index] = data[random_centroids[index]]

        for iteration in range(self.max_iters):
            centroids_old = copy.deepcopy(self.centroids)
            distance_instance_to_clusters = np.zeros((number_of_samples, self.k))
            # print(data[1])
            # print(self.centroids[1])
            
            for instance in range(number_of_samples):
                for cluster in range(self.k):
                    distance_instance_to_clusters[instance, cluster] = euclidean_distance(data[instance], self.centroids[cluster])
                # choose closets cluster for that instance
                labels[instance] = np.argmin(distance_instance_to_clusters[instance])
            
            for i in range(self.k):
                self.clusters = data[labels==i]
                self.centroids[i] = np.mean(self.clusters, axis=0)

            
            centroid_distances = [euclidean_distance(centroids_old[i], self.centroids[i]) for i in range(self.k)]
            
            if sum(centroid_distances) <= self.threshold:
                info("> Initial k-means clustering created", Fore.CYAN)
                # print(f'Converged somehow (?) at iter {iteration+1}')
                break
        return 
        
    def predict(self, data):
        distance_instance_to_clusters = np.zeros((len(data), self.k))
        labels = np.zeros(len(data))
        for instance in range(len(data)):
            for cluster in range(self.k):
                distance_instance_to_clusters[instance, cluster] = euclidean_distance(data[instance], self.centroids[cluster])
            labels[instance] = np.argmin(distance_instance_to_clusters[instance])
        info("Predicted on data, returning labels", Fore.CYAN)
        return labels
    
    def partial_fit(self, data, plot_partial_fit=False):
        new_labels = np.zeros(len(data))
        for indx in range(len(data)):
            new_labels[indx] = self.fit_instance(data[indx])
        info("> new data fitted to the model", Fore.CYAN)
    
    def fit_instance(self, instance):
        # print(instance)
        distance_instance_to_clusters = np.zeros(self.k)
        for i in range(self.k):
            distance_instance_to_clusters[i] = euclidean_distance(self.centroids[i], instance)
        label = np.argmin(distance_instance_to_clusters)
        self.counts[label] += 1
        self.centroids[label] = self.centroids[label] + (1 / self.counts[label]) * (instance - self.centroids[label])
        return label
