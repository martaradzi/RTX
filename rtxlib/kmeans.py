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

        # feat_size = len(data[0]) - 2 # this assumes that the last two features are time index and number of cars
        number_of_samples = len(data)        
        labels = np.zeros(number_of_samples)
        
        random_centroids = np.random.choice(number_of_samples, self.k, replace = False)
        # init the centroids
        for index in range(self.k):
            self.centroids[index] = data[random_centroids[index]]
            # print(self.centroids[index])
            # print(data[random_centroids[index]])

        for iteration in range(self.max_iters):
            centroids_old = copy.deepcopy(self.centroids)
            distance_instance_to_clusters = np.zeros((number_of_samples, self.k))
            
            for instance in range(number_of_samples):
                for cluster in range(self.k):
                    distance_instance_to_clusters[instance, cluster] = euclidean_distance(data[instance], self.centroids[cluster])
                # choose closets cluster for that instance
                labels[instance] = np.argmin(distance_instance_to_clusters[instance])
            
            for i in range(self.k):
                self.clusters = data[labels==i]
                self.centroids[i] = np.mean(self.clusters, axis=0)
            
            if self.plot_iter:
                self.plot_iteration(iteration, data, labels)
            
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
#                 print(self.centroids)
#                 print(self.centroids[cluster])
                # print(instance)
#                 print(data[instance])
                # print(distance_instance_to_clusters[instance, cluster])
                distance_instance_to_clusters[instance, cluster] = euclidean_distance(data[instance], self.centroids[cluster])
            labels[instance] = np.argmin(distance_instance_to_clusters[instance])
        info("Predicted on data, returning labels", Fore.CYAN)
        return labels
    
    def partial_fit(self, data, plot_partial_fit=False):
        new_labels = np.zeros(len(data))
        for indx in range(len(data)):
            new_labels[indx] = self.fit_instance(data[indx])
#         self.labels = np.append(self.labels, new_labels)
        if plot_partial_fit:
            self.plot_partial_fit(data, new_labels)
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
    
        
    def plot_partial_fit(self, data, labels):
        
        figure, axs = plt.subplots(nrows=1, ncols=2,figsize=(14,4))                       

        axs[0].scatter(data[:,5], data[:,4], c=labels, cmap='rainbow', alpha=0.7)
        axs[0]..set_ylim(0, 750)
        axs[0].set_ylabel('Number of cars')
        axs[0].set_xlabel('Time')

        axs[1].scatter(data[:,1], data[:,3], c=labels, cmap='rainbow', alpha=0.7)
        axs[1].set_ylabel('3rd Quartile')
        axs[1].set_xlabel('1st Quartlie')
        for i in range(len(self.centroids)):
            axs[1].scatter(self.centroids[i][1], self.centroids[i][3], c='black', marker='x')

        figure.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.show(figure)
        plt.close(figure)
        return 
    
    def plot_iteration(self, iteration, data, labels):
        
        axes_labels = ['median_overhead', 'q1_overhead', 'q3_overhead', 'p9_overhead',]
        axes_feats = [  [(0,1),(0,3),(0,2)],
                        [(1,3),(1,2),(2,3)],
                    ]
        nrows = 2
        ncols = 3
        figure, axs = plt.subplots(nrows=nrows, ncols=ncols,figsize=(13,7))
        figure.suptitle(f'iteration {iteration}', fontsize= 16)
        for row in range(nrows):
            for col in range(ncols):
                feat = axes_feats[row][col]
                figure.tight_layout(rect=[0, 0.03, 1, 0.95])
                axs[row, col].scatter(data[:, feat[0]], data[:, feat[1]],  c=labels, cmap='rainbow', alpha=0.7)
                axs[row, col].set_ylabel(axes_labels[feat[1]])
                axs[row, col].set_xlabel(axes_labels[feat[0]])
                for i in range(len(self.centroids)):
                    axs[row, col].scatter(self.centroids[i][feat[0]], self.centroids[i][feat[1]], c='black', marker='x')
        plt.show(figure)
        plt.close(figure)
        
        return
        