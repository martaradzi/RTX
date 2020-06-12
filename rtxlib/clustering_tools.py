from colorama import Fore
from rtxlib import info
from sklearn.cluster import Birch
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt
import numpy as np
import copy
import csv

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import wandb

def transfrom_to_nparray(data, feature_array):
    """ Transform the gathered data to numpy array to fit the model's requirements,
        only use the features spcified in feature array

        RETURNS: numpy.array 
    """
    transformed_data = []
    reshape = False # after many iteration, it might happen that there is only one data sample
    for row in data:
        # print(row)
        if isinstance(row, dict):
            t = ()
            for k, v in row.items():
                if k in feature_array:
                    t = t + (v,)
            transformed_data.append(t)
        else:
            reshape = True
    if reshape:
        data = np.array(data)
        return data.reshape(1, -1)
    else:
        return np.array(transformed_data)


@ignore_warnings(category=ConvergenceWarning)
def plot_silhouette_scores(model, test_data, n_clusters_min, n_clusters_max, folder, save_graph_name):
    """ Plot silhouette scores and return the best number of clusters
        
        RETURNS: int - most appropriate number of clusters
    """

    if len(model.subcluster_labels_) > 2:

        silhouette_scores = []

        clusters_range = range(n_clusters_min, n_clusters_max+1)
        results_dict = []
        # print(clusters_range)
        for number in clusters_range:
            # make a copy of the model so as not to mess up the 'correct' model
            model_cpy = model
            model_cpy.set_params(n_clusters=number)
            model_cpy.partial_fit()
            labels = model_cpy.predict(test_data)
            try: 
                s = metrics.silhouette_score(test_data, labels, metric='euclidean')
                silhouette_scores.append(s)
                results_dict.append((number, s))
            except ValueError: # not possible to calculate score for some values
                pass

        silhouette_range = [i[0] for i in results_dict]  
#         plt.plot(silhouette_range[:], silhouette_scores[:])
#         plt.xlabel('Number Of Clusers')
#         plt.ylabel('Silhouette Score')
#         plt.savefig(folder + 'silhouette_'+ save_graph_name +'.png')
        # plt.show()
#         plt.close()
        try: 
            max_score = max(silhouette_scores)
            for i in results_dict:
                if i[1] == max_score:
                    # print("The highest silhouette scores(" + str(max_score) + ") is for " + str(i[0]) + " clusers")
                    info("> Optimal number of clusters  | " + str(int(i[0])), Fore.CYAN)
                    info("> Silhouette Score     | " + str(max_score), Fore.CYAN)
                    wandb.log({f'Silhouette Score': max_score,
                              f'Number of clusters': int(i[0])})
                    return int(i[0])
                
        except ValueError:
            info("> Returned Minimal Number of Clusters  | " + str(n_clusters_min), Fore.CYAN)
            return n_clusters_min
    else:
        info("> Returned Minimal Number of Clusters  | " + str(n_clusters_min), Fore.CYAN)
        return n_clusters_min



def run_model(model, test_data, model_name, folder):
    """ Run the birch model """

    feature_array = list(test_data[0].keys())
    data_to_fit = transfrom_to_nparray(test_data, feature_array[3:])

    n_clusters = plot_silhouette_scores(model, data_to_fit, 2, 10, folder, ('global_fit_' + model_name))
    model.set_params(n_clusters = n_clusters)
    model.partial_fit()
    labels = model.predict(data_to_fit)

    for index in range(len(labels)):
        test_data[index].update({'label': labels[index]})
    feature_array.append('label')
    new_array = transfrom_to_nparray(test_data, feature_array)

    create_graphs(new_array, labels, folder, model_name)
    pca_plotting(new_array[:, 3:6], labels, folder, model_name)

    wandb.sklearn.plot_clusterer(model, new_array[:, 3:6], labels, labels=None, model_name=model_name)


    info("> New graphs were created", Fore.CYAN) 


def partial_clustering(model, data, data_for_clustering, feature_array,  folder, name):
    """ 
    This function will perform partial clustering, check if new sublcusters 
    were created in the process of clustering and plot the clusters in case there were

    """
    info("> ") 
    try:
        pre_number_of_subclusters = len(model.subcluster_labels_)
        pre_labels = model.subcluster_labels_
    except AttributeError: # in the first iteration of partial clustering, no data will be avaiable
        pre_number_of_subclusters = 0
        pre_labels = []

    data_for_clustering =  exclude_outliers_modified_z_score(data_for_clustering, feature_array[3:])
    model.partial_fit(data_for_clustering)
    current_number_of_subclusters =  len(model.subcluster_labels_)

    info("> Partial clustering performed", Fore.CYAN)    
    # check if number of sublusters is the same after partial clustering
    if pre_number_of_subclusters != current_number_of_subclusters:

        cpy_model = copy.deepcopy(model)
        run_model(cpy_model, data, (str(name) + '_partial_clustering_'), folder)

    # check if the labels before and after partial clustering are the same
    elif len(pre_labels) != 0:
        post_labels =  model.subcluster_labels_
        post_labels = post_labels[:len(pre_labels)]
        if not np.array_equal(np.array(pre_labels), np.array(post_labels)):
            cpy_model = copy.deepcopy(model)
            run_model(cpy_model, data, (str(name) + '_partial_clustering_'), folder)



def exclude_outliers_modified_z_score(partial_clustering_data, feature_array):
    """ Calculates the modified z-score for each of the samples for partial clustering
        and excludes the points that do exceed the modified z-score threshold

        RETURNS: numpy.array of the non-outlier samples
    """

    data_for_clustering = transfrom_to_nparray(partial_clustering_data, feature_array)
    excluded_index = []
    for i in [0, 1, 2, 3]:
        ys = data_for_clustering[:,i]
        median_y = np.median(ys)
        median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys])

        for y in range(len(data_for_clustering)):

            modified_z_score = 0.6745 * (data_for_clustering[y, i] - median_y) / median_absolute_deviation_y
            
            if np.abs(modified_z_score) > 3.5:
                excluded_index.append(y)

    info("> Number of outliers detected | " + str(len(excluded_index)), Fore.CYAN)
    data_for_clustering = np.delete(data_for_clustering, excluded_index, axis=0)
    return data_for_clustering



def write_raw_data(data, folder, features, header):
    """ Write the raw data of the experiment """

    with open(folder + 'raw_data.csv', 'a+') as f:
        try:
            writer = csv.DictWriter(f, fieldnames=features)
            if header:
                writer.writeheader()
            else:
                for dictionary in data:
                    writer.writerow(dictionary)
        except IndexError:
            # a sample with no data
            pass
    f.close()



def write_samples(sample, folder, feature_array, header):
    """ Write the data the model was trained on as well as the labels created after clustering"""
    
    with open(folder + 'data.csv', 'a+') as f:
        writer = csv.DictWriter(f, fieldnames=feature_array)
        if header:
            writer.writeheader()
        else:
            try: 
                writer.writerow(sample)
            except:
                pass
    f.close()



def write_description(duration, feature_array, folder, wf):
    ''' Writting the description file '''
    with open(folder + 'description.txt', 'w+') as f:
        f.write('The experiment took ' + str(duration/60000) + ' minutes to run as follows\n')
        f.write('Ignored results (in ticks): ' + str(wf.execution_strategy['ignore_first_n_ticks']) + '\n')
        f.write('Sample size: ' + str(wf.execution_strategy['sample_size']) + '\n')
        f.write('Samples gathered for: ' +str(wf.execution_strategy['ticks_per_sample']) + ' ticks\n')
        f.write('The partial clustering was performed on data of size: ' + str(wf.execution_strategy['partial_clustering_sample_size']) + '\n')
        f.write('Features the model trained on: \n')
        for feat in feature_array:
            f.write(str(feat) + '\n')
    f.close()



def create_graphs(new_array, labels, folder, model_name):
    '''plotting graphs for partial and global clustering for result inspection'''

    plt.scatter(new_array[:,0], new_array[:,1], c=labels, cmap='rainbow', alpha=0.7, edgecolors='b')
    plt.ylabel('Car Number')
    plt.xlabel('Index')
    plt.savefig(folder + model_name +'_car_over_time.png')
    wandb.log({f'Fit_{model_name}': plt})
    plt.close()
    
    plt.scatter(new_array[:,1], new_array[:,3], c=labels, cmap='rainbow', alpha=0.7, edgecolors='b')
    plt.ylabel('Overhead: Median')
    plt.xlabel('Car Number')
    plt.savefig(folder+ model_name +'_carVSmedian.png')
    plt.close()

    plt.scatter(new_array[:,1], new_array[:,4], c=labels, cmap='rainbow', alpha=0.7, edgecolors='b')
    plt.ylabel('Overhead: 1st Quartile')
    plt.xlabel('Car Number')
    plt.savefig(folder+ model_name +'_carVSq1.png')
    plt.close()

    plt.scatter(new_array[:,1], new_array[:,5], c=labels, cmap='rainbow', alpha=0.7, edgecolors='b')
    plt.ylabel('Overhead: 3rd Quartile')
    plt.xlabel('Car Number')
    plt.savefig(folder+ model_name +'_carVq3.png')
    plt.close()

    plt.scatter(new_array[:,1], new_array[:,6], c=labels, cmap='rainbow', alpha=0.7, edgecolors='b')
    plt.ylabel('Overhead: 90 Percentile')
    plt.xlabel('Car Number')
    plt.savefig(folder+ model_name +'_carVSp90.png')
    plt.close()

    plt.scatter(new_array[:,3], new_array[:,4], c=labels, cmap='rainbow', alpha=0.7, edgecolors='b')
    plt.ylabel('Overhead: 1st Quartile')
    plt.xlabel('Overhead: Median')
    plt.savefig(folder + model_name +'_q1VSmedian.png')
    plt.close()


    plt.scatter(new_array[:,5], new_array[:,3], c=labels, cmap='rainbow', alpha=0.7, edgecolors='b')
    plt.ylabel('Overhead: Median')
    plt.xlabel('Overhead: 3rd Quartile')
    plt.savefig(folder + model_name +'_medianVSq3.png')
    plt.close()

    plt.scatter(new_array[:,0], new_array[:,1], c=labels, cmap='rainbow', alpha=0.7, edgecolors='b')
    plt.ylabel('Number of cars')
    plt.xlabel('Time')
    plt.savefig(folder + model_name +'_time_cars.png')
    plt.close()


def pca_plotting(data, labels,  folder, model_name,):
    """ Plot the Principal Components instead of features """

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(data)

    plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c = labels, cmap='rainbow', alpha=0.7, edgecolors='b')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.savefig(folder + model_name +'_principal_components_plot.png')
    plt.close()
