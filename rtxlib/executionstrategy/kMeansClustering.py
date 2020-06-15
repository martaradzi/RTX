import os

from colorama import Fore
from rtxlib import info, error, current_milli_time
from rtxlib.execution import clusteringExperimentFunction
from rtxlib.clustering_tools import exclude_outliers_modified_z_score, transfrom_to_nparray, write_description, write_raw_data, write_samples, create_graphs
from rtxlib.kmeans import kMeans
from sklearn.cluster import Birch
import numpy as np
import copy
import csv
#import wandb


def start_k_menas_clustering_strategy(wf):
    """ executes all experiments from the definition file """

    info("> ExecStrategy   | SequentialKMeans Clustering", Fore.CYAN)
    wf.totalExperiments = wf.execution_strategy["sample_size"]
    
    start_time = current_milli_time()

    folder = wf.execution_strategy['save_in']
    os.makedirs(os.path.dirname(folder), exist_ok=True)

    sample_size = wf.execution_strategy["sample_size"]
    partial_clustering_size = wf.execution_strategy['partial_clustering_sample_size']
    
    feature_array = [
        'index',
        'totalCarNumber',
        'numberOfTrips',
        'median_overhead',
        'q1_overhead',
        'q3_overhead',
        'p9_overhead',
        ]
        
    features_for_raw_data = [
        # 'tick',
        'startCarNumber',
        'totalCarNumber',
        'overhead',
        'duration',
    ]

    #wandb.init(project='sequential_k-means_clustering', name="test_run_1")
    kmeansModel = kMeans()

    number_of_submodels_trained = 1

    data = []
    data_for_partial_clustering = []
    data_for_clutering = []
    sample_number = 0

    write_raw_data(None, folder,features_for_raw_data, header=True)
    write_samples(None, folder, feature_array, header = True)

    while sample_number < sample_size:
        result, new_sample= clusteringExperimentFunction(sample_number, folder, wf, {
            "ignore_first_n_results": wf.execution_strategy['ignore_first_n_ticks'],
            "window_size": wf.execution_strategy['ticks_per_sample'],
        })                    

        if new_sample is not None:
            sample_number += 1
            data_for_partial_clustering.append(new_sample)
            write_samples(new_sample, folder, feature_array, False)

            data.append(new_sample)

            # run partial clustering when the specified number of samples was created 
            if len(data_for_partial_clustering) == partial_clustering_size:
                data_for_partial_clustering = exclude_outliers_modified_z_score(data_for_partial_clustering, feature_array[3:])
                data_for_clutering = np.concat(data_for_clutering, data_for_partial_clustering, axis=0)
                if number_of_submodels_trained == 1:
                    kmeansModel.fit(data_for_partial_clustering)
                else:
                    kmeansModel.partial_fit(data_for_partial_clustering)

                data_for_partial_clustering = []                
                number_of_submodels_trained += 1
                
                copy_data = copy.deepcopy(data_for_clutering)
                data_for_clutering = exclude_outliers_modified_z_score(cpy_data, feature_array[3:])
                labels = kmeansModel.predict(data_for_clutering)
                plt.scatter(data_for_clutering[:,0], data_for_clutering[:,1], c=labels, cmap='rainbow', alpha=0.7, edgecolors='b')
                plt.ylabel('Median')
                plt.xlabel('90th percentile')
                plt.savefig(folder + number_of_submodels_trained +'_med_90thpercentile.png')
                #wandb.log({f'Fit_{number_of_submodels_trained}': plt})
                plt.close()
            
    # # at the end of the gathering process, if there is still data left for parial clustering, cluster it.
    if sample_number % partial_clustering_size != 0:
        cpy_data = copy.deepcopy(data)
        data_for_clutering = exclude_outliers_modified_z_score(cpy_data, feature_array[3:])
        if number_of_submodels_trained == 1:
            kmeansModel.fit(data_for_clutering)
        else:
            kmeansModel.partial_fit(data_for_clutering)
        data_for_partial_clustering = []
        number_of_submodels_trained += 1
    
    # # run the global clustering
    data = transfrom_to_nparray(data, feature_array[3:])
    labels = kmeansModel.predict(data)
    with open(folder+'labels.txt', 'w') as f:
        for label in labels:
            f.write(str(label))
            f.write(',\n')
    f.close()
    # duration = current_milli_time() - start_time

