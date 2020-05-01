import os

from colorama import Fore
from rtxlib import info, error, current_milli_time
from rtxlib.execution import clusteringExperimentFunction
from rtxlib.clustering_tools import run_model, partial_clustering, write_description

from sklearn.cluster import Birch
import numpy as np
import copy
# import wandb


def start_clustering_strategy(wf):
    """ executes all experiments from the definition file """

    info("> ExecStrategy   | BirchClustering", Fore.CYAN)
    wf.totalExperiments = wf.execution_strategy["sample_size"]
    
    start_time = current_milli_time()

    folder = wf.execution_strategy['save_in']
    os.makedirs(os.path.dirname(folder), exist_ok=True)

    sample_size = wf.execution_strategy["sample_size"]
    partial_clustering_size = wf.execution_strategy['partial_clustering_size']
    
    feature_array = [
        # 'avg_overhead',
        # 'std_overhead',
        # 'var_overhead',
        'median_overhead',
        'q1_overhead',
        'q3_overhead',
        'p9_overhead',
        ]

    # wandb.init(project='rtx-clustering', name="Two Features Run")
    birchModel = Birch(n_clusters=None, threshold=0.1)

    number_of_submodels_trained = 1

    data = []
    data_for_partial_clustering = []
    sample_number = 0

    while len(data) < sample_size:
        result, new_sample= clusteringExperimentFunction(sample_number, folder, wf, {
            "ignore_first_n_results": wf.execution_strategy['ignore_first_n_results'],
            "window_size": wf.execution_strategy['window_size_for_car_number_change'],
        })                    

        # number_of_trips_per_window.append(len(array_overheads))

        sample_number += 1
        data.append(new_sample)
        data_for_partial_clustering.append(new_sample)

        # run partial clustering when the specified number of samples was created 
        if len(data_for_partial_clustering) == partial_clustering_size:
            cpy_data = copy.deepcopy(data)
            partial_clustering(birchModel, cpy_data, data_for_partial_clustering, feature_array, folder, number_of_submodels_trained)
            data_for_partial_clustering = []
            number_of_submodels_trained += 1

    # at the end of the gathering process, if there is still data left for parial clustering, cluster it.
    if sample_number % partial_clustering_size != 0:
        cpy_data = copy.deepcopy(data)
        partial_clustering(birchModel, cpy_data, data_for_partial_clustering, feature_array, folder, number_of_submodels_trained)
        data_for_partial_clustering = []
        number_of_submodels_trained += 1
    
    # run the global clustering
    run_model(birchModel, data, 'final_global_', folder, True)
    
    
    duration = current_milli_time() - start_time

    write_description(duration, sample_size, partial_clustering_size, feature_array, folder, wf)

