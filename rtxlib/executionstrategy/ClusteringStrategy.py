from colorama import Fore

from rtxlib import info, error, current_milli_time
from rtxlib.execution import transfrom_to_nparray, clusteringExperimentFunction, run_model, partial_clustering
import os

from sklearn.cluster import Birch
import numpy as np
import csv
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
    # number_of_submodels_trained = 0
    
    feature_array = [
        # 'avg_overhead', \
        # 'std_overhead', \
        # 'var_overhead', \
        'median_overhead', \
        'q1_overhead', \
        'q3_overhead', \
        'p9_overhead'
        ]

    # create a birch model threshold = 1
    # wandb.init(project='rtx-clustering', name="Two Features Run")
    birchModel = Birch(n_clusters=None, threshold=0.1)

    number_of_submodels_trained = 1

    data = []
    data_for_partial_clustering = []
    sample_number = 1

    number_of_trips_per_window = []

    while len(data) < sample_size:
        result, new_sample, array_overheads = clusteringExperimentFunction(sample_number, folder, wf, {
            "ignore_first_n_results": wf.execution_strategy['ignore_first_n_results'],
            "window_size": wf.execution_strategy['window_size_for_car_number_change'],
        })                    

            
        number_of_trips_per_window.append(len(array_overheads))

        sample_number += 1
        data.append(new_sample)
        data_for_partial_clustering.append(new_sample)


        if len(data_for_partial_clustering) == partial_clustering_size:
            cpy_data = copy.deepcopy(data)
            partial_clustering(birchModel, cpy_data, data_for_partial_clustering, feature_array, folder, number_of_submodels_trained)
            # data_numpy = transfrom_to_nparray(data_cpy,feature_array)
            # birchModel.partial_fit(data_numpy)
            data_for_partial_clustering = []
            number_of_submodels_trained += 1
    
    run_model(birchModel, data, 'final_global_', folder, True)
    duration = current_milli_time() - start_time

    with open(folder + 'description.txt', 'w+') as f:
        f.write('The experiment took ' + str(duration/60000) + 'minutes to run as follows\n')
        f.write('Ignored results (in ticks): ' + str(wf.execution_strategy['ignore_first_n_results']) + '\n')
        f.write('Sample size: ' + str(sample_size) + '\n')
        f.write('Samples gathered for: ' +str(wf.execution_strategy['window_size_for_car_number_change']) + ' ticks\n')
        f.write('The partial clustering was performed on data of size: ' + str(partial_clustering_size) + '\n')
        f.write('Avg number of trips per windwo: ' + str(np.int_(np.average(number_of_trips_per_window))) + '\n\n')
        f.write('Features the model trained on: \n')
        for feat in list(data[0].keys())[1:-1]:
            f.write(str(feat) + '\n')
    f.close()
