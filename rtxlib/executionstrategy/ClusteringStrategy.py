from colorama import Fore

from rtxlib import info, error
from rtxlib.execution import clusteringExperimentFunction, plot_silhouette_scores, run_model, transfrom_to_nparray
import os

from sklearn.cluster import Birch
import numpy as np
# import wandb


def start_clustering_strategy(wf):
    """ executes all experiments from the definition file """
    info("> ExecStrategy   | Clustering", Fore.CYAN)
    wf.totalExperiments = len(wf.execution_strategy["knobs"])
    
    folder = wf.execution_strategy['save_in']
    os.makedirs(os.path.dirname(folder), exist_ok=True)

    sample_size = wf.execution_strategy["sample_size"]
    partial_clustering_size = wf.execution_strategy['partial_clustering_size']
    # number_of_submodels_trained = 0

    model_name = 'blah'
    
    feature_array = [
        'avg_overhead', \
        'std_overhead', \
        'var_overhead', \
        'median_overhead', \
        'q1_overhead', \
        'q3_overhead', \
        'p9_overhead'
        ]

    # create a birch model threshold = 1
    # wandb.init(project='rtx-clustering', name="Two Features Run")
    birchModel = Birch(n_clusters=None)

    number_of_submodels_trained = 0

    data = []
    data_for_partial_clustering = []

    while len(data) < sample_size:
        result, new_sample = clusteringExperimentFunction(wf, {
            "knobs": wf.execution_strategy["knobs"],
            "ignore_first_n_results": wf.execution_strategy['ignore_first_n_results'],
            "window_size": wf.execution_strategy['window_size_for_car_number_change'],
        })
        data.append(new_sample)
        data_for_partial_clustering.append(new_sample)


        if len(data_for_partial_clustering) == partial_clustering_size:
            data_cpy = data_for_partial_clustering
            data_numpy = transfrom_to_nparray(data_cpy,feature_array)
            birchModel.partial_fit(data_numpy)
            data_cpy, data_numpy, data_for_partial_clustering = [], [], []

    run_model(birchModel, data, model_name, folder)

    with open(folder + 'description.txt', 'w+') as f:
        # f.write('The experiment took ' + str(current_milli_time() - start_time) + 'ms to run as follows\n')
        f.write('Ignored results (in ticks): ' + str(wf.execution_strategy['ignore_first_n_results']) + '\n')
        f.write('Sample size (in ticks): ' + str(sample_size) + '\n')
        f.write('Number of cars change  every (in ticks): ' +str(wf.execution_strategy['window_size_for_car_number_change']) + '\n')
        f.write('The partial clustering was performed on data of size: ' + str(partial_clustering_size) + '\n\n')
        f.write('Features the model trained on: \n')
        for feat in list(data[0].keys())[1:-1]:
            f.write(str(feat) + '\n')
    f.close()


    # wandb.save("dynamic_car_number_change1.h5")
