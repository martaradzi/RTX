from colorama import Fore

from rtxlib import info, error
from rtxlib.execution import clusteringExperimentFunction
from rtxlib.execution import plot_silhouette_scores
from rtxlib.execution import run_model


from sklearn.cluster import Birch
import numpy as np
from sklearn.cluster import Birch
from matplotlib import pyplot as plt
import seaborn as sns


def start_clustering_strategy(wf):
    """ executes all experiments from the definition file """
    info("> ExecStrategy   | Sequential", Fore.CYAN)
    wf.totalExperiments = len(wf.execution_strategy["knobs"])
    
    # create a birch model 
    # can be moved to within the for loop to initialize for each knob , threshold = 1
    birchModel = Birch(n_clusters=None)

    # saves the windows of data in order to save the data while switching knobs
    window_overhead = []

    # data to predict on, made from data from differnt knobs
    data_to_test = []

    for kn in wf.execution_strategy["knobs"]:
        result, window_overhead, to_add = clusteringExperimentFunction(birchModel, window_overhead, wf, {
            "knobs":kn,
            # "knobs": {"forever": True},
            "ignore_first_n_results": wf.execution_strategy["ignore_first_n_results"],
            "sample_size": wf.execution_strategy["sample_size"],
            "window_size": wf.execution_strategy['window_size']
        })
        data_to_test += to_add

    test_data = np.array(data_to_test)
    
    # get silhouette scores using test set
    n_clusters = plot_silhouette_scores(birchModel, 3, test_data)
    run_model(birchModel, n_clusters, test_data)


