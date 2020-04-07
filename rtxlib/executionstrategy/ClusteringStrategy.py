from colorama import Fore

from rtxlib import info, error
from rtxlib.execution import clusteringExperimentFunction
from rtxlib.execution import plot_silhouette_scores
from rtxlib.execution import run_model

from sklearn.cluster import Birch
# import wandb


def start_clustering_strategy(wf):
    """ executes all experiments from the definition file """
    info("> ExecStrategy   | Clustering", Fore.CYAN)
    wf.totalExperiments = len(wf.execution_strategy["knobs"])
    
    # create a birch model 
    # can be moved to within the for loop to initialize for each knob , threshold = 1
    # wandb.init(project='rtx-clustering', name="Two Features Run")
    birchModel = Birch(n_clusters=None)

    number_of_submodels_trained = 0

    check_for_printing = []

    for kn in wf.execution_strategy["knobs"]:
        result, number_of_submodels_trained = clusteringExperimentFunction(birchModel,number_of_submodels_trained, check_for_printing,wf, {
            "knobs":kn,
            # "knobs": {"forever": True},
            "ignore_first_n_results": wf.execution_strategy["ignore_first_n_results"],
            "sample_size": wf.execution_strategy["sample_size"],
            "window_size": wf.execution_strategy['window_size_for_car_number_change'],
            "partial_clustering_size": wf.execution_strategy['partial_clustering_size'],
            'save_in': wf.execution_strategy['save_in']
        })

    # wandb.save("dynamic_car_number_change1.h5")
