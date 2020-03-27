# Simple sequantial run of knob values
import numpy as np

name = "CrowdNav-BirchClustering"

execution_strategy = {
    "ignore_first_n_results": 0,
    "sample_size": 4000,
    "window_size": 400,
    "type": "clustering",
    "knobs": [
        {'z': 1},
        {'z': 2},
        {'z': 3},
        {'z': 4},
        {'z': 5},
        # {'z': 6},
        # {'z': 7},
        # {'z': 8},
        # {'z': 9},
        # {'z': 10},
        # {'z': 11},
        # {'z': 12},
    ]
}

# 2 gather data
def primary_data_reducer(state, newData, wf, temp_array):
    # cnt = state["count"]
    overhead = temp_array[-1]
    # state['carCount'] = newData['carNumber']
    # state["overhead"] = overhead
    # state['avg_overhead'] = np.average(temp_array)
    # state['std_overhead'] = np.std(temp_array)
    state['var_overhead'] = np.var(temp_array)
    # state['median_overhead'] = np.median(temp_array)
    # state['q1_overhead'] = np.quantile(temp_array, .25)
    # state['q2_overhead'] = np.quantile(temp_array, .75)
    state['09_overhead'] = np.quantile(temp_array, .90)
    # state["count"] = cnt + 1
    return state


primary_data_provider = {
    "type": "kafka_consumer",
    "kafka_uri": "localhost:9092", # this was changed
    "topic": "crowd-nav-trips",
    "serializer": "JSON",
    "data_reducer": primary_data_reducer
}

change_provider = {
    "type": "kafka_producer",
    "kafka_uri": "localhost:9092", # this was changed
    "topic": "crowd-nav-commands",
    "serializer": "JSON",
}

# 3 this can be clustering can print that
# TODO: figure out what the evaluator has to be
def evaluator(model):

    return len(model.subcluster_labels_)


def state_initializer(state, wf):
    # state['carCount'] = 0
    # state["overhead"] = 0
    # state['avg_overhead'] = 0
    # state['std_overhead'] = 0   
    state['var_overhead'] = 0
    # state['median_overhead'] = 0
    # state['q1_overhead'] = 0
    # state['q2_overhead'] = 0
    state['09_overhead'] = 0
    # state["count"] = 0
    return state
