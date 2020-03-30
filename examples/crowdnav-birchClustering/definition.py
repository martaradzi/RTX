# Simple sequantial run of knob values
import numpy as np

name = "CrowdNav-BirchClustering"

execution_strategy = {
    "ignore_first_n_results": 0,
    "sample_size": 24000,
    "window_size": 1000,
    "type": "clustering",
    "knobs": [
        {'z': 1},
        # {'z': 2},
        # {'z': 3},
        # {'z': 4},
    ]
}

# 2 gather data

def secondary_data_reducer(state, wf, temp_array):
    # cnt = state["count"]
    # overhead = temp_array[-1]
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

def primary_data_reducer(state, newData, wf):
    cnt = state['tick']
    state['tick'] = newData['tick'] + cnt
    return state


secondary_data_providers = [{
    "type": "kafka_consumer",
    "kafka_uri": "localhost:9092",
    "topic": "crowd-nav-trips",
    "serializer": "JSON",
    "data_reducer": secondary_data_reducer
}]

primary_data_provider = {
    "type": "kafka_consumer",
    "kafka_uri": "localhost:9092", 
    "topic": "crowd-nav-ticks",
    "serializer": "JSON",
    "data_reducer": primary_data_reducer
}

change_provider = {
    "type": "kafka_producer",
    "kafka_uri": "localhost:9092", # this was changed
    "topic": "crowd-nav-commands",
    "serializer": "JSON",
}


def evaluator(model):

    return len(model.subcluster_labels_)


def state_initializer(state, wf):
    # state['tick'] = 0
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
