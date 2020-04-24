# Simple sequantial run of knob values
import numpy as np

name = "CrowdNav-BirchClustering"

execution_strategy = {
    "ignore_first_n_results": 2000,
    "sample_size": 20,
    "window_size_for_car_number_change": 3000,
    "partial_clustering_size": 5,
    "save_in": './experiments/start_number_filter_3_less_features/',
    "type": "clustering",
}

def secondary_data_reducer(state, wf, temp_dict):
    # cnt = state["count"]
    # overhead = temp_array[-1]
    state['duration'] = np.average([d['duration'] for d in temp_dict])
    state['totalCarNumber'] = int(np.median([d['totalCarNumber'] for d in temp_dict]))
    state['avg_overhead'] = np.average([d['overhead'] for d in temp_dict])
    state['std_overhead'] = np.std([d['overhead'] for d in temp_dict])
    state['var_overhead'] = np.var([d['overhead'] for d in temp_dict])
    state['median_overhead'] = np.median([d['overhead'] for d in temp_dict])
    state['q1_overhead'] = np.quantile([d['overhead'] for d in temp_dict], .25)
    state['q3_overhead'] = np.quantile([d['overhead'] for d in temp_dict], .75)
    state['p9_overhead'] = (np.percentile([d['overhead'] for d in temp_dict], 90))
    # state["count"] = cnt + 1
    return state

def primary_data_reducer(state, newData, wf):
    state['tick'] = newData['tick']
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
    "kafka_uri": "localhost:9092",
    "topic": "crowd-nav-commands",
    "serializer": "JSON",
}


def evaluator(resultState,wf):

    return resultState['duration']


def state_initializer(state, wf):
    # state['tick'] = 0
    state['totalCarNumber'] = 0
    state['duration'] = 0
    # state["overhead"] = 0
    state['avg_overhead'] = 0
    state['std_overhead'] = 0   
    state['var_overhead'] = 0
    state['median_overhead'] = 0
    state['q1_overhead'] = 0
    state['q3_overhead'] = 0
    state['p9_overhead'] = 0
    # state["count"] = 0
    return state
