# Simple sequantial run of knob values
import numpy as np

name = "CrowdNav-BirchClustering"

execution_strategy = {
    "ignore_first_n_results": 0,
    "sample_size": 250,
    "window_size": 250,
    "type": "clustering",
    "knobs": [
        {"total_car_counter": 100},
        {"total_car_counter": 200},
        # {"total_car_counter": 300},
        # {"total_car_counter": 400},
        # {"total_car_counter": 500},
        # {"total_car_counter": 600},
        # {"total_car_counter": 700},
        # {"total_car_counter": 100},
        # {"total_car_counter": 200},
        # {"total_car_counter": 300},
        # {"total_car_counter": 400},
        # {"total_car_counter": 500},
        # {"total_car_counter": 600},
        # {"total_car_counter": 700},
    ]
}

# 2 gather data
def primary_data_reducer(state, newData, wf, temp_array):
    # cnt = state["count"]
    overhead = temp_array[-1]
    state['carCount'] = newData['carNumber']
    state["overhead"] = overhead
    state['avg_overhead'] = np.average(temp_array)
    state['std_overhead'] = np.std(temp_array)
    state['var_overhead'] = np.var(temp_array)
    # state["moving_avg_overhead"] = (state["avg_overhead"] * cnt + overhead) / (cnt + 1) # moving average
    state['median_overhead'] = np.median(temp_array)
    state['q1_overhead'] = np.quantile(temp_array, .25)
    state['q2_overhead'] = np.quantile(temp_array, .75)
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
def evaluator(resultState, wf):

    return resultState['overhead']


def state_initializer(state, wf):
    state['carCount'] = 0
    state["overhead"] = 0
    state['avg_overhead'] = 0
    state['std_overhead'] = 0   
    state['var_overhead'] = 0
    # state["moving_avg_overhead"] = 0
    state['median_overhead'] = 0
    state['q1_overhead'] = 0
    state['q2_overhead'] = 0
    # state["count"] = 0
    return state
