from rtxlib import info, error, warn, direct_print, process, log_results, current_milli_time
from rtxlib.clustering_tools import write_raw_data


def _defaultChangeProvider(variables,wf):
    """ by default we just forword the message to the change provider """
    return variables


def experimentFunction(wf, exp):
    """ executes a given experiment """
    start_time = current_milli_time()
    # remove all old data from the queues
    wf.primary_data_provider["instance"].reset()

    # load change event creator or use a default
    if hasattr(wf, "change_event_creator"):
        change_creator = wf.change_event_creator
    else:
        change_creator = _defaultChangeProvider

    # start
    info(">")
    info("> KnobValues     | " + str(exp["knobs"]))
    # create new state
    exp["state"] = wf.state_initializer(dict(),wf)

    # apply changes to system
    try:
        wf.change_provider["instance"].applyChange(change_creator(exp["knobs"],wf))
    except:
        error("apply changes did not work")

    # ignore the first data sets
    to_ignore = exp["ignore_first_n_results"]
    if to_ignore > 0:
        i = 0
        while i < to_ignore:
            new_data = wf.primary_data_provider["instance"].returnData()
            if new_data is not None:
                i += 1
                process("IgnoreSamples  | ", i, to_ignore)
        print("")

    # start collecting data
    sample_size = exp["sample_size"]
    i = 0
    try:
        while i < sample_size:
            # we start with the primary data provider using blocking returnData
            new_data = wf.primary_data_provider["instance"].returnData()
            if new_data is not None:
                try:
                    # print(new_data)
                    exp["state"] = wf.primary_data_provider["data_reducer"](exp["state"], new_data,wf)
                except StopIteration:
                    raise StopIteration()  # just fwd
                except RuntimeError:
                    raise RuntimeError()  # just fwd
                except:
                    error("could not reducing data set: " + str(new_data))
                i += 1
                process("CollectSamples | ", i, sample_size)
            # now we use returnDataListNonBlocking on all secondary data providers
            if hasattr(wf, "secondary_data_providers"):
                for cp in wf.secondary_data_providers:
                    new_data = cp["instance"].returnDataListNonBlocking()
                    for nd in new_data:
                        try:
                            exp["state"] = cp["data_reducer"](exp["state"], nd,wf)
                        except StopIteration:
                            raise StopIteration()  # just
                        except RuntimeError:
                            raise RuntimeError()  # just fwd
                        except:
                            error("could not reducing data set: " + str(nd))
        print("")
    except StopIteration:
        # this iteration should stop asap
        error("This experiment got stopped as requested by a StopIteration exception")
    try:
        result = wf.evaluator(exp["state"],wf)
    except:
        result = 0
        error("evaluator failed")
    # we store the counter of this experiment in the workflow
    if hasattr(wf, "experimentCounter"):
        wf.experimentCounter += 1
    else:
        wf.experimentCounter = 1
    # print the results
    duration = current_milli_time() - start_time
    # do not show stats for forever strategy
    if wf.totalExperiments > 0:
        info("> Statistics     | " + str(wf.experimentCounter) + "/" + str(wf.totalExperiments)
             + " took " + str(duration) + "ms" + " - remaining ~" + str(
            (wf.totalExperiments - wf.experimentCounter) * duration / 1000) + "sec")
    info("> FullState      | " + str(exp["state"]))
    info("> ResultValue    | " + str(result))
    # log the result values into a csv file
    log_results(wf.folder, list(exp["knobs"].values()) + [result])
    # return the result value of the evaluator
    return result


def clusteringExperimentFunction(sample_number, folder, wf, exp):
    """ executes the online clustering experiment """

    start_time = current_milli_time()
    # remove all old data from the queues
    wf.primary_data_provider["instance"].reset()

    # load change event creator or use a default
    if hasattr(wf, "change_event_creator"):
        change_creator = wf.change_event_creator
    else:
        change_creator = _defaultChangeProvider

    # start
    info(">")
    info("> Sample Number     \t| " + str(sample_number + 1))
    # create new state
    exp["state"] = wf.state_initializer(dict(),wf)

    first_batch_tick = None
    # get the first tick of the new sample 
    while first_batch_tick is None:
        new_tick = wf. primary_data_provider['instance'].returnData()
        if new_tick is not None:
            first_batch_tick = list(new_tick.values())[0]


    # ignore the first data sets
    to_ignore = exp["ignore_first_n_results"]
    if to_ignore > 0:
        i = 0
        while i < to_ignore:
            new_tick = wf. primary_data_provider['instance'].returnData()
            if new_tick is not None:
                i = list(new_tick.values())[0] - first_batch_tick
                process("IgnoreSamples \t| ", i, to_ignore)
        print("")


    # get the tick after ignoring
    sample_beginning_tick = None
    while sample_beginning_tick is None:
        new_tick = wf. primary_data_provider['instance'].returnData()
        if new_tick is not None:
            sample_beginning_tick = list(new_tick.values())[0]

    window_size = exp['window_size']
    array_overheads= [] # this variable stores the overhead data for a specific size 
    new_sample = [] # this variable stores the final samples passed to the model


    i = 0
    try:
        while i < (window_size):
            new_tick = wf. primary_data_provider['instance'].returnData()
            if new_tick is not None:
                # ADD TO THE OVERHEAD ARRAY
                if hasattr(wf, "secondary_data_providers"):
                    for cp in wf.secondary_data_providers:
                        new_data = cp["instance"].returnDataListNonBlocking()
                        for nd in new_data:
                            try:
                                # check if trips started in the current iteration of the car change
#                                 if (nd['totalCarNumber'] - nd['startCarNumber']) <= 100 and (nd['totalCarNumber'] - nd['startCarNumber']) >= 0:
                                array_overheads.append({'startCarNumber':   nd['startCarNumber'], 
                                                            'totalCarNumber':   nd['totalCarNumber'], 
                                                            'overhead':         nd['overhead'], 
                                                            'duration':         nd['duration'],
                                                            })
                            except StopIteration:
                                raise StopIteration()  # just
                            except RuntimeError:
                                raise RuntimeError()  # just fwd
                            except:
                                error("could not reducing data set: " + str(nd))

                i = list(new_tick.values())[0] - sample_beginning_tick
                process("GatheredData \t\t| ", i, window_size)

        print("")
    except StopIteration:
        # this iteration should stop asap
        error("This experiment got stopped as requested by a StopIteration exception")


    if hasattr(wf, "secondary_data_providers"):
        for cp in wf.secondary_data_providers:
            try:
                exp["state"] = cp["data_reducer"](exp["state"], wf, array_overheads)

                new_sample = {'totalCarNumber': exp["state"]['totalCarNumber'],
                            'numberOfTrips':    len(array_overheads),
                    #       'avg_overhead':     exp["state"]['avg_overhead'],
                    #       'std_overhead':     exp["state"]['std_overhead'],
                            # 'var_overhead':     exp["state"]['var_overhead'],
                            'median_overhead':  exp["state"]['median_overhead'],
                            'q1_overhead':      exp["state"]['q1_overhead'],
                            'q3_overhead':      exp["state"]['q3_overhead'],
                            'p9_overhead':      exp["state"]['p9_overhead'],
                    #       'duration': exp["state"]['duration']
                    }

            except StopIteration:
                raise StopIteration()  # just
            except RuntimeError:
                raise RuntimeError()  # just fwd
            except:
                error("could not reducing data set")

    # Write the gathered data into a file
    if len(array_overheads) != 0:
        write_raw_data(array_overheads, folder, array_overheads[0].keys(), header=False)

    try:
        result = wf.evaluator(array_overheads)
    except:
        result = 0
        error("evaluator failed")
    
    # we store the counter of this experiment in the workflow
    if hasattr(wf, "experimentCounter"):
        wf.experimentCounter += 1
    else:
        wf.experimentCounter = 1
    # print the results
    duration = current_milli_time() - start_time
    # do not show stats for forever strategy
    if wf.totalExperiments > 0:
        info("> Statistics     | " + str(wf.experimentCounter) + "/" + str(wf.totalExperiments)
             + " took " + str(duration) + "ms" + " - remaining ~" + str(
            (wf.totalExperiments - wf.experimentCounter) * duration / 1000) + "sec")
    info("> FullState      | " + str(exp["state"]))
    info("> Number of samples processed \t| " + str(result))
    
    return result, new_sample
