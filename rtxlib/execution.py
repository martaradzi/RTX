from rtxlib import info, error, warn, direct_print, process, log_results, current_milli_time
import numpy as np
from sklearn.cluster import Birch
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# import wandb

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

def clusteringExperimentFunction(birchModel, number_of_submodels_trained, check_for_printing, wf, exp):
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
                process("IgnoreSamples \t| ", i, to_ignore)
        print("")

    # start collecting data
    sample_size = exp["sample_size"]

    array_overheads= [] # this variable stores the overhead data for a specific size 
    # window = exp['window_size']
    
    variance_and_percentile_array = []
    check_for_printing = []

    i = 0
    try:
        # tick_simulation_begun_at = 'blah'
        while i < (sample_size+1):
            new_tick = wf. primary_data_provider['instance'].returnData()
            # print(list(new_tick.values())[0])
            if new_tick is not None:
                # ADD TO THE OVERHEAD ARRAY
                if hasattr(wf, "secondary_data_providers"):
                    for cp in wf.secondary_data_providers:
                        new_data = cp["instance"].returnDataListNonBlocking()
                        for nd in new_data:
                            try:
                                # print('new data')
                                # print(nd)
                                array_overheads.append(nd['overhead'])
                                # print(array_overheads)

                            except StopIteration:
                                raise StopIteration()  # just
                            except RuntimeError:
                                raise RuntimeError()  # just fwd
                            except:
                                error("could not reducing data set: " + str(nd))

                i = list(new_tick.values())[0]

                if i % 1000 == 0:
                    # CLALUCATE THE VARIANCE AND PERCENTILE 
                    # FORGET THE OVERHEAD ARRAY

                    # IF LEN THE VARIANCE ARRAY = 4 
                    # DO CLUSTERING 
                    # print('at 1000 ticks')
                    if hasattr(wf, "secondary_data_providers"):
                        for cp in wf.secondary_data_providers:
                            try:
                                exp["state"] = cp["data_reducer"](exp["state"], wf, array_overheads)

                            except StopIteration:
                                raise StopIteration()  # just
                            except RuntimeError:
                                raise RuntimeError()  # just fwd
                            except:
                                error("could not reducing data set: " + str(nd))
                            
                            variance_and_percentile_array.append(np.array(list(exp["state"].values())))
                            check_for_printing.append(np.array(list(exp["state"].values())))
                            # print(exp["state"])
                            # print(variance_and_percentile_array)
                            array_overheads = []
                            # print(array_overheads)

                    
                    if len(variance_and_percentile_array) == 4:
                        print('trying to train model')
                        numpy_array = np.array(variance_and_percentile_array)
                        birchModel.partial_fit(numpy_array)
                        print('model trained')
                        model_cpy = birchModel
                        n = plot_silhouette_scores(birchModel, numpy_array, 2, 10, ('partial_fit_' + str(number_of_submodels_trained)))
                        run_model(model_cpy, n, numpy_array, ('partial_fit_' + str(number_of_submodels_trained)))
                        number_of_submodels_trained += 1
                        variance_and_percentile_array = []


                process("ticks \t| ", i, sample_size)

        print("")
    except StopIteration:
        # this iteration should stop asap
        error("This experiment got stopped as requested by a StopIteration exception")

    # import json
    # with open('2ksamplex12.txt', 'a+') as file1:
    #     file1.write(json.dumps(to_print))
    # print(exp['knobs'].values())
    # test_principalComponents = pca.fit_transform(scaled_test_data)

    n = plot_silhouette_scores(birchModel, np.array(check_for_printing), 3, 10, ('global_fit_' + str(number_of_submodels_trained)))
    run_model(birchModel, n, np.array(check_for_printing), ('global_fit_' + str(list(exp['knobs'].values())[0])))
    # wandb.log({'subclusters': len(birchModel.subcluster_centers_)}, step=(list(exp['knobs'].values())[0]))
    
    try:
        result = wf.evaluator(birchModel)
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
    info("> ResultClusterValue    | " + str(result))
    # log the result values into a csv file
    log_results(wf.folder, list(exp["knobs"].values()) + [result])
    
    # return the result value of the evaluator (which was returned before I started changing code)
    # as well as window_overhead (the current window of data to pass to the next knob iteration)
    # and to_add (with number of states for predicting in the clustering strategy)
    return result, number_of_submodels_trained

def plot_silhouette_scores(model, test_data, n_clusters_min, n_clusters_max, save_graph_name):
    """ Plot silhouette scores and return the best number of clusters"""
    silhouette_scores = [] # store scores for each number of clusters 
    # if statement only needed if the sample size is too small (when debugging)
    
    if len(model.subcluster_labels_) > 2:
        # clusters_range = range(initial_clusters_number, (n_clusters_model+1))
        clusters_range = range(n_clusters_min, n_clusters_max)
        results_dict = []
        # print(clusters_range)
        for number in clusters_range:
            # make a copy of the model so as not to mess up the 'correct' model
            model_cpy = model
            model_cpy.set_params(n_clusters=number)

            model_cpy.partial_fit()
            labels = model_cpy.predict(test_data)
            # print(labels)
            try: 
                s = metrics.silhouette_score(test_data, labels, metric='euclidean')
                silhouette_scores.append(s)
                results_dict.append((number, s))
            except ValueError:
                # print('impossible to check silhouette score for ' + str(number) + ' number of clusters')
                pass

        silhouette_range = [i[0] for i in results_dict]  
        plt.plot(silhouette_range[:], silhouette_scores[:])
        plt.xlabel('Number Of Clusers')
        plt.ylabel('Silhouette Score')
        plt.savefig('./graphs/ticks_experiment/silhouette'+ save_graph_name +'.png')
        # plt.show()
        plt.close() 
        max_score = max(silhouette_scores)
        for i in results_dict:
            if i[1] == max_score:
                # print("The highest silhouette scores(" + str(max_score) + ") is for " + str(i[0]) + " clusers")
                return int(i[0])
    else:
        return n_clusters_min
    
def run_model(model, n_clusters, test_data, model_name):

    # for the final fit of data, set the number of clusters

    model.set_params(n_clusters = n_clusters)
    model.partial_fit()
    # run predict to check the labels and for plotting later
    labels = model.predict(test_data)
    # print(len(np.unique(np.array(labels))))
    # cluster_labels = labels

    # wandb.sklearn.plot_clusterer(model, test_data, cluster_labels, labels=None, model_name=model_name)
    
    print(model.subcluster_centers_)

    plt.scatter(test_data[:,0], test_data[:,1], c=labels, cmap='rainbow', alpha=0.7, edgecolors='b')
    plt.ylabel('09Quantile')
    plt.xlabel('Variance')
    # plt.ylabel('PCA_1')
    # plt.xlabel('PCA_2')
    plt.savefig('./graphs/ticks_experiment/'+ model_name +'.png')
    # plt.show()
    plt.close()
    
    # import pickle
    # import joblib
    # joblib.dump(model, './models/filename.pkl')
    
    # saving the data (not the model)
    # with open(folder + 'labels_' + experiment_name + '.txt', 'w') as file1:
    #     file1.write("The number of the clusters is" + str(len(model.subcluster_labels_)) + "\n")
    #     file1.write("The centroids are: \n")
    #     file1.write(str(model.subcluster_centers_))
    # file1.close()

