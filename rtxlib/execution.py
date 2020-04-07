from rtxlib import info, error, warn, direct_print, process, log_results, current_milli_time

import os
import csv
import numpy as np
from sklearn.cluster import Birch
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import json

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
            new_tick = wf. primary_data_provider['instance'].returnData()
            if new_tick is not None:
                i = list(new_tick.values())[0]
                process("IgnoreSamples \t| ", i, to_ignore)
        print("")

    folder = exp['save_in']
    os.makedirs(os.path.dirname(folder), exist_ok=True)

    # start collecting data
    sample_size = exp["sample_size"]

    array_overheads= [] # this variable stores the overhead data for a specific size 
    window_size = exp['window_size']
    partial_cluster_counter = exp['partial_clustering_size']
    partial_fit_array = []
    check_for_printing = []

    i = 0
    try:

        while i < (sample_size+1):
            new_tick = wf. primary_data_provider['instance'].returnData()
            if new_tick is not None:
                # ADD TO THE OVERHEAD ARRAY
                if hasattr(wf, "secondary_data_providers"):
                    for cp in wf.secondary_data_providers:
                        new_data = cp["instance"].returnDataListNonBlocking()
                        for nd in new_data:
                            try:
                                array_overheads.append({'totalCarNumber': nd['totalCarNumber'], 'overhead': nd['overhead']})
                                # print(array_overheads)
                            except StopIteration:
                                raise StopIteration()  # just
                            except RuntimeError:
                                raise RuntimeError()  # just fwd
                            except:
                                error("could not reducing data set: " + str(nd))

                i = list(new_tick.values())[0]

                if i % window_size == 0:
                    if hasattr(wf, "secondary_data_providers"):
                        for cp in wf.secondary_data_providers:
                            try:
                                exp["state"] = cp["data_reducer"](exp["state"], wf, array_overheads)
                                partial_fit_array.append(np.array([
                                    exp["state"]['avg_overhead'],\
                                    exp["state"]['std_overhead'], \
                                    exp["state"]['var_overhead'], \
                                    exp["state"]['median_overhead'], \
                                    exp["state"]['q1_overhead'], \
                                    exp['state']['q3_overhead'], \
                                    exp["state"]['p9_overhead']
                                    ]))

                                check = {'totalCarNumber': exp["state"]['totalCarNumber'], \
                                    'avg_overhead': exp["state"]['avg_overhead'], \
                                    'std_overhead':  exp["state"]['std_overhead'], \
                                    'var_overhead': exp["state"]['var_overhead'], \
                                    'median_overhead': exp["state"]['median_overhead'], \
                                    'q1_overhead': exp["state"]['q1_overhead'], \
                                    'q3_overhead': exp["state"]['q3_overhead'], \
                                    'p9_overhead': exp["state"]['p9_overhead']}
                             
                                check_for_printing.append(check)
                                # print(check_for_printing)
                                array_overheads = []
                                check = []
                                # print(exp['state'])
                            except StopIteration:
                                raise StopIteration()  # just
                            except RuntimeError:
                                raise RuntimeError()  # just fwd
                            except:
                                error("could not reducing data set: " + str(nd))
                            # print(exp['state']) 
                    if len(partial_fit_array) == partial_cluster_counter:

                        numpy_array = np.array(partial_fit_array)
                        
                        birchModel.partial_fit(numpy_array)
                        
                        number_of_submodels_trained += 1
                        partial_fit_array = []

                process("ticks \t| ", i, sample_size)

        print("")
    except StopIteration:
        # this iteration should stop asap
        error("This experiment got stopped as requested by a StopIteration exception")

    # print(exp['knobs'].values())
    # test_principalComponents = pca.fit_transform(scaled_test_data)

    run_model(birchModel, check_for_printing, 'final_global_fit_', folder)

    # wandb.log({'subclusters': len(birchModel.subcluster_centers_)}, step=(list(exp['knobs'].values())[0]))
    
    with open(folder + 'description.txt', 'w+') as f:
        f.write('The experiment run with the following attributes: \n')
        f.write('Ignored results: ' + str(to_ignore) + '\n')
        f.write('Sample size (in ticks): ' + str(sample_size) + '\n')
        f.write('The number of cars chang (in ticks): ' +str(window_size) + '\n')
        f.write('The partial cluster was performed on data of size: ' + str(partial_cluster_counter) + '\n\n')
        
        if 'pca' in globals():
            f.write('PCA was performed\n\n')

        f.write('Features the model trained on: \n')
        for feat in list(check_for_printing[0].keys())[1:-1]:
            f.write(str(feat) + '\n')
    f.close()

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
    
    return result, number_of_submodels_trained

def plot_silhouette_scores(model, test_data, n_clusters_min, n_clusters_max, folder, save_graph_name):
    """ Plot silhouette scores and return the best number of clusters"""

    if len(model.subcluster_labels_) > 2:

        silhouette_scores = []

        clusters_range = range(n_clusters_min, n_clusters_max+1)
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
                pass

        silhouette_range = [i[0] for i in results_dict]  
        plt.plot(silhouette_range[:], silhouette_scores[:])
        plt.xlabel('Number Of Clusers')
        plt.ylabel('Silhouette Score')
        plt.savefig(folder + 'silhouette_'+ save_graph_name +'.png')
        # plt.show()
        plt.close() 
        max_score = max(silhouette_scores)
        for i in results_dict:
            if i[1] == max_score:
                print("The highest silhouette scores(" + str(max_score) + ") is for " + str(i[0]) + " clusers")
                return int(i[0])
    else:
        print('couldnt get the scores, plz help')
        print('returning number of clusters = ' + str(n_clusters_min))
        return n_clusters_min

def transfrom_to_nparray(data, feature_array):
    """ Transform the gathered data to numpy array to fit the model's requirements """
    transformed_data = []
    for row in data:
        t = ()
        for k, v in row.items():
            if k in feature_array:
                t = t + (v,)
        transformed_data.append(t)
    transformed_data = np.array(transformed_data)
    return transformed_data

def run_model(model,test_data, model_name, folder):

    # LEFT FOR DEBUGGINING
    # feature_array = [
    #     'avg_overhead', \
    #     'std_overhead', \
    #     'var_overhead', \
    #     'median_overhead', \
    #     'q1_overhead', \
    #     'q3_overhead', \
    #     'p9_overhead'
    #     ]

    feature_array = list(test_data[0].keys())
    data_to_fit = transfrom_to_nparray(test_data, feature_array[1:])

    n_clusters = plot_silhouette_scores(model, data_to_fit, 3, 10, folder, ('global_fit_' + model_name))
    model.set_params(n_clusters = n_clusters)
    model.partial_fit()
    
    labels = model.predict(data_to_fit)

    l = len(labels)
    for index in range(l):
        test_data[index].update({'label': labels[index]})

    feature_array.append('label')

    new_array = transfrom_to_nparray(test_data, feature_array)


    #####################################   PLOTING PART    #######################################
    plt.scatter(new_array[:,0], new_array[:,1], c=labels, cmap='rainbow', alpha=0.7, edgecolors='b')    
    plt.ylabel('Overhead: average')
    plt.xlabel('car number')
    plt.savefig(folder+ model_name +'_carVSavg.png')
    # plt.show()
    plt.close()

    plt.scatter(new_array[:,0], new_array[:,2], c=labels, cmap='rainbow', alpha=0.7, edgecolors='b')
    plt.ylabel('Overhead: STD')
    plt.xlabel('car number')
    plt.savefig(folder+ model_name +'_carVSstd.png')
    # plt.show()
    plt.close()

    plt.scatter(new_array[:,0], new_array[:,3], c=labels, cmap='rainbow', alpha=0.7, edgecolors='b')
    plt.ylabel('Overhead: Var')
    plt.xlabel('car number')
    plt.savefig(folder+ model_name +'_carVSvar.png')
    # plt.show()
    plt.close()

    plt.scatter(new_array[:,0], new_array[:,4], c=labels, cmap='rainbow', alpha=0.7, edgecolors='b')
    plt.ylabel('Overhead: Median')
    plt.xlabel('car number')
    plt.savefig(folder+ model_name +'_carVSmedian.png')
    # plt.show()
    plt.close()

    plt.scatter(new_array[:,0], new_array[:,5], c=labels, cmap='rainbow', alpha=0.7, edgecolors='b')
    plt.ylabel('Overhead: 1st Quartile')
    plt.xlabel('car number')
    plt.savefig(folder+ model_name +'_carVSq1.png')
    # plt.show()
    plt.close()

    plt.scatter(new_array[:,0], new_array[:,6], c=labels, cmap='rainbow', alpha=0.7, edgecolors='b')
    plt.ylabel('Overhead: 3rd Quartile')
    plt.xlabel('car number')
    plt.savefig(folder+ model_name +'_carVSq3.png')
    # plt.show()
    plt.close()

    plt.scatter(new_array[:,0], new_array[:,7], c=labels, cmap='rainbow', alpha=0.7, edgecolors='b')
    plt.ylabel('Overhead: 90th Percentile')
    plt.xlabel('car number')
    plt.savefig(folder+ model_name +'_carVSp90.png')
    # plt.show()
    plt.close()

    plt.scatter(new_array[:,1], new_array[:,3], c=labels, cmap='rainbow', alpha=0.7, edgecolors='b')
    plt.ylabel('Overhead: Variance')
    plt.xlabel('Overhead: Average')
    plt.savefig(folder + model_name +'_varVS90th.png')
    # plt.show()
    plt.close()
        
    plt.scatter(new_array[:,5], new_array[:,6], c=labels, cmap='rainbow', alpha=0.7, edgecolors='b')
    plt.ylabel('Overhead: 3rd Quartile')
    plt.xlabel('Overhead: 1st Quartile')
    plt.savefig(folder + model_name +'_1stVS3rd.png')
    # plt.show()
    plt.close()

    plt.scatter(new_array[:,4], new_array[:,2], c=labels, cmap='rainbow', alpha=0.7, edgecolors='b')
    plt.ylabel('Overhead: STD')
    plt.xlabel('Overhead: Median')
    plt.savefig(folder + model_name +'_medianVSstd.png')
    # plt.show()
    plt.close()

    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure(figsize=(15,10))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(new_array[:,1], new_array[:,0], new_array[:,2], c=new_array[:,3], cmap='rainbow', alpha=0.8, edgecolors='b')
    # ax.set_xlabel('Average')
    # ax.set_ylabel('cars')
    # ax.set_zlabel('Median')
    # fig.savefig(folder + model_name + '3D.png')

    # big_list = []
    # for label in np.unique(np.array(labels)):
    #     small_list = []
    #     for d in test_data:
    #         if d['label'] == label:
    #             small_list.append(d['totalCarNumber'])
    #     x = np.array(small_list) 
    #     x = np.unique(x)
    #     big_list.append(x.tolist())
    
    #################################   WRITTING PART    ###################
    
    with open(folder + 'data.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=feature_array)
        writer.writeheader()
        for dictionary in test_data:
            writer.writerow(dictionary)
    f.close()
    
    # write results to a file
    with open(folder + 'results.txt', 'w+') as f:
        f.write('number of clusters: ' + str(len(model.subcluster_labels_)) + '\n')
        for i in model.subcluster_centers_:
            f.write(str(i) + ',\n')
    f.close()


