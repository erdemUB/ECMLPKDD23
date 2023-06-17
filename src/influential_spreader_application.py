import sys
import networkx as nx
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.patches as mpatches

import main as global_main
import dependency_graph_insertion as IA_dependency
import dependency_graph_removal as RA_dependency

def draw_paper_performance(kshell, improved_kshell, RS_in, RS_out, IS_out, IS_in, CS_kshell, graph_name, save_fig = 0):
    n = np.arange(len(kshell))
    each_step = float(2)/len(kshell)/100
    x_axis = []
    x_ticks = []
    n1 = []
    for i in range(len(kshell)):
        x_axis.append(each_step*i)
        x_ticks.append(round(each_step*i, 2))
        if i % 2:
            n1.append(i)
    
    plt.subplot(1, 1, 1)
    plt.plot(n, RS_out, '--', label='Loaded from file!', marker='v', markersize=3, color='purple')
    plt.plot(n, RS_in, '--', label='Loaded from file!', marker='v', markersize=3, color='c')
    plt.plot(n, CS_kshell, '--', label='Loaded from file!', marker='s', markersize=3, color='blue')
    plt.plot(n, IS_out, '--', label='Loaded from file!', marker='v', markersize=3, color='black')
    plt.plot(n, IS_in, '--', label='Loaded from file!', marker='v', markersize=3, color='green')
    plt.plot(n, improved_kshell, label='Loaded from file!', marker='^', markersize=6, color='darkorange')
    plt.plot(n, kshell, label='Loaded from file!', marker="v", markersize=3, color='red')
#     plt.plot(n, inDegree_RS, label='Loaded from file!', marker="v", markersize=3, color='aqua')
    
    font_size, font_sz = 22, 15
    plt.xlabel("Infected time (t)", fontsize = font_size)
    plt.ylabel('S(t)', fontsize = font_size + 3)
    plt.title(graph_name, fontsize = font_size + 10)
    plt.xticks(n1, fontsize=20)
    plt.yticks(fontsize=20)
    
    
    plt.legend( [r'$RS_{OD}$', r'$RS_{ID}$', 'Core Strength', r'$IS_{OD}$', r'$IS_{ID}$', 'IKS', 'kshell'], title = None, 
               title_fontsize=font_sz-7, bbox_to_anchor=(.01, .990), loc='upper left', ncol=1, fontsize = font_sz-5)
    
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams["figure.figsize"] = (5.5, 4.5)
    plt.tight_layout()

    
    plt.grid(False)
    if save_fig:
        plt.savefig('influential_improved_figure/paper_figure/' + graph_name + '_line_chart.pdf')

    plt.show()

def getKshellSortedList(core_num, top_20_percentage ):
    sorted_list = dict(sorted(core_num.items(), key=lambda x: x[1], reverse=True))
    initial_affected_list, cnt = [], 0
    for key in sorted_list.keys():
        cnt += 1
        initial_affected_list.append(key)
        if cnt == top_20_percentage:
            break
    return initial_affected_list

def getImprovedKShellSortedList(graph, kshell, top_20_percentage):
    tot_degree, node_importance, node_information_entropy = 0.0, {}, {}
    for node in graph.nodes():
        tot_degree += graph.degree(node)

    for node in graph.nodes():
        node_importance[node] = graph.degree(node) * 1.0 / tot_degree
    for node in graph.nodes():
        neighbor_list = list(graph.neighbors(node))
        neigh_sum = 0.0
        for v in neighbor_list:
            ln_value =  math.log(node_importance[v]) * node_importance[v]
            neigh_sum += ln_value
        node_information_entropy[node] = neigh_sum * (-1.0)

    node_information_entropy = dict(sorted(node_information_entropy.items(), key=lambda x: x[1], reverse=True))

    group_sorted_kshell = {}
    max_selected = {}
    for key in kshell.keys():
        max_selected[key] = 0
        sorted_kshell = {}
        for node in kshell[key].nodes():
            sorted_kshell[node] = node_information_entropy[node]
        group_sorted_kshell[key] = dict(sorted(sorted_kshell.items(), key=lambda x: x[1], reverse=True))

    group_sorted_kshell = dict(sorted(group_sorted_kshell.items(), key=lambda x: x[0], reverse=True))
    improved_kshell_sorted_list = []

    #print(list(group_sorted_kshell[3].items())[0][0])
    #node_value_list = list(group_sorted_kshell[3].items())
    #print(node_value_list)
    while len(improved_kshell_sorted_list) < top_20_percentage:
        for kshell_num in group_sorted_kshell.keys():
           node_value_list = list(group_sorted_kshell[kshell_num].items())
           ind = max_selected[kshell_num]
           if ind >= len(kshell[kshell_num].nodes()):
               continue
           max_selected[kshell_num] += 1
           improved_kshell_sorted_list.append(node_value_list[ind][0])
           if len(improved_kshell_sorted_list) >= top_20_percentage:
               break
    #print(improved_kshell_sorted_list)

    return improved_kshell_sorted_list

def getIndividualRSISGroupSortedKshell(graph, kshell, type='CS'):
    CI_increase, node_cirs_ciis = {}, {}
    if type == 'CS':
        core_num = nx.core_number(graph)
        node_cirs_ciis = global_main.getCoreStrength(graph, core_num)
    elif 'insertion' in type:
        IS_ID, IS_OD, run_time = IA_dependency.dependencyGraphInsertion(graph)
        if type == 'inDegree_insertion':
            node_cirs_ciis = IS_ID
        else:
            node_cirs_ciis = IS_OD
    else:
        RS_ID, RS_OD, run_time = RA_dependency.dependencyGraphRemoval(graph)
        if type == 'inDegree_removal':
            node_cirs_ciis = RS_ID
        else:
            node_cirs_ciis = RS_OD

    group_sorted_kshell = {}
    for key in kshell.keys():
        sorted_kshell = {}
        for node in kshell[key].nodes():
            if int(node) in node_cirs_ciis.keys():
                sorted_kshell[node] = node_cirs_ciis[int(node)]
                # sorted_kshell[node] += (graph.degree(node)/graph.number_of_nodes())
        if 'inDegree_insertion_NOT' in type or 'inDegree_removal_NOT' in type or 'pagerank_removal_NOT' in type:
            group_sorted_kshell[key] = dict(sorted(sorted_kshell.items(), key=lambda x: x[1], reverse=False))
        else:
            group_sorted_kshell[key] = dict(sorted(sorted_kshell.items(), key=lambda x: x[1], reverse=True))

    group_sorted_kshell = dict(sorted(group_sorted_kshell.items(), key=lambda x: x[0], reverse=True))
    return group_sorted_kshell, node_cirs_ciis

def getRemovalInsertionStrengthSortedList(graph, kshell, top_20_percentage, type='CS'):
    group_sorted_kshell, node_cirs_ciis = getIndividualRSISGroupSortedKshell(graph, kshell, type=type)

    max_selected, max_selected_left, max_selected_right = {}, {}, {}
    for key in kshell.keys():
        max_selected[key], max_selected_left[key], max_selected_right[key] = 0, 0, 0
        max_selected_right[key] = len(group_sorted_kshell[key]) - 1

    removal_strength_sorted_list = []
    while len(removal_strength_sorted_list) < top_20_percentage:
        for kshell_num in group_sorted_kshell.keys():
            #if kshell_num <= 1:
             #   continue
            node_value_list = list(group_sorted_kshell[kshell_num].items())
            ind = max_selected[kshell_num]
            if ind >= len(node_value_list):
                continue
            max_selected[kshell_num] += 1
            removal_strength_sorted_list.append(node_value_list[ind][0])
            if len(removal_strength_sorted_list) >= top_20_percentage:
                break

    return removal_strength_sorted_list, node_cirs_ciis

def improvedKShellAlgo(exp_type, graph, beta, miu, core_num, kshell, initial_affected_list):

    #print("ewfdwefwe  >>>>>      ", graph_name)
    random.seed(10)
    initial_affected_percentage = len(initial_affected_list) / len(graph.nodes()) * 100
    #print(int(miu*100), len(initial_affected_list), initial_affected_percentage)

    average_affected = []
    for exp in range(0, 20):
        temp_affected_list = initial_affected_list.copy()
        affected_plus_recovered = set(initial_affected_list.copy())
        average_affected.append([])
        recovered_list = set()
        for t in range(2, 17):
            new_affected_list = set()
            #print(len(temp_affected_list), len(recovered_list))
            for node in temp_affected_list:
                neighbor_list = list(graph.neighbors(node))
                x = random.randint(1, 100)
                #print(x)
                if x > int(miu*100):
                    if node not in new_affected_list:
                        new_affected_list.add(node)
                        affected_plus_recovered.add(node)
                else:
                    recovered_list.add(node)
                    affected_plus_recovered.add(node)

                tot_select = int(np.round(len(neighbor_list) * beta))
                affected_neigh_list = random.sample(neighbor_list, tot_select)
                #print(beta, tot_select, affected_neigh_list)
                #new_affected_list.update(affected_neigh_list)
                #affected_plus_recovered.update(affected_neigh_list)
                #continue

                for v in neighbor_list:
                    if v in recovered_list or v in new_affected_list or v in temp_affected_list:
                        continue
                    x = random.randint(1, 100)
                    #if node_cirs_ciis and int(v) in node_cirs_ciis[graph_name].keys() and (exp_type == 'removal_kshell' or exp_type == 'insertion_kshell'):
                     #   x -= (node_cirs_ciis[graph_name][int(v)] * 10)

                    if x <= int(beta*100): # for random selection one by one
                    #if v in affected_neigh_list: # for random selection in a single turn
                        new_affected_list.add(v)
                        affected_plus_recovered.add(v)

                #if t == 3:
                 #   print(len(temp_affected_list), len(new_affected_list), len(recovered_list))

            temp_affected_list = list(new_affected_list.copy())
            affected_percentage = np.round((len(temp_affected_list) + len(recovered_list)) / len(graph.nodes()) * 100, 2)
            only_affected_percentage = np.round((len(affected_plus_recovered)) / len(graph.nodes()) * 100, 2)
            #print(t,  len(temp_affected_list) , len(recovered_list), affected_percentage, only_affected_percentage)
            average_affected[exp].append(affected_percentage)

    #print(average_affected)
    average_affected = np.mean(average_affected, axis = 0)
    average_affected = list(np.insert(average_affected, 0, initial_affected_percentage))
    average_affected = [np.round(num, 2) for num in average_affected]
    print(exp_type, average_affected, sep="")

    return average_affected

def getBetaValue(graph, degree_num):
    #print(type(graph.degree()), degree_num)
    first_moment = np.mean(list(degree_num.values()))
    second_moment = 0
    for key in degree_num.keys():
        second_moment += (degree_num[key] * degree_num[key])
    second_moment /= len(degree_num)
    #print(first_moment, second_moment, first_moment/second_moment)
    beta_min = np.round(first_moment/second_moment, 3)
    beta = np.round(beta_min + .005, 2)
    if beta < .02:
        beta = .02
    print("Beta value --->>>>  &  ", beta_min, "  & ",  beta)
    return beta_min, beta

def simulateInfluentialSpreader(graph, graph_name):
    core_num = nx.core_number(graph)
    degree_num = dict(graph.degree())
    CS = global_main.getCoreStrength(graph, core_num)
    knumber = [core_num[u] for u in core_num]
    kshell = global_main.generateShellSubgraph(graph, core_num)
    sorted_core_num = dict(sorted(core_num.items(), key=lambda x: x[1], reverse=True))
    top_20_percentage = int(len(graph.nodes()) / 5)

    miu = .01
    beta_min, beta = getBetaValue(graph, degree_num)

    inDegree_insertion_init, node_ciis = getRemovalInsertionStrengthSortedList(graph, kshell, top_20_percentage, type='inDegree_insertion')
    inDegree_insertion = improvedKShellAlgo('inDegree_insertion', graph, beta, miu, core_num, kshell, inDegree_insertion_init)

    outDegree_insertion_init, node_ciis = getRemovalInsertionStrengthSortedList(graph, kshell, top_20_percentage, type='outDegree_insertion')
    outDegree_insertion = improvedKShellAlgo('outDegree_insertion', graph, beta, miu, core_num, kshell, outDegree_insertion_init)

    inDegree_removal_init, node_ciis = getRemovalInsertionStrengthSortedList(graph, kshell, top_20_percentage, type='inDegree_removal')
    inDegree_removal = improvedKShellAlgo('inDegree_removal', graph, beta, miu, core_num, kshell, inDegree_removal_init)

    outDegree_removal_init, node_ciis = getRemovalInsertionStrengthSortedList(graph, kshell, top_20_percentage, type='outDegree_removal')
    outDegree_removal = improvedKShellAlgo('outDegree_removal', graph, beta, miu, core_num, kshell, outDegree_removal_init)

    CS_initial, node_cirs_ciis = getRemovalInsertionStrengthSortedList(graph, kshell, top_20_percentage, type = 'CS')
    CS_kshell_affected = improvedKShellAlgo('CS_kshell', graph, beta, miu, core_num, kshell, CS_initial)

    kshell_initial_affected_list = getKshellSortedList(core_num, top_20_percentage)
    kshell_affected = improvedKShellAlgo('kshell', graph, beta, miu, core_num, kshell, kshell_initial_affected_list)

    IKS_initial_affected_list = getImprovedKShellSortedList(graph, kshell, top_20_percentage)
    improved_kshell_affected = improvedKShellAlgo('improved_kshell', graph, beta, miu, core_num, kshell, IKS_initial_affected_list)

    draw_paper_performance(kshell_affected, improved_kshell_affected, inDegree_removal, outDegree_removal, 
                           outDegree_insertion, inDegree_insertion, CS_kshell_affected, graph_name, save_fig = 0)


if __name__ == '__main__':
    file_name = sys.argv[1]
    # print(file_name)
    graph = global_main.readGraph(file_name)
    graph_matrix = nx.to_numpy_matrix(graph)
    graph = nx.from_numpy_matrix(graph_matrix)

    file_name = file_name.split('/')
    graph_name = file_name[len(file_name) - 1].split('.')[0]
    # print(graph_name)
    
    # print(nx.info(graph))

    results = simulateInfluentialSpreader(graph, graph_name)