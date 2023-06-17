import sys
import networkx as nx
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.patches as mpatches
from datetime import datetime

import main as global_main
import dependency_graph_insertion as IA_dependency
import dependency_graph_removal as RA_dependency

def draw_paper_performance(IS_OD, IS_OD_NS, IS_ID, IS_ID_NS, core_strength, core_number, degree, random, graph_name, save_fig = 0):
    n = np.arange(len(core_strength))
    x_ticks_label = []
    n1 = []
    for i in range(1, len(core_strength) + 1):
        if (i+2) % 4  == 0:
            x_ticks_label.append((i+2)*50)
            n1.append((i+1))
    
    plt.subplot(1, 1, 1)
    plt.plot(n, IS_OD, '--', label='Loaded from file!', marker='v', markersize=3, color='c')
    plt.plot(n, IS_OD_NS, '--', label='Loaded from file!', marker='s', markersize=3, color='blue')
    plt.plot(n, IS_ID, '--', label='Loaded from file!', marker='v', markersize=3, color='purple')
    plt.plot(n, IS_ID_NS, '--', label='Loaded from file!', marker='s', markersize=3, color='red')
    plt.plot(n, core_strength, label='Loaded from file!', marker='^', markersize=3, color='black')
    plt.plot(n, core_number, '--', label='Loaded from file!', marker='s', markersize=3, color='green')
    plt.plot(n, degree, label='Loaded from file!', marker='^', markersize=6, color='darkorange')
    plt.plot(n, random, label='Loaded from file!', marker="v", markersize=3, color='y')    

    
    font_size, font_sz = 22, 15
#     plt.xlabel("Number of edge insertion (b)", fontsize = font_size)
#     plt.ylabel('Affected nodes, F (%)', fontsize = font_size -3)
    plt.xlabel("Budget (b)", fontsize = font_size+3)
    plt.ylabel('F', fontsize = font_size +3)
    # if not (graph_name =='as19971108' or graph_name =='ca-CondMat' or graph_name == 'inf-openflights' or graph_name =='p2p-Gnutella08'):
    plt.title(graph_name, fontsize = font_size + 5)
    plt.xticks(n1, x_ticks_label, fontsize=font_sz+5)
    plt.yticks(fontsize=font_sz+5)
    
    # if graph_name =='as19971108' or graph_name =='as19990309':
    plt.legend( [r'$IS_{OD}$', r'$IS_{OD}$ NS', r'$IS_{ID}$', r'$IS_{ID}$ NS', 'Core Strength', 'Core Number', 'Degree', 'Random'], title = None, title_fontsize=font_sz, 
            bbox_to_anchor=(.01, .9999), loc='upper left', ncol=1, fontsize = font_sz)
    
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams["figure.figsize"] = (5.5, 4.5)
    plt.tight_layout()
    plt.grid(False)
    if save_fig:
        plt.savefig('Edge_insertion_figure/paper_figure/' + graph_name + '_line_chart_insertion_neighbor_sum.pdf')

    plt.show()
    
def getTopNPercentileNodes(graph, core_num, top_n_percentile, degree = 'degree', cutoff = None):
    cnt = 0
    for node in graph.nodes():
        if graph.degree[node] <= 2:
            cnt += 1
    # print("Total node of degree <= 2   --->>>> ", cnt)

    degree_vals = [val for (node, val) in graph.degree()]
    degree_cutoff = sorted(degree_vals, reverse=False)[int(len(core_num) * (top_n_percentile) / 100)]
    if cutoff:
        degree_cutoff = cutoff
    crucial_degree_nodes = list([u for u in core_num if graph.degree[u] >= degree_cutoff])

    core_vals = list(core_num.values())
    core_cutoff = sorted(core_vals, reverse=False)[int(len(core_num) * top_n_percentile / 100)]
    if cutoff:
        core_cutoff = cutoff
    crucial_core_nodes = list([u for u in core_num if core_num[u] >= core_cutoff])
    # print("Cutoff ===   ", degree_cutoff, core_cutoff, len(crucial_degree_nodes),  len(crucial_core_nodes))
    if degree == 'degree':
        return  crucial_degree_nodes
    else:
        return crucial_core_nodes

def randomEdgeInsertion(graph, possible_new_edge, core_num, percentage, dest_name, exp=5, budget=None):
    changed_core_cnt = 0
    percentage_edge_number = int(graph.number_of_edges() * percentage / 100)
    if budget:
        percentage_edge_number = budget

    # print("Number of edges to be added -->>>    ", percentage_edge_number)

    result_dict, affected_percentage = {}, {}
    # print(percentage_edge_number, len(possible_new_edge))
    while len(possible_new_edge) < percentage_edge_number:
        possible_new_edge.append(possible_new_edge[0])
    #percentage_edge_number = min(percentage_edge_number, len(pos))
    for b in range(0, percentage_edge_number, 50):
        changed_core_cnt = 0
        for ex in range(0, exp):
            temp_graph = graph.copy()
            insert_edge_list = random.sample(list(possible_new_edge), b + 50)
            # print(len(delete_edge_list))

            temp_graph.add_edges_from(insert_edge_list)
            temp_core_number = nx.core_number(temp_graph)
            changed_core_cnt += coreNumberChange(graph, core_num, temp_core_number)

        changed_core_cnt = int(changed_core_cnt / exp)
        result_dict[b + 50] = changed_core_cnt
        affected_percentage[b + 50] = np.round(changed_core_cnt / graph.number_of_nodes() * 100, 2)


    # saveData(graph.number_of_nodes(), len(possible_new_edge), result_dict, dest_path, dest_name)

    print("Random", list(affected_percentage.values()), sep="")
    return list(affected_percentage.values())


def endPointSumEdgeInsertion(graph, possible_new_edge, core_num, percentage, dest_name, type='Degree', budget=None):
    changed_core_cnt = 0
    CI_increase, node_cirs_ciis = {}, {}
    #CI_increase, node_cirs_ciis = insertion_algo.depndencyGraphInsertion(graph)
    percentage_edge_number = int(graph.number_of_edges() * percentage / 100)
    if budget:
        percentage_edge_number = budget
    # print("Number of edges to be added -->>>    ", percentage_edge_number)
    if type == 'Core_strength':
        core_strength = global_main.getCoreStrength(graph, core_num)
    elif type == 'inDegree_insertion' or type == 'outDegree_insertion':
        IS_ID, IS_OD, run_time = IA_dependency.dependencyGraphInsertion(graph)
        if type == 'inDegree_insertion':
            node_cirs_ciis = IS_ID
        else:
            node_cirs_ciis = IS_OD

    deg_sum_dict = {}
    for edge in possible_new_edge:
        if type == 'inDegree_insertion' or type == 'outDegree_insertion':
            if edge[0] in node_cirs_ciis and edge[1] in node_cirs_ciis:
                deg_sum_dict[edge] = node_cirs_ciis[edge[0]] + node_cirs_ciis[edge[1]]
            else:
                deg_sum_dict[edge] = 100000
        elif type == 'Hcd':
            if edge[0] in node_cirs_ciis and edge[1] in node_cirs_ciis:
                deg_sum_dict[edge] = core_num[edge[0]] + core_num[edge[1]] - (node_cirs_ciis[edge[0]] + node_cirs_ciis[edge[1]])
            else:
                deg_sum_dict[edge] = 0
        elif type == 'Degree':
            deg_sum_dict[edge] = graph.degree[edge[0]] + graph.degree[edge[1]]
        elif type == 'Core_strength':
            deg_sum_dict[edge] = core_strength[edge[0]] + core_strength[edge[1]]
        elif type == 'Core_number':
            deg_sum_dict[edge] = core_num[edge[0]] + core_num[edge[1]]
            #deg_sum_dict[edge] = min(core_num[edge[0]], core_num[edge[1]])
    if type == "Core_number" or type == "Core_strength" or type == 'Degree':
        deg_sum_dict = dict(sorted(deg_sum_dict.items(), key=lambda item: item[1], reverse=True))
    else:
        deg_sum_dict = dict(sorted(deg_sum_dict.items(), key=lambda item: item[1], reverse=True))

    insertion_edge_list, visited, cnt = [], [], 0

    for edge in deg_sum_dict.keys():
        if edge[0] in visited and edge[1] in visited:
            continue
        insertion_edge_list.append(edge)
        visited.append(edge[0])
        visited.append(edge[1])
        cnt += 1
        if cnt == percentage_edge_number:
            break

    if cnt < percentage_edge_number:
        for edge in deg_sum_dict.keys():
            if edge not in insertion_edge_list:
                insertion_edge_list.append(edge)
                cnt += 1
                if cnt >= percentage_edge_number:
                    break

    temp_graph = graph.copy()
    result_dict, affected_percentage = {}, {}
    for b in range(0, percentage_edge_number, 50):
        temp_graph.add_edges_from(insertion_edge_list[b:b + 50])
        temp_core_number = nx.core_number(temp_graph)
        changed_core_cnt = coreNumberChange(graph, core_num, temp_core_number)
        # print("budget and accuracy  ", b + 50, changed_core_cnt)
        result_dict[b + 50] = changed_core_cnt
        affected_percentage[b + 50] = np.round(changed_core_cnt / graph.number_of_nodes() * 100, 2)
        #print(type, b, nx.average_shortest_path_length(temp_graph))

    # saveData(graph.number_of_nodes(), len(possible_new_edge), result_dict, dest_path, dest_name)
    print(type, list(affected_percentage.values()), sep="")
    return list(affected_percentage.values())

def coreNumberChange(graph, actual_core_number, changed_core_number):
    cnt = 0
    for node in graph.nodes():
        if actual_core_number[node] != changed_core_number[node]:
            cnt += 1

    return cnt


def modifyCiis(graph, ciis):
    new_ciis = {}
    for node in graph.nodes():
        if node in ciis:
            new_ciis[node] = ciis[node]
            cnt = 1
            for v in graph.neighbors(node):
                if v in ciis:
                    new_ciis[node] += ciis[v]
                    cnt += 1
            #new_ciis[node] /= cnt

    return new_ciis

def getInsertionEdgeList(temp_graph, deg_sum_dict, insertion_edge_cnt, visited_node, last_cnt, core_sum_dict, crucial_nodes =[]):
    insertion_edge_list = []
    cnt = 0
    visited = []
    temp_cnt = 0
    # Insert edges according to lower Edge strength and consider crucial nodes (top n'th percentile node by core number value)
    for edge in deg_sum_dict.keys():
        if temp_cnt < last_cnt:
            temp_cnt += 1
            continue
        temp_cnt += 1
        # print(edge, deg_sum_dict[edge])
        if crucial_nodes and (edge[0] not in crucial_nodes or edge[1] not in crucial_nodes):
            continue
        if edge[0] in visited_node and edge[1] in visited_node:
            continue
        insertion_edge_list.append(edge)
        visited_node.append(edge[0])
        visited_node.append(edge[1])
        # print(edge, deg_sum_dict[edge])
        cnt += 1
        if cnt >= insertion_edge_cnt:
            break
    #print("Before NEXT_LOOP total inserted   --->>>>   ", cnt)
    # Insert edges according to lower Edge strength and ignore crucial nodes
    for edge in deg_sum_dict.keys():
        if cnt >= insertion_edge_cnt:
            break
        if edge[0] in visited_node and edge[1] in visited_node:
            continue
        insertion_edge_list.append(edge)
        visited_node.append(edge[0])
        visited_node.append(edge[1])
        cnt += 1

    #'''
    # insert randomly
    #print("Before RANDOM total inserted   --->>>>   ", cnt)
    while cnt < insertion_edge_cnt:
        edge, ciis = random.choice(list(deg_sum_dict.items()))
        if edge not in insertion_edge_list:
            insertion_edge_list.append(edge)
            cnt += 1
    #'''

    if cnt < insertion_edge_cnt:
        for edge in core_sum_dict.keys():
            if edge in insertion_edge_list or temp_graph.has_edge(*edge):
                continue
            insertion_edge_list.append(edge)
            cnt += 1
            if cnt >= insertion_edge_cnt:
                break

    #print("Total Insertion-->    ", len(insertion_edge_list))
    return  insertion_edge_list, temp_cnt


def ciisSumEdgeInsertion(graph, possible_new_edge, core_num, percentage, kshell, dest_name, type='CI_strength', budget=None,
                           compute_interval=50, ns=False, mx_sum='mx'):

    """
    Returns Insertion strength sum result dictionary
    To consider Neighbor sum call modifyCiis function
    Check sorting order and save data or not
    """

    ciis_start_time = datetime.now()
    top_n_percentile = 0
    crucial_nodes = getTopNPercentileNodes(graph, core_num, top_n_percentile, degree='degree', cutoff= None)

    core_num_time = 0
    percentage_edge_number = int(graph.number_of_edges() * percentage / 100)
    if budget:
        percentage_edge_number = budget
    temp_graph = graph.copy()
    result_dict, affected_percentage, visited_node = {}, {}, []

    #CI_increase, node_ciis = insertion_algo.depndencyGraphInsertion(graph)
    x_interval, last_cnt = 50, 0
    delete_cnt = min(x_interval, compute_interval)
    IS_sum_dict, core_sum_dict = {}, {}
    #delete_cnt = 50
    for b in range(0, percentage_edge_number, delete_cnt):
        if b % compute_interval == 0:
            last_cnt = 0
            IS_ID, IS_OD, run_time = IA_dependency.dependencyGraphInsertion(graph)
            if type == 'inDegree_insertion':
                node_ciis = IS_ID
            else:
                node_ciis = IS_OD
            #initial_affected_list, node_ciis = IS.getRemovalInsertionStrengthSortedList(graph, kshell, 0, dest_name, removal=0)
            if ns: # if need to modify the ciis with neighbor sum
                node_ciis = modifyCiis(graph, node_ciis)
            IS_sum_dict, core_sum_dict = {}, {}
            for edge in possible_new_edge:
                core_sum_dict[edge] = core_num[edge[0]] + core_num[edge[1]]
                # if type == 'CI_strength':
                if edge[0] in node_ciis and edge[1] in node_ciis:
                    if mx_sum == 'sum':
                        IS_sum_dict[edge] = node_ciis[edge[0]] + node_ciis[edge[1]]
                    else:
                        IS_sum_dict[edge] = max(node_ciis[edge[0]], node_ciis[edge[1]])
                else:
                    IS_sum_dict[edge] = 100000
            if 'outDegree_insertion_NOT' in type or 'pagerank_insertion_NOT' in type or 'inDegree_insertion_NOT' in type:
                IS_sum_dict = dict(sorted(IS_sum_dict.items(), key=lambda item: item[1], reverse=True))
            else:
                IS_sum_dict = dict(sorted(IS_sum_dict.items(), key=lambda item: item[1], reverse=False))
            core_sum_dict = dict(sorted(core_sum_dict.items(), key=lambda item: item[1], reverse=True))
        #last_cnt = 0
        insertion_edge_list, last_cnt = getInsertionEdgeList(temp_graph, IS_sum_dict, delete_cnt, visited_node, last_cnt, core_sum_dict, crucial_nodes=crucial_nodes)
        #insertion_edge_list = getInsertionEdgeListDifferentKshell(graph, core_num, node_ciis, possible_new_edge, delete_cnt, visited_node, crucial_nodes=crucial_nodes)

        temp_graph.add_edges_from(insertion_edge_list)
        core_start_time = datetime.now()
        temp_core_number = nx.core_number(temp_graph)
        changed_core_cnt = coreNumberChange(graph, core_num, temp_core_number)
        core_end_time = datetime.now()
        core_num_time += (core_end_time - core_start_time).total_seconds()
        # print("budget and accuracy  ", dest_name,  b + delete_cnt, changed_core_cnt)
        if b % x_interval == 0:
            result_dict[b + delete_cnt] = changed_core_cnt
            affected_percentage[b + delete_cnt] = np.round(changed_core_cnt / graph.number_of_nodes() * 100, 2)

    if ns:
        type = type + '_ns'
    print(type, list(affected_percentage.values()), sep = "")
    return list(affected_percentage.values()), np.round(core_num_time, 2)



def simulateEdgeInsertion(graph, dest_name):
    core_num = nx.core_number(graph)
    kshell = global_main.generateShellSubgraph(graph, core_num)
    possible_edges_node, common_count, common_one_new_edge, common_two_new_edge,  common_node_list_edge = IA_dependency.getPossibleEdges(graph)
    possible_edge = common_two_new_edge
    per = 20

    Degree = endPointSumEdgeInsertion(graph, possible_edge, core_num, per, dest_name, type='Degree', budget=1000)
    Core_number = endPointSumEdgeInsertion(graph, possible_edge, core_num, per, dest_name, type='Core_number', budget=1000)
    Core_strength = endPointSumEdgeInsertion(graph, possible_edge, core_num, per, dest_name, type='Core_strength', budget=1000)
    Random = randomEdgeInsertion(graph, possible_edge, core_num, per, dest_name, exp=5, budget=1000)

    in_insertion, ct = ciisSumEdgeInsertion(graph, possible_edge, core_num, per, kshell, dest_name, 'inDegree_insertion', 1000, 1000,  False)
    in_insertion_ns, ct = ciisSumEdgeInsertion(graph, possible_edge, core_num, per, kshell, dest_name, 'inDegree_insertion', 1000, 1000, True)
    
    out_insertion, ct = ciisSumEdgeInsertion(graph, possible_edge, core_num, per, kshell, dest_name, 'outDegree_insertion', 1000, 1000,  False)
    out_insertion_ns, ct = ciisSumEdgeInsertion(graph, possible_edge, core_num, per, kshell, dest_name, 'outDegree_insertion', 1000, 1000, True)

    # out_insertion, ct = ciisSumEdgeInsertion(graph, possible_edge, core_num, per, kshell, dest_name, 'outDegree_insertion', 1000, 1000, False)

    draw_paper_performance(out_insertion, out_insertion_ns, in_insertion, in_insertion_ns, Core_strength, Core_number, 
                             Degree, Random, graph_name, save_fig = 0)


if __name__ == '__main__':
    file_name = sys.argv[1]
    # print(file_name)
    graph = global_main.readGraph(file_name)
    graph_matrix = nx.to_numpy_matrix(graph)
    graph = nx.from_numpy_matrix(graph_matrix)

    file_name = file_name.split('/')
    graph_name = file_name[len(file_name) - 1].split('.')[0]
    # print(graph_name)

    results = simulateEdgeInsertion(graph, graph_name)