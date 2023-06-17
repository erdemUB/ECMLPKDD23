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

def draw_paper_performance(RS_out, RS_in, Core_strength, Random, Degree, Core_number, graph_name, save_fig = 0):
    n = np.arange(len(Core_strength))
    x_ticks_label, n1 = [], []
    for i in range(1, len(Core_strength) + 1):
        if (i+2) % 4  == 0:
            x_ticks_label.append((i+2)*50)
            n1.append((i+1))
    
    plt.subplot(1, 1, 1)
    plt.plot(n, RS_out, '--', label='Loaded from file!', marker='v', markersize=3, color='purple')
    plt.plot(n, RS_in, '--', label='Loaded from file!', marker='s', markersize=3, color='red')
    plt.plot(n, Core_strength, '--', label='Loaded from file!', marker='s', markersize=3, color='green')
    plt.plot(n, Random, label='Loaded from file!', marker='^', markersize=6, color='darkorange')
    plt.plot(n, Degree, label='Loaded from file!', marker="v", markersize=3, color='blue')
    plt.plot(n, Core_number, label='Loaded from file!', marker='^', markersize=6, color='black')
    
    font_size = 22
    plt.xlabel("Number of edge removal (b)", fontsize = font_size)
    plt.ylabel('Affected nodes, F (%)', fontsize = font_size -3)
    plt.title(graph_name, fontsize = font_size + 10)
    
    plt.xticks(n1, x_ticks_label, fontsize=20)
    plt.yticks(fontsize=20)
    font_sz = 15
    plt.legend( [ r'$RS_{OD}$', r'$RS_{ID}$', 'Core Strength', 'Random', 'Degree', 'Core Number'],
               title = None, title_fontsize=font_sz-7, bbox_to_anchor=(.001, .990), loc='upper left', ncol=1, fontsize = font_sz-5)
   
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams["figure.figsize"] = (5.5, 4.5)
    plt.tight_layout()
    plt.grid(False)
    if save_fig:
        plt.savefig('Edge_deletion_figure/paper_figure/' + graph_name + '_line_chart.pdf')

    plt.show()

def randomEdgeRemoval(graph, core_num, percentage, dest_name, exp = 5, budget = None):
    changed_core_cnt = 0
    percentage_edge_number = int(graph.number_of_edges() * percentage / 100)
    if budget:
        percentage_edge_number = budget

    # print("Number of edges to be deleted -->>>    ", percentage_edge_number)

    affected_percentage, result_dict = {}, {}
    for b in range(0, percentage_edge_number, 50):
        changed_core_cnt = 0
        for ex in range (0, exp):
            temp_graph = graph.copy()
            delete_edge_list = random.sample(list(graph.edges()), b+50)
            #print(len(delete_edge_list))

            temp_graph.remove_edges_from(delete_edge_list)
            temp_core_number = nx.core_number(temp_graph)
            changed_core_cnt += coreNumberChange(graph, core_num, temp_core_number)

        changed_core_cnt = int(changed_core_cnt/exp)
        result_dict[b+50] = changed_core_cnt
        affected_percentage[b+50] = np.round(changed_core_cnt / graph.number_of_nodes() * 100, 2)

    print("Random", list(affected_percentage.values()), sep="")
    return list(affected_percentage.values())

def coreNumberChange(graph, actual_core_number, changed_core_number):
    cnt = 0
    for node in graph.nodes():
        if actual_core_number[node] != changed_core_number[node]:
            cnt += 1

    return cnt

def getEqualEdgeParent(graph, core_num):
    kcore = global_main.generateCoreSubgraph(graph, core_num)
    cs = global_main.getCoreStrength(graph, core_num)
    knumber = [core_num[u] for u in core_num]
    kmax = max(knumber)
    corona_clusters_nodes, corona_clusters_edges = {}, {}
    for k in range(2, kmax + 1):
        corona_clusters_nodes[k], corona_clusters_edges[k] = RA_dependency.findCoronaClusters(graph, kcore, core_num, k, cs)

    edge_parent = {}
    total_deletion = 0
    core_wise_len = []

    edge_deleted = 0
    for k in range(2, kmax + 1):
        core_wise_len.append(len(corona_clusters_nodes[k]))
        total_deletion += len(corona_clusters_nodes[k])
        # print(k, len(corona_clusters_nodes[k]), len(corona_clusters_edges[k]), corona_clusters_nodes[k])
        # continue
        for cluster_edge_list, cluster_node_list in zip(corona_clusters_edges[k], corona_clusters_nodes[k]):
            # print(cluster_node_list, cluster_edge_list)
            edge_deleted += 1
            RA_dependency.addEdgeParent(cluster_edge_list, edge_parent)

    return edge_parent

def endPointSumEdgeRemoval(graph, core_num, percentage, dest_name, type='Degree', budget=None, edge_parent=None):
    if type == 'Core_strength':
        core_strength = global_main.getCoreStrength(graph, core_num)
    elif type == 'Degree' or type == 'Core_number':
        node_cirs_ciis = []
    else:
        RS_ID, RS_OD, run_time = RA_dependency.dependencyGraphRemoval(graph)
        if type == 'inDegree_removal':
            node_cirs_ciis = RS_ID
        else:
            node_cirs_ciis = RS_OD
        # CI_decrease, node_cirs_ciis, edge_parent, node_parent = remove_algo.dependencyGraphRemoval(graph)
        if edge_parent is None:
            edge_parent = getEqualEdgeParent(graph, core_num)

    changed_core_cnt = 0
    percentage_edge_number = int(graph.number_of_edges() * percentage / 100)
    if budget:
        percentage_edge_number = budget
    #print("Number of edges to be deleted -->>>    ", percentage_edge_number)
    deg_sum_dict = {}
    for edge in graph.edges():
        if type == 'Core_number':
            deg_sum_dict[edge] = core_num[edge[0]] + core_num[edge[1]]
        elif type == 'Core_strength':
            deg_sum_dict[edge] = core_strength[edge[0]] + core_strength[edge[1]]
        elif type == 'Degree':
            deg_sum_dict[edge] = graph.degree[edge[0]] + graph.degree[edge[1]]
        else:
            if edge[0] in node_cirs_ciis and edge[1] in node_cirs_ciis:
                deg_sum_dict[edge] = node_cirs_ciis[edge[0]] + node_cirs_ciis[edge[1]]
            else:
                deg_sum_dict[edge] = 100000
    if type == 'Degree' or type == 'Core_number' or type == 'Core_strength' or type == 'outDegree_removal':
        deg_sum_dict = dict(sorted(deg_sum_dict.items(), key=lambda item: item[1], reverse=True))
    else:
        deg_sum_dict = dict(sorted(deg_sum_dict.items(), key=lambda item: item[1], reverse=False))

    delete_edge_list, delete_complement_edge_dict = [], {}
    cnt, last_edge = 0, None
    for edge in deg_sum_dict.keys():
        #print(edge, deg_sum_dict[edge])
        # edge_parent = None
        if edge_parent == None:
            delete_edge_list.append(edge)
            cnt += 1
        elif edge in edge_parent:
            cur_edge_parent = edge_parent[edge]
            if cur_edge_parent not in delete_edge_list:
                delete_edge_list.append(cur_edge_parent)
                cnt += 1
            else:
                delete_complement_edge_dict[edge] = core_num[edge[0]] + core_num[edge[1]]
        if cnt == percentage_edge_number:
            break

    delete_complement_edge_dict = dict(sorted(delete_complement_edge_dict.items(), key=lambda item: item[1], reverse=False))
    delete_complement_edge_list = [ elem for elem in deg_sum_dict.keys() if elem not in delete_edge_list]
    # print(len(delete_complement_edge_list), len(delete_complement_edge_dict), len(delete_edge_list), len(deg_sum_dict))

    # print(percentage_edge_number, len(delete_edge_list))

    random_list, random_exp = {}, 1
    if cnt < percentage_edge_number:
        # print("Random needed")
        random_exp = 10
        residual_edge_cnt = percentage_edge_number - len(delete_edge_list)
        # delete_edge_list.extend(random.sample(list(delete_complement_edge_list), residual_edge_cnt))
        for exp in range (0, random_exp):
            random_list[exp] = random.sample(list(delete_complement_edge_list), residual_edge_cnt)


    # print("Len of delete list ", len(delete_edge_list))
    avg_result_dict, avg_affected_percentage = {}, {}
    for b in range(0, percentage_edge_number, 50):
        avg_result_dict[b+50], avg_affected_percentage[b+50] = 0, 0
    result_dict, affected_percentage = {}, {}
    for exp in range (0, random_exp):
        if random_exp == 1:
            random_list[0] = []
        temp_graph = graph.copy()
        result_dict[exp], affected_percentage[exp] = {}, {}
        cur_delete_edge_list = delete_edge_list.copy() + random_list[exp]
        for b in range(0, percentage_edge_number, 50):
            temp_graph.remove_edges_from(cur_delete_edge_list[b:b + 50])
            temp_core_number = nx.core_number(temp_graph)
            changed_core_cnt = coreNumberChange(graph, core_num, temp_core_number)
            # print("budget and accuracy  ", b + 50, changed_core_cnt)
            result_dict[exp][b + 50] = changed_core_cnt
            avg_result_dict[b + 50] += result_dict[exp][b + 50]
            affected_percentage[exp][b+50] = np.round(changed_core_cnt / graph.number_of_nodes() * 100, 2)
            avg_affected_percentage[b+50] += affected_percentage[exp][b+50]

    for b in range(0, percentage_edge_number, 50):
        avg_affected_percentage[b + 50] = np.round(avg_affected_percentage[b + 50] / random_exp, 2)
    
    print(type, list(avg_affected_percentage.values()), sep="")
    return list(avg_affected_percentage.values())


def simulateEdgeRemoval(graph, graph_name):
    core_num = nx.core_number(graph)
    top_n_percentage = 20
    edge_parent = None
    random_result = randomEdgeRemoval(graph, core_num, top_n_percentage, graph_name, exp = 5, budget = 1000)
    degree_result = endPointSumEdgeRemoval(graph, core_num, top_n_percentage, graph_name, type = 'Degree', budget = 1000, edge_parent = edge_parent)
    core_num_result = endPointSumEdgeRemoval(graph, core_num, top_n_percentage, graph_name, type='Core_number', budget = 1000, edge_parent = edge_parent)
    CS_result = endPointSumEdgeRemoval(graph, core_num, top_n_percentage, graph_name, type='Core_strength', budget = 1000, edge_parent = edge_parent)

    edge_parent = getEqualEdgeParent(graph, core_num)
    inDegree_removal = endPointSumEdgeRemoval(graph, core_num, top_n_percentage, graph_name, type='inDegree_removal', budget=1000, edge_parent=edge_parent)
    outDegree_removal = endPointSumEdgeRemoval(graph, core_num, top_n_percentage, graph_name, type='outDegree_removal', budget=1000, edge_parent=edge_parent)

    draw_paper_performance(outDegree_removal, inDegree_removal, CS_result, random_result, degree_result,
                           core_num_result, graph_name, save_fig = 0)


if __name__ == '__main__':
    file_name = sys.argv[1]
    # print(file_name)
    graph = global_main.readGraph(file_name)
    graph_matrix = nx.to_numpy_matrix(graph)
    graph = nx.from_numpy_matrix(graph_matrix)

    file_name = file_name.split('/')
    graph_name = file_name[len(file_name) - 1].split('.')[0]
    # print(graph_name)

    results = simulateEdgeRemoval(graph, graph_name)