import main as global_main
import time
from datetime import datetime
import sys
import networkx as nx
import numpy as np
import streaming_algorithm as sa
import random
from collections import defaultdict

def getD2NeighborList(common_count_edge, common_one_new_edge):
    total_common_cnt, d2_edge_count, d2_neighbor_list, d2_edge_list, mx_common_cnt = {}, {}, {}, {}, 0

    for edge in common_one_new_edge:
        count = common_count_edge[edge[0]][edge[1]]
        mx_common_cnt = max(mx_common_cnt, count)
        if count not in total_common_cnt:
            total_common_cnt[count] = 1
        else:
            total_common_cnt[count] += 1
        if edge[0] not in d2_edge_count:
            d2_edge_count[edge[0]] = 1
            d2_neighbor_list[edge[0]] = [edge[1]]
            d2_edge_list[edge[0]] = [edge]
        else:
            d2_edge_count[edge[0]] += 1
            d2_neighbor_list[edge[0]].append(edge[1])
            d2_edge_list[edge[0]].append(edge)

        if edge[1] not in d2_edge_count:
            d2_edge_count[edge[1]] = 1
            d2_neighbor_list[edge[1]] = [edge[0]]
            d2_edge_list[edge[1]] = [edge]
        else:
            d2_edge_count[edge[1]] += 1
            d2_neighbor_list[edge[1]].append(edge[0])
            d2_edge_list[edge[1]].append(edge)
    return total_common_cnt, d2_edge_count, d2_neighbor_list, d2_edge_list, mx_common_cnt

def getPossibleEdges(graph):
    nodes = graph.nodes()
    sibling_nodes,  possible_edges_node = {}, {}
    common_one_new_edge, common_two_new_edge = [], []
    common_count, common_node_list_edge = defaultdict(dict), defaultdict(dict)
    for node in nodes:
        sibling = []
        for child in graph[node]:
            sibling.append(child)
        sibling_nodes[node] = sibling

        edge_list = []
        for i in range(0, len(sibling)):
            for j in range(i + 1, len(sibling)):
                if graph.has_edge(sibling[i], sibling[j]):
                    continue

                if sibling[i] in common_count and sibling[j] in common_count[sibling[i]]:
                    common_count[sibling[i]][sibling[j]] += 1
                    common_count[sibling[j]][sibling[i]] += 1
                    common_node_list_edge[sibling[i]][sibling[j]].append(node)
                    common_node_list_edge[sibling[j]][sibling[i]].append(node)
                else:
                    edge_list.append((sibling[i], sibling[j]))
                    common_count[sibling[i]][sibling[j]] = 1
                    common_count[sibling[j]][sibling[i]] = 1
                    common_one_new_edge.append((sibling[i], sibling[j]))
                    common_node_list_edge[sibling[i]][sibling[j]] = [node]
                    common_node_list_edge[sibling[j]][sibling[i]] = [node]

        possible_edges_node[node] = edge_list

    for edge in common_one_new_edge:
        if common_count[edge[0]][edge[1]] >= 2:
            common_two_new_edge.append(edge)
    # print("all unique edges, commmon one and two ==>>>>   ", len(common_one_new_edge), len(common_two_new_edge))

    return possible_edges_node, common_count, common_one_new_edge, common_two_new_edge, common_node_list_edge

def getInsertionCandidateGraph(graph, dest_name=None, random_cutoff=5, s1='S1'):
    graph_start_time = datetime.now()

    core_num = nx.core_number(graph)
    kcore = global_main.generateCoreSubgraph(graph, core_num)
    kshell = global_main.generateShellSubgraph(graph, core_num)
    knumber = [core_num[u] for u in core_num]
    possible_edges_node, common_count_edge, common_one_new_edge, common_two_new_edge, common_node_list_edge = getPossibleEdges(
        graph)
    ICG = nx.Graph()
    ICG.add_nodes_from(list(graph.nodes()))
    if s1 == 'S1':
        total_common_cnt, d2_edge_count, d2_neighbor_list, d2_edge_list, mx_common_cnt = getD2NeighborList(common_count_edge, common_one_new_edge)
    else:
        total_common_cnt, d2_edge_count, d2_neighbor_list, d2_edge_list, mx_common_cnt = getD2NeighborList(common_count_edge, common_two_new_edge)

    d2_edge_count = dict(sorted(d2_edge_count.items(), key=lambda item: item[0], reverse=False))
    # print(dest_name, sum(list(d2_edge_count.values()))/2)
    tmp_cnt = 0
    for node in graph.nodes():
        # All random insertion
        if node not in d2_edge_list or d2_edge_list[node] == 0:
            d2_neighbor_list[node] = []
            for i in range(0, random_cutoff):
                v = random.choice(list(ICG.nodes()))
                while (v in d2_neighbor_list[node] or ICG.has_edge(node, v)):
                    v = random.choice(list(ICG.nodes()))
                ICG.add_edge(node, v)
        # choose k node randomly from S1
        elif d2_edge_count[node] >= random_cutoff:
            inserted_edges = random.sample(d2_edge_list[node], random_cutoff)
            ICG.add_edges_from(inserted_edges)
        # choose all node from S1 and k-|S1(u)| node randomly from V
        else:
            ICG.add_edges_from(d2_edge_list[node])
            random_insert_cnt = random_cutoff - d2_edge_count[node]
            for i in range(0, random_insert_cnt):
                v = random.choice(list(ICG.nodes()))
                while (v in d2_neighbor_list[node] or ICG.has_edge(node, v)):
                    v = random.choice(list(ICG.nodes()))
                ICG.add_edge(node, v)

    # print(dest_name,  graph.number_of_edges(), len(common_one_new_edge), ICG.number_of_edges(), graph.number_of_nodes()*5)
    # print(dest_name, len(common_one_new_edge), ICG.number_of_edges())

    return ICG

def getHigherCoreDegree(graph, cnumber, nodes=None):
    data = {}
    if nodes is None:
        nodes = graph.nodes()

    for u in nodes:
        neighbors = graph.neighbors(u)
        hcd = len([v for v in neighbors if cnumber[u] < cnumber[v]])
        data[u] = hcd

    return data

def calculateNeighborHCD(graph, core_num, kshell, hcd):

    neighbor_HCD = {}
    for node in graph.nodes():
        neighbor_HCD[node] = []
        for nei in kshell[core_num[node]].neighbors(node):
            #print(nei, core_num[nei], hcd[nei])
            neighbor_HCD[node].append(hcd[nei])
        neighbor_HCD[node] = sorted(neighbor_HCD[node], reverse=True)
        #print(neighbor_HCD[node])
    return neighbor_HCD

def CIincreaseCheck(CI_increase, changed_core_nodes, edge, IDG = None):
    if edge[0] in changed_core_nodes:
        CI_increase[edge[0]] += 1
        if IDG:
            IDG.add_edge(edge[1], edge[0])
    if edge[1] in changed_core_nodes:
        CI_increase[edge[1]] += 1
        if IDG:
            IDG.add_edge(edge[0], edge[1])

def streamingAddEdgeNaive(graph, candidate_edges=None, IDG=None, traversal='Traversal', critical_size=None, subcore_node_parent=None, 
                          subcore_parent_H=None, subcore_parent_cd=None, dest_name= None):
    core_num = nx.core_number(graph)
    temp_graph = graph.copy()
    solution_edges, CI_increase, hcd = set(), {}, {}
    node_common_two_count = {}
    for node in graph.nodes():
        CI_increase[node], node_common_two_count[node] = 0, 0
    # hcd = IA.getHigherCoreDegree(graph, core_num)
    if candidate_edges == None:
        # possible_edges_node, common_count, candidate_edges, common_two_new_edge, common_node_list_edge = getPossibleEdges(graph)
        ICG = getInsertionCandidateGraph(graph, dest_name=dest_name, random_cutoff=5, s1='S1')
        candidate_edges = []
        for edge in list(ICG.edges()):
            if not graph.has_edge(*edge):
                candidate_edges.append(edge)
    if IDG == None:
        IDG = nx.DiGraph()
        IDG.add_nodes_from(list(graph.nodes()))

    naive_start_time = time.time()
    sc_time = []
    for edge in candidate_edges:
        # check for changed core nodes in naive
        if traversal == 'Traversal':
            changed_core_nodes = sa.traversalInsertEdge(temp_graph, core_num, edge, traversal=traversal, pc_time=sc_time)
        else:
            changed_core_nodes = sa.subCoreInsertEdge(temp_graph, core_num, edge, sc_time=sc_time, subcore=traversal)
        temp_graph.remove_edge(*edge)
        if len(changed_core_nodes) >= 1:
            CIincreaseCheck(CI_increase, changed_core_nodes, edge, IDG)
            solution_edges.add(edge)

    naive_end_time = np.round(time.time() -naive_start_time, 3)
    # print("Summation of finding the %s   --->>>>   " %(traversal), np.round(sum(sc_time), 3))

    # print("Naive approach run time   ", naive_end_time - naive_start_time)
    IS_ID, IS_OD = {}, {}
    for node in graph.nodes():
        if IDG.in_degree(node) != 0:
            IS_ID[node] = np.round(1 / IDG.in_degree(node), 3)
        else:
            IS_ID[node] = IDG.number_of_nodes()   # High resilience
        IS_OD[node] = IDG.out_degree(node)

    # print(IDG.number_of_edges())

    return IS_ID, IS_OD, naive_end_time


def dependencyGraphInsertion(graph, dest_name=None, candidate_edges=None, IDG=None, traversal='Traversal',
                               subcore_node_parent=None, subcore_parent_H=None, subcore_parent_cd=None, adaptive_critical_size=1000):
    core_num = nx.core_number(graph)
    kshell = global_main.generateShellSubgraph(graph, core_num)
    if candidate_edges == None:
        # possible_edges_node, common_count, candidate_edges, common_two_new_edge, common_node_list_edge = getPossibleEdges(graph)
        ICG = getInsertionCandidateGraph(graph, dest_name=dest_name, random_cutoff=5, s1='S1')
        candidate_edges = []
        for edge in list(ICG.edges()):
            if not graph.has_edge(*edge):
                candidate_edges.append(edge)
    if IDG == None:
        IDG = nx.DiGraph()
        IDG.add_nodes_from(list(graph.nodes()))

    hcd = getHigherCoreDegree(graph, core_num)
    neighbor_HCD = calculateNeighborHCD(graph, core_num, kshell, hcd)

    temp_graph = graph.copy()
    solution_edges, CI_increase = set(), {}
    node_common_two_count = {}

    for node in graph.nodes():
        CI_increase[node], node_common_two_count[node] = 0, 0

    alg_start_time = time.time()
    sc_time = []

    for edge in candidate_edges:
        u, v = edge[0], edge[1]
        if core_num[u] < core_num[v]:
            u, v = edge[1], edge[0]
        node_common_two_count[u] += 1
        node_common_two_count[v] += 1

        if core_num[u] == core_num[v]:
            if hcd[v] == core_num[v] and hcd[u] == core_num[u]:
                solution_edges.add(edge)
                CI_increase[u] += 1
                CI_increase[v] += 1
                if IDG:
                    IDG.add_edge(u, v)
                    IDG.add_edge(v, u)
            else:
                # check for changed core nodes using different insertion algorithm
                if traversal == 'Traversal':
                    changed_core_nodes = sa.traversalInsertEdge(temp_graph, core_num, edge, traversal=traversal, pc_time=sc_time)
                elif traversal == 'Subcore':
                    changed_core_nodes = sa.subCoreInsertEdge(temp_graph, core_num, edge, sc_time=sc_time, subcore=traversal)
                temp_graph.remove_edge(*edge)
                if len(changed_core_nodes) >= 1:
                    CIincreaseCheck(CI_increase, changed_core_nodes, edge, IDG)
                    solution_edges.add(edge)

        else:
            if hcd[v] == core_num[v]:
                solution_edges.add(edge)
                CI_increase[v] += 1
                if IDG:
                    IDG.add_edge(u, v)
            elif core_num[v] - hcd[v] == 1:
                if len(neighbor_HCD[v]) > 0 and neighbor_HCD[v][0] >= core_num[v]:
                    solution_edges.add(edge)
                    CI_increase[v] += 1
                    if IDG:
                        IDG.add_edge(u, v)
            else:
                # check for changed core nodes in naive
                if traversal == 'Traversal':
                    changed_core_nodes = sa.traversalInsertEdge(temp_graph, core_num, edge, traversal=traversal, pc_time=sc_time)
                elif traversal == 'Subcore':
                    changed_core_nodes = sa.subCoreInsertEdge(temp_graph, core_num, edge, sc_time=sc_time, subcore=traversal)
                temp_graph.remove_edge(*edge)
                if len(changed_core_nodes) >= 1:
                    CIincreaseCheck(CI_increase, changed_core_nodes, edge, IDG)
                    solution_edges.add(edge)

    alg_end_time = np.round(time.time() - alg_start_time, 3)
    #print(dest_name, common_greater_one, total_time, no_streaming_perform, common_greater_one - no_streaming_perform, len(solution_edges), streaming_perform)
    # print(total_time, len(solution_edges), len(candidate_edges), IDG.number_of_edges())
    # print("Algorithm total time taken   ", alg_end_time)

    IS_ID, IS_OD = {}, {}
    for node in graph.nodes():
        if IDG.in_degree(node) != 0:
            IS_ID[node] = np.round(1 / IDG.in_degree(node), 3)
        else:
            IS_ID[node] = IDG.number_of_nodes()   # High resilience
        IS_OD[node] = IDG.out_degree(node)

    # print(IDG.number_of_edges())

    return IS_ID, IS_OD, alg_end_time

if __name__ == '__main__':
    start_time = datetime.now()
    file_name = sys.argv[1]
    # print(file_name)
    graph = global_main.readGraph(file_name)
    graph_matrix = nx.to_numpy_matrix(graph)
    graph = nx.from_numpy_matrix(graph_matrix)

    traversal, approach = 'Traversal', 'Heuristic'
    run_time = None
    if len(sys.argv) > 2:
        traversal = sys.argv[2]
    if len(sys.argv) > 3 and sys.argv[3]=='Naive':
        approach = sys.argv[3]
        IS_ID, IS_OD, run_time = streamingAddEdgeNaive(graph, traversal=traversal)
    else:
        if traversal == 'Traversal' or traversal == 'Subcore':
            IS_ID, IS_OD, run_time = dependencyGraphInsertion(graph, traversal=traversal)
        else:
             print("Invalid input. Please provide the correct input format.")

    if run_time:
        print("Streaming algorithm type, Node Strength calculation Approach, and runtime: ", traversal, approach, run_time)
    # hcd, node_common_two_count, CI_increase = optimizedAddEdgeExperiment(graph, candidate_edges=candidate_edges, IDG=IDG, traversal=traversal,
    #                                                                          subcore_node_parent=snp, subcore_parent_H=sph, subcore_parent_cd=spcd)