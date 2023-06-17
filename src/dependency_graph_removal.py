import main as global_main
import time
from datetime import datetime
import sys
import networkx as nx
import numpy as np
import streaming_algorithm as sa
import random

dec_k_in_fun = 1

def findKcorona(graph, k, core_num, cs):
    temp_nodes = []
    for node in graph.nodes():
        # nodes = (v for v in core if k_filter(v, k, core))
        if core_num[node] == k and cs[node] == 1:
            temp_nodes.append(node)
    return graph.subgraph(temp_nodes).copy()

def findCoronaClusters(graph, kcore, core_num, k, cs=[]):
    # corona_clusters1 = nx.k_corona(graph, k, core_num)
    corona_clusters = findKcorona(graph, k, core_num, cs)
    # diff = nx.difference(corona_clusters, corona_clusters1)
    # print(diff.edges())

    # connected_clusters = nx.connected_components(corona_clusters)
    connected_clusters = [graph.subgraph(c).copy() for c in nx.connected_components(corona_clusters)]

    corona_clusters_nodes = []
    corona_clusters_edges = []
    for cluster in connected_clusters:
        equal_corona_edge = []
        for node in cluster.nodes():
           # print(node)
           #  for v in kcore[k].neighbors(node):
            for v in graph.neighbors(node):
                if core_num[v] >= core_num[node]:
                    x, y = node, v
                    if node > v:
                        x, y = v, node
                    if (x, y) not in equal_corona_edge:
                        equal_corona_edge.append((x, y))

        corona_clusters_nodes.append(cluster.nodes())
        corona_clusters_edges.append((equal_corona_edge))

    #print(corona_clusters_nodes)
    return corona_clusters_nodes, corona_clusters_edges

def addEdgeParent(edge_list, edge_parent):
    master_edge = edge_list[0]
    for edge in edge_list:
        edge_parent[edge] = master_edge

def addNodeParent(edge_list, changed_core_nodes, node_parent):
    #print("addNodeParnet Called-------------------------   ", edge_list[0], changed_core_nodes)
    master_edge = edge_list[0]
    for node in changed_core_nodes:
        node_parent[node] = master_edge

def naiveDependencyGraphRemoval(graph, traversal='Subcore', critical_size=1000, subcore_node_parent=None, subcore_parent_H=None, subcore_parent_cd=None):
    # print(traversal)
   
    core_num = nx.core_number(graph)
    edge_list = graph.edges()
    only_IA_time = 0
    tmp_time = time.time()
    DDG = nx.DiGraph()
    DDG.add_nodes_from(list(graph.nodes()))
    # temp_graph = graph.copy()
    for edge in edge_list:
        if dec_k_in_fun:
            core_num_copy = core_num.copy()
        else:
            core_num_copy = core_num

        if traversal == 'Subcore':
            core_changed_nodes = sa.subCoreRemoveEdge(graph, core_num_copy, edge, change_k=dec_k_in_fun)
        elif traversal == 'Traversal':
            core_changed_nodes = sa.traversalRemoveEdge(graph, core_num_copy, edge, traversal='Traversal', pc_time=[], change_k=dec_k_in_fun)
        graph.add_edge(*edge)
        
        if edge[0] in core_changed_nodes:
            DDG.add_edge(edge[1], edge[0])
        if edge[1] in core_changed_nodes:
            DDG.add_edge(edge[0], edge[1])
        
    naive_end_time = np.round((time.time() - tmp_time), 3)

    RS_ID, RS_OD = {}, {}
    for node in graph.nodes():
        if DDG.in_degree(node) != 0:
            RS_ID[node] = np.round(1 / DDG.in_degree(node), 3)
        else:
            RS_ID[node] = DDG.number_of_nodes() # high resilience
        RS_OD[node] = DDG.out_degree(node)

    return RS_ID, RS_OD, naive_end_time
    # print("naive IA time-->>  ", traversal, np.round(only_IA_time, 3))

def removeEdgeExperimentRandom(graph, core_num, cs, node_parent, CI_decrease, DDG=None, random_removal_cnt=5):
    # print(list(graph.neighbors(1)))
    # print("In exp  ", DDG.number_of_nodes())
    total_decrease = 0
    for node in graph.nodes():
        decrease_cnt = 0
        if len(list(graph.neighbors(node))) <= random_removal_cnt:
            random_neighbor_nodes = graph.neighbors(node)
        else:
            random_neighbor_nodes = random.sample(list(graph.neighbors(node)), random_removal_cnt)

        if cs[node] == 1:
            for v in random_neighbor_nodes:
                if core_num[v] >= core_num[node]:
                    decrease_cnt += 1
                    if DDG:
                        DDG.add_edge(v, node)
        else:
            for v in random_neighbor_nodes:
                if core_num[v] == core_num[node] and cs[v] == 1:
                    if node in node_parent.keys():
                        if node_parent[node] == node_parent[v]:
                            decrease_cnt += 1
                            if DDG:
                                DDG.add_edge(v, node)

        CI_decrease[node] = decrease_cnt
        total_decrease += decrease_cnt

    # print(total_decrease)
    return CI_decrease

def removeEdgeExperiment(graph, core_num, cs, node_parent, CI_decrease, DDG=None):
    total_decrease = 0
    for node in graph.nodes():
        decrease_cnt = 0
        if cs[node] == 1:
            for v in graph.neighbors(node):
                if core_num[v] >= core_num[node]:
                    decrease_cnt += 1
                    if DDG:
                        DDG.add_edge(v, node)
        else:
            for v in graph.neighbors(node):
                if core_num[v] == core_num[node] and cs[v] == 1:
                    if node in node_parent.keys():
                        if node_parent[node] == node_parent[v]:
                            decrease_cnt += 1
                            if DDG:
                                DDG.add_edge(v, node)

        CI_decrease[node] = decrease_cnt
        total_decrease += decrease_cnt
    #print("k and cnt ==   ", k, cnt, cnt2 + cnt3, cnt2, cnt3)
    #print("Total decrease --->>>  ", k, total_decrease)
    return CI_decrease

def removeCandidateEdge(graph, core_num, edge_list, traversal='Subcore', critical_size=1000,
                        subcore_node_parent=None, subcore_parent_H=None, subcore_parent_cd=None):
    # temp_graph1 = graph.copy()
    master_edge = edge_list[0]
    # temp_graph.remove_edge(*master_edge)
    # temp_core_num = nx.core_number(temp_graph)
    # temp_graph.add_edge(*master_edge)

    core_changed_nodes1 = []

    # for node in graph.nodes():
    #    if core_num[node] > temp_core_num[node]:
    #        core_changed_nodes.append(node)

    alg_time = 0
    if dec_k_in_fun:
        core_num_copy = core_num.copy()
    else:
        core_num_copy = core_num

    tmp_time = time.time()
    if traversal == 'Subcore':
        core_changed_nodes1 = sa.subCoreRemoveEdge(graph, core_num_copy, master_edge, change_k=dec_k_in_fun)
    elif traversal == 'Traversal':
        core_changed_nodes1 = sa.traversalRemoveEdge(graph, core_num_copy, master_edge, traversal='Traversal', pc_time=[], change_k=dec_k_in_fun)
    graph.add_edge(*master_edge)

    alg_time += (time.time()-tmp_time)
    # print("Only removal time   -->>  ", traversal, np.round(alg_time, 3))

    # print(core_changed_nodes1.sort())
    # print(core_changed_nodes.sort())
    return core_changed_nodes1

def dependencyGraphRemoval(graph, DDG=None, traversal='Subcore', critical_size=1000, subcore_node_parent=None,
                           subcore_parent_H=None, subcore_parent_cd=None, random_exp=0):
    core_num = nx.core_number(graph)
    cn = set(core_num.values())

    kcore = {}

    cs = global_main.getCoreStrength(graph, core_num)
    kmax = max(cn)
    corona_clusters_nodes, corona_clusters_edges = {}, {}

    alg_start_time = time.time()
    corona_time = time.time()
    for k in range(1, kmax + 1):
        corona_clusters_nodes[k], corona_clusters_edges[k] = findCoronaClusters(graph, kcore, core_num, k, cs=cs)

    corona_time = np.round(time.time() - corona_time, 3)

    only_IA_time = 0
    tmp_time = time.time()

    edge_parent, node_parent = {}, {}
    edge_deleted = 0
    for k in range(1, kmax + 1):
        for cluster_edge_list, cluster_node_list in zip(corona_clusters_edges[k], corona_clusters_nodes[k]):
            edge_deleted += 1
            # Compute KAES (S)
            addEdgeParent(cluster_edge_list, edge_parent)
            # Compute CCN_KAES(S)
            changed_core_nodes = removeCandidateEdge(graph, core_num, cluster_edge_list, traversal=traversal, critical_size=critical_size, subcore_node_parent=
                                subcore_node_parent, subcore_parent_H=subcore_parent_H, subcore_parent_cd=subcore_parent_cd)
            addNodeParent(cluster_edge_list, changed_core_nodes, node_parent)

    only_IA_time += (time.time() - tmp_time)


    CI_decrease = {}
 
    DDG = nx.DiGraph()
    DDG.add_nodes_from(list(graph.nodes()))
    CI_decrease = removeEdgeExperiment(graph, core_num, cs, node_parent, CI_decrease, DDG)
    
    alg_end_time = np.round(time.time() - alg_start_time, 3)
    
    RS_ID, RS_OD = {}, {}
    for node in graph.nodes():
        if DDG.in_degree(node) != 0:
            RS_ID[node] = np.round(1 / DDG.in_degree(node), 3)
        else:
            RS_ID[node] = DDG.number_of_nodes() # high resilience
        RS_OD[node] = DDG.out_degree(node)

    return RS_ID, RS_OD, alg_end_time
    
if __name__ == '__main__':
    start_time = datetime.now()
    file_name = sys.argv[1]
    # print(file_name)
    graph = global_main.readGraph(file_name)
    graph_matrix = nx.to_numpy_matrix(graph)
    graph = nx.from_numpy_matrix(graph_matrix)
    # RS_ID, RS_OD = dependencyGraphRemoval(graph)
    
    traversal, approach = 'Traversal', 'Heuristic'
    run_time = None
    if len(sys.argv) > 2:
        traversal = sys.argv[2]
    if len(sys.argv) > 3 and sys.argv[3]=='Naive':
        approach = sys.argv[3]
        RS_ID, RS_OD, run_time = naiveDependencyGraphRemoval(graph, traversal=traversal)
    else:
        if traversal == 'Traversal' or traversal == 'Subcore':
            RS_ID, RS_OD, run_time = dependencyGraphRemoval(graph, traversal=traversal)
        else:
             print("Invalid input. Please provide the correct input format.")

    if run_time:
        print("Streaming algorithm type, Node Strength calculation Approach, and runtime: ", traversal, approach, run_time)
    # print(RS_ID)
    # dependencyGraphRemoval(graph, DDG=DDG, traversal=traversal, critical_size=adaptive_critical_size, subcore_node_parent=snp,
                                    # subcore_parent_H=sph, subcore_parent_cd=spcd)
