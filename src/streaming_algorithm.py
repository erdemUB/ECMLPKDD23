import time
import csv
import os 
import sys
import networkx as nx
import numpy as np
import random
#import improve_resilience
import numpy
import pathlib
from datetime import datetime
import glob
from networkx.classes.function import neighbors
from scipy.special import comb
from scipy.stats import binom
import sympy
from itertools import combinations 
from ast import Num
from asyncore import write
import pickle

import collections
import heapq
import main as global_main
import bucket_sort

dec_k_in_fun = 0


def findSubCore1(graph, core_num, node):
    visited = []
    cd = {}
    H = nx.Graph()

    k = core_num[node]
    queue = deque([node])
    visited.append(node)

    while queue:  # Creating loop to visit each node
        u = queue.popleft()
        cd[u] = 0

        for v in graph[u]:
            if core_num[v] >= core_num[u]:
                cd[u] += 1
                if core_num[v] == k and v not in visited:
                    visited.append(v)
                    queue.append(v)
                    H.add_edge(u, v)

    return H, cd

def findSubCore(G, core_num, u):
    H = nx.Graph()
    k = core_num[u]
    cd = {}

    visited, queue = set([u]), collections.deque([u])

    while queue:
        v = queue.popleft()
        H.add_node(v)
        cd[v] = 0
        for w in G.neighbors(v):
            if core_num[w] >= k:
                cd[v] += 1
                if core_num[w] == k:
                    H.add_edge(v, w)
                if core_num[w] == k and w not in visited:
                    queue.append(w)
                    # H.add_edge(v, w)
                    visited.add(w)
    return H, cd

def findPureCore(G, core_num, u):
    H = nx.Graph()
    k = core_num[u]
    cd = {}
    
    visited, queue = set([u]), collections.deque([u])
    
    while queue:
        v = queue.popleft()
        H.add_node(v)
        cd[v] = 0
        for w in G.neighbors(v):
            if core_num[w] > k or (core_num[w] == k and coreDegree(G, core_num, u, k) > k):
                cd[v] += 1
                if core_num[w] == k:
                    H.add_edge(v, w)
                if core_num[w] == k and w not in visited:
                    queue.append(w)
                    # H.add_edge(v, w)
                    visited.add(w)
    return H, cd

def computeMCD(graph, cnumber):
    mcd = {}
    for u in graph.nodes():
        mcd[u] = 0
        for v in graph.neighbors(u):
            if cnumber[u] <= cnumber[v]:
                mcd[u] += 1
    return mcd

def computePCD(graph, cnumber, mcd):
    pcd = {}
    for u in graph.nodes():
        pcd[u] = 0
        for w in graph.neighbors(u):
            if  (cnumber[w] == cnumber[u] and mcd[w] > cnumber[u]) or (cnumber[w] > cnumber[u]):
                pcd[u] += 1
    return pcd

def coreDegree(graph, cnumber, u, core_of_root):
    mcd = 0
    for v in graph.neighbors(u):
        if cnumber[v] >= core_of_root:
            mcd += 1
    return mcd

def pureCoreDegree(graph, cnumber, u, core_of_root):
    pcd = 0
    for w in graph.neighbors(u):
        mcd_w = coreDegree(graph, cnumber, w, core_of_root)
        if (cnumber[w] == core_of_root and mcd_w > core_of_root) or (cnumber[w] > core_of_root):
            pcd += 1
    return pcd

def prepareRCD(graph, core_num):
    mcd = computeMCD(graph, core_num)
    pcd = computePCD(graph, core_num, mcd)
    return mcd, pcd

def subCoreInsertEdge(G, core_num, edge, sc_time=[], subcore='Subcore', H=None, cd=None):
    u1 = edge[0]
    u2 = edge[1]
    root = u1
    if core_num[u2] < core_num[u1]:
        root = u2

    G.add_edge(u1, u2)
    start_time = time.time()
    if subcore == 'Subcore':
        if H == None:
            H, cd = findSubCore(G, core_num, root)
        # else:
        #     H, cd = H.copy(), cd.copy()
    else:
        H, cd = findPureCore(G, core_num, root)
    sc_time.append(time.time()-start_time)
    # return
    k = core_num[root]

    bs = bucket_sort.Bucket(min(cd.values()), max(cd.values()))
    bs.initializeBucket(max(cd.values()))
    for v in cd.keys():
        bs.insertBucket(v, cd[v])

    changed_core_nodes = set()
    while bs.cur_min_buck_id <= bs.total_bucket:
        v, cd_v = bs.popMinFromBucket()
        if v == None or cd_v == None:
            break
        if cd_v <= k:
            for w in H.neighbors(v):
                if w in H and cd[w] > cd[v]:
                    bs.decCDValue(w, cd[w])
                    cd[w] -= 1
        else:
            changed_core_nodes.add(v)
            while bs.cur_min_buck_id <= bs.total_bucket:# and len(bs.bucket_array[bs.cur_min_buck_id]) > 0:
                v, cd_v = bs.popMinFromBucket()
                if v != None and cd_v:
                    changed_core_nodes.add(v)
                    # comment out next line to update core number
                    # core_num[v] += 1
                else:
                    break
            break


    # print((changed_core_nodes))
    return list(changed_core_nodes)

def propagateEviction(graph, core_num, cd, evicted, k, v):
    evicted[v] = 1
    for w in graph.neighbors(v):
        if core_num[w] == k:
            cd[w] = cd[w] - 1
            if cd[w] == k and evicted[w] == 0:
                propagateEviction(graph, core_num, cd, evicted, k, w)

def traversalInsertEdge(G, core_num, edge, mcd=None, pcd=None, traversal='Traversal', pc_time=[]):
    u1 = edge[0]
    u2 = edge[1]
    root = u1
    if core_num[u2] < core_num[u1]:
        root = u2
    G.add_edge(u1, u2)
    pre_calculate = 0
    # Calculated all nodes mcd and mcd. Sometime performs better.
    if pre_calculate:
        mcd, pcd = prepareRCD(G, core_num)
    stack = collections.deque()
    cd, visited, evicted, vis_list = {}, {}, {}, set()
    for v in G.nodes():
        cd[v], visited[v], evicted[v] = 0, 0, 0
    k = core_num[root]
    if pre_calculate:
        cd[root] = pcd[root]
    else:
        cd[root] = pureCoreDegree(G, core_num, root, k)
    stack.append(root)
    visited[root] = 1
    vis_list.add(root)

    while stack:
        v = stack.pop()
        if cd[v] > k:
            for w in G.neighbors(v):
                if pre_calculate:
                    mcd_w, pcd_w = mcd[w], pcd[w]
                else:
                    start_time = time.time()
                    mcd_w = coreDegree(G, core_num, w, k)
                    pcd_w = pureCoreDegree(G, core_num, w, k)
                    pc_time.append(time.time()-start_time)

                if core_num[w] == k and mcd_w > k and visited[w] == 0:
                    stack.append(w)
                    visited[w] = 1
                    vis_list.add(w)
                    cd[w] = cd[w] + pcd_w
        else:
            if evicted[v] == 0:
                propagateEviction(G, core_num, cd, evicted, k, v)
            #  Next 2 lines for early eviction on insertion dependency algorithm
            if traversal == 'EarlyEviction' and evicted[u1] and evicted[u2]:
                return []


    changed_core_nodes = []
    # print(u1, u2, root, mcd[u1], mcd[u2], pcd[u1], pcd[u2], vis_list, evicted[519], evicted[2055], core_num[u1], core_num[u2])
    for v in vis_list:
        if evicted[v] == 0:
            changed_core_nodes.append(v)
            # comment out next line to update core number
            # core_num[v] += 1

    return changed_core_nodes

def createSingleSubcore(G, core_num, root, visited):
    H = nx.Graph()
    k = core_num[root]
    cd = {}
    queue = collections.deque([root])
    visited[root] = 1
    while queue:
        v = queue.popleft()
        H.add_node(v)
        cd[v] = 0
        for w in G.neighbors(v):
            if core_num[w] >= k:
                cd[v] += 1
                if core_num[w] == k:
                    H.add_edge(v, w)
                if core_num[w] == k and w not in visited:
                    queue.append(w)
                    visited[w] = 1
    return H, cd


def generateAllSubcoreSubgraph(G, core_num):
    visited = {}
    subcore_node_parent, subcore_parent_cd, subcore_parent_H = {}, {}, {}
    subcore_node_cnt = {}
    subcore_size_list = []
    for node in G.nodes():
        if node not in visited:
            H, cd = createSingleSubcore(G, core_num, node, visited)
            for v in H.nodes():
                subcore_node_parent[v] = node
            subcore_parent_cd[node] = cd
            subcore_parent_H[node] = H
            total_node = len(cd)
            if total_node not in subcore_node_cnt:
                subcore_node_cnt[total_node] = 0
            subcore_node_cnt[total_node] += total_node
            subcore_size_list.append(total_node)

    return subcore_node_parent, subcore_parent_cd, subcore_parent_H


def subCoreRemoveEdge(G, core_num, edge, change_k=1, H=None, cd=None):
    root = edge[0]
    if core_num[edge[1]] < core_num[edge[0]]:
        root = edge[1]
    G.remove_edge(*edge)

    if H == None:
        H = nx.Graph()
        cd = {}
        if core_num[edge[1]] != core_num[edge[0]]:
            H, cd = findSubCore(G, core_num, root)
        else:
            H1, cd1 = findSubCore(G, core_num, edge[0])
            H2, cd2 = findSubCore(G, core_num, edge[1])
            H.add_edges_from(H1.edges)
            H.add_edges_from(H2.edges)
            H.remove_edges_from(nx.selfloop_edges(H))

            for key, value in cd1.items():
                cd[key] = value
            for key, value in cd2.items():
                cd[key] = value


    k = core_num[root]

    # print(cd)

    bs = bucket_sort.Bucket(min(cd.values()), max(cd.values()))
    bs.initializeBucket(max(cd.values()))
    for v in cd.keys():
        bs.insertBucket(v, cd[v])

    visited = set([])
    changed_core_nodes = set()
    while bs.cur_min_buck_id <= bs.total_bucket:
        v, cd_v = bs.popMinFromBucket()
        if v == None or cd_v == None:
            break
        bs.processed_id[v] = 1

        if cd_v < k:
            if change_k:
                core_num[v] = k - 1
            changed_core_nodes.add(v)
            for w in G.neighbors(v):
                if w in H and cd[w] > cd[v]:
                    bs.decCDValue(w, cd[w])
                    cd[w] -= 1

    # print("core number and cd of 5   ===============  ", core_num[5], cd[5])
    return list(changed_core_nodes)

def propagateDismissal(graph, core_num, cd, dismissed, visited, k, v, changed_core_nodes, pre_calculate=0, mcd=[]):
    dismissed[v] = 1
    core_num[v] -= 1
    changed_core_nodes.add(v)
    for w in graph.neighbors(v):
        if core_num[w] == k:
            # if visited[w] == 0:
            #     cd[w] = cd[w] + coreDegree(graph, core_num, w, k)
            #     visited[w] = 1
            # cd[w] = cd[w] - 1
            if pre_calculate:
                cd[w] = mcd[w]
            else:
                cd[w] = coreDegree(graph, core_num, w, k)
            if cd[w] < k and dismissed[w] == 0:
                propagateDismissal(graph, core_num, cd, dismissed, visited, k, w, changed_core_nodes)

def traversalRemoveEdge(G, core_num, edge, traversal='Traversal', pc_time=[], change_k=1):
    u1 = edge[0]
    u2 = edge[1]
    root = u1
    if core_num[u2] < core_num[u1]:
        root = u2
    G.remove_edge(u1, u2)
    pre_calculate = 0
    # Calculated all nodes mcd and mcd. Sometime performs better.
    mcd = []
    if pre_calculate:
        mcd = computeMCD(graph, core_num)
    cd, visited, dismissed, changed_core_nodes = {}, {}, {}, set()
    for v in G.nodes():
        cd[v], visited[v], dismissed[v] = 0, 0, 0
    k = core_num[root]

    if core_num[u1] != core_num[u2]:
        if visited[root] == 0:
            visited[root] = 1
            if pre_calculate:
                cd[root] += mcd[root]
            else:
                cd[root] += coreDegree(G, core_num, root, k)
        if dismissed[root] == 0 and cd[root] < k:
            propagateDismissal(G, core_num, cd, dismissed, visited, k, root, changed_core_nodes, pre_calculate=pre_calculate, mcd=mcd)
    else:
        if visited[u1] == 0:
            visited[u1] = 1
            if pre_calculate:
                cd[u1] += mcd[u1]
            else:
                cd[u1] += coreDegree(G, core_num, u1, k)
        if dismissed[u1] == 0 and cd[u1] < k:
            propagateDismissal(G, core_num, cd, dismissed, visited, k, u1, changed_core_nodes, pre_calculate=pre_calculate, mcd=mcd)

        if visited[u2] == 0:
            visited[u2] = 1
            if pre_calculate:
                cd[u2] += mcd[u2]
            else:
                cd[u2] += coreDegree(G, core_num, u2, k)
        if dismissed[u2] == 0 and cd[u2] < k:
            propagateDismissal(G, core_num, cd, dismissed, visited, k, u2, changed_core_nodes, pre_calculate=pre_calculate, mcd=mcd)


    if change_k == 0: # No needed to change the core number. Increment then node core num who has decreased their value. Just return the core changed node set.
        for node in changed_core_nodes:
            core_num[node] += 1
    return list(changed_core_nodes)


if __name__ == '__main__':
    start_time = datetime.now()
    root_dir = 'data/'
    # dir_path = os.path.dirname(os.path.realpath(__file__))

    for path, subdirs, files in os.walk(root_dir):
        for file_name in files:
            file_abs_path = os.path.join(path, file_name)
            graph_name = (file_name.split('.'))[0]
            # continue

            if graph_name == 'temp' or graph_name == 'email-Enron' or graph_name == 'loc-brightkite_edges':
                continue
            if graph_name == 'bio-yeast-protein-inter' or graph_name == 'facebook_combined' or graph_name == 'gplus' or graph_name == 'loc-brightkite_edges':
                continue

            if  not (graph_name == 'as19971108' or graph_name == 'soc-wiki-Vote' or graph_name == 'tech-routers-rf'):
                continue
            print("Graph file name  ", file_abs_path, graph_name)

            graph = global_main.readGraph(file_abs_path)
            actual_node_number_graph = graph.copy()
            graph_matrix = nx.to_numpy_matrix(graph)
            graph = nx.from_numpy_matrix(graph_matrix)
            core_num = nx.core_number(graph)
            kcore = global_main.generateCoreSubgraph(graph, core_num)
            kshell = global_main.generateShellSubgraph(graph, core_num)

            knumber = [core_num[u] for u in core_num]
            kmin = min(knumber)
            kmax = max(knumber)

            cs = global_main.getCoreStrength(graph, core_num)

            adaptiveRemoveEdgeExperiment(graph, actual_node_number_graph, graph_name)

    end_time = datetime.now()
    print("Time taken to run cr vs cis --- %s  ---" % (end_time - start_time)) 