import csv
import os
import sys
import networkx as nx
import numpy as np
import random
import numpy
import pathlib
from datetime import datetime
import glob
import math
import random
from networkx.classes.function import neighbors


def readGraph(fname):
    if fname.endswith('mtx'):
        edges = []
        with open(fname, 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            cnt = 0
            for row in reader:
                cnt += 1
                if cnt <= 2:
                    continue
                # print(row)
                if len(row) == 3 or len(row) == 2:
                    edges.append([row[0], row[1]])
        graph = nx.Graph()
        graph.add_edges_from(edges)

    else:
        graph = nx.read_edgelist(fname)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    #print(nx.info(graph))
    return graph

def getCoreStrength(graph, cnumber, nodes=None):
    data = {}
    cd1 = {}

    if nodes is None:
        nodes = graph.nodes()

    for u in nodes:
        neighbors = graph.neighbors(u)
        cd = len([v for v in neighbors if cnumber[u] <= cnumber[v]])
        data[u] = cd - cnumber[u] + 1
        cd1[u] = cd

    return data

def generateCoreSubgraph(graph, cnumber):
    cn = set(cnumber.values())
    core_subgraph = {}

    for c in cn:
        core_subgraph[c] = nx.k_core(graph, k=c, core_number=cnumber)
        # print core number and number of nodes in that core
        # print("core_info     ", c, core_subgraph[c].number_of_nodes())

    return core_subgraph

def generateShellSubgraph(graph, cnumber):
    cn = set(cnumber.values())
    shell_subgraph = {}

    for c in cn:
        shell_subgraph[c] = nx.k_shell(graph, k=c, core_number=cnumber)
    return shell_subgraph

def performExperiment(fname, dest_name):
    graph_start_time = datetime.now()

    graph = readGraph(fname)
    graph_matrix = nx.to_numpy_matrix(graph)
    graph = nx.from_numpy_matrix(graph_matrix)
    print(nx.info(graph))
    return

if __name__ == '__main__':    
    start_time = datetime.now()
    write_data_list = []

    root_dir = 'data/'

    for path, subdirs, files in os.walk(root_dir):
        for file_name in files:
            file_abs_path = os.path.join(path, file_name)
            # graph_name = (file_name.split('.'))[0]
            graph_name = file_name.split('/')
            graph_name = graph_name[len(graph_name) - 1].split('.')[0]

            if graph_name == 'temp' or graph_name == 'email-Enron' or graph_name == 'loc-brightkite_edges':
                continue
            if graph_name == 'bio-yeast-protein-inter' or graph_name == 'facebook_combined' or graph_name == 'gplus' or graph_name == 'loc-brightkite_edges':
                continue
            f_name = graph_name

            if not (f_name == 'inf-power'):
               continue
            print("Graph file name  ", file_abs_path, f_name)
            mx_node = performExperiment(file_abs_path, f_name)
            # max_node = max(mx_node, max_node)

    end_time = datetime.now()