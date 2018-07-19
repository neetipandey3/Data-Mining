import sys, time, os, copy
from pyspark import SparkContext, SparkConf
from itertools import combinations
import networkx as nx
import numpy as np
from operator import itemgetter
#import matplotlib.pyplot as plt


def construct_graph(dataset):
    users_movies = {}

    for user_movies in dataset:
        users_movies[user_movies[0]] = set(user_movies[1])

    user_pairs = combinations(users_movies.keys(), 2)

    G_edges = []
    for pair in user_pairs:
        if len(users_movies[pair[0]].intersection(users_movies[pair[1]])) >= edge_cutoff:
            G_edges.append(tuple(sorted(pair)))

    #print "Edges #",len(G_edges)
    #print G_edges

    G.add_edges_from(G_edges)


def traverse_bfs(root_node):
    queue, visited, distance, predecessor = list(), dict(), dict(), dict()

    queue.append(root_node)
    distance[root_node] = 0
    visited[root_node] = True
    for node in G[root_node]:
        visited[node] = True
        predecessor[node] = [root_node, ]
        queue.append(node)
        distance[node] = 1

    i = 1
    while i < len(queue):
        node = queue[i]
        dist = distance[node]
        visited[node] = True
        for child in G[node]:
            if child not in visited:
                if child not in distance:
                    queue.append(child)
                    distance[child] = dist + 1
                    predecessor[child] = [node,]
                else:
                    if distance[child] > dist:
                        predecessor[child].append(node)
                        distance[child] = dist + 1

        i += 1
    return queue, distance, predecessor

def calculate_betweenness(_bfs, distance, predecessor, betweenness):
    credits = dict()
    _bfs = _bfs[1:]
    while _bfs:
        node = _bfs.pop()
        if node not in credits:
            credits[node] = 1
        if len(predecessor[node]) != 0:
            weight = float(1) / len(predecessor[node])
            for parent in predecessor[node]:
                if parent not in credits:
                    credits[parent] = 1
                credit = weight * credits[node]
                credits[parent] += credit
                if parent < node:
                    edge = (parent, node)
                    if edge in betweenness:
                        betweenness[edge] += credit
                    else:
                        betweenness[edge] = credit


def get_betweenness():
    betweenness = {}
    for root in G:
        bfs_path, distance, predecessor = traverse_bfs(root)
        calculate_betweenness(bfs_path, distance, predecessor, betweenness)

    return betweenness



def get_modularity(G_, B):
    #print "calculating modularity"
    K_delta = [[0 for _ in range(max(G)+1)] for _ in range(max(G)+1)]
    m = nx.number_of_edges(G) # original graph
    _Q = 0

    partition_S = nx.connected_components(G_)
    for s in partition_S:
        for i in s:
            for j in s:
                K_delta[i][j] = 1
                #_Q += B[i, j]
    K_delta = np.array(K_delta)
    Q = np.sum(np.multiply(K_delta, B)) / float(2 * m)
    #Q = float(_Q) / float(2 * m)
    #print "calculated modularity", Q
    return Q




def find_community(betweenness):
    #print "finding community"

    node_degrees = {}
    P = np.zeros((max(G)+1, max(G)+1))
    G_ = copy.deepcopy(G)
    for node in G:
        node_degrees[node] = G.degree(node)


    A = nx.to_numpy_matrix(G)
    print A
    A = np.c_[np.zeros(max(G)), A]
    A = np.r_[np.zeros((1, max(G)+1)), A]
    m = nx.number_of_edges(G)
    #A = np.zeros((max(G)+1, max(G)+1))
    for i in G:
        for j in G:
            P[i, j] = float(node_degrees[i] * node_degrees[j]) / float(2 * m)
    B = A - P
    #B = B.round(decimals=2)
    print B

    max_Q = np.sum(B) / float(2*m)
    #print "initial modularity", max_Q
    communities = list(nx.connected_components(G))
    sorted_betweenness = sorted(betweenness.items(), key=itemgetter(1), reverse=True)
    for edge in sorted_betweenness:
        node1, node2 = edge[0][0], edge[0][1]
        G_.remove_edge(node1, node2)
        #del betweenness[edge]

        #if nx.number_connected_components(G_) > num_communities:
        if not nx.has_path(G_, node1, node2):
            Q = get_modularity(G_, B)
            if Q > max_Q:
                max_Q = Q
                communities = list(nx.connected_components(G_))
                num_comm = len(communities)
                #print "num communities", num_comm
                #print "MAX Q", max_Q


    return max_Q, communities



if __name__ == "__main__":
    start = time.time()

    conf = SparkConf().setAppName("MovieLensCommunity").setMaster("local[8]")\
        .set("spark.executor.memory", '4g').set("spark.driver.memory",'4g')

    sc = SparkContext(conf=conf)
    input_file_name = "ratings.csv"
    output_file_name = "Neeti_Pandey_Community.txt"

    edge_cutoff = 9

    dataset = sc.textFile(os.path.join(sys.argv[1], input_file_name))
    header = dataset.first()
    dataset = dataset.filter(lambda row: row != header)
    dataset = dataset.map(lambda row: row.split(',')).map(lambda cols: (int(cols[0]), int(cols[1]))).distinct().groupByKey().collect()

    G = nx.Graph()
    construct_graph(dataset)
    betweenness = get_betweenness()
    #betweenness = nx.edge_betweenness_centrality(G, normalized = False)
    betweenness.update((k, v / 2.0) for k, v in betweenness.items())
    betweenness.update((k, int(v * 1000) / 1000.0) for k, v in betweenness.items())


    modularity, communities = find_community(betweenness)


    output_file = open(output_file_name, 'w')
    print "\n\n\n %%%% COUNT $$$$$ = ", len(communities)
    for community in communities:
        output_file.write("{}\n".format(sorted(list(community))))
    output_file.close()



    print("\n\n\nExecution time = {}".format(time.time() - start))

    sc.stop()

