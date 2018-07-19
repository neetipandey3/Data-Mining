import sys, time, os, math
from pyspark import SparkContext, SparkConf
from itertools import combinations
import networkx as nx
from operator import itemgetter
from collections import deque


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
        predecessor[node] = [root_node,]
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




if __name__ == "__main__":
    start = time.time()

    conf = SparkConf().setAppName("MovieLensBetweenness").setMaster("local[8]")\
        .set("spark.executor.memory", '4g').set("spark.driver.memory",'4g')

    sc = SparkContext(conf=conf)
    input_file_name = "ratings.csv"
    output_file_name = "Betweenness.txt"

    edge_cutoff = 9

    dataset = sc.textFile(os.path.join(sys.argv[1], input_file_name))
    header = dataset.first()
    dataset = dataset.filter(lambda row: row != header)
    dataset = dataset.map(lambda row: row.split(',')).map(lambda cols: (int(cols[0]), int(cols[1]))).distinct().groupByKey().collect()

    G = nx.Graph()
    construct_graph(dataset)

    #betweenness = dict.fromkeys(G.edges, 0.0)
    betweenness = get_betweenness()

    #betweenness2 = nx.edge_betweenness_centrality(G, k=None, normalized=False, weight=None, seed=None)

    output_file = open(output_file_name, 'w')
    result = betweenness.items()
    result.sort(key=itemgetter(0, 1))
    print "\n\n\n %%%% COUNT $$$$$ = ", len(result)
    for res in result:
        output_file.write("({},{},{})\n".format(res[0][0], res[0][1], res[1]/2.0))
    output_file.close()



    print("\n\n\nExecution time = {}".format(time.time() - start))

    sc.stop()

