import sys, time, os
from pyspark import SparkContext, SparkConf
from itertools import combinations
import networkx as nx
import community
#import igraph as ig



if __name__ == "__main__":
    start = time.time()

    conf = SparkConf().setAppName("MovieLensCommunityDetetction").setMaster("local[8]")\
        .set("spark.executor.memory", '4g').set("spark.driver.memory",'4g')

    sc = SparkContext(conf=conf)
    input_file_name = "ratings.csv"
    output_file_name = "Communities.txt"

    edge_cutoff = 9

    dataset = sc.textFile(os.path.join(sys.argv[1], input_file_name))
    header = dataset.first()
    dataset = dataset.filter(lambda row: row != header)
    dataset = dataset.map(lambda row: row.split(',')).map(lambda cols: (int(cols[0]), int(cols[1]))).distinct().groupByKey().collect()

    #g = ig.Graph()
    G = nx.Graph()
    #construct_graph(dataset)

    users_movies = {}

    for user_movies in dataset:
        users_movies[user_movies[0]] = set(user_movies[1])

    user_pairs = combinations(users_movies.keys(), 2)

    G_edges = []
    nodes = []
    for pair in user_pairs:
        if len(users_movies[pair[0]].intersection(users_movies[pair[1]])) >= edge_cutoff:
            G_edges.append(tuple(sorted(pair)))
            nodes.append(pair[0])
            nodes.append(pair[1])

    #nodes = list(set(nodes))

    #g = ig.Graph(vertex_attrs={"label": nodes}, edges=G_edges, directed=False)
    print "Edges #", len(G_edges)
    G.add_edges_from(G_edges)

    partition = community.best_partition(G)
    values = [partition.get(node) for node in G.nodes()]

    num_communities = list(set(values))

    communities = {}
    for num in num_communities:
        communities[num] = [k for k, v in partition.items() if v == num]



    #print communities
    #print "len: ", len(communities)
    #communities = g.community_edge_betweenness(directed = False)

    #clusters = communities.as_clustering()
    #print "len: ", len(communities)
    #print float(len(set(communities.values())))


    #communities = g.community_edge_betweenness(directed = False)

    #communities = g.community_optimal_modularity(weights=None)

    #communities = g.community_infomap(edge_weights=None, vertex_weights=None, trials=10)

    output_file = open(output_file_name, 'w')
    #result.sort(key=itemgetter(0, 1))
    print "\n\n\n %%%% COUNT $$$$$ = ", len(communities)
    for k in communities:
        output_file.write("{}\n".format(sorted(list(communities[k]))))
    output_file.close()


    print("\n\n\nExecution time = {}".format(time.time() - start))

    sc.stop()

