
import sys, time, os
from pyspark import SparkContext, SparkConf
import random
import numpy as np
from operator import itemgetter

def generate_hashes(user_movie_dict, n, p):

    def get_coefficients():
        a_hash = []
        for _ in range(n):
            a = random.randint(1, p)
            if a % 2 == 0:
                a += 1
            a_hash.append(a)
        return a_hash

    def hash(x):
        hash_r = [((a * x) + b) % p for a, b in zip(a_hash, b_hash)]
        return hash_r

    a_hash = get_coefficients()
    b_hash = [random.randrange(p) for _ in xrange(n)]

    hashes = {}
    for user_id in user_movie_dict:
        hashes[user_id] = hash(user_id)

    #print "\n\n\n$$$$ HASH GENERATED ", hashes
    return hashes


def get_rows_list(num_bands, num_rows):
    rows_list = []
    for b in range(num_bands):
        for r in range(num_rows):
            rows_list.append(b * num_rows + r)
    return rows_list



def replace_with_min(new_hash, movies_list, signatures):
    #print "\n\n\n movie_list", movies_list
    new_hash = np.array(new_hash)
    for movie in movies_list:
        if movie in signatures:
            curr_hash = np.array(signatures[movie])
            signatures[movie] = np.minimum(curr_hash, new_hash).tolist()
        else:
            signatures[movie] = new_hash.tolist()
    #print("\n\n\n $$$$$\n\n\n  LEngth of signatures {}".format(len(signatures)))
    return signatures

def get_bands(min_hash_movie, NUM_ROWS):
    for i in xrange(0, len(min_hash_movie), NUM_ROWS):
        yield frozenset(min_hash_movie[i:i+NUM_ROWS])

def get_J_similarity(movie_dataset, candidates):
    j_similarity = {}

    for i in candidates:
        for j in candidates:
            if i != j:
                pair = tuple(sorted([j, i]))
                if pair not in j_similarity:
                    set1, set2 = set(movie_dataset[pair[0]]), set(movie_dataset[pair[1]])
                    j_sim = float(len(set1 & set2)) / float(len(set1 | set2))
                    if j_sim >= 0.5:
                        j_similarity[frozenset(list(pair))] = j_sim

    return j_similarity




def jaccard_similarity():
    start = time.time()


    conf = SparkConf().setAppName("LSH").setMaster("local[8]")

    sc = SparkContext.getOrCreate(conf=conf)
    input_file = sys.argv[1]  # ratings.csv input file path
    input_file_name = "ratings.csv"
    output_filename = "Neeti_Pandey_SimilarMovie_Jaccard.txt"
    data = sc.textFile(os.path.join(input_file, input_file_name))
    header = data.first()
    data = data.filter(lambda row: row != header)

    data_user_by_movie = data.map(lambda row: row.split(',')).map(lambda cols: (int(cols[0]), int(cols[1]))).distinct()\
        .groupByKey().map(lambda x: (x[0], sorted(list(x[1]))))
    data_movie_by_user = data.map(lambda row: row.split(',')).map(lambda cols: (int(cols[1]), int(cols[0]))).distinct() \
        .groupByKey().map(lambda x: (x[0], list(x[1])))
    users_movies = data_user_by_movie.collect()
    movies_users = data_movie_by_user.collect()
    max_num_rows = data_user_by_movie.count()
    usr_movie_dict = {}
    for user_movie in users_movies:
        usr_movie_dict[user_movie[0]] = user_movie[1]
    movie_usr_dict = {}
    for movie_user in movies_users:
        movie_usr_dict[movie_user[0]] = movie_user[1]


    hashes = generate_hashes(user_movie_dict=usr_movie_dict, n=NUM_HASHES, p=(max_num_rows+1))

    signatures = {}
    for user, movies in usr_movie_dict.iteritems():
        signatures = replace_with_min(hashes[user], movies, signatures)



    # Emit band_id, list(movie_ids_in_the_same_bucket)
    bands_similar_movies = sc.parallelize(signatures.items()).map(lambda x: [[(b, hash(band)), x[0]] for b, band in enumerate(get_bands(x[1], NUM_ROWS))]).\
        flatMap(lambda x: [(list(each[0])[0], list(each[0])[1], each[1]) for each in x]).groupBy(lambda x: x[:-1]).map(lambda y: y[0] + (tuple(x[-1] for x in y[1]),)). \
        map(lambda x: tuple(sorted(x[2]))).filter(lambda x: len(x) > 1).distinct()


    similar_movies = bands_similar_movies.map(lambda candidates: get_J_similarity(movie_usr_dict, candidates)).flatMap(lambda results: [(sorted(list(key))[0], sorted(list(key))[1], value) for key, value in results.items()]).distinct()

    similar_movies_result = similar_movies.collect()
    output_file = open(output_filename, 'w')
    similar_movies_result.sort(key=itemgetter(0, 1))
    #print "\n\n\n %%%% COUNT $$$$$ = ", len(similar_movies_result)
    for result in similar_movies_result:
        output_file.write("{}, {}, {}\n".format(result[0], result[1], result[2]))
    output_file.close()



    print("\n\n\nExecution time = {}".format(time.time() - start))


    '''
    Precision & Recall Calculation using Grount Truth as reference
    
    '''
    '''truth_values = os.path.join(input_file, "SimilarMovies.GroundTruth.05.csv")
    my_result = similar_movies.map(lambda x: (frozenset(sorted([x[0], x[1]])), 1))
    ground_truth = sc.textFile(truth_values, use_unicode=False).map(lambda row: row.split(',')).map(lambda x: (frozenset(sorted([int(x[0]), int(x[1])])), 1))
    for a in ground_truth.take(20):
        print "ground_truth ", a
    for a in my_result.take(20):
        print "my_result ", a
    t = my_result.subtract(ground_truth)
    trueP = my_result.subtract(t)

    print "\n\n\n\n\$$$$$$$$%%%%%%%%$$$#@@@!@!\n\n\n trueP.count", trueP.count()

    TP = trueP.count()
    falseP = my_result.subtractByKey(trueP)
    FP = falseP.count()
    FN = ground_truth.count() - my_result.count()



    precision = float(TP)/float(TP+FP)
    recall = float(TP)/float(TP+FN)

    print("\n\n TP = {}  FP = {} FN = {}".format(TP, FP, FN))
    print("\n\n precision {}".format(precision))
    print("\n\n recall {}".format(recall))
    '''


    sc.stop()



if __name__ == "__main__":
    NUM_ROWS = 2
    NUM_HASHES = 40
    jaccard_similarity()