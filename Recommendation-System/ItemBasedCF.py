
import sys, time, os, math, random
from pyspark import SparkContext, SparkConf
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
                    if j_sim >= 0.5: #thresold
                        j_similarity[frozenset(list(pair))] = j_sim

    return j_similarity




def jaccard_similarity(sc):
    NUM_ROWS = 2
    NUM_HASHES = 40

    input_file = sys.argv[1]  # ratings.csv input file path
    input_file_name = "ratings.csv"
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


    #similar_movies = bands_similar_movies.map(lambda candidates: get_J_similarity(movie_usr_dict, candidates)).flatMap(lambda results: [(key[0], key[1], value) for key, value in results.items()]).collect()
    similar_movies = bands_similar_movies.map(lambda candidates: get_J_similarity(movie_usr_dict, candidates)).flatMap(lambda results: [(sorted(list(key))[0], sorted(list(key))[1], value) for key, value in results.items()]).distinct().collect()
    #for a in similar_movies:
    #            print a


    return similar_movies





def get_pearson_correlation(movie1, movie2):
    users = {}
    if movie1 in movies_users_ratings:
        for user in movies_users_ratings[movie1]:
            if movie2 in movies_users_ratings and user in movies_users_ratings[movie2]:
                users[user] = 1

    num_co_rated = len(users)
    if num_co_rated == 0:
        return 0

    r_u = float(sum([movies_users_ratings[movie1][user] for user in users])) / float(num_co_rated)
    r_v = float(sum([movies_users_ratings[movie2][user] for user in users])) / float(num_co_rated)

    numerator = sum([(movies_users_ratings[movie1][user] - r_u) * (movies_users_ratings[movie2][user] - r_v) for user in users])
    denominator_1 = math.sqrt(sum([(movies_users_ratings[movie1][user] - r_u) ** 2 for user in users]))
    denominator_2 = math.sqrt(sum([(movies_users_ratings[movie2][user] - r_v) ** 2 for user in users]))

    denominator = denominator_1 * denominator_2

    if denominator == 0:
        return 0

    return float(numerator) / float(denominator)

def predict_with_LSH(user, movie):
    user_avg_rating = float(sum(user_movies_ratings[user].values())) / float(len(user_movies_ratings[user]))

    weighted_sum = 0.0
    weights = 0.0
    for movie2 in movies_users_ratings:
        if movie != movie2:
            pair = frozenset(sorted([movie, movie2]))
            if pair not in J_sim:
                continue
            w_i_n = J_sim[pair]
            if movie2 not in movies_users_ratings or user not in movies_users_ratings[movie2]:
                continue
            weighted_sum += movies_users_ratings[movie2][user] * w_i_n
            weights += abs(w_i_n)

    if weights == 0.0:
        return user, movie, user_avg_rating

    rating =  float(weighted_sum) / float(weights)


    #print "LSH user, movie, rating_pred", user, movie, rating
    return user, movie, rating



def predict(user, movie, correlation):
    #print "\n\n\nneighbours[movie]", len(neighbours[movie])
    '''a = neighbours[movie]
    k_similar_movies = sorted(a, key=a.get, reverse=True)
    k_similar_movies = k_similar_movies[:k_neighbours]
    movies_ratedby_users = []
    for neighbour in k_similar_movies:
        if user in movies_users_ratings[neighbour]:
            movies_ratedby_users.append(neighbour)


    if not movies_ratedby_users:
        return user, movie, user_avg_rating

    '''
    user_avg_rating = float(sum(user_movies_ratings[user].values())) / float(len(user_movies_ratings[user]))

    weighted_sum = 0.0
    weights = 0.0
    for movie2 in user_movies_ratings:
        if movie != movie2:
            pair = frozenset(sorted([movie, movie2]))

            if pair not in P_sim:
                continue
            w_i_n = P_sim[pair]

            if movie2 not in movies_users_ratings and user not in movies_users_ratings[movie2][user]:
                continue
            weighted_sum += movies_users_ratings[movie2][user] * w_i_n
            weights += abs(w_i_n)

    if weights == 0.0:
        return user, movie, user_avg_rating

    return user, movie, float(weighted_sum) / float(weights)

def predict_item_based_CF(user, movie):

    '''if movie not in neighbours:
        neighbs = {}
        for movie2 in movies_users_ratings:
            if movie != movie2:
                movie_pair = frozenset(sorted([movie, movie2]))
                if movie_pair not in P_sim:
                    P_sim[movie_pair] = get_pearson_correlation(movie, movie2)
                neighbs[movie2] = P_sim[movie_pair]

        neighbours[movie] = neighbs
        '''



    user, movie, rating = predict(user, movie, correlation="Pearson")
    print "Peasron - user, movie, rating_pred", user, movie, rating
    return user, movie, rating


def item_based_CF():
    start = time.time()


    conf = SparkConf().setAppName("MovieLensItemBasedCF").setMaster("local[8]").\
        set("spark.executor.memory", '10g').set("spark.driver.memory",'10g')

    sc = SparkContext.getOrCreate(conf=conf)
    input_file_name = "ratings.csv"
    test_file_name = "testing_small.csv"
    output_file_name = "Neeti_Pandey_ItemBasedCF.txt"

    data = sc.textFile(os.path.join(sys.argv[1], input_file_name))
    header = data.first()
    data = data.filter(lambda row: row != header)
    dataset = data.map(lambda row: row.split(',')).map(lambda cols: ((int(cols[0]), int(cols[1])), float(cols[2])))

    test_data = sc.textFile(os.path.join(sys.argv[2], test_file_name))
    header2 = test_data.first()
    test_data = test_data.filter(lambda row: row != header2)
    test_dataset = test_data.map(lambda row: row.split(',')).map(
        lambda cols: ((int(cols[0]), int(cols[1])), 1))

    training_dataset = dataset.subtractByKey(test_dataset)



    ground_truth = dataset.subtractByKey(training_dataset)


    data_movie_by_user = training_dataset.map(
        lambda x: (int(x[0][1]), (int(x[0][0]), float(x[1])))).groupByKey().collect()
    data_user_by_movie = training_dataset.map(
        lambda x: (int(x[0][0]), (int(x[0][1]), float(x[1])))).groupByKey().collect()


    for value in data_movie_by_user:
        sub_dict = {}
        for v in value[1]:
            sub_dict[v[0]] = v[1]
        movies_users_ratings[value[0]] = sub_dict


    for value in data_user_by_movie:
        sub_dict = {}
        for v in value[1]:
            sub_dict[v[0]] = v[1]
        user_movies_ratings[value[0]] = sub_dict



    J_similar_movies = jaccard_similarity(sc)
    for line in J_similar_movies:
        J_sim[frozenset(sorted([line[0], line[1]]))] = line[2]
    test_dataset_re = test_dataset.repartition(4)
    '''prediction_LSH = test_dataset_re.map(lambda x: predict_with_LSH(x[0][0], x[0][1])).map(lambda x: ((x[0], x[1]), x[2]))
    ratings_predictions = ground_truth.map(lambda r: ((int(r[0][0]), int(r[0][1])), float(r[1]))).join(prediction_LSH)
    output_file = open(output_file_name, 'w')
    result = ratings_predictions.collect()
    result.sort(key=itemgetter(0, 1))
    print "\n\n\n %%%% COUNT $$$$$ = ", len(result)
    for res in result:
        output_file.write("{},{},{}\n".format(res[0][0], res[0][1], res[1][1]))
    output_file.close()

    RMSE_LSH = math.sqrt(ratings_predictions.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean())

    abs_diff_LSH = ratings_predictions.map(lambda r: abs(r[1][0] - r[1][1]))

    l1 = abs_diff_LSH.filter(lambda x: x >= 0 and x < 1).count()
    l2 = abs_diff_LSH.filter(lambda x: x >= 1 and x < 2).count()
    l3 = abs_diff_LSH.filter(lambda x: x >= 2 and x < 3).count()
    l4 = abs_diff_LSH.filter(lambda x: x >= 3 and x < 4).count()
    l5 = abs_diff_LSH.filter(lambda x: x >= 4).count()
    print "\n\n\n\n\n\n\n COUNT of ratings = ", ratings_predictions.count()
    print "\nItem Based using LSH"
    print ">=0 and <1: ", l1
    print ">=1 and <2: ", l2
    print ">=2 and <3: ", l3
    print ">=3 and <4: ", l4
    print ">=4: ", l5
    print("Mean Squared Error (Using LSH)= " + str(RMSE_LSH))
    print("Total predictions = {}".format(l1 + l2 + l3 + l4 + l5))

    '''



    prediction_item_based_CF = test_dataset_re.map(lambda x: predict_item_based_CF(x[0][0], x[0][1])).map(lambda x: ((x[0], x[1]), x[2]))
    print "\n\n\n prediction_item_based_CF", prediction_item_based_CF.count()  # remove

    # prediction_item_based_CF = prediction_item_based_CF.map(lambda x: ((x[0], x[1]), x[2])).subtractByKey(temp1)
    ratings_pearson = ground_truth.map(lambda r: ((r[0][0], r[0][1]), r[1])).join(prediction_item_based_CF)
    RMSE_pearson = math.sqrt(ratings_pearson.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean())

    abs_diff_pearson = ratings_pearson.map(lambda r: abs(r[1][0] - r[1][1]))
    l1 = abs_diff_pearson.filter(lambda x: x >= 0 and x < 1).count()
    l2 = abs_diff_pearson.filter(lambda x: x >= 1 and x < 2).count()
    l3 = abs_diff_pearson.filter(lambda x: x >= 2 and x < 3).count()
    l4 = abs_diff_pearson.filter(lambda x: x >= 3 and x < 4).count()
    l5 = abs_diff_pearson.filter(lambda x: x >= 4).count()

    print "Item Based using Pearson Correlation"
    print ">=0 and <1: ", l1
    print ">=1 and <2: ", l2
    print ">=2 and <3: ", l3
    print ">=3 and <4: ", l4
    print ">=4: ", l5
    print("Mean Squared Error Pearson= " + str(RMSE_pearson))
    print("Total predictions = {}".format(l1 + l2 + l3 + l4 + l5))




    print("\n\n\nExecution time = {}".format(time.time() - start))
    sc.stop()


if __name__ == "__main__":
    user_movies_ratings = {}
    movies_users_ratings = {}
    J_sim = {}
    P_sim = {}
    neighbours = {}
    k_neighbours = 10

    item_based_CF()
