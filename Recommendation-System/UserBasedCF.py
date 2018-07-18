import sys, time, os
from pyspark import SparkContext, SparkConf
import math
from operator import itemgetter





def get_pearson_correlation(user1, user2):
    movies = {}
    for movie in user_movies_ratings[user1]:
        if movie in user_movies_ratings[user2]:
            movies[movie] = 1

    num_co_rated = len(movies)
    if num_co_rated == 0:
        return 0

    r_u = float(sum([user_movies_ratings[user1][movie] for movie in movies]))/float(num_co_rated)
    r_v = float(sum([user_movies_ratings[user2][movie] for movie in movies]))/float(num_co_rated)

    numerator = sum([(user_movies_ratings[user1][movie] - r_u) * (user_movies_ratings[user2][movie] - r_v) for movie in movies])
    denominator_1 = math.sqrt(sum([(user_movies_ratings[user1][movie] - r_u) ** 2 for movie in movies]))
    denominator_2 = math.sqrt(sum([(user_movies_ratings[user2][movie] - r_v) ** 2 for movie in movies]))

    denominator = denominator_1*denominator_2

    if denominator == 0:
        return 0

    cor = float(numerator)/float(denominator)
    return cor



def get_knn_users(active_user, k):
    neighbours = {}

    k_neighbours = {}
    for user in user_movies_ratings:
        if user != active_user:
            pair = frozenset(sorted([user, active_user]))
            if pair not in pearson_sim:
                pearson_sim[pair] = get_pearson_correlation(active_user, user)
            else:
                neighbours[user] = pearson_sim[pair]


    sorted_n_users = sorted(neighbours, key=neighbours.get, reverse=True)[:k]
    for user in sorted_n_users:
            k_neighbours[user] = neighbours[user]
    return k_neighbours

def case_amplification(weight):
    if weight >= 0.0:
        return weight ** 2.5
    else:
        return -((-weight) ** 2.5)

def reduce_weights():
    if num_co_rated < num_co_rated_thresold:
        return 1/float(num_co_rated_thresold)
    else:
        return 1

def inverse_user_frequency(movie):
    n_j = 0.0
    if not movie in inverse_user_freq:
        for user in user_movies_ratings:
            if user_movies_ratings[user]:
                if movie in user_movies_ratings[user].keys():
                    n_j += 1
        n = len(user_movies_ratings)
        if n_j == 0:
            inverse_user_freq[movie] =  1
        else:
            inverse_user_freq[movie] =  math.log10(n/n_j)
        print "n , nj, inverse_user_freq[movie]", n, n_j, inverse_user_freq[movie]

    return inverse_user_freq[movie]

def get_cosine_similarity(x, y):

    numerator = sum(a * b for a, b in zip(x, y))
    denominator = math.sqrt(sum([a * a for a in x])) * math.sqrt(sum([a * a for a in y]))
    return float(numerator) / float(denominator)

def vectorize():
    all_genres = set([genre for genre_sublist in movies_genres.values() for genre in genre_sublist])


    genres = list(all_genres)

    for movie in movies_genres:
        vector = []
        for genre in genres:
            if genre in movies_genres[movie]:
                vector.append(1)
            else:
                vector.append(0)
        movie_profiles[movie] = vector
    return movie_profiles


def content_based_rating(active_user, movie):

    movies_user_rated = user_movies_ratings[active_user].keys()
    weighted_rating = 0.0
    weights = 0.0
    for movie2 in movies_user_rated:
        movie_pair = frozenset(sorted([movie, movie2]))
        if movie_pair not in cosine_similarity:
            cosine_similarity[movie_pair] = get_cosine_similarity(movie_profiles[movie], movie_profiles[movie2])

        if cosine_similarity[movie_pair] > 0:
            weighted_rating += user_movies_ratings[active_user][movie2] * cosine_similarity[movie_pair]
            weights += cosine_similarity[movie_pair]

    if weights > 0.0:
        prediction = float(weighted_rating)/float(weights)
    else:
        prediction = float(sum(movies_user_rated)) / float(len(movies_user_rated))

    #print "user, movie, prediction", active_user, movie, prediction
    return prediction






def predict(active_user, movie):
    k_neighbours = get_knn_users(active_user, k)
    if not k_neighbours:
        return active_user, movie, content_based_rating(active_user, movie)


    r_a_avg = float(sum(user_movies_ratings[active_user].values()))/ float(len(user_movies_ratings[active_user]))


    users_rated_movie = []
    for neighbour in k_neighbours:
        if movie in user_movies_ratings[neighbour]:
            users_rated_movie.append(neighbour)


    if not users_rated_movie:
        #return active_user, movie, content_based_rating(active_user, movie)
        return active_user, movie, r_a_avg

    #if len(users_rated_movie) < 5:
        #return active_user, movie, content_based_rating(active_user, movie)
    #inverse_user_frequency(movie)
    sum_weighted_diff = 0.0
    sum_w_a_u = 0.0
    w_a = []
    weight_diff = []
    for user in users_rated_movie:
        w_a_u = pearson_sim[frozenset(sorted([user, active_user]))]
        # Use case_amplification
        if is_case_amplification:
            w_a_u = case_amplification(w_a_u)
        if is_reduce_weights:
           w_a_u = w_a_u*reduce_weights()
        r_u = float(sum(user_movies_ratings[user].values()) - user_movies_ratings[user][movie]) / float(len(user_movies_ratings[user])-1)
        sum_weighted_diff += (user_movies_ratings[user][movie] - r_u)* w_a_u
        weight_diff.append(sum_weighted_diff)
        #print "inverse_user_frequency(movie)" , inverse_user_frequency(movie)
        sum_w_a_u += abs(w_a_u)
        w_a.append(w_a_u)
    if active_user == 652 and movie == 4141:
        print"\n\n\n####### length of 652 who voted for 539 weights sum_w_a_u", sum_w_a_u
    if sum_w_a_u == 0.0:
        return active_user, movie, r_a_avg

    weight = (float(sum_weighted_diff) / float(sum_w_a_u))

    rating = r_a_avg + weight
    if rating > 5.0:
        rating = 5.0
    return active_user, movie, rating




def user_based_CF():
    start = time.time()

    conf = SparkConf().setAppName("MovieLensUserBasedCF").setMaster("local[8]")

    sc = SparkContext(conf=conf)
    input_file_name = "ratings.csv"
    test_file_name = "testing_small.csv"
    output_file_name = "Neeti_Pandey_UserBasedCF.txt"

    data = sc.textFile(os.path.join(sys.argv[1], input_file_name))
    header = data.first()
    data = data.filter(lambda row: row != header)
    dataset = data.map(lambda row: row.split(',')).map(lambda cols: (int(cols[0]), int(cols[1])))

    #Read the test data to predict on
    test_data = sc.textFile(os.path.join(sys.argv[2], test_file_name))
    header2 = test_data.first()
    test_data = test_data.filter(lambda row: row != header2)
    test_dataset = test_data.map(lambda row: row.split(',')).map(
        lambda cols: (int(cols[0]), int(cols[1])))

    training_dataset = dataset.subtract(test_dataset)


    #Read entire dataset
    all_data = data.map(lambda row: row.split(',')).map(lambda cols: ((int(cols[0]), int(cols[1])), float(cols[2]))).distinct()

    # Filter out just the training dataset
    training_dataset_bc = sc.broadcast(set(training_dataset.collect()))
    training_data = all_data.filter(lambda (k, v): k in training_dataset_bc.value)
    test_data_with_pred = all_data.subtract(training_data)

    print "\n\n\n test dataset count = ", test_data_with_pred.count()
    print "training data count = ", training_data.count()

    data_user_by_movie = training_data.map(
        lambda x: (int(x[0][0]), (int(x[0][1]), float(x[1])))).groupByKey().collect()



    # Create dict: {user_id: {movie_id1: rating, movie_id2: rating}}
    #user_movies_ratings = {}
    for value in data_user_by_movie:
        sub_dict = {}
        for v in value[1]:
            sub_dict[v[0]] = v[1]
        user_movies_ratings[value[0]] = sub_dict

    #Extract movie - genre
    movies_file_name = "movies.csv"
    movies_data = sc.textFile(os.path.join(sys.argv[1], movies_file_name), use_unicode=False)
    header = movies_data.first()
    movies_dataset = movies_data.filter(lambda row: row != header)


    movies_dataset = movies_dataset.map(lambda row: row.split(',')).map(lambda line: (int(line[0]), line[-1])).collect()

    for line in movies_dataset:
        movies_genres[line[0]] = line[1].split("|")

    vectorize()


    predictions = test_dataset.map(lambda x: predict(x[0], x[1])).map(lambda x: ((x[0], x[1]), x[2]))
    #for a in predictions.collect():
       #print(a)



    ratings_predictions = test_data_with_pred.map(lambda r: ((r[0][0], r[0][1]), r[1])).join(predictions)
    RMSE = math.sqrt(ratings_predictions.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean())

    abs_diff = ratings_predictions.map(lambda r: abs(r[1][0] - r[1][1]))

    l1 = abs_diff.filter(lambda x: x >= 0 and x < 1).count()
    l2 = abs_diff.filter(lambda x: x >= 1 and x < 2).count()
    l3 = abs_diff.filter(lambda x: x >= 2 and x < 3).count()
    l4 = abs_diff.filter(lambda x: x >= 3 and x < 4).count()
    l5 = abs_diff.filter(lambda x: x >= 4).count()
    #l6 = ratings_predictions.filter(lambda x: x[1][1] > 5).count()

    print ">=0 and <1: ", l1
    print ">=1 and <2: ", l2
    print ">=2 and <3: ", l3
    print ">=3 and <4: ", l4
    print ">=4: ", l5
    #print "Prediction >=5: ", l6

    #print("l1 = {}, l2 = {}, l3 = {}, l4 = {}, l5 = {}".format(l1, l2, l3, l4, l5))
    print("Mean Squared Error = " + str(RMSE))
    print("Total predictions = {}".format(l1 + l2 + l3 + l4 + l5))

    output_file = open(output_file_name, 'w')
    result = ratings_predictions.collect()
    result.sort(key=itemgetter(0, 1))
    print "\n\n\n %%%% COUNT $$$$$ = ", len(result)
    for res in result:
        output_file.write("{}, {}, {}\n".format(res[0][0], res[0][1], res[1][1]))
    output_file.close()


    print("\n\n\nExecution time = {}".format(time.time() - start))

    sc.stop()


if __name__ == "__main__":
    num_co_rated = 0
    num_co_rated_thresold = 50
    user_movies_ratings = {}
    movies_genres = {}
    # dict of all p similarities
    pearson_sim = {}
    cosine_similarity = {}
    inverse_user_freq = {}
    movie_profiles = {}

    #No of nearest neighbours
    k = 250
    is_reduce_weights = False
    is_case_amplification = False

    user_based_CF()