import sys, time, os, math
from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import ALS, Rating
from operator import itemgetter

def model_based_CF():
    start = time.time()

    conf = SparkConf().setAppName("MovieLensModelBasedCF").setMaster("local[8]")
        #set("spark.executor.memory", '4g').set("spark.driver.memory",'4g')

    sc = SparkContext(conf=conf)
    input_file_name = "ratings.csv"
    input_path = sys.argv[1]
    if "20" in input_path:
        test_file_name = "testing_20m.csv"
        output_file_name = "Neeti_Pandey_ModelBasedCF_Big.txt"
        rank = 5
        num_terations = 10
        lambda_ = 0.1
    else:

        test_file_name = "testing_small.csv"
        output_file_name = "Neeti_Pandey_ModelBasedCF_Small.txt"
        rank = 5
        num_terations = 10
        lambda_ = 0.1

    data = sc.textFile(os.path.join(input_path, input_file_name))
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

    print "COUNT = ", training_dataset.count()
    print "test count = ", test_dataset.count()

    #Read entire dataset
    all_data = data.map(lambda row: row.split(',')).map(lambda cols: ((int(cols[0]), int(cols[1])), float(cols[2])))

    # Filter out just the training dataset
    training_dataset_bc = sc.broadcast(set(training_dataset.collect()))
    training_data = all_data.filter(lambda (k, v): k in training_dataset_bc.value)
    test_data_with_pred = all_data.subtract(training_data)
    ratings = training_data.map(lambda x: Rating(x[0][0], x[0][1], x[1]))

    #rank = 5
    #num_terations = 8
    #lambda_ = 0.1
    model = ALS.train(ratings, rank, num_terations, lambda_)

    predictions = model.predictAll(test_dataset).map(lambda x: ((x[0], x[1]), x[2]))

    ratings_predictions = test_data_with_pred.map(lambda r: ((r[0][0], r[0][1]), r[1])).join(predictions)

    output_file = open(output_file_name, 'w')
    results = ratings_predictions.collect()
    results.sort(key=itemgetter(0, 1))
    print "\n\n\n %%%% COUNT $$$$$ = ", len(results)
    for result in results:
        output_file.write("{}, {}, {}\n".format(result[0][0], result[0][1], result[1][1]))
    output_file.close()

    RMSE = math.sqrt(ratings_predictions.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean())


    abs_diff = ratings_predictions.map(lambda r: abs(r[1][0] - r[1][1]))


    l1 = abs_diff.filter(lambda x: x >= 0 and x < 1).count()
    l2 = abs_diff.filter(lambda x: x >= 1 and x < 2).count()
    l3 = abs_diff.filter(lambda x: x >= 2 and x < 3).count()
    l4 = abs_diff.filter(lambda x: x >= 3 and x < 4).count()
    l5 = abs_diff.filter(lambda x: x >= 4).count()

    print ">=0 and <1: ", l1
    print ">=1 and <2: ", l2
    print ">=2 and <3: ", l3
    print ">=3 and <4: ", l4
    print ">=4: ", l5

    print("l1 = {}, l2 = {}, l3 = {}, l4 = {}, l5 = {}".format(l1, l2, l3, l4, l5))
    print("Mean Squared Error = " + str(RMSE))
    print("sum = {}".format(l1+l2+l3+l4+l5))


    #for a in test_data_rdd.collect():
       #print(a)

    print("\n\n\nExecution time = {}".format(time.time() - start))


if __name__ == "__main__":
    model_based_CF()