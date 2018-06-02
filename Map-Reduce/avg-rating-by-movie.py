import sys, csv, os
from pyspark import SparkContext
'''
Movie Average Rating by movieId

Input:
    :parameter #1: input file path (ratings.csv)
    :parameter #2: output file path

Output: 
    Neeti_Pandey_result_task1.csv; values = (movieId, avg_rating)
        
'''

class MovieRating:

    def getMovieAvgRating(self, rdd):
        # Prepare ratings from ratings data as (movieId, (ratings, 1.0))
        ratings_rdd = rdd.map(lambda row: row.split(',')).map(lambda cols: (int(cols[1]), (float(cols[2]), 1.0)))
        #ratings_sumcount_rdd = ratings_rdd.aggregateByKey((0, 0), lambda x, y: (x[0] + y, x[1]+1), lambda rdd1, rdd2: (rdd1[0]+rdd2[0], rdd1[1]+rdd2[1]))
        #ratings_avg_rdd = ratings_sumcount_rdd.map(lambda x: (int(x[0]), x[1][0] / x[1][1])).sortByKey(ascending=True)
        #return ratings_avg_rdd.collect()

        # Reduce to (movieID, (ratings_sum, total_no_of_ratings))
        sum_count_ratings_rdd = ratings_rdd.reduceByKey(lambda mov1, mov2: (mov1[0] + mov2[0], mov1[1] + mov2[1]))
        avg_rating_rdd = sum_count_ratings_rdd.mapValues(lambda sumAndCount: sumAndCount[0] / sumAndCount[1]).sortByKey(ascending=True)

        return avg_rating_rdd.collect()

    def writeOutput(self, movie_avgrating_rdd, op_path, op_filename):
        open_file = open(os.path.join(op_path, op_filename), 'w')
        writer = csv.writer(open_file)
        # add header to output csv
        writer.writerow(["movieId", "rating_avg"])
        for row in movie_avgrating_rdd:
            writer.writerow([row[0], row[1]])
        open_file.close()
        
def main():
    sc = SparkContext(appName="MovieRatingMovieLens")
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    input_filename = "ratings.csv"
    output_filename = "Neeti_Pandey_result_task1.csv"
    rdd_data = sc.textFile(os.path.join(input_path, input_filename))
    # get rid of the header of the input file
    csv_header = rdd_data.first()
    csv_header_rdd = sc.parallelize([csv_header])
    rdd = rdd_data.subtract(csv_header_rdd)

    task1 = MovieRating()
    movie_avgrating_rdd = task1.getMovieAvgRating(rdd)
    task1.writeOutput(movie_avgrating_rdd, output_path, output_filename)

if __name__ == "__main__":
    main()






