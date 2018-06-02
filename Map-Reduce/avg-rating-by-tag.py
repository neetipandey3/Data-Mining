import sys, csv, os
from pyspark import SparkContext

'''
Movie Average Rating by tag

Input:
    :parameter #1: input file path (ratings.csv)
    :parameter #2: input file path (tags.csv)
    :parameter #3: output file path 

Output: 
    Neeti_Pandey_result_task2.csv; values = (tag, avg_rating)

'''
class TagMovieAvgRating:
    def getTagMovieAvgRating(self, ratings, tags):

        # Prepare ratings from ratings data as (movieId, (ratings, 1.0))
        ratings_rdd = ratings.map(lambda row: row.split(',')).map(lambda cols: (int(cols[1]), (float(cols[2]), 1.0)))
        # Prepare tags and movie id as (movieId, tag)
        tags_rdd = tags.map(lambda row: row.split(',')).map(lambda cols: (int(cols[1]), cols[2]))
        #tags_join_ratings_rdd = tags_rdd.join(ratings_rdd)
        #tags_ratings_rdd =   tags_join_ratings_rdd.map(lambda x: (x[1][0], x[1][1]))
        #print tags_join_ratings_rdd.take(5)
        #print tags_ratings_rdd.take(5)
        #tag_ratings_sumcount_rdd = tags_ratings_rdd.aggregateByKey((0, 0),lambda x, y:(x[0]+y, x[1]+1), lambda rdd1, rdd2: (rdd1[0]+rdd2[0], rdd1[1]+rdd2[1]))
        #tags_ratings_avg_rdd = tag_ratings_sumcount_rdd.map(lambda x: (x[0], x[1][0] / x[1][1])).sortByKey(ascending=False)
        #return tags_ratings_avg_rdd.collect()

        #Reduce to (movieID, (ratings_sum, total_no_of_ratings))
        sum_count_ratings_rdd = ratings_rdd.reduceByKey(lambda mov1, mov2: (mov1[0] + mov2[0], mov1[1] + mov2[1]))
        tag_ratings_join_rdd = sum_count_ratings_rdd.join(tags_rdd).map(lambda row: (row[1][1], row[1][0])).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
        avg_rating_rdd = tag_ratings_join_rdd.mapValues(lambda sumAndCount: sumAndCount[0] / sumAndCount[1]).sortByKey(ascending=False)


        return avg_rating_rdd.collect()

    def writeOutput(self, tag_avgrating_rdd, op_path, op_filename):
        open_file = open(os.path.join(op_path, op_filename), 'w')
        writer = csv.writer(open_file)
        writer.writerow(["tag", "rating_avg"])         #add header to output csv
        for row in tag_avgrating_rdd:
            writer.writerow([row[0], row[1]])

        open_file.close()

def main():
    input_path1 = sys.argv[1]
    input_path2 = sys.argv[2]
    output_path = sys.argv[3]

    sc = SparkContext(appName="MovieRatingMovieLens")
    ratings_filename = "ratings.csv"
    tag_filename = "tags.csv"
    output_filename = "Neeti_Pandey_result_task2.csv"
    ratings_data = sc.textFile(os.path.join(input_path1, ratings_filename), use_unicode=False)
    tag_data = sc.textFile(os.path.join(input_path2, tag_filename), use_unicode=False)

    csv_hdr_rating = ratings_data.first()     # get rid of the header of the input files
    #ratings = ratings_data.filter(lambda x: x != csv_hdr_rating)
    csv_hdr_rating_rdd = sc.parallelize([csv_hdr_rating])
    ratings = ratings_data.subtract(csv_hdr_rating_rdd)

    csv_hdr_tag = tag_data.first()
    #tags = tag_data.filter(lambda x: x != csv_hdr_tag)
    csv_hdr_tag_rdd = sc.parallelize([csv_hdr_tag])
    tags = tag_data.subtract(csv_hdr_tag_rdd)

    task2 = TagMovieAvgRating()
    tag_avgrating_rdd = task2.getTagMovieAvgRating(ratings, tags)
    task2.writeOutput(tag_avgrating_rdd, output_path, output_filename)

if __name__ == "__main__":
    main()






