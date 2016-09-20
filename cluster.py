import json
import glob
from tweetutils import vectorize_tweets
from tweetutils import silhouette_analysis
from tweetutils import cluster_to_port
from tweetutils import clusterinfo
import shutil
import time

if __name__ == '__main__':
	
	# ongoing tweet collection list
	mastertweetlist = []
	roundnumber = 1

	while True:
		# check for new files
		if len(glob.glob('data/current_data/*.txt')) > 0:
			for fil in glob.glob('data/current_data/*.txt'):

				time.sleep(.5)
				print "loading ", fil
				f = open(fil, 'r')
				tweet_batch = json.load(f)

				# append to master list
				mastertweetlist = mastertweetlist + tweet_batch

				# movefiles once loaded
				newpath = "data/processed_data/" + fil[18:]

				shutil.move(fil, newpath)


			# vectorize master list
			vectorized_tweets, names = vectorize_tweets(mastertweetlist)

			# do silhouette analysis on master list
			n, tweet_pred = silhouette_analysis(vectorized_tweets)

			# create cluter information on master list
			cluster_json = clusterinfo(n = n, vectorized_tweets = vectorized_tweets, names = names, tweetlistmaster = mastertweetlist, tweet_pred = tweet_pred)

			# send master list to visualization
			cluster_to_port(cluster_json)

			print "finished batch ", roundnumber, " total tweets clustering on ", len(mastertweetlist)
			roundnumber = roundnumber + 1
