
import glob
import json
from pprint import pprint as pp
from etl import clean_tweet
from etl import vectorize_tweets
from etl import clusterinfo
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import scipy
from collections import Counter







if __name__ == '__main__':
	# testing play ground, is set up to work on the #howtoconfuseamillennial dataset.
	# make sure this reflects what is in etl.py's on_status method.

	# setting up list for tweets to go into this acts like tweetlistmaster in etl
	tweetlistmaster = []


	# this will grab the file name where data is stored
	for fil in glob.glob('data/clean/*.txt'):

		# only use millennial data
		if "HowtoConfuseaMillennial" in fil:
			print fil

			# open and load the file
			f = open(fil, 'r')
			tweet_batch = json.load(f)

			# append file data to full list
			tweetlistmaster = tweetlistmaster + tweet_batch
			userlist = [tweet["screen_name"] for tweet in tweetlistmaster]
			tweet_id = [tweet['tweet_id'] for tweet in tweetlistmaster]


			# vectorize
			vectorized_tweets, names = vectorize_tweets(tweetlistmaster)

			# need code for silhouette analysis
			n = 3
			# cluster
			clf = KMeans(n_clusters=n)
			tweet_pred = clf.fit_predict(vectorized_tweets)

			# we want to subset the vectorized tweets based on tweet_pred, also create a list of dictionarys with word counts per cluster
			
			cluster_json = clusterinfo(n = n, vectorized_tweets = vectorized_tweets, names = names, tweetlistmaster = tweetlistmaster, tweet_pred = tweet_pred)
			
	pp(cluster_json)
