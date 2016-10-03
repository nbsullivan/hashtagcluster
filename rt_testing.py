
import glob
import json
from pprint import pprint as pp
from tweetutils import clean_tweet
from tweetutils import vectorize_tweets
from tweetutils import clusterinfo
from tweetutils import counts_to_file
from tweetutils import silhouette_analysis
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import scipy
from collections import Counter
import pandas as pd
from pprint import pprint


def tweet_counter(tweet_text):
	

	return tweet_count

if __name__ == '__main__':
	# testing play ground, is set up to work on the #howtoconfuseamillennial dataset.
	# make sure this reflects what is in etl.py's on_status method.

	# setting up list for tweets to go into this acts like tweetlistmaster in etl
	tweetlistmaster = []


	# this will grab the file name where data is stored
	for k in range(0,24):

		base = 'data/millennial/HowtoConfuseaMillennial_batch'

		fil = base + '{0}.txt'.format(k)

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
			tweet_text = [tweet["text"] for tweet in tweetlistmaster]
			tweet_freq = [tweet_text.count(i) for i in tweet_text]
			tweet_count = dict(zip(tweet_text, tweet_freq))
			
			pprint(tweet_count)
			

			# # vectorize
			# vectorized_tweets, names = vectorize_tweets(tweetlistmaster)

			# # do silhouette analysis on master list
			# n, tweet_pred = silhouette_analysis(vectorized_tweets)

			# # create cluter information on master list
			# cluster_json = clusterinfo(n = n, vectorized_tweets = vectorized_tweets, names = names, tweetlistmaster = tweetlistmaster, tweet_pred = tweet_pred)

			# # write to file
			# counts_to_file(cluster_json, base, batchnumber = k)

			# f = open(base + '{0}.json'.format(k), 'w')
			# json.dump(cluster_json, f)
			# f.close()















