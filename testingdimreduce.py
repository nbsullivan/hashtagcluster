
import glob
import json
from pprint import pprint as pp
from tweetutils import clean_tweet
from tweetutils import vectorize_tweets
from tweetutils import clusterinfo
from tweetutils import counts_to_file
from tweetutils import silhouette_analysis
from tweetutils import tf_idf_tweets
from tweetutils import tf_idf_lsa_tweets
from tweetutils import tf_idf_pca_tweets
import numpy as np
import scipy
from collections import Counter
import pandas as pd




if __name__ == '__main__':
	# testing play ground, is set up to work on the #howtoconfuseamillennial dataset.
	# make sure this reflects what is in etl.py's on_status method.

	# setting up list for tweets to go into this acts like tweetlistmaster in etl
	tweetlistmaster = []


	# this will grab the file name where data is stored
	for k in range(0,820):

		base = 'data/brangelia/#Brangelina_batch'

		fil = base + '{0}.txt'.format(k)

		# only use millennial data
		if "Brangelina" in fil:
			print fil

			# open and load the file
			f = open(fil, 'r')
			tweet_batch = json.load(f)

			# append file data to full list
			tweetlistmaster = tweetlistmaster + tweet_batch
			userlist = [tweet["screen_name"] for tweet in tweetlistmaster]
			tweet_id = [tweet['tweet_id'] for tweet in tweetlistmaster]


			# vectorize
			hashed_tweets = tf_idf_pca_tweets(tweetlistmaster)

			# do silhouette analysis on master list
			n, tweet_pred = silhouette_analysis(hashed_tweets)


			# create cluster information on master list
			cluster_df = clusterinfo(n = n, vectorized_tweets = hashed_tweets, tweetlistmaster = tweetlistmaster, tweet_pred = tweet_pred)

			# write to file
			counts_to_file(cluster_df=cluster_df, base = base, batchnumber = k, n = n)

			# write cluster_df to file
			cluster_df.to_csv(base + 'pred'+ '{0}.csv'.format(k), encoding='utf-8')
















