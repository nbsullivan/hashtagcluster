
import glob
import json
from pprint import pprint as pp
from tweetutils import clean_tweet
from tweetutils import vectorize_tweets
from tweetutils import clusterinfo
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import scipy
from collections import Counter
import pandas as pd

def counts_to_file(cluster_json, base, batchnumber):
	# for a cluster_json write it to a csv file, for analysis

	word_list = []
	user_list = []
	for j in range(len(cluster_json['Clusterlist'])):
		print j
		word_row = {}
		user_row = {}
		

		for key in cluster_json['Clusterlist'][j]['bagofwords']:
			# print key
			word_row["Cluster"] = j
			word_row["Word"] = key
			word_row["Count"] = cluster_json['Clusterlist'][j]['bagofwords'][key]

			word_list.append(word_row)

			word_row = {}

		for key in cluster_json['Clusterlist'][j]['userscounts']:
			# print key
			user_row["Cluster"] = j
			user_row["User"] = key
			user_row["Count"] = cluster_json['Clusterlist'][j]['userscounts'][key]

			user_list.append(user_row)

			user_row = {}

	# put into  df for easy writing.
	user_df = pd.DataFrame(user_list)
	words_df = pd.DataFrame(word_list)

	# write to file
	user_df.to_csv(base + '{0}usercount.csv'.format(k), encoding='utf-8')
	words_df.to_csv(base + '{0}wordcount.csv'.format(k), encoding='utf-8')

	print user_df.head()
	print words_df.head()






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


			# vectorize
			vectorized_tweets, names = vectorize_tweets(tweetlistmaster)

			# need code for silhouette analysis
			n = 3
			# cluster
			clf = KMeans(n_clusters=n)
			tweet_pred = clf.fit_predict(vectorized_tweets)

			# we want to subset the vectorized tweets based on tweet_pred, also create a list of dictionarys with word counts per cluster
			
			cluster_json = clusterinfo(n = n, vectorized_tweets = vectorized_tweets, names = names, tweetlistmaster = tweetlistmaster, tweet_pred = tweet_pred)
			
			counts_to_file(cluster_json, base, batchnumber = k)













