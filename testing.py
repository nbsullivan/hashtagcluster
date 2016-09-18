
import glob
import json
from pprint import pprint as pp
from etl import clean_tweet
from etl import vectorize_tweets
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import scipy
from collections import Counter


def clusterinfo(n = 2, vectorized_tweets = None, names = None, tweetlistmaster = None, tweet_pred = None):
	# we want to subset the vectorized tweets based on tweet_pred, also create a list of dictionarys with word counts per cluster
	dict_list = []

	# put everything into the full_info dict
	full_info = {}


	# loop over number of subsets
	for k in range(n):

		# prepare a dictionary for cluster information
		cluster_dict = {}

		# rows with label k.
		indexes = [i for i, x in enumerate(tweet_pred) if x == k]
		
		# subset vectorized tweets
		subset_words = vectorized_tweets[np.array(indexes)]

		# sum the subset into  a single vector
		total_vector = subset_words.sum(axis = 0)

		# note .tolist will return a nested list
		total_vector = total_vector.tolist()
		total_vector = total_vector[0];

		# convert the long to int for entries in total_vector
		total_vector = map(int, total_vector)

		# form dictionary of words.
		word_dict = dict(zip(names, total_vector))

		# reduce the size of the word dictionary
		reduc_word = {k:v for k,v in word_dict.items() if v != 0}

		# pp(reduc_word)

		cluster_dict["words"] = reduc_word

		# get unigue users and number of tweets from them.
		cluster_users = [userlist[i] for i in indexes]
		user_dict = dict(Counter(cluster_users))


		cluster_dict["users"] = user_dict

		# get tweet ids users and number of tweets from them.
		tweet_id_list = [tweet_id[i] for i in indexes]

		cluster_dict["tweet_ids"] = tweet_id_list
		cluster_dict["tweetsize"] = len(tweet_id_list)

		dict_list.append(cluster_dict)

	full_info["Clusterlist"] = dict_list
	full_info["totaltweet"] = len(tweetlistmaster)


	return full_info







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
