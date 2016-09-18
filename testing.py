
import glob
import json
from pprint import pprint as pp
from etl import clean_tweet
from etl import vectorize_tweets
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import scipy




if __name__ == '__main__':
	# testing play ground, is set up to work on the #howtoconfuseamillennial dataset.
	# make sure this reflects what is in etl.py's on_status method.

	# setting up list for tweets to go into this acts like tweetlistmaster in etl
	tweetlistmaster = []


	# this will grab the file name where data is stored
	for fil in glob.glob('data/*.txt'):

		# only use millennial data
		if "#HowtoConfuseaMillennial" in fil:
			print fil

			# open and load the file
			f = open(fil, 'r')
			tweet_batch = json.load(f)

			# append file data to full list
			tweetlistmaster = tweetlistmaster + tweet_batch

			# vectorize
			vectorized_tweets = vectorize_tweets(tweetlistmaster)


			# wtf is vectorizer doing.
			vectorizer = CountVectorizer(min_df = 1)
			map(lambda tweet: tweet['text'], tweetlistmaster)
			vectorizer.fit_transform(map(lambda tweet: tweet['text'], tweetlistmaster))
			names = vectorizer.get_feature_names()

			n = 3
			# cluster
			clf = KMeans(n_clusters=n)
			tweet_pred = clf.fit_predict(vectorized_tweets)

			# we want to subset the vectorized tweets based on tweet_pred, also create a list of dictionarys with word counts per cluster
			subset_list = []
			dict_list = []


			# loop over number of subsets
			for k in range(n):

				# rows with label k.
				indexes = [i for i, x in enumerate(tweet_pred) if x == k]
				
				# subset vectorized tweets
				subset = vectorized_tweets[np.array(indexes)]

				# sum the subset into  a single vector
				total_vector = subset.sum(axis = 0)

				# note .tolist will return a nested list
				total_vector = total_vector.tolist()
				total_vector = total_vector[0];

				# total_vector appears to have some funnny data type that is storing for the numbers
				total_vector = map(int, total_vector)

				# form dictionary of words.
				word_dict = dict(zip(names, total_vector))

				# reduce the size of the word dictionary
				reduc_word = {k:v for k,v in word_dict.items() if v != 0}

				pp(reduc_word)



				""" todo:
					get number of unique users in a cluster
					get number of unique tweets in a cluster
					"""

