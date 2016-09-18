import tweepy
import json
from pprint import pprint as pp
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import scipy
from collections import Counter



def clean_tweet(tweet):
	# input a tweet and get back a dictionary form of its relevant content.
	# return tweet text, username, tweet_id, & geo-location (coordinates)
	tweet_dict = {}
	tweet_dict['text'] = tweet['text']
	tweet_dict['screen_name'] = tweet['user']['screen_name']
	tweet_dict['tweet_id'] = tweet['id']
		
	# return tweets with geo-location
	if tweet['geo'] != None:
		tweet_dict['geo'] = tweet['geo']['coordinates']

	return tweet_dict

def vectorize_tweets(tweetlist):
	# vectorized tweetlist inputs and outputs a sparce matrix representation
	vectorizer = CountVectorizer(min_df = 1)
	vectorized_tweets = vectorizer.fit_transform(map(lambda tweet: tweet['text'], tweetlist))
	names = vectorizer.get_feature_names()
	return vectorized_tweets, names


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
    


class MyStreamListener(tweepy.StreamListener):
	
	# tweetlist stores 100 tweets, tweetlistmaster stores all imported tweets
	tweetlist = []
	tweetlistmaster = []
	tweetcounter = 0
	batchnumber = 0

	def __init__(self, api=None, searchitem = None):
		# over ride the base _init_ method to accept a serach term parameter

		self.api = api
		self.searchterm = searchitem


	def on_status(self, status):
    	# we want to grab tweets in batches of 100 to send to the clustering algo, or maybe less than 100 or maybe more
    	# also at this stage we want to have something to process the tweets.

    	# clean the status, put in a list.
		cleantweet = clean_tweet(status._json)
		self.tweetlist.append(cleantweet)
		self.tweetlistmaster.append(cleantweet)
		self.tweetcounter += 1

		print self.tweetcounter

		if self.tweetcounter >= 100:
			# once 100 tweeets have been reached we do something. currently we write it to file, extract features, cluster and generate cluster information.

			# reset the counters
			self.tweetcounter = 0

			# dump to datafile for now
			f = open('data/{0}_batch{1}.txt'.format(self.searchterm, str(self.batchnumber)), 'w')
			json.dump(self.tweetlist, f)
			f.close()

			# wipe the tweetlist
			self.tweetlist = []

			# increment batch couter.
			self.batchnumber += 1

			# vectorize master tweetlist list
			vectorized_tweets, names = vectorize_tweets(self.tweetlistmaster)

			# for now use n = 3 this to be replaced with silhouette analysis.
			n = 3

			# send vectorized tweets to clustering algorithm
			clf = KMeans(n_clusters=n)
			tweet_pred = clf.fit_predict(vectorized_tweets)

			# create cluster_json list
			cluster_json = clusterinfo(n = n, vectorized_tweets = vectorized_tweets, names = names, tweetlistmaster = self.tweetlistmaster, tweet_pred = tweet_pred)

			# if you want to take a peak uncomment below.
			# pp(cluster_json)


			# UP NEXT: create happy json to send out for visualization

			# UP LATER: Shillouette analysis sklearn 






	



if __name__ == '__main__':

	# load the API keys
	with open('../keys.json', 'r') as fp:
		keys = json.load(fp)

	consumer_token = keys['APIkey']
	consumer_secret = keys['APIsecret']
	access_token = keys['AccessToken']
	access_secret = keys['AccessTokenSecret']

	# start a session with the twitter API
	auth = tweepy.OAuthHandler(consumer_token, consumer_secret)
	auth.set_access_token(access_token, access_secret)

	# create the API object
	api = tweepy.API(auth)

	# search term
	hashTag = ''
	print hashTag


	# creating stream listener
	SL = MyStreamListener(searchitem = hashTag)

	# starting stream? maybe?
	mystream = tweepy.Stream(auth = api.auth, listener = SL)

	# filtering?
	mystream.filter(track=[hashTag])
	