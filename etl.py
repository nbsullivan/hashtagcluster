import tweepy
import json
from pprint import pprint as pp
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import scipy
from collections import Counter
from tweetutils import clean_tweet






class MyStreamListener(tweepy.StreamListener):

	# tweetlist stores 100 tweets, tweetlistmaster stores all imported tweets
	tweetlist = []
	tweetcounter = 0
	batchnumber = 0

	def __init__(self, api=None, searchitem = None):
		"""over ride the base _init_ method to accept a serach term parameter"""

		self.api = api
		self.searchterm = searchitem


	def on_status(self, status):
		"""we want to grab tweets in batches of 100 to send to the clustering algo, or maybe less than 100 or maybe more
		also at this stage we want to have something to process the tweets."""

		# clean the status, put in a list.
		cleantweet = clean_tweet(status._json)
		self.tweetlist.append(cleantweet)
		self.tweetcounter += 1



		if self.tweetcounter >= 100:
			print self.searchterm, ' batch number ', self.batchnumber
			# once 100 tweeets have been reached we do something. currently we write it to file, extract features, cluster and generate cluster information.

			# reset the counters
			self.tweetcounter = 0

			# dump to datafile for now
			f = open('data/current_data/{0}_batch{1}.txt'.format(self.searchterm, str(self.batchnumber)), 'w')
			json.dump(self.tweetlist, f)
			f.close()

			# wipe the tweetlist
			self.tweetlist = []

			# increment batch couter.
			self.batchnumber += 1






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
	hashTag = '#WorldTeachersDay'
	print hashTag


	# creating stream listener
	SL = MyStreamListener(searchitem = hashTag)

	# starting stream? maybe?
	mystream = tweepy.Stream(auth = api.auth, listener = SL)

	# filtering?
	mystream.filter(track=[hashTag])
	
