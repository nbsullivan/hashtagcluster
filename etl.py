import tweepy
import json
from pprint import pprint as pp

class MyStreamListener(tweepy.StreamListener):

	tweetlist = []
	tweetcounter = 0
	batchnumber = 0

	def __init__(self, api=None, searchitem = None):
		# over ride the base _init_ method to accept a serach term parameter
		self.api = api
		self.searchterm = searchitem


	def on_status(self, status):

    	# we want to grab tweets in batches of 100 to send to the clustering algo, or maybe less than 100 or maybe more
    	# also at this stage we want to have something to process the tweets.
		self.tweetlist.append(self.clean_tweet(status._json))
		self.tweetcounter += 1
		print self.searchterm

		print self.tweetcounter

		if self.tweetcounter >= 100:
			# once 100 tweeets have been reached we do something. for now we are just writing to a file
			print "reached 100 tweets"

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

        	# send the tweets to the clustering algo.

	def clean_tweet(self, tweet):
    	# do something here to clean up the tweet for processing.
    	# this is where the preprocessing will take place.

		return tweet

    	



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
	hashTag = '#HowtoConfuseaMillennial'

	# "creating stream listener"
	SL = MyStreamListener(searchitem = hashTag)

	# "starting stream? maybe?"
	mystream = tweepy.Stream(auth = api.auth, listener = SL)

	# "filtering?"
	mystream.filter(track=[hashTag])
	



	# # get interatble cursor object.
	# cursor = tweepy.Cursor(api.search, q = hashTag).items(3000)

	# # load tweets into a list for writng to file
	# tweetlist = []
	# for item in cursor:
	# 	tweetlist.append(item._json)
	# 	print(item._json['text'])

	# do not uncomment unless you want to override the sample tweet set.
	# f = open('data/sampletweets.txt', 'w')
	# json.dump(tweetlist, f)
	# f.close()

	

	
	# just checking number of tweets recived.
	# print len(tweetlist)
