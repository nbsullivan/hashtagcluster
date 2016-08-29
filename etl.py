import tweepy
import json
from pprint import pprint as pp

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
	hashTag = '#tacobell'

	# get interatble cursor object.
	cursor = tweepy.Cursor(api.search, q = hashTag).items(3000)

	# load tweets into a list for writng to file
	tweetlist = []
	for item in cursor:
		tweetlist.append(item._json)
		print(item._json['text'])

	# do not uncomment unless you want to override the sample tweet set.
	# f = open('data/sampletweets.txt', 'w')
	# json.dump(tweetlist, f)
	# f.close()

	

	
	# just checking number of tweets recived.
	print len(tweetlist)
