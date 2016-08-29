import tweepy
import json
from pprint import pprint as pp

if __name__ == '__main__':

	with open('../keys.json', 'r') as fp:
		keys = json.load(fp)

	consumer_token = keys['APIkey']
	consumer_secret = keys['APIsecret']
	access_token = keys['AccessToken']
	access_secret = keys['AccessTokenSecret']
	auth = tweepy.OAuthHandler(consumer_token, consumer_secret)
	auth.set_access_token(access_token, access_secret)

	api = tweepy.API(auth)


	hashTag = '#tacobell'

	cursor = tweepy.Cursor(api.search, q = hashTag).items(3000)

	tweetlist = []
	for item in cursor:
		tweetlist.append(item._json)
		print(item._json['text'])

	

	

	print len(tweetlist)
