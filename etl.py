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

	tweets = api.search(q = hashTag, count = 3000 )

	tweetlist = []
	k = 1
	for item in tweets:
		tweetlist.append(item._json)
		print(item._json['text'])


	print len(tweetlist)
