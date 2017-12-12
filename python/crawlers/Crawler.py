'''
Store data at iphone.json
'''
import tweepy
from tweepy import OAuthHandler
import json
from tweepy import Stream
from tweepy.streaming import StreamListener

consumer_key = 'B3i8Mz3SQnHPVmy50R0KYQnJu'
consumer_secret = 'XCO6uQXzRJnqk5C9PfRGhHI1nVI1k037mjGZD54voR2EmReMDR'
access_token = '923266112594333698-657sEzUmA1KiI1vAgAYg2KhrSJzJWja'
access_secret = 'RUyawd6kzi3AWNfN5DVZJszQbEhsBMdSlBN5zxKQZngln'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)


class MyListener(StreamListener):
    def on_data(self, data):
        try:
            with open('data/iphone1.json', 'a') as f:
                f.write(data)
                return True
        except BaseException as e:
            print("Error on_data:%s" % str(e))
        return True

    def on_error(self, status):
        print(status)



twitter_stream = Stream(auth, MyListener())
twitter_stream.filter(track=['iPhone 8', 'all glass design', 'water and dust resistant', 'wireless charging', 'two sizes'])
