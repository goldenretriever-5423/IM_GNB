import pandas as pd
#para：chunkSize
chunkSize = 3000
tweets = pd.DataFrame()
chunk_df = pd.read_csv('/Volumes/TOSHIBA EXT/retweet/retweetstext.csv', \
                       header=None, \
                       names=['idretweet', 'text', 'date', 'regular_node', 'influencer', 'original_tweet_id'],\
                       iterator=True, sep='#1#8#3#', \
                       chunksize=chunkSize)
for chunk in chunk_df:
    tweets = tweets.append(chunk)
    break # just read first 1 million data
# print(tweets)