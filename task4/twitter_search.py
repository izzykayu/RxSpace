# Author: Izzy Metzger
# Last edited 0:49
# Usage: python twitter_search.py --help
from __future__ import print_function
import GetOldTweets3 as got
import argparse
import jsonlines
from io import open
import os
import datetime

today = datetime.datetime.today().strftime('%Y-%m-%d %H:%M')
print(today)


def mkdir_if_not_exist(path):
    """
    path to folder
    :param path: e.g. 'tweets'
    :return: a directory called tweets made
    """
    if not os.path.isdir(path):
        os.mkdir(path)


def print_description(which='all'):
    """
    this function prints
    :param which:
    :return:
    """
    csv_description = "csv file will be comma-delimited with the column names: \
        ['username', 'reply_to', 'date', 'mentions', 'hashtags', 'geo', 'urls', 'text']"
    if which == 'csv_info':
        print(csv_description)

    elif which == 'packages':
        with open(file='requirements.txt') as f:
            requirements = f.readlines()
            print('requirements needed for python3.6 environment')
            print(requirements)
    if which == 'columns':
        column_info = """username:  username of twitter account that posted (e.g., username Roivant refers to a tweet 
        from @Roivant account
        reply_to:   if tweet post is a reply_to then this will provide that text, if not 'NaN'
        date:   datetime string when tweet was posted in "YEAR-MONTH-DAY HOUR:MINUTE" format using 24-hr clock')
        mentions:   if other twitter accounts were mentioned in the post, they will be listed here delimited  by 
        commas (e.g., )')
        hashtags:   any #hashtag terms delimited by comma
        geo:    less than 2% of tweets are geo-tagged but if THEY are, lat and long will be returned in this value
        urls:   any urls in the tweet
        text:   the tweet text with utf-8 encoding
        """
        print(column_info)

        # TODO: write out all of the column names


def write_twitter_file(username, keyword, result_dir, verbose=True, set_since=None):
    """
    this function takes a keyword (or key phrase) and queries twitter for tweets containing it
    :param keyword: phrase or word to query twitter
    :param result_dir: directory to write file response
    :param verbose: if True, prints out tweets
    :param set_since: default is None, date passed to get tweets from
    :return: csv file written to a directory with naming convention
    "TwitteratureSearch-Date-<today%Y-%M-%Dformat>-phrase-<keyword>-numResults-<NumberReturned>.csv"
    with column names of ['username', 'reply_to', 'date', 'mentions', 'hashtags', 'geo', 'urls', 'text']
    e.g., output:
    ... Found 1020 tweets using search term/phrase "endometriosis AbbVie"
    wrote csv file to directory "tweets"
    file name is "TwitteratureSearch-Date-2019-07-17-phrase-endometriosis_AbbVie-numResults-1020.csv"

    """
    if username != None:
        tweetCriteria = got.manager.TweetCriteria().setUsername(username)
        tweets = got.manager.TweetManager.getTweets(tweetCriteria)
        file_name = 'TwitteratureSearch-Date-' + str(
            today.split(' ')[0]) + '-username-' + username + '-numResults-' + str(len(tweets)) + '.csv'

    elif set_since != None:
        tweetCriteria = got.manager.TweetCriteria().setQuerySearch(keyword).setSince(set_since)
        tweets = got.manager.TweetManager.getTweets(tweetCriteria)
        keyword_phrase = '_'.join(keyword.split(' '))
        file_name = 'TwitteratureSearch-Date-' + str(
            today.split(' ')[0]) + '-Keyword-' + keyword_phrase + '-setSince-' + set_since + '-numResults-' + str(
            len(tweets)) + '.csv'
    else:
        tweetCriteria = got.manager.TweetCriteria().setQuerySearch(keyword)
        tweets = got.manager.TweetManager.getTweets(tweetCriteria)
        keyword_phrase = '_'.join(keyword.split(' '))
        file_name = 'TwitteratureSearch-Date-' + str(
            today.split(' ')[0]) + '-Keyword-' + keyword_phrase + '-numResults-' + str(len(tweets)) + '.csv'

    temp_file_name = open(os.path.join(result_dir, file_name), 'w')
    counter = 0
    temp_file_name.write(";".join(['username', 'reply_to', 'date', 'mentions', 'hashtags', 'geo', 'urls', 'text'])
                         + os.linesep)
    for tweet in tweets:
        counter += 1

        if not tweet.to:
            date_full = str(tweet.date)

            temp_file_name.write(tweet.username +
                                 ";" + ";" +
                                 date_full + ";" + tweet.mentions + ";" + tweet.geo +
                                 ";" + tweet.urls + ";" +
                                 '"' + tweet.text + '"' + os.linesep)
            if verbose == True:
                print(counter)
                print(tweet.text)

        else:
            date_full = str(tweet.date)
            temp_file_name.write(tweet.username +
                                 ";" + tweet.to + ";" +
                                 date_full + ";" + tweet.mentions + ";" + tweet.geo +
                                 ";" + tweet.urls + ";" +
                                 '"' + tweet.text + '"' + os.linesep)
            if verbose == True:
                print(counter)
                print(tweet.text)

    if verbose == True:
        print("header names:\n['username', 'reply_to', 'date', 'mentions', 'hashtags', 'geo', 'urls', 'text']")
        print('... ' + str(counter) + ' tweets found using search term/phrase ' + keyword)
        print('wrote csv file to directory "' + result_dir + '"')
        print('file name is "' + file_name + '"')
    temp_file_name.close()


# Command line options
optargs = [
    ('--keyword', {
        'type': str,
        'default': 'announces topline results',
        'help': 'keyword (search phrase) you would like to search twitter. Default keyword is "roivant"',
        }),
    ('--result_dir', {
        'type': str,
        'default': 'tweets',
        'help': 'Path to output directory where the twitter csv file written to will be. Default result_dir is "tweets"'
        }),
    ('--verbose', {
        'type': bool,
        'default': True,
        'help': 'True/False for printing statement (default verbose is True)'
        }),
    ('--set_since', {
        'type': str,
        'default': None,
        'help': 'date to search tweets from (e.g., if you want to query tweets from 2019-07-13 to present,'
                ' you would pass the argument --set_since 2019-07-13). Default is None (meaning no date-limit and '
                'that tweets from all time are searched)'
        }),
    ('--username', {
        'type': str,
        'default': None,
        'help': 'if you want to get tweets from one tweeter via their username (e.g., @Roivant'

        }),
    # ('--description', {
    #     'type': bool,
    #     'default': ,
    #     'help': 'if True then will print out description',
    # })

    ]

if __name__ == '__main__':
    '''
    query with keyword and write tweets
    Usage python search_twitter.py
    e.g., python example.py --keyword "Roivant"
    '''
    parser = argparse.ArgumentParser()
    for opt, config in optargs:
        parser.add_argument(opt, **config)

    args = parser.parse_args()
    mkdir_if_not_exist(args.result_dir)
    # print_description(args.description)
    write_twitter_file(username=args.username, keyword=args.keyword, result_dir=args.result_dir, verbose=args.verbose,
                       set_since=args.set_since)
