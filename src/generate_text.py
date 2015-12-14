#!/usr/bin/env python

import nltk, random, re, argparse, sys, os.path


TEXT = 'definitions.txt'
LINELENGTH = 6
TWEET_CHARS = 116


def makePairs(arr):
    pairs = []
    for i in range(len(arr)):
        if i < len(arr)-1: 
            temp = (arr[i], arr[i+1])
            pairs.append(temp)
    return pairs

def generate(cfd, word = 'the', num = 50):
    words = []
    for i in range(num):
        arr = []
        for j in cfd[word]:
            for k in range(cfd[word][j]):
                arr.append(j)
        words.append(word)
        word = arr[int((len(arr))*random.random())]     
    return words


def make_tweet(cfd, words, num):
    outwords = []
    for w in words:
        start = w.replace(' ', '_')
        try:
            outwords += generate(cfd, start, args.number)
        except IndexError as e:
            print "ERROR: %s %s" % ( start, e )
    outwords = [ w.replace('_', ' ') for w in outwords ]
    tweet = ""
    for w in outwords:
        if len(tweet) + len(w) + 1 > TWEET_CHARS:
            break
        tweet += " " + w
    return tweet

    


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--number", type=int, default=LINELENGTH,help="Words per line")
parser.add_argument("-t", "--tweets", action='store_true', help="Limit output to %s characters" % TWEET_CHARS)
parser.add_argument("logfile",  type=str, help="Log file from neuralgia_control")


args = parser.parse_args()

if not os.path.isfile(args.logfile):
    print "Could not read %s" % args.logfile
    sys.exit(-1)



    


with open(TEXT, 'r') as f:
    corpus = f.read()
    corpus = corpus.split()
    pairs = makePairs(corpus)
    cfd = nltk.ConditionalFreqDist(pairs)

    line_re = re.compile('^(.*): (.*)$')

    with open(args.logfile, 'r') as lf:
        for l in lf:
            m = line_re.match(l)
            if m:
                image = m.group(1)
                bits = m.group(2)
                words = bits.split(', ')
                if args.tweets:
                    tweet = make_tweet(cfd, words, args.number)
                    print "%s:%s" % ( image, tweet )
                else:
                    print image
                    for w in words:
                        start = w.replace(' ', '_')
                        try:
                            line = ' '.join(generate(cfd, start, args.number))
                        except IndexError as e:
                            print "ERROR: %s %s" % ( start, e )
                        print line.replace('_', ' ')
                    print "\n"



        
