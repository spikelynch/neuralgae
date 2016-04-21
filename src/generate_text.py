#!/usr/bin/env python2

import nltk, random, re, argparse, sys, os.path, itertools



DEFINITIONS = './Definitions'
LINELENGTH = 6
TWEET_CHARS = 116
OLD_LINE_RE = re.compile('^(.*): (.*)$')

def parse_line(l):
    m = OLD_LINE_RE.match(l)
    if m:
        image = m.group(1)
        bits = m.group(2)
        words = bits.split(', ')
        return image, words
    bits = l.split(',')
    image = bits[0]
    words = [ bits[n] for n in range(0, len(bits)) if n % 2 ]
    return image, words


def makePairs(arr):
    pairs = []
    for i in range(len(arr)):
        if i < len(arr)-1: 
            temp = (arr[i], arr[i+1])
            pairs.append(temp)
    return pairs


def makeCfd(m):
    deffile = os.path.join(DEFINITIONS, '%s.txt' % m)
    if not os.path.isfile(deffile):
        print "Can't find %s" % deffile
        sys.exit(-1)
    with open(deffile, 'r') as f:
        corpus = f.read()
        corpus = corpus.split()
        pairs = makePairs(corpus)
        cfd = nltk.ConditionalFreqDist(pairs)
    return cfd

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
parser.add_argument("-m", "--model", type=str, default="googlenet", help="Model name (for definitions) (to alternate use model1,model2")
parser.add_argument("logfile",  type=str, help="Log file from neuralgia_control")


args = parser.parse_args()

if not os.path.isfile(args.logfile):
    print "Could not read %s" % args.logfile
    sys.exit(-1)

cfds = [ makeCfd(m) for m in args.model.split(',') ]

cfdi = itertools.cycle(cfds)


with open(args.logfile, 'r') as lf:
    for l in lf:
        image, words = parse_line(l)
        cfd = cfdi.next()
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



        
