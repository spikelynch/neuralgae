#!/usr/bin/env python2

import nltk, random, re, argparse, sys, os.path, itertools

import traceback

DEFINITIONS = './Definitions'
LINELENGTH = 6
TWEET_CHARS = 280 
OLD_LINE_RE = re.compile('^(.*): (.*)$')
WORD_RE = re.compile('^[a-zA-Z]')

def parse_line(l, model):
    """model = True - if this is true get model from the file"""
    m = OLD_LINE_RE.match(l)
    if m:
        image = m.group(1)
        bits = m.group(2)
        words = bits.split(', ')
        return image, words, None
    bits = l.split(',')
    image = bits[0]
    m = None
    if model:
        m = bits[1]
        bits = bits[1:]
    #words = [ bits[n] for n in range(0, len(bits)) if n % 2 ]
    words = [ b for b in bits if WORD_RE.match(b) ]
    words = words[1:]
    return image, words, m


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
        i = int((len(arr))*random.random())
        word = arr[i]     
    return words


def make_tweet(cfd, words, num, postlength):
    outwords = []
    tweet = ""
    done = False
    for w in words:
        start = w.replace(' ', '_')
        try:
            line = generate(cfd, start, args.number)
            line = [ w.replace('_', ' ') for w in line ]
            if tweet:
                tweet += '/'
	    for w in line:
                if len(tweet) + len(w) + 1 > postlength:
                    done = True
	            break
                if tweet:
                    if tweet[-1] != '/':
                        tweet += ' '
                tweet += w 
        except IndexError as e:
            print "ERROR: (%s) %s" % ( start, e )
            print traceback.format_exc()
        if done:
            break
    return tweet

    


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--number", type=int, default=LINELENGTH,help="Words per line")
parser.add_argument("-l", "--length", type=int, help="Twitter/Mastodon mode: limit size of output and use / for line breaks")
parser.add_argument("-m", "--model", type=str, default=None, help="Model name (to alternate use model1,model2)")
parser.add_argument("logfile",  type=str, help="Log file from neuralgia_control")


args = parser.parse_args()

if not os.path.isfile(args.logfile):
    print "Could not read %s" % args.logfile
    sys.exit(-1)

cfdi = None
cfds = {}
filemodel = True
if args.model:
    cfds = [ makeCfd(m) for m in args.model.split(',') ]
    cfdi = itertools.cycle(cfds)
    filemodel = False
  

with open(args.logfile, 'r') as lf:
    for l in lf:
        image, words, m = parse_line(l, filemodel)
        if cfdi:
            cfd = cfdi.next()
        else:
            if m:
                if not m in cfds:
                    cfds[m] = makeCfd(m)
                cfd = cfds[m]
            else:
                print "Couldn't work out a model for line\n%s\n\n" % l
                sys.exit(-1)
        if args.length:
            tweet = make_tweet(cfd, words, args.number, args.length)
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



        
