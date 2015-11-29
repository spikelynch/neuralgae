#!/usr/bin/env python

from nltk.corpus import wordnet as wn


from imagenet import ImageNet

CLASSES = '../../Classes/classes.txt'
OUTFILE = 'definitions.txt'

inet = ImageNet(CLASSES)

with open(OUTFILE, 'w') as f:
    for i in range(0, 1000):
        w = inet.name(i)
        w_ = w.replace(' ', '_')
        ss = wn.synsets(w_)
        print w
        if not ss:
            print "ERR: %d %s not found" % (i, w)
        for s in ss:
            f.write(w_ + " " + s.definition() + "\n")

        
