#!/usr/bin/env python

from nltk.corpus import wordnet as wn


from neuralgae import ImageCategories

OUTFILE = 'definitions.txt'

inet = ImageCategories('places')

with open(OUTFILE, 'w') as f:
    for i in range(0, 204):
        w = inet.name(i)
        w_ = w.replace(' ', '_')
        ss = wn.synsets(w_)
        print w
        if not ss:
            print "ERR: %d %s not found" % (i, w)
        for s in ss:
            f.write(w_ + " " + s.definition() + "\n")

        
