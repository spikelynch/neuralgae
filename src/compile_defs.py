#!/usr/bin/env python

from nltk.corpus import wordnet as wn


from neuralgae import ImageCategories

OUTFILE = 'definitions_places.txt'

inet = ImageCategories('places')

def find_synsets(w):
    ss = wn.synsets(w)
    if ss:
        return ss
    ss = []
    ws = w.split('_')
    for w1 in ws:
        ss1 = wn.synsets(w1)
        if ss1:
            ss += ss1
    return ss


with open(OUTFILE, 'w') as f:
    for i in range(0, 204):
        w = inet.name(i)
        w_ = w.replace(' ', '_')
        ss = find_synsets(w_)
        if not ss:
            print "ERR: %d %s not found" % (i, w)
            f.write(w_ + " ERR\n")
        else:
            for s in ss:
                f.write(w_ + " " + s.definition() + "\n")

        
