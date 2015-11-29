#!/usr/bin/env python3

import pystache, os, os.path, re, shutil

BASE = '/Users/mike/Desktop/DeepDream/deepdream/Neuralgia'
SRC = BASE
DEST = os.path.join(BASE, 'Text', 'Output')

class Chapter(object):
    def __init__(self, c, stanzas):
        self.c = c
        self.stanzas = stanzas

class Stanza(object):
    def __init__(self, c, n, lines):
        self.c = c
        self.n = n
        self.lines = [ { 'line': l } for l in lines ]


def copy_image(c, n):
    src = os.path.join(SRC, "Chapter%s" % c, "image%s.jpg" % n)
    dst = os.path.join(DEST, "img", c, "%s.jpg" % n)
    #print(src, dst)
    shutil.copyfile(src, dst)

parens_re = re.compile('\(.*\)')

def clean_line(l):
    l = l[:-1]
    if not parens_re.search(l):
        l = re.sub('[\(\)]', '', l)
    return l
    
        

def read_stanzas(file, c):
    img_re = re.compile('^\./Neuralgia/image(\d+)\.jpg')
    line_re = re.compile('^([A-Za-z].*)$')
    n = None
    lines = []
    stanzas = []
    with open(file, 'r') as f:
        for l in f:
            mi = img_re.match(l)
            if mi:
                if lines:
                    if n is not None:
                        s = Stanza(c, n, lines)
                        stanzas.append(s)
                    else:
                        print("Error: lines found before image")
                        print(lines)
                        print(n)
                    lines = []
                n = mi.group(1)
            ml = line_re.match(l)
            if ml:
                lines.append(clean_line(l))
    if lines and n:
        s = Stanza(c, n, lines)
        stanzas.append(s)
    return stanzas

renderer = pystache.Renderer()

for cn in [ '1', '2', '3', '4', '5' ]:
    s = read_stanzas('./Stanzas/chapter%s.txt' % cn, cn)
    c = Chapter(cn, s)
    of = './Output/chapter%s.html' % cn
    with open(of, 'w') as cf:
        cf.write(renderer.render(c))
        print("Wrote %s" % of)

    for stanza in s:
        copy_image(c.c, stanza.n)
    

