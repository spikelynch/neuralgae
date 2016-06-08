#!/usr/bin/env python2

import random

COLORFILE = '/opt/X11/share/X11/rgb.txt'

colors = []
with open(COLORFILE, 'r') as cf:
    for line in cf:
        if line[0] == '#':
            next
        parts = line[:-1].split()
        if len(parts) == 4:
            colors.append(parts[3])

cs = random.sample(colors, 1)

print cs[0]
