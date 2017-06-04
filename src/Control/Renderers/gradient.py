#!/usr/bin/env python

import json
import sys

# subdivide octaves


def lerp(x1, x2, k):
    return x1 + (x2 - x1) * k
    

def expand(o, scale1, divs):
    n = []
    layer = o['layer']
    if 'scale' not in o:
        scale2 = 1
    else:
        scale2 = o['scale']
    esc = (1.0 * scale2 / scale1) ** ( 1.0 / divs )
    iters = o['iter_n'] // divs
    sc = scale1
    for i in range(divs):
        k1 = i / divs
        k2 = (i+1) / divs
        sigma1 = lerp(o['start_sigma'], o['end_sigma'], k1)
        sigma2 = lerp(o['start_sigma'], o['end_sigma'], k2)
        step1 = lerp(o['start_step_size'], o['end_step_size'], k1)
        step2 = lerp(o['start_step_size'], o['end_step_size'], k2)
        n.append({
            'layer' : layer,
            'iter_n' : iters,
            'start_sigma': sigma1,
            'end_sigma': sigma2,
            'start_step_size': step1,
            'end_step_size': step2,
            'scale': esc
            })
    return n

divs = 10

o1 = json.load(sys.stdin)

o2 = []

scale = 1

for o in o1:
    o2 += expand(o, scale, divs)
    if 'scale' not in o:
        scale = 1
    else:
        scale = o['scale']

print(json.dumps(o2))

