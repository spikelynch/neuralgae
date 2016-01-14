#!/usr/bin/env python

import sys
import subprocess
import os.path
import shutil
import json
import random, math
import argparse
from neuralgae import ImageNet
import neuralgae
import classify

TARGETS = 20

DEFAULTS = {
    'nstart': 3,
    'ntween': 50,
    'nsample': 3,
    'scale': 80,
    'blend': 60,
    'model': 'googlenet',
    'size': 224,
    'iters': 300,
    'sigma': 0.33
}


def interpolate(v, k):
    if len(v) == 2:
        return v[0] + k * (v[1] - v[0])
    elif v[0] == 'random':
        k = random.random()
        return v[1] + k * (v[2] - v[1])
    elif v[0] == 'sinusoid':
        k = math.sin(k * math.pi)
        return v[1] + k * (v[2] - v[1])
    else:
        return v[1]

def interpolate_cf(cf, k):
    frame_cf = {}
    for key, value in cf.iteritems():
        if type(value) == list:
            frame_cf[key] = interpolate(value, k)
        else:
            frame_cf[key] = value
    print frame_cf
    return frame_cf
    


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--number", type=int, default=100, help="Number of frames to generate")
parser.add_argument("-i", "--initial", type=str, default=None, help="Initial image - if not supplied, will start with a random image")
parser.add_argument("-c", "--config", type=str, default=None, help="Global config file")
parser.add_argument("outdir",       type=str, help="Output directory")
parser.add_argument("outfile",      type=str, help="File to log image classes")

args = parser.parse_args()
imagen = ImageNet("./classes.txt")

outfile = os.path.join(args.outdir, args.outfile)

if args.config:
    with open(args.config, 'r') as f:
        cf = json.load(f)
else:
    cf = DEFAULTS

cfdump = os.path.join(args.outdir, 'global.conf')
with open(cfdump, 'w') as cfd:
    json.dump(cf, cfd, indent=4)
    
classes = ""

if args.initial:
    base = args.initial
    lastimage = "%s.jpg" % base
    initconf = "%s.json" % base
    t = neuralgae.read_config(initconf)
    if t:
        print t
        classes = ', '.join([imagen.name(c) for c in t])
else:
    start_targets = random.sample(range(0, TARGETS), cf['nstart'])
    conffile = os.path.join(args.outdir, 'conf0.json')
    print "Config file: %s " % conffile
    cf['targets'] = ','.join([ str(x) for x in start_targets ])
    frame_cf = interpolate_cf(cf, 0)
    neuralgae.write_config(frame_cf, conffile)
    subprocess.call(["./draw.sh", args.outdir, 'image0', conffile, str(cf['size']), str(int(frame_cf['scale'])), str(int(frame_cf['blend']))])
    lastimage = os.path.join(args.outdir, 'image0.jpg')
    classes = ', '.join([imagen.name(c) for c in start_targets])



with open(outfile, 'w') as f:
    f.write("%s: %s\n" % (lastimage, classes))

print "lastimage = %s" % lastimage

print "generating %d frames" % args.number

for i in range(1, args.number + 1):
    jsonfile = os.path.join(args.outdir, "conf%d.json" % i)
    targets = classify.classify(cf['model'], lastimage, cf['ntween'])
    if cf['nsample'] < cf['ntween']:
        targets = random.sample(targets, cf['nsample'])
    print targets
    #sys.exit(-1)
    cf['targets'] = ','.join([ str(x) for x in targets ])
    frame_cf = interpolate_cf(cf, 1.0 * i / (args.number + 1))
    neuralgae.write_config(frame_cf, jsonfile)
    subprocess.call(["./draw.sh", args.outdir, "image%d" % i, jsonfile, str(cf['size']), str(int(frame_cf['scale'])), str(int(frame_cf['blend']))])
    lastimage = os.path.join(args.outdir, "image%d.jpg" % i)
    t = neuralgae.read_config(jsonfile)
    if t:
        print t
        classes = ', '.join([imagen.name(c) for c in t])
    else:
        classes = ""
    with open(outfile, 'a') as f:
        f.write("%s: %s\n" % (lastimage, classes))
