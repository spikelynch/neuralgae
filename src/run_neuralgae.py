#!/usr/bin/env python

import sys
import subprocess
import os.path
import shutil
import json
import random
import argparse
from neuralgae import ImageNet
import neuralgae
import classify

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
    start_targets = random.sample(range(0, 1000), cf['nstart'])
    conffile = os.path.join(args.outdir, 'conf0.json')
    print "Config file: %s " % conffile
    cf['targets'] = ','.join([ str(x) for x in start_targets ])
    neuralgae.write_config(cf, conffile)
    subprocess.call(["./draw.sh", args.outdir, 'image0', conffile, str(cf['size']), str(cf['scale']), str(cf['blend'])])
    lastimage = os.path.join(args.outdir, 'image0.jpg')
    classes = ', '.join([imagen.name(c) for c in start_targets])



with open(outfile, 'w') as f:
    f.write("%s: %s\n" % (lastimage, classes))

print "lastimage = %s" % lastimage

print "generating %d frames" % args.number

for i in range(1, args.number + 1):
    jsonfile = os.path.join(args.outdir, "conf%d.json" % i)
    targets = classify.classify(lastimage, cf['ntween'])
    if cf['nsample'] < cf['ntween']:
        targets = random.sample(targets, cf['nsample'])
    print targets
    cf['targets'] = ','.join([ str(x) for x in targets ])
    neuralgae.write_config(cf, jsonfile)
    subprocess.call(["./draw.sh", args.outdir, "image%d" % i, jsonfile, str(cf['size']), str(cf['scale']), str(cf['blend'])])
    lastimage = os.path.join(args.outdir, "image%d.jpg" % i)
    t = neuralgae.read_config(jsonfile)
    if t:
        print t
        classes = ', '.join([imagen.name(c) for c in t])
    else:
        classes = ""
    with open(outfile, 'a') as f:
        f.write("%s: %s\n" % (lastimage, classes))
