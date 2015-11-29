#!/usr/bin/env python

import subprocess
import os.path
import json
import random
import argparse
from neuralgae import ImageNet
import neuralgae



NSTART = 8

NTWEEN = 100
NSAMPLE = 8
SIZE = "224"
SCALE = "8"
BLEND = "75"


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--number", type=int, default=100, help="Number of frames to generate")
parser.add_argument("-i", "--initial", type=str, default=None, help="Initial image - if not supplied, will start with a random image")
parser.add_argument("outdir",       type=str, help="Output directory")
parser.add_argument("outfile",      type=str, help="File to log image classes")

args = parser.parse_args()



outfile = os.path.join(args.outdir, args.outfile)

if args.initial:
    lastimage = args.initial
else:
    start_targets = random.sample(range(0, 1000), NSTART)
    conffile = os.path.join(args.outdir, 'conf0.json')
    neuralgae.write_config(start_targets, conffile)
    subprocess.call(["./draw.sh", args.outdir, 'image0', conffile, SIZE, SCALE, BLEND])
    lastimage = os.path.join(args.outdir, 'image0.jpg')

imagen = ImageNet("./classes.txt")

classes = ', '.join([imagen.name(c) for c in start_targets])

with open(outfile, 'w') as f:
    f.write("%s: %s\n" % (lastimage, classes))


for i in range(0, args.number):
    jsonfile = os.path.join(args.outdir, "conf%d.json" % i)
    subprocess.call(["./classify.py", str(NTWEEN), str(NSAMPLE), lastimage, jsonfile])
    subprocess.call(["./draw.sh", args.outdir, "image%d" % i, jsonfile, SIZE, SCALE, BLEND])
    lastimage = os.path.join(args.outdir, "image%d.jpg" % i)
    t = neuralgae.read_config(jsonfile)
    if t:
        print t
        classes = ', '.join([imagen.name(c) for c in t])
    else:
        classes = ""
    with open(outfile, 'a') as f:
        f.write("%s: %s\n" % (lastimage, classes))
