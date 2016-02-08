#!/usr/bin/env python


import sys
import subprocess
import os.path
import shutil
import json
import random, math, itertools
import argparse
from neuralgae import ImageCategories
import neuralgae
import classify


NTARGETS = {
    'googlenet': 1000,
    'caffenet': 1000,
    'places': 204,
    'flickr_style': 10,
    'oxford': 102
}

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
    

def model_iter(cfm):
    l = cfm.split(',')
    return itertools.cycle(l)

def class_names(model, t):
    imagen = ImageCategories(model)
    return ', '.join([imagen.name(c) for c in t])

    


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--number", type=int, default=100, help="Number of frames to generate")
parser.add_argument("-i", "--initial", type=str, default=None, help="Initial image - if not supplied, will start with a random image")
parser.add_argument("-c", "--config", type=str, default=None, help="Global config file")
parser.add_argument("-r", "--remote", type=str, default=None, help="Remote classify parameters")
parser.add_argument("outdir",       type=str, help="Output directory")
parser.add_argument("outfile",      type=str, help="File to log image classes")

args = parser.parse_args()

outfile = os.path.join(args.outdir, args.outfile)

if args.config:
    with open(args.config, 'r') as f:
        cf = json.load(f)
else:
    cf = DEFAULTS

if args.remote:
    with open(args.remote, 'r') as f:
        remote_cf = json.load(f)
else:
    remote_cf = None
    
imagen = ImageCategories(cf['model'])

cfdump = os.path.join(args.outdir, 'global.conf')
with open(cfdump, 'w') as cfd:
    json.dump(cf, cfd, indent=4)
    
classes = ""

mi = model_iter(cf['model'])

if args.initial:
    model = mi.next()
    base = args.initial
    lastimage = "%s.jpg" % base
    initconf = "%s.json" % base
    t = neuralgae.read_config(initconf)
    if t:
        print t
        classes = class_names(model, t)
else:
    model = mi.next()
    start_targets = random.sample(range(0, NTARGETS[model]), cf['nstart'])
    conffile = os.path.join(args.outdir, 'conf0.json')
    print "Config file: %s " % conffile
    cf['targets'] = ','.join([ str(x) for x in start_targets ])
    frame_cf = interpolate_cf(cf, 0)
    frame_cf['model'] = model
    neuralgae.write_config(frame_cf, conffile)
    subprocess.call(["./draw.sh", args.outdir, 'image0', conffile, str(cf['size']), str(int(frame_cf['scale'])), str(int(frame_cf['blend']))])
    lastimage = os.path.join(args.outdir, 'image0.jpg')
    classes = class_names(model, start_targets)



with open(outfile, 'w') as f:
    f.write("%s: %s\n" % (lastimage, classes))

print "lastimage = %s" % lastimage

print "generating %d frames" % args.number

for i in range(1, args.number + 1):
    model = mi.next()
    jsonfile = os.path.join(args.outdir, "conf%d.json" % i)
    targets = []
    if remote_cf:
        targets = classify.classify_remote(model, lastimage, cf['ntween'], remote_cf)
    else:
        targets = classify.classify(model, lastimage, cf['ntween'])
    if cf['nsample'] < cf['ntween']:
        targets = random.sample(targets, cf['nsample'])
    print targets
    cf['targets'] = ','.join([ str(x) for x in targets ])
    frame_cf = interpolate_cf(cf, 1.0 * i / (args.number + 1))
    frame_cf['model'] = model
    neuralgae.write_config(frame_cf, jsonfile)
    subprocess.call(["./draw.sh", args.outdir, "image%d" % i, jsonfile, str(cf['size']), str(int(frame_cf['scale'])), str(int(frame_cf['blend']))])
    lastimage = os.path.join(args.outdir, "image%d.jpg" % i)
    t = neuralgae.read_config(jsonfile)
    if t:
        print t
        classes = class_names(model, t)
    else:
        classes = ""
    with open(outfile, 'a') as f:
        f.write("%s: %s\n" % (lastimage, classes))
