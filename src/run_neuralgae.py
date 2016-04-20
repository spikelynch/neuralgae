#!/usr/bin/env python2


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
    'bg_script': 'bg_perlin1.sh',
    'bg_params': [ '80', '60' ],
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

def do_classify(cf, remote_cf, model, lastimage):
    if cf['weighted']:
        return do_classify_weighted(cf, remote_cf, model, lastimage)
    targets = []
    if remote_cf:
        targets = classify.classify_remote(model, lastimage, cf['ntween'], remote_cf)
    else:
        targets = classify.classify(model, lastimage, cf['ntween'])
    if cf['nsample'] < cf['ntween']:
        targets = random.sample(targets, cf['nsample'])
    print targets
    return ','.join([ str(x) for x in targets ])

def do_classify_weighted(cf, remote_cf, model, lastimage):
    if remote_cf:
        t = classify.classify_remote(model, lastimage, 0, remote_cf)
    else:
        t = classify.classify(model, lastimage, 0)
    targets = dict(sorted(t.items(), key=lambda x: -x[1])[:cf['ntween']])
    if cf['nsample'] < cf['ntween']:
        ts = random.sample(targets.keys(), cf['nsample'])
        ots = targets
        targets = { t:ots[t] for t in ts }
    print targets
    return targets


def class_names(model, t):
    imagen = ImageCategories(model)
    return ', '.join([imagen.name(c) for c in t])


def format_target(v):
    if type(v) == float:
        v = "%.5f" % v
    else:
        v = str(v)
    return v


def log_line(image, model, t):
    imagen = ImageCategories(model)
    td = t
    if not type(t) == dict:
        td = { int(i): 1 for i in t }
    else:
        td = { int(i): v for (i, v) in t.iteritems() }
    nd = { imagen.name(i): v for (i, v) in td.iteritems() }
    nld = sorted(nd.items(), key=lambda x: -x[1])
    row = [ str(e) for l in nld for e in l ]
    return image + ',' + ','.join(row) + "\n"
#         [int(i) for i in t.keys()]


def make_background(cf, bgfile):
    bg_script = './' + cf['bg_script']
    bg_params = cf['bg_params']
    args = [ bg_script, bgfile, str(cf['size']) ] + bg_params
    print args
    subprocess.call(args)
        
def deepdraw(conffile, infile, outdir, outfile):
    dd = [ './dream.py', '--config', conffile, '--basefile', outfile, infile, outdir ]
    print "deepdraw %s" % dd
    subprocess.call(dd)
    return os.path.join(outdir, outfile) + '.jpg'

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--number", type=int, default=42, help="Number of frames to generate")
parser.add_argument("-i", "--initial", type=str, default=None, help="Initial image - if not supplied, will start with a random image")
parser.add_argument("-c", "--config", type=str, default=None, help="Global config file")
parser.add_argument("-r", "--remote", type=str, default=None, help="Remote classify parameters")
parser.add_argument("-s", "--static", action='store_true', help="Same base image for all frames", default=False)
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
    

mi = model_iter(cf['model'])

start_targets = None

if args.initial:
    model = mi.next()
    base = args.initial
    lastimage = "%s.jpg" % base
    initconf = "%s.json" % base
    start_targets = neuralgae.read_config(initconf)
else:
    model = mi.next()
    start_targets = random.sample(range(0, NTARGETS[model]), cf['nstart'])
    conffile = os.path.join(args.outdir, 'conf0.json')
    print "Config file: %s " % conffile
    cf['target'] = ','.join([ str(x) for x in start_targets ])
    frame_cf = cf # interpolate_cf(cf, 0)
    frame_cf['model'] = model
    neuralgae.write_config(cf, conffile)
    make_background(cf, 'bg.jpg')
    lastimage = deepdraw(conffile, 'bg.jpg', args.outdir, 'image0')
    #sys.exit(-1)
    
with open(outfile, 'w') as f:
    f.write(log_line(lastimage, model, start_targets))

print "lastimage = %s" % lastimage

print "generating %d frames" % args.number

for i in range(1, args.number):
    model = mi.next()
    jsonfile = os.path.join(args.outdir, "conf%d.json" % i)
    cf['target'] = do_classify(cf, remote_cf, model, lastimage)
    frame_cf = cf # interpolate_cf(cf, 1.0 * i / (args.number + 1))
    frame_cf['model'] = model
    neuralgae.write_config(frame_cf, jsonfile)
    if not args.static:
        make_background(frame_cf, 'bg.jpg')
    lastimage = deepdraw(jsonfile, 'bg.jpg', args.outdir, 'image%d' % i)
    
    targets = neuralgae.read_config(jsonfile)
    with open(outfile, 'a') as f:
        f.write(log_line(lastimage, model, targets))
