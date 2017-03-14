#!/usr/bin/env python2


import sys
import subprocess
import copy
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
    'oxford': 102,
    'manga_tag': 1538,
    'manga': 4096
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
    'sigma': 0.33,
    'cutoff': 0.001,
    'octaves': None
}

CONF_TEMP = 'conf%03d'
IMG_TEMP = 'img%03d'

def conffilename(i):
    return ( CONF_TEMP % i ) + '.json'

def imgfilename(i):
    return IMG_TEMP % i


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
    return ','.join([ str(x) for x in targets ])

def do_classify_weighted(cf, remote_cf, model, lastimage):
    if remote_cf:
        t = classify.classify_remote(model, lastimage, 0, remote_cf)
    else:
        t = classify.classify(model, lastimage, 0)
    t0 = dict(sorted(t.items(), key=lambda x: -x[1])[:cf['ntween']])
    cutoff = 0
    print("classified: {}".format(t0))
    targets = { c: t0[c] for c in t0.keys() if t0[c] > cutoff }
    if len(targets) < len(t0):
        print "Discarded some zeroes"
    if cf['nsample'] < len(targets):
        ts = random.sample(targets.keys(), cf['nsample'])
        ots = targets
        targets = { t:ots[t] for t in ts }
    print("using: {}".format(targets))
    return targets


def flatten_targets(cf, ltargets):
    """Raise all to 1.0"""
    for t in cf['target'].keys():
        cf['target'][t] = 1.0
    return cf['target']


def perturb_targets(cf, ltargets):
    d = match_targets(ltargets, cf['target'])
    print("matchiness = {}".format(d))
    if d >= cf['reset']:
        print("Match between targets {} > {}".format(d, cf['reset']))
        cf['target'] = remove_max_targets(cf['target'])
        print("Remaining targets {}".format(cf['target']))
    if len(cf['target']):
      print("Rescaling remaining targets")
      cf['target'] = rescale_targets(cf['target'])
    else:
      print("Resetting all targets")
      cf['target'] = { t: 1 for t in random.sample(range(0, NTARGETS[model]), cf['ntween']) }
    return cf['target']


def match_targets(a1, a2):
    d1 = rescale_targets(a1)
    d2 = rescale_targets(a2)
    t = []
    l = len(d2)
    for k, v in d2.iteritems():
        if k in d1:
            t.append(d2[k] / (1 + (d2[k] - d1[k]) ** 2))
    if l:
        w = sum([ d2[k] for k in d2.keys() ])
        if w > 0:
            return sum(t) / w
        else:
            return 0
    else:
        return 0


def remove_max_targets(t):
    """Remove the greatest target"""
    g0 = max(t, key=lambda k: t[k])
    print "removing = {}".format(g0)
    return { k:t[k] for k in t.keys() if k != g0 }



def rescale_targets(t):
    g1 = max(t, key=lambda k: t[k])
    t1 = { k: t[k] / t[g1] for k in t.keys() }
    return t1



def class_names(model, t):
    imagen = ImageCategories(model)
    return ', '.join([imagen.name(c) for c in t])


def format_target(v):
    if type(v) == float:
        v = "%.5f" % v
    else:
        v = str(v)
    return v


def target_names(t):
    imagen = ImageCategories(model)
    if not type(t) == dict:
        td = { int(i): 1 for i in t }
    else:
        td = { int(i): v for (i, v) in t.iteritems() }
    nd = { imagen.name(i): v for (i, v) in td.iteritems() }
    nld = sorted(nd.items(), key=lambda x: -x[1])
    return nld

def show_targets(t):
    st = target_names(t)
    return ' '.join([ '{0}: {1:.3f}'.format(i[0], i[1]) for i in st])



def log_line(image, model, t):
    imagen = ImageCategories(model)
    nld = target_names(t)
    row = [ str(e) for l in nld for e in l ]
    row = [ image, model ] + row
    return ','.join(row) + "\n"
#         [int(i) for i in t.keys()]


def make_background(cf, bgfile):
    bg_script = cf['bg_script']
    args = [ bg_script ] + cf['bg_params'] + [ bgfile ]
#    args = [ bg_script, bgfile, str(cf['size']) ] + bg_params
    print args
    subprocess.check_output(args)
    #sys.exit(0)

def deepdraw(conffile, infile, outdir, outfile):
    dd = [ '../../deepdream/dream.py', '--config', conffile, '--basefile', outfile, infile, outdir ]
    print "deepdraw %s" % dd
    subprocess.check_output(dd, stderr=subprocess.STDOUT)
    return os.path.join(outdir, outfile) + '.jpg'

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--number", type=int, default=42, help="Number of frames to generate")
parser.add_argument("-i", "--initial", type=str, default=None, help="Initial image - if not supplied, will start with a random image")
parser.add_argument("-f", "--first", type=str, default=None, help="First set of classes")
parser.add_argument("-e", "--ever", action='store_true', help="Ignore classification, do first classes forever")
parser.add_argument("-c", "--config", type=str, default=None, help="Global config file")
parser.add_argument("-s", "--static", action='store_true', help="Same base image for all frames", default=False)
parser.add_argument("outdir",       type=str, help="Output directory")
parser.add_argument("outfile",      type=str, help="File to log image classes")

args = parser.parse_args()

outfile = os.path.join(args.outdir, args.outfile)

if args.config:
    with open(args.config, 'r') as f:
        cf = json.load(f)
    if 'remote' in cf and cf['remote']:
        remote_cf = cf['remote']
    else:
        remote_cf = None
else:
    cf = DEFAULTS
    remote_cf = None

# imagen = ImageCategories(cf['model'])

cfdump = os.path.join(args.outdir, 'global.conf')
with open(cfdump, 'w') as cfd:
    json.dump(cf, cfd, indent=4)


multi = False
if ',' in cf['model']:
    multi = True

mi = model_iter(cf['model'])

start_targets = None

if args.initial:
    model = mi.next()
    base = args.initial
    lastimage = ( IMG_TEMP + ".jpg" ) % base
    initconf = ( CONF_TEMP + ".json" ) % base
    start_targets = neuralgae.read_config(initconf)
else:
    model = mi.next()
    if args.first:
        start_targets = [ int(t) for t in args.first.split(',') ]
        out = [ t for t in start_targets if t >= NTARGETS[model] ]
        if out:
            print "Classes out of range"
            sys.exit(-1)
    else:
        start_targets = random.sample(range(0, NTARGETS[model]), cf['nstart'])
    conffile = os.path.join(args.outdir, conffilename(0))
    print "Config file: %s " % conffile
    cf['target'] = ','.join([ str(x) for x in start_targets ])
    frame_cf = cf # interpolate_cf(cf, 0)
    frame_cf['model'] = model
    neuralgae.write_config(cf, conffile)
    make_background(cf, 'bg.jpg')
    lastimage = deepdraw(conffile, 'bg.jpg', args.outdir, imgfilename(0))
    #sys.exit(-1)

with open(outfile, 'w') as f:
    f.write(log_line(lastimage, model, start_targets))

print "lastimage = %s" % lastimage

print "generating %d frames" % args.number

ltargets = None

# Note: instead of resetting the targets, try removing the
# largest one and renormalising the remainder

for i in range(1, args.number):
    model = mi.next()
    jsonfile = os.path.join(args.outdir, conffilename(i))
    if not args.ever:
        cf['target'] = do_classify(cf, remote_cf, model, lastimage)
        if 'flatten' in cf:
            cf['target'] = flatten_targets(cf, ltargets)
        elif not multi and (ltargets and 'reset' in cf):
            cf['target'] = perturb_targets(cf, ltargets)
        print("final = {}".format(show_targets(cf['target'])))
        ltargets = copy.deepcopy(cf['target'])
    frame_cf = cf # interpolate_cf(cf, 1.0 * i / (args.number + 1))
    frame_cf['model'] = model
    neuralgae.write_config(frame_cf, jsonfile)
    if not args.static:
        make_background(frame_cf, 'bg.jpg')
    lastimage = deepdraw(jsonfile, 'bg.jpg', args.outdir, imgfilename(i))
    targets = neuralgae.read_config(jsonfile)
    with open(outfile, 'a') as f:
        f.write(log_line(lastimage, model, targets))
