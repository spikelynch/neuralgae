#!/usr/bin/env python2


import sys
import subprocess
import copy
import os.path
import shutil
import json
import re
import random, math, itertools
import argparse
import logging
from neuralgae import ImageCategories
import neuralgae
import classify


NTARGETS = {
    'googlenet': 1000,
    'caffenet': 1000,
    'places': 205,
    'flickr_style': 10,
    'oxford': 102,
    'manga_tag': 1536,  #excludes the three safe/questionable/explicit tags
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
    'octaves': None,
    'logfile': 'neuralgae.log'
}

DEFAULT_LOGFILE = 'neuralgae.log'
DEFAULT_LOGFORM = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
CONF_TEMP = 'conf%03d'
IMG_TEMP = 'img%03d'

COLORFILE = '/opt/X11/share/X11/rgb.txt'

colornames = []
with open(COLORFILE) as cf:
    for line in cf:
        if line[0] == '#':
            next
        parts = line[:-1].split()
        if len(parts) == 4:
            colornames.append(parts[3])


logger = None

def conffilename(i):
    return ( CONF_TEMP % i ) + '.json'

def imgfilename(i):
    return ( IMG_TEMP % i )


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
    logger.info("Classify results: {}".format(show_targets(t0)))
    logger.info("Classify results: {}".format(t0))
    # hack to filter targets down to just the first 512 for manga_tag
    targets = { c: t0[c] for c in t0.keys() if ( t0[c] > cutoff and int(c) < NTARGETS[model]) }
    logger.info("Filtered results: {}".format(show_targets(targets)))
    logger.info("Classify results: {}".format(targets))
    if cf['nsample'] < len(targets):
        logger.debug("Sampling {} of {} targets".format(cf['nsample'], len(targets)))
        ts = random.sample(targets.keys(), cf['nsample'])
        ots = targets
        targets = { t:ots[t] for t in ts }
        logger.info("Sampled targets: {}".format(show_targets(targets)))
    return targets


def flatten_targets(cf):
    """Raise all to 1.0"""
    for t in cf['target'].keys():
        cf['target'][t] = 1.0
    return cf['target']



def perturb_targets(cf, ltargets, downscale, seen):
    match = match_targets(ltargets, cf['target'])
    if 'remove_seen' in cf:
        for s in seen.keys():
            if s in cf['targets']:
                logger.debug("Removing seen target: {}".format(target_name(s)))
                del cf['targets'][s]
    ds = downscale
    if match >= cf['reset']:
        logger.debug("Old: {}".format(show_targets(ltargets)))
        logger.debug("New: {}".format(show_targets(cf['target'])))
        logger.debug("Match between targets {} > {}".format(match, cf['reset']))
        if 'add_randoms' in cf:
            logger.debug("Added {} random targets".format(cf['add_randoms']))
            if cf['downscale']:
                ds = ds * cf['downscale']
                # Note- using old targets
                cf['target'] = rescale_targets(ltargets, ds)
                logger.debug("Downscaled originals by {}".format(ds))
            cf['target'] = add_randoms(cf['target'], cf['add_randoms'])
        else:
            logger.debug("Removing max target")
            cf['target'], s = remove_max_targets(cf['target'])
    else:
        ds = 1
    if len(cf['target']):
        if  'flatten' in cf:
            logger.debug("Flattening targets")
            cf['target'] = flatten_targets(cf)
        else:
            logger.debug("Rescaling remaining targets")
            cf['target'] = rescale_targets(cf['target'])
    else:
      logger.debug("No targets left: resetting all targets")
      cf['target'] = { t: 1 for t in random.sample(range(0, NTARGETS[model]), cf['ntween']) }
      #seen = {}
    logger.info("Final targets {}".format(show_targets(cf['target'])))
    return ( cf['target'], match, ds, seen )


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

def add_randoms(targets, nrandoms):
    tn = { str(t).encode('utf-8'):1 for t in range(0, NTARGETS[model]) }
    for t in targets:
        del(tn[t])
    for n in random.sample(tn.keys(), nrandoms):
        targets[n] = 1
    return targets


def remove_max_targets(t):
    """Remove the greatest target"""
    g0 = max(t, key=lambda k: t[k])
    return ( { k:t[k] for k in t.keys() if k != g0 }, g0 )



def rescale_targets(t, scale=1.0):
    g1 = max(t, key=lambda k: t[k])
    t1 = { k: scale * t[k] / t[g1] for k in t.keys() }
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

def target_name(t):
    imagen = ImageCategories(model)
    return imagen.name(t)


def show_targets(t):
    st = target_names(t)
    return ' '.join([ '{0}: {1:.3f}'.format(i[0], i[1]) for i in st])



def log_line(image, model, match, t):
    imagen = ImageCategories(model)
    nld = target_names(t)
    row = [ str(e) for l in nld for e in l ]
    row = [ image, model, "%.4f" % match ] + row
    return ','.join(row) + "\n"

def bg_param_sub(macro):
    m = macro.group(1)
    logger.debug("Substituting param macro {}".format(m))
    if m == 'COLOR':
        return random.choice(colornames)
    else:
        return ""


def bg_param(raw):
    return re.sub(r'\$\{([a-zA-Z]+)\}', bg_param_sub, raw)

def make_background(cf, bgfile):
    bg_script = cf['bg_script']
    params = [ bg_param(p) for p in cf['bg_params'] ]
    args = [ bg_script ] + params + [ bgfile ]
    logger.debug("Background {}".format(args))
#    args = [ bg_script, bgfile, str(cf['size']) ] + bg_params
    subprocess.check_output(args)
    #sys.exit(0)

def deepdraw(conffile, infile, outdir, outfile):
    logger.info("Drawing image {} -> {}".format(infile, outfile))
    dd = [ '../../deepdream/dream.py', '--config', conffile, '--basefile', outfile, infile, outdir ]
    subprocess.check_output(dd, stderr=subprocess.STDOUT)
    return os.path.join(outdir, outfile) + '.jpg'


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--number", type=int, default=42, help="Number of frames to generate")
parser.add_argument("-i", "--initial", type=int, default=None, help="Initial image - if not supplied, will start with a random image")
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


logger = logging.getLogger('neuralgae')

if 'logfile' in cf:
    logfile = cf['logfile']
else:
    logfile = DEFAULT_LOGFILE

logfile = os.path.join(args.outdir, logfile)

if 'logform' in cf:
    logform = logging.Formatter(cf['logform'])
else:
    logform = logging.Formatter(DEFAULT_LOGFORM)



lfh = logging.FileHandler(logfile)
lsh = logging.StreamHandler()

lfh.setFormatter(logform)
lsh.setFormatter(logform)

logger.addHandler(lfh)
logger.addHandler(lsh)
logger.setLevel(logging.DEBUG)


if args.config:
    logger.info("Starting run - global config file {}".format(args.config))
else:
    logger.info("Starting run with default config")

logger.info("Output {} - log file {}".format(args.outdir, args.outfile))
logger.info("Generating %d frames" % args.number)


multi = False
if ',' in cf['model']:
    multi = True
    multi_renderers = cf['deepdraw']

mi = model_iter(cf['model'])

start_targets = None

if args.initial:
    model = mi.next()
    base = args.initial
    lastimage = os.path.join(args.outdir, ( IMG_TEMP + ".jpg" ) % base)
    initconf = os.path.join(args.outdir, ( CONF_TEMP + ".json" ) % base)
    start_targets = neuralgae.read_config(initconf)
    logger.info("Initial from config " + show_targets(start_targets))
else:
    model = mi.next()
    if args.first:
        start_targets = [ int(t) for t in args.first.split(',') ]
        out = [ t for t in start_targets if t >= NTARGETS[model] ]
        if out:
            logger.error("Classes out of range: {}".format(out))
            sys.exit(-1)
    else:
        start_targets = random.sample(range(0, NTARGETS[model]), cf['nstart'])
    logger.info("NTARGETS[{}]: {}".format(model, NTARGETS[model]))
    logger.info("Start targets: " + show_targets(start_targets))
    conffile = os.path.join(args.outdir, conffilename(0))
    cf['target'] = ','.join([ str(x) for x in start_targets ])
    if multi:
        cf['deepdraw'] = multi_renderers[model]
    frame_cf = cf # interpolate_cf(cf, 0)
    frame_cf['model'] = model
    neuralgae.write_config(cf, conffile)
    make_background(cf, 'bg.jpg')
    lastimage = deepdraw(conffile, 'bg.jpg', args.outdir, imgfilename(0))
    #sys.exit(-1)

with open(outfile, 'w') as f:
    f.write(log_line(lastimage, model, 0.0, start_targets))


ltargets = None

seen_targets = {}

if args.ever:
    cf['target'] = cf['target'].split(',')

downscale = 1

if args.initial:
    start = args.initial
else:
    start = 1

for i in range(start, start + args.number):
    model = mi.next()
    jsonfile = os.path.join(args.outdir, conffilename(i))
    match = 0.0
    if not args.ever:
        cf['target'] = do_classify(cf, remote_cf, model, lastimage)
        if not multi and (ltargets and 'reset' in cf):
            cf['target'], match, downscale, seen_targets = perturb_targets(cf, ltargets, downscale, seen_targets)
        ltargets = copy.deepcopy(cf['target'])
    frame_cf = cf # interpolate_cf(cf, 1.0 * i / (args.number + 1))
    frame_cf['model'] = model
    if multi:
        frame_cf['deepdraw'] = multi_renderers[model]
    neuralgae.write_config(frame_cf, jsonfile)
    if not args.static:
        make_background(frame_cf, 'bg.jpg')
    lastimage = deepdraw(jsonfile, 'bg.jpg', args.outdir, imgfilename(i))
    targets = neuralgae.read_config(jsonfile)
    with open(outfile, 'a') as f:
        f.write(log_line(lastimage, model, match, targets))
