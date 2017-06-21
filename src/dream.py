#!/usr/bin/env python2
# imports and basic notebook setup
from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from google.protobuf import text_format

import argparse
import json
import re
import sys
import os, os.path

import caffe

if 'CAFFE_PATH' in os.environ:
    CAFFE_MODELS = os.path.join(os.environ['CAFFE_PATH'], 'models')
else:
    print """
You need to set the environment variable CAFFE_PATH to the location of your
Caffe installation
"""
    sys.exit(-1)


MODELS = {
    'googlenet': 'bvlc_googlenet',
    'places': 'googlenet_places205',
    'oxford': 'oxford102',
    'cnn_age': 'cnn_age',
    'cnn_gender': 'cnn_gender',
    'caffenet': 'bvlc_reference_caffenet',
    'ilsvrc13': 'bvlc_reference_rcnn_ilsvrc13',
    'flickr_style': 'finetune_flickr_style',
    'manga' : 'illustration2vec',
    'manga_tag' : 'illustration2vec_tag'
#    'cars' : 'cars'
}

DEFAULT_LAYERS = {
    'googlenet': 'inception_4c/output',
    'places': 'inception_4c/output',
    'cnn_age': 'pool5',
    'cnn_gender': 'pool5',
    'oxford': 'pool5',
    'cars': 'pool5',
    'caffenet': 'pool5',
    'ilsvrc13': 'pool5',
    'flickr_style': 'pool5',
    'manga': 'pool5',
    'manga_tag': 'pool6'
}

CLASS_TARGET_LAYER = {
    'googlenet': 'loss3/classifier',
    'places': 'loss3/classifier',
    'oxford': 'fc8_oxford_102',
    'flickr_style': 'fc8_flickr',
    'cars': 'fc8',
    'cnn_age': 'fc8',
    'cnn_gender': 'fc8',
    'ilsvrc13': 'fc-rcnn',
    'caffenet': 'fc8',
    'manga': 'encode1neuron',
    'manga_tag': 'conv6_4'
}

N_CLASSES = {
    'googlenet': 1000,
    'caffenet': 1000,
    'manga_tag': 1539,
    'places': 205
}

MAGIC_TARGETS = [ 'randomise' ]
        
CLASS_BACKGROUND = 128.0

MD_FILE = 'dream.json'


MEAN_BINARIES = {
    #    'cnn_age': 'cnn_age_gender/mean.binaryproto',
    'manga_tag': 'illustration2vec_tag/image_mean.npy',
    'manga': 'illustration2vec_tag/image_mean.npy'
}


# inception layers (for both of the above)
# 3a, 3b, 4a, 4b, 4c, 4d, 4e, 5a, 5b

models = MODELS.keys()

def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

def writearray(a, filename, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    PIL.Image.fromarray(a).save(filename, fmt)


def loadmean(filename):
    proto_data = open(filename, "rb").read()
    a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
    mean  = caffe.io.blobproto_to_array(a)[0]
    print "Loaded mean binary %s" % filename
    print mean.shape
    return mean

output_path = 'Output/'
default_layer = None

def load_net(model_name):
    model = MODELS[model_name]
    model_path = os.path.join(CAFFE_MODELS, model)
    net_fn   = os.path.join(model_path, 'deploy.prototxt')
    param_fn = os.path.join(model_path, model + '.caffemodel')
    default_layer = DEFAULT_LAYERS[model_name]

    # load mean binary if it's defined

    if model in MEAN_BINARIES:
        mean = loadmean('../caffe/models/' + MEAN_BINARIES[model])
    else:
        mean = np.float32([104.0, 116.0, 122.0])

    # Patching model to be able to compute gradients.
    # Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(net_fn).read(), model)
    model.force_backward = True
    open('tmp.prototxt', 'w').write(str(model))

    net = caffe.Classifier('tmp.prototxt', param_fn,
                           mean=mean, # ImageNet mean, training set dependent
                           channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB
    return net

# a couple of utility functions for converting to and from Caffe's input image layout

def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']

def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])


# Objective functions

# This is the default objective

def objective_L2(dst):
    dst.diff[:] = dst.data

# objective function based on a guide image

def make_objective_guide(net, guide, guide_layer):
    h, w = guide.shape[:2]
    src, dst = net.blobs['data'], net.blobs[guide_layer]
    src.reshape(1, 3, h, w)
    src.data[0] = preprocess(net, guide)
    net.forward(end=guide_layer)
    guide_features = dst.data[0].copy()
    return lambda d: objective_guide(guide_features, d)

def objective_guide(guide_features, dst):
    x = dst.data[0].copy()       # the data
    y = guide_features           # the guide image
    ch = x.shape[0]              # the shape
    print "objective_guide"
    print "before", x.shape, y.shape
    x = x.reshape(ch,-1)         # reshape these
    y = y.reshape(ch,-1)         # to match one another
    print "after", x.shape, y.shape
    A = x.T.dot(y)               # compute the matrix of dot-products with guide features
    dst.diff[0].reshape(ch,-1)[:] = y[:,A.argmax(1)] # select ones that match best


# Next idea: look at the API for loss layers (like loss3/classifier) and see what
# can be used from them to feedback into dst.diff (ie filter them by one or more
# target categories)


def make_objective_target(net, foci):
    return lambda d: objective_targets(foci, d)



def objective_targets(foci, dst):
    one_hot = np.zeros_like(dst.data)
    for focus, weight in foci.iteritems():
        one_hot.flat[focus] = 1. * weight
    dst.diff[:] = one_hot


def make_magic_targets(type, model):
    n = N_CLASSES[model]
    foci = {}
    for i in range(0, n):
        if np.random.randint(0, 2):
            foci[i] = 1.
        else:
            foci[i] = -1.
    return foci
        
    
def blur(img, sigma):
    if sigma > 0:
        img[0] = nd.filters.gaussian_filter(img[0], sigma, order=0)
        img[1] = nd.filters.gaussian_filter(img[1], sigma, order=0)
        img[2] = nd.filters.gaussian_filter(img[2], sigma, order=0)
    return img



def make_step(net, step_size=1.5, end=default_layer, jitter=32, clip=True, objective=objective_L2, sigma=0):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]    # the layer targeted by default_layer

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift

    net.forward(end=end)     # inference of features

    objective(dst)           # set an objective

    net.backward(start=end)  # retrain

    g = src.diff[0]

    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image

    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)

    if sigma:
        src.data[0] = blur(src.data[0], sigma)




def deepdream(net, base_img, verbose_file=None, iter_n=10, octave_n=4, octave_scale=1.4, tiling=False, end=default_layer, clip=True, **step_params):
    # prepare base images for all octaves

    octaves = [preprocess(net, base_img)]

    w0 = net.blobs['data'].width
    h0 = net.blobs['data'].height

    for i in xrange(octave_n-1):
        o_base = nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1)
        h, w = o_base.shape[-2:]
        if not tiling or (h > h0 and w > w0):
            octaves.append(o_base)

    src = net.blobs['data']
    for o in octaves:
        print o.shape
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    if tiling:
        image = np.zeros_like(base_img)
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)


        if not tiling:
            src.reshape(1, 3, h, w) # resize the network's input image size
            src.data[0] = octave_base + detail
            for i in xrange(iter_n):
                make_step(net, end=end, clip=clip, **step_params)
                vis = deprocess(net, src.data[0])
                if not clip: # adjust image contrast if clipping is disabled
                    vis = vis*(255.0/np.percentile(vis, 99.98))
                print octave, i, end
                if verbose_file:
                    filename = "%s_%d_%i.jpg" % ( verbose_file, octave, i )
                    writearray(vis, filename)
                    print "Wrote %s" % filename

            # extract details produced on the current octave
            detail = src.data[0] - octave_base
            writearray(deprocess(net, detail), "detail_%d.jpg" % octave)
        else:
#            image.reshape(1, 3, h, w)
            image = octave_base + detail
            print "tiling, image = %d %d" % ( h, w )
            tiles = make_tile_pattern(image, w0, h0)
            for i in xrange(iter_n):
                 print "Iter %d" % i
                 for x, y in tiles:
                    tile = get_tile(image, x, y, w0, h0)
                    # print "tile = ", tile.shape
                    if len(tile):
                        src.data[0] = tile
                        make_step(net, end=end, clip=clip, **step_params)
                        print src.data[0].shape
                        put_tile(image, src.data[0], x, y, w0, h0)
            detail = image - octave_base
            writearray(deprocess(net, detail), "detail_%d.jpg" % octave)

    # returning the resulting image
    if not tiling:
        return deprocess(net, src.data[0])
    else:
        return deprocess(net, image)



# TODO:

# parametrise the octaves so that they're unlinked from the above function

# figure out a way to scale up the input to loss3/classifier and/or use
# the other classifiers so that --target works on larger images.

# basic tiles for now

def deepdraw(net, base_img, verbose_file=None, iter_n=10, end=default_layer, clip=True, **step_params):

    # prepare base image

    image = preprocess(net, base_img)

    _, imw, imh = image.shape


    # get input dimensions from net
    w = net.blobs['data'].width
    h = net.blobs['data'].height

    src = net.blobs['data']
    print "Reshaping input image size %d, %d" % ( h, w )

    src.reshape(1,3,h,w) # resize the network's input image size

    tiles = make_tile_pattern(image, w, h)

    for i in xrange(iter_n):
        print "Iter %d" % i
        for x, y in tiles:
            tile = get_tile(image, x, y, w, h)
            if len(tile):
                src.data[0] = tile
                make_step(net, end=end, clip=clip, **step_params)
                put_tile(image, src.data[0], x, y, w, h)
    return deprocess(net, image)

# spiral outwards, overlapping

def make_tile_pattern(image, w, h):
    _, imgw, imgh = image.shape

    if imgw == w and imgh == h:
        return [ ( 0, 0 ) ]

    ox = (imgw - w) / 2
    oy = (imgh - h) / 2

    spiral = [  ]

    r = 1
    done = False

    while not done:
        sq = make_ring(imgw, imgh, w, h, r)
        if sq:
            spiral += sq
            r += 1
        else:
            done = True
    return spiral


def make_ring(imgw, imgh, w, h, r):
    sq = []
    ox = (imgw - r * w) / 2
    oy = (imgh - r * h) / 2
    for x in range(r):
        x1 = ox + x * w
        y1 = oy
        add_if_intersect(sq, imgw, imgh, x1, y1, w, h)
    for y in range(1, r):
        x1 = ox + (r - 1) * w
        y1 = oy + y * h
        add_if_intersect(sq, imgw, imgh, x1, y1, w, h)
    for x in range(r - 2, -1, -1):
        x1 = ox + x * w
        y1 = oy + (r - 1) * h
        add_if_intersect(sq, imgw, imgh, x1, y1, w, h)
    for y in range(r - 2, 0, -1):
        x1 = ox
        y1 = oy + y * h
        add_if_intersect(sq, imgw, imgh, x1, y1, w, h)
    if len(sq):
        print "Ring %d: %s" % ( r, sq )
    return sq


def add_if_intersect(sq, imgw, imgh, x, y, w, h):
    if x < imgw and x > -w and y < imgh and y > -h:
        sq.append((x, y))



def get_tile(image, x, y, w, h):
    _, imgw, imgh = image.shape
    #print "get tile at %d, %d (%d, %d) from %d, %d" % ( x, y, w, h, imgw, imgh )
    x2 = x + w
    y2 = y + h
    t = []
    px = 0
    py = 0
    if x < 0:
        px = x
        x = 0
    elif x2 > imgw:
        px = x2 - imgw
        x2 = imgw
    if y < 0:
        py = y
        y = 0
    elif y2 > imgh:
        py = y2 - imgh
        y2 = imgh
    tile = image[:, x:x2, y:y2]
    if px < 0:
        p = padding(-px, y2 - y, tile)
        tile = np.concatenate((p, tile), 1)
    elif px > 0:
        p = padding(px, y2 - y, tile)
        tile = np.concatenate((tile, p), 1)
    if py < 0:
        p = padding(w, -py, tile)
        tile = np.concatenate((p, tile), 2)
    elif py > 0:
        p = padding(w, py, tile)
        tile = np.concatenate((tile, p), 2)
    return tile



def padding(w, h, tile):
    """Generate padding with the average colour of tile"""
#     padding = np.full((3, w, h), CLASS_BACKGROUND)
#     return padding
    r = np.full((w, h), np.mean(tile[0]))
    g = np.full((w, h), np.mean(tile[1]))
    b = np.full((w, h), np.mean(tile[2]))
    p = np.array([ r, g, b ])
    return p



def put_tile(image, data, x, y, w, h):
    _, imgw, imgh = image.shape
    # x1/y1/x2/y2: in image
    # u1/v1/u2/v2: in data
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    u1 = 0
    v1 = 0
    u2 = w
    v2 = h
    if x < 0:
        u1 = -x
        x1 = 0
    elif x2 > imgw:
        u2 = w - (x2 - imgw)   #fix
        x2 = imgw
    if y < 0:
        v1 = -y
        y1 = 0
    elif y2 > imgh:
        v2 = h - (y2 - imgh)    #fix
        y2 = imgh
    image[:,x1:x2,y1:y2] = data[:,u1:u2,v1:v2]



def parse_classes(s, w):
    if s == 'nil':
        return {}
    try:
        il = []
        for c1 in s.split(','):
            c2 = c1.split('-')
            print len(c2)
            if len(c2) == 1:
                il.append(int(c2[0]))
            else:
                il += (range(int(c2[0]), int(c2[1]) + 1))
#        il = map(int, s.split(','))
        print il
        weight = 1.
        if w:
            weight = w
        c = { f: weight for f in il }
        return c
    except ValueError():
        print "Bad class"
        return {}


def write_json(bfile, args):
    jsonfile = bfile + '.json'
    with open(jsonfile, 'wb') as jf:
        jf.write(json.dumps(args, sort_keys=True, indent=4))

# TODO: add the octaves to the json output and input
# default octaves stored in a config file somewhere (actually all
# defaults)?

# this now takes the command line arg values which are not overriden
# in the config

def read_json(clargs):
    a = argparse.Namespace()
    with open(clargs.config, 'rb') as jf:
        data = json.load(jf)
        for arg, value in data.iteritems():
            a.__setattr__(arg, value)
    for arg, value in vars(clargs).iteritems():
        if not arg in a:
            a.__setattr__(arg, value)
    return a


def convert_focus(strfoci):
    foci = {}
    for target, weight in strfoci.iteritems():
        ti = int(target)
        foci[ti] = weight
    return foci

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input",        type=str, help="The source image")
    parser.add_argument("output",        type=str, help="The output directory")
    parser.add_argument("-c", "--config",  type=str, default=None, help="JSON config file")
    parser.add_argument("-m", "--model", type=str, help="The model", choices=models, default='googlenet')
    parser.add_argument("-l", "--layer", type=str, help="The layer")
    parser.add_argument("-b", "--basefile", type=str, help="Base filename", default=None)
    parser.add_argument("-g", "--guide", type=str, help="The guide image", default=None)
    parser.add_argument("-e", "--guidelayer", type=str, help="The guide layer", default='inception_3b/output')
    parser.add_argument("-t", "--target", type=str, help="ImageNet class(es) (comma separated)", default=None)
    parser.add_argument("-w", "--weight", type=float, help="Weight of ImageNet classes", default=None)
    parser.add_argument("-i", "--iters",  type=int, help="Number of iterations per octave", default=10)
    parser.add_argument("-o", "--octaves", type=int, help="Number of octaves", default=4)
    parser.add_argument("-s", "--sigma", type=float, help="Blur (sigma)", default=0)
    parser.add_argument("-u", "--glide", type=str, help="Glide between frames x,y", default=None)
    parser.add_argument("-v", "--verbose", action='store_true', help="Dump out a file for every iteration", default=False)
    parser.add_argument("-z", "--zoom", type=float, help="Zoom factor", default=0)
    parser.add_argument("-r", "--rotate", type=int, help="Rotate in degrees", default=0)
    parser.add_argument("-f", "--frames", type=int, help="Number of frames", default=1)
    parser.add_argument("-j", "--initial", type=int, help="Initial frame #", default=0)
    parser.add_argument("-k", "--keys", action='store_true', help="Dump a list of available layers", default=False)
    args = parser.parse_args()
     
    origfile = args.input
    output_path = args.output

    if os.path.exists(output_path):
        if os.path.isdir(output_path):
            print "Warning: %s already exists" % output_path
        else:
            print "Output path %s is a file: exiting" % output_path
            sys.exit(-1)
    else:
        os.makedirs(output_path)

    # basefile is never taken from config
    if args.basefile:
        bfile = os.path.join(output_path, args.basefile)
    else:
        f, e = os.path.splitext(os.path.basename(origfile))
        if e != '.jpg':
            print "Input must be a jpg"
            print "Got %s/%s" % ( f, e )
            sys.exit(-1)
        bfile = os.path.join(output_path, f)

    write_conf = True
        
    if args.config:
        if not os.path.isfile(args.config):
            print "Config file %s not found" % args.config
            sys.exit(-1)
        args = read_json(args)
        write_conf = False

    if args.target:
        if args.target in MAGIC_TARGETS and not args.config:
            print "make magic targets"
            foci = make_magic_targets(args.target, args.model)
            args.target = foci
        
    
    if write_conf:
        # don't write out a config file if one was passed in
        write_json(bfile, vars(args))


    vfile = None
    if args.verbose:
        vfile = bfile

    print "Loading %s" % origfile

    img = np.float32(PIL.Image.open(origfile))


    print "Starting neural net..."

    net = load_net(args.model)

    original_w = net.blobs['data'].width
    original_h = net.blobs['data'].height

    print "Data original size: %d %d" % ( original_w, original_h )

    if args.keys:
        print "Layers"
        for k in net.blobs.keys():
            print k
        exit()

    if args.layer:
        layer = args.layer
    else:
        layer = DEFAULT_LAYERS[args.model]

    print "Dreaming..."

    if args.guide:
        guide = np.float32(PIL.Image.open(args.guide))
        guide_layer = args.guidelayer
        obj_guide = make_objective_guide(net, guide, guide_layer)
        dreamer = lambda x: deepdream(net, x, verbose_file=vfile, iter_n=args.iters, octave_n=args.octaves, end=layer, objective=obj_guide)
    elif args.target is not None:
        if args.model not in CLASS_TARGET_LAYER:
            print "Can't do deepdraw on this model"
            sys.exit(-1)
        if type(args.target) is dict:
            foci = convert_focus(args.target)
        else:
            foci = parse_classes(args.target, args.weight)
        if not foci:
            print "Empty targets"
            foci = []
        print foci
        layer = CLASS_TARGET_LAYER[args.model]
        obj_class = make_objective_target(net, foci)
        dreamer = lambda x: deepdraw(net, x, verbose_file=vfile, iter_n=args.iters, end=layer, objective=obj_class, sigma=args.sigma)
        #dreamer = lambda x: deepdream(net, x, verbose_file=vfile, iter_n=args.iters, octave_n=args.octaves, tiling=True, end=layer, objective=obj_class, sigma=args.sigma)
    else:
        dreamer = lambda x: deepdream(net, x, verbose_file=vfile, iter_n=args.iters, octave_n=args.octaves, sigma=args.sigma, end=layer)

    # default value of args.frames is 1

    h, w = img.shape[:2]

    gx, gy = 0, 0
    if args.glide:
        g0 = args.glide.split(',')
        if len(g0) == 2:
            gx = int(g0[0])
            gy = int(g0[1])

    print "Shape" , img.shape
    s = args.zoom
    theta = args.rotate
    fi = args.initial
    for i in xrange(args.frames):
        img = dreamer(img)
        if args.frames > 1:
            filename = "%s_f%d.jpg" % ( bfile, fi )
        else:
            filename = "%s.jpg" % bfile
        writearray(img, filename)
        print "Wrote frame %s" % filename
        if theta != 0:
            print "rotate %d" % theta
            img = nd.rotate(img, theta, reshape=False)
        if s != 0:
            print "zoom %f" % s
            img = nd.affine_transform(img, [1-s,1-s,1], [h*s/2,w*s/2,0], order=1)
        if gx != 0 or gy != 0:
            print "glide %d ,%d" % ( gx, gy )
            img = nd.shift(img, [ gy, gx, 0], mode='nearest')
        fi += 1


    print "Done"
