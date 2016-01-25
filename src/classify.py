#!/usr/bin/env python

# code to classify an input image and return the top n matching targets

from cStringIO import StringIO

import numpy as np
import scipy.ndimage as nd
import PIL.Image
from google.protobuf import text_format

import argparse
import json
import re
import sys
import os, os.path, shutil, subprocess
import argparse
import matplotlib.pyplot as plt

#from neuralgae import ImageCategories
import neuralgae

import caffe
import random

DEFAULT_MODEL = 'caffenet'

CLASSES = './classes.txt'

if 'CAFFE_PATH' in os.environ:
    CAFFE_ROOT = os.path.join(os.environ['CAFFE_PATH'])
else:
    print """
You need to set the environment variable CAFFE_PATH to the location of your
Caffe installation
"""
    sys.exit(-1)

MEAN = os.path.join(CAFFE_ROOT, 'python/caffe/imagenet/ilsvrc_2012_mean.npy')

CAFFE_MODELS = os.path.join(CAFFE_ROOT, 'models')

IMAGE = os.path.join(CAFFE_ROOT, 'examples/images/fish-bike.jpg')

MODELS = {
    'googlenet': 'bvlc_googlenet',
    'places': 'googlenet_places205',
    'oxford': 'oxford102',
    'cnn_age': 'cnn_age',
    'cnn_gender': 'cnn_gender',
    'caffenet': 'bvlc_reference_caffenet',
    'ilsvrc13': 'bvlc_reference_rcnn_ilsvrc13',
    'flickr_style': 'finetune_flickr_style'
#    'cars' : 'cars'
}

LAYERS = {
    'oxford': 'fc8_oxford_102'
}

MODEL = DEFAULT_MODEL
model_name = MODELS[MODEL]


def classify(model_label, image, n):
    """Classifies an image file and returns the top n matching classes"""
    caffe.set_mode_cpu()
    model_name = MODELS[model_label]
    model_d = os.path.join(CAFFE_MODELS, model_name)
    proto = os.path.join(model_d, 'deploy.prototxt')
    model = os.path.join(model_d, '%s.caffemodel' % model_name)

    net = caffe.Net(proto, model, caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load(MEAN).mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(image))
    out = net.forward()
    targets = []
    nclasses = int(n) + 1
    layer = 'prob'
    if model_label in LAYERS:
        layer = LAYERS[model_label] 
    top_k = net.blobs[layer].data[0].flatten().argsort()
    #print net.blobs[layer].data[0]
    for i in top_k[-1:-nclasses:-1]:
        targets.append(i)
    return targets

def classify_remote(model, image, n, cf):
    """Runs the classify on a remote server"""
    script = os.path.join(cf['basedir'], 'classify.py')
    filename = os.path.basename(image)
    shutil.copyfile(image, os.path.join(cf['localdir'], filename))
    imagecp = os.path.join(cf['remotedir'], filename)
    command = ' '.join([ "source /etc/profile;", script, model, imagecp, str(n)])
    out = subprocess.check_output(["ssh", "-p", cf['port'], cf['host'], command])
    results = [int(s) for s in out.split(',')]
    return results

        

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('model', type=str, help="Model name")
    ap.add_argument('image', type=str, help="Image to classify")
    ap.add_argument('classes', type=int, help="Number of classes to return")
    args = ap.parse_args()
    model = args.model
    if model not in MODELS:
        print "Unknown moded %s" % model
        sys.exit(-1)
    image = args.image
    classes = args.classes
    classes = classify(model, image, classes)
    print ','.join([ str(c) for c in classes])

            

