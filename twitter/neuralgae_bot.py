#!/usr/bin/env python3

from twython import Twython
import yaml, sys, os.path, re

CF = 'neuralgae.yml'

DIR = '/Users/mike/Desktop/DeepDream/neuralgae/src/Output/Twitter/Seq3'

IMAGE = 'image10.jpg'
TEXT = 'rock beauty, rock python, hognose snake, badger, night snake, king crab, tailed frog, crayfish'


def post_neuralga(cf, img, text):
    """Post a Neuralga image and text to the twitter account.

Args:
    cf (dict): config file
    img (str): image file
    text (str): text part of tweet

Returns:
    id (str): the id of the new tweet
"""
    
    app_key = cf['app_key']
    app_secret = cf['app_secret']
    oauth_token = cf['oauth_token']
    oauth_token_secret = cf['oauth_token_secret']
    twitter = Twython(app_key, app_secret, oauth_token, oauth_token_secret)
    imgfile = os.path.join(DIR, IMAGE)
    with open(imgfile, 'rb') as ih:
        response = twitter.upload_media(media=ih)
        print(response['media_id'])
        out = twitter.update_status(status=TEXT, media_ids=response['media_id'])
        print(out)
        print("Done")

def read_index(cf):
    """Read the index file and return a dict by image filename

Args:
    cf (dict): the config

Returns:
    index (dict): dict by image filename, values are the class target lists
"""
    indexfile = os.path.join(cf['images'], cf['index'])
    index_re = re.compile(cf['index_re'])
    index = {}
    with open(indexfile, 'r') as f:
        for l in f:
            m = index_re.search(l)
            if m:
                index[m.group(1)] = m.group(2)
    return index
        
def get_next_post(cf):
    lfile = cf['lastfile']
    index = cf['index']

        
def load_config(conffile):
    config = None
    with open(conffile) as cf:
        try:
            config = yaml.load(cf)
        except yaml.YAMLError as exc:
            print("%s parse error: %s" % ( conffile, exc ))
            if hasattr(exc, 'problem_mark'):
                mark = exc.problem_mark
                print("Error position: (%s:%s)" % (mark.line + 1, mark.column + 1))
    if not config:
        print("Config error")
        sys.exit(-1)
    return config


def config_has(cf, keys):
    ok = True
    for k in keys:
        if not k in cf:
            print("Missing key %s" % k)
            ok = False
    return ok

if __name__ == '__main__':
    cf = load_config(CF)
    id = read_index(cf)
    print(id)
#    if config_has(cf, [ 'app_key', 'app_secret', 'oauth_token', 'oauth_token_secret']):
#        post_neuralga(cf, IMAGE, TEXT)
#    else:
#        print("Missing config - halted")
