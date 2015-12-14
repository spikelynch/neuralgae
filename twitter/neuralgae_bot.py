#!/usr/bin/env python3

from twython import Twython
import yaml, sys, os.path, re, argparse

ARGS = [ 'app_key', 'app_secret', 'oauth_token', 'oauth_token_secret', 'images', 'index', 'index_re', 'lastfile' ]


def post_neuralga(cf, img, text):
    """Post a Neuralga image and text to the twitter account.

Args:
    cf (dict): config file
    img (str): image file
    text (str): text part of tweet

Returns:
    status (bool): True if the post was successful
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
        print("Posted")
        return True
    return False

def read_last(cf):
    """Returns the last file posted, or None"""
    lastfile = os.path.join(cf['images'], cf['lastfile'])
    last = None
    if os.path.isfile(lastfile):
        with open(lastfile, 'r') as lf:
            last = lf.read().strip()
    return last

def write_last(cf, image):
    """Writes an image filename to lastfile"""
    lastfile = os.path.join(cf['images'], cf['lastfile'])
    with open(lastfile, 'w') as lf:
        lf.write(image)
    
    

        
def get_next_post(cf):
    """
Reads the image from lastfile and uses it to search for the next file in
the index.

Args:
    cf (dict): the config

Returns:
    ( image, text ): (str, str): the image file and text
    ( none, none ): if there were no more images

"""
    last = read_last(cf)
    indexfile = os.path.join(cf['images'], cf['index'])
    index_re = re.compile(cf['index_re'])
    index = {}
    with open(indexfile, 'r') as f:
        gn = False
        for l in f:
            m = index_re.search(l)
            if m:
                image = m.group(1)
                if not last or gn:
                    return ( image, m.group(2) )
                if image == last:
                    gn = True
    return ( None, None )
        

        
def load_config(conffile):
    """Reads the YAML config file and returns a dict"""
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
    """Checks for compulsory members of config and returns a bool"""
    ok = True
    for k in keys:
        if not k in cf:
            print("Missing config parameter %s" % k)
            ok = False
    return ok

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, type=str, help="Config file")
    parser.add_argument('-d', '--dry-run', action='store_true', help="Don't post or update the last-post file")
    args = parser.parse_args()
    cf = load_config(args.config)
    if config_has(cf, ARGS):
        img, text = get_next_post(cf)
        print("Next up: %s, %s" % (img, text))
        if args.dry_run:
            print("Dry run")
        else:
            if post_neuralga(cf, img, text):
                write_last(cf, img)
    else:
        print("Can't proceed")
