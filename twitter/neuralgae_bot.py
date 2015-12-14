#!/usr/bin/env python3

from twython import Twython
import yaml, sys

CF = 'neuralgae.yml'


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



if __name__ == '__main__':
    cf = load_config(CF)
    print(cf)
