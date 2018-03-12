#!/usr/bin/env python

import os, os.path, shutil, random, argparse

parser = argparse.ArgumentParser()
parser.add_argument("dir",       type=str, help="Directory")
parser.add_argument("file",      type=str, help="File to copy to")

args = parser.parse_args()

files = []
for f in os.listdir(args.dir):
    if f[-4:] == '.jpg':
        files.append(f)

f = random.choice(files)
print f
shutil.copy(os.path.join(args.dir, f), args.file)
