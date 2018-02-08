# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os

# Parse arguments
parser = argparse.ArgumentParser(description="Search target extention files.")
parser.add_argument('rootpath', help="Input search root path.")
parser.add_argument('--target_ext','-t', default='txt,jpg', help="Input target extensions with comma. [ex. txt,jpg,..]")
parser.add_argument('--outfile','-o', default=None, help="Input path to outfile(If input 'None', output to console)")
args = parser.parse_args()

# Search target files
def search_file():
  exts = args.target_ext
  tgt_exts = ['.{}'.format(ext) for ext in exts.split(',')]

  for root, dirs, files in os.walk(args.rootpath):

    for f in files:
      _, ext = os.path.splitext(f)

      if ext not in exts:
        absroot = os.path.abspath(root)
        
        if args.outfile is not None:
          with open(args.outfile, 'wt') as search_file:
            search_file.write(os.path.join(absroot, f))

        else:
          print(os.path.join(absroot, f))


if __name__ == '__main__':
  search_file()
        