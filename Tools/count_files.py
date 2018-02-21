"""
 Count files in directory
"""
from __future__ import print_function
import os


def count_files(rootpath):
  numfiles = 0
  for _, _, files in os.walk(rootpath):
    numfiles += len(files)
  return numfiles


if __name__ == '__main__':
  import sys
  if len(sys.argv) < 1:
    print("Usage: {} <target_path>".format(argv[0]))

  num_files = count_files(sys.argv[1])
  print('number of files in [{}]: {}'.format(sys.argv[1], num_files))
