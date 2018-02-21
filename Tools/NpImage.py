"""
 Image Processes with numpy
"""
import numpy as np
import argparse
from skimage.io import imread, imsave


class NpImage():
  """[Image process modules use Numpy]
  """

  def __init__(self, imgpath):
    self.image = imread(imgpath)
    self.output = self.image
  
  # To grayscale image
  def grayscale(self):
   self.output = self.image.dot([0.299, 0.587, 0.114])
   return self.output

  # Horizontal flip
  def hflip(self):
   self.output = self.image[:, ::-1]
   return self.output.astype('uint8')

  # Vertical flip
  def vflip(self):
    self.output = self.image[::-1, :]
    return self.output.astype('uint8')

  # Save image
  def save(self, filename, imgary=None):
    if imgary is not None:
      imsave(filename, imgary)
    else:
      imsave(filename, self.output.astype('uint8'))
