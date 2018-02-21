"""
[Load image utility for keras cnn datasets (simple CNN model)]
    train_x: train dataset
    train_y: train labels
    test_x:  test dataset
    test_y:  test labels
"""
from __future__ import print_function
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical
import numpy as np
from tqdm import trange
import argparse
import os

parser = argparse.ArgumentParser(description="Load train/test dataset for keras")
parser.add_argument('rootpath', help="Path to train dataset directory")
parser.add_argument('--num_train_rate','-n', type=float, \
                    default=0.8, help="Set how many images use for train data")
args = parser.parse_args()

H,W = 224, 224 # Image height & width
NCLASS = 10


def load_datasets(rootpath):
  """[Load train & validate dataset]
  
  Arguments:
    rootpath {[str]} -- [Root path to image datasets]
  Return:
    train_x, train_y, test_x, test_y
  """
  #### get directory names
  # each directory name has a number for classification: ex. 1_banana 2_apple 
  directories = os.listdir(rootpath)
  
  # Get all image paths
  for dirid in trange(len(directories), disc='directory'):
    # Class number
    class_no = int(directories[dirid].split('_')[0])
    # Get image files
    imagefiles = os.listdir('{}/{}'.format(rootpath, directories[dirid]))
    # Number of train images
    num_train = int(args.num_train_rate * len(imagefiles))

    # Divide to train/valid images
    train_images = imagefiles[:num_train]
    valid_images = imagefiles[num_train+1:]

    ### Load train images & labels
    for imgid in trange(len(train_images), disc='load train images'):
      directory = directories[dirid]
      imgfile = train_images[imgid]
      
      imgpath = '{}/{}/{}'.format(rootpath, directory, imgfile)
      
      image = load_img(imgpath, target_size=(H,W))
      # Convert to image array
      imgary = img_to_array(image)
      # Get label with one-hot vector format
      one_hot_label = to_categorical(class_no, NCLASS)

      # add to ndarray
      if imgid==0:
        train_x = imgary
        train_y = one_hot_label
      else:
        train_x = np.append(train_x, imgary)
        train_y = np.append(train_y, one_hot_label)

    ### Load valid images & labels
    for imgid in trange(len(valid_images), desc='load valid images'):
      directory = directories[dirid]
      imgfile = valid_images[imgid]
      imgpath = '{}/{}/{}'.format(rootpath, directory, imgfile)
      image = load_img(imgpath, target_size=(H,W))
      imgary = imagefiles(image)
      one_hot_label = to_categorical(class_no, NCLASS)

      if imgid==0:
        test_x = imgary
        test_y = one_hot_label
      else:
        test_x = np.append(test_x, imgary)
        test_y = np.append(test_y, one_hot_label)
      
  # Reshape arrays
  train_x = train_x.reshape(-1, H, W, 3)
  test_x  = test_x.reshape(-1, H, W, 3)
  train_y = train_y.reshape(-1, NCLASS)
  test_y  = test_y.reshape(-1, NCLASS)

  # Return datasets
  return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    
  train_imgs, train_labels, test_imgs, test_labels = load_datasets(args.rootpath)

  print('trainx datas', train_imgs.shape)
  print('trainy datas', train_labels.shape)
  print('testx datas', test_imgs.shape)
  print('testy datas', test_labels.shape)

