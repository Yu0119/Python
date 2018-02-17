# -*- coding: utf-8 -*-
"""
 Brief: Resnet50+PCAによるOneClassSVM
 Ref: https://hackernoon.com/one-class-classification-for-images-with-deep-features-be890c43455d
"""
from __future__ import print_function
from keras.applications.resnet50 import ResNet50
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn import svm
from sklearn.mixture import GaussianMixture
from sklearn.isotonic import IsotonicRegression
import numpy as np
import argparse
import six
import os

from keras.preprocessing.image import load_img, img_to_array

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--traindir','-tr', type=str, default='data/train', help='Path to train dataset directory')
parser.add_argument('--testdir','-te', type=str, default='data/test', help='Path to test dataset directory')
args = parser.parse_args()

# Image size
H,W = 224,224


def extract_resnet_features(x):
  """"[summary]
    Extract layer features from resnet50 model
  Arguments:
    x {[ndarray]} -- [input image data]
  
  Returns:
    [ndarray] -- [feature layer]
  """

  resnet_model = ResNet50(input_shape=(H, W, 3),
                          weights='imagenet',
                          include_top=False)
  features_ary = resnet_model.predict(x)
  
  return features_ary


train_paths = [ os.path.join(args.traindir, imgfile) for imgfile \
                in os.listdir(args.traindir) if imgfile.endswith('.jpg') ]
test_paths = [ os.path.join(args.testdir, imgfile) for imgfile \
                in os.listdir(args.testdir) if imgfile.endswith('.jpg') ]
# Extract feature arrays

print('num train images: {}'.format(len(train_paths)))
print('num test images: {}'.format(len(test_paths)))

for i, train_path in enumerate(train_paths):
  image = load_img(train_path, target_size=(H, W))
  imgarray = img_to_array(image)
  imgarray = np.reshape(imgarray, (-1, H, W, 3))
  feature = extract_resnet_features(imgarray)
  if i == 0:
    train_features = feature
  else:
    train_features = np.append(train_features, feature)
  print('finished load train {}'.format(os.path.basename(train_path)))

train_features = train_features.reshape(-1, 2048)
print(train_features.shape)

# train_features = extract_resnet_features(train_imgarys)

for j, test_path in enumerate(test_paths):
  image = load_img(test_path, target_size=(224, 224))
  imgarray = img_to_array(image)
  imgarray = np.reshape(imgarray, (-1, H, W, 3))
  feature = extract_resnet_features(imgarray)
  if j==0: 
    test_features = feature
  else:
    test_features = np.append(test_features, feature)
  print('finished load test {}'.format(os.path.basename(test_path)))

test_features = test_features.reshape(-1, 2048)
# for i in six.moves.range(len()):
  
# x_train is train data of feature arrays
# x_test is test data of feature arrays

st_scaler = StandardScaler()
st_scaler.fit(train_features)
X_train = st_scaler.transform(train_features)
X_test = st_scaler.transform(test_features)

pca = PCA(n_components=512, whiten=True)
pca = pca.fit(X_train)

print('Explained variance percentage = \
  {:3.2f}'.format(sum(pca.explained_variance_ratio_)))

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

oneclass_svm_classifier = svm.OneClassSVM(gamma=0.001,
                                          kernel='rbf',
                                          nu=0.08)

isolationforest_classifier = \
                IsolationForest(contamination=0.08,
                                max_features=1.0,
                                n_estimators=40)
# 学習データでフィッティング
oneclass_svm_classifier.fit(X_train)
isolationforest_classifier.fit(X_train)

# One Class SVMによる予測
oneclass_svm_predict = oneclass_svm_classifier.predict(X_test)
# 孤立した分布を探索
isolation_predict =  isolationforest_classifier.predict(X_test)

gausian_mixture_classifier = \
  GaussianMixture(covariance_type='spherical',
                  n_components=5,
                  max_iter=(1e7))

gausian_mixture_classifier.fit(X_train)

log_probs_value = \
  gausian_mixture_classifier.score_samples(X_val)

isotonic_regressor = IsotonicRegression(out_of_bounds='clip')

# テストデータは0:T, 1:Fとする
y_value = []
isotonic_regressor.fit(log_probs_value, y_value)

log_probs_test = gausian_mixture_classifier.score_samples(X_test)
test_probs = isotonic_regressor.predict(log_probs_value)

test_preds = [ 1 if prob >= 0.5 else 0 for prob in test_probs ]
