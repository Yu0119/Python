# -*- coding: utf-8 -*-
"""[Finetune Keras-Resnet50 model]

 Usage:
  python -tr <train_dir> -m <output_modelname> -e <num_epochs>
"""

from keras.applications.resnet50 import ResNet50 
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
# from keras.datasets import mnist
from keras.optimizers import SGD
import argparse
import os
from tqdm import trange
import pickle

# Argument parser
parser = argparse.ArgumentParser(description="Finetune Resnet50 model")
parser.add_argument('--train_dir','-tr', help="Path to train directory")
parser.add_argument('--output_model','-m', default=None, help="Output modelfile name")
parser.add_argument('--epoch','-e', type=int, default=30, help="Numbers of epochs")
args = parser.parse_args()

# vars settings
n_class = 10
n_epochs = args.epoch
imgw, imgh = 224,224 # 28, 28 # width/height of the image
num_train_rates = 0.8 # separate rate Train:Test

# Set output modelname
output_modelname = 'resnet50-finetune-modelweights'

# output_modelname = args.output_model

# Load mnist datasets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()


# Count files in target directory
def count_files(target_path):
    nfiles = 0
    for _, _, files in os.walk(target_path):
        nfiles += len(files)
    return nfiles


# Load 10 category image datasets
def load_10category_images(rootdir):

    directories = os.listdir(rootdir)
    for dir_index in trange(len(directories), desc='load image directory'):
        # Each directory has a number ex. 1_apple, 2_orange, ...
        directory = directories[dir_index]
        cls_number = int(directory.split('_')[0])
        imgfiles = os.listdir('{}/{}'.format(rootdir, directory))
        seperate_id = int(num_train_rates * len(imgfiles))

        # Separate Train/Test datasets 
        train_files = imgfiles[:seperate_id]
        test_files  = imgfiles[seperate_id+1:]

        one_hot_label = to_categorical(cls_number, n_class)

        # Train datasets
        #------------------------------------------------------------------
        for trid, train_file in enumerate(train_files):
            imgpath = '{}/{}/{}'.format(rootdir, directories[dir_index], train_file)
            train_image = load_img(imgpath, target_size=(imgw, imgh))
            train_imgary = img_to_array(train_image)

            if trid==0:
                train_images = train_imgary
                train_labels = one_hot_label
            else:
                train_images = np.append(train_images, train_imgary)
                train_labels = np.append(train_labels, one_hot_label)

        # Reconstruct
        train_images = train_images.reshape(-1, imgw, imgh, 3)
        train_labels = train_labels.reshape(-1, n_class)


        # Test datasets
        #------------------------------------------------------------------
        for teid, test_file in enumerate(test_files):
            imgpath = '{}/{}/{}'.format(rootdir, directory, test_file)
            test_image = load_img(imgpath, target_size=(imgw,imgh))
            test_imgary = img_to_array(test_image)

            if teid==0:
                test_images = test_imgary
                test_labels = one_hot_label
            else:
                test_images = np.append(test_images, test_imgary)
                test_labels = np.append(test_labels, one_hot_label)

        # Combine all datasets
        if dir_index==0:
            train_x = train_images
            train_y = train_labels
            test_x  = test_images
            test_y  = test_labels
        else:
            train_x = np.append(train_x, train_images)
            train_y = np.append(train_y, train_labels)
            test_x  = np.append(test_x, test_images)
            test_y  = np.append(test_y, test_labels)

    # Reconstruct datasets array
    train_x = train_x.reshape(-1, imgw, imgh, 3)
    train_y = train_y.reshape(-1, n_class)
    test_x  = test_x.reshape(-1, imgw, imgh, 3)
    test_y  = test_y.reshape(-1, n_class)

    # Check arrays
    print('train_image: ', train_x.shape )
    print('train_label: ', train_y.shape )
    print('test_image: ' , test_x.shape )
    print('test_labels: ', test_y.shape )

    # Return datasets
    return train_x, train_y, test_x, test_y


# Create & calculate
def model(x_train, y_train, base_model):
    x = base_model.output
    # Pooling layer
    x = GlobalAveragePooling2D()(x)
    # fully-connected(512)
    x = Dense(512, activation='relu')(x)
    # Output layer
    pred = Dense(n_class, activation='softmax')(x)
    print(pred)
    # Construct model
    model = Model(inputs=base_model.input, outputs=pred)

    # Freeze pretrained-layers
    for layer in base_model.layers:
        layer.trainable = False

    # Update weights
    # model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.compile(optimizer='sgd', loss='categorical_crossentropy')

    # Trainig start
    model.fit(x_train, y_train)

    # choose the layers which are updated by training
    layer_num = len(model.layers)
    for layer in model.layers[:int(layer_num * 0.9)]:
        layer.trainable = False

    for layer in model.layers[int(layer_num * 0.9):]:
        layer.trainable = True

    # Update weights
    optimizer = SGD(lr=0.0001, momentum=0.9)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    # Get results
    history = model.fit(x_train, y_train, epochs=args.epoch)

    return history


if __name__ == '__main__':
    
    TRAINPATH = 'data/mnist/train'
    TESTPATH = 'data/mnist/validate'
    dump_mnist_pickles(TRAINPATH, TESTPATH)
    x_train, y_train, x_test, y_test = combine_pickles()
    # DIR_TO_10CATEG_DATASETS = 'data/10CategoryImages'
    # x_train, y_train, x_test, y_test = load_10category_images(DIR_TO_10CATEG_DATASETS)

    # Create finetune resnet50 model
    res50_model = ResNet50(weights='imagenet', include_top=False, input_shape=(imgw,imgh,3))

    # Resnet50 history
    history_res50 = model(x_train, y_train, res50_model)
    print('resnet 50 finetune history: {}'.format(history_res50))

    # Evaluation
    evaluation_res_50 = history_res50.model.evaluate(x_test, y_test)
    print("resnet 50 loss: {}".format(evaluation_res_50))

    # Save Model weights
    if output_modelname is not None:
        res50_model.save_weights('{}.h5'.format(output_modelname))
