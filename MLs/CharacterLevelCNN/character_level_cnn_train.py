# -*- coding: utf-8 -*-
"""
 Character level analysis of CNN training script.
 Implementation: https://qiita.com/bokeneko/items/c0f0ce60a998304400c8
"""
from keras.models import Input, Model
from keras.layers import Convolution2D, MaxPooling2D, Dense, \
                        Dropout, BatchNormalization, Embedding, Reshape, merge
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
import numpy as np
import os, sys, random, argparse
from tqdm import tqdm


def create_model(embed_size=128, max_length=300, filter_sizes=(2,3,4,5), filter_num=64):
    inp = Input(shape=(max_length,))
    emb = Embedding(0xffff, embed_size)(inp)
    emb_ex = Reshape((max_length, embed_size, 1))(emb)
    convs = []

    for filter_size in filter_sizes:
        conv = Convolution2D(filter_num, filter_size, embed_size, activation='relu')(emb_ex)
        pool = MaxPooling2D(pool_size=(max_length - filter_size+1, 1))(conv)
        convs.append(pool)

    convs_merged = merge(convs, mode='concat')
    reshape = Reshape((filter_num * len(filter_sizes), ))(convs_merged)
    fc1 = Dense(64, activation='relu')(reshape)
    bn1 = BatchNormalization()(fc1)
    do1 = Dropout(0.5)(bn1)
    fc2 = Dense(1, activation='sigmoid')(do1)
    
    model = Model(inputs=inp, outputs=fc2)

    return model

# Spam datasetを読み込む
def load_spam_data(filepath, targets, max_length=300, min_length=10):
    texts = []
    tmp_texts = []

    with open(filepath, 'r', encoding='utf-8') as f:

        for line in f:
        
            # 読み込めるSpamデータのみが対象
            try:
                print(line)
                attrib, text = line.split('\t', 1)
            except:
                # utf-8非対応は除外
                continue

            # 文字単位でWordIDに変換
            text = [ ord(x) for x in text.strip() ]
            # print(text)
            # 長い部分は打ち切り
            text = text[:max_length]
            text_len = len(text)

            if text_len < min_length:
                continue

            if text_len < max_length:
                # 0 padding
                text += ([0] * (max_length-text_len))

            # target:1, others:0
            if attrib not in targets:
                tmp_texts.append((0, text))
            else:
                texts.append((1, text))

        random.shuffle(tmp_texts)
        texts.extend(tmp_texts[:len(texts)])
        random.shuffle(texts)

        return texts


def train(inputs, targets, batch_size=100, epoch_count=100, max_length=300, model_filepath='model.h5', lr=0.001):

    # 学習率を少しずつ下げるようにする
    start = lr
    stop = lr * 0.01
    learning_rates = np.linspace(start, stop, epoch_count)

    # モデルの作成
    model = create_model(max_length=max_length)
    optimizer = Adam(lr=lr)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # 学習
    model.fit(inputs, targets, nb_epoch=epoch_count, 
            batch_size=batch_size, verbose=1, validation_split=0.1, 
            shuffle=True, callbacks=[
                LearningRateScheduler(lambda epoch: learning_rates[epoch]),
            ])
    
    # モデルの保存
    model.save(model_filepath)


if __name__ == '__main__':
    # Parser
    parser = argparse.ArgumentParser(description="CNN character-level prediction argorithm.")
    parser.add_argument('--tgtpath','-t', default='data/smsspamcollection/SMSSpamCollection_train')
    parser.add_argument('--maxlength','-m', type=int, default=300)
    parser.add_argument('--batch_size','-b', type=int, default=32)
    parser.add_argument('--epoch_count','-e', type=int, default=30)
    args = parser.parse_args()
    
    print('loading datasets ...')
    Load spam dataset
    comments = load_spam_data(args.tgtpath, 'spam')
    # print(comments)

    input_values = []
    target_values = []

    for target_value, input_value in comments:
        input_values.append(input_value)
        target_values.append(target_value)
    
    input_values = np.array(input_values)
    target_values = np.array(target_values)
    
    # train start
    train( input_values
        , target_values
        , batch_size=args.batch_size
        , max_length=args.maxlength
        , epoch_count=args.epoch_count )
