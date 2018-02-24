# -*- coding:utf-8 -*-
"""
 Predict script for character-level-cnn
 Implementation: https://qiita.com/bokeneko/items/c0f0ce60a998304400c8
 
 Usage:
   python predict.py <words_predict> --model <model.h5>
 Args:
   words_predict : target text strings judge spam or not.
   --model : modelfile trained with spam datasets.
"""
import numpy as np
from keras.models import load_model
import argparse


def predict(comments, model_filepath="model.h5"):
    model = load_model(model_filepath)
    ret = model.predict(comments)
    return ret


def main_spam(tgt_comment):
    comment = [ ord(x) for x in tgt_comment.strip() ]
    comment = comment[:300]
    if len(comment) < 10:
        exit("too short!!")
    if len(comment) < 300:
        comment += ([0] * (300 - len(comment)))
    ret = predict(np.array([comment]))
    predict_result = ret[0][0]
    
    return predict_result



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict model with target words.")
    parser.add_argument('words_predict', help="Input words you wanted to predict.")
    parser.add_argument('--model','-m', default='modelname.h5', help="Input path to model file.")
    args = parser.parse_args()

    # Predict spam level
    target_words = args.words_predict
    predict = main_spam(target_words, args.model)
    # print(target_words)
    print("Spam level: {:3.4f}%".format(predict * 100))
    
    
