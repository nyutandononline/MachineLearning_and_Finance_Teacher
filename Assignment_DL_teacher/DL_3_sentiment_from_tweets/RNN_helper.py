import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import re
import h5py
import pickle
import sklearn
import tensorflow as tf
from nose.tools import assert_equal
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model


class rnn_helper():
    def __init__(self):
        self.DATA_DIR = "./Data"

        if not os.path.exists(self.DATA_DIR):
            self.DATA_DIR = "../resource/asnlib/publicdata/tweets/Data"

        self.tweet_file = "Apple-Twitter-Sentiment-DFE-1.csv"

    def getDataRaw(self):
      tweets_raw = pd.read_csv( os.path.join(self.DATA_DIR, self.tweet_file) )
      return tweets_raw

    def getTextClean(self, tweets_raw):
      # Filter out rows with invalid sentiment
      mask = tweets_raw["sentiment"] == 'not_relevant'
      tweets_raw = tweets_raw[ ~ mask ]
      
      docs = tweets_raw["text"].apply(self.cleanTxt).values
      
      # We will treat the sentiment values as Categorical, rather than numeric
      le = sklearn.preprocessing.LabelEncoder(  )
      sents = le.fit_transform( tweets_raw[ "sentiment"] )
      
      return docs, sents

    # Remove mentions, hash tags and someting else not important
    def cleanTxt(self, text):
      text = re.sub('@[A-Za-z0–9]+', '<MENTION>', text) #Removing @mentions
      text = re.sub('#', '', text) # Removing '#' hash tag
      text = re.sub('RT[\s]+', '<RT>', text) # Removing RT
      text = re.sub('https?:\/\/\S+', '<LINK>', text) # Removing hyperlink
      
      return text

    def show(self, tok, encoded_docs_padded):
      '''
      Display data by reversing index back to word
      '''
      # Special characters index directionary
      special = { "<PAD>": 0, 
           }
      
      word_index = tok.word_index
      
      # Add the special characters to the word to index map
      for word, idx in special.items():
          word_index[word] = idx

      # Reverse map: index to word
      # Index_word: map index to word
      index_word = { idx:w for (w, idx) in word_index.items() }

      for i, rev in enumerate(encoded_docs_padded[0:5]):
          # Map each index in the example back to word
          rev_words = [ index_word[idx] for idx in rev if idx != 0]
          print("{i:d}:\t{r:s}".format(i=i,  r= " ".join(rev_words)) )
          # sent = y_train[i]
          # print("{i:d}:\t({sent:s})\t{fr:s}".format(i=i, sent=sentiment[y_train[i]], r= " ".join(rev_words)) )
        
    def y_OHE(self, y):
        """
        Determine the encoding of y
        - False if it is one dimensional (or two dimensional with final dimension of 1
        - True if it is One Hot Encoded

        Parameters
        -----------
        y: ndarray

        Returns
        -------
        Bool: 
        - True if y is OHE
        - False otherwise
        """
        result = None
        if ( (y.ndim > 1) and (y.shape[-1] >1) ):
          result = True
        else:
          result = False

        return result

    def createOHE(self, max_features):
      mat = np.diag( np.ones(max_features))
      return mat
      
    def getExamples(self, encoded_docs_padded, sents, max_features):
      OHE = self.createOHE(max_features)
      X = np.zeros( (encoded_docs_padded.shape[0], encoded_docs_padded.shape[1], max_features) )

      # Convert each word to a OHE representation
      for doc_num in range(0, encoded_docs_padded.shape[0]):
          encoded_doc = encoded_docs_padded[doc_num]
          ohe_encoded_doc = OHE[ encoded_doc ]

          X[doc_num] = ohe_encoded_doc

      y = sents
      
      return X, y

    def modelPath(self, modelName):
      return os.path.join(".", "models", modelName)

    def saveModel(self, model, modelName): 
      model_path = self.modelPath(modelName)
      
      try:
          os.makedirs(model_path)
      except OSError:
          print("Directory {dir:s} already exists, files will be over-written.".format(dir=model_path))
          
      # Save model JSON to disk
      json_config = model.to_json()
      with open(os.path.join(model_path, 'config.json'), 'w') as json_file:
          json_file.write(json_config)

      # Save weights to disk
      model.save_weights(os.path.join(model_path, 'weights.h5'))

      # Save training config
      metrics = model.metrics_names
      loss    = model.loss
      if 'loss' in metrics:
        metrics.remove('loss')

      training_parms = { "metrics": metrics,
                         "loss"   : loss
                         }
      
      with open(os.path.join(model_path, 'training_parms.pkl'), 'wb') as f:
          pickle.dump(training_parms, f)

      
      print("Model saved in directory {dir:s}; create an archive of this directory and submit with your assignment.".format(dir=model_path))

    def loadModel(self, modelName):
        model_path = self.modelPath(modelName)

        # Reload the model from the files we saved
        with open(os.path.join(model_path, 'config.json')) as json_file:
            json_config = json_file.read()

        model = tf.keras.models.model_from_json(json_config)

        # Retrieve training parameters and restore them
        with open(os.path.join(model_path, 'training_parms.pkl'), 'rb') as f:
            training_parms = pickle.load(f)
            metrics, loss = ( training_parms[k] for k in ("metrics", "loss") )

        model.compile(loss=loss, metrics=metrics)
        model.load_weights(os.path.join(model_path, 'weights.h5'))

        return model

    def saveHistory(self, history, model_name):
      history_path = self.modelPath(model_name)

      try:
          os.makedirs(history_path)
      except OSError:
          print("Directory {dir:s} already exists, files will be over-written.".format(dir=history_path))

      # Save history
      with open(os.path.join(history_path, 'history'), 'wb') as f:
          pickle.dump(history.history, f)

    def loadHistory(self, model_name):
      history_path = self.modelPath(model_name)

      # Reload the model from the 2 files we saved
      with open(os.path.join(history_path, 'history'), 'rb') as f:
          history = pickle.load(f)

      return history
    
    def acc_key(self, history=None, model=None):
        """
        Parameters
        ----------
        model:   A Keras model object
        history: "history" object return by "fit" method applied to a Keras model

        Returns
        -------
        key_name: String.  The key to use in indexing into the dict contained in the history object returned by the "fit" method applied to a Keras model

        You should supply only ONE of these parameters (priority given to "model")

        Newer versions of Keras have changed the name of the metric that measures
        accuracy from "acc" to "accuracy".  Either name is allowed in the "compile" statement.

        The key in the history.history dictionary (returned by applying the "fit" method to the model object) will depend on the exact name of the metric supplied in the "compile" statement.

        This method will return the string to use as a key in history.history by examining
        - The model object (if given)
        - The keys of history.history (if history is given)
        """

        key_name = None

        if model is not None:
          key_name = "accuracy" if "accuracy" in model.metrics_names else "acc"
        else:
          key_name = "accuracy" if "accuracy" in history.history.keys() else "acc"

        return key_name

    def plotModel(self, model, modelName):
        plotFile = modelName + ".png"
        plot_model(model, plotFile, show_shapes=True)

        return plotFile

    def plot_training(self, history, metric="acc"):
        """
        Plot training and validation statistics
        - accuracy vs epoch number
        - loss     vs epoch number

        From https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/
        """  
        # Determine the name of the key that indexes into the accuracy metric
        acc_string = self.acc_key(history=history)
    
        # Accuracy
        acc = history.history[acc_string]
        val_acc = history.history['val_' + acc_string]

        # Loss
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(acc))

        plt.plot(epochs, acc, 'b', label='Training ' + acc_string)
        plt.plot(epochs, val_acc, 'r', label='Validation ' + acc_string)
        plt.title('Training and validation ' + acc_string)
        plt.legend()

        plt.figure()

        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.show()

