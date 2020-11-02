import math

import numpy as np
import matplotlib.pyplot as plt

import os
import h5py
import pickle
import tensorflow as tf
from nose.tools import assert_equal
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical


import json

class Helper():
    def __init__(self):
        # Data directory
        self.DATA_DIR = "./Data"

        if not os.path.exists(self.DATA_DIR):
            self.DATA_DIR = "../resource/asnlib/publicdata/ships_in_satellite_images/data"


        self.json_file =  "shipsnet.json"
    def acc_key(history=None, model=None):
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
        
    def scaleData(self, data, labels, one_hot):
        X = data / 255.
        if one_hot:
            y = to_categorical(labels, num_classes=2)
        else:
            y = labels
        return X, y
    
    def getData(self):
        data,labels = self.json_to_numpy( os.path.join(self.DATA_DIR, self.json_file) )
        return data, labels

    def showData(self, data, labels, num_cols=5):
        # Plot the first num_rows * num_cols images in X
        (num_rows, num_cols) = ( math.ceil(data.shape[0]/num_cols), num_cols)

        fig = plt.figure(figsize=(10,10))
        # Plot each image
        for i in range(0, data.shape[0]):
            img, img_label = data[i], labels[i]
            ax  = fig.add_subplot(num_rows, num_cols, i+1)
            _ = ax.set_axis_off()
            ax.set_title(img_label)

            _ = plt.imshow(img)
        fig.tight_layout()

        return fig

    def modelPath(self, modelName):
        return os.path.join(".", "models", modelName)

    def saveModel(self, model, modelName): 
        model_path = self.modelPath(modelName)

        try:
            os.makedirs(model_path)
        except OSError:
            print("Directory {dir:s} already exists, files will be over-written.".format(dir=model_path))

        tf.keras.models.save_model(model, model_path)

        print("Model saved in directory {dir:s}; create an archive of this directory and submit with your assignment.".format(dir=model_path))

    def loadModel(self, modelName):
        model_path = self.modelPath(modelName)

        model = tf.keras.models.load_model(model_path)

        return model

    def saveModelNonPortable(self, model, modelName): 
        model_path = self.modelPath(modelName)

        try:
            os.makedirs(model_path)
        except OSError:
            print("Directory {dir:s} already exists, files will be over-written.".format(dir=model_path))

        model.save( model_path )

        print("Model saved in directory {dir:s}; create an archive of this directory and submit with your assignment.".format(dir=model_path))
   
    def loadModelNonPortable(self, modelName):
        model_path = self.modelPath(modelName)
        model = self.load_model( model_path )

        # Reload the model 
        return model

    def saveHistory(self, history, model_name):
        history_path = self.modelPath(model_name)

        try:
            os.makedirs(history_path)
        except OSError:
            print("Directory {dir:s} already exists, files will be over-written.".format(dir=history_path))

        # Save JSON config to disk
        with open(os.path.join(history_path, 'history'), 'wb') as f:
            pickle.dump(history.history, f)

    def loadHistory(self, model_name):
        history_path = self.modelPath(model_name)
      
        # Reload the model from the 2 files we saved
        with open(os.path.join(history_path, 'history'), 'rb') as f:
            history = pickle.load(f)
      
        return history

    def json_to_numpy(self, json_file):
        # Read the JSON file
        f = open(json_file)
        dataset = json.load(f)
        f.close()

        data = np.array(dataset['data']).astype('uint8')
        labels = np.array(dataset['labels']).astype('uint8')

        # Reshape the data
        data = data.reshape([-1, 3, 80, 80]).transpose([0,2,3,1])

        return data, labels

    def plotTrain(self, history, model_name="???"):
        # Determine the name of the key that indexes into the accuracy metric
        acc_string = self.acc_key(history=history)
        
        fig, axs = plt.subplots( 1, 2, figsize=(12, 5) )

        # Plot loss
        axs[0].plot(history.history['loss'])
        axs[0].plot(history.history['val_loss'])
        axs[0].set_title(model_name + " " + 'model loss')
        axs[0].set_ylabel('loss')
        axs[0].set_xlabel('epoch')
        axs[0].legend(['train', 'validation'], loc='upper left')

        # Plot accuracy
        axs[1].plot(history.history[acc_string])
        axs[1].plot(history.history['val_'+acc_string])
        axs[1].set_title(model_name + " " +'model accuracy')
        axs[1].set_ylabel('accuracy')
        axs[1].set_xlabel('epoch')
        axs[1].legend(['train', 'validation'], loc='upper left')

        return fig, axs
