import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


class Helper():
    def __init__(self):
        self.DATA_DIR = './Data'
        
        if not os.path.isdir(self.DATA_DIR):
            self.DATA_DIR = '../resource/asnlib/publicdata/'
        self.data_file = 'data.csv'
        
    
    def getData(self):
        train = pd.read_csv( os.path.join(self.DATA_DIR, 'train', self.data_file) )
        holdout = pd.read_csv( os.path.join(self.DATA_DIR, 'holdout', self.data_file) )
        return train, holdout


    def plot_attr(self, df, y_train, attr, trunc=.01):
	    X = df[attr]
	    
	    # Remove outliers, to improve clarity
	    mask = (X > X.quantile(trunc)) & (X < X.quantile(1-trunc))
	    X_trunc, y_trunc = X[ mask  ], y_train[ mask ]

	    bins = np.linspace( int(X_trunc.min()), int(X_trunc.max() +1), 30)
	    
	    fig, ax = plt.subplots( figsize=(8,4))
	    color = 'tab:green'
	    ax.set_xlabel(attr)
	    ax.set_ylabel('count', color=color)
	    data = X_trunc[ y_trunc == 0 ]
	    ax.hist( data, bins, alpha=0.5, label='0', color=color, weights=np.ones(len(data)) / len(data))
	    ax.tick_params(axis='y', labelcolor=color)

	    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

	    color = 'tab:red'
	    ax2.set_xlabel(attr)
	    ax2.set_ylabel('count', color=color)
	    data =  X_trunc[ y_trunc == 1 ]
	    ax2.hist( data, bins, alpha=0.5, label='1', color=color, weights=np.ones(len(data)) / len(data))
	    ax2.tick_params(axis='y', labelcolor=color)

    def save_data(self, X_train, X_test, y_train, y_test):
        if not os.path.exists('./mydata'):
            os.mkdir('./mydata')
        np.savez_compressed('./mydata/train_test_data.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

        
    def load_data(self):
        data_dir = np.load('./mydata/train_test_data.npz')
        X_train = data_dir['X_train']
        X_test = data_dir['X_test']
        y_train = data_dir['y_train']
        y_test = data_dir['y_test']
        
        return X_train, X_test, y_train, y_test
