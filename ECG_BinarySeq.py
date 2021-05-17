import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import layers, losses
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model

#https://www.kaggle.com/devavratatripathy/ecg-dataset

'''
   init()
   This is a temp function to check data structures.
   Argument:
   Return: 
'''
def init():
    #Use pandas (pd) to read the csv file into a data file (df)
    #Use pandas API call to print the first 5 rows to check the data
    df = pd.read_csv('./ecg.csv', header=None)
    print(df.head())

    #Now we will separate the data and labels so that it will be easy for us
    #To just work with the data later on.. we don't need the labels - we aren't
    #Trying to do Machine Learning.. yet.. maybe??
    data = df.iloc[:,:-1].values
    labels = df.iloc[:,-1].values
    #print(labels)

    #Separate the data in case we want to do any ML later on.
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = 0.2, random_state = 21)
   
    #Now lets Normalize the data
    #First we will calculate the maximum and minimum value from the training set 
    min = tf.reduce_min(train_data)
    max = tf.reduce_max(train_data)

    #Now we will use the formula (data - min)/(max - min)
    train_data = (train_data - min)/(max - min)
    test_data = (test_data - min)/(max - min)

    #I have converted the data into float
    train_data = tf.cast(train_data, dtype=tf.float32)
    test_data = tf.cast(test_data, dtype=tf.float32)

    #The labels are either 0 or 1, so I will convert them into boolean(true or false) 
    train_labels = train_labels.astype(bool)
    test_labels = test_labels.astype(bool)

    #Now let's separate the data for normal ECG from that of abnormal ones
    #Normal ECG data
    n_train_data = train_data[train_labels]
    n_test_data = test_data[test_labels]

    #Abnormal ECG data
    an_train_data = train_data[~train_labels]
    an_test_data = test_data[~test_labels]
    print(n_train_data)
   
    #Lets plot a normal ECG - this is cool.. 
    #However the problem we have is we need to find out how 
    #TODO: how do you get the fiducual points? PQRST and their primes?
    #Some how we use the 140 columns in the data file.. and identify them.
    #The columns are just floating points.. there are math models to get this.
    #But this is a start.
    plt.plot(np.arange(140), train_data[0])
    plt.grid()
    plt.title('Normal ECG')
    plt.show()
   

if __name__ == '__main__':
    init()
