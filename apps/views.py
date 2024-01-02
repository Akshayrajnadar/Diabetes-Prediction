from django.shortcuts import render
# import numpy as np
# import pandas as pd

# import tensorflow as tf
# from tensorflow.keras import datasets, layers, models
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# import keras
# from keras.models import Sequential
# from keras.layers import Dense , Input
# from keras.layers import LSTM
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# Create your views here.
def index(request):
    return render(request, 'index.html')

def home(request):
    return render(request, 'home.html')

def reload(request):

    # training_data = pd.read_csv(r'D:\Django and Flask video\Django practice\Diabetese_prediction\diabetes.csv')

    # X = training_data.drop("Outcome", axis = 1)
    # Y = training_data['Outcome']

    # X_train, X_test , Y_train, Y_test = train_test_split(X, Y, test_size= 0.30)

    # model = models.Sequential()
    # model.add(Input( shape=[8,]))

    # model.add(Dense(50, activation='relu'))
    # model.add(Dense(50, activation='relu'))
    # model.add(Dense(50, activation='relu'))
    # model.add(Dense(40, activation='relu'))

    # model.add(Dense(1, activation='sigmoid'))

    # model.compile(optimizer='sgd',
    #           loss='binary_crossentropy',
    #           metrics=['accuracy'])
    
    # model.fit(X_train, Y_train, epochs=200,validation_data=(X_test, Y_test))
    
    from keras.models import load_model
    model = load_model('D:\Django and Flask video\Django practice\Diabetese_prediction\new_diabatese_weights.hdf5')
    
    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])

    
    print(pred[0][0])

    result1 = ""
    if pred[0][0] < 0.5:
        result1 = f'There is {round(pred[0][0] *100)} % probability of having diabetese'
    elif pred[0][0] > 0.5:
        result1 = f'There is {round(pred[0][0] *100)} % probability of having diabetese'
    elif pred[0][0] > 0.7:
        result1 = f'There is {round(pred[0][0] *100)} % probability of having diabetese'
    elif pred[0][0] > 0.8:
        result1 = f'There is {round(pred[0][0] *100)} % probability of having diabetese'
    else:
        result1 = f'There is {round(pred[0][0] *100)} % probability of having diabetese'

    return render(request, 'result.html', {"result2":result1})
