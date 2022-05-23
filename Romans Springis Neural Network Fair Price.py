from operator import index
from matplotlib.axis import XAxis, YAxis
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#inputs


#matrix_input = #[[6.01, 0.1485, 0.15, 26.09],
    #[4.90, 0.2170, 0.15, 167.55],
    #[12.94, 0.4727, 0.15, 4.57],
    #[2.25, 0.0724, 0.15, 26.82],
    #[10.04, 0.1297, 0.15, 23.64],
    #[9.39, 0.1740, 0.15, 32.35],
    #[4.87, 0.0835, 0.15, 29.51],
    #[3.79, 0.1534, 0.15, 35.54],
    #[84.60, 0.2964, 0.15, 3.30],
    #[3.70, 0.1140, 0.15, 24.02]]
matrix_input = [[2.25, 0.0724, 0.15, 26.82],
[3.70, 0.1140, 0.15, 24.02],
[3.79, 0.1534, 0.15, 35.54],
[4.87, 0.0835, 0.15, 29.51],
[4.90, 0.2170, 0.15, 167.55],
[6.01, 0.1485, 0.15, 26.09],
[9.39, 0.1740, 0.15, 32.35],
[10.04, 0.1297, 0.15, 23.64],
[12.94, 0.4727, 0.15, 4.57],
[84.60, 0.2964, 0.15, 3.30]]
#output
#fairPrice = [154.97, 1366.71, 547.77, 32.18, 202.19, 365.82, 84.08, 138.32, 286.70, 66.75]
fairPrice = [32.18, 66.75, 138.32, 84.08, 1366.71, 154.97, 365.82, 202.19, 47.77, 286.70]
x = matrix_input
y = fairPrice

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state = 42)
tf.random.set_seed(42)
#Create
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="selu"),
    tf.keras.layers.Dense(100, activation="selu"),
    tf.keras.layers.Dense(100, activation="selu"),
    tf.keras.layers.Dense(100, activation="selu"),
    tf.keras.layers.Dense(100, activation="selu"),
    tf.keras.layers.Dense(100, activation="selu"),
    tf.keras.layers.Dense(100, activation="selu"),
    tf.keras.layers.Dense(100, activation="selu"),
    tf.keras.layers.Dense(100, activation="selu"),
    tf.keras.layers.Dense(100, activation="selu"),
    tf.keras.layers.Dense(1)

])

#Compile
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.Adam(lr=0.001),
              metrics=["mae"])

#Fit
model.fit(x_train, y_train, epochs = 500)

array = model.evaluate(x_test, y_test)
print(array)
print(plt.plot(array))
#dense model
#https://www.youtube.com/watch?v=tpCFfeUEGs8
