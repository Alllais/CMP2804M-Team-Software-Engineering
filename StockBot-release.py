from operator import index
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import PySimpleGUI as sg
import os


from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load Data

layoutTicker = [[sg.Text("Input stock ticker")], [sg.InputText()], [sg.Button("Generate Model")]],

# Create the window
window = sg.Window("StockBot V1.0", layoutTicker)

# Create an event loop
while True:
    event, values = window.read()
    # End program if user closes window or
    # presses the OK button
    if event == "Generate Model":

        event, values = window.read()
        company = values[0]
        break

    if event == event == sg.WIN_CLOSED:
        #
        #
        break




#company = 'ORCL'

start = dt.datetime(2019,1,2)
end = dt.datetime(2020,1,2)

data = web.DataReader(company, 'yahoo', start, end)

start2 = dt.datetime(2019,1,3)
end2 = dt.datetime(2020,1,3)

data2 =  web.DataReader(company, 'yahoo', start2, end2)

#Preprocessing
ct = make_column_transformer(
    (MinMaxScaler(), ["High", "Low", "Open", "Close", "Volume", "Adj Close"])
)




#x represents the data, y represents "High"
x = data
#Offset the data by 1
y = data2["High"]
print(len(x))
#print(len(y))
#If (high > close) {buy;} else if (high < close) {sell;}
#80/20 train/test split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state = 42)

#ct.fit(x_train)
#x_train_normal = ct.transform(x_train)
#x_test_normal = ct.transform(x_test)


tf.random.set_seed(42)

#Include the epoch in the file name (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

batch_size = 32

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=5*batch_size)


#Create a model
stock_prediction_model = tf.keras.Sequential([
                                                tf.keras.layers.Dense(1000, activation ="selu"),
                                                tf.keras.layers.Dense(1000, activation = "selu"),
                                                tf.keras.layers.Dense(1000, activation = "relu"),
                                                tf.keras.layers.Dense(1000, activation = "selu"),
                                                tf.keras.layers.Dense(1000, activation = "selu"),
                                                tf.keras.layers.Dense(1000, activation = "relu"),
                                                tf.keras.layers.Dense(1000, activation = "selu"),
                                                tf.keras.layers.Dense(1000, activation = "selu"),
                                                tf.keras.layers.Dense(1000, activation = "selu"),
                                                tf.keras.layers.Dense(1000, activation = "selu"),
                                                tf.keras.layers.Dense(1000, activation = "selu"),
                                                tf.keras.layers.Dense(1000, activation = "selu"),
                                                tf.keras.layers.Dense(1000, activation = "selu"),
                                                tf.keras.layers.Dense(1000, activation = "selu"),
                                                tf.keras.layers.Dense(1000, activation = "selu"),
                                                tf.keras.layers.Dense(1000, activation = "selu"),
                                                tf.keras.layers.Dense(1000, activation = "selu"),
                                                tf.keras.layers.Dense(1000, activation = "selu"),
                                                tf.keras.layers.Dense(1000, activation = "selu"),
                                                tf.keras.layers.Dense(1000, activation = "selu"),
                                                tf.keras.layers.Dense(1000, activation = "selu"),
                                                tf.keras.layers.Dense(1000, activation = "selu"),
                                                tf.keras.layers.Dense(1)
])

stock_prediction_model._name = "Stock_Prediction_Model"

#Compile the model
stock_prediction_model.compile ( loss = tf.keras.losses.mae,
                                optimizer = tf.keras.optimizers.Adam(lr=0.0005),
                                metrics = ["mae"])

#Fit the model
history = stock_prediction_model.fit(x_train, y_train, epochs = 50)



#Actually use it
test_start = dt.datetime(2020,1,1)
test_end = dt.datetime(2020,1,2)

test_data = web.DataReader(company, 'yahoo', test_start, test_end)

layout = [[sg.Text("Model Generated")], [sg.Button("Save Current Model")]]

# Create the window
window = sg.Window("StockBot V1.0", layout)

# Create an event loop
while True:
    event, values = window.read()
    # End program if user closes window or
    # presses the OK button
    if event == "Save Current Model" or event == sg.WIN_CLOSED:
        
        print("Model Saved")
        stock_prediction_model.save_weights('./checkpoints/currentxvc.ckpt')
        break

window.close()
