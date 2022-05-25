Welcome to the StockBot AI Model Generator!

After installing the dependancies, using this model generator is simple. 

Step 1: Run the StockBot V1.0-ReleaseCandidate.py file

Step 2: Input the stock ticker (abbreviated stock name such as SNAP or AAPL) and double-click the generate model button

Step 3: When finished a prompt to save the model appears, click save and navigate to your user folder and then checkpoints.

Step 4: The latest saved checkpoint will be saved as currentxvc.ckpt in two parts (index and data file) which can be then used to rebuild the model for use in other projects.

The promotional demonstration video is available here: 

https://www.youtube.com/watch?v=eGam8GHy1XY


The generated model's MAE (mean average error) can be found in the Python console as the AI generates. 
We would recommend values under 15 to be both common and usable for the models, however some stocks are more unpredictable for the AI models.
In these cases we would caution against the use of these models in any investment or financial analysis tools or environments.
Stocks such as AAPL and SNAP are a good benchmark to check the software is working, as when tested our MAE values were around 6.5 for AAPL and 2.5 for SNAP.

This project has dependancies on TensorFlow, Pandas Datareader and PySimpleGUI which can be installed with the following commands:

pip install pysimplegui

pip install tensorflow

pip install pandas-datareader

