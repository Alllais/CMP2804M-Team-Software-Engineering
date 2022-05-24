from accessToAPI import *
import csv

### The following code is used to build the dataset. The API used only allows for 100 requests per day. ###
### We wanted it faster than that, so multiple API keys were used, with the code being used for 100 tickers at a time. ###

# Create the csv file
csvFile = open("5data.csv", "w", newline='')
# Define the header for the file (the labels)
csvHeader = ["employees", "volume", "eps", "profit", "income", "cash", "liab", "assets", "cap", "shares", "ticker"]
# Create the writer object
csvWriter = csv.writer(csvFile)
# Write the header to the file
csvWriter.writerow(csvHeader)

# Reference the file containing the tickers of the companies in S&P 500
tickFile = open("sp500.txt", "r").readlines()
for i in range(0,len(tickFile)):
    # Remove the newline character
    tickFile[i] = tickFile[i].replace("\n", "")

# Go back through, printing the data for each ticker for debugging
for i in range(0,len(tickFile)):
    print(i, tickFile[i])

# The range of tickers, to be changed each time the API is used (100 at a time)
tickers = tickFile[400:499]

for ticker in tickers:
    # Some tickers do not have all of the data, so we skip these using the try/except
    try:
        # Get the data from the API for the given ticker
        data = getData(ticker)
        # Clear the row
        csvRow = []
        # Add the data to the row
        for datapoint in data:
            csvRow.append(data.get(datapoint))
        # Write the data to the file
        csvWriter.writerow(csvRow)
    except Exception:
        # If a data point is missing, we just skip adding it to the file
        print("Failed to process", ticker)