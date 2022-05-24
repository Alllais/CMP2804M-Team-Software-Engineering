import requests
import json

def getData(ticker):
    # API required a key, this is stored here
    apiKey = "gBUqdgyfSO5ZAYIqMEXeT4xjMA7xlLzh5nfxewQ4"
    # The URL used for the API, specifically the Quote Summary for a specific ticker
    url = "https://yfapi.net/v11/finance/quoteSummary/" + str(ticker)

    # State what data we want to get
    querystring = {"modules": "balanceSheetHistory,incomeStatementHistory,earningsHistory,assetProfile,summaryDetail,defaultKeyStatistics"}
    # Have the API key as a header, as requested by the API documentation
    headers = {'x-api-key': apiKey}

    # Make the request
    response = requests.request("GET", url, headers=headers, params=querystring)

    # Load the response into a JSON object
    data = json.loads(response.content)["quoteSummary"]["result"][0]

    # Return the data
    return {
        "employees" : data["assetProfile"]["fullTimeEmployees"],
        "volume" : data["summaryDetail"]["volume"]["raw"],
        "eps" : data["earningsHistory"]["history"][0]["epsActual"]["raw"],
        "profit" : data["incomeStatementHistory"]["incomeStatementHistory"][0]["grossProfit"]["raw"],
        "income" : data["incomeStatementHistory"]["incomeStatementHistory"][0]["incomeBeforeTax"]["raw"],
        "cash" : data["balanceSheetHistory"]["balanceSheetStatements"][0]["cash"]["raw"],
        "liab" : data["balanceSheetHistory"]["balanceSheetStatements"][0]["totalLiab"]["raw"],
        "assets" : data["balanceSheetHistory"]["balanceSheetStatements"][0]["totalAssets"]["raw"],
        "cap" : data["summaryDetail"]["marketCap"]["raw"],
        "shares" : data["defaultKeyStatistics"]["sharesOutstanding"]["raw"],
        "ticker" : ticker
    }

def getCurrentPrice(ticker):
    apiKey = "LTIXeAVEBp7i2Of5goSM55jZlQ0eiPP02FNhfJ1I"
    url = "https://yfapi.net/v6/finance/quote"

    querystring = {"symbols": ticker}
    headers = {'x-api-key': apiKey}

    response = requests.request("GET", url, headers=headers, params=querystring)

    data = json.loads(response.content)

    return data["quoteResponse"]["result"][0]["regularMarketPrice"]

def getHistoricalPrice(ticker, interval, period):
    # API required a key, this is stored here
    apiKey = "LTIXeAVEBp7i2Of5goSM55jZlQ0eiPP02FNhfJ1I"
    headers = { 'x-api-key': apiKey }

    # Structure used for modularity; not implemented here
    url = "https://yfapi.net/v8/finance/spark?"
    parameters = {
        "symbols":ticker,
        "range":period,
        "interval":interval,
        "lang":"en"
    }

    response = requests.request("GET", url, headers=headers, params=parameters)
    data = json.loads(response.content) # Using JSON for this implementation

    # Lists here to store stock values and time
    dataList = []

    for selectedTicker in data:
        thisTicker = data[selectedTicker]
        for i in range(0, len(thisTicker["close"]) ):
            dataList.append([ thisTicker["symbol"], int(round(((thisTicker["timestamp"][i] - thisTicker["timestamp"][0])/86400) + 1, 0)), thisTicker["timestamp"][i] , float(round(thisTicker["close"][i], 2)) ]) # Use two-dimensional arrays

    return dataList
