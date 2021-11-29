import copy


def createIndex(oldindex):
    day = '01'
    month = str(oldindex)[4:6]
    year = str(oldindex)[0:4]
    return year + month + day


def R2_adjusted(y_true, y_pred):
    import numpy as np
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return 1 - (np.sum((y_true - y_pred) ** 2) / np.sum(y_true ** 2))


def prepare_data(all_data, market_orig, risk_free_rate, stock, feature_list, with_market=True, scale=True):
    from sklearn.preprocessing import MinMaxScaler
    import pandas as pd
    market = copy.deepcopy(market_orig)
    # Filter relevant Stock and Columns
    stock_data = copy.deepcopy(all_data[all_data['permno'] == stock])
    stock_data = stock_data[['RET'] + feature_list]
    # Build index to match S&P500 index (start of month)
    new_index = []
    for index in stock_data.index:
        new_index.append(createIndex(index))
    stock_data.index = new_index
    # Replace nan with 0
    stock_data = stock_data.fillna(0)

    # Index comparison
    for row in market.index:
        if stock_data.index.to_list().count(row) != 1:
            market = market[market.index != row]

    for row in stock_data.index:
        if market.index.to_list().count(row) != 1:
            stock_data = stock_data[stock_data.index != row]

    risk_free_rate.index = risk_free_rate.index.astype(str)
    for index in stock_data.index:
        if risk_free_rate.index.to_list().count(index) == 1:
            stock_data.loc[index]["RET"] = stock_data.loc[index]["RET"] - risk_free_rate.loc[index]["DGS3M"]
            market.loc[index]["Market Return"] = market.loc[index]["Market Return"] - risk_free_rate.loc[index]["DGS3M"]
        else:
            stock_data.drop(index, inplace=True)
            market.drop(index, inplace=True)


    # Concat stock data and market data
    if with_market:
        if len(feature_list) > 0:
            nn_data = pd.concat([stock_data, market], axis=1)
        else:
            nn_data = pd.concat([stock_data, market, market], axis=1)
    else:
        nn_data = stock_data

    # Split into features and target
    nn_data_features = nn_data.drop("RET", axis=1)
    nn_data_target = nn_data["RET"]
    # Scale features to Interval [-1,1]
    if scale:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(nn_data_features)
        nn_data_features = scaler.transform(nn_data_features)
        nn_data_features = pd.DataFrame(nn_data_features)
    return nn_data_features, nn_data_target


def get_market_ret(startstring, endstring):
    import yfinance as yf
    # Import S&P 500 Data for train data
    sp500 = yf.download('^GSPC', start=startstring, end=endstring, interval="1mo")
    sp500 = sp500["Adj Close"]
    # Adjust datetime
    sp500.index = sp500.index.strftime('%Y%m%d')
    # Calculate of return
    market_ret = (sp500 / sp500.shift()) - 1
    # Rename Column
    market_ret = market_ret.to_frame()
    market_ret.rename(columns={"Adj Close": "Market Return"}, inplace=True)
    # Drop NAN
    market_ret = market_ret.dropna()
    return market_ret


def plot_CAPM(x, y, result, stock, r2_adj):
    import matplotlib.pyplot as plt
    # Plot manuell gebaut
    plt.suptitle("Stock: " + f'{stock}')
    plt.title("R2: " + f'{r2_adj}')
    plt.plot(x, y, 'bo')
    xbottom, xtop = plt.xlim()
    ybottom, ytop = plt.ylim()
    xbottom = min(xbottom, ybottom)
    xtop = max(xtop, ytop)
    ybottom = xbottom
    ytop = xtop
    plt.ylim(ybottom, ytop)
    plt.xlim(xbottom, xtop)
    frame_prediction = x * result.params["Market Return"] + result.params["const"]
    plt.plot(x, frame_prediction["Market Return"], 'r-')
    plt.show()