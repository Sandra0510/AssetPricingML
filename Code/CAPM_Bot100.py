import statsmodels.api as sm
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

from Helper import get_market_ret, prepare_data, R2_adjusted, plot_CAPM

start = 19991201
end = 20210101

target_stock_count = 100

df = pd.read_csv(r"GKX_20201231.csv", index_col="DATE")
df = df.loc[start:end]

df_risk_free = pd.read_csv(r"DGS3MO.csv", index_col="DATE")
df_risk_free = df_risk_free.loc[start:end]

candidate_stocks = pd.DataFrame(df.sort_values('mvel1', ascending=True)["permno"].unique())[0].values.tolist()
counter_added_stocks = 0
max_len_stocks = df["permno"].value_counts().max()
list_chosen_stocks = []
for stock in candidate_stocks:
    df_current_stock = df[df["permno"] == stock]
    if len(df_current_stock) / max_len_stocks >= 0.5:
        list_chosen_stocks.append(stock)
        counter_added_stocks = counter_added_stocks + 1
    if counter_added_stocks == target_stock_count:
        break

# Check if enough stocks were found which match the criteria
if counter_added_stocks < target_stock_count:
    print("Not enough stocks matching the criteria found")
    exit()
else:
    print("The chosen stocks are:")
    print(list_chosen_stocks)

market_ret = get_market_ret('2000-01-01', '2021-01-01')

resultdict = {}

for stock in list_chosen_stocks:
    market_ret_single_stock, stock_ret = prepare_data(df, market_ret, df_risk_free, stock, [], with_market=True,
                                                      scale=False)
    market_ret_single_stock = market_ret_single_stock.iloc[:, :1]  # remove second market return column

    # Split data into training data and test data
    X_train, X_test, y_train, y_test = train_test_split(market_ret_single_stock, stock_ret, test_size=0.2,
                                                        random_state=42)

    # Regression with market ret (x) and stock return (y)
    X_sm = sm.add_constant(X_train)
    X_sm.index = y_train.index
    model = sm.OLS(y_train, X_sm)
    results = model.fit()

    # Calculation of Accuracy (MSE and R2)
    metric = tf.keras.losses.MeanSquaredError()
    pred = X_test * results.params[1] + results.params[0]
    real = y_test
    #pd.concat([pd.DataFrame(real), pd.DataFrame(pred)], axis=1).to_csv(f'Predictions\{stock}_prediction_CAPM.csv')

    resultdict[stock] = (metric(real, pred).numpy(), R2_adjusted(real, pred.squeeze()))
    # plot_CAPM(market_ret_single_stock, stock_ret, results, stock, R2_adjusted(real, pred.squeeze()))

sum_r2_adj = 0
resultfile = open("results_CAPM_Bot100.csv", "w")
resultfile.write(f'stock;MSE;R2_adjusted\n')
for stock in resultdict.keys():
    print(f'Stock:{stock} MSE:{resultdict[stock][0]} R2_Adj:{resultdict[stock][1]}')
    resultfile.write(f'{stock};{resultdict[stock][0]};{resultdict[stock][1]}\n')
    sum_r2_adj += resultdict[stock][1]
print(f'Mean R2_adj: {sum_r2_adj / len(resultdict.keys())}')
resultfile.write(f'Mean R2_Adj: {sum_r2_adj / len(resultdict.keys())}')
resultfile.close()
