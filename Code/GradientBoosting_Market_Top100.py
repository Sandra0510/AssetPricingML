import pandas as pd

from GB_Helper import GB_single_run
from Helper import get_market_ret

# Set number of stocks
target_stock_count = 100

# Set time frame
start = 19991201
end = 20210101

# Set filename for results
result_filename = "results_GB_Market_Top100.csv"

# Data and RF Settings
data_preparation_settings = [(['mom1m', 'mvel1'], True), (['mom1m', 'mvel1', 'retvol'], True)]

loss_fnc = ["squared_error", "absolute_error", "huber"]
estimators = [*range(100, 400, 100)]
lr= [0.01, 0.1]
crit = ["friedman_mse"]
depth = [4,5,6]

# Import Data
df = pd.read_csv(r"GKX_20201231.csv", index_col="DATE")
df = df.loc[start:end]

df_risk_free = pd.read_csv(r"DGS3MO.csv", index_col="DATE")
df_risk_free = df_risk_free.loc[start:end]

candidate_stocks = pd.DataFrame(df.sort_values('mvel1', ascending=False)["permno"].unique())[
    0].values.tolist()  # Choose ascending = True to pick the bottom 100
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

gb_settings = []
for loss in loss_fnc:
    for est in estimators:
        for learning_rate in lr:
            for criteria in crit:
                for dep in depth:
                    gb_settings.append((loss, est, learning_rate, criteria, dep))

progress = 0
for stock in list_chosen_stocks:
    resultlist = []
    for data_setting in data_preparation_settings:
        for gb_setting in gb_settings:
            resultlist.append((GB_single_run(df, market_ret, df_risk_free, stock, data_setting, gb_setting), data_setting, gb_setting))

    best_result = resultlist[0]
    for result in resultlist:
        if result[0][2] > best_result[0][2]:
            best_result = result

    resultdict[stock] = best_result
    progress += 1
    print(f'Progress: {progress}%')

resultfile = open(result_filename, "w")
resultfile.write(f'stock;MSE;R2_adjusted;features;gb_settings\n')
sum_r2_adj = 0

for stock in resultdict.keys():
    # plot_loss(resultdict[stock][0][0], stock, resultdict[stock][0][1], resultdict[stock][0][2])
    print(
        f'Stock:{stock} MSE:{resultdict[stock][0][1]} R2_Adj:{resultdict[stock][0][2]} DataSettings:{resultdict[stock][1]} GBSettings {resultdict[stock][2]}')
    resultfile.write(
        f'{stock};{resultdict[stock][0][1]};{resultdict[stock][0][2]};{resultdict[stock][1]};{resultdict[stock][2]};')
    resultfile.write('\n')
    sum_r2_adj += resultdict[stock][0][2]
print(f'Mean best r2_adj: {sum_r2_adj / len(resultdict.keys())}')
resultfile.write(f'Mean best r2_adj: {sum_r2_adj / len(resultdict.keys())}')
resultfile.close()


