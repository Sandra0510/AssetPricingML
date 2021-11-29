def random_forest(features_train, features_test, target_train, target_test, criterion, depth):
    import tensorflow as tf
    import numpy.random
    from Helper import R2_adjusted
    from sklearn.ensemble import RandomForestRegressor
    # Set random seed for consistent results
    numpy.random.seed(28102021)
    tf.random.set_seed(28102021)

    # Metric Variables
    metric = tf.keras.losses.MeanSquaredError()

    # Build RF
    RF = RandomForestRegressor(criterion=criterion, n_estimators=300, max_depth=depth, random_state=7, max_features="sqrt")
    history = RF.fit(features_train, target_train)

    # Prediction with test data
    y_pred_test = RF.predict(features_test)

    # Calculation of Metrics
    R2 = R2_adjusted(target_test, y_pred_test)
    MSE = metric(target_test, y_pred_test)

    return history, MSE, R2, y_pred_test

def RF_single_run(df, market_return, df_risk_free, stock, data_setting, rf_setting, return_predictions=False):
    from Helper import prepare_data
    from sklearn.model_selection import train_test_split

    X, y = prepare_data(df, market_return, df_risk_free, stock, data_setting[0], data_setting[1],scale=False)

    # Split data into training data and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    hist, mse, r2, pred = random_forest(X_train, X_test, y_train, y_test, rf_setting[0], rf_setting[1])
    if return_predictions:
        return hist, mse, r2, pred
    else:
        return hist, mse, r2