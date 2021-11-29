def gradient_boosting(features_train, features_test, target_train, target_test, loss_fnc, estimators, lr, crit, depth):
    import tensorflow as tf
    from Helper import R2_adjusted
    import numpy.random
    from sklearn.ensemble import GradientBoostingRegressor

    # Set random seed for consistent results
    numpy.random.seed(28102021)
    tf.random.set_seed(28102021)

    # Metric Variables
    metric = tf.keras.losses.MeanSquaredError()

    # Build RF
    gb = GradientBoostingRegressor(random_state=7, loss=loss_fnc, n_estimators=estimators, learning_rate=lr, criterion=crit, max_depth=depth, max_features="sqrt")
    history = gb.fit(features_train, target_train)

    # Prediction with test data
    y_pred_test = gb.predict(features_test)

    # Calculation of Metrics
    R2 = R2_adjusted(target_test, y_pred_test)
    MSE = metric(target_test, y_pred_test)

    return history, MSE, R2, y_pred_test

def GB_single_run(df, market_return, df_risk_free, stock, data_setting, gb_setting, return_predictions=False):
    from Helper import prepare_data
    from sklearn.model_selection import train_test_split

    X, y = prepare_data(df, market_return, df_risk_free, stock, data_setting[0], data_setting[1],scale=False)

    # Split data into training data and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    hist, mse, r2, pred = gradient_boosting(X_train, X_test, y_train, y_test, gb_setting[0], gb_setting[1], gb_setting[2], gb_setting[3], gb_setting[4])
    if return_predictions:
        return hist, mse, r2, pred
    else:
        return hist, mse, r2