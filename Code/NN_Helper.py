def build_and_compile_model(norm, layers_neuron, layers_activation, optimizer, opt_learning_rate):
    from tensorflow import keras
    from tensorflow.keras import layers
    # Layers variable
    if len(layers_neuron) == len(layers_activation):
        model = keras.Sequential()
        model.add(norm)
        for i in range(0, len(layers_neuron)):
            model.add(layers.Dense(layers_neuron[i], activation=layers_activation[i]))
        model.add(layers.Dense(1))

        # Optimizer und Learning Rate variabel
        model.compile(loss='mean_squared_error', optimizer=optimizer(learning_rate=opt_learning_rate))

        return model
    else:
        print('Length mismatch')
        exit()

def neural_net(features_train, features_test, target_train, target_test, layers_neuron, layers_activation, optimizer,opt_learning_rate):
    import tensorflow as tf
    import numpy.random
    from Helper import R2_adjusted

    # Set random seed for consistent results
    numpy.random.seed(28102021)
    tf.random.set_seed(28102021)

    # Metric Variables
    metric = tf.keras.losses.MeanSquaredError()

    # Normalize features
    normalizer = tf.keras.layers.Normalization()
    normalizer.adapt(features_train)

    # Create model
    dnn_model = build_and_compile_model(normalizer, layers_neuron, layers_activation, optimizer, opt_learning_rate)
    callback_loss = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    callback_val_loss = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    history = dnn_model.fit(features_train, target_train, validation_split=0.2, verbose=0, epochs=500,
                            callbacks=[callback_loss], workers=8, use_multiprocessing=True)

    # Prediction with test data
    y_pred = dnn_model.predict(features_test).flatten()

    # Calculation of Metrics
    y_true = target_test
    return history, metric(y_true, y_pred).numpy(), R2_adjusted(y_true, y_pred), y_pred

def NN_single_run(df, market_return, df_risk_free, stock, data_setting, nn_setting, return_predictions=False):
    from sklearn.model_selection import train_test_split
    from Helper import prepare_data

    X, y = prepare_data(df, market_return, df_risk_free, stock, data_setting[0], data_setting[1], scale=True)

    # Split data into training data and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Run NN
    hist, mse, r2, pred = neural_net(X_train, X_test, y_train, y_test, nn_setting[0], nn_setting[1],nn_setting[2], nn_setting[3])
    if return_predictions:
        return hist, mse, r2, pred
    else:
        return hist, mse, r2

def plot_loss(history, title, mse, r2_adj):
    import matplotlib.pyplot as plt
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [Return]')
    plt.legend()
    plt.suptitle(f'Stock:{title}')
    plt.title(f'MSE: {mse} R2_adj: {r2_adj}')
    plt.grid(True)
    plt.show()