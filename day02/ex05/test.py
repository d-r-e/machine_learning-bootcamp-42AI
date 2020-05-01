import pandas as pd
import numpy as np
from log_reg import LogisticRegressionBatchGd as LogReg

if __name__ == "__main__":

    # We load and prepare our train and test dataset into x_train, y_train and x_test, y_test
    df_train = pd.read_csv('train_dataset_clean.csv', delimiter=',', header=None, index_col=False)
    x_train, y_train = np.array(df_train.iloc[:, 1:82]), df_train.iloc[:, 0]
    df_test = pd.read_csv('test_dataset_clean.csv', delimiter=',', header=None, index_col=False)
    x_test, y_test = np.array(df_test.iloc[:, 1:82]), df_test.iloc[:, 0]
    # We set our model with our hyperparameters : alpha, max_iter, verbose and learning_rate
    model = LogReg(alpha=0.01, max_iter=1500, verbose=False, learning_rate='constant')
    # We fit our model to our dataset and display the score for the train and test datasets
    model.fit(x_train, y_train)
    print(f'Score on train dataset : {model.score(x_train, y_train)}')
    y_pred = model.predict(x_test)
    print(f'Score on test dataset : {(y_pred == y_test).mean()}')
