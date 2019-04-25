# -*- encoding: utf-8 -*-
"""
==========
Regression
==========

The following example shows how to fit a simple regression model with
*auto-sklearn*.
"""
import numpy as np 
import sklearn.model_selection
from sklearn.linear_model import SGDRegressor
from sklearn.externals import joblib
import sklearn.datasets
import sklearn.metrics

import autosklearn.regression

def main():
    # X, y = sklearn.datasets.load_boston(return_X_y=True)
    # feature_types = (['numerical'] * 3) + ['categorical'] + (['numerical'] * 9)
    # X_train, X_test, y_train, y_test = \
    #     sklearn.model_selection.train_test_split(X, y, random_state=1)

    # metadata = np.load('./10_australian_big_metadata.npy')
    metadata1 = np.load('./query_time50/2australian30_big_metadata50.npy')
    metadata2 = np.load('./query_time50/20australian30_big_metadata50.npy')
    metadata3 = np.load('./query_time50/60australian30_big_metadata50.npy')

    metadata = np.vstack((metadata1, metadata2, metadata3))
    n_samples = metadata.shape[0]
    index = np.arange(n_samples)
    np.random.shuffle(index)

    X_train = metadata[index, 0:396]
    y_train = metadata[index, 396]

    X = metadata[:, 0:396]
    y = metadata[:, 396]

    automl = autosklearn.regression.AutoSklearnRegressor(
        # time_left_for_this_task=120,
        # per_run_time_limit=30,
        tmp_folder='./tmp1/autosklearn_regression_example_tmp/',
        output_folder='./tmp1/autosklearn_regression_example_out/',
        n_jobs=8,
        delete_output_folder_after_terminate=False,
        delete_tmp_folder_after_terminate=False,
        time_left_for_this_task=3600*2,
        ml_memory_limit=1024*10,
    )
    automl.fit(X_train, y_train, dataset_name='australian_big_qt50_lr')
    joblib.dump(automl, './automl.joblib')
    print('='*10)
    print(automl.show_models())
    print('='*10)

    # print(automl.)
    predictions = automl.predict(X)
    print("R2 score:", sklearn.metrics.r2_score(y, predictions))

    # sgdr = SGDRegressor()
    # sgdr.fit(X_train, y_train)
    # sgd_pred = sgdr.predict(X)
    # print("SGD R2 score:", sklearn.metrics.r2_score(y, sgd_pred))


if __name__ == '__main__':
    main()
