# -*- encoding: utf-8 -*-

import numpy as np 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

import autosklearn.classification

def main():

    metadata = np.load('./10_australian_big_metadata.npy')
    n_samples = metadata.shape[0]
    print("the num of the metadata: ", n_samples)
    index = np.arange(n_samples)
    np.random.shuffle(index)

    # X = metadata[:, 0:396]
    # y = metadata[:, 396]

    # X_train = metadata[index[0:1000], 0:396]
    # y_train = metadata[index[0:1000], 396]

    X_train = metadata[index, 0:396]
    y_train = metadata[index, 396]

    y_train[np.where(y_train>0)[0]] = 1
    y_train[np.where(y_train<=0)[0]] = -1



    autoclassifier = autosklearn.classification.AutoSklearnClassifier(
        tmp_folder='./tmp_all10_australian_metadata/autosklearn_classification_tmp/',
        output_folder='./tmp2_all10_australian_metadata/autosklearn_classification_output/',
        n_jobs=8,
        # delete_output_folder_after_terminate=False,
        # delete_tmp_folder_after_terminate=False,
        time_left_for_this_task=60*30,

    )

    autoclassifier.fit(X_train, y_train)
    autoclassifier.fit_ensemble(y_train)
    joblib.dump(autoclassifier, './all10_autoclassifier.joblib')

    print(autoclassifier.show_models())
    pred = autoclassifier.predict(X_train)
    print("Accuracy score: ", accuracy_score(y_train, pred))

    # lr = LogisticRegression()
    # lr.fit(X_train, y_train)
    # lr_pred = lr.predict(X_train)
    # print("LogisticRegression Accuracy score: ", accuracy_score(y_train, lr_pred))
    



if __name__ == "__main__":
    main()    