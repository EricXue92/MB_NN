# https://stackoverflow.com/questions/72087719/shap-summary-plot-and-mean-values-displaying-together
# https://shap-lrjball.readthedocs.io/en/latest/generated/shap.DeepExplainer.html
import csv  
import os 
import shap
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder 
# from tensorflow.keras.utils import to_categorical
from MB_nn_last import MB_nn
from keras.utils import to_categorical  
from sklearn.metrics import classification_report
import statistics

def load_pred_MB(filename):
    filepath = 'Pre_MB_Dataset/' + filename 
    MB_pred = pd.read_csv(filepath)
    MB_dict = {}
    for col in MB_pred:
        MB_dict[col] = [str(int(i)) for i in list(MB_pred[col]) if not np.isnan(i) ]  
    return MB_dict   # return {'2':[1,2,4], '4':[1,5,8]}

# In DATASET_FOR_pyCausalFS file(alarm.csv, child.csv, insurance.csv)

def load_mb_data(dataset, target, MB_dict):

    filepath = 'Raw_DATASET/' + dataset
    df = pd.read_csv(filepath, index_col=False)
    mappings = { col_name: num for col_name, num in zip(df.columns, range(0,len(df.columns) ))} 
    target_num = mappings[target]
    MB_list_nums = MB_dict[str(target_num)]
    key_list = list(mappings.keys())
    val_list = list(mappings.values())
    MB_list_names =[ key_list[ val_list.index(int(i)) ] for i in MB_list_nums ]
    X = df.loc[:, MB_list_names]
    # recording =  { x: y for x, y in zip(X.columns, range(0,len(X.columns)))}
    column_names = X.columns
    print(column_names)

    # One-hot encoding for X
    X = X.astype(str)
    X = pd.get_dummies(X)

    X = X.to_numpy()


    y = df[target]
    #https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
    print(y.value_counts())
    num_class = len( y.value_counts() )
    encoder = LabelEncoder() 
    y = encoder.fit_transform(y)
    # Convert integers to dummy variables (i.e. one hot encoded)
    y = to_categorical(y)

    return X, y, column_names, num_class

def load_real_MB(dataset, MB_list, target):
    filepath = 'Raw_DATASET/' + dataset
    df = pd.read_csv(filepath, index_col=False)
    X = df.loc[:, MB_list]
    column_names = X.columns
    print(column_names)
    X = X.to_numpy()
    # encoder = OneHotEncoder(sparse=False) 
    # X = encoder.fit_transform(X) 
    y = df[target]
    num_class = len(y.value_counts())
    encoder = LabelEncoder() 
    y = encoder.fit_transform(y)

    # Convert integers to dummy variables (i.e. one hot encoded)
    y = to_categorical(y)
    return X, y, column_names, num_class

def load_all_data(target,  data_name):
    data_path = 'Raw_DATASET/' + data_name
    df = pd.read_csv(data_path, index_col=False)
    y = df.pop(target)
    column_names = df.columns
    num_class = len(y.value_counts())
    encoder = LabelEncoder() 
    y = encoder.fit_transform(y)

    y = to_categorical(y)
    columns_names = df.columns
    X = df.to_numpy()
    return X, y, column_names,num_class

def main(flag_MB):
    target = 'CarValue'
    # MB_list = ['ThisCarDam','RuggedAuto', 'Mileage','Antilock', 'Cushioning','Age', 'DrivQuality','OtherCarCost' , 'MedCost','ILiCost']
   
    if flag_MB == 1:
        
        MB_dict = load_pred_MB('Pred_insurance.csv')

        X, y, column_names, num_class = load_mb_data('insurance.csv', target, MB_dict)

    elif flag_MB  == 0:
        X, y, column_names, num_class = load_all_data(target, 'alarm.csv')

    else:
        X, y, column_names, num_class = load_real_MB('insurance.csv', MB_list, target )

    input_shape = X.shape[1]
    no_epochs = 200
    seed = 9
    res = {'precision':[], 'recall':[], 'f1-score':[]}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= seed)
    kf = KFold(n_splits = 5, random_state = seed, shuffle = True )
    tmp_acc = 0
    test_loss = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):

        train_X, val_X = X_train[train_idx], X_train[val_idx]
        train_Y, val_Y = y_train[train_idx], y_train[val_idx]

        MB_model = MB_nn( input_shape, num_class ) 
        MB_model.assign_data( train_X, train_Y, val_X , val_Y, X_test, y_test )

        initial_weights = MB_model.model.get_weights()
        optim = tf.keras.optimizers.legacy.Adam()
        MB_model.train(no_epochs, optim)

        # Choose the best weights on the validation data from 10 fold results 
        MB_model.model.set_weights(MB_model.best_weights)

        y_pred = MB_model.model.predict(MB_model.X_test)
        MB_model.model.reset_states()
        # print(np.mean(y_pred, axis = 0))

        ess = tf.keras.losses.CategoricalCrossentropy()
        Entropy_Loss = ess(y_test, y_pred).numpy()
        #print(Entropy_Loss)

        # tf.print('------------')
        # tf.print(y_test)
        # tf.print(y_pred)
        # tf.print('------------')

        test_loss.append(Entropy_Loss)

        y_pred = np.argmax(y_pred, axis=1)
        y_test_temp = np.argmax(y_test, axis=1)

        report = classification_report(y_test_temp, y_pred, output_dict=True)
        df_classification_report = pd.DataFrame(report).transpose()


        res['precision'].append(df_classification_report.loc['weighted avg',  'precision'])
        res['recall'].append(df_classification_report.loc['weighted avg',  'recall'])
        res['f1-score'].append(df_classification_report.loc['weighted avg',  'f1-score'])


        # df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)

        # Plot the better accuracy 
        if res['precision'][fold] > tmp_acc:
            tmp_acc = res['precision'][fold]
            # Explanation (Compute SHAP values)
            shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
            # select a set of background examples to takr an expectation over
            explainer = shap.DeepExplainer(MB_model.model, MB_model.train_X ) 
            shap_values = explainer.shap_values(MB_model.val_X)
            fig = plt.gcf()
            shap.summary_plot(shap_values, MB_model.val_X , max_display = MB_model.val_X.shape[1],
               feature_names=column_names)
            # shap.summary_plot(shap_values, MB_model.val_X , max_display = MB_model.val_X.shape[1],
            #     feature_names= column_names)
            fig.savefig('scratch.pdf', bbox_inches='tight')

    res = pd.DataFrame(data = res)

    # row = [ f'{target}', round(res['precision'].mean(), 3), round(res['recall'].mean(),3), round(res['f1-score'].mean(),3), 
    # round(res['precision'].std(),3), round(res['recall'].std(),3),  round(res['f1-score'].std(),3) ]
    

    # file_MB = 'results/MB_insurance.csv'
    # file_non_MB = 'results/insurance.csv'
    # header = ['Target', 'Precision_Mean', 'Recall_Mean','F1-score_Mean',
    # 'Precision_std','Recall_std','F1-score_std',]

    # if flag_MB == 1:
    #     if os.path.exists(file_MB): 
    #         with open(file_MB, 'a+', encoding='UTF8') as f:
    #             writer = csv.writer(f)
    #             writer.writerow(row)
    #     else:
    #         with open(file_MB, 'a+', encoding='UTF8') as f:
    #             writer = csv.writer(f)
    #             writer.writerow(header)
    #             writer.writerow(row)
    # else:
    #     if os.path.exists(file_non_MB):
    #         with open(file_non_MB, 'a+', encoding='UTF8') as f:
    #             writer = csv.writer(f)
    #             writer.writerow(row)
    #     else:
    #         with open(file_non_MB, 'a+', encoding='UTF8') as f:
    #             writer = csv.writer(f)
    #             writer.writerow(header)
    #             writer.writerow(row)

    print(test_loss)

if __name__ == "__main__":
    main(1)