import numpy as np
import pandas as pd 
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.utils import np_utils
import tensorflow as tf
from MB_nn import MB_nn
from keras.utils.np_utils import to_categorical  
from sklearn.metrics import classification_report

# MB_true = {
#             '2':[4],
#             '4': [1, 2, 3, 5],
#             '5':[0,4,6,3],
#             '6':[5,3,35,34],
#             '8':[7,34],
#             '11':[10,34],
#            '14': [13,33,36,35,12,20,32 ],
#            '19':[18,31,20,23],
#            '24': [23,17,31,25,30,22,29,16],
#             '25':[24,29,16],
#            '29': [26,28,25,30,16,24],
#            '30': [16,17,31,15,29,24,32],
#            '33': [14,12,20,32,34],
#            '34': [33,7,10,35,8,11,9],
#            '35':[36,34,6,14],
#     }

# MB_pre = {'2': [4], 
        # '4': [1, 2, 3, 5], 
        # '5': [0, 3, 4, 6], 
        # '6': [34, 3, 35, 4, 5], 
        # '8': [34, 7], 
        # '11': [34, 10], 
        # '14': [32, 33, 35, 36], 
        # '19': [18, 20, 23, 31], 
        # '24': [16, 17, 23, 25, 29, 30, 31], 
        # '25': [24, 16, 29], 
        # '29': [24, 25, 26, 28, 30], 
        # '30': [17, 24, 31, 29, 15], 
        # '33': [32, 34, 14], 
        # '34': [33, 35, 6, 7, 8, 9, 10, 11], 
        # '35': [34, 36, 6, 14] }

def load_pred_MB(filename):
    MB_pred = pd.read_csv(filename)
    MB_dict = {}
    for col in MB_pred:
        MB_dict[col] = [str(int(i)) for i in list(MB_pred[col]) if not np.isnan(i) ]  
    return MB_dict   # return {'2':[1,2,4], '4':[1,5,8]}

# Like 'Alarm.csv' and numerical column names 

def load_mb_data(dataset, target, MB_dict):
    df = pd.read_csv(dataset)
    if target in MB_dict:
        MB = MB_dict[target]
    else:
        print('Target Input Error!')
        sys.exit()
    X = df.loc[:, MB] 
    encoder = OneHotEncoder(sparse=False) 
    X = encoder.fit_transform(X) # X = pd.get_dummy(X)
    y = df[target]
    encoder = LabelEncoder() 
    y = encoder.fit_transform(y)
    # Convert integers to dummy variables (i.e. one hot encoded)
    y = np_utils.to_categorical(y)
    return X, y 

    # Original dataset "ALARM.csv"
def load_all_data(target,  data_name = 'ALARM.csv'):
    #https://stackoverflow.com/questions/43515877/should-binary-features-be-one-hot-encoded
    data_path = 'DATASET/' + data_name
    df = pd.read_csv( data_path, index_col=False ) 

    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].map({True: 1, False: 0})   
    y = df.pop(target)
    encoder = LabelEncoder() 
    y = encoder.fit_transform(y)
    y = np_utils.to_categorical(y)

    encoder = OneHotEncoder(sparse=False) 
    X = encoder.fit_transform(df) 
    return X, y 

# '4' : 'LVEDVOLUME'
# '5' : 'LVFAILURE'  
# '6' : 'STROKEVOLUME'
# '14' : 'TPR'
# '24' : 'INTUBATION'
# '29' : 'VENTTUBE'
# '30': 'VENTLUNG'
#  '34' : 'HR'
def main(flag_MB = 0):

    if flag_MB == 1:
        # MB features as input  
        MB_dict = load_pred_MB('Pred_MB_MBOR.csv')
        # 'ALARM_SAMPLES.csv': numerical columns and rows
        X, y = load_mb_data('ALARM_SAMPLES.csv' ,'34', MB_dict)
        name = 'MB'
    else:
        # All features as input  
        X, y = load_all_data('HR')
        name = 'All'

    input_shape = X.shape[1]
    num_class = len(y[0])
    res = []
    no_epochs = 200


    seed = 15
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= seed)
    print(X_train)
    print(y_test.shape)
    kf = KFold(n_splits = 10, random_state = seed, shuffle = True )

#https://www.kaggle.com/questions-and-answers/236902

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):

        train_X, val_X = X_train[train_idx], X_train[val_idx]
        train_Y, val_Y = y_train[train_idx], y_train[val_idx]

        MB_model = MB_nn(input_shape , num_class) 

        MB_model.assign_data(train_X, train_Y, val_X , val_Y, X_test, y_test)

        initial_weights = MB_model.model.get_weights()
        optim = tf.keras.optimizers.Adam()
        MB_model.train(no_epochs, optim)

        # Choose the best weights on the validation data from 10 fold results 
        MB_model.model.set_weights(MB_model.best_weights)

        y_pred = MB_model.model.predict(MB_model.X_test)

        ess = tf.keras.losses.CategoricalCrossentropy()
        Entropy_Loss = ess(y_test, y_pred).numpy()
        res.append([fold, Entropy_Loss])
        print (fold + 1, Entropy_Loss)


        y_pred = np.argmax(y_pred, axis=1)
        y_test_temp = np.argmax(y_test, axis=1)
        report = classification_report(y_test_temp, y_pred, output_dict=True)

        df_classification_report = pd.DataFrame(report).transpose()
        #df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)
        print(df_classification_report)
 
        MB_model.model.reset_states()


    df_results = pd.DataFrame(res, columns = ['Run', 'Entropy_Loss'])
    df_results.to_csv(f'{name}.csv', index = False)

    print (df_results['Entropy_Loss'].mean())
    print (df_results['Entropy_Loss'].std()) 
    
if __name__ == "__main__":
    main(0)