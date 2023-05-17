import pandas as pd 
import numpy as np
import random

def load_data(data_name = 'ALARM.csv'):
    # Keep the same in main function 

    seed = 2
    data_path = 'DATASET/' + data_name
    df = pd.read_csv(data_path, index_col=False)
    columns = df.columns
    print(columns)

    for col in columns:
        print(col +'\n' + str(df[col].value_counts()))

    df.rename(columns={x: y for x, y in zip(df.columns, range(0,len(df.columns)))}).to_csv('Alarm.csv')
    Mappings = {'ZERO':3, 'LOW':0,'NORMAL':1,'HIGH':2, 'ONESIDED':0, 'ESOPHAGEAL':2}
    
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].map({True: 1, False: 0})
        elif col =='CATECHOL' or col == 'SHUNT':
            df[col] = df[col].map({'NORMAL':0,'HIGH':1})
        else:
            df[col] = df[col].map(Mappings)    

    df = df.sample( n = 2000,  random_state = seed,  ignore_index = True )
    df = df.rename( columns={x: y for x, y in zip(df.columns, range(0,len(df.columns)))} )
    #df.to_csv( 'ALARM_SAMPLES.csv', index = False )
    df.to_csv('pyCausalFS-main/pyCausalFS/CBD/data/ALARM_SAMPLES.csv', index = False)
    return 

if __name__ == "__main__":
    load_data()

