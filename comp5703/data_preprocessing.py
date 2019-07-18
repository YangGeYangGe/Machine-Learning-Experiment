import numpy as np
# import datetime
import pandas as pd
#read file
import talib
from pandas.api.types import is_numeric_dtype




def read_file(file):
    
    '''
    Args:
        file: CSV data file of a currency pair. The format need to like the following:
        
        <TICKER>,<DTYYYYMMDD>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>
        AUDUSD,20010102,230100,0.5617,0.5617,0.5617,0.5617,4
        ......
        ......
    Return:
        Pandas dataframe which looks like the following
            <DTYYYYMMDD>	<TIME>	<OPEN>	<HIGH>	<LOW>	<CLOSE>	<DATETIME>
        0	20010102	   230100	0.5617	0.5617	0.5617	0.5617	2001-01-02 23:01:00
        1 ...........
        2 ...........
    '''
    data = pd.read_csv(file)
    is_numeric_dtype(data["<TIME>"])
    data["<TICKER>"] = data["<TICKER>"].astype(str)
    data["<TIME>"] = data["<TIME>"].astype(str)
    data["<DTYYYYMMDD>"] = data["<DTYYYYMMDD>"].astype(str)
    data["<OPEN>"] = data["<OPEN>"].astype(float)
    data["<HIGH>"] = data["<HIGH>"].astype(float)
    data["<LOW>"] = data["<LOW>"].astype(float)
    data["<CLOSE>"] = data["<CLOSE>"].astype(float)
    data["<VOL>"] = data["<VOL>"].astype(int)
    data=data.drop(['<TICKER>'],axis = 1)
    data=data.drop(['<VOL>'],axis = 1)
    #ensure all values in <TIME> in right H%M%S format
    data['<TIME>'] = data['<TIME>'].str.zfill(6)
    #add date to time
    data["<DATETIME>"] = data["<DTYYYYMMDD>"] + data["<TIME>"]
    # convert to right datetime format
    data["<DATETIME>"] = pd.to_datetime(data["<DATETIME>"], format='%Y%m%d%H%M%S')
    return data

def Nodate_HLOC(data,time_frame):
    
    '''
    Args:
        data: A pandas dataframe returned by function read_file().
        time_frame: The time frame you want. Usually "1D", "4H", "1H", "15MIN"
        
    Return:
        A pandas dataframe based on the timeframe.
        If the time_frame is set to "1D" (1 Day), this function return like the following:
        
                <HIGH>	<LOW>	<OPEN>	<CLOSE>
        <DATETIME>				
        2001-01-02	0.5622	0.5614	0.5617	0.5622
        2001-01-03	0.5652	0.5547	0.5618	0.5572
        2001-01-04	0.5695	0.5554	0.5569	0.5687
        ................
    '''
    data_high = data.groupby(pd.Grouper(key='<DATETIME>', freq=time_frame)).agg({'<HIGH>' : ['max']})
    data_low = data.groupby(pd.Grouper(key='<DATETIME>', freq=time_frame)).agg({'<LOW>' : ['min']})
    data_open = data.groupby(pd.Grouper(key='<DATETIME>', freq=time_frame)).agg({'<OPEN>' : ['first']})
    data_close = data.groupby(pd.Grouper(key='<DATETIME>', freq=time_frame)).agg({'<CLOSE>' : ['last']})
    c = pd.merge(data_high, data_low, left_index=True, right_index=True)
    d = pd.merge(c, data_open, left_index=True, right_index=True)
    e = pd.merge(d, data_close, left_index=True, right_index=True)
    e.columns = ['<HIGH>', '<LOW>', '<OPEN>','<CLOSE>']
    f = e.dropna()
    
    return f


def label_data(final_data,pattern,during_before,during_later):
    
    '''
    This function is used in function set_features()
    
    Args:
        final_data: A pandas dataframe based on a timeframe, returned by function Nodate_HLOC()
        
        pattern: The pattern 
        Pattern has to be one of the following:
            'HAMMER'
            'HANGINGMAN'
            'GRAVESTONEDOJI'
            'BELTHOLD'
            'CLOSINGMARUBOZU'
            'HIGHWAVE'
            'HIKKAKE'
            'LONGLEGGEDDOJI'
            'LONGLINE'
            'SHORTLINE'
            'SPINNINGTOP'
            'DOJI'
        
        during_before: label window size
        
        during_later: labe window size, usually same as during_before
        
    Return:
        labeled: labeled all the data. -1 for points where pattern not happen; 1 for up trend,
        0 for down trend and stable.
        
        pattern_index: The index of data points where pattern happen.

        pat: the output of ta-lib recognition function. 0 for points where pattern not happen,
        100 and -100 for points where pattern happen.
        
    '''

    labeled = np.full((len(final_data),), -1.0)

    pattern_index = None

    if pattern == 'HAMMER':
        pat = talib.CDLHAMMER(final_data["<OPEN>"],final_data["<HIGH>"],final_data["<LOW>"],final_data["<CLOSE>"])
        pattern_index = list(np.where(pat != 0))[0]
    elif pattern == 'HANGINGMAN':
        pat = talib.CDLHANGINGMAN(final_data["<OPEN>"],final_data["<HIGH>"],final_data["<LOW>"],final_data["<CLOSE>"])
        pattern_index = list(np.where(pat != 0))[0]
    elif pattern == 'GRAVESTONEDOJI':
        pat = talib.CDLGRAVESTONEDOJI(final_data["<OPEN>"],final_data["<HIGH>"],final_data["<LOW>"],final_data["<CLOSE>"])
        pattern_index = list(np.where(pat != 0))[0]
    elif pattern == 'BELTHOLD':      
        pat = talib.CDLBELTHOLD(final_data["<OPEN>"],final_data["<HIGH>"],final_data["<LOW>"],final_data["<CLOSE>"])
        pattern_index = list(np.where(pat != 0))[0]
    elif pattern == 'CLOSINGMARUBOZU':
        pat = talib.CDLCLOSINGMARUBOZU(final_data["<OPEN>"],final_data["<HIGH>"],final_data["<LOW>"],final_data["<CLOSE>"])
        pattern_index = list(np.where(pat != 0))[0]
    elif pattern == 'HIGHWAVE':
        pat = talib.CDLHIGHWAVE(final_data["<OPEN>"],final_data["<HIGH>"],final_data["<LOW>"],final_data["<CLOSE>"])
        pattern_index = list(np.where(pat != 0))[0]
    elif pattern == 'HIKKAKE':
        pat = talib.CDLHIKKAKE(final_data["<OPEN>"],final_data["<HIGH>"],final_data["<LOW>"],final_data["<CLOSE>"])
        pattern_index = list(np.where(pat != 0))[0]
    elif pattern == 'LONGLEGGEDDOJI':
        pat = talib.CDLLONGLEGGEDDOJI(final_data["<OPEN>"],final_data["<HIGH>"],final_data["<LOW>"],final_data["<CLOSE>"])
        pattern_index = list(np.where(pat != 0))[0]     
    elif pattern == 'LONGLINE':
        pat = talib.CDLLONGLINE(final_data["<OPEN>"],final_data["<HIGH>"],final_data["<LOW>"],final_data["<CLOSE>"])
        pattern_index = list(np.where(pat != 0))[0]
    elif pattern == 'SHORTLINE':
        pat = talib.CDLSHORTLINE(final_data["<OPEN>"],final_data["<HIGH>"],final_data["<LOW>"],final_data["<CLOSE>"])
        pattern_index = list(np.where(pat != 0))[0]
    elif pattern == 'SPINNINGTOP':
        pat = talib.CDLSPINNINGTOP(final_data["<OPEN>"],final_data["<HIGH>"],final_data["<LOW>"],final_data["<CLOSE>"])
        pattern_index = list(np.where(pat != 0))[0]
    elif pattern == 'DOJI':
        pat = talib.CDLDOJI(final_data["<OPEN>"],final_data["<HIGH>"],final_data["<LOW>"],final_data["<CLOSE>"])
        pattern_index = list(np.where(pat != 0))[0]
        
        
    for i in pattern_index:
        labeled[i] = 0
        if i < (len(final_data)-during_later): 
            change = (final_data["<CLOSE>"][i+1:i+during_later+1]).mean() - (final_data["<CLOSE>"][i-during_before:i]).mean()
            if (change > 0):
                labeled[i] = 1
            else:
                labeled[i] = 0 
            
    return labeled, pattern_index,pat


def set_features(dataset,pattern,feature_list, window):
    '''
    Args:
        dataset: A pandas dataframe, it is based on timeframe, returned by function Nodate_HLOC().
        
        pattern: pattern number
        
        feature_list: indicators or raw ohlc data needed in training
        
        window: label window size.
        
    Return:
        features: All the data (features, X) needed for  training. 
        
        y: All the labels (target) corresponding to features
        
        new_test_df: The whole dataframe.
        
    '''
    
    
    dataset['50day MA'] = dataset['<CLOSE>'].shift(1).rolling(window = 50).mean()
    dataset['Std_dev']= dataset['<CLOSE>'].rolling(5).std()
    dataset['RSI'] = talib.RSI(dataset['<CLOSE>'].values, timeperiod = 9)
    dataset['Williams %R'] = talib.WILLR(dataset['<HIGH>'].values, dataset['<LOW>'].values, dataset['<CLOSE>'].values, 7)
    dataset['50EMA'] = talib.EMA(dataset['<CLOSE>'], timeperiod=50)
    
    macd, macdsignal, macdhist = talib.MACD(dataset['<CLOSE>'], fastperiod=12, slowperiod=26, signalperiod=9)
    dataset['MACD'] = macd
    dataset['SAR'] = talib.SAR(dataset['<HIGH>'], dataset['<LOW>'], acceleration=0, maximum=0)
    
    #BBANDS
    upperband, middleband, lowerband = talib.BBANDS(dataset['<LOW>'], timeperiod=30, nbdevup=2, nbdevdn=2, matype=0)
    dataset['upperband'] = upperband
    dataset['middleband'] = middleband
    dataset['lowerband'] = lowerband
    
    #Stochastic
    slowk, slowd = talib.STOCH(dataset['<HIGH>'], dataset['<LOW>'], dataset['<CLOSE>'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    dataset['STOCHslowk'] = slowk
    dataset['STOCHslowd'] = slowd
    
    #ADX
    adx = talib.ADX(dataset['<HIGH>'], dataset['<LOW>'], dataset['<CLOSE>'], timeperiod=14)
    dataset['ADX'] = adx
    
    #CCI
    cci = talib.CCI(dataset['<HIGH>'], dataset['<LOW>'], dataset['<CLOSE>'], timeperiod=14)
    dataset['CCI'] = cci
    
    #ADXR
    real = talib.ADXR(dataset['<HIGH>'], dataset['<LOW>'], dataset['<CLOSE>'], timeperiod=14)
    dataset['ADXR'] = real
    
    real = talib.APO(dataset['<CLOSE>'], fastperiod=12, slowperiod=26, matype=0)
    dataset['APO'] = real
    
    aroondown, aroonup = talib.AROON(dataset['<HIGH>'], dataset['<LOW>'],timeperiod=14)
    dataset['AROONDOWN'] = aroondown
    dataset['AROONUP'] = aroonup
    
    real = talib.AROONOSC(dataset['<HIGH>'], dataset['<LOW>'],timeperiod=14)
    dataset['AROONOSC'] = real
    
    real = talib.BOP(dataset['<OPEN>'], dataset['<HIGH>'], dataset['<LOW>'], dataset['<CLOSE>'])
    dataset['BOP'] = real
    
    real = talib.CMO(dataset['<CLOSE>'], timeperiod=14)
    dataset['CMO'] = real
    
    real = talib.DX(dataset['<HIGH>'], dataset['<LOW>'], dataset['<CLOSE>'], timeperiod=14)
    dataset['DX'] = real
    
    acd, macdsignal, macdhist = talib.MACDEXT(dataset['<CLOSE>'], fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
    dataset['ACDEXT'] = acd
    dataset['MACDSIGNALEXT'] = macdsignal
    dataset['MACDHISTEXT'] = macdhist
    
    macd, macdsignal, macdhist = talib.MACDFIX(dataset['<CLOSE>'], signalperiod=9)
    dataset['MACDFIX'] = macd
    dataset['MACDSIGNALFIX'] = macdsignal
    dataset['MACDHISTFIX'] = macdhist
    
    real = talib.MINUS_DI(dataset['<HIGH>'], dataset['<LOW>'], dataset['<CLOSE>'], timeperiod=14)
    dataset['MINUS_DI'] = real
    
    real = talib.MINUS_DM(dataset['<HIGH>'], dataset['<LOW>'], timeperiod=14)
    dataset['MINUS_DM'] = real
    
    real = talib.MOM(dataset['<CLOSE>'], timeperiod=10)
    dataset['MOM'] = real
    
    real = talib.PLUS_DI(dataset['<HIGH>'], dataset['<LOW>'], dataset['<CLOSE>'], timeperiod=14)
    dataset['PLUS_DI'] = real
    
    real = talib.PLUS_DM(dataset['<HIGH>'], dataset['<LOW>'], timeperiod=14)
    dataset['PLUS_DM'] = real
    
    real = talib.PPO(dataset['<CLOSE>'], fastperiod=12, slowperiod=26, matype=0)
    dataset['PPO'] = real
    
    real = talib.ROC(dataset['<CLOSE>'], timeperiod=10)
    dataset['ROC'] = real
    
    real = talib.ROCP(dataset['<CLOSE>'], timeperiod=10)
    dataset['ROCP'] = real
    
    real = talib.ROCR(dataset['<CLOSE>'], timeperiod=10)
    dataset['ROCR'] = real
    
    real = talib.ROCR100(dataset['<CLOSE>'], timeperiod=10)
    dataset['ROCR100'] = real
    
    fastk, fastd = talib.STOCHF(dataset['<HIGH>'], dataset['<LOW>'], dataset['<CLOSE>'], fastk_period=5, fastd_period=3, fastd_matype=0)
    dataset['STOCHFFASTK'] = fastk
    dataset['STOCHFFASTD'] = fastd
    
    fastk1, fastd1 = talib.STOCHRSI(dataset['<CLOSE>'], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    dataset['STOCHRSIFASTK'] = fastk1
    dataset['STOCHRSIFASTD'] = fastd1
    
    real = talib.TRIX(dataset['<CLOSE>'], timeperiod=30)
    dataset['TRIX'] = real
    
    real = talib.ULTOSC(dataset['<HIGH>'], dataset['<LOW>'], dataset['<CLOSE>'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
    dataset['ULTOSC'] = real
    
    #DEMA
    dataset['DEMA'] = talib.DEMA(dataset['<CLOSE>'], timeperiod=30)

    #HT_TRENDLINE - Hilbert Transform - Instantaneous Trendlin
    
    dataset['HT_TRENDLINE'] = talib.HT_TRENDLINE(dataset['<CLOSE>'])
    
    #KAMA
    dataset['KAMA'] = talib.KAMA(dataset['<CLOSE>'], timeperiod=30)

#     MIDPOINT - MidPoint over period
    dataset['MIDPOINT']  = talib.MIDPOINT(dataset['<CLOSE>'], timeperiod=14)
#     MIDPRICE - Midpoint Price over period
    dataset['MIDPRICE'] = talib.MIDPRICE(dataset['<HIGH>'], dataset['<LOW>'], timeperiod=14)

#     SAREXT - Parabolic SAR - Extended
    dataset['SAREXT']  = talib.SAREXT(dataset['<HIGH>'], dataset['<LOW>'], startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)
    
#     SMA - Simple Moving Average
    dataset['SMA'] = talib.SMA(dataset['<CLOSE>'], timeperiod=30)
    
#     T3 - Triple Exponential Moving Average (T3)
# NOTE: The T3 function has an unstable period.

    dataset['T3'] = talib.T3(dataset['<CLOSE>'], timeperiod=5, vfactor=0)

#     TEMA - Triple Exponential Moving Average
    dataset['TEMA'] = talib.TEMA(dataset['<CLOSE>'], timeperiod=30)
    
#     TRIMA - Triangular Moving Average
    dataset['TRIMA'] = talib.TRIMA(dataset['<CLOSE>'], timeperiod=30)
    
#     WMA - Weighted Moving Average
    dataset['WMA'] = talib.WMA(dataset['<CLOSE>'], timeperiod=30)
    
    
    test_df = dataset.copy()
    new_test_df = test_df.dropna()
    
    la, idx, all_idx= label_data(new_test_df,pattern,window,window)
    
    new_test_df["pattern"] = all_idx
    
    new_test_df["label"] = la
    new_test_df["label"] = new_test_df["label"].astype(int)
    
    new_test_df.loc[new_test_df['pattern'] != 0]

    
    features = new_test_df.loc[new_test_df['label'] != -1][feature_list].values
    y = new_test_df.loc[new_test_df['label'] != -1]['label'].values
    return features, y,new_test_df



def get_XY_data(file, timeframe, pattern_name, window_size,feature_list):
    data = read_file(file)
    dataset =Nodate_HLOC(data,timeframe)
    features,y,all_df=set_features(dataset,pattern_name,feature_list,window_size)
    
    # To ensure the X and y have the same amount
    assert(y.shape[0] == features.shape[0])

    # To ensure only have two class in y
    assert(y.shape[0] == y[y==0].shape[0] + y[y==1].shape[0])
    return features, y

import itertools
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        pass
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# Compute confusion matrix
