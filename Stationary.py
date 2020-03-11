from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot as plt

# https://www.kaggle.com/sumi25/understand-arima-and-tune-p-d-q

##################################################################################
# Techniques to Make Stationary
##################################################################################
#---------------------------------------------------------------------------------
def sub_rolling_mean(data):
    """ Takes the log of the data and then removes a rolling average from each value
    """
    df_log = np.log(data)
    rolling_mean = df_log.rolling(window=12).mean()
    df_log_minus_mean = df_log - rolling_mean
    return df_log_minus_mean.dropna(inplace=True)



##################################################################################
# Techniques to Determine if Stationary
##################################################################################
#--------------------------------------------------------------------------------
def examine(timeseries, window_length = 28):
    test(timeseries)
    plot(timeseries, window_length)

#--------------------------------------------------------------------------------
def test(timeseries,cutoff = 0.001):
    """ Runs augmented Dickey-Fuller (ADF) tests to determine if stationary or not.
        Key is interpreting the p-value, which represents the probability the model
        has a unit root (we want it to reject that, so want small p-value).
    """
    result = adfuller(timeseries)
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    if(result[1] < cutoff):
        print("*** Appears stationary. ***")
    else:
        print("!!! Not stationary. !!!")

    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

#--------------------------------------------------------------------------------
def plot(timeseries, window_length = 28):   
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=window_length).mean()
    rolstd = timeseries.rolling(window=window_length).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()