from statsmodels.tsa.stattools import adfuller
from matplotlib import pyplot as plt

#--------------------------------------------------------------------------------
def examine(timeseries, window_length = 28):
    test(timeseries)
    plot(timeseries, window_length)

#--------------------------------------------------------------------------------
def test(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
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