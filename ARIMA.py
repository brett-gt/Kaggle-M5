import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf #TODO
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

#https://www.kaggle.com/brunoborges95/m5-time-series-forecasting-using-mapa-sarimax
#https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/133904

# https://people.duke.edu/~rnau/411arim3.htm

# https://people.duke.edu/~rnau/Slides_on_ARIMA_models--Robert_Nau.pdf

# Rule 6: If the PACF of the differenced series displays a sharp cutoff and/or the lag-1 autocorrelation 
# is positive--i.e., if the series appears slightly "underdifferenced"--then consider adding an AR term
#  to the model. The lag at which the PACF cuts off is the indicated number of AR terms.

#--------------------------------------------------------------------------------
def apply(data):
    print("Running ARIMA")

    # Order of terms is [AR, differencing, MA]
    # autoregressive (AR) - restoring force that tends to pull outliers towards mean.  Larger
    #               values return to mean more quickly
    # 
    # moving-average (MA) - has a period of "random shocks" whose effects are felt in two or more periods

    # Applying Seasonal ARIMA model to forcast the data 
    mod = sm.tsa.SARIMAX(data['data'], trend='c', order=(2,1,0), seasonal_order=(0,1,2,12))
    results = mod.fit()
    print(results.summary())

    data['forecast'] = results.predict(start = 1700, end= 1900, dynamic= False)  
    data[['data', 'forecast']].plot(figsize=(12, 8))
    plt.show()

    # Plot residual errors
    residuals = pd.DataFrame(results.resid)
    fig, ax = plt.subplots(1,2)
    residuals.plot(title="Residuals", ax=ax[0])
    residuals.plot(kind='kde', title='Density', ax=ax[1])
    plt.show()

    # Actual vs Fitted
    #mod = ARIMA(data['data'], (2,1,0))
    #results.plot_predict(1000, 1100, dynamic=False)
    #plt.show()

#--------------------------------------------------------------------------------
def find_AR(timeseries):
    """ Find the AR term by plotting partial autocorrelation function.  Looking for
        furthest out spikes in the plot, which is the order we want to use for the
        AR (p term) in the ARIMA model

        If ACF dies out gradually and PACF cuts off sharpy, AR signature
        if ACF cuts of sharpy and PACF dies out gradually, MA signature
    """
    plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

    fig, axes = plt.subplots(1, 2, sharex=False)
    axes[0].plot(timeseries.data.diff()); axes[0].set_title('1st Differencing')
    axes[1].set(ylim=(0,5))
    plot_pacf(timeseries.data.diff().dropna(), ax=axes[1])
    plt.show()


#--------------------------------------------------------------------------------
def find_differncing(timeseries):
    """ Differencing helps make the data set stationary.  Plot multiple levels
        of differencing to see if we can get a stationary signal (most points inside
        the significance bound).
    """
    # Original Series
    fig, axes = plt.subplots(3, 2, sharex=False)
    axes[0, 0].plot(timeseries.data); axes[0, 0].set_title('Original Series')
    plot_acf(timeseries.data, ax=axes[0, 1])

    # 1st Differencing
    axes[1, 0].plot(timeseries.data.diff()); axes[1, 0].set_title('1st Order Differencing')
    plot_acf(timeseries.data.diff().dropna(), ax=axes[1, 1])

    # 2nd Differencing
    axes[2, 0].plot(timeseries.data.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
    plot_acf(timeseries.data.diff().diff().dropna(), ax=axes[2, 1])

    plt.show()


#--------------------------------------------------------------------------------
def find_MA(timeseries):
    """ 
    """
    fig, axes = plt.subplots(1, 2, sharex=True)
    axes[0].plot(timeseries.data.diff()); axes[0].set_title('1st Differencing')
    axes[1].set(ylim=(0,1.2))
    plot_acf(timeseries.data.diff().dropna(), ax=axes[1])

    plt.show()