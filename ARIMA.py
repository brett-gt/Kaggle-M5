import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot as plt

#https://www.kaggle.com/brunoborges95/m5-time-series-forecasting-using-mapa-sarimax
#https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/133904


#--------------------------------------------------------------------------------
def apply(data):
    print("Running ARIMA")

    # Applying Seasonal ARIMA model to forcast the data 
    #mod = sm.tsa.SARIMAX(data['data'], trend='c', order=(1,1,1), seasonal_order=(1,1,1,12))
    mod = sm.tsa.SARIMAX(data['data'], trend='c', order=(6,1,0), seasonal_order=(0,1,1,7))
    results = mod.fit()
    print(results.summary())

    data['forecast'] = results.predict(start = 100, end= 200, dynamic= True)  
    data[['data', 'forecast']].plot(figsize=(12, 8))
    plt.show()