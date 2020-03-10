"""Contest defines the use of Weighted Root Mean Squared Scaled Error (WRMSSE).
"""
# https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/133834

#evaluation test
#--------------------------------------------------------------------------------
def rmse(actual, predict, title="No title given"):
    from sklearn import metrics
    rmse = np.sqrt(metrics.mean_squared_error(actual, predict))
    print('The RMSE of ' + title + ' is:', rmse)

#--------------------------------------------------------------------------------
def calc_WRMSSE(val_data, output_data):
    print("Calculating RMSSE...")

#--------------------------------------------------------------------------------
def calc_RMSSE(val_data, output_data):
    print("Calculating RMSSE...")
        
