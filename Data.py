import pandas as pd
import numpy as np

#------------------------------------------------------------------------------

ID_COL = 'id'
LOOKUP_COL = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

VAR_COL = 'day'
VALUE_COL = 'demand'   

CAL_DROP_COLS = ['weekday', 'wday', 'month', 'year']

#--------------------------------------------------------------------------------
def merge_data_sets(data_path = "Data/", filename = "combined_data.pkl"):
    """ Merge the various csv files provided into a single data set.  Save
        result to a pickle so we can use it quickly later

        Arguments:
            data_path: path of the data to be read
            filename: name of the file to be read
    """
    print("merge_data_sets()")
    print("Reading sales_train_validation.csv...")
    sales = pd.read_csv(data_path + "sales_train_validation.csv")

    print("Melting sales data...")
    #data = data.drop(LOOKUP_COL, axis=1)
    sales = pd.melt(sales, id_vars = LOOKUP_COL, var_name = VAR_COL, value_name = VALUE_COL)
    print("Sales size = {}\nShape ={}\nShape[0] x Shape[1] = {}".
           format(sales.size, sales.shape, sales.shape[0]*sales.shape[1])) 
    print(sales.head())

    print("\n\nAdding calendar data...")
    print("Reading calendar.csv...")
    calendar = pd.read_csv(data_path + "calendar.csv")
    calendar.drop(CAL_DROP_COLS, inplace = True, axis = 1)
    print(calendar.head())

    print("\nMerging calendar prices with the data set...")
    sales = pd.merge(sales, calendar, how = 'left', left_on = [VAR_COL], right_on = ['d'])
    sales.drop(['d', 'day'], inplace = True, axis = 1)
    print(sales.head())
    del calendar

    print("\n\nAdding sales price...")
    print("Reading sell_price.csv...")
    sell_prices = pd.read_csv(data_path + "sell_prices.csv")
    print(sell_prices.head())

    print("\nMerging sell prices with the data set...")
    sales = pd.merge(sales, sell_prices, how = 'left', on = ['store_id', 'item_id', 'wm_yr_wk'])
    print(sales.head())
    del sell_prices

    sales = reduce_mem_usage(sales, True)

    print("\n\nSaving to a pickle...")
    sales.to_pickle(data_path + filename)

#--------------------------------------------------------------------------------
def augment_data(dataset):
    print("Augmenting data...")
    add_lag(dataset, VALUE_COL, 28)
    print(dataset.head())

#--------------------------------------------------------------------------------
def add_lag(data, source_col, lag_days):
    """ Add a shifted set of data to the dataframe.  Allows us to easily add relevant
        historic data (i.e. price 1 week or 1 month ago).  No idea why I made this a seperate
        function other than it looked cleaner.

        Arguments:
            data - dataframe to be acted on
            source_col - column to be lagged
            lag_days - number of days to lag the source_col values
    """
    print("Add_lag " + str(lag_days))
    col_name = 'lag_t' + str(lag_days)
    data[col_name] = data.groupby([ID_COL])[source_col].transform(lambda x: x.shift(lag_days))
 

#----------------------------------------------------------------------------
def add_rolling(data, source_col, lag_days, rolling_days, function = 'mean'):
    """ Add a historic data statistics to the dataframe.  Allows us to easily add relevant
        historic data (i.e. price 1 week or 1 month ago).  

        Arguments:
            data - dataframe to be acted on
            source_col - column to be lagged
            lag_days - number of days to lag the source_col values
            rolling_days - period over which function is applied
            function - string name of the function
    """
    print("add_rolling: lag=" + str(lag_days) + " rolling=" + str(rolling_days) + " function=" + function)
    dest_col = col_name = 'lag_t' + str(lag_days) + "_" + function
    if(function == 'mean'):
        data[dest_col] = data.groupby([ID_COL])[source_col].transform(lambda x: x.shift(lag_days).rolling(rolling_days).mean())
    elif(function == 'std'):
        data[dest_col] = data.groupby([ID_COL])[source_col].transform(lambda x: x.shift(lag_days).rolling(rolling_days).std())
    elif(function == 'skew'):
        data[dest_col] = data.groupby([ID_COL])[source_col].transform(lambda x: x.shift(lag_days).rolling(rolling_days).skew())
    elif(function == 'kurt'):
        data[dest_col] = data.groupby([ID_COL])[source_col].transform(lambda x: x.shift(lag_days).rolling(rolling_days).kurt())
    else:
        print("!!! add_rolling: Unidentified function !!!")

#--------------------------------------------------------------------------------
def read_data(data_path = "Data/", filename = "combined_data.pkl"):
    """ Read in a merged data set and return the dataframe

        Arguments:
            filename: name of the file to be read
    """
    print("Reading pickle: " + data_path + filename)
    return pd.read_pickle(data_path + filename)

#----------------------------------------------------------------------------
# Borrowed from: https://www.kaggle.com/mayer79/m5-forecast-keras-with-categorical-embeddings-v2
def reduce_mem_usage(data, verbose=True):
    """ Try to reduce some memory usage by smartly choosing data types
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = data.memory_usage().sum() / 1024**2    
    for col in data.columns:
        col_type = data[col].dtypes
        if col_type in numerics:
            c_min = data[col].min()
            c_max = data[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    data[col] = data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    data[col] = data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    data[col] = data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    data[col] = data[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    data[col] = data[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    data[col] = data[col].astype(np.float32)
                else:
                    data[col] = data[col].astype(np.float64)    
    end_mem = data.memory_usage().sum() / 1024**2

    if verbose: 
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return data
 
        
