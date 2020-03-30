import pandas as pd
import numpy as np
import DataExam
from sklearn import preprocessing, metrics
import gc

#------------------------------------------------------------------------------

ID_COL = 'id'
LOOKUP_COL = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
VAR_COL = 'day'
VALUE_COL = 'demand'   

# These we can add later using built in python functions
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
    print('Melted sales data has {} rows and {} columns'.format(sales.shape[0], sales.shape[1]))
    

    sales = extend(sales) #Add the d_cols for the evaluation data sets
    #TODO: Temporarily dropping the eval data
    sales = sales[sales['part'] != 'test2']
    #END TODO
    print('Extended sales data has {} rows and {} columns'.format(sales.shape[0], sales.shape[1]))

    print("\n\nAdding calendar data...")
    print("Reading calendar.csv...")
    calendar = pd.read_csv(data_path + "calendar.csv")
    calendar.drop(CAL_DROP_COLS, inplace = True, axis = 1) 
    print(calendar.head())
    print('Calendar data has {} rows and {} columns'.format(sales.shape[0], sales.shape[1]))

    print("\nMerging calendar prices with the data set...")
    sales = pd.merge(sales, calendar, how = 'left', left_on = [VAR_COL], right_on = ['d'])
    sales.drop(['d', 'day'], inplace = True, axis = 1)
    print('Merged sales and calendar data has {} rows and {} columns'.format(sales.shape[0], sales.shape[1]))
    del calendar

    print("\n\nAdding sales price...")
    print("Reading sell_price.csv...")
    sell_prices = pd.read_csv(data_path + "sell_prices.csv")

    print("\nMerging sell prices with the data set...")
    sales = pd.merge(sales, sell_prices, how = 'left', on = ['store_id', 'item_id', 'wm_yr_wk'])

    DataExam.date_summary(sales)
    print('Merged sales, calendar, and price data has {} rows and {} columns'.format(sales.shape[0], sales.shape[1]))
    del sell_prices

    sales = reduce_mem_usage(sales, True)

    print("\n\nSaving to a pickle " + data_path + filename)
    sales.to_pickle(data_path + filename)

    gc.collect()
    return sales

#--------------------------------------------------------------------------------
def subset(data, nrows=55000000):
    print("\n\nData.subset()\nGrabbing subset number of rows...")
    start_mem = data.memory_usage().sum() / 1024**2   
    data = data.loc[nrows:]
    end_mem = data.memory_usage().sum() / 1024**2
    DataExam.date_summary(data)
    print('Subset data has {} rows and {} columns'.format(data.shape[0], data.shape[1]))
    print('Subset: Mem. usage decreased to {:5.2f} Mb ({:.1f}% increase)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return data
    
#--------------------------------------------------------------------------------
def seperate(data, end_train_date, end_val_date):
    print("\n\nSeperating test, train, and val data...")
    DataExam.date_summary(data)

    x_train = data[data['date'] <= end_train_date]
    y_train = x_train['demand']
    x_val = data[(data['date'] > end_train_date) & (data['date'] <= end_val_date)]
    y_val = x_val['demand']
    test = data[(data['date'] > end_val_date)]
    print('Train data has {} rows and {} columns'.format(x_train.shape[0], x_train.shape[1]))
    print('Val data has {} rows and {} columns'.format(x_val.shape[0], x_val.shape[1]))
    print('Test data has {} rows and {} columns'.format(test.shape[0], test.shape[1]))
    return x_train, y_train, x_val, y_val, test

    
#--------------------------------------------------------------------------------
def extend(data):
    """ Extends our data set to include d_cols relating to the evaluation data.
        Also, make sure only working with rows that are in the submission file.
    """
    #TODO: Make variable
    submission = pd.read_csv('Data/sample_submission.csv')

    # Grab items from the submission file incase there are extras in the training and aux data
    test1_rows = [row for row in submission['id'] if 'validation' in row]
    test2_rows = [row for row in submission['id'] if 'evaluation' in row]
    test1 = submission[submission['id'].isin(test1_rows)]
    test2 = submission[submission['id'].isin(test2_rows)]

    # change column names
    test1.columns = ['id', 'd_1914', 'd_1915', 'd_1916', 'd_1917', 'd_1918', 'd_1919', 'd_1920', 'd_1921', 'd_1922', 'd_1923', 'd_1924', 'd_1925', 'd_1926', 'd_1927', 'd_1928', 'd_1929', 'd_1930', 'd_1931', 
                      'd_1932', 'd_1933', 'd_1934', 'd_1935', 'd_1936', 'd_1937', 'd_1938', 'd_1939', 'd_1940', 'd_1941']

    test2.columns = ['id', 'd_1942', 'd_1943', 'd_1944', 'd_1945', 'd_1946', 'd_1947', 'd_1948', 'd_1949', 'd_1950', 'd_1951', 'd_1952', 'd_1953', 'd_1954', 'd_1955', 'd_1956', 'd_1957', 'd_1958', 'd_1959', 
                      'd_1960', 'd_1961', 'd_1962', 'd_1963', 'd_1964', 'd_1965', 'd_1966', 'd_1967', 'd_1968', 'd_1969']

    # get product table
    product = data[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']].drop_duplicates()
    
    # merge with product table
    test2['id'] = test2['id'].str.replace('_evaluation','_validation')
    test1 = test1.merge(product, how = 'left', on = 'id')
    test2 = test2.merge(product, how = 'left', on = 'id')
    test2['id'] = test2['id'].str.replace('_validation','_evaluation')

    test1 = pd.melt(test1, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')
    test2 = pd.melt(test2, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'day', value_name = 'demand')
    
    data['part'] = 'train'
    test1['part'] = 'test1'
    test2['part'] = 'test2'
    
    data = pd.concat([data, test1, test2], axis = 0)

    del test1, test2
    return data


#--------------------------------------------------------------------------------
def augment(data, data_path = "Data/", filename = ""):
    print("\n\nAugmenting data...")
    start_mem = data.memory_usage().sum() / 1024**2 

    print("Condensing SNAP...")
    data['snap'] = np.where(((data.state_id == 'CA') & (data.snap_CA == 1)) |
                            ((data.state_id == 'TX') & (data.snap_TX == 1)) |
                            ((data.state_id == 'WI') & (data.snap_WI == 1)),
                            1, 0)

    print("Encoding data...")
    data = encode(data)
    
    # Demand stuff
    col = 'demand'
    add_lag(data, col, 28)
    add_lag(data, col, 29)
    add_lag(data, col, 30)

    add_rolling(data, col, 'mean', 28, 7)
    add_rolling(data, col, 'std', 28, 7)
    add_rolling(data, col, 'mean', 28, 30)
    add_rolling(data, col, 'std', 28, 30)
    add_rolling(data, col, 'skew', 28, 30)
    add_rolling(data, col, 'kurt', 28, 30)
    add_rolling(data, col, 'mean', 28, 90)
    add_rolling(data, col, 'mean',28, 180)

    col = 'sell_price'
    add_delta(data, col, "shift", 1)
    add_delta(data, col, "max", 365)
    add_rolling(data, col, 'std', 0, 30)
    add_rolling(data, col, 'std', 0, 7)

    print(data.head())

    print("Adding date information...")
    data = add_date(data)

    end_mem = data.memory_usage().sum() / 1024**2
    DataExam.date_summary(data)
    print('augment_data: Mem. usage increased to {:5.2f} Mb ({:.1f}% increase)'.format(end_mem, 100 * (end_mem - start_mem) / start_mem))
    print('augment_data: data has {} rows and {} columns'.format(data.shape[0], data.shape[1]))

    print("Compressing augment_data")
    data = reduce_mem_usage(data, True)

    #TODO: Might be better to use some other criteria for determining save
    if(filename != ""):
        print("\n\nSaving to a pickle...")
        data.to_pickle(data_path + filename)

    gc.collect()
    return data

#--------------------------------------------------------------------------------
def add_date(data):
    data['date'] = pd.to_datetime(data['date'])
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['week'] = data['date'].dt.week
    data['day'] = data['date'].dt.day
    data['dayofweek'] = data['date'].dt.dayofweek
    return data

#--------------------------------------------------------------------------------
def split():
    """ Designate data as training/test sets.  Also trim data if we don't want
        to use the entire thing.
    """
    print("Getting validation/test columns...")
    print("Reading sample_submission.csv...")
    submission = pd.read_csv(data_path + "sample_submission.csv")

    val_rows = [row for row in submission['id'] if 'validation' in row]
    eval_rows = [row for row in submission['id'] if 'evaluation' in row]

    val_data = submission[submission['id'].isin(val_rows)]
    eval_data = submission[submission['id'].isin(eval_rows)]

    print(TEST_ROWS)

#--------------------------------------------------------------------------------
def encode(data):
    """ Clean up and encode categorical features
    """
    nan_features = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for feature in nan_features:
        data[feature].fillna('unknown', inplace = True)
        
    cat = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for feature in cat:
        encoder = preprocessing.LabelEncoder()
        data[feature] = encoder.fit_transform(data[feature])
    
    return data

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
    dest_col = source_col + '_lag_t' + str(lag_days)
    print("add_lag: data[" + dest_col + "] -> shifting:" + str(lag_days))
    data[dest_col] = data.groupby([ID_COL])[source_col].transform(lambda x: x.shift(lag_days))
 
#----------------------------------------------------------------------------
def add_delta(data, source_col, function, time_period):
    """ Calculate amount the source value changes relative to some other value.
    """
    dest_col = source_col + '_change_t' + str(time_period)
    print("add_delta: data[" + dest_col + "] -> function: " + function + " time period:" + str(time_period))

    if(function == 'shift'):
        temp = data.groupby([ID_COL])[source_col].transform(lambda x: x.shift(time_period))
    elif(function == 'max'):
        temp = data.groupby([ID_COL])[source_col].transform(lambda x: x.shift(1).rolling(time_period).max())

    data[dest_col] = (temp - data[source_col]) / temp

#----------------------------------------------------------------------------
def add_rolling(data, source_col,  function, lag_days, rolling_days):
    """ Add a historic data statistics to the dataframe.  Allows us to easily add relevant
        historic data (i.e. price 1 week or 1 month ago).  

        Arguments:
            data - dataframe to be acted on
            source_col - column to be lagged
            lag_days - number of days to lag the source_col values
            rolling_days - period over which function is applied
            function - string name of the function
    """
    dest_col = source_col + "_rolling_" + function + '_t' + str(rolling_days)

    print("add_rolling: data[" + dest_col + "] -> lag=" + str(lag_days) + " rolling=" + str(rolling_days) + " function=" + function)
    
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
def read(data_path = "Data/", filename = "combined_data.pkl"):
    """ Read in a merged data set and return the dataframe

        Arguments:
            data_path: path where file is stored
            filename: name of the file to be read
    """
    print("\n\nData.read()")
    print("Reading pickle: " + data_path + filename)
    data = pd.read_pickle(data_path + filename)
    print('read data: data has {} rows and {} columns'.format(data.shape[0], data.shape[1]))
    mem_usage = data.memory_usage().sum() / 1024**2
    print('read data size: {:5.2f} Mb'.format(mem_usage))   
    return data

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
 

