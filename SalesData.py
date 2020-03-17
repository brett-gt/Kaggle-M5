import pandas as pd
import numpy as np
import gc
import pickle as pickle

import Globals as globals
from Calendar import cCalendar

class cSalesData:
    """Top level class to handle distribution of sales data.  
    """
    sales = []
    lookup = []

    ID_COL = 'id'
    LOOKUP_COL = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

    VAR_COL = 'day'
    VALUE_COL = 'demand'
    AUGMENT_COLS = [VALUE_COL]


    # TODO: Define a common output format

    #----------------------------------------------------------------------------
    def __init__(self, path):
        print("SalesData: Initializing data...")
        print("Reading sales_train_validation.csv...")
        data = pd.read_csv(path + "sales_train_validation.csv")

        print("Creating lookup dataframe...")
        self.lookup = pd.concat([data[self.ID_COL], data[self.LOOKUP_COL]], axis=1)
        self.lookup = self.lookup.set_index(self.ID_COL)
        print(self.lookup.head())

        print("Melting sales data...")
        #data = data.drop(self.LOOKUP_COL, axis=1)
        self.sales = pd.melt(data, id_vars = self.LOOKUP_COL, var_name = self.VAR_COL, value_name = self.VALUE_COL)
        print("Sales size = {}\nShape ={}\nShape[0] x Shape[1] = {}".
              format(self.sales.size, self.sales.shape, self.sales.shape[0]*self.sales.shape[1])) 
        print(self.sales.head())
        del data


        print("\n\nAdding calendar data...")
        print("Reading calendar.csv...")
        #self.calendar = cCalendar(path)
        calendar = pd.read_csv(path + "calendar.csv")
        calendar.drop(['weekday', 'wday', 'month', 'year'], inplace = True, axis = 1)
        print(calendar.head())
        print("\nMerging calendar prices with the data set...")
        self.sales = pd.merge(self.sales, calendar, how = 'left', left_on = ['day'], right_on = ['d'])
        self.sales.drop(['d', 'day'], inplace = True, axis = 1)
        print(self.sales.head())
        del calendar

        print("\n\nAdding sales price...")
        print("Reading sell_price.csv...")
        sell_prices = pd.read_csv(path + "sell_prices.csv")
        print(sell_prices.head())
        print("\nMerging sell prices with the data set...")
        self.sales = pd.merge(self.sales, sell_prices, how = 'left', on = ['store_id', 'item_id', 'wm_yr_wk'])
        print(self.sales.head())
        del sell_prices


        print("\n\nSAving to a pickle...")
        self.sales.to_pickle("combined_data.pkl")

        self.reduce_mem_usage(True)


        print("Augmenting data...")
        self.add_lag(self.VALUE_COL, 28)
        self.add_lag(self.VALUE_COL, 29)
        self.add_lag(self.VALUE_COL, 30)

        self.add_rolling(self.VALUE_COL, 28, 7, 'mean')
        self.add_rolling(self.VALUE_COL, 28, 7, 'std')
        self.add_rolling(self.VALUE_COL, 28, 30, 'mean')
        self.add_rolling(self.VALUE_COL, 28, 30, 'std')
        self.add_rolling(self.VALUE_COL, 28, 30, 'skew')
        self.add_rolling(self.VALUE_COL, 28, 30, 'kurt')
        self.add_rolling(self.VALUE_COL, 28, 90, 'mean')
        self.add_rolling(self.VALUE_COL, 28, 180, 'mean')


  
     
   
        print("SalesData init complete...")
        gc.collect()



    #----------------------------------------------------------------------------
    def store_item_to_id(self, store_id, item_id):
        """Take in a store_id and item_id and return the combined item id
        """
        #TODO: May have to do more than this if not everything end in _validation
        return store_id + "_" + item_id + "_validation"

    #----------------------------------------------------------------------------
    def add_lag(self, source_col, lag_days):
        print("Add_lag " + str(lag_days))
        self.AUGMENT_COLS.append('lag_t' + str(lag_days))
        self.sales[col_name] = self.sales.groupby([self.ID_COL])[source_col].transform(lambda x: x.shift(lag_days))

    #----------------------------------------------------------------------------
    def add_rolling(self, source_col, lag_days, rolling_days, function = 'mean'):
        print("add_rolling: lag=" + str(lag_days) + " rolling=" + str(rolling_days) + " function=" + function)
        self.AUGMENT_COLS.append('rolling_' + function + "_t" + str(rolling_days))

        if(function == 'mean'):
            self.sales[dest_col] = self.sales.groupby([ID_COL])[source_col].transform(lambda x: x.shift(lag_days).rolling(rolling_days).mean())
        elif(function == 'std'):
            self.sales[dest_col] = self.sales.groupby([ID_COL])[source_col].transform(lambda x: x.shift(lag_days).rolling(rolling_days).std())
        elif(function == 'skew'):
            self.sales[dest_col] = self.sales.groupby([ID_COL])[source_col].transform(lambda x: x.shift(lag_days).rolling(rolling_days).skew())
        elif(function == 'kurt'):
            self.sales[dest_col] = self.sales.groupby([ID_COL])[source_col].transform(lambda x: x.shift(lag_days).rolling(rolling_days).kurt())
        else:
            print("!!! add_rolling: Unidentified function !!!")


    #----------------------------------------------------------------------------
    def get_by_id(self, id):
        """ Get a sales for a particular id only
        """
        print("\n\nget_by_id")
        return self.sales.loc[self.sales['id'] == id]

    #----------------------------------------------------------------------------
    def get_dcol_minus(self, d_today, minus_start, length, id = "all"):
        """ Get a range of sales values in the past.
        """
        result = pd.DataFrame()
        index = self.sales.loc[self.sales[self.VAR_COL]==d_today].index[0]

        #Keep in mind need to reverse since going backwards... 
        #the "start" number is bigger than the "end" number
        end = index - minus_start
        if(end <= 0):
            end = 1;
        
        start = index - minus_start - length + 1
        if(start <= 0):
            start = 1;

        first_d_col = self.sales.loc[self.sales[self.VAR_COL]=="d_1"].index[0]

        result = self.sales.iloc[np.r_[0:first_d_col, start:end],:]
        print(result.head())

        if(id == "all"):
            return result
        else:
            return self.get_by_id(id, result)

    #--------------------------------------------------------------------------------
    def get_date_range(self, date_start, date_end):
        """ Get a range of rows from a dataset using the d_XXXX value.

            Arguments:
                data- dataset that contains a "d" column (d_XXXX)
                d_start - first d_XXXX
                d_end   - end d_XXXX

            Returns:
                data frame
        """  
        d_start = self.calendar.get_d_from_date(date_start)
        d_end = self.calendar.get_d_from_date(date_end)
        return self.get_d_range(d_start, d_end)

    #--------------------------------------------------------------------------------
    def get_d_range(self, d_start, d_end):
        """ Get a range of rows from a dataset using the d_XXXX value.

            Arguments:
                data- dataset that contains a "d" column (d_XXXX)
                d_start - first d_XXXX
                d_end   - end d_XXXX

            Returns:
                data frame
        """  
        if "d_1" not in self.d_cols:
            return None
        if d_start not in self.d_cols:
            return None
        if d_end not in self.d_cols:
            return None

        first_d_col = self.sales.day.get_loc("d_1")
        d_first_col = self.sales.day.get_loc(d_start)
        d_end_col = self.sales.day.get_loc(d_end) + 1

        result = self.sales.iloc[np.r_[0:first_d_col, d_first_col:d_end_col],:]
        print(result.head())
        return result

    #--------------------------------------------------------------------------------
    # TODO
    def get_values_by_store(self):
        d_cols = [c for c in self.sales.columns if 'd_' in c]
        return self.sales.groupby(['store_id'])[d_cols].values[0]

    #--------------------------------------------------------------------------------
    # TODO
    def get_values_by_state(self):

        d_cols = [c for c in self.sales.columns if 'd_' in c]
        return self.sales.groupby(['state_id'])

    #--------------------------------------------------------------------------------
    # TODO
    def get_values_by_dept(self):
        d_cols = [c for c in self.sales.columns if 'd_' in c]
        return self.sales.groupby(['dept_id'])

    #--------------------------------------------------------------------------------
    # TODO
    def get_values_by_item(self):
        d_cols = [c for c in self.sales.columns if 'd_' in c]
        return self.sales.groupby(['item_id'])

    #----------------------------------------------------------------------------
    def limit_d_cols(self, col_list):
        """ Limit a list of d_XXXX columns to those found in the sales dataset
            Useful since some of the other files include d_XXXX columns that will
            only be present in the test data
        """
        c_set = set(col_list)
        d_set = set(self.d_cols)
        return c_set.intersection(d_set)


    #----------------------------------------------------------------------------
    # TODO
    def basic_summary(self, filename):
        results = pd.DataFrame()
        results['id'] = self.sales['id']
        summary = self.create_summary(self.sales)
        results = pd.concat([results, summary], axis=1, sort=False)
        results.set_index('id')
        results.to_csv(filename)

    #----------------------------------------------------------------------------
    # TODO
    def week_summary(self, filename, days=globals.WEEK_SEARCH):
        results = pd.DataFrame()
        results['id'] = self.sales['id']

        for day in days:
            d_cols = self.calendar.getDays(day)
            subset = self.sales[self.limit_d_cols(d_cols)]
            
            name = "default"
            if(isinstance(day,str)):
                name = day 
            elif(isinstance(day,list)):
                name = "_".join(str(x) for x in day)

            summary = self.create_summary(subset, name + "_")
            results = pd.concat([results, summary], axis=1, sort=False)

        results.set_index('id')
        results.to_csv(filename)

    #----------------------------------------------------------------------------
    # TODO
    def create_summary(self, data, prefix = ""):
        d_cols = [c for c in data.columns if 'd_' in c]
        
        results = pd.DataFrame()
        length = len(d_cols)
        results[prefix + 'count'] = length
        results[prefix + 'min'] = data[d_cols].min(axis=1)
        results[prefix + 'max'] = data[d_cols].max(axis=1)
        results[prefix + 'sum'] = data[d_cols].sum(axis=1)
        results[prefix + 'mean'] = data[d_cols].mean(axis=1)
        results[prefix + 'median'] = data[d_cols].median(axis=1)
        results[prefix + 'std'] = data[d_cols].std(axis=1)
        results[prefix + 'zeros'] = (data[d_cols] == 0).astype(int).sum(axis=1)
        results[prefix + 'pcnt_zeros'] = ((data[d_cols] == 0).astype(int).sum(axis=1))/len(d_cols)
        return results

    #----------------------------------------------------------------------------
    def reduce_mem_usage(self, verbose=True):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = self.sales.memory_usage().sum() / 1024**2    
        for col in self.sales.columns:
            col_type = self.sales[col].dtypes
            if col_type in numerics:
                c_min = self.sales[col].min()
                c_max = self.sales[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        self.sales[col] = self.sales[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        self.sales[col] = self.sales[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        self.sales[col] = self.sales[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        self.sales[col] = self.sales[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        self.sales[col] = self.sales[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        self.sales[col] = self.sales[col].astype(np.float32)
                    else:
                        self.sales[col] = self.sales[col].astype(np.float64)    
        end_mem = self.sales.memory_usage().sum() / 1024**2

        if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

 
        



     








