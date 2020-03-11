import pandas as pd
import numpy as np

import Globals as globals
import Calendar as calendar


class cSalesData:
    """Top level class to handle distribution of sales data.  
    """
    sales = []
    lookup = []

    ID_COL = ['id']
    LOOKUP_COL = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    

    #----------------------------------------------------------------------------
    def __init__(self, path):
        self.sales = pd.read_csv(path + "sales_train_validation.csv")
        self.d_cols = [c for c in self.sales.columns if 'd_' in c]

        self.lookup = self.sales[self.ID_COL + self.LOOKUP_COL]
        self.lookup = self.lookup.set_index(self.ID_COL)
        
        self.sales = self.sales.drop(self.LOOKUP_COL, axis=1)
        self.sales = self.sales.set_index(self.ID_COL).transpose().reset_index().rename(columns={'index':'d_col'})
        #TODO: d_col isn't index, as seen in get_by_id usage.  

        self.calendar = calendar.cCalendar(path)

    #----------------------------------------------------------------------------
    # TODO: Define a common output format
    # TODO: d_col isn't in the d_col format, prob an issue init
    def get_by_id(self, id):
  
        result = pd.DataFrame()
        result['d_col'] = self.sales.index
        result = result.set_index('d_col')
        result['data'] = self.sales[id]

        print("\n\nget_by_id")
        print(result.head())
        return result

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

        first_d_col = self.sales.index.get_loc("d_1")
        d_first_col = self.sales.index.get_loc(d_start)
        d_end_col = self.sales.index.get_loc(d_end) + 1

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

 
        



     








