import pandas as pd
import Globals as globals
import Calendar as calendar

class cSalesData:
    """description of class"""
    sales = []

    #----------------------------------------------------------------------------
    def __init__(self, path):
        self.sales = pd.read_csv(path + "sales_train_validation.csv")
        self.calendar = calendar.cCalendar(path)
        #TODO: d_cols = [c for c in data.columns if 'd_' in c]

    #--------------------------------------------------------------------------------
    def get_d_range(d_start, d_end, columns=True):
        """ Get a range of rows from a dataset using the d_XXXX value.

            Arguments:
                data- dataset that contains a "d" column (d_XXXX)
                d_start - first d_
                d_end   - end _d
                columns - the target is a dataframe that has columns named d_

            Returns:
                data frame
        """  
        cols = self.sales.columns

        if "d_1" not in cols:
            return None
        if d_start not in cols:
            return None
        if d_end not in cols:
            return None

        first_d_col = sales.columns.get_loc("d_1")
        d_first_col = sales.columns.get_loc(d_start)
        d_end_col = sales.columns.get_loc(d_end) + 1

        result = sales.iloc[:, np.r_[0:first_d_col, d_first_col:d_end_col]]
        return result



    #----------------------------------------------------------------------------
    def basic_summary(self, filename):
        results = pd.DataFrame()
        results['id'] = self.sales['id']
        summary = self.create_summary(self.sales)
        results = pd.concat([results, summary], axis=1, sort=False)
        results.set_index('id')
        results.to_csv(filename)

    #----------------------------------------------------------------------------
    def limit_d_cols(self, col_list):
        c_set = set(col_list)
        d_set = set([c for c in self.sales.columns if 'd_' in c])
        return c_set.intersection(d_set)

    #----------------------------------------------------------------------------
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

 
        



     








