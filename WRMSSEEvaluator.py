""" This is sakami (https://www.kaggle.com/sakami) implementation
    From https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/133834 

    Example usage:

        evaluator = WRMSSEEvaluator(train_data, valid_data, results, calendar, prices)
        evaluator.score(results)
"""
    
from typing import Union

import numpy as np
import pandas as pd

class WRMSSEEvaluator(object):
    
    group_ids = ( 'all_id', 'state_id', 'store_id', 'cat_id', 'dept_id', 'item_id',
        ['state_id', 'cat_id'],  ['state_id', 'dept_id'], ['store_id', 'cat_id'],
        ['store_id', 'dept_id'], ['item_id', 'state_id'], ['item_id', 'store_id'])

    #----------------------------------------------------------------------------
    def __init__(self, path = "Data/"):
        print("\n\nInitializing WRMSSE Evaluator")

        #Added this stuff because the way I am doing it I don't keep these around
        print("Reading data...")
        sales_data = pd.read_csv(path + "sales_train_validation.csv")
        train_df = sales_data.iloc[:,:-28]
        valid_df = sales_data.iloc[:,-28:]

        calendar = pd.read_csv(path + "calendar.csv")
        prices = pd.read_csv(path + "sell_prices.csv")

        train_target_columns = [i for i in train_df.columns if i.startswith('d_')]
        weight_columns = train_df.iloc[:, -28:].columns.tolist()

        train_df['all_id'] = 'all'  # for lv1 aggregation
        id_columns = [i for i in train_df.columns if not i.startswith('d_')]
        valid_target_columns = [i for i in valid_df.columns if i.startswith('d_')]

        #TODO:
        #valid_target_columns = ['d_1914', 'd_1915', 'd_1916', 'd_1917', 'd_1918', 'd_1919', 'd_1920', 'd_1921', 'd_1922', 
        #                        'd_1923', 'd_1924', 'd_1925', 'd_1926', 'd_1927', 'd_1928', 'd_1929', 'd_1930', 'd_1931', 
        #                        'd_1932', 'd_1933', 'd_1934', 'd_1935', 'd_1936', 'd_1937', 'd_1938', 'd_1939', 'd_1940', 'd_1941']

        if not all([c in valid_df.columns for c in id_columns]):
            valid_df = pd.concat([train_df[id_columns], valid_df], axis=1, sort=False)

        self.train_series = self.trans_30490_to_42840(train_df, 
                                                      train_target_columns, 
                                                      self.group_ids)

        self.valid_series = self.trans_30490_to_42840(valid_df, 
                                                      valid_target_columns, 
                                                      self.group_ids)

        # Set values we want to keep
        self.valid_df = valid_df
        self.weight_columns = weight_columns
        self.id_columns = id_columns
        self.valid_target_columns = valid_target_columns

        # Calculate weights
        self.weights = self.get_weight_df(calendar, prices, train_df)
        self.scale = self.get_scale()


    #----------------------------------------------------------------------------
    def get_scale(self):
        '''
        scaling factor for each series ignoring starting zeros
        '''
        scales = []
        for i in range(len(self.train_series)):
            series = self.train_series.iloc[i].values
            series = series[np.argmax(series!=0):]
            scale = ((series[1:] - series[:-1]) ** 2).mean()
            scales.append(scale)
        return np.array(scales)

    #----------------------------------------------------------------------------
    def get_name(self, i):
        '''
        convert a str or list of strings to unique string 
        used for naming each of 42840 series
        '''
        if type(i) == str or type(i) == int:
            return str(i)
        else:
            return "--".join(i)

    #----------------------------------------------------------------------------
    def get_weight_df(self, calendar, prices, train_df) -> pd.DataFrame:
        """
        returns weights for each of 42840 series in a dataFrame
        """
        day_to_week = calendar.set_index("d")["wm_yr_wk"].to_dict()
        weight_df = train_df[["item_id", "store_id"] + self.weight_columns].set_index(["item_id", "store_id"])
        weight_df = (weight_df.stack().reset_index().rename(columns={"level_2": "d", 0: "value"}))
        weight_df["wm_yr_wk"] = weight_df["d"].map(day_to_week)
        weight_df = weight_df.merge(prices, how="left", on=["item_id", "store_id", "wm_yr_wk"])

        weight_df["value"] = weight_df["value"] * weight_df["sell_price"]
        weight_df = weight_df.set_index(["item_id", "store_id", "d"]).unstack(level=2)["value"]
        weight_df = weight_df.loc[zip(train_df.item_id, train_df.store_id), : ].reset_index(drop=True)
        weight_df = pd.concat([train_df[self.id_columns], weight_df], axis=1, sort=False )
        weights_map = {}

        for i, group_id in enumerate(self.group_ids):
            lv_weight = weight_df.groupby(group_id)[self.weight_columns].sum().sum(axis=1)
            lv_weight = lv_weight / lv_weight.sum()
            for i in range(len(lv_weight)):
                weights_map[self.get_name(lv_weight.index[i])] = np.array([lv_weight.iloc[i]])
        weights = pd.DataFrame(weights_map).T / len(self.group_ids)

        return weights

    #----------------------------------------------------------------------------
    def trans_30490_to_42840(self, df, cols, group_ids, dis=False):
        '''
        transform 30490 sries to all 42840 series
        '''
        series_map = {}
        for i, group_id in enumerate(group_ids):
            tr = df.groupby(group_id)[cols].sum()
            for i in range(len(tr)):
                series_map[self.get_name(tr.index[i])] = tr.iloc[i].values
        return pd.DataFrame(series_map).T

    #----------------------------------------------------------------------------
    def get_rmsse(self, valid_preds) -> pd.Series:
        '''
        returns rmsse scores for all 42840 series
        '''
        score = ((self.valid_series - valid_preds) ** 2).mean(axis=1)
        rmsse = (score / self.scale).map(np.sqrt)
        return rmsse

    #----------------------------------------------------------------------------
    def score_submission(self, results, d_col_start = 1886) -> float:
        """ Adjust submission column headers to match what is expected by the default
            WRMSSE calculations.

            Internal testing starts at 1886 (assuming last 28 held): 3/27/16

            Submission starts at F1 (val) = D_1914 = 4/24/16
                                 F1 (eval) = D_1942 = 5/22/16
        """
        results = results[~results.id.str.contains("evaluation")]
        results = results.drop('id',axis =1)

        col_dic = {}
        for i in range(28):
            col_dic['F'+str(i+1)] = 'd_' + str(d_col_start+i)
        results.rename(columns=col_dic, inplace = True)
        return(self.score(results))

    #----------------------------------------------------------------------------
    def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]) -> float:      
        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape

        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)

        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds], axis=1, sort=False)

        valid_preds = self.trans_30490_to_42840(valid_preds, 
                                                self.valid_target_columns, 
                                                self.group_ids, 
                                                True)
        rmsse = self.get_rmsse(valid_preds)
        contributors = pd.concat([self.weights, rmsse], 
                                  axis=1, 
                                  sort=False).prod(axis=1)

        score = np.sum(contributors)
        print("WRMSSE Score: " + str(score))
        return score
