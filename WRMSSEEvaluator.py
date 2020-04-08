""" This is sakami (https://www.kaggle.com/sakami) implementation
    From https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/133834 

    Example usage:

        evaluator = WRMSSEEvaluator(train_data, valid_data, results, calendar, prices)
        evaluator.score(results)
"""
    
from typing import Union

import numpy as np
import pandas as pd
#from tqdm.notebook import tqdm_notebook as tqdm


class WRMSSEEvaluator(object):
    
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

        train_y = train_df.loc[:, train_df.columns.str.startswith('d_')]
        train_target_columns = train_y.columns.tolist()
        weight_columns = train_y.iloc[:, -28:].columns.tolist()

        train_df['all_id'] = 'all'  # for lv1 aggregation

        id_columns = train_df.loc[:, ~train_df.columns.str.startswith('d_')].columns.tolist()

        valid_target_columns = valid_df.loc[:, valid_df.columns.str.startswith('d_')].columns.tolist()
        #TODO:
        #valid_target_columns = ['d_1914', 'd_1915', 'd_1916', 'd_1917', 'd_1918', 'd_1919', 'd_1920', 'd_1921', 'd_1922', 
        #                        'd_1923', 'd_1924', 'd_1925', 'd_1926', 'd_1927', 'd_1928', 'd_1929', 'd_1930', 'd_1931', 
        #                        'd_1932', 'd_1933', 'd_1934', 'd_1935', 'd_1936', 'd_1937', 'd_1938', 'd_1939', 'd_1940', 'd_1941']

        if not all([c in valid_df.columns for c in id_columns]):
            valid_df = pd.concat([train_df[id_columns], valid_df], axis=1, sort=False)

        self.train_df = train_df
        self.valid_df = valid_df
        self.calendar = calendar
        self.prices = prices

        self.weight_columns = weight_columns
        self.id_columns = id_columns
        self.valid_target_columns = valid_target_columns


        # Calculate weights
        weight_df = self.get_weight_df()

        self.group_ids = (
            'all_id',
            'state_id',
            'store_id',
            'cat_id',
            'dept_id',
            ['state_id', 'cat_id'],
            ['state_id', 'dept_id'],
            ['store_id', 'cat_id'],
            ['store_id', 'dept_id'],
            'item_id',
            ['item_id', 'state_id'],
            ['item_id', 'store_id']
        )

        for i, group_id in enumerate(self.group_ids): #tqdm(self.group_ids)
            train_y = train_df.groupby(group_id)[train_target_columns].sum()
            scale = []
            for _, row in train_y.iterrows():
                series = row.values[np.argmax(row.values != 0):]
                scale.append(((series[1:] - series[:-1]) ** 2).mean())
            setattr(self, f'lv{i + 1}_scale', np.array(scale))
            setattr(self, f'lv{i + 1}_train_df', train_y)
            setattr(self, f'lv{i + 1}_valid_df', valid_df.groupby(group_id)[valid_target_columns].sum())

            lv_weight = weight_df.groupby(group_id)[weight_columns].sum().sum(axis=1)
            setattr(self, f'lv{i + 1}_weight', lv_weight / lv_weight.sum())

    #----------------------------------------------------------------------------
    def get_weight_df(self) -> pd.DataFrame:
        day_to_week = self.calendar.set_index('d')['wm_yr_wk'].to_dict()
        weight_df = self.train_df[['item_id', 'store_id'] + self.weight_columns].set_index(['item_id', 'store_id'])
        weight_df = weight_df.stack().reset_index().rename(columns={'level_2': 'd', 0: 'value'})
        weight_df['wm_yr_wk'] = weight_df['d'].map(day_to_week)

        weight_df = weight_df.merge(self.prices, how='left', on=['item_id', 'store_id', 'wm_yr_wk'])
        weight_df['value'] = weight_df['value'] * weight_df['sell_price']
        weight_df = weight_df.set_index(['item_id', 'store_id', 'd']).unstack(level=2)['value']
        weight_df = weight_df.loc[zip(self.train_df.item_id, self.train_df.store_id), :].reset_index(drop=True)
        weight_df = pd.concat([self.train_df[self.id_columns], weight_df], axis=1, sort=False)
        return weight_df

    #----------------------------------------------------------------------------
    def rmsse(self, valid_preds: pd.DataFrame, lv: int) -> pd.Series:
        valid_y = getattr(self, f'lv{lv}_valid_df')
        score = ((valid_y - valid_preds) ** 2).mean(axis=1)
        scale = getattr(self, f'lv{lv}_scale')
        return (score / scale).map(np.sqrt)

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

        all_scores = []
        for i, group_id in enumerate(self.group_ids):
            lv_scores = self.rmsse(valid_preds.groupby(group_id)[self.valid_target_columns].sum(), i + 1)
            weight = getattr(self, f'lv{i + 1}_weight')
            lv_scores = pd.concat([weight, lv_scores], axis=1, sort=False).prod(axis=1)
            all_scores.append(lv_scores.sum())

        self.all_scores = all_scores
        score = np.mean(all_scores)
        print("WRMSSE Score: " + str(score))
        return score
