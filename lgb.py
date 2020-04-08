
import lightgbm as lgb
import numpy as np
import pandas as pd
import Data as Data
import time

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

from sklearn import preprocessing, metrics
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from WRMSSEEvaluator import WRMSSEEvaluator

# define list of features
features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'year', 'month', 'week', 'day', 
            'dayofweek', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 
            'snap', 'sell_price', 'demand_lag_t28', 'demand_lag_t29', 'demand_lag_t30', 
            'demand_rolling_mean_t7', 'demand_rolling_std_t7', 'demand_rolling_mean_t30', 'demand_rolling_mean_t90', 
            'demand_rolling_mean_t180', 'demand_rolling_std_t30', 'sell_price_change_t1', 'sell_price_change_t365', 
            'sell_price_rolling_std_t7', 'sell_price_rolling_std_t30', 'demand_rolling_skew_t30', 'demand_rolling_kurt_t30']


class LGB(object):

    #--------------------------------------------------------------------------------
    def __init__(self, data):
        self.model = []
        self.evaluator = WRMSSEEvaluator()
        print(data.shape)
        self.split_data(data)

    #--------------------------------------------------------------------------------
    def split_data(self, data):
        print("Splitting data...")
        self.x_train, self.y_train, self.x_val, self.y_val, self.test = Data.seperate(data, '2016-03-27', '2016-04-24')
        print(self.x_train.shape)
        print(self.y_train.shape)
        print(self.x_val.shape)
        print(self.y_val.shape)
        print(self.x_train[features].shape)
        print(self.x_val[features].shape)
        self.d_col_val = 1886

    #--------------------------------------------------------------------------------
    def train_and_predict(self, show_plots=False):
        print("\n\n\nlgb.Run()")
 
        # define random hyperparammeters
        params = {
            'boosting_type': 'gbdt',
            'metric': 'rmse',
            'objective': 'regression',
            'n_jobs': -1,
            'seed': 236,
            'learning_rate': 0.08,
            'bagging_fraction': 0.75,
            'bagging_freq': 10, 
            'colsample_bytree': 0.75}

        train_set = lgb.Dataset(self.x_train[features], self.y_train)
        val_set = lgb.Dataset(self.x_val[features], self.y_val)    
  
        print("Training model...")
        evals_result = {}
        #TODO:
        # 'metric':'auc'
        self.model = lgb.train(params, 
                               train_set, 
                               num_boost_round = 250,  #TODO: 250 
                               early_stopping_rounds = 50, 
                               valid_sets = val_set, #[train_set, val_set], 
                               verbose_eval = 20,
                               feval = self.WRMSSE_val_loss,    #TODO NEW
                               evals_result = evals_result)         

        # TODO: Getting error with this once I moved it into class
        #if(show_plots):
        #    ax = lgb.plot_importance(self.model, max_num_features=20)
        #    plt.show()

        #    ax = lgb.plot_split_value_histogram(self.model, feature='store_id', bins='auto')
        #    plt.show()
 
    #--------------------------------------------------------------------------------
    def WRMSSE_val_loss(self, pred, train_data):
        #predictions = y_pred[['id', 'date', 'demand']]
        #predictions = pd.pivot(predictions, index = 'id', columns = 'date', values = 'demand').reset_index()
        #predictions.columns = ['id'] + ['d_' + str(self.d_col_val + i) for i in range(28)]
        #predictions = predictions.drop('id',axis =1) #WRMSSE only want data columns

        #print(predictions.columns)

        pred = pred.reshape(int(len(pred)/28) , 28)
        loss = self.evaluator.score(pred)

        #Note:  Want smallest error, so higher is worse
        return "custom_loss", loss, False

    
    #--------------------------------------------------------------------------------
    def make_submission(self):
        #TODO: This was at the end of train
        print("Predicting with model...")
        val_pred = self.model.predict(self.x_val[features])
        val_score = np.sqrt(metrics.mean_squared_error(val_pred, self.y_val))

        print(f'Our val rmse score is {val_score}')
        y_pred = self.model.predict(self.test[features])
        self.test['demand'] = y_pred
        #return test

        print("Preparing submission...")
        submission = pd.read_csv('Data/sample_submission.csv')

        predictions = self.test[['id', 'date', 'demand']]
        predictions = pd.pivot(predictions, index = 'id', columns = 'date', values = 'demand').reset_index()
        predictions.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]

        evaluation_rows = [row for row in submission['id'] if 'evaluation' in row] 
        evaluation = submission[submission['id'].isin(evaluation_rows)]

        validation = submission[['id']].merge(predictions, on = 'id')
        final = pd.concat([validation, evaluation])
        final.to_csv('submission.csv', index = False)
        return final

#END CLASS


#--------------------------------------------------------------------------------
# TODO: Log/Output names are hardcoded
# TODO: Option to read in log for future use
# https://www.kaggle.com/tilii7/bayesian-optimization-of-xgboost-parameters
# https://www.kaggle.com/clair14/tutorial-bayesian-optimization
def bayes_optimize(data, 
                   init_round=15, #TODO
                   opt_round=25,  #TODO
                   n_folds=5, random_seed=6, 
                   n_estimators=10000, 
                   learning_rate=0.05, 
                   logging = True,
                   output_process=False):


    print("\n\n\bayes_optimize")
    print("Splitting data...")
    x_train, y_train, x_val, y_val, test = Data.seperate(data, '2016-03-27', '2016-04-24')

    # prepare data
    train_data = lgb.Dataset(x_train[features], y_train)

    # parameters
    def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight):
         params = {'application':'regression_l1',
                   'num_iterations': n_estimators, 
                   'learning_rate':learning_rate, 
                   'early_stopping_round':100, 
                   'metric':'auc'}

         params["num_leaves"] = int(round(num_leaves))
         params['feature_fraction'] = max(min(feature_fraction, 1), 0)
         params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
         params['max_depth'] = int(round(max_depth))
         params['lambda_l1'] = max(lambda_l1, 0)
         params['lambda_l2'] = max(lambda_l2, 0)
         params['min_split_gain'] = min_split_gain
         params['min_child_weight'] = min_child_weight
         cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval =200, metrics=['auc'])
         return max(cv_result['auc-mean'])

    # range 
    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (24, 4000),
                                            'max_depth': (5, 60),
                                            'lambda_l1': (0, 5),
                                            'lambda_l2': (0, 3),
                                            'feature_fraction': (0.1, 0.9),
                                            'bagging_fraction': (0.8, 1),
                                            'min_split_gain': (0.001, 0.1),
                                            'min_child_weight': (5, 50)}, random_state=0)
    # Logging
    if logging:
        logger = JSONLogger(path = "./lboLog.json")
        lgbBO.subscribe(Events.OPTMIZATION_STEP, logger)

    # optimize
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)

    print(lgbBO.max)
    
    # output optimization process
    if output_process==True: 
        lgbBO.points_to_csv("bayes_opt_result.csv")
    
    # return best parameters
    return lgbBO.res['max']['max_params']




##################################################################################
# https://www.kaggle.com/ogrellier/feature-selection-with-null-importances
##################################################################################

#--------------------------------------------------------------------------------
def null_importance(data):
    np.random.seed(123)

    start = time.time()
    dsp = ''

    null_imp_df = pd.DataFrame()
    runs = 80

    for i in range(80):
        #Get current run importance
        imp_df = get_feature_importance(data)
        imp_df['run'] = i+1

        #Concat with previous
        null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)

        # Erase previous message
        for l in range(len(dsp)):
            print('\b', end='', flush=True)

        # Display current run and time used
        spent = (time.time() - start) / 60
        dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
        print(dsp, end='', flush=True)


#--------------------------------------------------------------------------------
def get_feature_importance(data, verbose=False):

    print("Evaluting feature importance using lightgbm")

    print("Splitting data...")
    x_train, y_train, x_val, y_val, test = Data.seperate(data, '2016-03-27', '2016-04-24')

    # define random hyperparammeters
    # TODO: Changing this could affect feature importance
    params = {
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'objective': 'regression',
        'n_jobs': -1,
        'seed': 236,
        'learning_rate': 0.08,
        'bagging_fraction': 0.75,
        'bagging_freq': 1, 
        'colsample_bytree': 0.75}

    train_set = lgb.Dataset(x_train[features], y_train)
    val_set = lgb.Dataset(x_val[features], y_val)    
    del x_train, y_train

    print("Training model...")
    evals_result = {}
    model = lgb.train(params, 
                      train_set, 
                      num_boost_round = 2500, 
                      early_stopping_rounds = 50, 
                      valid_sets = [train_set, val_set], 
                      verbose_eval = 100)

    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(train_features)
    imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
    imp_df["importance_split"] = clf.feature_importance(importance_type='split')
    imp_df['trn_score'] = roc_auc_score(y, clf.predict(data[train_features]))

    if verbose:
        print(imp_df.head(10))

    return imp_df

#--------------------------------------------------------------------------------
def display_distributions(actual_imp_df_, null_imp_df_, feature_):
    plt.figure(figsize=(13, 6))
    gs = gridspec.GridSpec(1, 2)

    # Plot Split importances
    ax = plt.subplot(gs[0, 0])
    a = ax.hist(null_imp_df_.loc[null_imp_df_['feature'] == feature_, 'importance_split'].values, label='Null importances')
    ax.vlines(x=actual_imp_df_.loc[actual_imp_df_['feature'] == feature_, 'importance_split'].mean(), 
               ymin=0, ymax=np.max(a[0]), color='r',linewidth=10, label='Real Target')
    ax.legend()
    ax.set_title('Split Importance of %s' % feature_.upper(), fontweight='bold')
    plt.xlabel('Null Importance (split) Distribution for %s ' % feature_.upper())

    # Plot Gain importances
    ax = plt.subplot(gs[0, 1])
    a = ax.hist(null_imp_df_.loc[null_imp_df_['feature'] == feature_, 'importance_gain'].values, label='Null importances')
    ax.vlines(x=actual_imp_df_.loc[actual_imp_df_['feature'] == feature_, 'importance_gain'].mean(), 
               ymin=0, ymax=np.max(a[0]), color='r',linewidth=10, label='Real Target')
    ax.legend()
    ax.set_title('Gain Importance of %s' % feature_.upper(), fontweight='bold')
    plt.xlabel('Null Importance (gain) Distribution for %s ' % feature_.upper())
        

