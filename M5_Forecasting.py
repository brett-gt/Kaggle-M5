

import Visualization as visual
import ARIMA
import pandas as pd

import Data as Data
import DataExam 
import lgb as lgb_wrap

import Stationary as stationary
import Calendar as calendar
import Fst_Copy 
from WRMSSEEvaluator import WRMSSEEvaluator

#owing to the fact that so many factors affect the sales on a given day. On certain days, the sales quantity is 0, which indicates that a certain product may not be available on that day (as noted by Rob in his kernel).

#https://www.kaggle.com/li325040229/more-eda-and-a-simple-lstm-model

#https://www.sciencedirect.com/science/article/pii/S0169207019301153

# VERY GOOD
#https://www.kaggle.com/ashishpatel26/lstm-demand-forecasting


#https://www.kaggle.com/screech/ensemble-of-arima-and-lstm-model-for-wiki-pages


#PRIMARY TECHNIQUES

#https://eng.uber.com/m4-forecasting-competition/

#https://www.sciencedirect.com/science/article/pii/S0169207019301153?via%3Dihub

#https://www.kaggle.com/ceshine/lgbm-starter/comments

#https://github.com/Mcompetitions/M4-methods/blob/slaweks_ES-RNN/118%20-%20slaweks17/ES_RNN_SlawekSmyl.pdf

#https://github.com/gregaw/sales-forecasting-with-nn


data_path = "Data/"
out_path = "Output/"

#-------------------------------------------------------------------------------------
def main():   
    print("Main...")
    #Fst_Copy.run()

    # Phase 1
    #Data.merge_data_sets()


    # Phase 2 
    #data = Data.read(filename="combined_data.pkl")
    #data = Data.subset(data, nrows=27500000)
    #data = Data.augment(data, filename='augmented_data.pkl')
    #print(data.columns)

    # Phase 3 
    data = Data.read(filename="augmented_data.pkl")
    data.drop(['snap_CA', 'snap_TX', 'snap_WI'], inplace = True, axis = 1)
    #DataExam.date_summary(data)
    print('Our final dataset to train has {} rows and {} columns'.format(data.shape[0], data.shape[1]))
    print(data.columns)

    LGB = lgb_wrap.LGB(data)
    LGB.train_and_predict()
    final = LGB.make_submission()

    print("Running our WRMSEE...")
    evaluator = WRMSSEEvaluator()
    print(evaluator.score_submission(final))



    #dataexam.basic_summary("summary.csv")




    #series = sales_data.get_date_range("2015-01-01", "2016-01-01")
    #test_data = sales_data.get_by_id('FOODS_3_586_CA_3_validation')
    #ARIMA.find_AR(test_data)
    #ARIMA.find_differncing(test_data)
    #ARIMA.find_MA(test_data)
    #ARIMA.apply(test_data)

    #stationary.examine(series)
 
    #visual.plot_rolling_window(series['FOODS_3_694_TX_1_validation'])





    #train_val_set = pd.read_csv(data_path + "sales_train_validation.csv")
    #d_cols = [c for c in train_val_set if 'd_' in c]
    #series = train_val_set[train_val_set['id'] == 'FOODS_3_694_TX_1_validation'][d_cols]
    #print(series.head())
    #visual.test_stationary(series)

    #data.week_summary(out_path + "sale_summary.csv")

    #calendar, prices, sample_sub, sales_val_data = pre.load_data()
    #visual.compare_sales_date_range(calendar, sales_val_data, "FOODS_3_120_WI_1_validation", "d_1914", "d_1942")

    #visual.show_sales(sales_val_data)
    #visual.show_denoise(validation_set, "average")
    #visual.candle_stick(validation_set, calendar)

    #train_data.to_csv("train_data.csv")
    #val_data.to_csv("val_data.csv")

    #train_data, val_data = pre.split_data(sales_val_data, 30)

    #visual.display_train_val(train_data, val_data)

#-------------------------------------------------------------------------------------
def WRMSSEE_test():
    evaluator = WRMSSEEvaluator()
    final = pd.read_csv('submission.csv')
    print(evaluator.score_submission(final))


#-------------------------------------------------------------------------------------
def lgb_opt():
    data = Data.read(filename="augmented_data.pkl")
    data.drop(['snap_CA', 'snap_TX', 'snap_WI'], inplace = True, axis = 1)
    DataExam.date_summary(data)
    print('Our final dataset to train has {} rows and {} columns'.format(data.shape[0], data.shape[1]))
    print(data.columns)
    print("\n\n\n")

    opt_param = lgb_wrap.bayes_optimize(data)
    print(opt_param)


#-------------------------------------------------------------------------------------
def cal_test():
    print("Reading calendar...")
    cal = calendar.cCalendar(data_path)
    d_cols = cal.get_dcol_for_year(2011)

    print("Results")
    print(d_cols)

    d_cols = cal.get_dcol_for_year(2013)

    print("Results")
    print(d_cols)

#-------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
