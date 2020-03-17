

import PreProcess as pre
import Visualization as visual
from SalesData import cSalesData
import ARIMA
import pandas as pd

import Stationary as stationary

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

    #sales_train_validation = pd.read_csv(data_path + "sales_train_validation.csv")
    #sales_train_validation = sales_train_validation.drop(['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], axis=1)
    #sales_train_validation = pd.melt(sales_train_validation, id_vars = ['id'], var_name = 'day', value_name = 'demand')
    #print('Melted sales train validation has {} rows and {} columns'.format(sales_train_validation.shape[0], sales_train_validation.shape[1]))
    #print(sales_train_validation.head(50))


    sales_data = cSalesData(data_path)

    test_data = sales_data.get_by_id('FOODS_3_586_CA_3_validation')
    print(test_data.head(50))
    print(sales_data.AUGMENT_COLS)


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
# OLD
def range_test():
    calendar, prices, sample_sub, sales_val_data = pre.load_data()

    print("Invalid value")
    print(pre.get_d_from_date(calendar,"2001-02-11"))

    print("Valid value")
    print(pre.get_d_from_date(calendar,"2011-02-11"))

    d_start = pre.get_d_from_date(calendar,"2016-04-19")
    #d_end = pre.get_d_from_date(calendar,"2016-04-24")
    d_end = pre.get_d_from_date(calendar,"2016-04-25")

    d_range = pre.get_d_range(sales_val_data, d_start, d_end)

    if (d_range is None ):
        print("d_range = None")
    else:
        print(d_range.columns)
        print(d_range.head())


#-------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
