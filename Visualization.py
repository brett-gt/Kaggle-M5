from matplotlib import pyplot as plt

import plotly as plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

import numpy as np
import datetime as datetime

import PreProcess as pre
import Globals as globals

#python -m pip install plotly --user


#https://www.kaggle.com/tarunpaparaju/m5-competition-eda-models




#--------------------------------------------------------------------------------
def compare_sales_date_range(calendar, sales_val_data, id, d_start, d_end):
    """ Goal here is to take in the d_start and d_end columns of the validation set,
        find the date range that covers, then plot that date range in all training data.

        Arguments: 
            calendar - data frame from calendar.csv
            sales_val_data - data from sales_train_validation.csv file
            id = unique identified from sales_train_validation.csv
            d_start = start in the form of d_XXXX
            d_end = end in the form of d_XXXX
    """
    start_date = pre.get_date_from_d(calendar, d_start)
    print(start_date.strftime(globals.DATE_FORMAT))

    end_date = pre.get_date_from_d(calendar, d_end)
    print(end_date.strftime(globals.DATE_FORMAT))

    if start_date is None:
        print("compare_sales_date_range: d_start " + str(d_start) + " not found.")
        return
    if end_date is None:
        print("compare_sales_date_range: d_end " + str(d_end) + " not found.")
        return


    fig = make_subplots(rows=len(globals.YEARS), cols=1)

    for row, cur_year in enumerate(globals.YEARS):
        print("Plotting year: " + str(cur_year));
        start_date_str = start_date.replace(year = cur_year).strftime(globals.DATE_FORMAT)
        end_date_str = end_date.replace(year = cur_year).strftime(globals.DATE_FORMAT)

        start_d_col = pre.get_d_from_date(calendar, start_date_str)
        end_d_col = pre.get_d_from_date(calendar, end_date_str)

        d_cols = globals.make_d_col_range(start_d_col, end_d_col)
        print(d_cols)

        X = sales_val_data.loc[sales_val_data['id'] == id].set_index('id')[d_cols].values[0]
        print(X)

        fig.add_trace(go.Scatter(x=np.arange(len(X)), y=X, showlegend=False,
                                 mode='lines', name=cur_year,
                                 marker=dict(color="mediumseagreen")),
                                 row=row+1, col=1)

    fig.update_layout(height=1200, width=800, title_text="Sales for " + str(id))
    plotly.offline.plot(fig, filename="test.html")


#--------------------------------------------------------------------------------
def show_sales(sales_train_val):
    ids = sorted(list(set(sales_train_val['id'])))
    d_cols = [c for c in sales_train_val.columns if 'd_' in c]
    x_1 = sales_train_val.loc[sales_train_val['id'] == ids[2]].set_index('id')[d_cols].values[0]
    x_2 = sales_train_val.loc[sales_train_val['id'] == ids[1]].set_index('id')[d_cols].values[0]
    x_3 = sales_train_val.loc[sales_train_val['id'] == ids[17]].set_index('id')[d_cols].values[0]

    fig = make_subplots(rows=3, cols=1)

    fig.add_trace(go.Scatter(x=np.arange(len(x_1)), y=x_1, showlegend=False,
                        mode='lines', name="First sample",
                             marker=dict(color="mediumseagreen")),
                 row=1, col=1)

    fig.add_trace(go.Scatter(x=np.arange(len(x_2)), y=x_2, showlegend=False,
                        mode='lines', name="Second sample",
                             marker=dict(color="violet")),
                 row=2, col=1)

    fig.add_trace(go.Scatter(x=np.arange(len(x_3)), y=x_3, showlegend=False,
                        mode='lines', name="Third sample",
                             marker=dict(color="dodgerblue")),
                 row=3, col=1)

    fig.update_layout(height=1200, width=800, title_text="Sample sales")
    fig.show()

#------------------------------------------------------------------------------------------
def show_denoise(sales_train_val, denoise_type = "wavelet"):
    ids = sorted(list(set(sales_train_val['id'])))
    d_cols = [c for c in sales_train_val.columns if 'd_' in c]
    x_1 = sales_train_val.loc[sales_train_val['id'] == ids[2]].set_index('id')[d_cols].values[0]
    x_2 = sales_train_val.loc[sales_train_val['id'] == ids[1]].set_index('id')[d_cols].values[0]
    x_3 = sales_train_val.loc[sales_train_val['id'] == ids[17]].set_index('id')[d_cols].values[0]

    if(denoise_type == "average"):
        y_w1 = pre.average_denoise(x_1)
        y_w2 = pre.average_denoise(x_2)
        y_w3 = pre.average_denoise(x_3)
    else:
        y_w1 = pre.wavelet_denoise(x_1)
        y_w2 = pre.wavelet_denoise(x_2)
        y_w3 = pre.wavelet_denoise(x_3)

    fig = make_subplots(rows=3, cols=1)

    fig.add_trace(
        go.Scatter(x=np.arange(len(x_1)), mode='lines+markers', y=x_1, marker=dict(color="mediumaquamarine"), showlegend=False,
                   name="Original signal"),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=np.arange(len(x_1)), y=y_w1, mode='lines', marker=dict(color="darkgreen"), showlegend=False,
                   name="Denoised signal"),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=np.arange(len(x_2)), mode='lines+markers', y=x_2, marker=dict(color="thistle"), showlegend=False),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=np.arange(len(x_2)), y=y_w2, mode='lines', marker=dict(color="purple"), showlegend=False),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=np.arange(len(x_3)), mode='lines+markers', y=x_3, marker=dict(color="lightskyblue"), showlegend=False),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(x=np.arange(len(x_3)), y=y_w3, mode='lines', marker=dict(color="navy"), showlegend=False),
        row=3, col=1
    )

    fig.update_layout(height=1200, width=800, title_text="Original (pale) vs. Denoised (dark) sales")
    fig.show()


#------------------------------------------------------------------------------------------
def candle_stick(sales_train_val, calendar):
    d_cols = [c for c in sales_train_val.columns if 'd_' in c]

    store_list = sales_train_val['store_id'].unique()
    
    past_sales = sales_train_val.set_index('id')[d_cols] \
        .T \
        .merge(calendar.set_index('d')['date'],
               left_index=True,
               right_index=True,
                validate='1:1') \
        .set_index('date')

    fig = go.Figure()
    for i, s in enumerate(store_list):
            store_items = [c for c in past_sales.columns if s in c]
            data = past_sales[store_items].sum(axis=1).rolling(90).mean()
            fig.add_trace(go.Box(x=[s]*len(data), y=data, name=s))
    
    fig.update_layout(yaxis_title="Sales", xaxis_title="Time", title="Rolling Average Sales vs. Store name ")
    fig.show()

#------------------------------------------------------------------------------------------
def bar(sales_train_val):
    df = pd.DataFrame(np.transpose([means, store_list]))
    df.columns = ["Mean sales", "Store name"]
    px.bar(df, y="Mean sales", x="Store name", color="Store name", title="Mean sales vs. Store name")
    fig.show()

#------------------------------------------------------------------------------------------
def display_train_val(train_data, val_data):
    fig = make_subplots(rows=3, cols=1)
    fig.add_trace(
        go.Scatter(x=np.arange(70), mode='lines', y=train_data.loc[0].values, marker=dict(color="dodgerblue"), showlegend=False,
                   name="Original signal"),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=np.arange(70, 100), y=val_data.loc[0].values, mode='lines', marker=dict(color="darkorange"), showlegend=False,
                   name="Denoised signal"),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=np.arange(70), mode='lines', y=train_data.loc[1].values, marker=dict(color="dodgerblue"), showlegend=False),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=np.arange(70, 100), y=val_data.loc[1].values, mode='lines', marker=dict(color="darkorange"), showlegend=False),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=np.arange(70), mode='lines', y=train_data.loc[2].values, marker=dict(color="dodgerblue"), showlegend=False),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(x=np.arange(70, 100), y=val_data.loc[2].values, mode='lines', marker=dict(color="darkorange"), showlegend=False),
        row=3, col=1
    )

    fig.update_layout(height=1200, width=800, title_text="Train (blue) vs. Validation (orange) sales")
    fig.show()
