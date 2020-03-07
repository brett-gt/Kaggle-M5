

#--------------------------------------------------------------------------------
def ARIMA(train_data, val_data):
    predictions = []
    for row in tqdm(train_data[train_data.columns[-30:]].values[:3]):
        fit = sm.tsa.statespace.SARIMAX(row, seasonal_order=(0, 1, 1, 7)).fit()
        predictions.append(fit.forecast(30))
    predictions = np.array(predictions).reshape((-1, 30))
    error_arima = np.linalg.norm(predictions[:3] - val_data.values[:3])/len(predictions[0])

#--------------------------------------------------------------------------------
def plot_ARIMA(predictions, train_data, val_data):
    pred_1 = predictions[0]
    pred_2 = predictions[1]
    pred_3 = predictions[2]

    fig = make_subplots(rows=3, cols=1)

    fig.add_trace(
        go.Scatter(x=np.arange(70), mode='lines', y=train_data.loc[0].values, marker=dict(color="dodgerblue"),
                   name="Train"),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=np.arange(70, 100), y=val_data.loc[0].values, mode='lines', marker=dict(color="darkorange"),
                   name="Val"),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=np.arange(70, 100), y=pred_1, mode='lines', marker=dict(color="seagreen"),
                   name="Pred"),
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
        go.Scatter(x=np.arange(70, 100), y=pred_2, mode='lines', marker=dict(color="seagreen"), showlegend=False,
                   name="Denoised signal"),
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

    fig.add_trace(
        go.Scatter(x=np.arange(70, 100), y=pred_3, mode='lines', marker=dict(color="seagreen"), showlegend=False,
                   name="Denoised signal"),
        row=3, col=1
    )

    fig.update_layout(height=1200, width=800, title_text="ARIMA")
    fig.show()
