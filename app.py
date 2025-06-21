import streamlit as st
import pandas as pd
import pandas_ta as ta
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
import plotly.graph_objects as go
from datetime import timedelta
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from prophet.diagnostics import cross_validation, performance_metrics
from plotly.subplots import make_subplots
import numpy as np
from keras.layers import LSTM, GRU, Dense, Dropout, Softmax
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Lambda 
from sklearn.preprocessing import MinMaxScaler
from pmdarima import auto_arima
from keras.callbacks import EarlyStopping
from keras.layers import Bidirectional, Attention, Input, Concatenate, Flatten
from keras.models import Model
from sklearn.model_selection import KFold
import tensorflow as tf
from keras.layers import Bidirectional, Input
import time
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")

# Center the title using HTML and markdown
st.markdown(
    """
    <h1 style='text-align: center;'>Stock Technical Analysis & Forecasting App</h1>
    <h1 style='text-align: center;'>By Remah Saber</h1>
    """,
    unsafe_allow_html=True
)
st.info("Make Sure to Upload your stock data in CSV format with columns: Open, High, Low, Closed, Volume, Date Range")
# File uploader for CSV file
uploaded_file = st.file_uploader("Upload Your Stock Data File (CSV File)", type="csv")
# Update progress bar after file upload
if uploaded_file is not None:
    pass  # Placeholder to satisfy indentation requirement

# Call show_final_progress(progress_placeholder, start_time) at the very end of your workflow
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'Date Range' in df.columns:
        df['Date Range'] = pd.to_datetime(df['Date Range'])
        df = df.sort_values('Date Range').reset_index(drop=True)
        st.success("Converted 'Date Range' to datetime and sorted by date.")
    else:
        st.warning("'Date Range' column not found in uploaded file.")
        st.stop()

    # Data preparation
    try:
        df = df[['Open', 'High', 'Low', 'Closed', 'Volume', 'Date Range']]
    except KeyError:
        st.error("CSV must contain columns: Open, High, Low, Closed, Volume, Date Range")
        st.stop()

    df['Previous_Close'] = df['Closed'].shift(1)
    df['Close_shifted'] = df['Closed'].shift(1)
    df['Open_shifted'] = df['Open'].shift(1)
    df['High_shifted'] = df['High'].shift(1)
    df['Low_shifted'] = df['Low'].shift(1)

    # Technical indicators
    df = df.dropna(subset=['Close_shifted'])  # Drop rows with NaN in 'Close_shifted'
    df['SMA_50'] = ta.sma(df['Close_shifted'], length=50)
    df['EMA_50'] = ta.ema(df['Close_shifted'], length=50)
    df['RSI'] = ta.rsi(df['Close_shifted'], length=14)
    macd = ta.macd(df['Close_shifted'], fast=12, slow=26, signal=9)
    if macd is not None and 'MACD_12_26_9' in macd.columns and 'MACDs_12_26_9' in macd.columns:
        df['MACD'] = macd['MACD_12_26_9']
        df['Signal_Line'] = macd['MACDs_12_26_9']
    else:
        df['MACD'] = np.nan
        df['Signal_Line'] = np.nan
    bollinger = ta.bbands(df['Close_shifted'], length=20, std=2)
    if bollinger is not None and all(col in bollinger.columns for col in ['BBU_20_2.0', 'BBM_20_2.0', 'BBL_20_2.0']):
        df['BB_Upper'] = bollinger['BBU_20_2.0']
        df['BB_Middle'] = bollinger['BBM_20_2.0']
        df['BB_Lower'] = bollinger['BBL_20_2.0']
    else:
        df['BB_Upper'] = np.nan
        df['BB_Middle'] = np.nan
        df['BB_Lower'] = np.nan
    stoch = ta.stoch(df['High_shifted'], df['Low_shifted'], df['Close_shifted'], k=14, d=3)
    if stoch is not None and 'STOCHk_14_3_3' in stoch.columns and 'STOCHd_14_3_3' in stoch.columns:
        df['%K'] = stoch['STOCHk_14_3_3']
        df['%D'] = stoch['STOCHd_14_3_3']
    else:
        df['%K'] = np.nan
        df['%D'] = np.nan
    df['ATR'] = ta.atr(df['High_shifted'], df['Low_shifted'], df['Close_shifted'], length=14)
    df.dropna(inplace=True)

    window_size = 20
    indicators = ['SMA_50', 'EMA_50', 'RSI', 'MACD', 'Signal_Line', 'BB_Upper', 'BB_Middle', 'BB_Lower', '%K', '%D', 'ATR', 'Close_shifted', 'Previous_Close']
    results = {indicator: {'predictions': [], 'actual': [], 'daily_mae': []} for indicator in indicators}

    for i in range(window_size, len(df) - 1):
        train_df = df.iloc[i - window_size:i]
        test_index = i + 1
        actual_close_price = df['Closed'].iloc[test_index]
        for indicator in indicators[:-1]:
            X_train = train_df[[indicator, 'Previous_Close']]
            y_train = train_df['Closed']
            X_train = sm.add_constant(X_train, has_constant='add')
            model = sm.OLS(y_train, X_train).fit()
            # Ensure X_test columns match X_train columns
            X_test = pd.DataFrame({indicator: [df[indicator].iloc[test_index]], 'Previous_Close': [df['Previous_Close'].iloc[test_index]]})
            X_test = sm.add_constant(X_test, has_constant='add')
            X_test = X_test[X_train.columns]  # <-- Ensure same column order and names
            prediction = model.predict(X_test)[0]
            results[indicator]['predictions'].append(prediction)
            results[indicator]['actual'].append(actual_close_price)
            daily_mae = mean_absolute_error([actual_close_price], [prediction])
            results[indicator]['daily_mae'].append(daily_mae)
        
    accuracy_data = {'Indicator': [], 'MAE': [], 'MSE': []}
    for indicator in indicators[:-1]:
        if results[indicator]['actual']:
            mae = mean_absolute_error(results[indicator]['actual'], results[indicator]['predictions'])
            mse = mean_squared_error(results[indicator]['actual'], results[indicator]['predictions'])
            accuracy_data['Indicator'].append(indicator)
            accuracy_data['MAE'].append(mae)
            accuracy_data['MSE'].append(mse)
    accuracy_df = pd.DataFrame(accuracy_data).sort_values(by='MAE').reset_index(drop=True)

    # Faceted plot of daily MAE
    y_values = [results[indicator]['daily_mae'] for indicator in indicators[:-1]]
    y_min = min(min(y) for y in y_values)
    y_max = max(max(y) for y in y_values)
    fig_mae = make_subplots(rows=len(indicators)-1, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                            subplot_titles=[f"{indicator} Daily MAE" for indicator in indicators[:-1]])
    for idx, indicator in enumerate(indicators[:-1]):
        fig_mae.add_trace(
            go.Scatter(
                x=df.index[window_size + 1:], 
                y=results[indicator]['daily_mae'],
                mode='lines',
                name=f'{indicator} Daily MAE'
            ),
            row=idx + 1, col=1
        )
    fig_mae.update_yaxes(range=[y_min, y_max])
    fig_mae.update_xaxes(title_text="Index", row=len(indicators)-1, col=1)
    fig_mae.update_layout(
        height=150 * (len(indicators)-1),
        title="Daily MAE of Each Technical Indicator on Closing Price",
        yaxis_title="Daily MAE",
        showlegend=False,
        template="plotly_white"
    )
    #st.plotly_chart(fig_mae, use_container_width=True)

    # Overlay plot
    st.markdown("<h3 style='text-align: center;'>Overlay of Technical Indicators on Close Price</h3>", unsafe_allow_html=True)
    fig_overlay = go.Figure()
    fig_overlay.add_trace(go.Scatter(x=df['Date Range'], y=df['Closed'], mode='lines', name='Close Price', line=dict(color='black', width=1)))
    fig_overlay.add_trace(go.Scatter(x=df['Date Range'], y=df['SMA_50'], mode='lines', name='SMA 50', line=dict(color='yellow', width=1)))
    fig_overlay.add_trace(go.Scatter(x=df['Date Range'], y=df['EMA_50'], mode='lines', name='EMA 50', line=dict(color='orange', width=1)))
    fig_overlay.add_trace(go.Scatter(x=df['Date Range'], y=df['BB_Upper'], mode='lines', name='BB Upper', line=dict(color='blue', width=1, dash='dot')))
    fig_overlay.add_trace(go.Scatter(x=df['Date Range'], y=df['BB_Lower'], mode='lines', name='BB Lower', line=dict(color='blue', width=1, dash='dot')))
    fig_overlay.add_trace(go.Scatter(x=df['Date Range'], y=df['BB_Middle'], mode='lines', name='BB Middle', line=dict(color='blue', width=1)))
    fig_overlay.add_trace(go.Scatter(x=df['Date Range'], y=df['MACD'], mode='lines', name='MACD', line=dict(color='cyan', width=1)))
    fig_overlay.add_trace(go.Scatter(x=df['Date Range'], y=df['Signal_Line'], mode='lines', name='Signal Line', line=dict(color='purple', width=1)))
    fig_overlay.update_layout(
        title="Overlay of Technical Indicators on Close Price",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color="black"),
        width=900,
        height=600
    )
    st.plotly_chart(fig_overlay, use_container_width=True)

    st.markdown("<h3 style='text-align: center;'>Prophet Forecast Using SMA20 & RSI & Stochastic Oscillator & ATR & ADX as Regressors</h3>", unsafe_allow_html=True)
    # Prepare DataFrame for Prophet
    if uploaded_file is not None:
        # Use the uploaded file for Prophet
        prophet_df = df[['Date Range', 'Closed']].rename(columns={'Date Range': 'ds', 'Closed': 'y'}).copy()

        # Add technical indicators as regressors
        prophet_df['SMA20'] = prophet_df['y'].rolling(20).mean()
        prophet_df['RSI14'] = ta.rsi(prophet_df['y'], length=14)
        prophet_df['Stoch_K'] = ta.stoch(prophet_df['y'], prophet_df['y'], prophet_df['y'], k=14, d=3)["STOCHk_14_3_3"]
        prophet_df['ATR'] = ta.atr(prophet_df['y'], prophet_df['y'], prophet_df['y'], length=14)
        prophet_df['ADX'] = ta.adx(prophet_df['y'], prophet_df['y'], prophet_df['y'], length=14)["ADX_14"]
        prophet_df = prophet_df.dropna().reset_index(drop=True)
        def forecast_arima(series, steps=180):
            from pmdarima import auto_arima
            auto_model = auto_arima(series, seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore')
            order = auto_model.order
            model = ARIMA(series, order=order).fit()
            return model.forecast(steps=steps)

        rsi_forecast = forecast_arima(prophet_df["RSI14"])
        sma_forecast = forecast_arima(prophet_df["SMA20"])
        stochk_forecast = forecast_arima(prophet_df["Stoch_K"])
        atr_forecast = forecast_arima(prophet_df["ATR"])
        adx_forecast = forecast_arima(prophet_df["ADX"])

        model = Prophet(seasonality_mode='multiplicative')
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        regressors = ["SMA20", "RSI14", "Stoch_K", "ATR", "ADX"]
        for reg in regressors:
            model.add_regressor(reg)
        model.fit(prophet_df[["ds", "y"] + regressors])

        future = model.make_future_dataframe(periods=180, freq='B')
        future = future.merge(prophet_df[["ds"] + regressors], on="ds", how="left")
        future.loc[future["SMA20"].isna(), "SMA20"] = sma_forecast.values
        future.loc[future["RSI14"].isna(), "RSI14"] = rsi_forecast.values
        future.loc[future["Stoch_K"].isna(), "Stoch_K"] = stochk_forecast.values
        future.loc[future["ATR"].isna(), "ATR"] = atr_forecast.values
        future.loc[future["ADX"].isna(), "ADX"] = adx_forecast.values

        forecast = model.predict(future)

        df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='180 days')
        df_p = performance_metrics(df_cv)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prophet_df["ds"], y=prophet_df["y"], name="Actual Price"))
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast"))
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(
            x=forecast["ds"], y=forecast["yhat_lower"],
            line=dict(width=0), fill='tonexty',
            fillcolor='rgba(0,100,80,0.2)', showlegend=False
        ))
        fig.update_layout(title="Prophet Forecast", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)
        st.write("Model Performance:")
        st.dataframe(df_p[["horizon", "mape", "rmse"]])
        st.write("Forecasted Values:")


    # --- Predict next 6 months using ARIMA with all indicators as exogenous variables ---
    st.markdown("<h3 style='text-align: center;'>ARIMA Forecast All Indicators as Exogenous Variables Using RSI & MACD & SMA & EMA & Bollinger Bands & Stochastic Oscillator & ATR</h3>", unsafe_allow_html=True)
    if uploaded_file is not None:
        # Prepare endogenous and exogenous variables
        arima_series = df.set_index('Date Range')['Closed']
        exog_cols = ['SMA_50', 'EMA_50', 'RSI', 'MACD', 'Signal_Line', 'BB_Upper', 'BB_Middle', 'BB_Lower', '%K', '%D', 'ATR']
        exog = df.set_index('Date Range')[exog_cols]
        # Ensure exogenous variables are aligned with the series
        if len(arima_series) != len(exog):
            st.error("Length of ARIMA series and exogenous variables do not match. Please check your data.")
            st.stop()
        # Set frequency if not already set
        if arima_series.index.freq is None:
            inferred_freq = pd.infer_freq(arima_series.index)
            if inferred_freq:
                arima_series = arima_series.asfreq(inferred_freq)
                exog = exog.asfreq(inferred_freq)
            else:
                arima_series = arima_series.asfreq('B')
                exog = exog.asfreq('B')

        # Fill missing values in exogenous variables
        exog = exog.fillna(method='ffill').fillna(method='bfill')

        # Check for stationarity and difference if needed
        from statsmodels.tsa.stattools import adfuller
        adf_result = adfuller(arima_series.dropna())
        diffed = False
        if adf_result[1] > 0.05:
            st.warning("Series is non-stationary (ADF p-value > 0.05). Differencing will be applied.")
            arima_series = arima_series.diff().dropna()
            exog = exog.iloc[1:]
            diffed = True

        # Use auto_arima to select best order with exogenous variables
        try:
            auto_model = auto_arima(
                arima_series, exogenous=exog, seasonal=False, stepwise=True,
                suppress_warnings=True, error_action='ignore', max_p=5, max_q=5, max_d=2
            )
            order = auto_model.order
            st.info(f"Auto-selected ARIMA order: {order}")
        except Exception as e:
            order = (3, 1, 0)
            st.warning(f"auto_arima failed, using default ARIMA{order}")

        # Prepare exogenous variables for forecasting
        future_periods = 126
        last_date = df['Date Range'].max()
        freq = pd.infer_freq(df['Date Range'])
        if freq is None:
            freq = 'B'
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=future_periods, freq=freq)

        # For exogenous variables, forecast using last available value (or rolling mean)
        last_exog = exog.iloc[-1]
        future_exog = pd.DataFrame([last_exog.values] * future_periods, columns=exog_cols, index=future_dates)
        
        # Align indices (intersection only) before fitting ARIMA
        arima_series, exog = arima_series.align(exog, join='inner', axis=0)

        # Fit ARIMA with exogenous variables
        arima_model = ARIMA(arima_series, order=order, exog=exog)
        arima_result = arima_model.fit()

        # Forecast
        arima_forecast = arima_result.forecast(steps=future_periods, exog=future_exog)

        # If differenced, invert the differencing
        if diffed:
            last_value = df['Closed'].iloc[-1]
            arima_forecast = last_value + arima_forecast.cumsum()

        # Plot
        fig_arima = go.Figure()
        fig_arima.add_trace(go.Scatter(x=df['Date Range'], y=df['Closed'], name="Actual Price"))
        fig_arima.add_trace(go.Scatter(x=future_dates, y=arima_forecast, name="ARIMAX Forecast (with indicators)"))
        fig_arima.update_layout(title="ARIMA Forecast with Exogenous Variables", xaxis_title="Date", yaxis_title="Close Price")
        st.plotly_chart(fig_arima, use_container_width=True)

        # Show forecasted values
        arima_forecast_df = pd.DataFrame({'Date': future_dates, 'ARIMA_Predicted_Close': arima_forecast.to_numpy()})
        st.dataframe(arima_forecast_df)

    # --- Predict next 6 months using Deep Learning (LSTM, with more robust features and improved validation) ---
    st.markdown("<h3 style='text-align: center;'>LSTM Deep Learning Prediction using Momentum & RSI & SMA_20 & EMA_20 & MACD & ATR & Volume & Change in Price</h3>", unsafe_allow_html=True)

    # Feature engineering: Add more features for LSTM
    required_cols = ['Closed', 'High', 'Low', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Column '{col}' not found in DataFrame. Please check your CSV file.")
            st.stop()
    df['Momentum_10'] = df['Closed'] - df['Closed'].shift(10)
    df['RSI_14'] = ta.rsi(df['Closed'], length=14)
    df['SMA_20'] = ta.sma(df['Closed'], length=20)
    df['EMA_20'] = ta.ema(df['Closed'], length=20)
    macd_result = ta.macd(df['Closed'], fast=12, slow=26, signal=9)
    if macd_result is not None:
        if isinstance(macd_result, pd.DataFrame) and 'MACD_12_26_9' in macd_result.columns:
            df['MACD'] = macd_result['MACD_12_26_9']
        elif isinstance(macd_result, pd.Series):
            df['MACD'] = macd_result
        else:
            df['MACD'] = np.nan
    else:
        df['MACD'] = np.nan
    df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Closed'], length=14)
    df['Log_Volume'] = np.log1p(df['Volume'])
    df['Returns'] = df['Closed'].pct_change()
    # Preserve 'Date Range' column after dropping NaNs for correct indexing and plotting
    keep_cols = ['Date Range', 'Closed', 'High', 'Low', 'Volume', 'Momentum_10', 'RSI_14', 'SMA_20', 'EMA_20', 'MACD', 'ATR_14', 'Log_Volume', 'Returns']
    df = df.dropna(subset=['Momentum_10', 'RSI_14', 'SMA_20', 'EMA_20', 'MACD', 'ATR_14', 'Log_Volume', 'Returns', 'Closed'])
    missing_cols = [col for col in keep_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns after feature engineering: {missing_cols}")
        st.stop()
    if df.empty:
        st.warning("No data left after feature engineering for LSTM prediction. Please check your input data.")
        st.stop()
    df = df[keep_cols].reset_index(drop=True)

    feature_cols = ['Momentum_10', 'RSI_14', 'SMA_20', 'EMA_20', 'MACD', 'ATR_14', 'Log_Volume', 'Returns']

    # Check if DataFrame is empty after dropping NaNs
    if df.empty or len(df) < 80:
        st.warning("Not enough data after feature engineering for LSTM prediction. Please upload a longer time series.")
    else:
        X = df[feature_cols].values
        y = df['Closed'].values

        # Prepare sequences
        seq_len = 40
        X_seq, y_seq = [], []
        for i in range(seq_len, len(X)):
            X_seq.append(X[i-seq_len:i])
            y_seq.append(y[i])
        X_seq, y_seq = np.array(X_seq), np.array(y_seq)

        # Train/val split (time-based, no shuffle)
        split_idx = int(len(X_seq) * 0.85)
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

        # Scaling
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        scaler_X.fit(X.reshape(-1, X.shape[-1]))
        X_train_scaled = scaler_X.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled = scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
        y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))

        # Improved LSTM model: deeper, dropout, bidirectional, regularization
        def build_lstm(input_shape):
            model = Sequential([
                Bidirectional(LSTM(64, return_sequences=True, dropout=0.25, recurrent_dropout=0.1), input_shape=input_shape),
                LSTM(32, return_sequences=False, dropout=0.25, recurrent_dropout=0.1),
                Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                Dropout(0.25),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            return model

        lstm_model = build_lstm((seq_len, len(feature_cols)))
        early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        history = lstm_model.fit(
            X_train_scaled, y_train_scaled,
            epochs=120, batch_size=16, verbose=0,
            validation_data=(X_val_scaled, y_val_scaled),
            callbacks=[early_stop]
        )

        # Validation performance
        lstm_val_pred = scaler_y.inverse_transform(lstm_model.predict(X_val_scaled, verbose=0))
        y_val_true = scaler_y.inverse_transform(y_val_scaled)
        st.write(f"LSTM Validation MAE: {mean_absolute_error(y_val_true, lstm_val_pred):.3f}")
        rmse = np.sqrt(mean_squared_error(y_val_true, lstm_val_pred))
        st.write(f"LSTM Validation RMSE: {rmse:.3f}")
        # Plot validation predictions
        val_plot = pd.DataFrame({
            'Actual': y_val_true.flatten(),
            'Predicted': lstm_val_pred.flatten()
        }, index=df['Date Range'].iloc[-len(y_val_true):])
        st.line_chart(val_plot)

        # Forecast next 6 months (126 business days)
        future_periods = 126
        if 'Date Range' in df.columns:
            # Ensure 'Date Range' is datetime
            df['Date Range'] = pd.to_datetime(df['Date Range'])
            last_date = df['Date Range'].max()
            freq = pd.infer_freq(df['Date Range'])
            if freq is None:
                freq = 'B'
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=future_periods, freq=freq)
        else:
            st.error("'Date Range' column not found in DataFrame. Please check your CSV file.")
            future_dates = None

        # Rolling forecast
        preds = []
        if future_dates is not None:
            history_df = df.copy()
            for i in range(len(future_dates)):
                last_rows = history_df.iloc[-seq_len:]
                X_input = last_rows[feature_cols].values
                if X_input.shape[0] == 0:
                    # If there is no data, fill with zeros
                    X_input = np.zeros((seq_len, len(feature_cols)))
                elif X_input.shape[0] < seq_len:
                    pad = np.tile(X_input[0], (seq_len - X_input.shape[0], 1))
                    X_input = np.vstack([pad, X_input])
                X_input_scaled = scaler_X.transform(X_input).reshape(1, seq_len, len(feature_cols))
                pred_scaled = lstm_model.predict(X_input_scaled, verbose=0)[0][0]
                pred = scaler_y.inverse_transform(np.array([[pred_scaled]]))[0][0]
                preds.append(pred)
                # Add new row to history for next prediction
                new_row = history_df.iloc[-1].copy(deep=True)
                new_row['Date Range'] = future_dates[i]
                new_row['Closed'] = pred
                # Set 'High' and 'Low' to previous values if missing (for ATR calculation)
                if pd.isnull(new_row['High']):
                    new_row['High'] = history_df['High'].iloc[-1]
                if pd.isnull(new_row['Low']):
                    new_row['Low'] = history_df['Low'].iloc[-1]
                # Update features for new row
                closes = history_df['Closed'].tolist() + [pred]
                if len(closes) > 10:
                    new_row['Momentum_10'] = pred - closes[-11]
                else:
                    new_row['Momentum_10'] = pred - closes[0]
                # RSI
                if len(closes) >= 15:
                    delta = np.diff(closes[-15:])
                    up = np.where(delta > 0, delta, 0)
                    down = np.where(delta < 0, -delta, 0)
                    avg_gain = np.mean(up[-14:]) if np.any(up[-14:]) else 0
                    avg_loss = np.mean(down[-14:]) if np.any(down[-14:]) else 0
                    if avg_loss != 0:
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                    else:
                        rsi = 100
                else:
                    rsi = 50
                new_row['RSI_14'] = rsi
                # SMA/EMA
                if len(closes) >= 20:
                    new_row['SMA_20'] = np.mean(closes[-20:])
                    prev_ema = history_df['EMA_20'].iloc[-1] if not pd.isnull(history_df['EMA_20'].iloc[-1]) else np.mean(closes[-20:])
                    new_row['EMA_20'] = (pred * (2/(20+1))) + (prev_ema * (1 - (2/(20+1))))
                else:
                    new_row['SMA_20'] = np.mean(closes)
                    new_row['EMA_20'] = pred
                # MACD
                if len(closes) >= 26:
                    ema12 = pd.Series(closes[-26:]).ewm(span=12, adjust=False).mean().iloc[-1]
                    ema26 = pd.Series(closes[-26:]).ewm(span=26, adjust=False).mean().iloc[-1]
                    new_row['MACD'] = ema12 - ema26
                else:
                    new_row['MACD'] = 0
                # ATR
                if len(history_df) >= 14:
                    highs = history_df['High'].tolist()[-13:] + [new_row['High']]
                    lows = history_df['Low'].tolist()[-13:] + [new_row['Low']]
                    closes_prev = history_df['Closed'].tolist()[-14:]
                    tr1 = np.array(highs) - np.array(lows)
                    tr2 = np.abs(np.array(highs) - np.array(closes_prev))
                    tr3 = np.abs(np.array(lows) - np.array(closes_prev))
                    tr = np.maximum.reduce([tr1, tr2, tr3])
                    new_row['ATR_14'] = np.mean(tr)
                else:
                    new_row['ATR_14'] = 0
                # Log_Volume and Returns
                new_row['Log_Volume'] = np.log1p(history_df['Volume'].iloc[-1])
                # Calculate Returns for the new row
                prev_close = history_df['Closed'].iloc[-1] if len(history_df) > 0 else pred
                if prev_close != 0:
                    new_row['Returns'] = (pred - prev_close) / prev_close
                else:
                    new_row['Returns'] = 0.0
                for col in history_df.columns:
                    if col not in new_row or pd.isnull(new_row[col]):
                        # Fill missing columns with 0 or np.nan as appropriate
                        if col in ['Momentum_10', 'RSI_14', 'SMA_20', 'EMA_20', 'MACD', 'ATR_14', 'Log_Volume', 'Returns']:
                            new_row[col] = 0.0
                        elif col == 'Date Range':
                            new_row[col] = future_dates[i]
                        elif col == 'Closed':
                            new_row[col] = pred
                        elif col in ['High', 'Low', 'Volume']:
                            new_row[col] = history_df[col].iloc[-1] if len(history_df) > 0 else 0.0
                        else:
                            new_row[col] = np.nan
                new_row_df = pd.DataFrame([new_row])
                history_df = pd.concat([history_df, new_row_df], ignore_index=True)

            # Adjust predictions to flow with the trend
            # Compute the trend from the last N days (e.g., 60) and apply a correction to the forecast
            N = 60
            if len(df) > N:
                trend_slope = (df['Closed'].iloc[-1] - df['Closed'].iloc[-N]) / N
            else:
                trend_slope = (df['Closed'].iloc[-1] - df['Closed'].iloc[0]) / max(1, len(df)-1)
            # Apply cumulative trend to predictions
            preds_trend = []
            for i, p in enumerate(preds):
                preds_trend.append(p + trend_slope * (i+1))
            preds = preds_trend

            future_df = pd.DataFrame({
                'Date': future_dates,
                'LSTM_Predicted_Close': np.array(preds)
            })
            future_df = future_df.set_index('Date')
            future_df = future_df[~future_df.index.duplicated(keep='first')]
            st.line_chart(future_df)
            st.write("STM Deep Learning Prediction using Momentum & RSI & SMA_20 & EMA_20 & MACD & ATR & Volume & Change in Price:")
            st.dataframe(future_df)
            lstm_preds = pd.Series(np.array(preds), index=future_dates)
 

    # --- Best Month to Buy/Sell & Trend Detection (Prophet, ARIMA, LSTM) ---
    st.markdown("<h3 style='text-align: center;'>Best Month to Buy/Sell & Trend Detection</h3>", unsafe_allow_html=True)

    # Helper: get monthly average price and returns
    def monthly_stats(series, dates):
        df_month = pd.DataFrame({'Date': dates, 'Value': series})
        df_month['Month'] = pd.to_datetime(df_month['Date']).dt.month
        df_month['Year'] = pd.to_datetime(df_month['Date']).dt.year
        # Monthly average price
        monthly_avg = df_month.groupby(['Year', 'Month'])['Value'].mean().reset_index()
        # Monthly return (last of month vs previous month)
        monthly_last = df_month.groupby(['Year', 'Month'])['Value'].last().pct_change().reset_index()
        monthly_last.rename(columns={'Value': 'Return'}, inplace=True)
        return monthly_avg, monthly_last

    # Prophet: Use historical and forecast for recommendations
    if uploaded_file is not None and 'forecast' in locals():
        # Use historical + forecast for best month
        prophet_hist = prophet_df[['ds', 'y']].rename(columns={'ds': 'Date', 'y': 'Value'})
        prophet_fore = forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Value'})
        prophet_all = pd.concat([prophet_hist, prophet_fore], ignore_index=True)
        prophet_all = prophet_all.drop_duplicates(subset='Date', keep='last')
        prophet_all = prophet_all.sort_values('Date')
        prophet_month_avg, prophet_month_ret = monthly_stats(prophet_all['Value'], prophet_all['Date'])
        # Best buy: month with lowest average price
        best_buy_month_prophet = prophet_month_avg.groupby('Month')['Value'].mean().idxmin()
        best_sell_month_prophet = prophet_month_avg.groupby('Month')['Value'].mean().idxmax()
        st.write(f"**Prophet Recommendation (historical+forecast):**")
        st.write(f"Best month to BUY: **{best_buy_month_prophet}** (lowest avg price)")
        st.write(f"Best month to SELL: **{best_sell_month_prophet}** (highest avg price)")
        # Trend detection (last 1 year of forecast)
        last_12 = prophet_all['Value'].iloc[-252:]
        if last_12.iloc[-1] > last_12.iloc[0] * 1.03:
            trend_prophet = "Uptrend"
        elif last_12.iloc[-1] < last_12.iloc[0] * 0.97:
            trend_prophet = "Downtrend"
        else:
            trend_prophet = "Sideways"
        st.write(f"Prophet detected trend: **{trend_prophet}**")

    # ARIMA: Use historical and forecast for recommendations
    if uploaded_file is not None and 'arima_forecast' in locals():
        arima_hist = df[['Date Range', 'Closed']].rename(columns={'Date Range': 'Date', 'Closed': 'Value'})
        arima_fore = pd.DataFrame({'Date': future_dates, 'Value': arima_forecast})
        arima_all = pd.concat([arima_hist, arima_fore], ignore_index=True)
        arima_all = arima_all.drop_duplicates(subset='Date', keep='last')
        arima_all = arima_all.sort_values('Date')
        arima_month_avg, arima_month_ret = monthly_stats(arima_all['Value'], arima_all['Date'])
        best_buy_month_arima = arima_month_avg.groupby('Month')['Value'].mean().idxmin()
        best_sell_month_arima = arima_month_avg.groupby('Month')['Value'].mean().idxmax()
        st.write(f"**ARIMA Recommendation (historical+forecast):**")
        st.write(f"Best month to BUY: **{best_buy_month_arima}** (lowest avg price)")
        st.write(f"Best month to SELL: **{best_sell_month_arima}** (highest avg price)")
        # Trend detection
        last_12 = arima_all['Value'].iloc[-252:]
        if last_12.iloc[-1] > last_12.iloc[0] * 1.03:
            trend_arima = "Uptrend"
        elif last_12.iloc[-1] < last_12.iloc[0] * 0.97:
            trend_arima = "Downtrend"
        else:
            trend_arima = "Sideways"
        st.write(f"ARIMA detected trend: **{trend_arima}**")

    # LSTM: Use historical and forecast for recommendations
    if uploaded_file is not None and 'lstm_preds' in locals() and 'future_dates' in locals():
        lstm_hist = df[['Date Range', 'Closed']].rename(columns={'Date Range': 'Date', 'Closed': 'Value'})
        lstm_fore = pd.DataFrame({'Date': future_dates, 'Value': lstm_preds})
        lstm_all = pd.concat([lstm_hist, lstm_fore], ignore_index=True)
        lstm_all = lstm_all.drop_duplicates(subset='Date', keep='last')
        lstm_all = lstm_all.sort_values('Date')
        lstm_month_avg, lstm_month_ret = monthly_stats(lstm_all['Value'], lstm_all['Date'])
        best_buy_month_lstm = lstm_month_avg.groupby('Month')['Value'].mean().idxmin()
        best_sell_month_lstm = lstm_month_avg.groupby('Month')['Value'].mean().idxmax()
        st.write(f"**LSTM Recommendation (historical+forecast):**")
        st.write(f"Best month to BUY: **{best_buy_month_lstm}** (lowest avg price)")
        st.write(f"Best month to SELL: **{best_sell_month_lstm}** (highest avg price)")
        # Trend detection
        last_12 = lstm_all['Value'].iloc[-252:]
        if last_12.iloc[-1] > last_12.iloc[0] * 1.03:
            trend_lstm = "Uptrend"
        elif last_12.iloc[-1] < last_12.iloc[0] * 0.97:
            trend_lstm = "Downtrend"
        else:
            trend_lstm = "Sideways"
        st.write(f"LSTM detected trend: **{trend_lstm}**")
st.info("Please upload a CSV file to begin analysis.")