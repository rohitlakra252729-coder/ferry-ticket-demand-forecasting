
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Ferry Demand Forecasting",
    page_icon="⛴️",
    layout="wide"
)

st.title("⛴️ Short-Term Ferry Ticket Demand Forecasting")
st.subheader("Toronto Island Ferry | Predictive Decision Support System")

st.markdown(
    """
    Forecast ticket sales and redemptions for the next 15 minutes
    to 2 hours to support scheduling, staffing, and crowd management.
    """
)

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------
@st.cache_data
def load_data():
    file_name = "Toronto_Island_Ferry_Tickets.csv"
    data = pd.read_csv(file_name)
    data.columns = data.columns.str.strip()

    required = ["Timestamp", "Sales Count", "Redemption Count"]
    missing = [col for col in required if col not in data.columns]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + ". Available columns: "
            + ", ".join(data.columns)
        )

    data["Timestamp"] = pd.to_datetime(
        data["Timestamp"], errors="coerce"
    )
    data["Sales Count"] = pd.to_numeric(
        data["Sales Count"], errors="coerce"
    )
    data["Redemption Count"] = pd.to_numeric(
        data["Redemption Count"], errors="coerce"
    )

    data = data.dropna(
        subset=["Timestamp", "Sales Count", "Redemption Count"]
    )
    data = data.sort_values("Timestamp")
    data = data.drop_duplicates(subset=["Timestamp"], keep="last")

    data = data.set_index("Timestamp")

    # Standardize observations to 15-minute intervals.
    data = data[["Sales Count", "Redemption Count"]].resample("15min").mean()

    # Interpolate internal missing intervals only.
    data = data.interpolate(method="time", limit_area="inside")
    data = data.dropna()
    data = data.clip(lower=0)

    return data


try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "Dataset not found. Place Toronto_Island_Ferry_Tickets.csv "
        "in the same repository folder as app.py."
    )
    st.stop()
except Exception as error:
    st.error(f"Unable to load the dataset: {error}")
    st.stop()

if len(df) < 30:
    st.error("The dataset has too few valid observations for reliable forecasting.")
    st.stop()

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------
st.sidebar.header("Forecast Controls")

target = st.sidebar.selectbox(
    "Select demand measure",
    ["Sales Count", "Redemption Count"]
)

model_name = st.sidebar.selectbox(
    "Select forecasting model",
    [
        "Naive Forecast",
        "Moving Average",
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting"
    ]
)

horizon_label = st.sidebar.selectbox(
    "Forecast horizon",
    ["15 minutes", "30 minutes", "1 hour", "2 hours"]
)

horizon_map = {
    "15 minutes": 1,
    "30 minutes": 2,
    "1 hour": 4,
    "2 hours": 8
}
steps = horizon_map[horizon_label]

st.sidebar.caption("Each forecasting step represents 15 minutes.")

# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------
def create_features(series):
    frame = pd.DataFrame({"target": series})

    for lag in [1, 2, 4, 8]:
        frame[f"lag_{lag}"] = series.shift(lag)

    previous = series.shift(1)

    frame["rolling_mean_4"] = previous.rolling(4).mean()
    frame["rolling_mean_8"] = previous.rolling(8).mean()
    frame["rolling_std_4"] = previous.rolling(4).std()
    frame["rolling_max_4"] = previous.rolling(4).max()

    idx = frame.index
    frame["hour"] = idx.hour
    frame["minute"] = idx.minute
    frame["day_of_week"] = idx.dayofweek
    frame["month"] = idx.month
    frame["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    return frame.dropna()


def build_model(name):
    if name == "Linear Regression":
        return LinearRegression()

    if name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

    return GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=42
    )


def forecast_values(history, name, number_of_steps):
    history = history.astype(float).copy()
    predictions = []

    feature_columns = [
        "lag_1", "lag_2", "lag_4", "lag_8",
        "rolling_mean_4", "rolling_mean_8",
        "rolling_std_4", "rolling_max_4",
        "hour", "minute", "day_of_week",
        "month", "is_weekend"
    ]

    training_frame = create_features(history)

    if name in ["Linear Regression", "Random Forest", "Gradient Boosting"]:
        if len(training_frame) < 20:
            raise ValueError(
                "Not enough historical observations to train this model."
            )

        X_train = training_frame[feature_columns]
        y_train = training_frame["target"]
        model = build_model(name)
        model.fit(X_train, y_train)

    for _ in range(number_of_steps):
        next_time = history.index[-1] + pd.Timedelta(minutes=15)

        if name == "Naive Forecast":
            prediction = history.iloc[-1]

        elif name == "Moving Average":
            prediction = history.iloc[-4:].mean()

        else:
            recent = history.iloc[-8:]
            row = {
                "lag_1": history.iloc[-1],
                "lag_2": history.iloc[-2],
                "lag_4": history.iloc[-4],
                "lag_8": history.iloc[-8],
                "rolling_mean_4": recent.iloc[-4:].mean(),
                "rolling_mean_8": recent.mean(),
                "rolling_std_4": recent.iloc[-4:].std(),
                "rolling_max_4": recent.iloc[-4:].max(),
                "hour": next_time.hour,
                "minute": next_time.minute,
                "day_of_week": next_time.dayofweek,
                "month": next_time.month,
                "is_weekend": int(next_time.dayofweek >= 5)
            }
            prediction = model.predict(
                pd.DataFrame([row], columns=feature_columns)
            )[0]

        prediction = max(0.0, float(prediction))
        predictions.append((next_time, prediction))

        history.loc[next_time] = prediction

    return pd.Series(
        [value for _, value in predictions],
        index=pd.DatetimeIndex([time for time, _ in predictions]),
        name="Forecast"
    )


# --------------------------------------------------
# HISTORICAL ANALYSIS
# --------------------------------------------------
st.header("1. Historical Demand Overview")

total_sales = df["Sales Count"].sum()
total_redemptions = df["Redemption Count"].sum()
average_sales = df["Sales Count"].mean()
average_redemptions = df["Redemption Count"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Tickets Sold", f"{total_sales:,.0f}")
c2.metric("Total Redemptions", f"{total_redemptions:,.0f}")
c3.metric("Average Sales / 15 min", f"{average_sales:,.1f}")
c4.metric("Average Redemptions / 15 min", f"{average_redemptions:,.1f}")

history_chart = go.Figure()
history_chart.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Sales Count"],
        mode="lines",
        name="Ticket Sales"
    )
)
history_chart.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Redemption Count"],
        mode="lines",
        name="Ticket Redemptions"
    )
)
history_chart.update_layout(
    title="Historical Ferry Ticket Activity",
    xaxis_title="Timestamp",
    yaxis_title="Tickets per 15-minute interval",
    hovermode="x unified"
)
st.plotly_chart(history_chart, use_container_width=True)

# --------------------------------------------------
# TIME PATTERN ANALYSIS
# --------------------------------------------------
st.header("2. Demand Patterns")

pattern_tab1, pattern_tab2 = st.tabs(
    ["Hourly Demand", "Day-of-Week Demand"]
)

with pattern_tab1:
    hourly = df.groupby(df.index.hour)[
        ["Sales Count", "Redemption Count"]
    ].mean().reset_index()
    hourly = hourly.rename(columns={"Timestamp": "Hour"})
    hourly.columns = ["Hour", "Sales Count", "Redemption Count"]

    fig_hour = px.line(
        hourly,
        x="Hour",
        y=["Sales Count", "Redemption Count"],
        markers=True,
        title="Average Demand by Hour of Day"
    )
    st.plotly_chart(fig_hour, use_container_width=True)

with pattern_tab2:
    weekday = df.groupby(df.index.dayofweek)[
        ["Sales Count", "Redemption Count"]
    ].mean().reset_index()
    weekday.columns = ["Day", "Sales Count", "Redemption Count"]
    weekday["Day"] = weekday["Day"].map({
        0: "Monday", 1: "Tuesday", 2: "Wednesday",
        3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"
    })

    fig_day = px.bar(
        weekday,
        x="Day",
        y=["Sales Count", "Redemption Count"],
        barmode="group",
        title="Average Demand by Day of Week"
    )
    st.plotly_chart(fig_day, use_container_width=True)

# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------
st.header("3. Forecast Model Evaluation")

series = df[target].astype(float)
features = create_features(series)

feature_columns = [
    "lag_1", "lag_2", "lag_4", "lag_8",
    "rolling_mean_4", "rolling_mean_8",
    "rolling_std_4", "rolling_max_4",
    "hour", "minute", "day_of_week",
    "month", "is_weekend"
]

split_at = int(len(features) * 0.8)
train = features.iloc[:split_at]
test = features.iloc[split_at:]

if len(train) < 20 or len(test) < 5:
    st.warning("The dataset is small; model evaluation may be unreliable.")
    st.stop()

actual = test["target"]
evaluation = []

# Baseline models use only information available before each test interval.
naive_pred = test["lag_1"].clip(lower=0)
moving_pred = test["rolling_mean_4"].clip(lower=0)

evaluation.append(("Naive Forecast", naive_pred))
evaluation.append(("Moving Average", moving_pred))

for candidate in [
    "Linear Regression", "Random Forest", "Gradient Boosting"
]:
    try:
        candidate_model = build_model(candidate)
        candidate_model.fit(train[feature_columns], train["target"])
        candidate_pred = np.maximum(
            0,
            candidate_model.predict(test[feature_columns])
        )
        evaluation.append(
            (candidate, pd.Series(candidate_pred, index=test.index))
        )
    except Exception:
        continue


def calculate_metrics(actual_values, predicted_values):
    actual_values = np.asarray(actual_values, dtype=float)
    predicted_values = np.asarray(predicted_values, dtype=float)

    mae = mean_absolute_error(actual_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))

    nonzero = actual_values != 0
    if nonzero.any():
        mape = (
            np.abs(
                (actual_values[nonzero] - predicted_values[nonzero])
                / actual_values[nonzero]
            ).mean() * 100
        )
    else:
        mape = np.nan

    return mae, rmse, mape


metric_rows = []
for candidate, prediction in evaluation:
    mae, rmse, mape = calculate_metrics(actual, prediction)
    metric_rows.append({
        "Model": candidate,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape
    })

metrics_df = pd.DataFrame(metric_rows).sort_values("MAE")
st.dataframe(
    metrics_df.round(2),
    use_container_width=True,
    hide_index=True
)

best_model_name = metrics_df.iloc[0]["Model"]
st.info(f"Lowest holdout MAE in this evaluation: {best_model_name}")

metric_chart = px.bar(
    metrics_df,
    x="Model",
    y="MAE",
    title="Model Comparison — Mean Absolute Error (Lower Is Better)"
)
st.plotly_chart(metric_chart, use_container_width=True)

st.caption(
    "Evaluation uses a chronological 80/20 split and one-step-ahead "
    "predictions on the holdout period. MAPE excludes zero-valued actuals. "
    "These scores do not guarantee future forecast accuracy."
)

# --------------------------------------------------
# FUTURE FORECAST
# --------------------------------------------------
st.header("4. Short-Term Demand Forecast")

try:
    future = forecast_values(series, model_name, steps)
except Exception as error:
    st.error(f"Forecast could not be generated: {error}")
    st.stop()

# Approximate error band based on the selected model's holdout residuals.
selected_eval = next(
    (pred for name, pred in evaluation if name == model_name),
    None
)

if selected_eval is not None:
    residuals = actual - selected_eval
    error_scale = float(residuals.std())
else:
    error_scale = float((actual - test["lag_1"]).std())

if not np.isfinite(error_scale):
    error_scale = 0.0

step_numbers = np.arange(1, len(future) + 1)
band = 1.96 * error_scale * np.sqrt(step_numbers)

lower = np.maximum(0, future.values - band)
upper = future.values + band

forecast_chart = go.Figure()

forecast_chart.add_trace(
    go.Scatter(
        x=df.index[-min(96, len(df)):],
        y=series.iloc[-min(96, len(df)):],
        mode="lines",
        name="Recent Actual"
    )
)

forecast_chart.add_trace(
    go.Scatter(
        x=future.index,
        y=upper,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        name="Upper uncertainty bound"
    )
)

forecast_chart.add_trace(
    go.Scatter(
        x=future.index,
        y=lower,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(99,110,250,0.18)",
        name="Approximate uncertainty band"
    )
)

forecast_chart.add_trace(
    go.Scatter(
        x=future.index,
        y=future.values,
        mode="lines+markers",
        name="Forecast"
    )
)

forecast_chart.update_layout(
    title=f"{target} Forecast — Next {horizon_label}",
    xaxis_title="Timestamp",
    yaxis_title="Tickets per 15-minute interval",
    hovermode="x unified"
)
st.plotly_chart(forecast_chart, use_container_width=True)

forecast_table = pd.DataFrame({
    "Forecast Timestamp": future.index,
    "Predicted Demand": np.round(future.values, 2),
    "Approx. Lower Bound": np.round(lower, 2),
    "Approx. Upper Bound": np.round(upper, 2)
})

st.dataframe(
    forecast_table,
    use_container_width=True,
    hide_index=True
)

st.metric(
    f"Total predicted demand over next {horizon_label}",
    f"{future.sum():,.1f} tickets"
)

# --------------------------------------------------
# OPERATIONAL INSIGHTS
# --------------------------------------------------
st.header("5. Operational Decision Support")

last_value = float(series.iloc[-1])
average_value = float(series.tail(min(96, len(series))).mean())
next_value = float(future.iloc[0])

if next_value > average_value * 1.25:
    st.warning(
        "Potential demand increase: consider checking staffing, queue "
        "readiness, and ferry dispatch capacity."
    )
elif next_value < average_value * 0.75:
    st.info(
        "Forecast demand is below the recent average. Review staffing "
        "flexibility while maintaining required safety coverage."
    )
else:
    st.success(
        "Forecast demand is near the recent average. Continue routine "
        "monitoring and compare forecasts with incoming ticket data."
    )

st.write(f"- Most recent interval: **{last_value:.1f} tickets**")
st.write(f"- Recent average per interval: **{average_value:.1f} tickets**")
st.write(f"- Next-interval forecast: **{next_value:.1f} tickets**")

st.caption(
    "Planning support only. Forecasts and uncertainty bands are estimates, "
    "not guarantees. Operational decisions should also consider weather, "
    "service capacity, safety rules, and live conditions."
)

# --------------------------------------------------
# DOWNLOAD FORECAST
# --------------------------------------------------
csv_data = forecast_table.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Forecast Results (CSV)",
    data=csv_data,
    file_name="ferry_demand_forecast.csv",
    mime="text/csv"
)

st.divider()
st.caption(
    "Project: Unified Mentor | Toronto Island Ferry Ticket Demand "
    "Forecasting & Predictive Decision Support System"
)

