import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(
    page_title="AI Sales Prediction",
    page_icon="📈",
    layout="wide"
)

df = pd.read_csv("sales_data.csv")

df["date"] = pd.to_datetime(df["date"])
df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

features = [
    "promotion",
    "holiday",
    "day",
    "month",
    "day_of_week",
    "is_weekend"
]

X = df[features]
y = df["sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

linear_rmse = mean_squared_error(y_test, linear_predictions) ** 0.5
linear_r2 = r2_score(y_test, linear_predictions)

rf_rmse = mean_squared_error(y_test, rf_predictions) ** 0.5
rf_r2 = r2_score(y_test, rf_predictions)

st.title("📈 AI Sales Prediction System")
st.write("Machine Learning based sales forecasting dashboard")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Average Sales", f"{df['sales'].mean():.2f}")

with col2:
    st.metric("Highest Sales", f"{df['sales'].max()}")

with col3:
    st.metric("Total Records", len(df))

st.divider()

st.subheader("🔮 Predict Future Sales")

col1, col2, col3 = st.columns(3)

with col1:
    date_input = st.date_input("Select Date")

with col2:
    promotion_input = st.selectbox("Promotion", ["No", "Yes"])

with col3:
    holiday_input = st.selectbox("Holiday", ["No", "Yes"])

if st.button("Predict Sales"):

    promotion = 1 if promotion_input == "Yes" else 0
    holiday = 1 if holiday_input == "Yes" else 0

    date = pd.to_datetime(date_input)

    new_data = pd.DataFrame({
        "promotion": [promotion],
        "holiday": [holiday],
        "day": [date.day],
        "month": [date.month],
        "day_of_week": [date.dayofweek],
        "is_weekend": [1 if date.dayofweek >= 5 else 0]
    })

    prediction = linear_model.predict(new_data)

    st.success(f"Predicted Sales: {prediction[0]:.2f} units")

st.divider()

st.subheader("📊 Daily Sales Trend")

fig = plt.figure(figsize=(10, 4))
plt.plot(df["date"], df["sales"], marker="o")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Daily Sales Trend")
plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

st.divider()

st.subheader("🤖 Model Performance")

results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "RMSE": [linear_rmse, rf_rmse],
    "R² Score": [linear_r2, rf_r2]
})

st.dataframe(results, use_container_width=True)

st.info("Lower RMSE and higher R² indicate better model performance.")
