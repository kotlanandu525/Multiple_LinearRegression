import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Multiple Linear Regression",
    layout="centered"
)

# --------------------------------------------------
# Load CSS
# --------------------------------------------------
def load_css(file_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(base_dir, file_name)

    try:
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("style.css not found")
load_css("style.css")


# --------------------------------------------------
# Title
# --------------------------------------------------
st.markdown(
    """
    <div class="card">
        <h1>Multiple Linear Regression</h1>
        <p>Train and test Multiple Linear Regression using any CSV dataset</p>
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Upload CSV
# --------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📂 Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is None:
    st.info("👆 Please upload a CSV file to continue")
    st.stop()

df = pd.read_csv(uploaded_file)

# --------------------------------------------------
# Dataset preview
# --------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🔍 Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Select numeric columns
# --------------------------------------------------
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) < 2:
    st.error("Dataset must contain at least two numeric columns")
    st.stop()

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("⚙️ Feature Selection")

X_cols = st.multiselect(
    "Select Independent Variables (X)",
    numeric_cols
)

y_col = st.selectbox(
    "Select Target Variable (y)",
    numeric_cols
)
st.markdown('</div>', unsafe_allow_html=True)

if len(X_cols) < 1:
    st.warning("Select at least one independent variable")
    st.stop()

if y_col in X_cols:
    st.error("Target variable cannot be included in features")
    st.stop()

# --------------------------------------------------
# Prepare data
# --------------------------------------------------
X = df[X_cols]
y = df[y_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# Train model
# --------------------------------------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# --------------------------------------------------
# Metrics
# --------------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 Model Performance")

c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")
c3.metric("R²", f"{r2:.2f}")

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Coefficients
# --------------------------------------------------
coef_df = pd.DataFrame({
    "Feature": X_cols,
    "Coefficient": model.coef_
})

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📈 Model Coefficients")
st.dataframe(coef_df)
st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Prediction section
# --------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🎯 Make a Prediction")

input_data = []

for col in X_cols:
    val = st.number_input(
        f"Enter value for {col}",
        float(df[col].min()),
        float(df[col].max()),
        float(df[col].mean())
    )
    input_data.append(val)

if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)[0]

    st.markdown(
        f"""
        <div class="prediction-box">
            Predicted <b>{y_col}</b>: {prediction:.2f}
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# Visualization
# --------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📉 Visualization (First Feature vs Target)")

fig, ax = plt.subplots()
ax.scatter(df[X_cols[0]], y, alpha=0.6)
ax.set_xlabel(X_cols[0])
ax.set_ylabel(y_col)

st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)
