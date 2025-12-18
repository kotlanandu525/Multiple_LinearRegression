import numpy as np
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import *
import matplotlib.pyplot  as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

st.markdown("""
<style>
.card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}

.prediction-box {
    background-color: #e6f4ff;
    padding: 15px;
    border-radius: 8px;
    font-size: 20px;
    font-weight: bold;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


st.set_page_config("Multiple_Linear_Regression",layout="centered")
def load_css(file):
    try:
        with open(file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("style.css not found. App running without custom styles.")


#title of the page
st.markdown("""
    <div class="card">
            <h1>Multiple_Linear_Regression</h1>
            <p>Predict <b>Tip Amount</b> from <b>total bill</b> using LinearRegression </p>

    </div>
    """,unsafe_allow_html=True)

#Load_data
@st.cache_data

def load_data():
    return sns.load_dataset("tips")
df=load_data()

#Dtaset_preview
st.markdown('<div class ="card">',unsafe_allow_html=True)
st.subheader("DataSet_Preview")
st.dataframe(df.head())
st.markdown('<div>',unsafe_allow_html=True)
#prepare datta

#prepare datta
x,y=df[['total_bill','size']],df['tip']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#Train Model
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
#Metrics
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
r2=r2_score(y_test,y_pred)
adj_r2=1-(1-r2)*(len(y_test)-1)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Total Bill vs Tip (Multiple Linear Regression)")

fig, ax = plt.subplots()
ax.scatter(df["total_bill"], df["tip"], alpha=0.6)

# Fix size at mean
size_mean = df["size"].mean()

x_line = np.linspace(
    df["total_bill"].min(),
    df["total_bill"].max(),
    100
)

X_line = np.column_stack((x_line, np.full_like(x_line, size_mean)))
X_line_scaled = scaler.transform(X_line)

y_line = model.predict(X_line_scaled)

ax.plot(x_line, y_line, color="red", label="Regression line (size = mean)")
ax.set_xlabel("Total Bill ($)")
ax.set_ylabel("Tip ($)")
ax.legend()

st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div>',unsafe_allow_html=True)

#Performance
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader('Model_performance')
c1,c2=st.columns(2)
c1.metric("MAE",f"{mae:.2f}")
c2.metric("RMSE",f"{rmse:.2f}")
c3,c4=st.columns(2)
c1.metric("R2",f"{r2:.2f}")
c2.metric("adj r2",f"{adj_r2:.2f}")
st.markdown('<div>',unsafe_allow_html=True)

#m & c#

st.markdown(
    f"""
    <div class="card">
        <h3>Model Interpretation</h3>
        <p>
            <b>Coefficient:</b> {model.coef_[0]:.3f}<br>
            <b>Intercept:</b> {model.intercept_:.3f}
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
# Prediction
min_bill = float(df["total_bill"].min())
max_bill = float(df["total_bill"].max())

st.markdown('<div class="card">', unsafe_allow_html=True)

bill = st.slider(
    "Enter Total Bill ($)",
    min_value=min_bill,
    max_value=max_bill,
    value=30.0
)

size = st.slider(
    "Group Size",
    min_value=int(df["size"].min()),
    max_value=int(df["size"].max()),
    value=int(df["size"].mean()),
    step=1
)

tip = model.predict(scaler.transform([[bill, size]]))[0]

st.markdown(
    f'<div class="prediction-box">Predicted Tip: ${tip:.2f}</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)



