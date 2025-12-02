import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("🎬 Movie Rating Predictor")
st.markdown("### Linear Regression: Rotten Tomatoes vs. IMDb")

st.sidebar.header("Model Parameters")
test_size = st.sidebar.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Random State", value=10, step=1)

data_file = st.sidebar.file_uploader("top_100_movies_full_best_effort.csv", type=['csv'])

try:
    if data_file is not None:
        df = pd.read_csv(data_file)
    else:
        df = pd.read_csv('top_100_movies_full_best_effort.csv')
        st.info("Using local file: `top_100_movies_full_best_effort.csv`")
    if st.checkbox("Show Raw Data"):
        st.dataframe(df.head())

    df_clean = df.dropna(subset=['Rotten Tomatoes %', 'IMDb Rating'])

    x = df_clean[["Rotten Tomatoes %"]]
    y = df_clean[["IMDb Rating"]]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    lr = LinearRegression()
    lr.fit(x_train, y_train)

    m = lr.coef_[0][0]
    c = lr.intercept_[0]

    col1, col2 = st.columns(2)
    col1.metric("Coefficient (m)", f"{m:.4f}")
    col2.metric("Intercept (c)", f"{c:.4f}")

    st.latex(f"y = {m:.4f}x + {c:.4f}")

    y_pred = lr.predict(x_test)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(x_test, y_test, label="Actual Data", color='blue', alpha=0.7)

    ax.plot(x_test, y_pred, label="Best Fit Line", linewidth=2, color='red')

    ax.set_xlabel("Rotten Tomatoes %")
    ax.set_ylabel("IMDb Rating")
    ax.set_title("Rotten Tomatoes % vs IMDb Rating")
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.7)

    # Display the plot in Streamlit
    st.pyplot(fig)

except FileNotFoundError:
    st.error("File `top_100_movies_full_best_effort.csv` not found. Please upload a CSV file in the sidebar.")
except Exception as e:
    st.error(f"An error occurred: {e}")
