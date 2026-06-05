import streamlit as st
import pandas as pd
import plotly.express as px

# Page Configuration
st.set_page_config(page_title="Titanic AI Dashboard", layout="wide")

# Title
st.title("🚢 Titanic AI Dashboard")

# Load Dataset
df = pd.read_csv("titanic.csv")

# Dataset Overview
st.header("📌 Dataset Overview")
st.write("Dataset Shape:", df.shape)
st.dataframe(df.head())

# Missing Values Before Cleaning
st.header("🔍 Missing Values Before Cleaning")
st.write(df.isnull().sum())

# Data Cleaning
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

if "Cabin" in df.columns:
    df.drop("Cabin", axis=1, inplace=True)

# Missing Values After Cleaning
st.header("✅ Missing Values After Cleaning")
st.write(df.isnull().sum())

# Sidebar Filters
st.sidebar.header("🎛 Interactive Filters")

gender = st.sidebar.multiselect(
    "Select Gender",
    options=df["Sex"].unique(),
    default=df["Sex"].unique()
)

pclass = st.sidebar.multiselect(
    "Select Passenger Class",
    options=sorted(df["Pclass"].unique()),
    default=sorted(df["Pclass"].unique())
)

filtered_df = df[
    (df["Sex"].isin(gender)) &
    (df["Pclass"].isin(pclass))
]

# KPI Metrics
st.header("📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Passengers", len(filtered_df))
col2.metric("Survived", int(filtered_df["Survived"].sum()))
col3.metric("Average Age", round(filtered_df["Age"].mean(), 1))
col4.metric("Average Fare", round(filtered_df["Fare"].mean(), 2))

# Visualization 1
st.subheader("1️⃣ Survival Distribution")
fig1 = px.histogram(filtered_df, x="Survived", color="Survived")
st.plotly_chart(fig1, use_container_width=True)

# Visualization 2
st.subheader("2️⃣ Gender Distribution")
fig2 = px.pie(filtered_df, names="Sex")
st.plotly_chart(fig2, use_container_width=True)

# Visualization 3
st.subheader("3️⃣ Passenger Class Distribution")
fig3 = px.histogram(filtered_df, x="Pclass", color="Pclass")
st.plotly_chart(fig3, use_container_width=True)

# Visualization 4
st.subheader("4️⃣ Age Distribution")
fig4 = px.histogram(filtered_df, x="Age", nbins=30)
st.plotly_chart(fig4, use_container_width=True)

# Visualization 5
st.subheader("5️⃣ Fare vs Age")
fig5 = px.scatter(
    filtered_df,
    x="Age",
    y="Fare",
    color="Survived"
)
st.plotly_chart(fig5, use_container_width=True)

# Insights
st.header("🔍 Insights and Findings")
st.write("""
• Female passengers had a higher survival rate.

• Most passengers traveled in 3rd class.

• Passenger class influenced survival chances.

• Younger passengers had slightly better survival rates.

• Higher ticket fares were generally associated with better survival rates.
""")

# Conclusion
st.header("✅ Conclusion")
st.success("""
The Titanic AI Dashboard helps analyze passenger information,
survival patterns, age distribution, fare trends, and class-wise behavior
through interactive visualizations and filters.
""")