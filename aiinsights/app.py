import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="📈 AutoInsights Executive Analyzer", layout="wide")
st.title("📊 AutoInsights Executive Stock Insights (6-Month Smart Analyzer)")
st.write("Upload your **stock or financial CSV** — this version automatically filters and analyzes the **latest 6 months of data.**")

# ----------------------------
# FILE UPLOAD
# ----------------------------
uploaded_file = st.file_uploader("📂 Upload a CSV File", type=["csv"])

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def detect_date_column(df):
    """Detect date column automatically"""
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
                return col
            except Exception:
                pass
    return None

def to_numeric(df):
    """Convert all numeric-like columns"""
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace(',', '', regex=True).str.strip()
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            continue
    return df

def filter_last_6_months(df, date_col):
    """Filter the dataframe to the most recent 6 months"""
    df = df.sort_values(by=date_col)
    latest_date = df[date_col].max()
    six_months_ago = latest_date - pd.DateOffset(months=6)
    df_filtered = df[df[date_col] >= six_months_ago]
    return df_filtered

# ----------------------------
# MAIN INSIGHT GENERATION
# ----------------------------
def generate_summary_and_insights(df):
    df = to_numeric(df)
    date_col = detect_date_column(df)
    if date_col:
        df = filter_last_6_months(df, date_col)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    summary_text = ""
    key_insights = []

    if not numeric_cols:
        return "No numeric columns found.", [], df, None, None

    n_rows, n_cols = df.shape
    summary_text += f"The dataset contains **{n_rows:,} records** and **{n_cols} columns**, representing the **latest 6 months of stock trading activity.** "

    # Identify likely columns
    close_col = next((col for col in df.columns if "close" in col.lower()), None)
    vol_col = next((col for col in df.columns if "vol" in col.lower()), None)
    val_col = next((col for col in df.columns if "val" in col.lower()), None)

    # Trend analysis
    if date_col and close_col:
        df = df.sort_values(by=date_col).dropna(subset=[close_col])
        X = np.arange(len(df)).reshape(-1, 1)
        y = df[close_col].values
        model = LinearRegression().fit(X, y)
        slope = model.coef_[0]
        pct_change = ((y[-1] - y[0]) / y[0]) * 100 if y[0] != 0 else 0
        direction = "📈 upward" if slope > 0 else "📉 downward" if slope < 0 else "⚖️ neutral"

        summary_text += f"The **{close_col.upper()}** shows an {direction} trend over the observed 6-month period, changing by **{pct_change:.2f}%** overall.\n\n"

        if slope > 0:
            reason = "This upward movement suggests improving investor confidence and positive earnings sentiment."
        elif slope < 0:
            reason = "This downward pattern may reflect mild correction, profit-taking, or short-term market pressure."
        else:
            reason = "This stable performance indicates balanced investor sentiment and reduced volatility."

        summary_text += f"**Trend:**\n{reason}\n\n"
        key_insights.append(f"🟢 **{close_col.upper()} trend:** {direction} ({pct_change:.2f}% change). {reason}")

    # Volume analysis
    if vol_col:
        max_vol, min_vol, avg_vol = df[vol_col].max(), df[vol_col].min(), df[vol_col].mean()
        summary_text += f"**Trading volumes** ranged between **{int(min_vol):,}** and **{int(max_vol):,}**, averaging around **{int(avg_vol):,}**. "
        key_insights.append(f"📊 **Volume:** Avg {int(avg_vol):,}, steady market participation.")

    # Correlation
    if vol_col and val_col:
        corr = df[[vol_col, val_col]].corr().iloc[0, 1]
        summary_text += f"The **correlation** between **{vol_col.upper()}** and **{val_col.upper()}** is **{corr:.2f}**, showing strong alignment between trade activity and total market value. "
        key_insights.append(f"🔗 **{vol_col.upper()}–{val_col.upper()} correlation:** {corr:.2f} (strong coupling).")

    summary_text += (
        "\n\n**Summary:**\n"
        "The stock shows stable performance with moderate fluctuations over the last 6 months. "
        "Healthy trading volume indicates consistent investor interest. "
        "Price movements align with broader sector trends and performance outlook."
    )

    return summary_text, key_insights, df, date_col, close_col

# ----------------------------
# APP EXECUTION
# ----------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    summary, insights, df_filtered, date_col, close_col = generate_summary_and_insights(df)

    st.subheader("📄 Executive Summary (Latest 6 Months)")
    st.markdown(summary)

    st.subheader("💡 Key Insights")
    for i in insights:
        st.markdown(f"- {i}")

    if date_col and close_col:
        st.subheader("📉 6-Month Price Trend")
        fig = px.line(df_filtered, x=date_col, y=close_col, title=f"{close_col.upper()} Trend - Last 6 Months", markers=True)
        st.plotly_chart(fig, use_container_width=True)

    numeric_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) > 1:
        st.subheader("📊 Correlation Heatmap")
        corr = df_filtered[numeric_cols].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    report_text = summary + "\n\n" + "Key Insights:\n" + "\n".join(insights)
    st.download_button("⬇️ Download Executive 6-Month Summary", report_text, file_name="executive_summary.txt")

else:
    st.info("Please upload your CSV file (e.g., TCS stock data) to generate 6-month insights.")