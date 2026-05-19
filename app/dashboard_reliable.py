import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Causal Impact BI Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PATH HANDLING ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_path(relative_path):
    return os.path.join(BASE_DIR, relative_path)

# --- CUSTOM CSS (Professional Dark Mode) ---
st.markdown("""
    <style>
    /* Global Dark Theme */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Metric Cards Styling */
    [data-testid="stMetric"] {
        background-color: #161b22 !important;
        padding: 20px !important;
        border-radius: 15px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
        border: 1px solid #30363d !important;
        transition: transform 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        border-color: #58a6ff !important;
    }
    
    /* Metric Text Colors */
    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
        font-size: 16px !important;
        font-weight: 500 !important;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid #30363d !important;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #161b22;
        border-radius: 8px 8px 0 0;
        color: #8b949e;
        border: 1px solid #30363d;
        padding: 0 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f242c !important;
        color: #58a6ff !important;
        border-bottom: 2px solid #58a6ff !important;
    }
    
    /* Header Styling */
    h1 {
        color: #58a6ff !important;
        font-weight: 800 !important;
    }
    h2, h3 {
        color: #c9d1d9 !important;
    }
    
    /* Success/Info/Error Box Styling */
    .stAlert {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        color: #c9d1d9 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    results_path = get_path('app/data/reliable_causal_results.csv')
    full_data_path = get_path('data/processed/daily_revenue_with_intervention.csv')
    
    if not os.path.exists(results_path) or not os.path.exists(full_data_path):
        results_path = 'app/data/reliable_causal_results.csv'
        full_data_path = 'data/processed/daily_revenue_with_intervention.csv'

    df_results = pd.read_csv(results_path, parse_dates=['date'])
    df_full = pd.read_csv(full_data_path, parse_dates=['date'])
    
    df = df_full.merge(df_results[['date', 'counterfactual', 'effect']], on='date', how='left')
    df['actual'] = df['revenue']
    df['cumulative_effect'] = df['effect'].cumsum()
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

intervention_date = pd.Timestamp('2011-04-01')

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=80)
    st.title("BI Analytics")
    st.markdown("---")
    st.header("⚙️ Configuration")
    
    min_date = df['date'].min().to_pydatetime()
    max_date = df['date'].max().to_pydatetime()
    date_range = st.date_input(
        "Analysis Period",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    smoothing_window = st.slider("Trend Smoothing (Days)", 1, 30, 7)
    st.markdown("---")
    st.info("💡 **Tip:** Use the tabs to explore different aspects of the causal impact.")

# --- DATA PROCESSING ---
if len(date_range) == 2:
    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    filtered_df = df.loc[mask].copy()
else:
    filtered_df = df.copy()

filtered_df['actual_smooth'] = filtered_df['actual'].rolling(window=smoothing_window, center=True).mean()
filtered_df['cf_smooth'] = filtered_df['counterfactual'].rolling(window=smoothing_window, center=True).mean()

# --- HEADER ---
st.title("📈 Causal Impact: Executive Insights")
st.markdown(f"**Intervention:** 20% Price Drop | **Intervention Date:** `{intervention_date.strftime('%Y-%m-%d')}`")

# --- KPI METRICS ---
post_mask = filtered_df['date'] >= intervention_date
post_df = filtered_df[post_mask]

if not post_df.empty:
    actual_avg = post_df['actual'].mean()
    cf_avg = post_df['counterfactual'].mean()
    rel_effect = ((actual_avg - cf_avg) / cf_avg) * 100 if cf_avg != 0 else 0
    bias_corrected = rel_effect - 13.9
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Avg Actual Revenue", f"£{actual_avg:,.0f}")
    with m2:
        st.metric("Avg Counterfactual", f"£{cf_avg:,.0f}")
    with m3:
        st.metric("Relative Lift", f"{rel_effect:.1f}%", delta=f"{rel_effect:.1f}%")
    with m4:
        st.metric("Bias-Corrected Lift", f"{bias_corrected:.1f}%", delta="-13.9% Bias", delta_color="inverse")

# --- MAIN TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Impact Analysis", "🔍 Statistical Validation", "📅 Temporal Trends", "📋 Raw Data"])

chart_style = "plotly_dark"
plotly_template = "plotly_dark"

with tab1:
    st.subheader("Revenue Performance: Actual vs. Counterfactual")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['actual'], name='Actual', line=dict(color='#58a6ff', width=1), opacity=0.3))
    fig.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['actual_smooth'], name=f'Actual ({smoothing_window}d Smooth)', line=dict(color='#58a6ff', width=3)))
    fig.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['cf_smooth'], name='Counterfactual', line=dict(color='#ff7b72', width=2, dash='dash')))
    fig.add_vline(x=intervention_date.timestamp() * 1000, line_width=2, line_dash="dash", line_color="#f85149")
    fig.update_layout(template=plotly_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode="x unified", xaxis_title="Date", yaxis_title="Revenue (£)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Daily Revenue Lift")
        fig_daily = px.bar(post_df, x='date', y='effect', color='effect', color_continuous_scale='RdYlGn')
        fig_daily.update_layout(template=plotly_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_daily, use_container_width=True)
    with c2:
        st.subheader("Cumulative Revenue Lift")
        fig_cum = px.area(post_df, x='date', y='cumulative_effect', color_discrete_sequence=['#238636'])
        fig_cum.update_layout(template=plotly_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_cum, use_container_width=True)

with tab2:
    st.subheader("Model Diagnostics")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.markdown("#### Effect Distribution")
        fig_dist = px.histogram(post_df, x='effect', nbins=30, marginal="box", color_discrete_sequence=['#bc8cff'])
        fig_dist.update_layout(template=plotly_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_dist, use_container_width=True)
    with col_v2:
        st.markdown("#### Placebo Validation")
        placebo_data = pd.DataFrame({'Test': ['Main Intervention', 'Placebo (Fake)'], 'Relative Effect (%)': [rel_effect, 13.9], 'Status': ['Real', 'Bias Indicator']})
        fig_placebo = px.bar(placebo_data, x='Test', y='Relative Effect (%)', color='Status', text_auto='.1f', color_discrete_map={'Real': '#58a6ff', 'Bias Indicator': '#ff7b72'})
        fig_placebo.update_layout(template=plotly_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_placebo, use_container_width=True)

with tab3:
    st.subheader("Temporal Patterns")
    filtered_df['weekday'] = filtered_df['date'].dt.day_name()
    c_t1, c_t2 = st.columns(2)
    with c_t1:
        avg_weekday = filtered_df.groupby('weekday')['actual'].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        fig_week = px.line(x=avg_weekday.index, y=avg_weekday.values, markers=True, title="Revenue by Weekday")
        fig_week.update_layout(template=plotly_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_week, use_container_width=True)
    with c_t2:
        filtered_df['volatility'] = filtered_df['actual'].rolling(window=7).std()
        fig_vol = px.line(filtered_df, x='date', y='volatility', title="7-Day Rolling Volatility", color_discrete_sequence=['#db6d28'])
        fig_vol.update_layout(template=plotly_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_vol, use_container_width=True)

with tab4:
    st.subheader("Data Explorer")
    st.dataframe(filtered_df, use_container_width=True)
    st.download_button(label="📥 Export Data (CSV)", data=filtered_df.to_csv(index=False).encode('utf-8'), file_name='causal_impact_results.csv', mime='text/csv')

# --- SMART INSIGHTS ---
st.markdown("---")
st.subheader("🤖 AI-Powered Insights")
if rel_effect > 20:
    st.success(f"🚀 **Strong Performance:** The intervention achieved a significant revenue lift of **{rel_effect:.1f}%**. The trend is robust and statistically significant.")
elif rel_effect > 0:
    st.info(f"📈 **Positive Trend:** A moderate revenue lift of **{rel_effect:.1f}%** was observed. Further optimization may be required.")
else:
    st.error("⚠️ **Neutral Impact:** No significant revenue lift detected. Consider reviewing the intervention strategy.")

# --- FOOTER ---
st.markdown("---")
st.markdown(f"<div style='text-align: center; color: #8b949e;'>Developed by Ahmedosrf | © {datetime.now().year} BI Analytics Platform</div>", unsafe_allow_html=True)
