import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Causal Impact BI Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 14px;
        color: #6c757d;
    }
    .delta-positive {
        color: #28a745;
        font-weight: bold;
    }
    .delta-negative {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    # Load results data
    df_results = pd.read_csv('data/reliable_causal_results.csv', parse_dates=['date'])
    
    # Load full historical data for context
    df_full = pd.read_csv('data/processed/daily_revenue_with_intervention.csv', parse_dates=['date'])
    
    # Merge or align data
    df = df_full.merge(df_results[['date', 'counterfactual', 'effect']], on='date', how='left')
    df['actual'] = df['revenue']
    
    # Calculate cumulative effect
    df['cumulative_effect'] = df['effect'].cumsum()
    
    return df

df = load_data()
intervention_date = pd.Timestamp('2011-04-01')

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🛠️ Dashboard Controls")

# Theme & Style
theme = st.sidebar.selectbox("Select Theme", ["Light", "Dark"])
chart_style = "plotly_white" if theme == "Light" else "plotly_dark"

# Date Filter
min_date = df['date'].min().to_pydatetime()
max_date = df['date'].max().to_pydatetime()
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Smoothing
smoothing_window = st.sidebar.slider("Smoothing Window (Days)", 1, 30, 7)

# Intervention Simulation (What-if)
st.sidebar.markdown("---")
st.sidebar.subheader("🧪 What-if Analysis")
price_drop_sim = st.sidebar.slider("Simulated Price Drop %", 0, 50, 20)
simulated_lift = 1 + (price_drop_sim / 100) * 2.5 # Simple elasticity assumption

# --- DATA PROCESSING ---
if len(date_range) == 2:
    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    filtered_df = df.loc[mask].copy()
else:
    filtered_df = df.copy()

# Apply smoothing
filtered_df['actual_smooth'] = filtered_df['actual'].rolling(window=smoothing_window, center=True).mean()
filtered_df['cf_smooth'] = filtered_df['counterfactual'].rolling(window=smoothing_window, center=True).mean()

# --- HEADER ---
st.title("📈 Causal Impact Analysis: Executive Dashboard")
st.markdown(f"**Intervention:** 20% Price Drop | **Start Date:** {intervention_date.strftime('%B %d, %Y')}")

# --- KPI METRICS ---
post_mask = filtered_df['date'] >= intervention_date
post_df = filtered_df[post_mask]

if not post_df.empty:
    actual_avg = post_df['actual'].mean()
    cf_avg = post_df['counterfactual'].mean()
    total_impact = post_df['effect'].sum()
    rel_effect = ((actual_avg - cf_avg) / cf_avg) * 100 if cf_avg != 0 else 0
    bias_corrected = rel_effect - 13.9 # Based on placebo test
    
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.metric("Avg Actual Revenue", f"£{actual_avg:,.0f}", help="Average daily revenue after intervention")
    with m2:
        st.metric("Avg Counterfactual", f"£{cf_avg:,.0f}", help="Estimated revenue if no intervention occurred")
    with m3:
        delta_color = "normal" if rel_effect > 0 else "inverse"
        st.metric("Relative Lift", f"{rel_effect:.1f}%", delta=f"{rel_effect:.1f}%", delta_color=delta_color)
    with m4:
        st.metric("Bias-Corrected Lift", f"{bias_corrected:.1f}%", delta="-13.9% Bias", delta_color="inverse")

# --- MAIN TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Impact Analysis", "🔍 Statistical Validation", "📅 Temporal Trends", "📋 Raw Data"])

with tab1:
    st.subheader("Causal Impact: Actual vs. Counterfactual")
    
    fig = go.Figure()
    
    # Actual Data
    fig.add_trace(go.Scatter(
        x=filtered_df['date'], y=filtered_df['actual'],
        name='Actual Revenue', line=dict(color='#1f77b4', width=1.5),
        opacity=0.4
    ))
    
    # Smoothed Actual
    fig.add_trace(go.Scatter(
        x=filtered_df['date'], y=filtered_df['actual_smooth'],
        name=f'Actual ({smoothing_window}d Smooth)', line=dict(color='#1f77b4', width=3)
    ))
    
    # Counterfactual
    fig.add_trace(go.Scatter(
        x=filtered_df['date'], y=filtered_df['cf_smooth'],
        name='Counterfactual (Trend)', line=dict(color='#ff7f0e', width=2, dash='dash')
    ))
    
    # Intervention Line
    fig.add_vline(x=intervention_date.timestamp() * 1000, line_width=2, line_dash="dash", line_color="red")
    fig.add_annotation(x=intervention_date, y=filtered_df['actual'].max(), text="Intervention Start", showarrow=True, arrowhead=1)
    
    fig.update_layout(
        template=chart_style,
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Revenue (£)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Daily Point Effect")
        fig_daily = px.bar(post_df, x='date', y='effect', color='effect', 
                          color_continuous_scale='RdYlGn', title="Daily Revenue Lift/Loss")
        fig_daily.update_layout(template=chart_style)
        st.plotly_chart(fig_daily, use_container_width=True)
        
    with c2:
        st.subheader("Cumulative Impact")
        fig_cum = px.area(post_df, x='date', y='cumulative_effect', 
                         title="Total Accumulated Revenue Lift", color_discrete_sequence=['#2ca02c'])
        fig_cum.update_layout(template=chart_style)
        st.plotly_chart(fig_cum, use_container_width=True)

with tab2:
    st.subheader("Model Diagnostics & Validation")
    
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        st.markdown("#### Distribution of Effects")
        fig_dist = px.histogram(post_df, x='effect', nbins=30, marginal="box", 
                               title="Revenue Effect Distribution", color_discrete_sequence=['#9467bd'])
        fig_dist.update_layout(template=chart_style)
        st.plotly_chart(fig_dist, use_container_width=True)
        
    with col_v2:
        st.markdown("#### Placebo Test Results")
        # Mocking placebo data for visualization based on the 13.9% bias mentioned
        placebo_data = pd.DataFrame({
            'Test': ['Main Intervention', 'Placebo (Fake)'],
            'Relative Effect (%)': [rel_effect, 13.9],
            'Status': ['Real', 'Bias Indicator']
        })
        fig_placebo = px.bar(placebo_data, x='Test', y='Relative Effect (%)', color='Status',
                            text_auto='.1f', title="Validation: Real vs Placebo Effect")
        fig_placebo.update_layout(template=chart_style)
        st.plotly_chart(fig_placebo, use_container_width=True)

with tab3:
    st.subheader("Temporal Aggregations")
    
    filtered_df['weekday'] = filtered_df['date'].dt.day_name()
    filtered_df['month'] = filtered_df['date'].dt.month_name()
    
    c_t1, c_t2 = st.columns(2)
    
    with c_t1:
        avg_weekday = filtered_df.groupby('weekday')['actual'].mean().reindex(
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        )
        fig_week = px.line(x=avg_weekday.index, y=avg_weekday.values, markers=True,
                          title="Average Revenue by Weekday", labels={'x': 'Day', 'y': 'Avg Revenue'})
        fig_week.update_layout(template=chart_style)
        st.plotly_chart(fig_week, use_container_width=True)
        
    with c_t2:
        # Volatility Analysis
        filtered_df['volatility'] = filtered_df['actual'].rolling(window=7).std()
        fig_vol = px.line(filtered_df, x='date', y='volatility', title="7-Day Rolling Volatility",
                         color_discrete_sequence=['#e377c2'])
        fig_vol.update_layout(template=chart_style)
        st.plotly_chart(fig_vol, use_container_width=True)

with tab4:
    st.subheader("Project Data Explorer")
    st.dataframe(filtered_df, use_container_width=True)
    
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name='causal_impact_results.csv',
        mime='text/csv',
    )

# --- AUTOMATED INSIGHTS ---
st.markdown("---")
st.subheader("🤖 Smart Insights")

if rel_effect > 20:
    st.success(f"**Strong Positive Impact:** The intervention resulted in a significant revenue lift of {rel_effect:.1f}%. Even after bias correction, the effect remains robust at ~{bias_corrected:.1f}%.")
elif rel_effect > 0:
    st.info(f"**Moderate Positive Impact:** A positive trend was detected, but it may be partially influenced by underlying bias ({rel_effect:.1f}% raw vs {bias_corrected:.1f}% corrected).")
else:
    st.error("**Negative or Neutral Impact:** The intervention did not result in the expected revenue lift during the selected period.")

st.info(f"**Volatility Note:** The average daily volatility is £{filtered_df['actual'].std():,.0f}. High variance on weekends (zero-revenue days) is a known characteristic of this dataset.")

# --- FOOTER ---
st.markdown("---")
st.markdown(f"**Author:** Ahmedosrf | **Last Updated:** {datetime.now().strftime('%Y-%m-%d')}")
