import streamlit as st
import pandas as pd
import numpy as np
import os

# ---- PAGE CONFIG ----
st.set_page_config(page_title="NIR POL Analysis Metrics", layout="wide", initial_sidebar_state="expanded")
DATA_PATH = "nir_comparative.csv"

# ---- LOAD DATA ----
@st.cache_data(ttl=3600)
def load_nir_data():
    """Load CSV file and ensure required columns exist."""
    try:
        if not os.path.exists(DATA_PATH):
            st.error(f"Data file not found at: {DATA_PATH}")
            return None
        df = pd.read_csv(DATA_PATH)
        # Ensure these columns exist
        required_cols = ['Date', 'Shift', 'Conv_POL', 'NIR_POL']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return None
        # Convert Date column to datetime - try MM/DD/YYYY format first
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
        # If that fails, try other common formats
        if df['Date'].isnull().all():
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Filter date range: 09/26/2024 to 6/22/2025
        start_date = pd.to_datetime('2024-09-26')
        end_date = pd.to_datetime('2025-06-22')
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        # Remove rows where Conv_POL = 0 or is NaN
        df = df[(df['Conv_POL'] != 0) & (df['Conv_POL'].notna())]
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# ---- MANUAL METRIC CALCULATION ----
def calculate_metrics(actual, predicted):
    """
    Manually compute MAE, RMSE, and R² between Conventional (actual) and NIR (predicted).
    """
    try:
        # Remove NaN values and rows where Conventional = 0
        mask = ~(np.isnan(actual) | np.isnan(predicted) | (actual == 0))
        actual_clean, predicted_clean = actual[mask], predicted[mask]
        n = len(actual_clean)

        if n == 0:
            return None, None, None

        # --- MAE: Mean Absolute Error ---
        mae = np.sum(np.abs(actual_clean - predicted_clean)) / n

        # --- RMSE: Root Mean Squared Error ---
        mse = np.sum((actual_clean - predicted_clean) ** 2) / n
        rmse = np.sqrt(mse)

        # --- R²: Coefficient of Determination ---
        ss_res = np.sum((actual_clean - predicted_clean) ** 2)             # residual sum of squares
        ss_tot = np.sum((actual_clean - np.mean(actual_clean)) ** 2)       # total sum of squares
        r2 = 1 - (ss_res / ss_tot if ss_tot != 0 else 0)

        return mae, rmse, r2
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        return None, None, None

# ---- FILTER DATA ----
@st.cache_data
def filter_data(df, selected_date, selected_shift):
    """Filter dataframe by selected date and shift."""
    try:
        filtered = df.copy()
        # Force selection: must pick one date and one shift
        if selected_date is None:
            st.warning("Please select a date.")
            return pd.DataFrame()
        if selected_shift is None:
            st.warning("Please select a shift.")
            return pd.DataFrame()
        filtered = filtered[filtered['Date'].dt.date == selected_date]
        filtered = filtered[filtered['Shift'] == selected_shift]
        return filtered
    except Exception as e:
        st.error(f"Error filtering data: {str(e)}")
        return df

# ---- MAIN APP ----
def main():
    st.title("NIR vs Conventional POL Metrics")
    st.markdown("**Displays MAE, RMSE, and R² for POL measurements for a selected date and shift.**")
    st.markdown("**Date Range: September 26, 2024 - June 22, 2025**")
    
    # ---- METRICS INTERPRETATION GUIDE ----
    with st.expander("📊 Metrics Interpretation Guide", expanded=False):
        st.markdown("""
        | **Metric** | **Value Range** | **Interpretation** |
        |------------|-----------------|-------------------|
        | **MAE** (Mean Absolute Error) | **0 – 0.2** | Excellent – Very low average error. |
        | | **0.2 – 0.5** | Good – Acceptable prediction accuracy. |
        | | **0.5 – 1.0** | Moderate – Predictions have noticeable errors. |
        | | **>1.0** | Poor – High average prediction error. |
        | **RMSE** (Root Mean Square Error) | **Close to 0** | Better – Lower overall prediction errors. |
        | | **0.2 – 0.5** | Good – Errors are within acceptable range. |
        | | **>0.5** | Poor – Larger deviations from actual values. |
        | **R²** (Coefficient of Determination) | **0.9 – 1.0** | Excellent – Model explains almost all variance. |
        | | **0.7 – 0.9** | Good – Model explains most variance. |
        | | **0.5 – 0.7** | Moderate – Model explains some variance. |
        | | **<0.5** | Poor – Model explains little variance. |
        """)
    
    st.markdown("---")

    # Load data
    with st.spinner("Loading NIR POL data..."):
        df = load_nir_data()
    if df is None:
        st.stop()

    # ---- SIDEBAR FILTERS ----
    st.sidebar.header("📊 Filters")
    available_dates = sorted(df['Date'].dropna().dt.date.unique())
    
    if len(available_dates) == 0:
        st.error("No data available in the specified date range (09/26/2024 - 06/22/2025)")
        st.stop()
    
    selected_date = st.sidebar.selectbox(
        "Select Date:",
        options=available_dates,
        index=0
    )
    available_shifts = sorted(df['Shift'].dropna().unique().tolist())
    selected_shift = st.sidebar.selectbox("Select Shift:", options=available_shifts, index=0)

    # Apply filters
    filtered_df = filter_data(df, selected_date, selected_shift)

    # ---- DATA SUMMARY ----
    st.sidebar.info(f"**Data Summary**\n- Total records: {len(df)}\n- Filtered records: {len(filtered_df)}")

    if len(filtered_df) == 0:
        st.warning("No POL data available for the selected date and shift.")
        st.stop()

    # ---- METRIC CALCULATIONS FOR POL ----
    st.subheader(f"POL Metrics for {selected_shift} on {selected_date}")
    
    if 'Conv_POL' in filtered_df.columns and 'NIR_POL' in filtered_df.columns:
        conventional_pol = filtered_df['Conv_POL'].values
        nir_pol = filtered_df['NIR_POL'].values
        mae, rmse, r2 = calculate_metrics(conventional_pol, nir_pol)
        n_samples = len(filtered_df)
        
        results = [{
            'Parameter': 'POL',
            'MAE': f"{mae:.4f}" if mae is not None else "N/A",
            'RMSE': f"{rmse:.4f}" if rmse is not None else "N/A",
            'R²': f"{r2:.4f}" if r2 is not None else "N/A",
            'Sample Size': n_samples
        }]
    else:
        results = [{'Parameter': 'POL', 'MAE': "N/A", 'RMSE': "N/A", 'R²': "N/A", 'Sample Size': 0}]

    # ---- DISPLAY RESULTS ----
    st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

    # ---- DISPLAY RAW DATA ----
    st.subheader("Raw POL Data Used for Calculation")
    # Include TLS_DM column if it exists in the data
    display_cols = ['Date', 'TLS_DM', 'Shift', 'Conv_POL', 'NIR_POL']
    # Only include columns that actually exist in the dataframe
    available_cols = [col for col in display_cols if col in filtered_df.columns]
    st.dataframe(filtered_df[available_cols], use_container_width=True)

    # ---- REFRESH BUTTON ----
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

if __name__ == "__main__":
    main()
