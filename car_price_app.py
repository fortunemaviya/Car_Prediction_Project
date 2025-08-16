import streamlit as st
import pandas as pd
import numpy as np
import joblib
from forex_python.converter import CurrencyRates

# Load the trained model
model = joblib.load('car_price.pkl')

# Setting the application title and introduction
st.set_page_config(page_title='Car Price Predictor', layout='wide')
st.title('Car Price Prediction APP')
st.write('Fill in the car details below')

# ===== Currency Conversion Section =====
st.sidebar.header("Currency Settings")
conversion_method = st.sidebar.radio("Conversion Method", 
                                    ["Manual Rate", "Live Rate (Internet Required)"],
                                    index=0)

exchange_rate = 0.22  # Default manual rate (1 INR = 0.22 ZAR)

if conversion_method == "Manual Rate":
    exchange_rate = st.sidebar.number_input(
        "INR to ZAR Exchange Rate (1 INR = ? ZAR)",
        min_value=0.01,
        max_value=1.0,
        value=0.22,
        step=0.01
    )
else:
    try:
        c = CurrencyRates()
        exchange_rate = c.get_rate('INR', 'ZAR')
        st.sidebar.success(f"Live Rate: 1 INR = {exchange_rate:.4f} ZAR")
    except:
        st.sidebar.error("Live rate unavailable. Using default rate")
        exchange_rate = 0.22

# Create input fields for the exact features used in training
col1, col2 = st.columns(2)

with col1:
    st.subheader("Vehicle Information")
    
    vehicle_age = st.slider('Vehicle Age (years)', 
                           min_value=0, 
                           max_value=30, 
                           value=5,
                           help="Age of the vehicle in years")
    
    km_driven = st.number_input('Kilometers Driven', 
                               min_value=0, 
                               max_value=1000000, 
                               value=50000,
                               step=1000,
                               help="Total kilometers driven")

with col2:
    st.subheader("Performance Specifications")
    
    mileage = st.number_input('Mileage (kmpl)', 
                             min_value=5.0, 
                             max_value=50.0, 
                             value=15.0,
                             step=0.1,
                             help="Fuel efficiency in kilometers per liter")
    
    max_power = st.number_input('Maximum Power (bhp)', 
                               min_value=50.0, 
                               max_value=1000.0, 
                               value=100.0,
                               step=1.0,
                               help="Maximum power output in brake horsepower")

# Add some spacing
st.markdown("---")

# Prediction button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button('🚗 Predict Car Price', type='primary', use_container_width=True):
        try:
            # Create a DataFrame with the exact features used in training
            input_data = pd.DataFrame({
                'vehicle_age': [vehicle_age],
                'km_driven': [km_driven],
                'mileage': [mileage],
                'max_power': [max_power]
            })
            
            # Make prediction
            prediction = model.predict(input_data)
            
            # Display the result with formatting
            predicted_price_inr = prediction[0]
            predicted_price_zar = predicted_price_inr * exchange_rate
            
            # Create a nice display for the prediction
            st.success('🎉 Prediction Complete!')
            
            # Display the predicted price prominently
            html_content = f"""
            <div style="text-align: center; padding: 20px; background-color: #f0f8ff; border-radius: 10px; margin: 20px 0;">
                <h2 style="color: #1f4e79;">Predicted Car Price (ZAR)</h2>
                <h1 style="color: #2e8b57; font-size: 3em;">R{predicted_price_zar:,.0f}</h1>
                <p style="color: #666; font-size: 0.9em;">
                    Original INR: ₹{predicted_price_inr:,.0f} | Rate: 1 INR = {exchange_rate:.4f} ZAR
                </p>
            </div>
            """
            st.markdown(html_content, unsafe_allow_html=True)
            
            # Show input summary
            st.subheader("📋 Input Summary")
            summary_col1, summary_col2 = st.columns(2)
            
            with summary_col1:
                st.metric("Vehicle Age", f"{vehicle_age} years")
                st.metric("Kilometers Driven", f"{km_driven:,} km")
            
            with summary_col2:
                st.metric("Mileage", f"{mileage} kmpl")
                st.metric("Max Power", f"{max_power} bhp")
            
            # Additional insights
            st.info('💡 This prediction is based on vehicle age, usage, fuel efficiency, and engine power.')
            
        except Exception as e:
            st.error(f'❌ Error making prediction: {str(e)}')
            st.info('Please make sure all fields are filled correctly and the model file is available.')

# Additional information section
st.markdown("---")
st.subheader("📊 About This Prediction Model")

# Create expandable sections for more information
with st.expander("🔍 How the prediction works"):
    st.write("""
    This car price prediction model uses a **Random Forest Regressor** trained on historical car sales data. 
    The model considers four key factors:
    
    - **Vehicle Age**: Older cars typically have lower values due to depreciation
    - **Kilometers Driven**: Higher mileage usually indicates more wear and tear
    - **Mileage (Fuel Efficiency)**: Better fuel economy can increase a car's value
    - **Maximum Power**: Higher engine power often correlates with higher car prices
    """)

with st.expander("📈 Model Performance"):
    st.write("""
    The Random Forest algorithm was chosen for its ability to:
    - Handle non-linear relationships between features
    - Provide robust predictions with multiple decision trees
    - Reduce overfitting through ensemble learning
    
    The model was trained on 80% of the dataset and tested on the remaining 20%.
    """)

with st.expander("⚠️ Important Notes"):
    st.write("""
    - Predictions are estimates based on historical data and may not reflect current market conditions
    - Actual car prices can vary based on additional factors like brand, model, condition, and location
    - This tool is for informational purposes and should not be the sole basis for buying/selling decisions
    - Market fluctuations, seasonal trends, and regional differences can affect actual prices
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; padding: 15px; background-color: #f8f9fa; border-radius: 12px; margin-top: 30px;">
        <h4 style="color:#333;">🚗 Built with <span style="color:#FF4B4B;">Streamlit</span> 🚀 | 
        <span style="color:#1E90FF;">Car Price Prediction Model</span></h4>
        <p style="margin: 8px 0; font-size: 16px;">Created by <strong>Fortune Maviya</strong> ✨</p>
        
        <a href="https://github.com/fortunemaviya" target="_blank">
            <button style="background-color:#24292f; color:white; border:none; padding:10px 18px; 
            margin:5px; border-radius:8px; font-size:14px; cursor:pointer;">
            🐙 GitHub
            </button>
        </a>
        
        <a href="https://www.linkedin.com/in/fortune-maviya" target="_blank">
            <button style="background-color:#0A66C2; color:white; border:none; padding:10px 18px; 
            margin:5px; border-radius:8px; font-size:14px; cursor:pointer;">
            💼 LinkedIn
            </button>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)













