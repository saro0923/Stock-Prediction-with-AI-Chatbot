import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import cohere
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="Stock Price Predictions", layout="wide")
st.title('Stock Price Predictions & AI Chatbot')

COHERE_API_KEY = "jJvToSpBrN0EHfiCsa0wNg3DRTfKqcQxbFwKRza9"
scaler = StandardScaler()

# --- Sidebar ---
option = st.sidebar.text_input('Enter a Stock Symbol', value='SPY').upper()

# Get current price
def get_current_stock_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.history(period='1d')['Close'].iloc[-1]
    except Exception:
        st.sidebar.error("❌ Error fetching stock price.")
        return None

current_price = get_current_stock_price(option)
if current_price:
    st.sidebar.success(f"Current Price of {option}: ${current_price:.2f}")

today = datetime.date.today()
duration = st.sidebar.number_input('Enter the duration (days)', value=3000)
before = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('Start Date', value=before)
end_date = st.sidebar.date_input('End Date', today)

# Fetch data
if st.sidebar.button('Fetch Data'):
    if start_date < end_date:
        st.sidebar.success(f"Fetched data from {start_date} to {end_date}")
        data = yf.download(option, start=start_date, end=end_date, progress=False)
        st.session_state['stock_data'] = data
        st.session_state['data_loaded'] = True
    else:
        st.sidebar.error('End date must be after start date.')

# --- Pages ---
def dataframe():
    if 'stock_data' in st.session_state:
        st.header("📊 Recent Data")
        st.dataframe(st.session_state['stock_data'].tail(10))
    else:
        st.warning("⚠️ Please fetch stock data first.")

def predict():
    if 'stock_data' not in st.session_state:
        st.warning("⚠️ Please fetch stock data first.")
        return

    data = st.session_state['stock_data']
    model_choice = st.radio('Choose a Model', ['Linear Regression', 'Random Forest', 'Extra Trees', 'K-Neighbors', 'XGBoost'])
    num_days = int(st.number_input('How many days forecast?', value=5))

    if st.button('Predict'):
        df = data[['Close']].copy()
        df['preds'] = df['Close'].shift(-num_days)
        x = scaler.fit_transform(df.dropna().drop(['preds'], axis=1))
        y = df.dropna()['preds'].values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(),
            'Extra Trees': ExtraTreesRegressor(),
            'K-Neighbors': KNeighborsRegressor(),
            'XGBoost': XGBRegressor()
        }

        model = models[model_choice]
        model.fit(x_train, y_train)
        preds = model.predict(x_test)

        st.text(f'R2 Score: {r2_score(y_test, preds):.4f}\nMAE: {mean_absolute_error(y_test, preds):.4f}')

        forecast = model.predict(scaler.transform(df[['Close']].tail(num_days)))
        for i, price in enumerate(forecast, 1):
            st.text(f'Day {i}: ${price:.2f}')

def extract_ticker(user_input):
    words = user_input.split()
    for word in words:
        if word.isupper() and 1 <= len(word) <= 5:
            return word
    return None

def chatbot():
    st.header("💬 AI Chatbot")

    cohere_api_key = COHERE_API_KEY
    if not cohere_api_key:
        st.error("❌ Cohere API key not found.")
        return

    try:
        co = cohere.Client(cohere_api_key)
    except Exception as e:
        st.error(f"❌ Failed to initialize Cohere: {e}")
        return

    user_input = st.text_input("Ask me anything:")
    if user_input:
        if "stock" in user_input.lower() or "price" in user_input.lower():
            ticker = extract_ticker(user_input)
            if ticker:
                stock_price = get_current_stock_price(ticker)
                if stock_price:
                    st.text_area("Chatbot:", f"The current price of {ticker} is ${stock_price:.2f}", height=100)
                    return
                else:
                    st.text_area("Chatbot:", f"❌ Couldn't fetch stock price for {ticker}.", height=100)
                    return

        with st.spinner("Thinking..."):
            try:
                response = co.chat(message=user_input)
                st.text_area("Chatbot:", response.text, height=100)
            except Exception as e:
                st.error(f"❌ Error from Cohere: {e}")

def main():
    page = st.sidebar.selectbox("Choose an Option", ['Recent Data', 'Predict', 'Chatbot'])
    if page == 'Recent Data':
        dataframe()
    elif page == 'Predict':
        predict()
    elif page == 'Chatbot':
        chatbot()

if __name__ == '__main__':
    main()
