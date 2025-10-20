import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(layout="wide")
st.title("Market Price Crop Prediction (India Dataset)")
st.write("Upload your crop price dataset in CSV format.")

# Upload CSV
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # ---------------------------
    # Preprocessing
    # ---------------------------
    st.subheader("Data Preprocessing")

    # Use 'Max Price' as target (instead of Modal_Price)
    target_col = 'Max Price'
    if target_col not in data.columns:
        st.error(f"Dataset must have '{target_col}' column as target.")
    else:
        # Fill missing numerical values
        num_cols = data.select_dtypes(include=np.number).columns
        for col in num_cols:
            data[col].fillna(data[col].median(), inplace=True)

        # Encode categorical columns
        cat_cols = data.select_dtypes(include='object').columns.tolist()
        le_dict = {}
        for col in cat_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            le_dict[col] = le

        st.write("Data after preprocessing")
        st.dataframe(data.head())

        # ---------------------------
        # Feature Selection
        # ---------------------------
        X = data.drop(target_col, axis=1)
        y = data[target_col]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # ---------------------------
        # Train Model
        # ---------------------------
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        st.write(f"Model RMSE: {rmse:.2f}")
        st.write(f"Model R² Score: {r2:.2f}")

        # ---------------------------
        # Prediction UI
        # ---------------------------
        st.subheader("Predict Crop Prices")
        sample_input = {}
        for col in X.columns:
            if col in cat_cols:
                # show original classes if available
                options = list(le_dict[col].classes_)
                val = st.selectbox(f"{col}", options)
                sample_input[col] = le_dict[col].transform([val])[0]
            else:
                # safe defaults
                try:
                    minv = float(data[col].min())
                    maxv = float(data[col].max())
                    medv = float(data[col].median())
                except Exception:
                    minv, maxv, medv = 0.0, 1.0, 0.0
                val = st.number_input(
                    f"{col}", min_value=minv, max_value=maxv, value=medv, format="%f"
                )
                sample_input[col] = val

        if st.button("Predict Price"):
            input_df = pd.DataFrame([sample_input])
            pred_price = model.predict(input_df)[0]
            st.success(f"Predicted Max Price: ₹{pred_price:.2f} per quintal")

        # ---------------------------
        # Add Most Profitable Crops + Graphs
        # ---------------------------
        st.subheader("Most profitable crops & visualizations")

        # Add predictions to full dataset
        data['Predicted_Price'] = model.predict(X)

        # Try to find a crop column (common names)
        possible_crop_cols = ['Crop', 'crop', 'Commodity', 'commodity', 'Crop_Name', 'Commodity_Name', 'Variety']
        crop_col = None
        for c in possible_crop_cols:
            if c in uploaded_file.name:
                # unlikely to be useful but keep
                pass
        for c in possible_crop_cols:
            if c in data.columns:
                crop_col = c
                break
        if not crop_col:
            # fallback to first original categorical column (before encoding) if available
            # Note: after label encoding the original string classes are lost in the dataframe; but we kept label encoders
            if len(cat_cols) > 0:
                crop_col = cat_cols[0]
            else:
                crop_col = None

        # Try to find a market column
        possible_market_cols = ['Market', 'market', 'City', 'city', 'Town', 'town']
        market_col = None
        for c in possible_market_cols:
            if c in data.columns:
                market_col = c
                break

        # Try to find a date column
        date_col = None
        for c in ['Date', 'date', 'TRADE_DATE', 'trade_date']:
            if c in data.columns:
                date_col = c
                break

        # If crop_col is a label-encoded column, we can retrieve original class names from the encoder
        crop_classes = None
        if crop_col and crop_col in le_dict:
            crop_classes = list(le_dict[crop_col].classes_)

        # Compute average predicted price per crop (if crop column exists)
        if crop_col:
            try:
                # If crop column was encoded, map back to names for presentation
                if crop_col in le_dict:
                    data['_crop_name'] = le_dict[crop_col].inverse_transform(data[crop_col].astype(int))
                else:
                    data['_crop_name'] = data[crop_col].astype(str)

                grouped = data.groupby('_crop_name')['Predicted_Price'].mean().sort_values(ascending=False)
                st.write("Top 5 crops by average predicted price:")
                st.table(grouped.head(5).reset_index().rename(columns={'_crop_name':'Crop','Predicted_Price':'Avg_Predicted_Price'}))

                # Recommend the single best crop
                best_crop = grouped.index[0]
                best_price = grouped.iloc[0]
                st.success(f"Recommended crop to grow for highest price: {best_crop} (avg predicted price ₹{best_price:.2f})")

                # Bar chart of top 10 crops
                st.write("Average predicted price — top 10 crops")
                fig, ax = plt.subplots()
                grouped.head(10).plot(kind='bar', ax=ax)
                ax.set_ylabel('Avg Predicted Price (₹)')
                ax.set_xlabel('Crop')
                ax.set_title('Top 10 Crops by Avg Predicted Price')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)

                # If market column exists: show top crop per market
                if market_col:
                    st.write(f"Top crop per {market_col}")
                    # prepare readable market and crop names
                    if market_col in le_dict:
                        data['_market_name'] = le_dict[market_col].inverse_transform(data[market_col].astype(int))
                    else:
                        data['_market_name'] = data[market_col].astype(str)

                    top_per_market = data.groupby(['_market_name','_crop_name'])['Predicted_Price'].mean().reset_index()
                    top_per_market = top_per_market.sort_values(['_market_name','Predicted_Price'], ascending=[True,False])
                    # show first crop for each market
                    top_crops = top_per_market.groupby('_market_name').first().reset_index()
                    st.dataframe(top_crops.rename(columns={'_market_name':market_col,'_crop_name':'Top_Crop','Predicted_Price':'Avg_Predicted_Price'}))

            except Exception as e:
                st.error(f"Couldn't compute crop recommendations: {e}")
        else:
            st.info("No crop-like column found to compute 'best crop'. If your dataset has a crop/commodity column, name it 'Crop' or 'Commodity' or similar.")

        # Visualization for a selected crop: price over time (if date exists) or avg price by market
        st.subheader("Crop price visualization")
        if crop_col:
            # let user pick a crop to visualize
            crop_options = data['_crop_name'].unique().tolist() if '_crop_name' in data.columns else list(data[crop_col].unique())
            sel_crop = st.selectbox('Select crop to visualize', crop_options)

            # filter
            df_crop = data[data.get('_crop_name', data.columns[0]) == sel_crop] if '_crop_name' in data.columns else data[data[crop_col] == sel_crop]

            if date_col and date_col in data.columns:
                try:
                    df_crop = df_crop.copy()
                    df_crop[date_col] = pd.to_datetime(df_crop[date_col], errors='coerce')
                    df_time = df_crop.sort_values(date_col).groupby(date_col)['Max Price'].mean().reset_index()
                    fig, ax = plt.subplots()
                    ax.plot(df_time[date_col], df_time['Max Price'])
                    ax.set_title(f'Price trend for {sel_crop}')
                    ax.set_ylabel('Max Price (₹)')
                    ax.set_xlabel('Date')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Could not plot time series: {e}")
            elif market_col:
                try:
                    # average max price by market for this crop
                    if market_col in le_dict:
                        df_crop['_market_name'] = le_dict[market_col].inverse_transform(df_crop[market_col].astype(int))
                    else:
                        df_crop['_market_name'] = df_crop[market_col].astype(str)

                    by_market = df_crop.groupby('_market_name')['Max Price'].mean().sort_values(ascending=False)
                    fig, ax = plt.subplots()
                    by_market.plot(kind='bar', ax=ax)
                    ax.set_title(f'Average Max Price by {market_col} for {sel_crop}')
                    ax.set_ylabel('Avg Max Price (₹)')
                    ax.set_xlabel(market_col)
                    plt.xticks(rotation=45, ha='right')
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Could not plot market-wise prices: {e}")
            else:
                # fallback: show price distribution for this crop
                try:
                    fig, ax = plt.subplots()
                    ax.boxplot(df_crop['Max Price'].dropna())
                    ax.set_title(f'Price distribution for {sel_crop}')
                    ax.set_ylabel('Max Price (₹)')
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Could not plot distribution: {e}")
        else:
            st.info("No crop column to visualize. Add a crop/commodity column named 'Crop' or similar.")

        # ---------------------------
        # Download Predictions
        # ---------------------------
        csv = data.to_csv(index=False)
        st.download_button(
            label="Download CSV with Predictions",
            data=csv,
            file_name='predicted_prices.csv',
            mime='text/csv'
        )
