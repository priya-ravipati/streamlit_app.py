import streamlit as st
import pandas as pd
import math
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to calculate safety stock
def calculate_safety_stock(z_score, std_dev_lead_time, avg_demand, std_avg_demand, avg_lead_time):
    if pd.isna(z_score) or pd.isna(std_dev_lead_time) or pd.isna(avg_demand) or pd.isna(std_avg_demand) or pd.isna(avg_lead_time):
        return np.nan
    term1 = z_score * math.sqrt(avg_lead_time) * std_avg_demand
    term2 = z_score * std_dev_lead_time * avg_demand
    safety_stock = term1 + term2
    return round(safety_stock)

def load_files_page():
    st.title("Safety Stock and Reorder Point Prediction")
    st.sidebar.title("Upload Excel file with sheets for each month")

    if 'merged_data' not in st.session_state:
        st.session_state.merged_data = None

    uploaded_file = st.sidebar.file_uploader("Upload Excel file", type="xlsx")

    if uploaded_file:
        months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        all_data = pd.DataFrame()

        for month in months:
            sheet_data = pd.read_excel(uploaded_file, sheet_name=month)
            sheet_data['Month'] = month
            all_data = pd.concat([all_data, sheet_data])

        all_data['Demand'] = pd.to_numeric(all_data['Demand'], errors='coerce')
        all_data['Lead Times'] = pd.to_numeric(all_data['Lead Times'], errors='coerce')
        all_data['Lead Times(m)'] = all_data['Lead Times'] / 30.42
        all_data['Service Level'] = pd.to_numeric(all_data['Service Level'], errors='coerce')

        numeric_cols = ['Demand', 'Lead Times(m)']
        all_data[numeric_cols] = all_data[numeric_cols].fillna(all_data[numeric_cols].mean())

        # Group by 'Product Code' and aggregate mean and std for 'Demand' and 'Lead Times'
        grouped_data = all_data.groupby(['Product Code']).agg({
            'Demand': ['mean', 'std'],
            'Lead Times(m)': ['mean', 'std']
        }).reset_index()

        grouped_data.columns = ['Product Code', 'Demand_mean', 'Demand_std', 'Lead_Times_mean', 'Lead_Times_std']

        # Merge the grouped data with the original data to get the monthly Service Level and calculate Z-score
        merged_data = pd.merge(all_data, grouped_data, on='Product Code', how='left')
        merged_data['z_score'] = norm.ppf(merged_data['Service Level'])

        # Calculate safety stock for each product and month
        merged_data['Safety_Stock'] = merged_data.apply(
            lambda row: calculate_safety_stock(
                row['z_score'], row['Lead_Times_std'], row['Demand_mean'], row['Demand_std'], row['Lead_Times_mean']
            ), axis=1
        )

        merged_data['Reorder Point'] = (merged_data['Demand_mean'] + merged_data['Safety_Stock']).round().astype('Int64')

        average_safety_stock = merged_data.groupby('Product Code')['Safety_Stock'].mean().reset_index()
        average_safety_stock.rename(columns={'Safety_Stock': 'Average_Safety_Stock'}, inplace=True)

        merged_data = pd.merge(merged_data, average_safety_stock, on='Product Code')

        merged_data['Safety_Stock_Flag'] = merged_data.apply(
            lambda row: '1' if row['Safety_Stock'] > row['Average_Safety_Stock'] else '0', axis=1
        )

        st.write("Uploaded Data:")
        st.dataframe(merged_data)

        st.session_state.merged_data = merged_data

        # Model Training and Prediction
        X = merged_data[['Demand_mean', 'Demand_std', 'Lead_Times_mean', 'Lead_Times_std', 'Service Level']]
        y = merged_data['Safety_Stock']

        # Remove rows with NaN values in y
        non_nan_indices = ~y.isna()
        X = X[non_nan_indices]
        y = y[non_nan_indices]

        # Train a Random Forest Regressor
        rf_regressor = RandomForestRegressor(n_estimators=50, random_state=42)
        rf_regressor.fit(X, y)

        # Store the model in session state for future use
        st.session_state.rf_regressor = rf_regressor
        st.session_state.merged_data = merged_data

        # Sidebar for Prediction Parameters
        st.sidebar.header("Prediction Parameters")
        product_code = st.sidebar.selectbox("Select Product Code", merged_data['Product Code'].unique(), key='product_code')
        month = st.sidebar.selectbox("Select Month", months, key='month')

        # Function to generate predictions
        def generate_predictions(product_code, month):
            filtered_data = merged_data[(merged_data['Product Code'] == product_code) & (merged_data['Month'] == month)]
            if not filtered_data.empty:
                future_data = pd.DataFrame({
                             'Demand_mean': [filtered_data['Demand_mean'].mean()],
                             'Demand_std': [filtered_data['Demand_std'].mean()],
                             'Lead_Times_mean': [filtered_data['Lead_Times_mean'].mean()],
                             'Lead_Times_std': [filtered_data['Lead_Times_std'].mean()],
                             'Service Level': [filtered_data['Service Level'].mean()]
                                         })

                predicted_safety_stock = rf_regressor.predict(future_data)
                predicted_reorder_point = (future_data['Demand_mean'][0] + predicted_safety_stock).round().astype(int)

                return pd.DataFrame({
                'Product Code': [product_code],
                'Month': [month],
                'Demand': future_data['Demand_mean'][0],
                'Service Level': future_data['Service Level'][0],
                'Safety_Stock': filtered_data['Safety_Stock'].mean(),
                'Reorder Point': filtered_data['Reorder Point'].mean(),
                'Predicted_Safety_Stock': round(predicted_safety_stock[0]),
                'Predicted_Reorder_Point': predicted_reorder_point[0]
                                    })
            else:
                return pd.DataFrame()  # Return an empty DataFrame if no data is available for the selected filters

        # Generate and display predictions
        predictions = generate_predictions(product_code, month)
        st.write("Predictions:")
        st.dataframe(predictions)

def visualization_page():
    st.title("Visualization")
    st.sidebar.title("Visualization Parameters")

    if st.session_state.merged_data is not None:
        merged_data = st.session_state.merged_data
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        product_code = st.sidebar.selectbox("Select Product Code", ['All'] + list(merged_data['Product Code'].unique()))
        month = st.sidebar.selectbox("Select Month", ['All'] + months)

        filtered_data = merged_data if product_code == 'All' else merged_data[merged_data['Product Code'] == product_code]
        if month != 'All':
            filtered_data = filtered_data[filtered_data['Month'] == month]

        x_axis = st.sidebar.selectbox("X-Axis", ['Demand', 'Service Level', 'Lead_Times_mean', 'Safety_Stock', 'Reorder Point'])
        
        # Adding a selection for the Y-Axis when scatter plot is selected
        plot_type = st.sidebar.selectbox("Plot Type", ['Heatmap', 'Scatter Plot'])
        y_axis = None
        if plot_type == 'Scatter Plot':
            y_axis = st.sidebar.selectbox("Y-Axis", ['Safety_Stock', 'Reorder Point'])

        if not filtered_data.empty:
            plt.figure(figsize=(10, 6))

            if plot_type == 'Heatmap':
                if x_axis == 'Demand':
                    pivot_data = filtered_data.pivot_table(index='Month', columns='Product Code', values='Demand', aggfunc='mean')
                    st.write("Heatmap of Demand")
                elif x_axis == 'Service Level':
                    pivot_data = filtered_data.pivot_table(index='Month', columns='Product Code', values='Service Level', aggfunc='mean')
                    st.write("Heatmap of Service Level")
                elif x_axis == 'Lead_Times_mean':
                    pivot_data = filtered_data.pivot_table(index='Month', columns='Product Code', values='Lead_Times_mean', aggfunc='mean')
                    st.write("Heatmap of Lead Times")
                elif x_axis == 'Safety_Stock':
                    pivot_data = filtered_data.pivot_table(index='Month', columns='Product Code', values='Safety_Stock', aggfunc='mean')
                    st.write("Heatmap of Safety Stock")
                elif x_axis == 'Reorder Point':
                    pivot_data = filtered_data.pivot_table(index='Month', columns='Product Code', values='Reorder Point', aggfunc='mean')
                    st.write("Heatmap of Reorder Point")

                sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap="YlGnBu")
                plt.title(f'{x_axis} for Product Code: {product_code} ({month})')
                st.pyplot(plt.gcf())

            elif plot_type == 'Scatter Plot':
                if month == 'All':
                    if product_code == 'All':
                        plt.scatter(merged_data[x_axis], merged_data[y_axis], alpha=0.7)
                        plt.title(f'Scatter Plot of {x_axis} vs {y_axis} for All Product Codes (All Months)')
                    else:
                        plt.scatter(filtered_data[x_axis], filtered_data[y_axis], alpha=0.7)
                        plt.title(f'Scatter Plot of {x_axis} vs {y_axis} for Product Code: {product_code} (All Months)')
                else:
                    if product_code == 'All':
                        plt.scatter(merged_data[merged_data['Month'] == month][x_axis], 
                                    merged_data[merged_data['Month'] == month][y_axis], alpha=0.7)
                        plt.title(f'Scatter Plot of {x_axis} vs {y_axis} for All Product Codes ({month})')
                    else:
                        plt.scatter(filtered_data[x_axis], filtered_data[y_axis], alpha=0.7)
                        plt.title(f'Scatter Plot of {x_axis} vs {y_axis} for Product Code: {product_code} ({month})')

                plt.xlabel(x_axis)
                plt.ylabel(y_axis)
                st.pyplot(plt.gcf())
        else:
            st.write("No data available for the selected filters.")
    else:
        st.write("Please upload files on the Load Files page first.")

# Streamlit page selection, default to 'Load Files' page
page = st.sidebar.selectbox("Select Page", ["Load Files", "Visualization"])

if page == "Load Files":
    load_files_page()
else:
    visualization_page()
