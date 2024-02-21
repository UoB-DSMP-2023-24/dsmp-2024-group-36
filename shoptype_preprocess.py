import pandas as pd
import os

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    df['not_happened_yet_date'] = pd.to_datetime(df['not_happened_yet_date'], format='%d/%m/%Y')
    df['day'] = df['not_happened_yet_date'].dt.dayofyear
    grouped = df.groupby(['Shop_type', 'day'])['monopoly_money_amount'].sum().reset_index()

    time_series_data = {}
    shop_types = grouped['Shop_type'].unique()
    for shop_type in shop_types:
        shop_data = grouped[grouped['Shop_type'] == shop_type]
        time_series_data[shop_type] = shop_data[['day', 'monopoly_money_amount']].set_index('day')

    return time_series_data

def save_to_csv(time_series_data, output_dir='shop_sales_time_series'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for shop_type, data in time_series_data.items():
        filename_safe_shop_type = shop_type.replace('/', '_').replace(' ', '_').replace('&', 'and')
        output_path = os.path.join(output_dir, f"{filename_safe_shop_type}_sales_time_series.csv")
        data.to_csv(output_path)

    print("hello world")

filepath = 'updated_shop_transfer_data.csv'

time_series_data = load_and_preprocess_data(filepath)

save_to_csv(time_series_data)
