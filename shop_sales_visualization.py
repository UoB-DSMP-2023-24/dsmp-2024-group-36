import pandas as pd
import matplotlib.pyplot as plt
import os

def visualize_and_save_time_series(input_dir='shop_sales_time_series', output_dir='shop_sales_visualizations'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            shop_type = filename.split('_sales_time_series')[0].replace('_', ' ')
            filepath = os.path.join(input_dir, filename)
            data = pd.read_csv(filepath, index_col='day')

            plt.figure(figsize=(12, 6))
            plt.plot(data.index, data['monopoly_money_amount'], marker='o', linestyle='-', label=shop_type)
            plt.title(f"Sales Time Series for {shop_type}")
            plt.xlabel('Day of Year')
            plt.ylabel('Sales Amount')
            plt.legend()
            plt.grid(True)
            output_filepath = os.path.join(output_dir, f"{shop_type.replace(' ', '_')}_sales_visualization.png")
            plt.savefig(output_filepath)
            plt.close()

    print("visualize completed")
visualize_and_save_time_series()
