import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os


def load_and_predict(model_name, file_path, look_back=1, future_steps=1, models_dir='saved_models'):
    model_path = os.path.join(models_dir, f'{model_name}.h5')
    model = load_model(model_path)

    df = pd.read_csv(file_path)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['monopoly_money_amount'].values.reshape(-1, 1))

    # 使用最后look_back个数据点作为输入进行预测
    last_data_points = scaled_data[-look_back:]
    last_data_points = last_data_points.reshape((1, look_back, 1))
    predictions = []
    for _ in range(future_steps):
        pred = model.predict(last_data_points)
        # Ensure pred is correctly shaped for appending
        pred = np.squeeze(pred)  # This changes pred shape from (1, 1, 1) to (1,)
        # Now, append the prediction to the predictions list
        predictions.append(pred)  # Append the raw prediction value
        # Reshape pred for appending to last_data_points
        pred = pred.reshape(1, 1, 1)  # Reshape pred to (1, 1, 1) to match last_data_points' expected input shape
        # Append the reshaped pred to last_data_points
        last_data_points = np.append(last_data_points[:, 1:, :], pred, axis=1)

    predictions_scaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions_scaled


def visualize_future_predictions(actual_data, predictions, shop_type,
                                 visualizations_dir='future_prediction_visualizations'):
    if not os.path.exists(visualizations_dir):
        os.makedirs(visualizations_dir)

    plt.figure(figsize=(10, 6))
    plt.plot(actual_data, label='Actual Sales')
    plt.plot(np.arange(len(actual_data), len(actual_data) + len(predictions)), predictions,
             label='Future Predicted Sales', linestyle='--')
    plt.title(f'{shop_type} Future Sales Prediction')
    plt.legend()
    plt.savefig(os.path.join(visualizations_dir, f'{shop_type}_future_prediction.png'))


def main():
    models_dir = 'saved_models'
    shop_types_dir = 'shop_sales_time_series'
    visualizations_dir = 'future_prediction_visualizations'
    future_steps = 7  # 假设我们想要预测未来30个时间步长的销售

    for shop_type_csv in os.listdir(shop_types_dir):
        if shop_type_csv.endswith('.csv'):
            shop_type = shop_type_csv.replace('_sales_time_series.csv', '').replace('_', ' ')
            file_path = os.path.join(shop_types_dir, shop_type_csv)

            print(f'Predicting future sales for shop type: {shop_type}')
            predictions = load_and_predict(shop_type, file_path, look_back=1, future_steps=future_steps,
                                           models_dir=models_dir)

            df = pd.read_csv(file_path)
            actual_data = df['monopoly_money_amount'].values
            visualize_future_predictions(actual_data, predictions, shop_type, visualizations_dir)


if __name__ == '__main__':
    main()
