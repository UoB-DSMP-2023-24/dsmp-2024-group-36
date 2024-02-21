import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os


def preprocess_shop_data(file_path, look_back=1):
    df = pd.read_csv(file_path)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['monopoly_money_amount'].values.reshape(-1, 1))
    generator = TimeseriesGenerator(scaled_data, scaled_data, length=look_back, batch_size=1)

    return generator, scaler


def train_lstm_model(generator, look_back=1):
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    model.fit(generator, epochs=25, verbose=1, callbacks=[early_stop])

    return model

def save_model(model, model_name, models_dir='saved_models'):
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    model_path = os.path.join(models_dir, f'{model_name}.h5')
    model.save(model_path)


def visualize_predictions(generator, model, scaler, shop_type, visualizations_dir='prediction_visualizations'):
    if not os.path.exists(visualizations_dir):
        os.makedirs(visualizations_dir)

    predictions = []
    for i in range(len(generator)):
        x, y = generator[i]
        pred = model.predict(x)
        predictions.append(pred[0, 0])

    predictions_scaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    actual_scaled = scaler.inverse_transform(generator.targets.reshape(-1, 1))

    plt.figure(figsize=(10, 6))
    plt.plot(actual_scaled, label='Actual Sales')
    plt.plot(predictions_scaled, label='Predicted Sales')
    plt.title(f'{shop_type} Sales Prediction')
    plt.legend()
    plt.savefig(os.path.join(visualizations_dir, f'{shop_type}_prediction.png'))


def main():
    shop_types_dir = 'shop_sales_time_series'
    models_dir = 'saved_models'
    visualizations_dir = 'prediction_visualizations'

    for shop_type_csv in os.listdir(shop_types_dir):
        if shop_type_csv.endswith('.csv'):
            shop_type = shop_type_csv.replace('_sales_time_series.csv', '').replace('_', ' ')
            file_path = os.path.join(shop_types_dir, shop_type_csv)

            print(f'shop type: {shop_type}')
            generator, scaler = preprocess_shop_data(file_path, look_back=1)
            model = train_lstm_model(generator, look_back=1)
            save_model(model, shop_type, models_dir)
            visualize_predictions(generator, model, scaler, shop_type, visualizations_dir)


if __name__ == '__main__':
    main()

