import numpy as np
import pandas as pd
from datetime import datetime
from keras.models import load_model
import joblib


def predict_top_consumptions(user_id, date_str, model, top_k=10):
    # 加载保存的模型和转换器
    model = load_model('combined_model.h5')
    scaler_amount = joblib.load('scaler_amount.pkl')
    encoder_shop = joblib.load('encoder_shop.pkl')
    encoder_date = joblib.load('encoder_date.pkl')

    # 解析日期并创建日期特征
    date = datetime.strptime(date_str, '%d/%m/%Y')
    day_of_week = date.weekday()
    day_of_month = date.day
    month = date.month
    date_features = pd.DataFrame({'day_of_week': [day_of_week], 'day_of_month': [day_of_month], 'month': [month]})

    # 独热编码日期特征
    date_features_one_hot = encoder_date.transform(date_features).toarray()

    # 构建输入特征数组，并添加用户ID
    user_id_scaled = scaler_amount.transform([[user_id]])  # 假设用户ID需要缩放
    input_features = np.concatenate([user_id_scaled, date_features_one_hot], axis=1)

    # 如果需要，根据模型期待的特征数量进行零填充
    num_expected_features = 68
    num_current_features = input_features.shape[1]
    if num_current_features < num_expected_features:
        padding = np.zeros((1, num_expected_features - num_current_features))
        input_features = np.concatenate([input_features, padding], axis=1)

    # 调整形状以适配模型输入
    input_features = input_features.reshape((1, input_features.shape[1], 1))

    # 使用模型进行预测
    predictions = model.predict(input_features)

    # 解析预测结果
    predicted_amount = predictions[0][0][0]
    shop_output_probs = predictions[1][0]
    top_indices = np.argsort(shop_output_probs)[-top_k:][::-1]
    top_shops = [
        encoder_shop.inverse_transform(np.eye(encoder_shop.categories_[0].shape[0])[index].reshape(1, -1))[0][0] for
        index in top_indices]
    top_probs = shop_output_probs[top_indices]

    results = [{
        'predicted_amount': predicted_amount,
        'predicted_shop': top_shops[i],
        'probability': top_probs[i]
    } for i in range(top_k)]

    return results

def main():
    # 加载模型和转换器，这部分不变
    model = load_model('combined_model.h5')
    scaler_amount = joblib.load('scaler_amount.pkl')
    encoder_shop = joblib.load('encoder_shop.pkl')
    encoder_date = joblib.load('encoder_date.pkl')

    # 获取用户输入，这部分不变
    user_id = input("Enter user ID: ")
    date_str = input("Enter date (DD/MM/YYYY): ")
    user_id = int(user_id)

    # 修改后的函数调用，去除了多余的参数
    top_consumptions = predict_top_consumptions(user_id, date_str, model, top_k=10)

    # 打印结果，这部分不变
    print("Predicted Top Consumptions:")
    for idx, result in enumerate(top_consumptions, start=1):
        print(f"Top {idx}:")
        print(f"  Predicted Shop: {result['predicted_shop']}")
        # print(f"  Predicted Amount: {result['predicted_amount']:.2f}")
        print(f"  Probability: {result['probability']:.4f}")
        print()

if __name__ == "__main__":
    main()
