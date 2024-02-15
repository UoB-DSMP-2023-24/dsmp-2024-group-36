import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # 使用正确的日期格式转换日期
    df['not_happened_yet_date'] = pd.to_datetime(df['not_happened_yet_date'], format='%d/%m/%Y')

    # 分解日期特征
    df['day_of_week'] = df['not_happened_yet_date'].dt.dayofweek
    df['day_of_month'] = df['not_happened_yet_date'].dt.day
    df['month'] = df['not_happened_yet_date'].dt.month

    # 对金额进行归一化处理
    scaler_amount = MinMaxScaler()
    df['monopoly_money_amount'] = scaler_amount.fit_transform(df[['monopoly_money_amount']])

    # 对店铺类型进行独热编码
    encoder_shop = OneHotEncoder()
    shop_types_one_hot = encoder_shop.fit_transform(df[['Shop_type']]).toarray()
    shop_types_df = pd.DataFrame(shop_types_one_hot, columns=encoder_shop.get_feature_names_out(['Shop_type']),
                                 index=df.index)

    # 对日期特征进行独热编码
    encoder_date = OneHotEncoder()
    date_features = encoder_date.fit_transform(df[['day_of_week', 'day_of_month', 'month']]).toarray()
    date_features_df = pd.DataFrame(date_features, columns=encoder_date.get_feature_names_out(
        ['day_of_week', 'day_of_month', 'month']), index=df.index)

    # 合并处理后的特征
    X = pd.concat([df[['from_totally_fake_account']], date_features_df, shop_types_df], axis=1)
    y_amount = df['monopoly_money_amount']
    y_shop = shop_types_df

    X_train, X_test, y_amount_train, y_amount_test, y_shop_train, y_shop_test = train_test_split(X, y_amount, y_shop,
                                                                                                 test_size=0.2,
                                                                                                 random_state=42)

    # 保存MinMaxScaler实例和OneHotEncoder实例
    joblib.dump(scaler_amount, 'scaler_amount.pkl')
    joblib.dump(encoder_shop, 'encoder_shop.pkl')
    joblib.dump(encoder_date, 'encoder_date.pkl')

    return X_train, X_test, y_amount_train, y_amount_test, y_shop_train, y_shop_test, scaler_amount, encoder_shop


filepath = 'processed_shop_transfer_data.csv'