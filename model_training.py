from keras.models import Model
from keras.layers import Input, LSTM, Dense,Dropout
from keras.callbacks import EarlyStopping
import data_preprocessing


# 加载和预处理数据
X_train, X_test, y_amount_train, y_amount_test, y_shop_train, y_shop_test, scaler, encoder = data_preprocessing.load_and_preprocess_data('processed_shop_transfer_data.csv')

# 将DataFrame转换为numpy数组，并调整形状以适配模型
X_train = X_train.to_numpy().reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.to_numpy().reshape((X_test.shape[0], X_test.shape[1], 1))


# 构建模型
input_layer = Input(shape=(X_train.shape[1], 1))

# 第一个LSTM层
lstm_layer1 = LSTM(50, return_sequences=True)(input_layer)  # 注意：return_sequences设置为True以堆叠LSTM层
# 可以在LSTM层之间添加Dropout层以减少过拟合
dropout_layer1 = Dropout(0.2)(lstm_layer1)

# 第二个LSTM层
lstm_layer2 = LSTM(50, return_sequences=False)(dropout_layer1)  # 最后一个LSTM层的return_sequences设置为False
# 在最后一个LSTM层和输出层之间添加全连接层（Dense层）
dense_layer = Dense(50, activation='relu')(lstm_layer2)
dropout_layer2 = Dropout(0.2)(dense_layer)  # 添加Dropout层以减少过拟合

amount_output = Dense(1, name='amount_output')(dropout_layer2)
shop_output = Dense(y_shop_train.shape[1], activation='softmax', name='shop_output')(dropout_layer2)

model = Model(inputs=input_layer, outputs=[amount_output, shop_output])

# 编译模型
model.compile(optimizer='adam', loss={'amount_output': 'mean_squared_error', 'shop_output': 'categorical_crossentropy'}, metrics={'shop_output': 'accuracy'})

# 定义提前停止
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# 训练模型
model.fit(X_train, {'amount_output': y_amount_train, 'shop_output': y_shop_train},
          validation_data=(X_test, {'amount_output': y_amount_test, 'shop_output': y_shop_test}),
          epochs=1,
          batch_size=1024,
          callbacks=[early_stopping])

# 保存模型
model.save('combined_model.h5')