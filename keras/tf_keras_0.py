# 導入函式庫
import numpy as np  

# tf: change namespace
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

from matplotlib import pyplot as plt
import os
from datetime import datetime

start=datetime.now()

# 載入 MNIST 資料庫的訓練資料，並自動分為『訓練組』及『測試組』
#(X_train, y_train), (X_test, y_test) = mnist.load_data(os.path.join(os.getcwd(), 'datasets/mnist.npz'))
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# 建立簡單的線性執行的模型
model = tf.keras.models.Sequential()
# Add Input layer, 隱藏層(hidden layer) 有 256個輸出變數
model.add(Dense(units=256, input_dim=784, kernel_initializer='normal', activation='relu')) 
# model.add(Dense(units=64, input_dim=784, kernel_initializer='normal', activation='relu')) 
# Add output layer
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))

# 編譯: 選擇損失函數、優化方法及成效衡量方式
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

# 將 training 的 label 進行 one-hot encoding，例如數字 7 經過 One-hot encoding 轉換後是 0000000100，即第8個值為 1
# tf: np_utils.to_categorical ==> tf.keras.utils.to_categorical
y_TrainOneHot = tf.keras.utils.to_categorical(y_train) 
y_TestOneHot = tf.keras.utils.to_categorical(y_test) 


# 將 training 的 input 資料轉為2維
X_train_2D = X_train.reshape(60000, 28*28).astype('float32')  
X_test_2D = X_test.reshape(10000, 28*28).astype('float32')  

x_Train_norm = X_train_2D/255
x_Test_norm = X_test_2D/255

# 進行訓練, 訓練過程會存在 train_history 變數中
# tf: 語法修改
#train_history = model.fit(x=x_Train_norm, y=y_TrainOneHot, validation_split=0.2, epochs=10, batch_size=800, verbose=2)  
# https://rylan.iteye.com/blog/2386155, 分配固定數量的顯存，否則可能會在開始訓練的時候直接報錯
# 下面兩行我的不須加
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)  
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  
train_history = model.fit(x_Train_norm, y_TrainOneHot, validation_split=0.2, epochs=10, batch_size=800, verbose=2)

print(model.summary())

# 顯示訓練成果(分數)
scores = model.evaluate(x_Test_norm, y_TestOneHot)  
print()  
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))  

# 預測(prediction)
X = x_Test_norm
predictions = model.predict(X)
# get prediction result
print('prediction:', predictions[0:20])
print('actual    :', y_test[0:20])

print(datetime.now()-start)

# # 顯示錯誤的資料圖像
# X2 = X_test[8,:,:]
# plt.imshow(X2.reshape(28,28))
# plt.show() 

    
# 模型及訓練結果存檔
# tf 要加 ./
model.save_weights("./tf_model", save_format='h5')

from sklearn.metrics import confusion_matrix
confmat = confusion_matrix(y_true=y_test, y_pred=predictions)

# draw confusion_matrix
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('Predicted label')
plt.ylabel('True label')

plt.tight_layout()
#plt.savefig('images/06_09.png', dpi=300)
plt.show()



# 取得模型組態
print("config = ", model.get_config())
# 取得模型所有權重
print("weights = ", model.get_weights())
# 取得模型彙總資訊
print("summary = ", model.summary())
# 取得網路層資訊
print("layer = ", model.get_layer(index=1).name)
# 取得參數總數
print("params = ", model.count_params())

# from keras.utils import plot_model
# plot_model(model, to_file='model.png')
