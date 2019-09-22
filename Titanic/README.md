# Titanic
官方提供許多關於乘客的資訊做訓練，像是乘客的性別(Sex)、姓名(Name)、出發港口(Embarked)、住的艙等(Pclass)、房間號碼(Cabin)、年齡(Age)、兄弟姐妹配偶的數量(Sibsp)、父母子女數的數量(parch)、票價(Fare)、船票的號碼(Ticket)這些資訊，目標是預估乘客是否會在鐵達尼號沈船的意外中生存下來
[競賽網址](https://www.kaggle.com/c/titanic)

## Simple_Model

### 前處理方法
- 空格
	- 年齡: 空格處使用均值填空
	- 票價: 空格處使用均值填空
	- 出發港口: 空格處使用眾數填空
- 非數值
	- 性別: 男性填0、女性填1
	- 出發港口: 將值轉為one-hot

### 訓練模型
先隨便使用以下訓練模型訓練
model.add(Dense(16, input_dim=train_Features.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

### 超參數
batch_size = 5
epochs = 200

### 預測結果
loss: 0.3923
accuracy: 0.8452
val_loss: 0.3348
val_accuracy: 0.8444

Kaggle 分數: 0.76555