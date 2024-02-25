# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

This dataset presents a captivating challenge due to the intricate relationship between the input and output columns. The complex nature of this connection suggests that there may be underlying patterns or hidden factors that are not readily apparent.

## Neural Network Model

![dl0](https://github.com/yoheshkumar/basic-nn-model/assets/119393568/78531e6c-15a4-4dc9-82bc-a497fa2c6f9e)

## DESIGN STEPS

## Step 1: Loading the Dataset
1. Load the dataset containing features and target variables into memory.
2. Check for data consistency and handle any missing values or anomalies.

## Step 2: Splitting the Dataset
1. Divide the dataset into training and testing subsets, ensuring a representative distribution of data in each subset.
2. Shuffle the data before splitting to avoid any inherent ordering bias.

## Step 3: Data Normalization
1. Normalize the features using MinMaxScaler to scale them within a predefined range, typically [0, 1].
2. Fit the scaler to the training data and transform both training and testing data accordingly.

## Step 4: Building the Neural Network Model
1. Design the architecture of the neural network model, specifying the number of layers and neurons per layer.
2. Compile the model by defining the loss function, optimizer, and any additional metrics to monitor during training.

## Step 5: Training the Model
1. Train the neural network model using the training data, specifying the number of epochs and batch size.
2. Monitor the training process for convergence and potential overfitting by observing the loss on both training and validation data.

## Step 6: Plotting Performance
1. Visualize the training process by plotting the training and validation loss over epochs.
2. Plot any additional metrics such as accuracy or precision to assess the model's performance.

## Step 7: Evaluating the Model
1. Evaluate the trained model's performance using the testing data.
2. Compute relevant metrics such as accuracy, precision, recall, and F1-score to assess the model's effectiveness.

## PROGRAM
### Name: YOHESHKUMAR R.M
### Register Number: 212222240118
#### DEPENDENCIES:
```py
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
```
#### DATA FROM SHEETS:
```py
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('exp1').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
```
#### DATA VISUALIZATION:
```py
import pandas as pd
import seaborn as sns
df['Input 1 (Number)'] = pd.to_numeric(df['Input 1 (Number)'])
sns.pairplot(df)

df['Input 1 (Number)'] = pd.to_numeric(df['Input 1 (Number)'])
df['Output'] = pd.to_numeric(df['Output'])
X = df['Input 1 (Number)']
y=df['Output']
```
#### DATA SPLIT AND PREPROCESSING:
```PY
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

x_train = x_train.values.reshape(-1, 1)
x_test = x_test.values.reshape(-1, 1)

from sklearn.preprocessing import MinMaxScaler
M = MinMaxScaler()
x_train = M.fit_transform(x_train)
```
#### REGRESSIVE MODEL:
```py
model = Sequential()
model.add(Dense(15,activation='relu',input_shape=x_train.shape))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

model.summary()

model.compile(optimizer='rmsprop',loss='mse')
model.fit(x_train,y_train,epochs=80)
model.history
```
#### LOSS CALCULATION:
```py
loss_df = pd.DataFrame(model.history.history)
loss_df.plot()
```
#### PREDICTION:
```py
y_pred=model.predict(x_test)
y_pred
```

## Dataset Information
![dl1](https://github.com/yoheshkumar/basic-nn-model/assets/119393568/967fc676-093c-46c7-9a2e-d20d34ca228a)


## OUTPUT
### Pairplot(data)
![dl2](https://github.com/yoheshkumar/basic-nn-model/assets/119393568/a970bc5e-4935-49d3-9676-2e2b0fd2bbfe)


### ARCHITECTURE AND TRAINING:
![dl3](https://github.com/yoheshkumar/basic-nn-model/assets/119393568/6025e853-c3b3-473d-aded-132fda1bc6d4)


![dl4](https://github.com/yoheshkumar/basic-nn-model/assets/119393568/52b3dc97-25c4-4375-ac58-a205f3f36ee7)

### Training Loss Vs Iteration Plot
![dl5](https://github.com/yoheshkumar/basic-nn-model/assets/119393568/57fc49d4-7636-4383-b08e-22ad9f49e730)


### Test Data Root Mean Squared Error

![dl6](https://github.com/yoheshkumar/basic-nn-model/assets/119393568/ac244e97-0a8c-4d83-ba0f-464d6da04fea)

### New Sample Data Prediction
![dl7](https://github.com/yoheshkumar/basic-nn-model/assets/119393568/96d88941-b285-4125-9a51-0ebb9b1b7230)


## RESULT

Summarize the overall performance of the model based on the evaluation metrics obtained from testing data as a regressive neural network based prediction has been obtained.
