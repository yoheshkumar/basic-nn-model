# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

This dataset presents a captivating challenge due to the intricate relationship between the input and output columns. The complex nature of this connection suggests that there may be underlying patterns or hidden factors that are not readily apparent.

## Neural Network Model
![model1](https://github.com/yoheshkumar/basic-nn-model/assets/119393568/758a728d-56e9-4ff4-a62f-3c881cf1a1a0)



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
# Importing the libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Data from sheets
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('Deep-1').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
```
```py
# Data Visualisation
df=df.astype({'Input':'float'})
df=df.astype({'Output':'float'})
df.head()
x=df[['Input']].values
y=df[['Output']].values

# Spliting and Preprocessing the data
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
```
```py
# Building and compiling the model
ai_brain=Sequential([
    Dense(units=7,input_shape=[1]),
    Dense(units=5,activation='relu'),
    Dense(units=3,activation='relu'),
    Dense(units=1)
])
ai_brain.compile(optimizer = 'rmsprop', loss = 'mse')
ai_brain.fit(X_train1,y_train,epochs = 2000)
```
```py
# Loss Calculation
loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()

# Analysing the performance
X_test1 = Scaler.transform(X_test)
ai_brain.evaluate(X_test1,y_test)
X_n1 = [[57]]
X_n1_1 = Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)
```

## Dataset Information
![dataset](https://github.com/yoheshkumar/basic-nn-model/assets/119393568/a78fb694-b21e-499d-9a82-13c2cbb1e575)
## Training Loss Vs Iteration Plot
![dl1](https://github.com/yoheshkumar/basic-nn-model/assets/119393568/683ef2c2-115a-4d2d-b244-ba6e4c63c017)


## OUTPUT

### Test Data Root Mean Squared Error
![dlll](https://github.com/yoheshkumar/basic-nn-model/assets/119393568/2c6f961d-a343-458f-b313-1131684869aa)


### New Sample Data Prediction
![dl3](https://github.com/yoheshkumar/basic-nn-model/assets/119393568/1cb58388-b42a-4969-9a40-79896caea061)


## RESULT

Summarize the overall performance of the model based on the evaluation metrics obtained from testing data as a regressive neural network based prediction has been obtained.
