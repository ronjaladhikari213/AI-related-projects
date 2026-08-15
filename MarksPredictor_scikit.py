import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

#Data
X=np.array([[1],
            [2],
            [3],
            [4],
            [5],
            [6],
            [7],
            [8]])
y=np.array([20,30,40,50,60,70,80,90])

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=420
)

#Creation of Model
model=LinearRegression()

#Training the Model
model.fit(X_train,y_train)

#Predict
y_pred=model.predict(X_test)

#Evaluate
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

print("Predictions: ",y_pred)
print("MSE: ",mse)
print("R2: ",r2)