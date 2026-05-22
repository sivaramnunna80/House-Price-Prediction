import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error,r2_score,root_mean_squared_error

df=pd.read_csv("house prediction\housing_price_dataset.csv")



df=pd.get_dummies(df,columns=['Neighborhood'],drop_first=True)


x=df.drop(columns=['Price'])
y=df['Price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

scaler=StandardScaler()


x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)


model=LinearRegression()

model.fit(x_train_scaled,y_train)
