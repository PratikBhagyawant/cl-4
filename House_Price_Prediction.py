import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("USA_Housing.csv")

df.head()

df.columns

df.shape

df.isnull().sum()

df.describe()

df.info()

sns.scatterplot(x="Avg. Area Income", y="Price", data=df)
plt.title("Price vs Avg. Area Income")
plt.show()

sns.scatterplot(x="Avg. Area House Age", y="Price", data=df)
plt.title("Price vs House Age")
plt.show()

sns.scatterplot(x="Area Population", y="Price", data=df)
plt.title("Price vs Population")
plt.show()

df_num = df.drop('Address', axis=1)

plt.figure(figsize=(10, 6))
sns.heatmap(df_num.corr(), annot=True, cmap="YlGnBu")
plt.title("Correlation Heatmap")
plt.show()

sns.histplot(df['Price'], kde=True)
plt.title("Distribution of House Prices")
plt.xlabel("Price")
plt.show()

x=df.drop(['Price', 'Address'],axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Accuracy: {model.score(X_test, y_test)*100 : 0.2f}%")

print("Mean Squared Error:",mean_squared_error(y_test,y_pred))
print("R2 Score:",r2_score(y_test,y_pred))
print("Mean Absolute Error:",mean_absolute_error(y_test, y_pred))

new_data = [[50000, 20, 7, 3, 200000]] 
feature_names = x.columns  
new_data_df = pd.DataFrame(new_data, columns=feature_names) 
predicted_price = model.predict(new_data_df)  
print(f'Predicted House Price: {predicted_price[0]}')

plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()