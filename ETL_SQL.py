import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sqlalchemy import create_engine

df=pd.read_excel("sales data.xlsx")

df.head()

df.isnull().sum()

df.duplicated().sum()

df.drop_duplicates(inplace=True)

df.info()

df.shape

df.describe()

df['order_value_EUR'].fillna(np.mean(df['order_value_EUR']),inplace=True)

df['device_type'].fillna(method='ffill',inplace=True)

df['date']=pd.to_datetime(df['date'])

df['cost'] = pd.to_numeric(df['cost'], errors='coerce')

df.columns

country_sales = df.groupby('country')['order_value_EUR'].sum().sort_values(ascending=False)
country_sales.plot(kind='bar', title='Total Order Value by Country', ylabel='EUR')
plt.xticks(rotation=45)
plt.show()

df['profit'] = df['order_value_EUR'] - df['cost']
sns.histplot(df['profit'], bins=30, kde=True)
plt.title('Profit Distribution')
plt.xlabel('Profit (EUR)')
plt.tight_layout()
plt.show()

category_sales = df.groupby('category')['order_value_EUR'].sum().sort_values(ascending=False)
category_sales.plot(kind='bar', title='Sales by Category', ylabel='EUR', color='orange')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

manager_sales = df.groupby('sales_manager')['order_value_EUR'].sum().sort_values(ascending=True)
manager_sales.plot(kind='barh', title='Sales by Sales Manager', xlabel='EUR', color='green')
plt.tight_layout()
plt.show()

device_counts = df['device_type'].value_counts()
device_counts.plot(kind='pie', autopct='%1.1f%%', title='Orders by Device Type')
plt.ylabel('')
plt.tight_layout()
plt.show()

df.groupby('customer_name')['order_value_EUR'].sum().sort_values(ascending=False).head(5).plot(kind='bar', title='Top 5 Customers by Order Value', color='green')
plt.xlabel('Customer Name')
plt.ylabel('Total Order Value (EUR)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

avg_order_value = df.groupby('category')['order_value_EUR'].mean().sort_values(ascending=False)
avg_order_value.plot(kind='bar', title='Avg Order Value by Category', ylabel='Average EUR', color='red')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

sales_rep_sales = df.groupby('sales_rep')['order_value_EUR'].sum().sort_values(ascending=False).head(10)
sales_rep_sales.plot(kind='bar', title='Top 10 Sales Reps by Sales', ylabel='EUR')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

category_profit = df.groupby('category')['profit'].sum().sort_values(ascending=False)
category_profit.plot(kind='bar', color='teal', title='Total Profit by Category', ylabel='Profit (EUR)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

country_orders = df['country'].value_counts()
sns.barplot(x=country_orders.index, y=country_orders.values, palette='viridis')
plt.title('Number of Orders by Country')
plt.ylabel('Order Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

sns.boxplot(data=df, x='category', y='order_value_EUR')
plt.title('Order Value Distribution by Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

engine_target = create_engine(
    "mysql+mysqlconnector://root:Pass%40123@localhost:3306/Salenew"
)


# -----------------------------
# Step 4: Load Data into TargetDB
# -----------------------------
# Excel → MySQL
df.to_sql(
    name="sales_excel",
    con=engine_target,
    if_exists="replace",
    index=False
)