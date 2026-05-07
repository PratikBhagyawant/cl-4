import pandas as pd
from sqlalchemy import create_engine

# -----------------------------
# Step 1: Read Data from Excel
# -----------------------------
excel_file = "sales data.xlsx"

data_excel = pd.read_excel(excel_file)

print("Excel Data:")
print(data_excel.head())


# -----------------------------
# Step 2: Read Data from MySQL (salesdb)
# -----------------------------
engine_source = create_engine(
    "mysql+mysqlconnector://root:Pass%40123@localhost:3306/salesdb"
)

query = "SELECT * FROM Orders"

data_sql = pd.read_sql(query, engine_source)

print("\nMySQL Data:")
print(data_sql.head())


# -----------------------------
# Step 3: Connect Target Database
# -----------------------------
engine_target = create_engine(
    "mysql+mysqlconnector://root:Pass%40123@localhost:3306/TargetDB"
)


# -----------------------------
# Step 4: Load Data into TargetDB
# -----------------------------
# Excel → MySQL
data_excel.to_sql(
    name="sales_excel",
    con=engine_target,
    if_exists="replace",
    index=False
)

# MySQL → MySQL
data_sql.to_sql(
    name="sales_sql",
    con=engine_target,
    if_exists="replace",
    index=False
)

print("\nData Loaded Successfully into TargetDB!")