# %%
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Read data from files
train_file_path = 'train.csv'
train_data = pd.read_csv(train_file_path)
test_file_path = 'test.csv'
test_data = pd.read_csv(test_file_path)

# Create train and test data sets
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath',
            'BedroomAbvGr', 'TotRmsAbvGrd']
y = train_data.SalePrice
X = train_data[features]
# train_y = train_data.SalePrice
# train_X = train_data[features]
# test_y = test_data.SalePrice
# test_X = test_data[features]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Create and fit a model
rf_model = RandomForestRegressor()
rf_model.fit(train_X, train_y)
rf_predictions = rf_model.predict(val_X)
mae = mean_absolute_error(rf_predictions, val_y)

print("MAE of random forest", mae)

# %%
# Show columns with missing values
missing_val_count_by_column = (train_data.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

# %%
# Possible solutions
# 1) Drop Columns
cols_with_missing = [col for col in train_data.columns
                     if train_data[col].isnull().any()]
reduced_train_data = train_data.drop(cols_with_missing, axis=1)
reduced_test_data = test_data.drop(cols_with_missing, axis=1)

#%%
