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
# y_train = train_data.SalePrice
# X_train = train_data[features]
# y_test = test_data.SalePrice
# X_test = test_data[features]
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.7, 
                                                  test_size=0.3, random_state=0)

# Create and fit a model
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_val)
mae = mean_absolute_error(rf_predictions, y_val)

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

# %%
# 2) Imputation of mean values
from sklearn.impute import SimpleImputer
my_imputer_mean = SimpleImputer()
data_with_imputed_values = my_imputer_mean.fit_transform(X_train)

# from sklearn.impute import SimpleImputer

# my_imputer = SimpleImputer()
# imputed_X_train = my_imputer.fit_transform(X_train)
# imputed_X_val = my_imputer.transform(X_val)
# print("Mean Absolute Error from Imputation:")
# print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))

# %%
# 3) Imputation of predictions
# make copy to avoid changing original data (when Imputing)
new_data = train_data.copy()

# make new columns indicating what will be imputed
cols_with_missing = (col for col in new_data.columns 
                     if new_data[col].isnull().any())
for col in cols_with_missing:
    new_data[col + '_was_missing'] = new_data[col].isnull()

# Imputation
my_imputer_predict = SimpleImputer()
new_data = pd.DataFrame(my_imputer_predict.fit_transform(new_data))
new_data.columns = train_data.columns