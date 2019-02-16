# %%
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


# Load data
iowa_data = pd.read_csv('train.csv')

iowa_target = iowa_data.SalePrice
iowa_predictors = iowa_data.drop(['SalePrice'], axis=1)

# For the sake of keeping the example simple, we'll use only numeric predictors
iowa_numeric_predictors = iowa_predictors.select_dtypes(exclude=['object'])

X_train, X_val, y_train, y_val = train_test_split(iowa_numeric_predictors, 
                                                  iowa_target,
                                                  train_size=0.7, 
                                                  test_size=0.3, 
                                                  random_state=0)

def score_dataset(X_train, X_val, y_train, y_val):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return mean_absolute_error(y_val, preds)

# %% Drop missing values
cols_with_missing = [col for col in X_train.columns 
                                 if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_val = X_val.drop(cols_with_missing, axis=1)
print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(reduced_X_train, reduced_X_val, y_train, y_val))

# %% Impute missing values
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_val = my_imputer.transform(X_val)
print("Mean Absolute Error from Imputation:")
print(score_dataset(imputed_X_train, imputed_X_val, y_train, y_val))

# %% Smart imputation
imputed_X_train_plus = X_train.copy()
imputed_X_val_plus = X_val.copy()

cols_with_missing = (col for col in X_train.columns 
                                 if X_train[col].isnull().any())
for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_val_plus[col + '_was_missing'] = imputed_X_val_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_val_plus = my_imputer.transform(imputed_X_val_plus)

print("Mean Absolute Error from Imputation while Track What Was Imputed:")
print(score_dataset(imputed_X_train_plus, imputed_X_val_plus, y_train, y_val))

#%%
