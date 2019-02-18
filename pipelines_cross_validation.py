# %%
import pandas as pd
from sklearn.model_selection import train_test_split

# Data pre-processing
# Read data
data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
X_train = data.drop(['Id', 'SalePrice'], axis=1)
y = data.SalePrice

low_cardinality_cols = [cname for cname in X_train.columns if 
                        X_train[cname].nunique() < 10 and
                        X_train[cname].dtype == "object"]
numeric_cols = [cname for cname in X_train.columns if 
                X_train[cname].dtype in ['int64', 'float64']]
my_cols = low_cardinality_cols + numeric_cols
X_train = X_train[my_cols]
X_test = test_data[my_cols]

# One hot encoding
X_train_ohe = pd.get_dummies(X_train)
X_test_ohe = pd.get_dummies(X_test)

# Align train and test data for the model not breaking
X_train, X_test = X_train_ohe.align(X_test_ohe, join='left', axis=1)

# %%
# Making a pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

gb_pipeline = make_pipeline(SimpleImputer(),
                            GradientBoostingRegressor(loss='ls',
                                                      n_estimators=500,
                                                      learning_rate=0.1,
                                                      max_depth=3))
lr_pipeline = make_pipeline(SimpleImputer(), LinearRegression())

# %%
# Cross-validation
from sklearn.model_selection import cross_val_score
gb_scores = cross_val_score(gb_pipeline, X_train, y, cv=4,
                            scoring='neg_mean_absolute_error')
lr_scores = cross_val_score(lr_pipeline, X_train, y, cv=4,
                            scoring='neg_mean_absolute_error')
print(gb_scores.mean(), lr_scores.mean(), sep='\n')

#%%
