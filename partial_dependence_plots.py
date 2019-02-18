# %%
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.preprocessing import Imputer

# This is a function from a tutorial but I extracted it in plain code further
# def get_some_data(cols_to_use):
    # data = pd.read_csv('train.csv')
    # y = data.SalePrice
    # X = data[cols_to_use]
    # my_imputer = Imputer()
    # imputed_X = my_imputer.fit_transform(X)
    # return imputed_X, y
    
data = pd.read_csv('train.csv')
y = data.SalePrice
X = data.select_dtypes(include=['number'])
#cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
X = X.drop(['SalePrice', 'Id'], axis=1)
my_imputer = Imputer()
imputed_X = my_imputer.fit_transform(X)
# X, y = get_some_data(cols_to_use)
my_model = GradientBoostingRegressor()
my_model.fit(imputed_X, y)

# %%
features = [2, 5]
my_plots = plot_partial_dependence(my_model, 
                                    features=features, 
                                    X=imputed_X, 
                                    feature_names=X.columns, 
                                    grid_resolution=10)


#%%
