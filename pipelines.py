# %%
import pandas as pd
from sklearn.model_selection import train_test_split

# Read data
data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
X = data.drop(['SalePrice'])
y = data.SalePrice

low_cardinality_cols = [cname for cname in X.columns if 
                        X[cname].nunique() < 10 and
                        X[cname].dtype == "object"]
numeric_cols = [cname for cname in X.columns if 
                                X[cname].dtype in ['int64', 'float64']]
my_cols = low_cardinality_cols + numeric_cols
X = X[my_cols]
