import functions_lib as udf 

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# # model selection & validation
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold, KFold, cross_val_predict, train_test_split
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, cross_validate

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

## Pandas need cufflinks to link with plotly and add the iplot method:
## plotly and cufflinks
import plotly 
import plotly.express as px
import cufflinks as cf #cufflink connects plotly with pandas to create graphs and charts of dataframes directly
import plotly.graph_objs as go
import plotly.offline as py
from plotly.offline import iplot
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

## regression/prediction
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor, plot_importance

import optuna

df = pd.read_csv("df_eBike_UVP.csv")

udf.first_looking(df)
udf.duplicate_values(df)

df.drop(columns=["dämpfer", "kassette"], inplace=True)

target = "uvp_€"

X = df[['gabel_federweg_mm',
        'akkukapazität_wh',
        'rahmenmaterial',
        'gänge',
        'kategorie',
        'hersteller']] # .drop(target, axis=1)


y = df[target]

X_train, X_test, y_train, y_test  = train_test_split(X, y, 
                                                     test_size=0.2, 
                                                     random_state=42, 
                                                     shuffle=True)

udf.shape_control(df, X_train, y_train, X_test, y_test)

numerics = X.select_dtypes(include="number").astype("float64")
categorics = X.select_dtypes(include=["object", "category", "bool"])

numeric_transformer = Pipeline([('Scaler', StandardScaler()), 
                                ('R_Scaler', RobustScaler()),
                                ('Pw_Scaler', PowerTransformer())])

categorical_transformer = Pipeline([('OHE', OneHotEncoder(handle_unknown="ignore"))])

transformer = ColumnTransformer([('numeric', numeric_transformer, numerics.columns),
                                 ('categoric', categorical_transformer, categorics.columns)])

pipeline_model_rf = Pipeline([('transform', transformer), ('prediction', RandomForestRegressor())])    

pipeline_model_rf.fit(X_train, y_train)

y_pred = pipeline_model_rf.predict(X_test)

def eval_metric(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"r2 score: {r2.round(2)}")
    return r2, mae, mse, rmse

r2, mae, mse, rmse = eval_metric(y_test, y_pred)

print("****************************************************************************************************")
print(f"Inputs: {X.columns.tolist()}")
print(f"Target: {target}")
print("****************************************************************************************************")
print(f"{pipeline_model_rf[-1]} >> r2: {r2}, MAE: {mae}, MSE: {mse}, RMSE: {rmse}")
print("****************************************************************************************************")

import joblib
joblib.dump(pipeline_model_rf, 'pipeline_model_rf')