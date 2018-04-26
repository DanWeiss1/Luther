import pprint
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm

# read in data subset variables
df = pd.read_pickle('Luther/bikedataset.pkl')
df.head()

print(df.columns)
# in x keep only numeric variables and add polynomials
df['post2'] = df['postLength'] ** 2
df['post3'] = df['postLength'] ** 3
df['pics2'] = df['num_pics'] ** 2
df['pics3'] = df['num_pics'] ** 3

X = df.drop(['attributes', 'location', 'numPics', 'price', 'text', 'title', 'url',
             'textCleaner', 'size', 'model', 'stringUpper', 'logPrice', 'manufacturer'], axis=1)
y = df['logPrice']
np.random.seed(42)
X['random'] = np.random.random_sample(X.shape[0])
X['random'].hist()
X.rename(columns={'condition_like new': 'condition_like_new'}, inplace=True)


def hist_plot_resid(resids, test):
    """ Plots a histogram of the residuals from a model
    Inputs: resids: residuals
            test: string for whether these are test or training data"""
    plt.hist(resids)
    plt.title("Histogram of Residuals\n" + test)
    plt.xlabel("Residuals")
    plt.show()


def scatter_resid(x, resids, test):
    """ Plots a scatter plot of the residuals against predicted values
    Inputs:  X  your fitted values from a model
    resids: your residuals from a model
    test: string that indicates either test or training data"""

    plt.scatter(x, resids)
    plt.xlabel("Predicted Y Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted Values\n" + test)
    plt.show()


def run_model(model_columns, x, y):
    """ Runs a linear regression model on set of training data
     and plots residuals"""
    independent_vars = x[model_columns]
    ind_vars_train, ind_vars_test, dep_train, dep_test = train_test_split(independent_vars, y, test_size=0.25,
                                                                          random_state=42)

    ols = LinearRegression()
    cv_arr = cross_val_score(ols, ind_vars_train, dep_train, cv=5)
    print(cross_val_score(ols, ind_vars_train, dep_train, cv=5))

    ols.fit(ind_vars_train, dep_train)
    hist_plot_resid(ols.predict(ind_vars_train) - dep_train, "Training Data")
    plt.show()
    hist_plot_resid(ols.predict(ind_vars_test) - dep_test, "Test Data")
    plt.show()
    scatter_resid(ols.predict(ind_vars_train), ols.predict(ind_vars_train) - dep_train, "Training Data")
    plt.show()
    scatter_resid(ols.predict(ind_vars_test), ols.predict(ind_vars_test) - dep_test, "Test Data")
    plt.show()
    d = {}
    d['cross_val_score'] = np.mean(cv_arr)
    d['coeffs'] = list(zip(model_columns, ols.coef_))
    d['intercept'] = ols.intercept_
    d['R2 train'] = ols.score(ind_vars_train, dep_train)
    d['R2 test'] = ols.score(ind_vars_test, dep_test)
    d['RMSE train'] = np.sqrt(np.mean((10**ols.predict(ind_vars_train) - 10**dep_train) ** 2))
    d['RMSE test'] = np.sqrt(np.mean((10**ols.predict(ind_vars_test) - 10**dep_test) ** 2))
    pprint.pprint(d)
    x_train = sm.add_constant(ind_vars_train)
    results = sm.OLS(dep_train, x_train).fit()
    print(results.summary())
    return d


model1_columns = ['random']
model1_results = run_model(model1_columns, X, y)

model2_columns = ['ebike', 'kids', 'pro', 'med', 'entry', 'num_pics', 'postLength']
model2_results = run_model(model2_columns, X, y)

model3_columns = ['ebike', 'kids', 'pro', 'med', 'entry', 'num_pics', 'postLength', 'area_eby', 'area_nby',
                  'area_pen', 'area_sby', 'area_scz']
model3_results = run_model(model3_columns, X, y)

model4_columns = ['ebike', 'kids', 'pro', 'med', 'entry', 'num_pics', 'postLength', 'area_eby', 'area_nby',
                  'area_pen', 'area_sby', 'area_scz', 'model_listed', 'size_listed', 'brand_listed']
model4_results = run_model(model4_columns, X, y)

model5_columns = ['ebike', 'kids', 'pro', 'med', 'entry', 'num_pics', 'postLength', 'area_eby', 'area_nby',
                  'area_pen', 'area_sby', 'area_scz', 'model_listed', 'size_listed', 'brand_listed',
                  'condition_excellent', 'condition_fair', 'condition_good', 'condition_like_new',
                  'condition_new']
model5_results = run_model(model5_columns, X, y)

model6_columns = ['ebike', 'kids', 'pro', 'med', 'entry', 'num_pics', 'postLength', 'post2', 'post3',
                  'area_eby', 'area_nby', 'area_pen', 'area_sby', 'area_scz', 'model_listed', 'size_listed',
                  'brand_listed',
                  'condition_excellent', 'condition_fair', 'condition_good', 'condition_like_new',
                  'condition_new']
model6_results = run_model(model6_columns, X, y)

model7_columns = ['ebike', 'kids', 'pro', 'med', 'entry', 'num_pics', 'postLength', 'pics2', 'pics3',
                  'area_eby', 'area_nby', 'area_pen', 'area_sby', 'area_scz', 'model_listed', 'size_listed',
                  'brand_listed',
                  'condition_excellent', 'condition_fair', 'condition_good', 'condition_like_new',
                  'condition_new']
model7_results = run_model(model7_columns, X, y)

model8_columns = ['ebike', 'kids', 'pro', 'med', 'entry', 'num_pics', 'postLength', 'pics2',
                  'area_eby', 'area_nby', 'area_pen', 'area_sby', 'area_scz', 'model_listed', 'size_listed',
                  'brand_listed',
                  'condition_excellent', 'condition_fair', 'condition_good', 'condition_like_new',
                  'condition_new']
model8_results = run_model(model8_columns, X, y)

# Model 5 returns best Cross Validated Results

pprint.pprint(model5_results)