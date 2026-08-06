import pandas as pd
from model import (
    load_and_prepare_data,
    train_linear_regression,
    train_ridge,
    train_lasso,
    evaluate_model,
)

print('Loading data...')
X_train, X_test, y_train, y_test = load_and_prepare_data('data/housing_data.csv')
print('Training Linear Regression...')
lin = train_linear_regression(X_train, y_train)
preds, mse, r2 = evaluate_model(lin, X_test, y_test)
print('Linear: MSE=', mse, 'R2=', r2)
print('Training Ridge...')
ridge = train_ridge(X_train, y_train, alpha=1.0)
preds, mse_r, r2_r = evaluate_model(ridge, X_test, y_test)
print('Ridge: MSE=', mse_r, 'R2=', r2_r)
print('Training Lasso...')
lasso = train_lasso(X_train, y_train, alpha=0.1)
preds, mse_l, r2_l = evaluate_model(lasso, X_test, y_test)
print('Lasso: MSE=', mse_l, 'R2=', r2_l)
print('Done')
