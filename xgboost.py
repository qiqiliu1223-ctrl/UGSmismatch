import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, KFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error


data = pd.read_csv(filename)
target_col = [col for col in data.columns if "target" in col.lower()][0]
X = data.drop(columns=[target_col])
Y = data[target_col]

seed=99
test_size = 0.2
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=test_size, random_state=seed)
num_folds = 10
kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)

model = xgb.XGBRegressor(
    random_state=seed,
    objective='reg:squarederror',
    eval_metric='rmse',
    tree_method='hist',
    booster='gbtree'
)

param_grid = {
    'n_estimators': range(800, 3501, 300),
    'learning_rate': [0.005, 0.01, 0.02, 0.05, 0.08, 0.1],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 1.0],
    'reg_lambda': [0, 0.5, 1, 2, 5, 10],
    'reg_alpha': [0, 0.001, 0.01, 0.1, 1, 5],
    'gamma': [0, 0.1, 0.2, 0.3, 0.5],
    'min_child_weight': [1, 3, 5, 7, 10],
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=100,
    scoring='neg_root_mean_squared_error',
    cv=kfold,
    verbose=1,
    n_jobs=-1,
    random_state=seed
)

random_search.fit(train_X, train_Y)

print("Best Parameter Combination：", random_search.best_params_)
print("Best RMSE：", abs(random_search.best_score_))

best_model = random_search.best_estimator_

scores_rmse = -cross_val_score(best_model, train_X, train_Y, cv=kfold, scoring='neg_root_mean_squared_error')
scores_mae = -cross_val_score(best_model, train_X, train_Y, cv=kfold, scoring='neg_mean_absolute_error')
scores_r2 = cross_val_score(best_model, train_X, train_Y, cv=kfold, scoring='r2')

for i in range(len(scores_rmse)):
    print(f'Fold {i + 1}: RMSE = {scores_rmse[i]:.4f}, MAE = {scores_mae[i]:.4f}, R2 = {scores_r2[i]:.4f}')

print("Cross-Validation Results：")
print('Mean RMSE: %f (%f)' % (np.mean(scores_rmse), np.std(scores_rmse)))
print('Mean MAE: %f (%f)' % (np.mean(scores_mae), np.std(scores_mae)))
print('Mean R2: %f (%f)' % (np.mean(scores_r2), np.std(scores_r2)))

predictions = best_model.predict(train_X)
df = pd.DataFrame({'Actual Value': train_Y, 'Predicted Value': predictions})
df.to_excel('predictions2.xlsx', index=False)

predictions = best_model.predict(test_X)
test_rmse = mean_squared_error(test_Y, predictions, squared=False)
test_mae = mean_absolute_error(test_Y, predictions)
test_r2 = r2_score(test_Y, predictions)
test_mape = mean_absolute_percentage_error(test_Y, predictions)

print("Test Set Results：")
print("RMSE:", test_rmse)
print("MAE:", test_mae)
print("R²:", test_r2)
print("MAPE:", test_mape)

df = pd.DataFrame({'Actual Value': test_Y, 'Predicted Value': predictions})
df.to_excel('predictions1.xlsx', index=False)

explainer = shap.Explainer(best_model)
shap_values = explainer.shap_values(X)


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.dpi"] = 300

shap.decision_plot(explainer.expected_value, shap_values[100:1000, :], feature_names=feature_names)

shap.summary_plot(shap_values, X, feature_names=feature_names)

shap.summary_plot(shap_values, X, plot_type="bar", feature_names=feature_names, show=False)

for f in feature_names:
    shap.dependence_plot(f, shap_values, X, interaction_index=None, feature_names=feature_names)


