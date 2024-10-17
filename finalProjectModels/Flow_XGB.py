import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import os
import traceback
import xgboost as xgb
os.chdir('D:/KTH/AI/Project/Data')

train_dir = 'portalSplit/train/grouped/'
evalf_dir = 'portalSplit/eval_final/grouped/'
out_dir = 'result/XGB'
result_df = pd.DataFrame(columns=['portal','mae','mse','r2'])

for file_name in os.listdir(train_dir):

    if file_name.endswith('.csv'):

        print(file_name)

        try:
            # ----------------- TRAINING -------------------#

            train_df = pd.read_csv(os.path.join(train_dir, file_name))
            #train_df['Date'] = train_df['Date'].astype(int).astype(str)
            train_df['Date'] =pd.to_datetime(train_df['Date'] ,format='%Y-%m-%d')
            # peak period 1 min interval
            start_time = 420  # 7:00
            end_time = 525  # 8:45

            # Filter data in peak period
            train_df = train_df[(train_df['Interval_1'] >= start_time) & (train_df['Interval_1'] <= end_time)]

            #sns.histplot(x=train_df['speed_avg'])

            target = 'FLOW_sum'

            X_li = []  # inputs
            y_li = []  # outputs

            window = 15

            # for current time
            features = ['FLOW_sum',
                        'is_weekend',
                        'FLOW_sum_rolling_15',
                        'visibility', 'temp',
                        'precip',
                        'speed_avg'
                        ]

            for day in train_df['Date'].unique():
                day_data = train_df[train_df['Date'] == day].copy()
                day_data.sort_values(by='Time')

                # create lag features
                day_data.loc[:, 'FLOW_sum_lag1'] = day_data['FLOW_sum'].shift(1)
                day_data['FLOW_sum_rolling_15'] = day_data['FLOW_sum'].rolling(window=15, min_periods=1).mean()

                if len(day_data) == end_time - start_time + 1:  # skip days with missing minutes

                    current_time = start_time + 16  # start from day_data minute 16

                    for current_time in range(current_time, end_time - 16 + 1):  # end at 8:29, predict 8:30-8:45

                        p_start = current_time + 1
                        p_end = p_start + window

                        X_li.append(day_data[day_data['Interval_1'] == current_time][features])
                        y_li.append(np.mean(day_data[(day_data['Interval_1'] >= p_start) & (day_data['Interval_1'] <= p_end)][target]))

                else:
                    print("Skipped day: ", day)

            X = np.array(X_li)
            y = np.array(y_li)


            X = X.reshape(-1, X.shape[2])


            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            params = {
                'n_estimators': [150, 200, 180],
                'max_depth': [4, 5, 6],
                'learning_rate': [0.2, 0.15, 0.1],
                'subsample': [0.9, 1.1, 1.0]
            }


            xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', random_state=0)


            grid_search = GridSearchCV(estimator=xgb_regressor, param_grid=params, cv=5,
                                       scoring='neg_mean_squared_error', verbose=1)
            grid_search.fit(X_train, y_train)


            best_params = grid_search.best_params_
            print("Best Hyperparameters:", best_params)


            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)


            mae_general = mean_absolute_error(y_test, y_pred)
            mse_general = mean_squared_error(y_test, y_pred)
            r2_general = r2_score(y_test, y_pred)
            '''''
            #print(f"Cross-validation MAE: {mean_mae:.4f} ± {std_mae:.4f}")
            #print(f"Cross-validation R2: {mean_r2:.4f} ± {std_r2:.4f}")
            print('Train-test evaluaion: ')
            print(f"Mean Absolute Error general: {mae_general}")
            print(f"Mean Squared Error general: {mse_general}")
            print(f"R-squared general: {r2_general}")

            # Create a scatter plot to visualize the relationship
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title(file_name[:-4])
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='red', lw=2)
            plt.show()
            '''''
            # Get feature importance scores
            importances = best_model.feature_importances_

            # Create a DataFrame for better visualization
            feature_importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)

            # Print the feature importance scores
            print(feature_importance_df)


            # -------------------------------------------------------------------FINAL EVALUATION -------------------------------------#

            eval_df = pd.read_csv(os.path.join(evalf_dir, file_name))
            eval_df['Date'] = pd.to_datetime(eval_df['Date'], format='%Y-%m-%d')
           # eval_df.sort_values(by=['Date', 'Time'], inplace=True)

            # Filter data in peak period
            eval_df = eval_df[(eval_df['Interval_1'] >= start_time) & (eval_df['Interval_1'] <= end_time)]

            #sns.histplot(x=eval_df['speed_avg'])

            #target = 'FLOW_sum'

            X_li = []  # inputs
            y_li = []  # outputs

            window = 15

            for day in eval_df['Date'].unique():
                day_data = eval_df[eval_df['Date'] == day].copy()
                day_data.sort_values(by='Time')
                day_data.loc[:, 'FLOW_sum_lag1'] = day_data['FLOW_sum'].shift(1)
                day_data['FLOW_sum_rolling_15'] = day_data['FLOW_sum'].rolling(window=15, min_periods=1).mean()

                if len(day_data) == end_time - start_time + 1:  # skip days with missing minutes

                    current_time = start_time + 16  # start from day_data minute 16

                    for current_time in range(current_time, end_time - 16 + 1):  # end at 8:29, predict 8:30-8:45

                        p_start = current_time + 1
                        p_end = p_start + window

                        X_li.append(day_data[day_data['Interval_1'] == current_time][features])
                        y_li.append(np.mean(
                            day_data[(day_data['Interval_1'] >= p_start) & (day_data['Interval_1'] <= p_end)][target]))

                else:
                    print("Skipped day: ", day)

            X_eval = np.array(X_li)
            y_eval = np.array(y_li)
            X_eval = X_eval.reshape(-1, X_eval.shape[2])

            # Predict whole dataset
            y_pred = best_model.predict(X_eval)
            mae = mean_absolute_error(y_eval, y_pred)
            mse = mean_squared_error(y_eval, y_pred)
            r2 = r2_score(y_eval, y_pred)
            output_filename = file_name[5:-8]
            result_df.loc[len(result_df)] = [output_filename, mae, mse, r2]
            result_df.to_csv(os.path.join(out_dir, 'flow_RF.csv'))
            print('Final evaluation dataset: ')
            print(f"Mean Absolute Error: {mae}")
            print(f"Mean Squared Error: {mse}")
            print(f"R-squared: {r2}")
            print('')

            # Create a scatter plot to visualize the relationship
            plt.figure(figsize=(8, 6))
            plt.scatter(y_eval, y_pred, alpha=0.5)  # Plot actual vs. predicted values
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title(output_filename +"- Actual vs. Predicted Values- XGBoost")
            plt.plot([y_eval.min(), y_eval.max()], [y_eval.min(), y_eval.max()], linestyle='--', color='red', lw=2)
            #plt.show()

            plt.savefig(os.path.join(out_dir, output_filename + '.jpg'), format='jpg')

        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            traceback.print_exc()
            break
