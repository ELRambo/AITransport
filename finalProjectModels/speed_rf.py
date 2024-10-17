# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 14:50:41 2024

@author: 10449
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:14:04 2024

@author: 10449
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import os
import traceback
os.chdir('D:\Msc\AITransport\Project\Data')

# In[]

train_dir = 'portalSplit/trainFillNan/'
evalf_dir = 'portalSplit/evalf/'
out_dir = 'result/'

weather = pd.read_csv('portalSplit/weather.csv')
weather['datetime'] = pd.to_datetime(weather['datetime'], format='%Y-%m-%d').dt.date

result_df = pd.DataFrame(columns=['portal','mae','mse','r2'])

def convert_to_minutes(timestamp):
    dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S")
    return dt.hour * 60 + dt.minute

for file_name in os.listdir(train_dir):
    
    if file_name.endswith('.csv'):
        
        print(file_name)
        
        try: 
            #----------------- TRAINING -------------------#
            
            train_df = pd.read_csv(os.path.join(train_dir, file_name))
            
            train_df['Date'] = train_df['Date'].astype(int).astype(str)
            train_df['Date'] = pd.to_datetime(train_df['Date'], format='%Y%m%d').dt.date
            
            train_df = pd.merge(train_df, weather, left_on='Date', right_on='datetime')
            
            label_encoder = LabelEncoder()
            train_df['conditions_encoded'] = label_encoder.fit_transform(train_df['conditions'])
            train_df['sunrise'] = train_df['sunrise'].apply(convert_to_minutes)
            
            # peak period 1 min interval
            start_time = 420  # 7:00
            end_time = 525  # 8:45
            
            # Filter data in peak period
            train_df = train_df[(train_df['Interval_1'] >= start_time) & (train_df['Interval_1'] <= end_time)]
            
            # sns.histplot(x=train_df['SPEED_MS_AVG'])
            
            target = 'SPEED_MS_AVG'
            
            X_li = [] # inputs
            y_li = []  # outputs
                        
            window = 15
            
            # for current time
            features = ['SPEED_MS_AVG', 'speed_avg_rolling_15', 
                        'is_weekend', 'sunrise',
                        'visibility', 'temp', 'precip'
                        ]
            
            for day in train_df['Date'].unique(): 
                
                for sensor in train_df['DP_ID'].unique():
                    
                    day_data = train_df[(train_df['Date'] == day) & (train_df['DP_ID'] == sensor)].copy()
                    day_data.sort_values(by='Time')
                    
                    day_data['speed_avg_rolling_15'] = day_data['SPEED_MS_AVG'].rolling(window=15, min_periods=1).mean()
                                        
                    if len(day_data) == end_time-start_time+1:  # skip days with missing minutes
                                                                            
                        current_time = start_time + 16  # start from day_data minute 16
                        
                        for current_time in range(current_time, end_time-16+1):  # end at 8:29, predict 8:30-8:45
                        
                            p_start = current_time + 1
                            p_end = p_start + window
                            
                            X_li.append(day_data[day_data['Interval_1'] == current_time][features])
                            y_li.append(np.mean(day_data[(day_data['Interval_1'] >= p_start) & (day_data['Interval_1'] < p_end)][target]))
                                           
                    # else:
                    #    print("Skipped day and sensor: ", day, sensor) 
            X_li_df = pd.concat(X_li, ignore_index=True)
            y_li_df = pd.DataFrame(y_li, columns=['Speed_next15'])
            X_li_df['Speed_next15'] = y_li_df
            sns.pairplot(data=X_li_df, hue='Speed_next15').savefig('result/figs/pair_' + file_name[:-4]+'.png')
            break
            
            X = np.array(X_li)
            y = np.array(y_li)
            
            X = X.reshape(-1, X.shape[2])  # reshape for random forest
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # X_train, y_train = resample_data(X_train, y_train)
            
            model = RandomForestRegressor(n_estimators=100, 
                                          max_depth=20,
                                          max_features='sqrt',
                                          min_samples_split=2,
                                          random_state=42)
            
            
            # # LSTM
            # model = Sequential()
            # model.add(LSTM(16, input_shape=(X_train.shape[1], X_train.shape[2])))
            # model.add(Dense(1))
            # model.compile(optimizer='adam', loss='mse')
            # early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            # model.fit(X_train, y_train, epochs=50, batch_size=64, callbacks=[early_stopping])

            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            cv_scores_mae = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
            cv_scores_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            mean_mae = -np.mean(cv_scores_mae)
            std_mae = np.std(cv_scores_mae)
            mean_r2 = np.mean(cv_scores_r2)
            std_r2 = np.std(cv_scores_r2)

            mae_general = mean_absolute_error(y_test, y_pred)
            mse_general = mean_squared_error(y_test, y_pred)
            r2_general = r2_score(y_test, y_pred)
            
            print(f"Cross-validation MAE: {mean_mae:.4f} ± {std_mae:.4f}")
            print(f"Cross-validation R2: {mean_r2:.4f} ± {std_r2:.4f}")
            print('Train-test evaluaion: ')
            print(f"Mean Absolute Error general: {mae_general}")
            print(f"Mean Squared Error general: {mse_general}")
            print(f"R-squared general: {r2_general}")
                        
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title(file_name[:-4])
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='red', lw=2)
            plt.show()
            
            # Get feature importance scores
            importances = model.feature_importances_
            
            # Create a DataFrame for better visualization
            feature_importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            
            # Print the feature importance scores
            print(feature_importance_df)
            
            #-----------------FINAL EVALUATION -------------------#
            eval_df = pd.read_csv(os.path.join(evalf_dir, file_name))
            eval_df['Date'] = eval_df['Date'].astype(int).astype(str)
            eval_df['Date'] = pd.to_datetime(eval_df['Date'], format='%Y%m%d').dt.date
            
            eval_df = pd.merge(eval_df, weather, left_on='Date', right_on='datetime')
            
            eval_df['conditions_encoded'] = label_encoder.fit_transform(eval_df['conditions'])
            eval_df['sunrise'] = eval_df['sunrise'].apply(convert_to_minutes)
                        
            # Filter data in peak period
            eval_df = eval_df[(eval_df['Interval_1'] >= start_time) & (eval_df['Interval_1'] <= end_time)]
            
            # sns.histplot(x=eval_df['SPEED_MS_AVG'])
                        
            X_li = [] # inputs
            y_li = []  # outputs
            day_li = []  # store days
            sensor_li = []  # store sensor IDs
            minute_li = []  # store Interval_1
                        
            window = 15
            
            for day in eval_df['Date'].unique(): 
                
                for sensor in eval_df['DP_ID'].unique():
                    
                    day_data = eval_df[(eval_df['Date'] == day) & (eval_df['DP_ID'] == sensor)].copy()
                    day_data.sort_values(by='Time')
                    
                    day_data['speed_avg_rolling_15'] = day_data['SPEED_MS_AVG'].rolling(window=15, min_periods=1).mean()
                                        
                    if len(day_data) == end_time-start_time+1:  # skip days with missing minutes
                                                                            
                        current_time = start_time + 16  # start from day_data minute 16
                        
                        for current_time in range(current_time, end_time-16+1):  # end at 8:29, predict 8:30-8:45
                        
                            p_start = current_time + 1
                            p_end = p_start + window
                            
                            X_li.append(day_data[day_data['Interval_1'] == current_time][features])
                            y_li.append(np.mean(day_data[(day_data['Interval_1'] >= p_start) & (day_data['Interval_1'] < p_end)][target]))
                            
                            # Track sensor ID and minute for each prediction
                            day_li.append(day)
                            sensor_li.append(sensor)  # Store the corresponding sensor ID
                            minute_li.append(current_time)  # Store the corresponding minute (Interval_1)

                    # else:
                    #    print("Skipped day and sensor: ", day, sensor) 
            
            X_eval = np.array(X_li)
            y_eval = np.array(y_li)
            X_eval = X_eval.reshape(-1, X_eval.shape[2])

            # Predict whole dataset
            y_pred = model.predict(X_eval)
            mae = mean_absolute_error(y_eval, y_pred)
            mse = mean_squared_error(y_eval, y_pred)
            r2 = r2_score(y_eval, y_pred)
            
            pred_df = pd.DataFrame({
                'DP_ID': sensor_li,
                'Date': day_li,
                'Interval_1': minute_li, 
                'Actual': y_eval,
                'Predicted': y_pred 
            })
            
            li = [file_name[:-4], mae, mse, r2]
            result_df.loc[len(result_df)] = li
            
            pred_df.to_csv(os.path.join(out_dir, 'sensors/' + file_name))
            
            print('Final evaluation dataset: ')
            print(f"Mean Absolute Error: {mae}")
            print(f"Mean Squared Error: {mse}")
            print(f"R-squared: {r2}")
            print('')

            plt.figure(figsize=(8, 6))
            plt.scatter(y_eval, y_pred, alpha=0.5)  # Plot actual vs. predicted values
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title(file_name[:-4])
            plt.plot([y_eval.min(), y_eval.max()], [y_eval.min(), y_eval.max()], linestyle='--', color='red', lw=2)
            plt.savefig(os.path.join(out_dir, f'figs/{file_name[:-4]}.png'))
            plt.show()
            # break
            result_df.to_csv(os.path.join(out_dir, 'diagnostics.csv'))
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            traceback.print_exc()
            break
