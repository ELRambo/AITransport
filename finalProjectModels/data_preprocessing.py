# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 14:10:05 2024

@author: 10449
"""

# In[]

import pandas as pd
import numpy as np
import os
import datetime
os.chdir('D:\Msc\AITransport\Project\Data')
    
# In[split weekdays and weekends]

df = pd.read_csv('evaluation_dataset.csv', sep=';')
# df = pd.read_csv('Data/evaluation_dataset.csv', sep=';')
# df = pd.read_csv('Data/final_evaluation_dataset.csv', sep=';')

df['is_weekend'] = np.nan
# df['is_snow'] = np.nan

for index, row in df.iterrows():
    
    date_obj = datetime.datetime.strptime(str(row['Date']), '%Y%m%d')

    # weekday or weekend
    if date_obj.weekday() < 5:  # weekday
        df.loc[index, 'is_weekend'] = 0
    else:  # weekend
        df.loc[index, 'is_weekend'] = 1

#     if date_obj.month in [12, 1, 2, 3]:  # snow season
#         df.loc[index, 'is_snow'] = 1  
#     else:
#         df.loc[index, 'is_snow'] = 0
        
# In[split portals]

portals = df['PORTAL'].unique()

for portal in portals:
    
    print(portal)
    
    li = []
    
    for index, row in df.iterrows():
        if row['PORTAL'] == portal:
            li.append(row)
            
    pt = pd.DataFrame(li).reset_index(drop=True)
    pt.to_csv('portalSplit/eval/' + portal + '.csv')
    
# In[]

# in_dir = 'portalSplit/train/weekdays/'
# out_dir = 'portalSplit/train/weekdays_avg/' 

in_dir = 'portalSplit/eval/'
out_dir = 'portalSplit/eval/grouped' 

for file_name in os.listdir(in_dir):
    
    if file_name.endswith('.csv'):
        
        try:
            df = pd.read_csv(os.path.join(in_dir, file_name))
            
            pt_sum = df.groupby(['Date','Time']).agg({
                'FLOW': 'sum',
                'SPEED_MS_AVG': 'mean',
            }).reset_index()
            
            pt_sum.rename(columns={
                'FLOW': 'FLOW_sum',
                'SPEED_MS_AVG': 'speed_avg'
            }, inplace=True)
            
            df_merged  = pd.merge(df, pt_sum, on=['Date','Time'], how='left')
            
            df_merged = df_merged.drop_duplicates(subset=['Date', 'Time'], keep='first')
            df_merged.sort_values(by=['Time', 'Date'], ascending=True, inplace=True)
                        
            df_merged.to_csv(os.path.join(out_dir, file_name), index=False)
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")