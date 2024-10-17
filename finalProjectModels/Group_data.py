import pandas as pd
import numpy as np
import os
import datetime



os.chdir('D:/KTH/AI/Project/Data')
in_dir = 'portalSplit/eval_final/'
out_dir = 'portalSplit/eval_final/grouped'

for file_name in os.listdir(in_dir):

    if file_name.endswith('.csv'):

        try:
            df = pd.read_csv(os.path.join(in_dir, file_name))

            pt_sum = df.groupby(['Date', 'Time']).agg({
                'FLOW': 'sum',
                'SPEED_MS_AVG': 'mean',
            }).reset_index()

            pt_sum.rename(columns={
                'FLOW': 'FLOW_sum',
                'SPEED_MS_AVG': 'speed_avg'
            }, inplace=True)

            df_merged = pd.merge(df, pt_sum, on=['Date', 'Time'], how='left')

            df_merged = df_merged.drop_duplicates(subset=['Date', 'Time'], keep='first')
            df_merged.sort_values(by=['Date', 'Time'], ascending=True, inplace=True)
            df_merged.drop(['FLOW', 'SPEED_MS_AVG', 'DP_ID'], axis=1,inplace = True)

            df_merged.to_csv(os.path.join(out_dir, f'group{file_name}.csv'), index=False)

        except Exception as e:
            print(f"Error processing {file_name}: {e}")