import pandas as pd
import numpy as np
import os
import datetime

os.chdir('D:/KTH/AI/Project/Data')

# In[split weekdays and weekends]

df = pd.read_csv('final_evaluation_dataset.csv', sep=';')

df['is_weekend'] = np.nan
df['is_snow'] = np.nan
df['is_congestion'] = np.nan

# 设置周末和雪天标识
for index, row in df.iterrows():
    date_obj = datetime.datetime.strptime(str(row['Date']), '%Y%m%d')

    # weekday or weekend
    if date_obj.weekday() < 5:  # weekday
        df.loc[index, 'is_weekend'] = 0
    else:  # weekend
        df.loc[index, 'is_weekend'] = 1

    # snow season
    if date_obj.month in [12, 1, 2, 3]:
        df.loc[index, 'is_snow'] = 1
    else:
        df.loc[index, 'is_snow'] = 0

# In[split portals by date]

portals = df['PORTAL'].unique()
dates = df['Date'].unique()
save_path = 'PortalSplit/eval_final'

if not os.path.exists(save_path):
    os.makedirs(save_path)

# 外层循环：按 portal 处理
for portal in portals:
    print(f"Processing portal: {portal}")

    # 过滤出当前 portal 的数据
    portal_data = df[df['PORTAL'] == portal].reset_index(drop=True)

    full_portal_data = pd.DataFrame()  # 创建一个空的 DataFrame 来存储所有日期的合并数据

    # 内层循环：针对每个日期处理
    for date in dates:
        print(f"Processing date: {date} for portal: {portal}")

        # 过滤当天的数据
        date_data = portal_data[portal_data['Date'] == date].reset_index(drop=True)

        # 获取唯一的时间点和 DP_ID
        unique_times = date_data['Time'].unique()
        unique_dpid = date_data['DP_ID'].unique()

        # Creates a Cartesian product of all times and DP_IDs
        full_grid = pd.MultiIndex.from_product([unique_times, unique_dpid], names=['Time', 'DP_ID']).to_frame(
            index=False)
        df_full = pd.merge(full_grid, date_data, on=['Time', 'DP_ID'], how='left')

        # 对于 SPEED_MS 和 FLOW 列，使用相同时间的其他 DP_ID 的均值来填补空值
        df_full['SPEED_MS_AVG'] = df_full.groupby('Time')['SPEED_MS_AVG'].transform(lambda x: x.fillna(x.mean()))
        df_full['FLOW'] = df_full.groupby('Time')['FLOW'].transform(lambda x: x.fillna(x.mean()))

        # 对于 is_weekend 和 is_snow 列，使用同一时间的数据填补缺失值
        df_full['is_weekend'] = df_full.groupby('Time')['is_weekend'].transform(lambda x: x.ffill().bfill())
        df_full['is_snow'] = df_full.groupby('Time')['is_snow'].transform(lambda x: x.ffill().bfill())
        df_full['PORTAL'] = df_full.groupby('Time')['PORTAL'].transform(lambda x: x.ffill().bfill())
        df_full['Date'] = df_full.groupby('Time')['Date'].transform(lambda x: x.ffill().bfill())
        df_full['Interval_1'] = df_full.groupby('Time')['Interval_1'].transform(lambda x: x.ffill().bfill())
        df_full['Interval_5'] = df_full.groupby('Time')['Interval_5'].transform(lambda x: x.ffill().bfill())
        df_full['Interval_15'] = df_full.groupby('Time')['Interval_15'].transform(lambda x: x.ffill().bfill())
        df_full['Interval_30'] = df_full.groupby('Time')['Interval_30'].transform(lambda x: x.ffill().bfill())
        df_full['Interval_60'] = df_full.groupby('Time')['Interval_60'].transform(lambda x: x.ffill().bfill())
        # 排序：按时间排序
        df_full = df_full.sort_values(by=['Time'])

        # 将处理好的当天数据添加到总的 DataFrame 中
        full_portal_data = pd.concat([full_portal_data, df_full], ignore_index=True)

    weather_data = pd.read_csv('Stockholm,Sweden 2021-06-01 to 2022-06-30.csv')
    weather_data.rename(columns={'datetime':'Date'}, inplace=True)
    full_portal_data['Date'] = pd.to_datetime(full_portal_data['Date'], format='%Y%m%d')  # 根据实际格式调整
    weather_data['Date'] = pd.to_datetime(weather_data['Date'], format='%Y-%m-%d')  # 根据实际格式调整

    # 根据 'date' 列合并两个 DataFrame
    merged_data = pd.merge(full_portal_data, weather_data, on='Date', how='inner')  # 使用 'inner' 只保留匹配日期的数据

    # 输出合并后的 DataFrame
    print(merged_data.head())


    # 保存每个 portal 的完整数据到一个文件
    output_file = os.path.join(save_path, f"{portal}.csv")
    merged_data.to_csv(output_file, index=False)

    print(f"Saved completed data for portal {portal} to {output_file}")

