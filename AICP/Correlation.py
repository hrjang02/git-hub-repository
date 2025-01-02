#%%
'''
10월만 그린거
'''
import numpy as np
from netCDF4 import Dataset
import pandas as pd
import pyresample
import xarray as xr
import matplotlib.pyplot as plt

year = '2021'
date = '202110'
start_date = '2021-10-01'
end_date = '2021-10-31'
var = 'O3'

#관측데이터 가져오기
file = pd.read_csv(f'/home/hrjang2/{year}_O3.csv',usecols=lambda column: column != 'Unnamed: 0')
obs_stn = file.columns[2:].str.split('_').str[0]

file['측정일시'] = pd.to_datetime(file['측정일시'])
file.set_index('측정일시', inplace=True)

header_str = list(obs_stn)
Korea_columns = [col for col in file.columns if any(col.startswith(stnid) for stnid in header_str)]

data_AK_tmp = file[Korea_columns]
# data_AK_tmp = df_AK
data_AK_tmp.index = data_AK_tmp.index.tz_localize('Asia/Seoul')
data_AK_tmp.index = data_AK_tmp.index.tz_convert('UTC')
data_AK_UTC = data_AK_tmp.loc[start_date:end_date]

stn_nums = [col.split('_')[0] for col in data_AK_UTC.columns]
lon_stn = [float(col.split('_')[1]) for col in data_AK_UTC.columns]
lat_stn = [float(col.split('_')[2]) for col in data_AK_UTC.columns]


#CMAQ 가져오기
path_cmaq_o3 = f'/data05/RIST/combine/base/202101/d02/COMBINE_ACONC_v54_intel_{year}01'
cmaq_o3 = Dataset(path_cmaq_o3, mode='r')

grid_CMAQ = xr.open_dataset(f'/data05/RIST/mcip/KLC_30/202101/d02/GRIDCRO2D_{year}01_d02.nc')
lat_CMAQ = grid_CMAQ['LAT'].squeeze()
lon_CMAQ = grid_CMAQ['LON'].squeeze()
AK_def = pyresample.geometry.SwathDefinition(lons=lon_stn, lats=lat_stn)
CMAQ_def = pyresample.geometry.GridDefinition(lons=lon_CMAQ, lats=lat_CMAQ)


data_list = []
cmaq_datetime_list = []
data_reCMAQ = pd.DataFrame(columns=data_AK_UTC.columns)

ACONC_CMAQ = xr.open_dataset(f'/data05/RIST/combine/base/{date}/d02/COMBINE_ACONC_v54_intel_{date}')
daily_CONC_CMAQ = ACONC_CMAQ.variables[var]
daily_TFLAG_CMAQ = ACONC_CMAQ.variables['TFLAG'][:, 0, :].values


for i in range(daily_CONC_CMAQ.shape[0]):    #shape[0] means first dimension size of that xarray
    ozone_CMAQ = np.squeeze((daily_CONC_CMAQ[i,:,:,:])).to_numpy()
    ozone_re = pyresample.kd_tree.resample_nearest(CMAQ_def, ozone_CMAQ, AK_def, radius_of_influence=10000, fill_value=np.nan)
    ozone_re_df = pd.DataFrame(ozone_re.reshape(1, -1), columns=data_AK_UTC.columns)
    datetime_CMAQ = pd.to_datetime((np.int64(daily_TFLAG_CMAQ[i, 0]) * 1000000 + daily_TFLAG_CMAQ[i, 1]).astype(str), format='%Y%j%H%M%S')
    data_list.append(ozone_re_df)
    cmaq_datetime_list.append(datetime_CMAQ)
    print(datetime_CMAQ)

data_reCMAQ = pd.concat(data_list, ignore_index=True)
data_reCMAQ.index = cmaq_datetime_list
data_reCMAQ.index = data_reCMAQ.index.tz_localize('UTC')

Night_o3 = pd.DataFrame({'CMAQ_O3': data_reCMAQ.stack(), 'OBS_O3': data_AK_UTC.stack()})
Night_o3['OBS_O3'] = Night_o3['OBS_O3'] * 1000  # Scale OBS_O3 as per your requirements
Night_o3['Difference'] = Night_o3['CMAQ_O3'] - Night_o3['OBS_O3']  # Calculate difference
Night_o3 = Night_o3.reset_index().rename(columns={'level_0': 'datetime'})  # Adjust index structure

Night_o3.set_index('datetime', inplace=True)  # Set 'datetime' as index
#Night_o3 = Night_o3.resample(rule='D').mean(numeric_only=True)  # Resample to daily mean
Night_o3.index = Night_o3.index.tz_convert('Asia/Seoul')
Night_o3['hour']=Night_o3.index.hour

Night_o3 = Night_o3[(Night_o3['hour'] >= 21) | (Night_o3['hour'] < 3)]

obs_data = pd.to_numeric(Night_o3['OBS_O3'], errors='coerce').values
cmaq_data = pd.to_numeric(Night_o3['CMAQ_O3'], errors='coerce').values


valid_mask = ~np.isnan(obs_data) & ~np.isnan(cmaq_data)
obs_data_valid = obs_data[valid_mask]
cmaq_data_valid = cmaq_data[valid_mask]


correlation_matrix = np.corrcoef(obs_data_valid, cmaq_data_valid)
correlation_coefficient = correlation_matrix[0, 1]
print(f'상관 계수 (R): {correlation_coefficient}')

slope, intercept = np.polyfit(obs_data_valid, cmaq_data_valid, 1)
regression_line = slope * obs_data_valid + intercept

fig, ax = plt.subplots(figsize=(8, 6))

hb = ax.hexbin(obs_data_valid, cmaq_data_valid, gridsize=20, cmap='hot_r', mincnt=1)
ax.set_xlabel('Observed Concentration (ppb)')
ax.set_ylabel('CMAQ Concentration (ppb)')
ax.set_title(f'{date} Night Ozone Correlation', fontweight='bold')
fig.colorbar(hb, ax=ax, label='Counts')

ax.plot(obs_data_valid, regression_line, 'r-', label=f'Regression line\ny = {slope:.2f}x + {intercept:.2f}')
ax.plot([min(obs_data_valid), max(obs_data_valid)], [min(obs_data_valid), max(obs_data_valid)], 'k--', linewidth=2, label='1:1 Line')

ax.set_xlim([0, 80])  
ax.set_ylim([0, 80])  
ax.grid(True)
ax.legend()

plt.savefig(f'/home/hrjang2/{date}_Night_validation.png')
#%%
# ASOS data loading and integration
path = '/home/hrjang2/met/ASOS/'
variables = ['hm', 'icsr', 'pa', 'ps', 'pv', 'rn', 'ss', 'ta', 'td', 'wd', 'ws']
dataframes = {}

for var in variables:
    df = pd.read_csv(f'{path}2021_{var}.csv')
    df.set_index('tm', inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df.loc[start_date:end_date]  # Limit data to desired date range
    df = df[df.index.hour == 0]  # Filter data to midnight hours
    df[f'{var}_mean'] = df.mean(axis=1) 
    mean_list = df[f'{var}_mean'].tolist() 
    Night_o3[f'{var}_mean'] = mean_list   # Add ASOS variable to Night_o3 DataFrame

   


# CORR plot for CMAQ - OBS difference vs ASOS variables
import seaborn as sns

column = ['Difference', 'hm_mean', 'icsr_mean', 'pa_mean', 'ps_mean',
          'pv_mean', 'rn_mean', 'ss_mean', 'ta_mean', 'td_mean', 
          'wd_mean', 'ws_mean']

corr_matrix = Night_o3[column].corr()  # Calculate correlation matrix

fig, ax = plt.subplots(figsize=(12, 10))
ax.set_title(f'{date} Midnight (CMAQ - OBS) VS. (ASOS)\nCORR plot')
sns.heatmap(data=corr_matrix, annot=True, cmap='coolwarm', ax=ax, vmin=-1.0, vmax=1.0)  # Plot heatmap
plt.savefig(f'/home/hrjang2/{date}_heatmap.png')
  

#%%
'''
2021년도 Heatmap 그리기(CMAQ)

'''

import numpy as np
from netCDF4 import Dataset
import pandas as pd
import pyresample
import xarray as xr
import matplotlib.pyplot as plt

# 2021년 1월, 4월, 7월, 10월 가져오기
months = ['01', '04', '07', '10']
start_dates = ['2021-01-01', '2021-04-01', '2021-07-01', '2021-10-01']
end_dates = ['2021-01-31', '2021-04-30', '2021-07-31', '2021-10-31']
year = '2021'
var = 'O3'

# 관측 데이터 가져오기
file = pd.read_csv(f'/home/hrjang2/{year}_O3.csv', usecols=lambda column: column != 'Unnamed: 0')
obs_stn = file.columns[2:].str.split('_').str[0]

file['측정일시'] = pd.to_datetime(file['측정일시'])
file.set_index('측정일시', inplace=True)

header_str = list(obs_stn)
Korea_columns = [col for col in file.columns if any(col.startswith(stnid) for stnid in header_str)]
data_AK_tmp = file[Korea_columns]
data_AK_tmp.index = data_AK_tmp.index.tz_localize('Asia/Seoul')
data_AK_tmp.index = data_AK_tmp.index.tz_convert('UTC')
#%%
# 1월, 4월, 7월, 10월 데이터를 담을 곳
all_data = pd.DataFrame()  

for month, start_date, end_date in zip(months, start_dates, end_dates):
    data_AK_UTC = data_AK_tmp.loc[start_date:end_date]
    
    stn_nums = [col.split('_')[0] for col in data_AK_UTC.columns]
    lon_stn = [float(col.split('_')[1]) for col in data_AK_UTC.columns]
    lat_stn = [float(col.split('_')[2]) for col in data_AK_UTC.columns]

    # GRID 데이터 가져오기
    path_cmaq_o3 = f'/data05/RIST/combine/base/2021{month}/d02/COMBINE_ACONC_v54_intel_{year}{month}'
    cmaq_o3 = Dataset(path_cmaq_o3, mode='r')

    grid_CMAQ = xr.open_dataset(f'/data05/RIST/mcip/KLC_30/2021{month}/d02/GRIDCRO2D_{year}{month}_d02.nc')
    lat_CMAQ = grid_CMAQ['LAT'].squeeze()
    lon_CMAQ = grid_CMAQ['LON'].squeeze()
    AK_def = pyresample.geometry.SwathDefinition(lons=lon_stn, lats=lat_stn)
    CMAQ_def = pyresample.geometry.GridDefinition(lons=lon_CMAQ, lats=lat_CMAQ)

    data_list = []
    cmaq_datetime_list = []

    # CMAQ 데이터 가져오기
    ACONC_CMAQ = xr.open_dataset(f'/data05/RIST/combine/base/2021{month}/d02/COMBINE_ACONC_v54_intel_{year}{month}')
    daily_CONC_CMAQ = ACONC_CMAQ.variables[var]
    daily_TFLAG_CMAQ = ACONC_CMAQ.variables['TFLAG'][:, 0, :].values

    for i in range(daily_CONC_CMAQ.shape[0]):
        ozone_CMAQ = np.squeeze((daily_CONC_CMAQ[i, :, :, :])).to_numpy()
        ozone_re = pyresample.kd_tree.resample_nearest(CMAQ_def, ozone_CMAQ, AK_def, radius_of_influence=10000, fill_value=np.nan)
        ozone_re_df = pd.DataFrame(ozone_re.reshape(1, -1), columns=data_AK_UTC.columns)
        
        datetime_CMAQ = pd.to_datetime((np.int64(daily_TFLAG_CMAQ[i, 0]) * 1000000 + daily_TFLAG_CMAQ[i, 1]).astype(str), format='%Y%j%H%M%S')
        
        data_list.append(ozone_re_df)
        cmaq_datetime_list.append(datetime_CMAQ)

    data_reCMAQ = pd.concat(data_list, ignore_index=True)
    data_reCMAQ.index = cmaq_datetime_list
    data_reCMAQ.index = data_reCMAQ.index.tz_localize('UTC')

    # 정리할 데이터프레임: 21시에서 03시 데이터만 필터링
    Night_o3 = pd.DataFrame({'CMAQ_O3': data_reCMAQ.stack(), 'OBS_O3': data_AK_UTC.stack()})
    Night_o3['OBS_O3'] = Night_o3['OBS_O3'] * 1000
    Night_o3['Difference'] = Night_o3['CMAQ_O3'] - Night_o3['OBS_O3']
    Night_o3 = Night_o3.reset_index().rename(columns={'level_0': 'datetime'})
    Night_o3.set_index('datetime', inplace=True)

    # 전체 시간대 필터링
    Night_o3['hour'] = Night_o3.index.hour

    # 21시에서 03시 데이터만 필터링
    #Night_o3_filtered = Night_o3[(Night_o3['hour'] >= 21) | (Night_o3['hour'] < 3)]

    # 데이터를 하나로 합치기
    all_data = pd.concat([all_data, Night_o3], axis=0)
#%%
# 결측치 제거
all_data = all_data.dropna(axis=0)
all_data['hour'] = all_data.index.hour

# 다시 필터링
#all_data = all_data[(all_data['hour'] >= 21) | (all_data['hour'] < 3)]

#%%
'''
시계열 그래프 그리기
'''
all_data['month'] = all_data.index.month
all_data = all_data.groupby('month').mean(numeric_only=True)

filtered_data = all_data.loc[all_data.index.get_level_values('month').isin([1, 4, 7, 10])]

fig, ax = plt.subplots(figsize=(5, 3))

ax.plot(filtered_data.index, filtered_data['CMAQ_O3'], label='CMAQ_O3', color='red', marker='o')
ax.plot(filtered_data.index, filtered_data['OBS_O3'], label='OBS_O3', color='black', marker='o')

months_to_show = filtered_data.index.get_level_values('month').unique()
ax.set_xticks([filtered_data[filtered_data.index.get_level_values('month') == month].index[0] for month in months_to_show])

ax.set_xticklabels(['1', '4', '7', '10'])

ax.set_title('CMAQ_O3 vs OBS_O3', fontsize=16)
ax.set_xlabel('Month', fontsize=14)
ax.set_ylabel('Ozone Concentration', fontsize=14)

ax.legend()

plt.grid(True)
plt.savefig(f'/home/hrjang2/{year}_O3_timeseries.png')
#%%
'''
validation 그리기
'''
obs_data = pd.to_numeric(all_data['OBS_O3'], errors='coerce').values
cmaq_data = pd.to_numeric(all_data['CMAQ_O3'], errors='coerce').values

valid_mask = ~np.isnan(obs_data) & ~np.isnan(cmaq_data)
obs_data_valid = obs_data[valid_mask]
cmaq_data_valid = cmaq_data[valid_mask]

correlation_matrix = np.corrcoef(obs_data_valid, cmaq_data_valid)
correlation_coefficient = correlation_matrix[0, 1]

def calculator(actual, predicted):
    ME2 = np.abs(predicted-actual)
    ME = np.mean(ME2)
    NME = (np.sum(ME2)/np.sum(actual))*100
    MB = np.mean(predicted-actual)
    NMB = (np.sum(predicted-actual)/np.sum(actual))*100
    RMSE = np.sqrt(np.mean((predicted-actual)**2))
    
    ME = format(ME,"5.2f")
    NME = format(NME,"5.2f")
    MB = format(MB,"5.2f")
    NMB = format(NMB,"5.2f")
    RMSE = format(RMSE,"5.2f")

    return RMSE,MB,NMB,ME,NME
print(f'상관 계수 (R): {correlation_coefficient}')
print(f'RMSE:{calculator(obs_data_valid, cmaq_data_valid)[0]}')
print(f'MB:{calculator(obs_data_valid, cmaq_data_valid)[1]}')  
print(f'NMB:{calculator(obs_data_valid, cmaq_data_valid)[2]}')
print(f'ME:{calculator(obs_data_valid, cmaq_data_valid)[3]}')
print(f'NME:{calculator(obs_data_valid, cmaq_data_valid)[4]}')

slope, intercept = np.polyfit(obs_data_valid, cmaq_data_valid, 1)
regression_line = slope * obs_data_valid + intercept

fig, ax = plt.subplots(figsize=(5, 4))

hb = ax.hexbin(obs_data_valid, cmaq_data_valid, gridsize=100, cmap='hot_r', mincnt=1)
ax.set_xlabel('Observed Concentration (ppb)')
ax.set_ylabel('CMAQ Concentration (ppb)')
ax.set_title('2021 Ozone', fontweight='bold')

ax.plot(obs_data_valid, regression_line, 'r-', label=f'Regression line (R={correlation_coefficient:.2f})')  # Regression line
#\ny = {slope:.2f}x + {intercept:.2f}')
ax.plot([min(obs_data_valid), max(obs_data_valid)], [min(obs_data_valid), max(obs_data_valid)], 'k--', linewidth=2, label='1:1 Line')

ax.set_xlim([0, 80])  
ax.set_ylim([0, 80])  
ax.grid(True)
ax.legend()

plt.savefig(f'/home/hrjang2/{year}_validation.png')

#%%
# ASOS 데이터 !!
path = '/home/hrjang2/met/ASOS/'
variables = ['hm', 'icsr', 'pa', 'ps', 'pv', 'rn', 'ss', 'ta', 'td', 'wd', 'ws']


for month, start_date, end_date in zip(months, start_dates, end_dates):
    for var in variables:
        df = pd.read_csv(f'{path}2021_{var}.csv')
        df.set_index('tm', inplace=True)
        df.index = pd.to_datetime(df.index)
        
        df = df.loc[start_date:end_date]       
        df = df[(df.index.hour >= 21) | (df.index.hour <= 3)]

        df[f'{var}'] = df.mean(axis=1)
        mean_list = df[f'{var}'].tolist()
        
        # all_data와 같은 길이로 맞추기
        if len(mean_list) > len(all_data.loc[start_date:end_date]):
            mean_list = mean_list[:len(all_data.loc[start_date:end_date])]
        elif len(mean_list) < len(all_data.loc[start_date:end_date]):
            mean_list.extend([np.nan] * (len(all_data.loc[start_date:end_date]) - len(mean_list)))
            
        
        # all_data에 추가
        all_data.loc[start_date:end_date, f'{var}_ASOS'] = mean_list

#%% 
'''
CORR plot for CMAQ - OBS 오차 vs ASOS variables 
'''
import seaborn as sns

column = ['Difference', 'hm_ASOS', 'icsr_ASOS', 'pa_ASOS', 'ps_ASOS',
          'pv_ASOS', 'rn_ASOS', 'ta_ASOS', 'td_ASOS',
          'wd_ASOS', 'ws_ASOS']

corr_matrix = all_data[column].corr()

plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12)
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title(f'{year} CMAQ Difference _ ASOS variables', fontsize=15, fontweight="bold")
sns.heatmap(data=corr_matrix, annot=True, cmap='coolwarm', ax=ax, vmin=-1.0, vmax=1.0)
plt.savefig(f'/home/hrjang2/heatmap_(CMAQ - OBS) VS. (ASOS).png')

'''
'''
difference_corr = corr_matrix['Difference'].drop('Difference')
fig, ax = plt.subplots(figsize=(10, 6))

# 색상 맵 설정
norm = plt.Normalize(vmin=-1.0, vmax=1.0)
colors = plt.cm.coolwarm(norm(difference_corr.values))

bars = ax.bar(difference_corr.index, difference_corr.values, color=colors)

ax.set_title('CMAQ Difference _ ASOS variables', fontsize=16)
ax.set_xlabel('Variables', fontsize=14)
ax.set_ylabel('Difference', fontsize=14)
ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
ax.tick_params(axis='x', rotation=45) 

sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)

cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Correlation Value', fontsize=14)
plt.savefig(f'/home/hrjang2/heatmap_graph_(CMAQ - OBS) VS (ASOS).png')


#%%

        # MCIP_WD = nc_data.variables['WDIR10'][:]  # WDIR10 풍향
        # MCIP_WS = nc_data.variables['WSPD10'][:]  # WSPD10 풍속
        # MCIP_RN = nc_data.variables['RN'][:]      # RN 강수량
        # MCIP_RC = nc_data.variables['RC'][:]      # RC 강수량 총계
        # MCIP_PA = nc_data.variables['PRSFC'][:]   # PRSFC 기압


#%%
'''
WRF 계산해야함!!!!!(은별이가 줬당^^ 행복)
'''

WRF_data = pd.read_csv('/home/hrjang2/WRF_data.csv')
WRF_data['datetime'] = pd.to_datetime(WRF_data['datetime'])
WRF_data.set_index('datetime', inplace=True)
all_data.index = all_data.index.tz_convert('Asia/Seoul')


WRF_data['ta_WRF'] = WRF_data['ta_WRF'] - 273.15  # 온도 변환
WRF_data['wd_WRF'] = WRF_data['wd_WRF']           # 풍향 그대로 사용
WRF_data['ws_WRF'] = WRF_data['ws_WRF']           # 풍속 그대로 사용
WRF_data['rn_WRF'] = WRF_data['RN'] + WRF_data['RC']  # 강수량과 강설량의 합
WRF_data['hm_WRF'] = WRF_data['hm_WRF']               # 습도 그대로 사용
WRF_data['pa_WRF'] = WRF_data['pa_WRF']            # 기압 그대로 사용


combined_data = all_data.join(WRF_data)
# %%
'''
CORR plot for CMAQ - OBS 오차 vs WRF variables 
'''

column = ['Difference', 'hm_WRF', 'pa_WRF', 'rn_WRF', 'ta_WRF','wd_WRF', 'ws_WRF']

corr_matrix = combined_data[column].corr()

plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12)
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title(f'{year} CMAQ Difference _ WRF variables', fontsize=15, fontweight="bold")
sns.heatmap(data=corr_matrix, annot=True, cmap='coolwarm', ax=ax, vmin=-1.0, vmax=1.0)
plt.savefig(f'/home/hrjang2/heatmap_(CMAQ - OBS) VS. (WRF).png')

'''
'''
difference_corr = corr_matrix['Difference'].drop('Difference')


fig, ax = plt.subplots(figsize=(8, 5))

norm = plt.Normalize(vmin=-1.0, vmax=1.0)
colors = plt.cm.coolwarm(norm(difference_corr.values))

bars = ax.bar(difference_corr.index, difference_corr.values, color=colors)

ax.set_title('CMAQ Difference _ WRF variables', fontsize=16, fontweight='bold')
#ax.set_xlabel('Variables', fontsize=14)
ax.set_ylabel('Correlation', fontsize=14)
ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
ax.tick_params(axis='x', rotation=45) 

sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
sm.set_array([])  
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Correlation Value', fontsize=14)
plt.savefig(f'/home/hrjang2/heatmap_graph_(CMAQ - OBS) VS (WRF).png')

# %%
'''
CORR plot for CMAQ - OBS 오차 vs WRF 오차variables 
'''

combined_data['hm']=combined_data['hm_WRF']-combined_data['hm_ASOS']
combined_data['pa']=combined_data['pa_WRF']-combined_data['pa_ASOS']
combined_data['rn']=combined_data['rn_WRF']-combined_data['rn_ASOS']
combined_data['ta']=combined_data['ta_WRF']-combined_data['ta_ASOS']
combined_data['wd']=combined_data['wd_WRF']-combined_data['wd_ASOS']
combined_data['ws']=combined_data['ws_WRF']-combined_data['ws_ASOS']


column = ['Difference', 'hm', 'pa', 'rn', 'ta','wd', 'ws']

corr_matrix = combined_data[column].corr()

plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12)
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title(f'{year} CMAQ Difference _ WRF variables Difference', fontsize=15, fontweight="bold")
sns.heatmap(data=corr_matrix, annot=True, cmap='coolwarm', ax=ax, vmin=-1.0, vmax=1.0)
plt.savefig(f'/home/hrjang2/heatmap_(CMAQ - OBS) VS (WRF-ASOS).png')

#%%
'''
'''
difference_corr = corr_matrix['Difference'].drop('Difference')


fig, ax = plt.subplots(figsize=(8, 5))

norm = plt.Normalize(vmin=-1.0, vmax=1.0)
colors = plt.cm.coolwarm(norm(difference_corr.values))

bars = ax.bar(difference_corr.index, difference_corr.values, color=colors)

ax.set_title('CMAQ Difference _ WRF variables Difference', fontsize=16, fontweight='bold')
# ax.set_xlabel('Variables', fontsize=14)
ax.set_ylabel('Correlation', fontsize=14)
ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
ax.tick_params(axis='x', rotation=45) 

sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
sm.set_array([])  
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Correlation Value', fontsize=14)
plt.savefig(f'/home/hrjang2/heatmap_graph_(CMAQ - OBS) VS (WRF-ASOS).png')


# %%
