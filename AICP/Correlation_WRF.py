#%%
'''
 Heatmap(WRF)
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

# 여러 변수를 리스트로 저장
variables = ['TEMPG', 'WDIR10', 'WSPD10', 'RN', 'RC', 'Q2', 'PRSFC']  # 여기에 필요한 변수를 추가

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

# 전체 데이터를 담을 곳
all_data = pd.DataFrame()

for var in variables:
    var_data = pd.DataFrame()

    for month, start_date, end_date in zip(months, start_dates, end_dates):
        data_AK_UTC = data_AK_tmp.loc[start_date:end_date]

        stn_nums = [col.split('_')[0] for col in data_AK_UTC.columns]
        lon_stn = [float(col.split('_')[1]) for col in data_AK_UTC.columns]
        lat_stn = [float(col.split('_')[2]) for col in data_AK_UTC.columns]

        # GRID 데이터 가져오기
        path_WRF_o3 = f'/data05/RIST/combine/base/2021{month}/d02/COMBINE_ACONC_v54_intel_{year}{month}'
        WRF_o3 = Dataset(path_WRF_o3, mode='r')

        grid_WRF = xr.open_dataset(f'/data05/RIST/mcip/KLC_30/2021{month}/d02/GRIDCRO2D_{year}{month}_d02.nc')
        lat_WRF = grid_WRF['LAT'].squeeze()
        lon_WRF = grid_WRF['LON'].squeeze()
        AK_def = pyresample.geometry.SwathDefinition(lons=lon_stn, lats=lat_stn)
        WRF_def = pyresample.geometry.GridDefinition(lons=lon_WRF, lats=lat_WRF)

        data_list = []
        WRF_datetime_list = []
        data_reWRF = pd.DataFrame(columns=data_AK_UTC.columns)

        # WRF 데이터
        ACONC_WRF = xr.open_dataset(f'/data05/RIST/mcip/KLC_30/2021{month}/d02/METCRO2D_{year}{month}_d02.nc')
        daily_CONC_WRF = ACONC_WRF.variables[var]
        daily_TFLAG_WRF = ACONC_WRF.variables['TFLAG'][:, 0, :].values

        for i in range(daily_CONC_WRF.shape[0]):
            ozone_WRF = np.squeeze((daily_CONC_WRF[i, :, :, :])).to_numpy()
            ozone_re = pyresample.kd_tree.resample_nearest(WRF_def, ozone_WRF, AK_def, radius_of_influence=10000, fill_value=np.nan)
            ozone_re_df = pd.DataFrame(ozone_re.reshape(1, -1), columns=data_AK_UTC.columns)

            datetime_WRF = pd.to_datetime((np.int64(daily_TFLAG_WRF[i, 0]) * 1000000 + daily_TFLAG_WRF[i, 1]).astype(str), format='%Y%j%H%M%S')

            data_list.append(ozone_re_df)
            WRF_datetime_list.append(datetime_WRF)

        data_reWRF = pd.concat(data_list, ignore_index=True)
        data_reWRF.index = WRF_datetime_list
        data_reWRF.index = data_reWRF.index.tz_localize('UTC')

        # 정리할 데이터프레임: 21시에서 03시 데이터만 필터링
        Night_o3 = pd.DataFrame({f'{var}': data_reWRF.stack()})
        Night_o3 = Night_o3.reset_index().rename(columns={'level_0': 'datetime'})
        Night_o3.set_index('datetime', inplace=True)

        Night_o3_filtered = Night_o3[(Night_o3.index.hour >= 21) | (Night_o3.index.hour <= 3)]
        Night_o3_filtered = Night_o3_filtered.resample(rule='H').mean(numeric_only=True)
        var_data = pd.concat([var_data, Night_o3_filtered], axis=0)
    var_data = var_data[var_data.index.month.isin([1, 4, 7, 10])]
    all_data = pd.concat([all_data, var_data], axis=1)

all_data = all_data.dropna(axis=0)


# %%
all_data.to_csv('/home/hrjang2/WRF_data.csv')


# %%
