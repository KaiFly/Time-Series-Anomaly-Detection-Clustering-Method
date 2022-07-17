import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm_notebook

def adj_consumption(row):
    if np.isnan(row.CONSUMPTION):
        return np.nan
    elif row.SO_CTO_SHIFT1 != row.SO_CTO:
        return 0
    elif row.IMPORT_KWH_SHIFT1 > 90000 and row.IMPORT_KWH < 10000:
        return row.IMPORT_KWH + 100000 - row.IMPORT_KWH_SHIFT1
    return row.CONSUMPTION

def large_change_handling(df):
    df.NGAYGIO = pd.to_datetime(df.NGAYGIO)
    df.sort_values(by='NGAYGIO',inplace=True)
    df['IMPORT_KWH_SHIFT1'] = df.groupby("MA_DIEMDO").IMPORT_KWH.shift(1)
    df['SO_CTO_SHIFT1'] = df.groupby("MA_DIEMDO").SO_CTO.shift(1)
    df['CONSUMPTION'] = df['IMPORT_KWH'] - df['IMPORT_KWH_SHIFT1']
    df['ADJ_CONSUMPTION'] = df.apply(adj_consumption, axis=1)
    df.drop(['SO_CTO_SHIFT1','SPIDER_READ','IMPORT_KWH_SHIFT1'],axis=1,inplace=True)
    return df

def normalize_data_frequency(df_inp):
    df = df_inp.copy()
    df.sort_values(by='NGAYGIO',inplace=True)
    df['NGAYGIO_SHIFT1'] = df.groupby("MA_DIEMDO").NGAYGIO.shift(1)
    df['ADJ_IMPORT_KWH'] = df.groupby("MA_DIEMDO").ADJ_CONSUMPTION.cumsum().fillna(0)
    df['NEAREST_HOUR'] = df.NGAYGIO.dt.floor('D') + timedelta(hours=1)*((df.NGAYGIO.dt.hour + df.NGAYGIO.dt.minute/60)/6).round(0) * 6
    df['TIME_DISTANCE'] = (df.NGAYGIO-df.NEAREST_HOUR).abs()
    df.sort_values(by=['MA_DIEMDO','NEAREST_HOUR','TIME_DISTANCE'],inplace=True)
    df = df.drop_duplicates(subset=['MA_DIEMDO','NEAREST_HOUR'],keep='first')
    df['TIME_DELTA'] = (df.NGAYGIO - df.NGAYGIO_SHIFT1)/np.timedelta64(1, 'h') 
    df.TIME_DISTANCE = (df.NEAREST_HOUR - df.NGAYGIO)/np.timedelta64(1, 'h') 
    df['MILESTONE_IMPORT_KWH'] = df.ADJ_IMPORT_KWH + df.ADJ_CONSUMPTION/df.TIME_DELTA * df.TIME_DISTANCE
#     print("POSTPROCESSED NAN COUNT", df.MA_DIEMDO.isna().sum())
    res = []
    pbar = tqdm_notebook(df.MA_DIEMDO.unique())
    for d in pbar:
        pbar.set_description(f"Processing DIEMDO: {d}")
        diemdo_df = df[df.MA_DIEMDO == d][['NEAREST_HOUR',"MILESTONE_IMPORT_KWH"]]
        time_index = pd.date_range(start = diemdo_df.NEAREST_HOUR.min(),end=diemdo_df.NEAREST_HOUR.max(),freq='6H')
        diemdo_df = diemdo_df.set_index("NEAREST_HOUR").loc[time_index]
        diemdo_df['MA_DIEMDO'] = d
        diemdo_df.MILESTONE_IMPORT_KWH = diemdo_df.MILESTONE_IMPORT_KWH.interpolate()
        res.append(diemdo_df)
    return pd.concat(res)

def monthly_consumption(normalized_df):
    #normalized_df = normalized_df.sort_values(by='index')
    normalized_df['IMPORT_KWH_SHIFT1'] = normalized_df.groupby("MA_DIEMDO").MILESTONE_IMPORT_KWH.shift(1)
    normalized_df['CONSUMPTION'] = normalized_df.MILESTONE_IMPORT_KWH - normalized_df.IMPORT_KWH_SHIFT1
    normalized_df['MONTH'] = normalized_df.index.values.astype('datetime64[M]')
    monthly_consp = pd.DataFrame([])
    monthly_consp = normalized_df.groupby(["MA_DIEMDO",'MONTH']).agg({"CONSUMPTION":"sum"}).reset_index()
    monthly_consp = monthly_consp[monthly_consp.MONTH < "2018-04-01"]
    monthly_consp.sort_values(by=['MA_DIEMDO','MONTH'],inplace=True)
    monthly_consp['CONSP_LAST_MONTH'] = monthly_consp.groupby("MA_DIEMDO").CONSUMPTION.shift(1)
    monthly_consp['CONSP_LAST_2MONTH'] = monthly_consp.groupby("MA_DIEMDO").CONSUMPTION.shift(2)
    monthly_consp['CONSP_LAST_3MONTH'] = monthly_consp.groupby("MA_DIEMDO").CONSUMPTION.shift(3)
    monthly_consp['CONSP_LAST_YEAR'] = monthly_consp.groupby("MA_DIEMDO").CONSUMPTION.shift(12)
    monthly_consp['PCT_LAST_MONTH'] = (monthly_consp.CONSUMPTION/monthly_consp.CONSP_LAST_MONTH - 1).abs()*100
    monthly_consp['PCT_LAST_3MONTH'] = (monthly_consp.CONSUMPTION/((monthly_consp.CONSP_LAST_MONTH + monthly_consp.CONSP_LAST_2MONTH + monthly_consp.CONSP_LAST_3MONTH)/3) - 1).abs()*100
    monthly_consp['PCT_LAST_YEAR'] = (monthly_consp.CONSUMPTION/monthly_consp.CONSP_LAST_YEAR - 1).abs()*100
    return monthly_consp.drop(['CONSP_LAST_MONTH','CONSP_LAST_2MONTH','CONSP_LAST_3MONTH','CONSP_LAST_YEAR'], axis=1)

def weekly_comsumption(normalized_df):
    normalized_df['WEEK'] = normalized_df.index.floor("D") - pd.to_timedelta(normalized_df.index.weekday, unit='D')
    normalized_df = normalized_df[normalized_df.WEEK < '2018-04-01']
    normalized_df.MILESTONE_IMPORT_KWH = normalized_df.MILESTONE_IMPORT_KWH.interpolate() 
    normalized_df['MILESTONE_IMPORT_KWH_SHIFT1'] = normalized_df.groupby("MA_DIEMDO").MILESTONE_IMPORT_KWH.shift(1)
    normalized_df['CONSUMPTION'] = normalized_df.MILESTONE_IMPORT_KWH - normalized_df.MILESTONE_IMPORT_KWH_SHIFT1
    weekly_consp = normalized_df.groupby(["MA_DIEMDO",'WEEK']).agg({"CONSUMPTION":"sum"}).reset_index()
    weekly_consp['CONSP_LAST_WEEK'] = weekly_consp.groupby('MA_DIEMDO').CONSUMPTION.shift(1)
    weekly_consp['CONSP_LAST_2WEEK'] = weekly_consp.groupby('MA_DIEMDO').CONSUMPTION.shift(2)
    weekly_consp['CONSP_LAST_3WEEK'] = weekly_consp.groupby('MA_DIEMDO').CONSUMPTION.shift(3)
    weekly_consp['CONSP_LAST_YEAR'] = weekly_consp.groupby('MA_DIEMDO').CONSUMPTION.shift(52)
    weekly_consp['PCT_LAST_WEEK'] = (weekly_consp.CONSUMPTION/weekly_consp.CONSP_LAST_WEEK - 1).abs()*100
    weekly_consp['PCT_LAST_3WEEK'] = (weekly_consp.CONSUMPTION/((weekly_consp.CONSP_LAST_WEEK + weekly_consp.CONSP_LAST_2WEEK + weekly_consp.CONSP_LAST_3WEEK)/3) - 1).abs()*100
    weekly_consp['PCT_LAST_YEAR'] = (weekly_consp.CONSUMPTION/weekly_consp.CONSP_LAST_YEAR - 1).abs()*100
    return weekly_consp

def scale_timeseries(normalized_df, scaler):
    res = []
    pbar = tqdm_notebook(normalized_df.MA_DIEMDO.unique())
    for i in pbar:
        df_diemdo = normalized_df[normalized_df.MA_DIEMDO == str(i)]
        df_diemdo = df_diemdo.dropna(subset=['CONSUMPTION'], how='all')
        if df_diemdo.shape[0] > 1:
            df_diemdo['NGAYGIO'] = df_diemdo.index
            df_diemdo["SCALED_CONSUMPTION"] = scaler().fit_transform(df_diemdo.CONSUMPTION.values.reshape(-1,1))
            df_diemdo = df_diemdo.reset_index(drop=True)
            res.append(df_diemdo)
    return pd.concat(res).reset_index(drop=True)

def encode_timeseries(scaled_df):
    q25 = scaled_df.SCALED_CONSUMPTION.quantile(0.25)
    q50 = scaled_df.SCALED_CONSUMPTION.quantile(0.5)
    q75 = scaled_df.SCALED_CONSUMPTION.quantile(0.75)
    scaled_df['ENCODE'] = ""
    scaled_df['ENCODE'] = pd.cut(scaled_df['SCALED_CONSUMPTION'], bins=[-np.inf, q25, q50, q75, np.inf], labels = ['a','b','c','d'])
    scaled_df.ENCODE = scaled_df.ENCODE.astype(str)
    res = []
    lst_diemdo = scaled_df.MA_DIEMDO.unique()
    for i in lst_diemdo:
        df_diemdo = scaled_df[scaled_df.MA_DIEMDO == i]
        series = ''.join(df_diemdo.ENCODE.values)
        res.append(series)
    return dict(zip(lst_diemdo, res))

def max_num_consecutive_outliers(consecutive_outliers):
    # Consecutive outliers in format: [False, True, True, False, False, True, False]
    # IN this example will return (2, 1) where 2 is the max nummber of consecutive True 
    # and 2 is the index of the end of the max length sequence
    max_count = 0
    max_index = 0
    current_count = 0
    for idx, val in enumerate(consecutive_outliers):
        if val:
            current_count += 1
        else:
            current_count = 0
        if current_count >= max_count:
            max_count = current_count
            max_index = idx
    return (max_count, max_index)