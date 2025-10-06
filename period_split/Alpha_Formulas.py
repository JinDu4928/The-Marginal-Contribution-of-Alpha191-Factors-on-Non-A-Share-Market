# (-1 * CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
from scipy.stats import rankdata
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
def rapply(a, b, c):
    if str(pd.__version__) >= '0.23.0':
        temp = a.rolling(b, min_periods=b).apply(c, raw=True)
    elif str(pd.__version__) >= '0.19.0':
        temp = a.rolling(b, min_periods=b).apply(c)
    else:
        temp = pd.rolling_apply(a, b, c, min_periods=b)
    return temp
def func_decaylinear(a, d):
    seq1 = (np.arange(0, d, 1) + 1)
    weight1 = seq1 / sum(seq1)
    part1 = lambda x: sum(x * weight1)
    output = rapply(a, d, part1)
    output = output.iloc[d - 1:, :]
    return output
def rcorr(a, b, c):
    if str(pd.__version__) >= '0.19.0':
        temp = a.rolling(c, min_periods=0).corr(b)
    else:
        temp = pd.rolling_corr(a, b, c, min_periods=0)
    # https://github.com/pandas-dev/pandas/issues/18430
    valid = (temp < np.inf) & (temp > -np.inf) & ((temp <= -1e-5) | (temp >= 1e-5))
    return temp.where(valid, np.nan)
def rsum(a, b):
    if str(pd.__version__) >= '0.19.0':
        temp = a.rolling(b, min_periods=0).sum()
    else:
        temp = pd.rolling_sum(a, b, min_periods=0)
    return temp
def rstd(a, b):
    if str(pd.__version__) >= '0.19.0':
        temp = a.rolling(b, min_periods=0).std()
    else:
        temp = pd.rolling_std(a, b, min_periods=0)
    return temp
def rmean(a, b):
    if str(pd.__version__) >= '0.19.0':
        temp = a.rolling(b, min_periods=0).mean()
    else:
        temp = pd.rolling_mean(a, b, min_periods=0)
    return temp
def func_tsrank(a, n):
    output = pd.DataFrame()
    def rk(x):
        if np.isnan(x[-1]):
            return np.nan
        return rankdata(x)[-1] / max(len(x[~np.isnan(x)]), 1)
    for stock in a.columns:
        output[stock] = rapply(a[stock], n, rk)
    return output
def ewmm(a, b):
    if str(pd.__version__) >= '0.19.0':
        temp = a.ewm(b, adjust=False).mean()
    else:
        temp = pd.ewma(a, b, adjust=False)
    return temp
def func_ret(code):
    return code['adj_close'].pct_change().fillna(0.0)
def func_wma(a, n):
    output = pd.DataFrame()
    for stock in a.columns:
        temp = 0
        for i in range(n):
            temp = a[stock][i] * 0.9 * (i + 1) + temp
        output[stock] = pd.Series(temp)
        output = pd.DataFrame(output)
    return output
def rmin(a, b):
    if str(pd.__version__) >= '0.19.0':
        temp = a.rolling(b, min_periods=0).min()
    else:
        temp = pd.rolling_min(a, b, min_periods=0)
    return temp
def rmax(a, b):
    if str(pd.__version__) >= '0.19.0':
        temp = a.rolling(b, min_periods=0).max()
    else:
        temp = pd.rolling_max(a, b, min_periods=0)
    return temp
def calculate_avg(code):
    return (code['volume'] * code['adj_close']).cumsum() / code['volume'].cumsum()
def func_dbm(code):
    cond1 = code['open'] >= code['open'].shift()
    data1 = np.maximum(code['open'] - code['low'], code['open'] - code['open'].shift())
    data1[cond1] = 0
    return data1
def func_dtm(code):
    cond1 = code['open'] <= code['open'].shift()
    data1 = np.maximum(code['high'] - code['open'], code['open'] - code['open'].shift())
    data1[cond1] = 0
    return data1
def get_bench(code):
    datas = code['volume'].index.sort_values()
    bio_ = pd.DataFrame(index=datas, columns=code['volume'].columns, dtype=np.float64)
    matched_values1 = code['bench_open'].loc[bio_.index]
    bio_.loc[:, :] = matched_values1['openprice'].values[:, None]
    bic_ = pd.DataFrame(index=datas, columns=code['volume'].columns, dtype=np.float64)
    matched_values2 = code['bench_close'].loc[bic_.index]
    bic_.loc[:, :] = matched_values2['closeprice'].values[:, None]
    return bio_, bic_
def roll_cov(a, b, c):
    if str(pd.__version__) >= '0.19.0':
        temp = a.rolling(c, min_periods=0).cov(b)
    else:
        temp = pd.rolling_cov(a, b, c, min_periods=0)
    return temp
def func_lowday(a, n):
    return a.iloc[:-(n + 1):-1].reset_index(drop=True).idxmin()
def func_highday(a, n):
    return a.iloc[:-(n + 1):-1].reset_index(drop=True).idxmax()
def alpha_001(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(7) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    data1 = np.log(volume_).diff(1).rank(axis=1, pct=True)
    data2 = ((close_ - open_) / open_).rank(axis=1, pct=True)
    alpha = -data1.iloc[-6:, :].corrwith(data2.iloc[-6:, :])
    return alpha.dropna()
# -1 * delta((((close-low)-(high-close))/((high-low)),1))
def alpha_002(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(2) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    temp = (((close_ - low_) - (high_ - close_)) / (high_ - low_)).diff(1)
    alpha = temp.iloc[-1, :]
    return (alpha * -1).dropna()
# SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6)
def alpha_003(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(7) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    delay1 = close_.shift().iloc[-6:]
    # condtion1 = (close_ == delay1)
    condition2 = (close_.iloc[-6:] > delay1)
    condition3 = (close_.iloc[-6:] < delay1)

    part2 = close_.iloc[-6:] - np.minimum(delay1.where(condition2, np.nan), low_.iloc[-6:].where(condition2, np.nan))  # 取最近的6位数据
    part3 = close_.iloc[-6:] - np.maximum(delay1.where(condition3, np.nan), high_.iloc[-6:].where(condition3, np.nan))

    result = part2.fillna(0) + part3.fillna(0)
    alpha = result.sum()
    return alpha.dropna()
# ((((SUM(CLOSE,8)/8)+STD(CLOSE,8))<(SUM(CLOSE,2)/2))?(-1*1):(((SUM(CLOSE,2)/2)<((SUM(CLOSE,8)/8)-STD(CLOSE,8)))?1:(((1<(VOLUME/MEAN(VOLUME,20)))||((VOLUME/MEAN(VOLUME,20))==1))?1:(-1*1))))
def alpha_004(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(20) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    condition1 = (rsum(close_, 8).iloc[-1] / 8 + rstd(close_, 8).iloc[-1] < rsum(close_, 2).iloc[-1] / 2)
    condition2 = (rsum(close_, 2).iloc[-1] / 2 < (rsum(close_, 8).iloc[-1] / 8 - rstd(close_, 8).iloc[-1]))
    condition3 = 1 <= (volume_.iloc[-1] / rmean(volume_, 20).iloc[-1])

    alpha = pd.Series(-1, index=code['open'].columns)
    alpha.loc[~condition1 & (condition2 | condition3)] = 1

    return alpha.dropna()
'''# (-1*TSMAX(CORR(TSRANK(VOLUME,5),TSRANK(HIGH,5),5),3))
def alpha_005(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(11) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    ts_volume = func_tsrank(volume_.iloc[-11:], 5).iloc[-7:]
    ts_high = func_tsrank(high_.iloc[-11:], 5).iloc[-7:]
    corr_ts = rcorr(ts_high, ts_volume.iloc[-11:], 5).iloc[-3:]
    alpha = -corr_ts.max()
    alpha = alpha.where((alpha < np.inf) & (alpha > -np.inf), np.nan)
    return alpha.dropna()'''
# (RANK(SIGN(DELTA((((OPEN*0.85)+(HIGH*0.15))),4)))*-1)
def alpha_006(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(5) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    temp_1 = open_.iloc[-5:] * 0.85 + high_.iloc[-5:] * 0.15
    alpha = -1 * np.sign(temp_1.diff(4)).iloc[-1, :].rank(pct=True)
    return alpha.dropna()
# ((RANK(MAX((VWAP-CLOSE),3))+RANK(MIN((VWAP-CLOSE),3)))*RANK(DELTA(VOLUME,3)))
def alpha_007(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code) 
    temp = (avg_ - close_).iloc[-1]
    temp_1 = temp.clip(lower=3).rank(pct=True)
    temp_2 = temp.clip(upper=3).rank(pct=True)
    temp_3 = volume_.diff(3).iloc[-1].rank(pct=True)
    alpha = (temp_1 + temp_2) * temp_3
    return alpha.dropna()
# RANK(DELTA(((((HIGH+LOW)/2)*0.2)+(VWAP*0.8)),4)*-1
def alpha_008(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    temp_1 = ((high_ + low_) / 2) * 0.2 + avg_ * 0.8
    temp_2 = temp_1.diff(4) * -1
    alpha = temp_2.iloc[-1].rank(pct=True)
    return alpha.dropna()
# SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,7,2)
def alpha_009(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    temp = ((high_ + low_) * 0.5 - (high_.shift() + low_.shift()) * 0.5) * (
            high_ - low_) / volume_ 
    # 1 / (1 + com) = 2 / 7
    result = ewmm(temp, 2.5)
    alpha = result.iloc[-1, :]
    return alpha.dropna()
# (RANK(MAX(((RET<0)?STD(RET,20):CLOSE)^2,5)))
def alpha_010(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(25) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    temp = close_.iloc[-25:].pct_change().iloc[1:].fillna(0.0)
    condition = temp.iloc[-5:] < 0
    part1 = rstd(temp, 20).iloc[-5:]
    part1[condition] = close_.iloc[-5:]
    result = (part1 ** 2).max()
    alpha = result.rank(pct=True)
    return alpha.dropna()
# SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,6)
def alpha_011(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(6) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    temp_1 = (close_ - low_) - (high_ - close_)
    temp_2 = (high_ - low_)
    temp = (temp_1 / temp_2) * volume_
    alpha = (temp.iloc[-6:, :]).sum()
    alpha[alpha == 0] = np.nan
    return alpha.dropna()
# (RANK((OPEN-(SUM(VWAP,10)/10))))*(-1*(RANK(ABS((CLOSE-VWAP)))))
def alpha_012(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    temp1 = open_.iloc[-1] - (avg_.iloc[-10:, :]).sum() / 10.
    part1 = temp1.rank(pct=True)
    temp2 = (close_.iloc[-1] - avg_.iloc[-1]).abs()
    part2 = -temp2.rank(pct=True)
    alpha = part1 * part2 * -1
    return alpha.dropna()
# (((HIGH*LOW)^0.5)-VWAP)
def alpha_013(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    result = ((high_ * low_) ** 0.5) - avg_
    alpha = result.iloc[-1, :]
    return alpha.dropna()
# CLOSE-DELAY(CLOSE,5)
def alpha_014(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(6) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    temp = close_ - close_.shift(5)
    alpha = temp.iloc[-1, :]
    return alpha.dropna()
# OPEN/DELAY(CLOSE,1)-1
def alpha_015(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(2) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    temp = open_ / close_.shift(1) - 1
    alpha = temp.iloc[-1, :]
    return alpha.dropna()
'''# (-1*TSMAX(RANK(CORR(RANK(VOLUME),RANK(VWAP),5)),5))
def alpha_016(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    temp1 = volume_.rank(axis=1, pct=True)
    temp2 = avg_.rank(axis=1, pct=True)
    part = rcorr(temp1, temp2, 5).iloc[-5:, :]
    part = part[(part < np.inf) & (part > -np.inf)].rank(axis=1, pct=True)
    alpha = -part.max()
    return alpha.dropna()'''
# RANK((VWAP-TSMAX(VWAP,15)))^DELTA(CLOSE,5)
def alpha_017(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    temp1 = avg_.iloc[-15:].max()
    temp2 = (avg_.iloc[-1] - temp1)
    part1 = temp2.rank(pct=True)
    part2 = close_.diff(5).iloc[-1]
    alpha = part1 ** part2
    return alpha.dropna()
# CLOSE/DELAY(CLOSE,5)
def alpha_018(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(6) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    delay5 = close_.shift(5).iloc[-1]
    alpha = close_.iloc[-1] / delay5
    return alpha.dropna()
# (CLOSE<DELAY(CLOSE,5)?(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5):(CLOSE=DELAY(CLOSE,5)?0:(CLOSE-DELAY(CLOSE,5))/CLOSE))
def alpha_019(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(6) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    delay5 = close_.shift(5).iloc[-1]
    close = close_.iloc[-1]
    alpha = pd.Series(0.0, index=close.index)
    condition1 = (close < delay5)
    condition3 = (close > delay5)
    alpha[condition1] = close.where(condition1, np.nan) / delay5.where(condition1, np.nan) - 1.
    alpha[condition3] = 1 - delay5.where(condition3, np.nan) / close.where(condition3, np.nan)
    return alpha.dropna()
# (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100
def alpha_020(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(7) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    delay6 = close_.shift(6).iloc[-1]
    alpha = (close_.iloc[-1] - delay6) * 100 / delay6
    return alpha.dropna()
# EGBETA(MEAN(CLOSE,6),SEQUENCE(6))
def alpha_021(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(11) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    A = rmean(close_, 6).iloc[-6:, :]
    B = pd.Series(np.arange(1, 7), index=A.index) 
    corr = A.corrwith(B)
    alpha = corr * A.std() / B.std()
    return alpha.dropna()
# SMEAN(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)
def alpha_022(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    part1 = (close_ - rmean(close_, 6)) / rmean(close_, 6)
    temp = (close_ - rmean(close_, 6)) / rmean(close_, 6)
    part2 = temp.shift(3)
    result = part1 - part2
    # 1 / (1 + com) = 1 / 12
    result = ewmm(result, 11)
    alpha = result.iloc[-1, :]
    return alpha.dropna()
# SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE:20),0),20,1)/(SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)+SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1))*100
def alpha_023(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    condition1 = (close_ > close_.shift())
    temp1 = rstd(close_, 20).where(condition1, np.nan)
    temp1 = temp1.fillna(0)
    temp2 = rstd(close_, 20).where(~condition1, np.nan)
    temp2 = temp2.fillna(0)
    # 1 / (1 + com) = 1 / 20
    part1 = ewmm(temp1, 19)
    part2 = ewmm(temp2, 19)
    alpha = part1.iloc[-1] * 100 / (part1.iloc[-1] + part2.iloc[-1])
    return alpha.dropna()
# SMA(CLOSE-DELAY(CLOSE,5),5,1)
def alpha_024(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    delay5 = close_.shift(5)
    result = close_ - delay5
    # 1 / (1 + com) = 1 / 5
    result = ewmm(result, 4)
    alpha = result.iloc[-1, :]
    return alpha.dropna()
# ((-1*RANK((DELTA(CLOSE,7)*(1-RANK(DECAYLINEAR((VOLUME/MEAN(VOLUME,20)),9))))))*(1+RANK(SUM(RET,250))))
def alpha_025(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    temp_1 = close_.iloc[-8:].diff(7).iloc[-1]
    temp_2 = 1 - func_decaylinear(volume_.iloc[-9:] / rmean(volume_.iloc[-28:], 20).iloc[-9:], 9).iloc[-1].rank(pct=True)
    part_1 = (temp_1 * temp_2).rank(pct=True)
    part_2 = (func_ret(code).iloc[-250:]).sum().rank(pct=True)
    alpha = -part_1 * (part_2 + 1)
    alpha[alpha == 0] = np.nan
    return alpha.dropna()
# ((((SUM(CLOSE, 7) / 7) - CLOSE)) + ((CORR(VWAP, DELAY(CLOSE, 5), 230))))
def alpha_026(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    part1 = (close_.iloc[-7:]).sum() / 7 - close_.iloc[-1]
    delay5 = close_.shift(5)
    part2 = rcorr(avg_.iloc[-230:], delay5.iloc[-230:], 230)
    part2 = part2.iloc[-1, :]
    alpha = part1 + part2
    return alpha.dropna()
# WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100,12)
def alpha_027(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    temp_1 = ((close_ - close_.shift(3)) * 100) / close_.shift(3)
    temp_2 = ((close_ - close_.shift(6)) * 100) / close_.shift(6)
    alpha = func_wma(temp_1.iloc[6:, ] + temp_2.iloc[6:, ], 12)
    alpha = alpha.iloc[-1, :]
    return alpha.dropna()
# 3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)-2*SMA(SMA((CLOSE-TSMIN(LOW,9))/( MAX(HIGH,9)-TSMAX(LOW,9))*100,3,1),3,1)
def alpha_028(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    temp1 = close_ - rmin(low_, 9)
    temp2 = rmax(high_, 9) - rmin(low_, 9)
    # 1 / (1 + com) = 1 / 3
    temp3 = ewmm(temp1 * 100 / temp2, 2)
    part1 = 3 * temp3
    part2 = 2 * ewmm(temp3, 2)
    result = part1 - part2
    alpha = result.iloc[-1, :] 
    return alpha.dropna()
# (CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME
def alpha_029(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(7)for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    delay6 = close_.shift(6).iloc[-1]
    alpha = ((close_.iloc[-1] - delay6) / delay6) * volume_.iloc[-1]
    return alpha.dropna()
# WMA((REGRESI(CLOSE/DELAY(CLOSE)-1,MKT,SMB,HML，60))^2,20)
# CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100
def alpha_031(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(12) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    alpha = (close_.iloc[-1] - rmean(close_, 12).iloc[-1]) * 100 / rmean(close_, 12).iloc[-1]
    return alpha.dropna()
# (-1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3))
def alpha_032(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(5) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    temp1 = high_.iloc[-5:].rank(axis=1, pct=True)
    temp2 = volume_.iloc[-5:].rank(axis=1, pct=True)
    temp3 = rcorr(temp1, temp2, 3).iloc[-3:, :]
    result = temp3.rank(axis=1, pct=True)
    alpha = -result.sum()
    return alpha.dropna()
# ((((-1*TSMIN(LOW,  5))+DELAY(TSMIN(LOW,5),5))*RANK(((SUM(RET,240)-SUM(RET,20))/220)))*TSRANK(VOLUME,5))
def alpha_033(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    ret = close_.pct_change().fillna(0.0)
    temp1 = rmin(low_, 5)  # TS_MIN
    part1 = temp1.shift(5) - temp1
    part1 = part1.iloc[-1, :]
    temp2 = (rsum(ret.iloc[-240:], 240).iloc[-1] - rsum(ret.iloc[-20], 20).iloc[-1]) / 220
    part2 = temp2.rank(pct=True)
    temp3 = volume_.iloc[-5:, :]
    part3 = temp3.rank(axis=0, pct=True)  # TS_RANK
    part3 = part3.iloc[-1, :]
    alpha = part1 * part2 * part3
    return alpha.dropna()
# MEAN(CLOSE,12)/CLOSE
def alpha_034(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    result = rmean(close_, 12) / close_
    alpha = result.iloc[-1, :]
    return alpha.dropna()
# (MIN(RANK(DECAYLINEAR(DELTA(OPEN,1),15)),RANK(DECAYLINEAR(CORR((VOLUME),((OPEN *0.65)+(OPEN*0.35)),17),7)))*-1)
def alpha_035(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(23) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    temp_1 = open_.diff(1)
    temp_1 = temp_1.fillna(0)
    part_1 = func_decaylinear(temp_1.iloc[-15:], 15).iloc[-1]
    cond1 = part_1 == 0
    part_1[cond1] = np.nan
    part_1 = part_1.rank(pct=True)
    temp_2 = open_.iloc[-23:] * 0.65 + open_.iloc[-23:] * 0.35
    temp_3 = rcorr(volume_.iloc[-23:], temp_2, 17).iloc[-7:]
    temp_3 = temp_3.fillna(0)
    part_2 = func_decaylinear(temp_3, 7).iloc[-1]
    cond2 = part_2 == 0
    part_2[cond2] = np.nan
    part_2 = part_2.rank(pct=True)
    alpha = np.minimum(part_1, part_2)
    return alpha.dropna()
'''# RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP), 6), 2))
def alpha_036(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    temp1 = volume_.rank(axis=1, pct=True)
    temp2 = avg_.rank(axis=1, pct=True)
    part1 = rcorr(temp1, temp2, 6).iloc[-2:]
    result = part1.sum()
    result = result.rank(pct=True)
    alpha = result
    return alpha.dropna()'''
# (-1*RANK(((SUM(OPEN,5)*SUM(RET,5))-DELAY((SUM(OPEN,5)*SUM(RET,5)),10))))
def alpha_037(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(16) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    ret = close_.pct_change().fillna(0.0)
    temp = rsum(open_, 5) * rsum(ret, 5)
    part1 = temp.rank(axis=1, pct=True)
    part2 = temp.shift(10)
    result = -part1 - part2
    alpha = result.iloc[-1, :]
    return alpha.dropna()
# (((SUM(HIGH, 20) / 20) < HIGH) ? (-1 * DELTA(HIGH, 2)) : 0)
def alpha_038(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(20) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    sum_20 = high_.iloc[-20:].sum() / 20
    delta2 = high_.iloc[-3:].diff(2).iloc[-1]
    condition = (sum_20 < high_.iloc[-1])
    alpha = -delta2.where(condition, 0)
    return alpha.dropna()
# ((RANK(DECAYLINEAR(DELTA((CLOSE), 2),8)) - RANK(DECAYLINEAR(CORR(((VWAP * 0.3) + (OPEN * 0.7)), SUM(MEAN(VOLUME,180), 37), 14), 12))) * -1
def alpha_039(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    part_1 = func_decaylinear(close_.iloc[-10:].diff(2), 8).iloc[-1]
    cond1 = part_1 == 0
    part_1[cond1] = np.nan
    part_1 = part_1.rank(pct=True)
    temp_1 = avg_ * 0.3 + open_ * 0.7
    temp_2 = rsum(rmean(volume_, 180), 37)
    part_2 = func_decaylinear(rcorr(temp_1.iloc[-25:], temp_2.iloc[-25:], 14), 12).iloc[-1]
    cond2 = part_2 == 0
    part_2[cond2] = np.nan
    part_2 = part_2.rank(pct=True)
    alpha = part_1 - part_2
    return alpha.dropna()
# SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:0),26)/SUM((CLOSE<=DELAY(CLOSE,1)?VOLUME:0),26)*100
def alpha_040(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(27) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    delay1 = close_.shift()
    condition = (close_ > delay1)
    vol = volume_.where(condition, np.nan).fillna(0)
    vol_sum = rsum(vol.iloc[-26:], 26)
    vol1 = volume_[~condition].fillna(0)
    vol1_sum = rsum(vol1.iloc[-26:], 26)
    result = 100 * vol_sum / vol1_sum
    result = result.iloc[-1, :]
    alpha = result
    return alpha.dropna()
# (RANK(MAX(DELTA((VWAP), 3), 5))* -1)
def alpha_041(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    delta_avg = avg_.iloc[-8:].diff(3).iloc[-5:]
    part = delta_avg.max()
    result = -part.rank(pct=True)
    alpha = result
    return alpha.dropna()
# (-1 * RANK(STD(HIGH, 10))) * CORR(HIGH, VOLUME, 10))
def alpha_042(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(10) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    part1 = rcorr(high_.iloc[-10:], volume_.iloc[-10:], 10).iloc[-1]
    part2 = rstd(high_.iloc[-10:], 10).iloc[-1]
    part2 = part2.rank(pct=True)
    result = -part1 * part2
    alpha = result
    return alpha.dropna()
# SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),6)
def alpha_043(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(7) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    delay1 = close_.shift().iloc[-6:]
    condition1 = (close_.iloc[-6:] > delay1)
    condition2 = (close_.iloc[-6:] < delay1)
    temp1 = volume_.iloc[-6:].where(condition1, np.nan).fillna(0)
    temp2 = -volume_.iloc[-6:].where(condition2, np.nan).fillna(0)
    result = temp1 + temp2
    alpha = result.sum()
    return alpha.dropna()
# (TSRANK(DECAYLINEAR(CORR(((LOW )), MEAN(VOLUME,10), 7), 6),4) + TSRANK(DECAYLINEAR(DELTA((VWAP), 3), 10), 15))
def alpha_044(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    temp_1 = rcorr(low_.iloc[-15:], rmean(volume_.iloc[-24:], 10).iloc[-15:], 7).iloc[-9:]
    part_1 = func_decaylinear(temp_1, 6).iloc[-4:]
    cond1 = part_1 == 0
    part_1[cond1] = np.nan
    part_1 = part_1.rank(axis=0, pct=True).iloc[-1]

    temp_2 = func_decaylinear(avg_.diff(3).iloc[-24:], 10).iloc[-15:]
    cond2 = temp_2 == 0
    temp_2[cond2] = np.nan
    part_2 = temp_2.rank(axis=0, pct=True).iloc[-1]
    alpha = part_1 + part_2
    return alpha.dropna()
# (RANK(DELTA((((CLOSE * 0.6) + (OPEN *0.4))), 1)) * RANK(CORR(VWAP, MEAN(VOLUME,150), 15)))
def alpha_045(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    temp1 = close_.iloc[-2:] * 0.6 + open_.iloc[-2:] * 0.4
    part1 = temp1.diff().iloc[-1]
    part1 = part1.rank(pct=True)
    temp2 = rmean(volume_, 150).iloc[-15:]
    part2 = rcorr(avg_.iloc[-15:], temp2, 15).iloc[-1]
    part2 = part2.rank(pct=True)
    alpha = part1 * part2
    return alpha.dropna()
# (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)
def alpha_046(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(24) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    part_1 = close_.iloc[-3:].mean()
    part_2 = close_.iloc[-6:].mean()
    part_3 = close_.iloc[-12:].mean()
    part_4 = close_.iloc[-24:].mean()
    alpha = (part_1 + part_2 + part_3 + part_4) / (4 * close_.iloc[-1])
    return alpha.dropna()
# SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,9,1)
def alpha_047(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    part1 = rmax(high_, 6) - close_
    part2 = rmax(high_, 6) - rmin(low_, 6)
    # 1 / (1 + com) = 1 / 9
    result = ewmm(100 * part1 / part2, 8)
    alpha = result.iloc[-1, :]
    return alpha.dropna()
#  (-1*((RANK(((SIGN((CLOSE-DELAY(CLOSE,1)))+SIGN((DELAY(CLOSE,1) - DELAY(CLOSE,2))))+SIGN((DELAY(CLOSE,2)-DELAY(CLOSE,3))))))*SUM(VOLUME,5))/SUM(VOLUME,20))
def alpha_048(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(20) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    close_diff = close_.iloc[-4:].diff()
    summ=(np.sign(close_diff.iloc[-1]) + np.sign(close_diff.iloc[-2]) + np.sign(close_diff.iloc[-3])).rank(pct=True)
    alpha = -summ * volume_.iloc[-5:].sum() / volume_.iloc[-20:].sum()
    return alpha.dropna()
# SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(L OW,1)))),12)/(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(L OW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HI GH,1)),ABS(LOW-DELAY(LOW,1)))),12))
def alpha_049(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(13) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    delay_high = high_.shift()
    delay_low = low_.shift()
    condition1 = (high_ + low_ >= delay_high + delay_low)
    condition2 = (high_ + low_ <= delay_high + delay_low)
    part = np.maximum(np.abs(high_ - delay_high), np.abs(low_ - delay_low))
    part1 = part.copy()
    part1[condition1] = 0
    part1 = part1.iloc[-12:].sum()
    part2 = part.copy()
    part2[condition2] = 0
    part2 = part2.iloc[-12:].sum()
    result = part1 / (part1 + part2)
    alpha = result
    return alpha.dropna()
# SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(L OW,1)))),12)/(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(L OW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HI GH,1)),ABS(LOW-DELAY(LOW,1)))),12))-SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0: MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELA Y(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))
def alpha_050(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(13) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    delay_high = high_.shift()
    delay_low = low_.shift()
    cond1 = (high_ + low_ <= delay_high + delay_low)
    cond2 = (high_ + low_ >= delay_high + delay_low)
    part = np.maximum(abs(high_ - high_.shift()), abs(low_ - low_.shift()))
    part1 = part.copy()
    part1[cond1] = 0
    part1 = part1.iloc[-12:].sum()
    part2 = part.copy()
    part2[cond2] = 0
    part2 = part2.iloc[-12:].sum()
    alpha = (part1 - part2) / (part1 + part2)
    return alpha.dropna()
#  SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(L OW,1)))),12)/(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(L OW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HI GH,1)),ABS(LOW-DELAY(LOW,1)))),12))
def alpha_051(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(13) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    delay_high = high_.shift()
    delay_low = low_.shift()
    condition1 = (high_ + low_ >= delay_high + delay_low)
    condition2 = (high_ + low_ <= delay_high + delay_low)
    part = np.maximum(np.abs(high_ - delay_high), np.abs(low_ - delay_low))
    part1 = part.copy()
    part1[condition1] = 0
    part1 = part1.iloc[-12:].sum()
    part2 = part.copy()
    part2[condition2] = 0
    part2 = part2.iloc[-12:].sum()
    result = part2 / (part1 + part2)
    alpha = result
    return alpha.dropna()
# SUM(MAX(0,HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)),26)/SUM(MAX(0,DELAY((HIGH+LOW+CLOSE)/3,1)-L),26)* 100
def alpha_052(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(27) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    delay = ((high_ + low_ + close_) / 3).shift()
    part1 = (np.maximum(high_ - delay, 0)).iloc[-26:, :]

    part2 = (np.maximum(delay - low_, 0)).iloc[-26:, :]
    alpha = part1.sum() + part2.sum()
    cond1 = alpha == 0
    alpha[cond1] = np.nan
    return alpha.dropna()
# COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100
def alpha_053(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(13) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    delay = close_.shift()
    condition = close_ > delay
    result = close_.where(condition, np.nan).iloc[-12:, :]
    alpha = result.count() * 100 / 12
    return alpha.dropna()
# (-1 * RANK((STD(ABS(CLOSE - OPEN)) + (CLOSE - OPEN)) + CORR(CLOSE, OPEN,10)))
def alpha_054(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(10) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    part1 = (close_ - open_).abs()
    part1 = part1.iloc[-5:, :].std()
    part2 = (close_ - open_).iloc[-1, :]
    part3 = close_.iloc[-10:, :].corrwith(open_.iloc[-10:, :])
    result = (part1 + part2 + part3)
    alpha = result.rank(pct=True)
    return alpha.dropna()
# SUM(16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/((ABS(HIGH-DELAY(CL OSE,1))>ABS(LOW-DELAY(CLOSE,1))&ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOS E,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))   & ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLO SE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OP EN,1))/4)))*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1))),20)
def alpha_055(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(21) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    part1 = 16 * (close_.iloc[-20:] - close_.shift().iloc[-20:] + (
                close_.iloc[-20:] - open_.iloc[-20:]) / 2 + close_.shift().iloc[-20:] - open_.shift().iloc[-20:])
    h_dc = abs(high_.iloc[-20:] - close_.shift().iloc[-20:])
    l_dc = abs(low_.iloc[-20:] - close_.shift().iloc[-20:])
    h_dl = abs(high_.iloc[-20:] - low_.shift().iloc[-20:])
    dc_do = abs(close_.shift().iloc[-20:] - open_.shift().iloc[-20:])
    cond1 = (h_dc > l_dc) & (h_dc > h_dl)
    cond2 = (l_dc > h_dl) & (l_dc > h_dc)
    part2 = h_dl + dc_do / 4
    part2[cond2] = l_dc + h_dc / 2 + dc_do / 4
    part2[cond1] = h_dc + l_dc / 2 + dc_do / 4
    part3 = np.maximum(h_dc, l_dc)
    alpha = (part1 / part2 * part3).sum()
    return alpha.dropna()
# (RANK((OPEN-TSMIN(OPEN,12)))<RANK((RANK(CORR(SUM(((HIGH+LOW)/2),19),SUM(MEAN(VOLUME,40),19),13))^5)))
def alpha_056(code):
    open_, close_, high_, low_, volume_ = (code[k].iloc[-70:] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    part1 = (open_.iloc[-1] - open_.iloc[-12:].min()).rank(pct=True).sort_index()
    temp1 = rsum((high_ + low_) / 2, 19).iloc[-13:]
    temp2 = rsum(rmean(volume_.iloc[-70:], 40).iloc[-32:], 19).iloc[-13:]
    temp3 = rcorr(temp1, temp2, 13).iloc[-1].rank(pct=True)
    part2 = (temp3 ** 5).rank(pct=True).sort_index()
    cond = part1 < part2
    alpha = part1
    alpha[cond] = 1
    alpha[~cond] = -1
    return alpha.dropna()
# SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)
def alpha_057(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    rlow = rmin(low_, 9)
    rhigh = rmin(high_, 9)
    part1 = close_ - rlow
    part2 = rhigh - rlow
    # 1 / (1 + com) = 1 / 3
    result = ewmm(100 * part1 / part2, 2)
    alpha = result.iloc[-1, :]
    return alpha.dropna()
# COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100
def alpha_058(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(21) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    delay = close_.shift()
    condition = close_ > delay
    result = close_.where(condition, np.nan).iloc[-20:, :]
    alpha = result.count() * 100 / 20
    return alpha.dropna()
# SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,D ELAY(CLOSE,1)))),20)
def alpha_059(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(21) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    delay = close_.shift()
    condition1 = (close_ > delay).iloc[-20:]
    condition2 = (close_ < delay).iloc[-20:]
    part1 = np.minimum(low_.iloc[-20:].where(condition1, np.nan), delay.iloc[-20:].where(condition1, np.nan)).fillna(0)
    part2 = np.maximum(high_.iloc[-20:].where(condition2, np.nan), delay.iloc[-20:].where(condition2, np.nan)).fillna(0)
    result = close_.iloc[-20:] - part1 - part2
    alpha = result.sum()
    return alpha.dropna()
# SUM(((CLOSE-LOW)-(HIGH-CLOSE))./(HIGH-LOW).*VOLUME,20)
def alpha_060(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(20) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    part1 = (close_.iloc[-20:, :] - low_.iloc[-20:, :]) - (
                high_.iloc[-20:, :] - close_.iloc[-20:, :])
    part2 = high_.iloc[-20:, :] - low_.iloc[-20:, :]
    result = volume_.iloc[-20:, :] * part1 / part2
    alpha = result.sum()
    cond1 = alpha == 0
    alpha[cond1] = np.nan
    return alpha.dropna()
# (MAX(RANK(DECAYLINEAR(DELTA(VWAP,1),12)),RANK(DECAYLINEAR(RANK(CORR((LOW),MEAN(VOLUME,80),8)),17))) * -1)
def alpha_061(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    part_1 = func_decaylinear(avg_.diff(1).iloc[-12:], 12).iloc[-1]
    cond1 = part_1 == 0
    part_1[cond1] = np.nan
    part_1 = part_1.rank(pct=True)
    temp_1 = rcorr(low_.iloc[-103:], rmean(volume_.iloc[-103:], 80).iloc[-24:], 8).iloc[-17:].rank(axis=1, pct=True)
    part_2 = func_decaylinear(temp_1, 17).iloc[-1]
    cond2 = part_2 == 0
    part_2[cond2] = np.nan
    part_2 = part_2.rank(pct=True)
    alpha = np.maximum(part_1, part_2) * -1
    return alpha.dropna()
# (-1 * CORR(HIGH, RANK(VOLUME), 5))
def alpha_062(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(5) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    part_1 = volume_.iloc[-5:].rank(axis=1, pct=True)
    alpha = high_.iloc[-5:].corrwith(part_1) * -1
    return alpha.dropna()
# SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100
def alpha_063(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    part1 = np.maximum(close_ - close_.shift(), 0)
    # 1 / (1 + com) = 1 / 6
    part1 = ewmm(part1, 5)
    part2 = (close_ - close_.shift()).abs()
    part2 = ewmm(part2, 5)
    result = part1 * 100 / part2
    alpha = result.iloc[-1, :]
    return alpha.dropna()
'''# (MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP),RANK(VOLUME),4),4)),RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE), RANK(MEAN(VOLUME,60)), 4), 13), 14))) * -1)
def alpha_064(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    temp_1 = rcorr(avg_.iloc[-7:].rank(axis=1, pct=True), volume_.iloc[-7:].rank(axis=1, pct=True), 4).iloc[-4:]
    part_1 = func_decaylinear(temp_1, 4).iloc[-1]
    cond1 = part_1 == 0
    part_1[cond1] = np.nan
    part_1 = part_1.rank(pct=True)
    temp_2 = rcorr(close_.iloc[-29:].rank(axis=1, pct=True),
                   rmean(volume_.iloc[-88:], 60).iloc[-29:].rank(axis=1, pct=True),
                   4
                   ).iloc[-26:]
    temp_3 = rmax(temp_2, 13).iloc[-14:]
    part_2 = func_decaylinear(temp_3, 14).iloc[-1]
    cond2 = part_2 == 0
    part_2[cond2] = np.nan
    part_2 = part_2.rank(pct=True)
    alpha = np.maximum(part_1, part_2) * -1
    return alpha.dropna()'''
# MEAN(CLOSE,6)/CLOSE
def alpha_065(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(6) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    alpha = rmean(close_.iloc[-6:], 6) / close_.iloc[-6:]
    alpha = alpha.iloc[-1, :]
    return alpha.dropna()
# (CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100
def alpha_066(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(6) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    alpha = ((close_.iloc[-6:] / rmean(close_.iloc[-6:], 6)) - 1) * 100
    alpha = alpha.iloc[-1, :]
    return alpha.dropna()
# SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1)*100
def alpha_067(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    temp1 = close_ - close_.shift()
    part1 = np.maximum(temp1, 0)
    # 1 / (1 + com) = 1 / 24
    part1 = ewmm(part1, 23)
    temp2 = temp1.abs()
    part2 = ewmm(temp2, 23)
    result = part1 * 100 / part2
    alpha = result.iloc[-1, :]
    return alpha.dropna()
# SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,15,2)
def alpha_068(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    part1 = (high_ + low_ - high_.shift(1) - low_.shift(1)).iloc[1:] / 2
    part2 = ((high_ - low_) / volume_).iloc[1:]
    result = (part1 * part2) * 100
    # 1 / (1 + com) = 2 / 15
    result = ewmm(result, 6.5)
    alpha = result.iloc[-1, :]
    return alpha.dropna()
# (SUM(DTM,20)>SUM(DBM,20)？(SUM(DTM,20)-SUM(DBM,20))/SUM(DTM,20)：(SUM(DTM,20)=SUM(DBM,20)？0：(SUM(DTM,20)-SUM(DBM,20))/SUM(DBM,20)))
def alpha_069(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(21) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    dtm20 = (func_dtm(code).iloc[-20:]).sum()
    dbm20 = (func_dbm(code).iloc[-20:]).sum()
    condition1 = (dtm20 > dbm20)
    condition2 = (dtm20 < dbm20)

    alpha = 1 - dbm20 / dtm20
    alpha[(~condition1) & ~condition2] = 0
    alpha[condition2] = dtm20 / dbm20 - 1
    return alpha.dropna()
# STD(AMOUNT,6)
# (CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100
def alpha_071(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(24) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    data = close_.iloc[-1] / close_.iloc[-24:].mean() - 1
    alpha = data * 100
    return alpha.dropna()
# SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)
def alpha_072(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    rhigh = rmax(high_, 6)
    data1 = rhigh - close_
    data2 = rhigh - rmin(low_, 6)
    # 1 / (1 + com) = 1 / 15
    alpha = ewmm(data1 / data2 * 100, 14).iloc[-1]
    return alpha.dropna()
# ((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((CLOSE),VOLUME,10),16),4),5)-RANK(DECAYLINEAR(CORR(VWAP, MEAN(VOLUME,30),4),3)))*-1)
def alpha_073(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    temp_1 = rcorr(close_.iloc[-32:], volume_.iloc[-32:], 10).iloc[-23:]
    temp_2 = func_decaylinear(temp_1, 16).iloc[-8:]
    part_1 = func_decaylinear(temp_2, 4).iloc[-5:]
    cond1 = part_1 == 0
    part_1[cond1] = np.nan
    part_1 = part_1.rank(axis=0, pct=True).iloc[-1]
    temp_3 = rcorr(avg_.iloc[-6:], rmean(volume_.iloc[-35:], 30).iloc[-6:], 4).iloc[-3:]
    part_2 = func_decaylinear(temp_3, 3).iloc[-1]
    cond2 = part_2 == 0
    part_2[cond2] = np.nan
    part_2 = part_2.rank(pct=True)
    alpha = (part_1 - part_2) * -1
    return alpha.dropna()
'''# (RANK(CORR(SUM(((LOW * 0.35) + (VWAP * 0.65)), 20), SUM(MEAN(VOLUME,40), 20), 7)) + RANK(CORR(RANK(VWAP), RANK(VOLUME), 6)))
def alpha_074(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    temp_1 = rsum(low_.iloc[-26:] * 0.35 + avg_.iloc[-26:] * 0.65, 20).iloc[-7:]
    temp_2 = rsum(rmean(volume_.iloc[-65:], 40).iloc[-26:], 20).iloc[-7:]
    part_1 = temp_1.corrwith(temp_2).rank(pct=True)

    temp_3 = avg_.iloc[-6:].rank(axis=1, pct=True)
    temp_4 = volume_.iloc[-6:].rank(axis=1, pct=True)
    part_2 = temp_3.corrwith(temp_4).rank(pct=True)
    alpha = part_1 + part_2
    return alpha.dropna()'''
# COUNT(CLOSE>OPEN&BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)/COUNT(BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)
'''def alpha_075(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(50) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    bic_, bio_ = get_bench(code)
    condition1 = close_.iloc[-50:] > open_.iloc[-50:]
    condition2 = bic_.iloc[-50:] < bio_.iloc[-50:]
    part_1 = close_.iloc[-50:][condition1 & condition2].count()
    part_2 = condition2.where(condition2, np.nan).count()
    alpha = part_1 / part_2
    return alpha.dropna()'''
# STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)
def alpha_076(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(21) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    part = (abs(close_ / close_.shift(1) - 1) / volume_).iloc[-20:]
    alpha = part.std() / part.mean()
    return alpha.dropna()
# MIN(RANK(DECAYLINEAR(((((HIGH + LOW) / 2) + HIGH) - (VWAP + HIGH)), 20)), RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 3), 6)))
def alpha_077(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    temp_1 = (high_.iloc[-20:] + low_.iloc[-20:]) / 2 + high_.iloc[-20:] - avg_.iloc[-20:] - high_.iloc[-20:]
    part_1 = func_decaylinear(temp_1, 20).iloc[-1]
    cond1 = part_1 == 0
    part_1[cond1] = np.nan
    part_1 = part_1.rank(pct=True)
    temp_2 = rcorr((high_.iloc[-8:] + low_.iloc[-8:]) / 2,
                   rmean(volume_.iloc[-47:], 40).iloc[-8:],
                   3).iloc[-6:]
    part_2 = func_decaylinear(temp_2, 6).iloc[-1]
    cond2 = part_2 == 0
    part_2[cond2] = np.nan
    part_2 = part_2.rank(pct=True)
    alpha = np.minimum(part_1, part_2)
    return alpha.dropna()
# ((HIGH+LOW+CLOSE)/3-MA((HIGH+LOW+CLOSE)/3,12))/(0.015*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,12)),12))
def alpha_078(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(23) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    part_1 = ((high_.iloc[-1] + low_.iloc[-1] + close_.iloc[-1]) / 3 -
              (high_.iloc[-12:] + low_.iloc[-12:] + close_.iloc[-12:]).mean() / 3)
    temp_1 = abs(close_.iloc[-12:] -
                 rmean((high_.iloc[-23:] + low_.iloc[-23:] + close_.iloc[-23:]) / 3, 12).iloc[-12:])
    part_2 = rmean(temp_1, 12).iloc[-1] * 0.015
    alpha = part_1 / part_2
    return alpha.dropna()
# SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100
def alpha_079(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    # 1 / (1 + com) = 1 / 12
    part_1 = ewmm(np.maximum((close_ - close_.shift(1)), 0), 11)
    part_2 = ewmm(abs(close_ - close_.shift(1)), 11)
    alpha = (part_1 / part_2 * 100).iloc[-1, :]
    return alpha.dropna()
# (VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100
def alpha_080(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(6) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    part_1 = volume_.iloc[-1] - volume_.shift(5).iloc[-1]
    part_2 = volume_.shift(5).iloc[-1]
    alpha = part_1 / part_2 * 100
    return alpha.dropna()
# SMA(VOLUME,21,2)
def alpha_081(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    # 1 / (1 + com) = 2 / 21
    result = ewmm(volume_, 9.5)
    alpha = result.iloc[-1, :]
    return alpha.dropna()
# SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,20,1)
def alpha_082(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    rmaxhigh = rmax(high_, 6)
    part1 = rmaxhigh.iloc[6:] - close_.iloc[6:]
    part2 = rmaxhigh.iloc[6:] - rmin(low_, 6).iloc[6:]
    # 1 / (1 + com) = 1 / 20
    result = ewmm(100 * part1 / part2, 19)
    alpha = result.iloc[-1, :]
    return alpha.dropna()
'''# (-1 * RANK(COVIANCE(RANK(HIGH), RANK(VOLUME), 5)))
def alpha_083(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    part1 = high_.iloc[-5:, :].rank(axis=1, pct=True)
    part2 = volume_.iloc[-5:, :].rank(axis=1, pct=True)
    result = part1.corrwith(part2) * part1.std() * part2.std()
    alpha = -(result.rank(pct=True))
    return alpha.dropna()'''
# SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),20)
def alpha_084(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(21) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    condition1 = (close_ > close_.shift()).iloc[-20:]
    condition2 = (close_ < close_.shift()).iloc[-20:]
    result = volume_.iloc[-20:].copy()
    result[~condition1 & ~condition2] = 0
    result[condition2] = -volume_.iloc[-20:]
    alpha = result.sum()
    return alpha.dropna()
# (TSRANK((VOLUME / MEAN(VOLUME,20)), 20) * TSRANK((-1 * DELTA(CLOSE, 7)), 8))
def alpha_085(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(39) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    temp_1 = volume_.iloc[-20:] / rmean(volume_.iloc[-39:], 20).iloc[-20:]
    part_1 = temp_1.rank(axis=0, pct=True)
    temp_2 = close_.iloc[-15:].diff(7).iloc[-8:] * -1
    part_2 = temp_2.rank(axis=0, pct=True)
    alpha = part_1.iloc[-1, :] * part_2.iloc[-1, :]
    return alpha.dropna()
# ((0.25 < (((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10))) ? (-1 * 1) :(((((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10)) < 0) ? 1 : ((-1 * 1) * (CLOSE - DELAY(CLOSE, 1)))))
def alpha_086(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(21) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    delay10 = close_.shift(10).iloc[-1]
    delay20 = close_.shift(20).iloc[-1]

    temp = (delay20 - delay10) / 10 - (delay10 - close_.iloc[-1]) / 10
    condition1 = temp > 0.25
    condition2 = temp < 0
    alpha = -(close_.iloc[-1] - close_.shift().iloc[-1])
    alpha[condition1] = -1
    alpha[condition2] = 1
    return alpha.dropna()
# ((RANK(DECAYLINEAR(DELTA(VWAP, 4), 7)) + TSRANK(DECAYLINEAR(((((LOW * 0.9) + (LOW * 0.1)) - VWAP) / (OPEN - ((HIGH + LOW) / 2))), 11), 7)) * -1)
def alpha_087(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    part_1 = func_decaylinear(avg_.iloc[-11:].diff(4).iloc[-7:], 7).iloc[-1]
    cond1 = part_1 == 0
    part_1[cond1] = np.nan
    part_1 = part_1.rank(pct=True)
    temp_1 = low_ * 0.9 + low_ * 0.9 - avg_
    temp_2 = open_ - (high_ + low_) / 2
    part_2 = func_decaylinear((temp_1 / temp_2).iloc[-18:], 11).iloc[-7:]
    cond2 = part_2 == 0
    part_2[cond2] = np.nan
    part_2 = part_2.rank(axis=0, pct=True).iloc[-1]
    alpha = (part_1 + part_2) * -1
    return alpha.dropna()
# (CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100
def alpha_088(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(21) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    alpha = (close_.iloc[-1] / close_.shift(20).iloc[-1]) - 1
    return alpha.dropna()
# 2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))
def alpha_089(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    # 1 / (1 + com) = 2 / 13
    data1 = ewmm(close_, 5.5)
    # 1 / (1 + com) = 2 / 27
    data2 = ewmm(close_, 12.5)
    # 1 / (1 + com) = 2 / 10
    data3 = ewmm(data1 - data2, 4)
    alpha = ((data1 - data2 - data3) * 2).iloc[-1, :]
    return alpha.dropna()
'''# ( RANK(CORR(RANK(VWAP), RANK(VOLUME), 5)) * -1)
def alpha_090(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    data1 = avg_.iloc[-5:].rank(axis=1, pct=True)
    data2 = volume_.iloc[-5:].rank(axis=1, pct=True)
    corr = data1.corrwith(data2)
    rank1 = corr.rank(pct=True)
    alpha = rank1 * -1
    return alpha.dropna()'''
# ((RANK((CLOSE - MAX(CLOSE, 5)))*RANK(CORR((MEAN(VOLUME,40)), LOW, 5))) * -1)
def alpha_091(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(44) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    data1 = close_.iloc[-1] - close_.iloc[-1].clip(upper=5.)
    rank1 = data1.rank(pct=True)
    mean = rmean(volume_.iloc[-44:], 40)
    corr = mean.iloc[-5:, :].corrwith(low_.iloc[-5:, :])
    rank2 = corr.rank(pct=True)
    alpha = rank1 * rank2 * (-1)
    return alpha.dropna()
# (MAX(RANK(DECAYLINEAR(DELTA(((CLOSE*0.35) + (VWAP*0.65)), 2), 3)), TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)), CLOSE, 13)), 5), 15)) * -1)
def alpha_092(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    temp_1 = close_.iloc[-5:] * 0.35 + avg_.iloc[-5:] * 0.65
    part_1 = func_decaylinear(temp_1.diff(2).iloc[-3:], 3).iloc[-1]
    cond1 = part_1 == 0
    part_1[cond1] = np.nan
    part_1 = part_1.rank(pct=True)
    temp_2 = abs(rcorr(rmean(volume_.iloc[-240:], 180).iloc[-31:], close_.iloc[-31:], 13).iloc[-19:])
    part_2 = func_decaylinear(temp_2, 5).iloc[-15:]
    cond2 = part_2 == 0
    part_2[cond2] = np.nan
    part_2 = part_2.rank(axis=0, pct=True).iloc[-1]
    alpha = np.maximum(part_1, part_2) * -1
    return alpha.dropna()
# SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20)
def alpha_093(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(21) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    cond = (open_ >= open_.shift()).iloc[-20:]
    data1 = open_.iloc[-20:] - low_.iloc[-20:]
    data2 = open_.iloc[-20:] - open_.shift().iloc[-20:]
    data = np.maximum(data1, data2)
    data[cond] = 0
    alpha = data.sum()
    return alpha.dropna()
# SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30)
def alpha_094(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(31) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    cond1 = (close_ > close_.shift()).iloc[-30:]
    cond2 = (close_ < close_.shift()).iloc[-30:]
    value = -volume_.copy().iloc[-30:]
    value[~cond2] = 0
    value[cond1] = volume_.iloc[-30:][cond1]
    alpha = value.sum()
    return alpha.dropna()
# STD(AMOUNT,20)
# SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)
def alpha_096(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    rmax_high = rmin(high_, 9)
    rmin_low = rmin(low_, 9)
    # 1 / (1 + com) = 1 / 3
    temp_1 = ewmm(100 * (close_.iloc[9:] - rmin_low.iloc[9:]) / (rmax_high.iloc[9:] - rmin_low.iloc[9:]), 2)
    alpha = ewmm(temp_1, 2).iloc[-1, :]
    return alpha.dropna()
# STD(VOLUME,10)
def alpha_097(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(10) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    alpha = volume_.iloc[-10:, :].std(axis=0)
    return alpha.dropna()
# ((((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) < 0.05) || ((DELTA((SUM(CLOSE, 100) / 100), 100) /DELAY(CLOSE, 100)) == 0.05)) ? (-1 * (CLOSE - TSMIN(CLOSE, 100))) : (-1 * DELTA(CLOSE, 3)))
def alpha_098(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    sum_close = rsum(close_.iloc[-200:], 100).iloc[-101:] / 100
    cond = (sum_close.iloc[-1] - sum_close.shift(100).iloc[-1]) / close_.shift(100).iloc[-1] <= 0.05
    left_value = -(close_.iloc[-1] - close_.iloc[-100:].min())
    right_value = -(close_.iloc[-1] - close_.shift(3).iloc[-1])
    alpha = left_value.where(cond, right_value)
    return alpha.dropna()
# (-1 * RANK(COVIANCE(RANK(CLOSE), RANK(VOLUME), 5)))
def alpha_099(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(5) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    alpha = -roll_cov(close_.iloc[-5:].rank(axis=1, pct=True),
                      volume_.iloc[-5:].rank(axis=1, pct=True),
                      5).iloc[-1, :].rank(pct=True)
    return alpha.dropna()
# STD(VOLUME,20)
def alpha_100(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(20) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    alpha = volume_.iloc[-20:, :].std(axis=0)
    return alpha.dropna()
'''# ((RANK(CORR(CLOSE, SUM(MEAN(VOLUME,30), 37), 15)) < RANK(CORR(RANK(((HIGH * 0.1) + (VWAP * 0.9))),RANK(VOLUME), 11))) * -1)
def alpha_101(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(80) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    temp_1 = rsum(rmean(volume_.iloc[-80:], 30).iloc[-51:], 37).iloc[-15:]
    part_1 = rcorr(close_, temp_1, 15).iloc[-1].rank(pct=True)

    temp_2 = (high_.iloc[-11:] * 0.1 + avg_.iloc[-11:] * 0.9).rank(axis=1, pct=True)
    temp_3 = volume_.iloc[-11:].rank(axis=1, pct=True)
    part_2 = temp_2.corrwith(temp_3)
    alpha = part_1 - part_2
    cond1 = alpha < 0
    cond2 = alpha >= 0
    alpha[cond1] = -1
    alpha[cond2] = 1
    return alpha.dropna()'''
# SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100
def alpha_102(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    max_data = (volume_ - volume_.shift()).clip(0.).iloc[1:]
    # 1 / (1 + com) = 1 / 6
    sma1 = ewmm(max_data, 5)
    sma2 = ewmm((volume_ - volume_.shift()).abs().iloc[1:], 5)
    alpha = sma1.iloc[-1, :] / sma2.iloc[-1, :] * 100
    return alpha.dropna()
# ((20-LOWDAY(LOW,20))/20)*100
def alpha_103(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(20) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    alpha = (20 - func_lowday(low_, 20)) / 20 * 100
    return alpha.dropna()
# (-1 * (DELTA(CORR(HIGH, VOLUME, 5), 5) * RANK(STD(CLOSE, 20))))
def alpha_104(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(20) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    part_1 = rcorr(high_.iloc[-10:], volume_.iloc[-10:], 5).iloc[-6:].diff(5).iloc[-1]
    part_2 = close_.iloc[-20:].std().rank(pct=True)
    alpha = -part_1 * part_2
    return alpha.dropna()
# (-1 * CORR(RANK(OPEN), RANK(VOLUME), 10))
def alpha_105(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(10) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    alpha = open_.iloc[-10:].rank(axis=1, pct=True).corrwith(
        volume_.iloc[-10:].rank(axis=1, pct=True)
    )
    return alpha.dropna()
# CLOSE-DELAY(CLOSE,20)
def alpha_106(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(21) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    alpha = close_.iloc[-1] - close_.shift(20).iloc[-1]
    return alpha.dropna()
# (((-1 * RANK((OPEN - DELAY(HIGH, 1)))) * RANK((OPEN - DELAY(CLOSE, 1)))) * RANK((OPEN - DELAY(LOW, 1))))
def alpha_107(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(2) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    rank1 = (open_.iloc[-1] - high_.shift().iloc[-1]).rank(pct=True)
    rank2 = (open_.iloc[-1] - close_.shift().iloc[-1]).rank(pct=True)
    rank3 = (open_.iloc[-1] - low_.shift().iloc[-1]).rank(pct=True)
    alpha = -rank1 * rank2 * rank3
    return alpha.dropna()
# ((RANK((HIGH - MIN(HIGH, 2)))^RANK(CORR((VWAP), (MEAN(VOLUME,120)), 6))) * -1)
def alpha_108(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    part_1 = (high_.iloc[-1] - high_.iloc[-2:].min()).rank(pct=True)
    part_2 = avg_.iloc[-6:].corrwith(
        rmean(volume_.iloc[-125:], 120).iloc[-6:]
    ).rank(pct=True)
    alpha = (part_1 ** part_2) * -1
    return alpha.dropna()
# SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)
def alpha_109(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    data = high_ - low_
    # 1 / (1 + com) = 2 / 10
    sma1 = ewmm(data, 4)
    sma2 = ewmm(sma1, 4)
    alpha = sma1.iloc[-1] / sma2.iloc[-1]
    return alpha.dropna()
# SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100
'''def alpha_110(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(21) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    data1 = high_.iloc[-20:] - close_.shift().iloc[-20:]
    data2 = close_.shift().iloc[-20:] - low_.iloc[-20:]
    alpha = (data1.clip(lower=0.)).sum() / (data2.clip(lower=0.)).sum() * 100
    return alpha.dropna()'''
# SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)-SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),4,2)
def alpha_111(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    data1 = volume_ * ((close_ - low_) - (high_ - close_)) / (
                high_ - low_)
    # 1 / (1 + com) = 2 / 11
    x = ewmm(data1, 4.5)
    # 1 / (1 + com) = 2 / 4
    y = ewmm(data1, 1)
    alpha = x.iloc[-1, :] - y.iloc[-1, :]
    return alpha.dropna()
# ((SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)-SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))/(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)+SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))*100
def alpha_112(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(13) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    data = close_.iloc[-12:] - close_.shift().iloc[-12:]
    cond1 = data > 0
    cond2 = data < 0
    data1 = data.copy()
    data1[~cond1] = 0
    data2 = data.copy().abs()
    data2[~cond2] = 0
    sum1 = data1.sum()
    sum2 = data2.sum()
    alpha = (sum1 - sum2) / (sum1 + sum2) * 100
    return alpha.dropna()
# (-1 * ((RANK((SUM(DELAY(CLOSE, 5), 20) / 20)) * CORR(CLOSE, VOLUME, 2)) * RANK(CORR(SUM(CLOSE, 5),SUM(CLOSE, 20), 2))))
def alpha_113(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(25) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    temp_1 = (close_.shift(5).iloc[-20:]).sum() / 20
    temp_2 = close_.iloc[-2:].corrwith(volume_.iloc[-2:])
    part_1 = (temp_1 * temp_2).rank(pct=True)
    part_2 = rsum(close_.iloc[-6:], 5).iloc[-2:].corrwith(
        rsum(close_.iloc[-21:], 20).iloc[-2:]
    ).rank(pct=True)
    alpha = -part_1 * part_2
    return alpha.dropna()
# ((RANK(DELAY(((HIGH - LOW) / (SUM(CLOSE, 5) / 5)), 2)) * RANK(RANK(VOLUME))) / (((HIGH - LOW) / (SUM(CLOSE, 5) / 5)) / (VWAP - CLOSE)))
def alpha_114(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    temp_1 = (high_.shift(2).iloc[-1] - low_.shift(2).iloc[-1]) / ((close_.shift(2).iloc[-5:]).sum() / 5)
    part_1 = temp_1.rank(pct=True)
    part_2 = volume_.iloc[-1].rank(pct=True).rank(pct=True)
    part_3 = (high_.iloc[-1] - low_.iloc[-1]) / ((close_.iloc[-5:]).sum() / 5) / (avg_.iloc[-1] - close_.iloc[-1])
    alpha = (part_1 * part_2) / part_3
    return alpha.dropna()
'''# ((RANK(CORR(((HIGH * 0.9) + (CLOSE * 0.1)), MEAN(VOLUME,30), 10))^RANK(CORR(TSRANK(((HIGH + LOW) /  2), 4), TSRANK(VOLUME, 10), 7)))
def alpha_115(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(39) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    part_1 = (high_.iloc[-10:] * 0.9 + close_.iloc[-10:] * 0.1).corrwith(
        rmean(volume_.iloc[-39:], 30).iloc[-10:]
    ).rank(pct=True)
    temp_1 = func_tsrank((high_.iloc[-10:] + low_.iloc[-10:]) / 2, 4).iloc[-7:]
    temp_2 = func_tsrank(volume_.iloc[-16:], 10).iloc[-7:]
    part_2 = temp_1.corrwith(temp_2).rank(pct=True)
    alpha = part_1 ** part_2
    return alpha.dropna()'''
# REGBETA(CLOSE,SEQUENCE,20)
def alpha_116(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(20) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    sequence = pd.Series(range(1, 21), index=close_.iloc[-20:, ].index)  # 1~20
    corr = close_.iloc[-20:, :].corrwith(sequence)
    alpha = corr * close_.iloc[-20:, :].std() / sequence.std()
    return alpha.dropna()
# ((TSRANK(VOLUME, 32) * (1 - TSRANK(((CLOSE + HIGH) - LOW), 16))) * (1 - TSRANK(RET, 32)))
'''def alpha_117(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(33) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    part_1 = volume_.iloc[-32:].rank(axis=0, pct=True).iloc[-1]
    part_2 = 1 - (close_.iloc[-16:] + high_.iloc[-16:] - low_.iloc[-16:]).rank(axis=0, pct=True).iloc[-1]
    part_3 = 1 - func_ret(code).iloc[-32:].rank(axis=0, pct=True).iloc[-1]
    alpha = part_1 * part_2 * part_3
    return alpha.dropna()'''
# SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100
def alpha_118(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(20) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    part_1 = (high_.iloc[-20:] - open_.iloc[-20:]).sum()
    part_2 = (open_.iloc[-20:] - low_.iloc[-20:]).sum()
    alpha = part_1 / part_2 * 100
    return alpha.dropna()
# ((RANK(DECAYLINEAR(CORR(VWAP, SUM(MEAN(VOLUME,5), 26), 5), 7))-RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN), RANK(MEAN(VOLUME,15)), 21), 9), 7), 8)))
def alpha_119(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    temp_1 = rcorr(avg_.iloc[-11:], rsum(rmean(volume_.iloc[-40:], 5).iloc[-36:], 26).iloc[-11:], 5).iloc[-7:]
    part_1 = func_decaylinear(temp_1, 7).iloc[-1]
    cond1 = part_1 == 0
    part_1[cond1] = np.nan
    part_1 = part_1.rank(pct=True)
    temp_2 = rcorr(open_.iloc[-42:].rank(axis=1, pct=True), rmean(volume_.iloc[-56:], 15).iloc[-42:].rank(axis=1, pct=True), 21).iloc[-22:]
    temp_3 = func_tsrank(rmin(temp_2, 9).iloc[-14:], 7).iloc[-8:]
    part_2 = func_decaylinear(temp_3, 8).iloc[-1]
    cond2 = part_2 == 0
    part_2[cond2] = np.nan
    part_2 = part_2.rank(pct=True)
    alpha = part_1 - part_2
    return alpha.dropna()
# (RANK((VWAP - CLOSE)) / RANK((VWAP + CLOSE)))
def alpha_120(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    data1 = (avg_.iloc[-1] - close_.iloc[-1]).rank(pct=True)
    data2 = (avg_.iloc[-1] + close_.iloc[-1]).rank(pct=True)
    alpha = data1 / data2
    return alpha.dropna()
'''# ((RANK((VWAP - MIN(VWAP, 12)))^TSRANK(CORR(TSRANK(VWAP, 20), TSRANK(MEAN(VOLUME,60), 2), 18), 3))*-1)
def alpha_121(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    part_1 = (avg_.iloc[-1] - avg_.iloc[-12:].min()).rank(pct=True)
    temp_1 = func_tsrank(avg_.iloc[-39:], 20).iloc[-20:]
    temp_2 = func_tsrank(rmean(volume_.iloc[-80:], 60).iloc[-21:], 2).iloc[-20:]
    part_2 = rcorr(temp_1, temp_2, 18).iloc[-3:].rank(axis=0, pct=True).iloc[-1]
    alpha = part_1 ** part_2 * -1
    return alpha.dropna()'''
# (SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)-DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1))/DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)
def alpha_122(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    log_close = np.log(close_)
    # 1 / (1 + com) = 2 / 13
    data = ewmm(ewmm(ewmm(np.log(close_), 5.5), 5.5), 5.5)
    alpha = (data.iloc[-1, :] / data.iloc[-2, :]) - 1
    return alpha.dropna()
# ((RANK(CORR(SUM(((HIGH + LOW) / 2), 20), SUM(MEAN(VOLUME,60), 20), 9)) < RANK(CORR(LOW, VOLUME,6))) * -1)
def alpha_123(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(87) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    temp_1 = rsum((high_.iloc[-28:] + low_.iloc[-28:]) / 2, 20).iloc[-9:]
    temp_2 = rsum(rmean(volume_.iloc[-87:], 60).iloc[-28:], 20).iloc[-9:]
    part_1 = temp_1.corrwith(temp_2).rank(pct=True)
    part_2 = low_.iloc[-6:].corrwith(volume_.iloc[-6:]).rank(pct=True)
    alpha = part_1 - part_2
    cond = alpha < 0
    alpha[cond] = -1
    alpha[~cond] = 1
    return alpha.dropna()
# (CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE, 30)),2)
def alpha_124(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    part_1 = close_.iloc[-1] - avg_.iloc[-1]
    temp_1 = rmax(close_.iloc[-31:], 30).iloc[-2:].rank(axis=1, pct=True)
    part_2 = func_decaylinear(temp_1, 2).iloc[-1]
    alpha = part_1 / part_2
    return alpha.dropna()
# (RANK(DECAYLINEAR(CORR((VWAP), MEAN(VOLUME,80),17), 20)) / RANK(DECAYLINEAR(DELTA(((CLOSE * 0.5)+ (VWAP * 0.5)), 3), 16)))
def alpha_125(code, end_date=None, fq='pre'):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    temp_1 = rcorr(avg_.iloc[-36:], rmean(volume_.iloc[-115:], 80).iloc[-36:], 17).iloc[-20:]
    part_1 = func_decaylinear(temp_1, 20).iloc[-1]
    cond1 = part_1 == 0
    part_1[cond1] = np.nan
    part_1 = part_1.rank(pct=True)
    temp_2 = close_.shift(3).iloc[-16:] * 0.5 + avg_.shift(3).iloc[-16:] * 0.5
    part_2 = func_decaylinear(temp_2, 16).iloc[-1]
    cond2 = part_2 == 0
    part_2[cond2] = np.nan
    part_2 = part_2.rank(pct=True)
    alpha = part_1 / part_2
    return alpha.dropna()
# ((CLOSE+HIGH+LOW)/3)
def alpha_126(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(1) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    alpha = close_.iloc[-1] + high_.iloc[-1] + low_.iloc[-1]
    alpha = alpha / 3
    return alpha.dropna()
# (MEAN((100*(CLOSE-MAX(CLOSE,12))/(MAX(CLOSE,12)))^2))^(1/2)
'''def alpha_127(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(12) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    temp_1 = close_.iloc[-1]
    temp_2 = close_.iloc[-12:].max()
    alpha = 100 * (((temp_1 / temp_2) - 1) ** 2) ** 0.5
    return alpha.dropna()'''
# 100-(100/(1+SUM(((HIGH+LOW+CLOSE)/3>DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0),14)/SUM(((HIGH+LOW+CLOSE)/3<DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0), 14)))
def alpha_128(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(15) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    temp = (high_.iloc[-15:] + low_.iloc[-15:] + close_.iloc[-15:]) / 3
    cond1 = temp.iloc[-14:] > temp.shift(1).iloc[-14:]
    cond2 = temp.iloc[-14:] < temp.shift(1).iloc[-14:]
    part = temp.iloc[-14:] * volume_.iloc[-14:]
    part1 = part.copy()
    part1[~cond1] = 0
    part2 = part.copy()
    part2[~cond2] = 0
    part1 = part1.sum()
    part2 = part2.sum()
    alpha = 100 - (100 / (1 + part1 / part2))
    return alpha.dropna()
# SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12)
def alpha_129(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(13) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    data = close_.iloc[-13:].diff(1).iloc[-12:].clip(upper=0).abs()
    alpha = data.sum()
    return alpha.dropna()
''' (RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 9), 10))/RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 7),3)))# 
def alpha_130(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    temp_1 = rcorr((high_.iloc[-18:] + low_.iloc[-18]) / 2, rmean(volume_.iloc[-57:], 40).iloc[-18:], 9).iloc[-10:]
    part_1 = func_decaylinear(temp_1, 10).iloc[-1]
    cond1 = part_1 == 0
    part_1[cond1] = np.nan
    part_1 = part_1.rank(pct=True)
    temp_2 = rcorr(avg_.iloc[-9:].rank(axis=1, pct=True), volume_.iloc[-9:].rank(axis=1, pct=True), 7).iloc[-3:]
    part_2 = func_decaylinear(temp_2, 3).iloc[-1]
    cond2 = part_2 == 0
    part_2[cond2] = np.nan
    part_2 = part_2.rank(pct=True)
    alpha = part_1 / part_2
    return alpha.dropna()'''
# (RANK(DELTA(VWAP, 1))^TSRANK(CORR(CLOSE,MEAN(VOLUME,50), 18), 18))
def alpha_131(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    part_1 = avg_.iloc[-2:].diff(1).iloc[-1].rank(pct=True)
    part_2 = rcorr(close_.iloc[-35:], rmean(volume_.iloc[-84:], 50).iloc[-35:], 18).iloc[-18:].rank(axis=0, pct=True).iloc[-1]
    alpha = part_1 ** part_2
    return alpha.dropna()
# MEAN(AMOUNT,20)
# ((20-HIGHDAY(HIGH,20))/20)*100-((20-LOWDAY(LOW,20))/20)*100
def alpha_133(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(20) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    part_1 = ((20 - func_highday(high_, 20)) / 20) * 100
    part_2 = ((20 - func_lowday(low_, 20)) / 20) * 100
    alpha = part_1 - part_2
    return alpha.dropna()
# (CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME
def alpha_134(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(13) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    alpha = (close_.iloc[-1] - close_.shift(12).iloc[-1]) / close_.shift(12).iloc[-1] * volume_.iloc[-1]
    return alpha.dropna()
# SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1)
def alpha_135(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    temp_1 = (close_ / close_.shift(20)).shift(1).iloc[21:]
    # 1 / (1 + com) = 1 / 20
    alpha = ewmm(temp_1, 19)
    alpha = alpha.iloc[-1, :]
    return alpha.dropna()
# ((-1 * RANK(DELTA(RET, 3))) * CORR(OPEN, VOLUME, 10))
def alpha_136(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(10) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    part_1 = -1 * (func_ret(code).iloc[-4:].diff(3).iloc[-1].rank(pct=True))
    part_2 = open_.iloc[-10:].corrwith(volume_.iloc[-10:])
    alpha = part_1 * part_2
    return alpha.dropna()
# 16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/((ABS(HIGH-DELAY(CLOSE,1))>ABS(LOW-DELAY(CLOSE,1)) &ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOS E,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))    & ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLO SE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OP EN,1))/4)))*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1)))
def alpha_137(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(2) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    part1 = (close_.iloc[-1] - close_.shift(1).iloc[-1] + (
                close_.iloc[-1] - open_.iloc[-1]) / 2 + close_.shift(1).iloc[-1] - open_.shift(1).iloc[-1]) * 16
    h_dc = abs(high_.iloc[-1] - close_.shift().iloc[-1])
    l_dc = abs(low_.iloc[-1] - close_.shift().iloc[-1])
    h_dl = abs(high_.iloc[-1] - low_.shift().iloc[-1])
    dc_do = abs(close_.shift().iloc[-1] - open_.shift().iloc[-1])
    cond1 = (h_dc > l_dc) & (h_dc > h_dl)
    cond2 = (l_dc > h_dl) & (l_dc > h_dc)
    part2 = h_dl + dc_do / 4
    part2[cond2] = l_dc + h_dc / 2 + dc_do / 4
    part2[cond1] = h_dc + l_dc / 2 + dc_do / 4
    part3 = np.maximum(h_dc, l_dc)
    alpha = part1 / part2 * part3
    return alpha.dropna()
# ((RANK(DECAYLINEAR(DELTA((((LOW*0.7)+(VWAP*0.3))),3),20))-TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW,8),TSRANK(MEAN(VOLUME,60),17),5),19),16),7))*-1)
def alpha_138(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    temp_1 = (low_.iloc[-23:] * 0.7 + avg_.iloc[-23:] * 0.3).diff(3).iloc[-20:]
    part_1 = func_decaylinear(temp_1, 20).iloc[-1].rank(pct=True)
    temp_2 = rcorr(func_tsrank(low_.iloc[-51:], 8).iloc[-44:],
                   func_tsrank(rmean(volume_.iloc[-119:], 60).iloc[-60:], 17).iloc[-44:],
                   5).iloc[-40:].fillna(0.0)
    part_2 = func_decaylinear(func_tsrank(temp_2, 19).iloc[-22:], 16).iloc[-7:].rank(axis=0, pct=True).iloc[-1]
    alpha = (part_1 - part_2) * -1
    return alpha.dropna()
# (-1 * CORR(OPEN, VOLUME, 10))
def alpha_139(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(10) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    alpha = -1 * open_.iloc[-10:].corrwith(volume_.iloc[-10:])
    return alpha.dropna()
'''# MIN(RANK(DECAYLINEAR(((RANK(OPEN)+RANK(LOW))-(RANK(HIGH)+RANK(CLOSE))),8)),TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE,8),TSRANK(MEAN(VOLUME,60),20), 8), 7), 3))
def alpha_140(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(94) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    temp_1 = ((open_.iloc[-8:].rank(axis=1, pct=True) + low_.iloc[-8:].rank(axis=1, pct=True)) -
              (high_.iloc[-8:].rank(axis=1, pct=True) + close_.iloc[-8:].rank(axis=1, pct=True)))
    part_1 = func_decaylinear(temp_1, 8).iloc[-1]
    cond1 = part_1 == 0
    part_1[cond1] = np.nan
    part_1 = part_1.rank(pct=True)
    temp_2 = rcorr(func_tsrank(close_.iloc[-23:], 8).iloc[-16:],
                   func_tsrank(rmean(volume_.iloc[-94:], 60).iloc[-35:], 20).iloc[-16:],
                   8).iloc[-9:]
    temp_2 = func_decaylinear(temp_2, 7).iloc[-3:]
    cond2 = temp_2 == 0
    temp_2[cond2] = np.nan
    part_2 = temp_2.rank(axis=0, pct=True).iloc[-1]
    alpha = np.minimum(part_1, part_2)
    return alpha.dropna()'''
# (RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME,15)), 9))* -1)
def alpha_141(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(23) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    alpha = -high_.iloc[-9:].rank(axis=1, pct=True).corrwith(
                rmean(volume_.iloc[-23:], 15).iloc[-9:].rank(axis=1, pct=True)
            ).rank(pct=True)
    return alpha.dropna()
# (((-1 * RANK(TSRANK(CLOSE, 10))) * RANK(DELTA(DELTA(CLOSE, 1), 1))) * RANK(TSRANK((VOLUME/MEAN(VOLUME,20)), 5)))
def alpha_142(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(24) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    part_1 = -close_.iloc[-10:].rank(axis=0, pct=True).iloc[-1].rank(pct=True)
    part_2 = close_.iloc[-3:].diff().diff().iloc[-1].rank(pct=True)
    part_3 = (volume_.iloc[-5:] / rmean(volume_.iloc[-24:], 20).iloc[-5:]).rank(axis=0, pct=True).iloc[-1].rank(pct=True)
    alpha = part_1 * part_2 * part_3
    return alpha.dropna()
# CLOSE>DELAY(CLOSE,1)?(CLOSE/DELAY(CLOSE,1))*SELF:SELF
def alpha_143(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    data = (close_ / close_.shift()).iloc[1:]
    data[data <= 1] = 1
    alpha = np.prod(data)
    return alpha.dropna()
#  SUMIF(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT,20,CLOSE<DELAY(CLOSE,1))/COUNT(CLOSE<DELAY(CLOSE, 1),20)
# (MEAN(VOLUME,9)-MEAN(VOLUME,26))/MEAN(VOLUME,12)*100
def alpha_145(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(26) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    alpha = (volume_.iloc[-9:].mean() - volume_.iloc[-26:].mean()) / volume_.iloc[-12:].mean() * 100
    return alpha.dropna()
#  MEAN((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2),20)*((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2))/SMA(((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE, 1))/DELAY(CLOSE,1),61,2)))^2,60)
def alpha_146(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    temp_1 = ((close_ - close_.shift()) / close_.shift()).iloc[1:]
    # 1 / (1 + com) = 2 / 61
    temp_2 = temp_1 - ewmm(temp_1, 29.5)

    part_1 = temp_1.iloc[-20:].mean() - temp_2.iloc[-20:].mean()
    part_2 = temp_1.iloc[-1] - temp_2.iloc[-1]
    # 1 / (1 + com) = 2 / 60
    part_3 = ewmm(temp_1 ** 2, 29).iloc[-1]
    alpha = part_1 * part_2 / part_3
    return alpha.dropna()
# REGBETA(MEAN(CLOSE,12),SEQUENCE(12))
def alpha_147(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(23) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    temp_1 = rmean(close_.iloc[-23:], 12).iloc[-12:]
    sequence = pd.Series(range(1, 13), index=temp_1.index)
    corr = temp_1.corrwith(sequence)
    alpha = corr * temp_1.std() / sequence.std()
    return alpha.dropna()
# ((RANK(CORR((OPEN), SUM(MEAN(VOLUME,60), 9), 6)) < RANK((OPEN - TSMIN(OPEN, 14)))) * -1)
def alpha_148(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(73) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    part_1 = open_.iloc[-6:].corrwith(
        rsum(rmean(volume_.iloc[-73:], 60).iloc[-14:], 9).iloc[-6:]
    ).rank(pct=True)
    part_2 = (open_.iloc[-1] - open_.iloc[-14:].min()).rank(pct=True)
    alpha = part_1 - part_2
    cond = alpha < 0
    alpha[cond] = -1
    alpha[~cond] = 1
    return alpha.dropna()
# REGBETA(FILTER(CLOSE/DELAY(CLOSE,1)-1,BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1)),FILTER(BANCHMARKINDEXCLOSE/DELAY(BANCHMARKINDEXCLOSE,1)-1,BANCHMARKINDEXCLOSE<DELA Y(BANCHMARKINDEXCLOSE,1)),252)
def alpha_149(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])    
    bic_, bio_ = get_bench(code)
    temp1 = close_.iloc[-252:] / close_.shift().iloc[-252:] - 1
    temp2 = bic_.iloc[-252:, 0] / bic_.shift().iloc[-252:, 0] - 1
    cond = (temp2 < 0)
    part1 = temp1.loc[cond[cond].index]
    part2 = temp2.loc[cond[cond].index]
    result = part1.corrwith(part2)
    alpha = result * part1.std() / part2.std()
    cond1 = alpha == 0
    alpha[cond1] = np.nan
    return alpha.dropna()
# (CLOSE+HIGH+LOW)/3*VOLUME
def alpha_150(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(1) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    alpha = (high_.iloc[-1] + close_.iloc[-1] + low_.iloc[-1]) / 3 * volume_.iloc[-1]
    return alpha.dropna()
# SMA(CLOSE-DELAY(CLOSE,20),20,1)
def alpha_151(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    # 1 / (1 + com) = 1 / 20
    alpha = ewmm((close_ - close_.shift(20)).iloc[20:], 19)
    alpha = alpha.iloc[-1, :]
    return alpha.dropna()
# SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)-MEAN(DELAY(SMA(DELAY(CLOSE/DELAY (CLOSE,9),1),9,1),1),26),9,1)
def alpha_152(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    # 1 / (1 + com) = 1 / 9
    temp_1 = ewmm((close_ / close_.shift(9)).shift().iloc[10:], 8).shift().iloc[1:]
    part_1 = rmean(temp_1, 12)
    part_2 = rmean(temp_1, 26)
    alpha = ewmm(part_1 - part_2, 8)
    alpha = alpha.iloc[-1, :]
    return alpha.dropna()
# (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4
def alpha_153(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(24) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    part_1 = close_.iloc[-3:].mean()
    part_2 = close_.iloc[-6:].mean()
    part_3 = close_.iloc[-12:].mean()
    part_4 = close_.iloc[-24:].mean()
    alpha = (part_1 + part_2 + part_3 + part_4) / 4
    return alpha.dropna()
# (((VWAP - MIN(VWAP, 16))) < (CORR(VWAP, MEAN(VOLUME,180), 18)))
def alpha_154(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    part_1 = avg_.iloc[-1] - avg_.iloc[-16:].min()
    part_2 = avg_.iloc[-18:].corrwith(rmean(volume_.iloc[-197:], 180).iloc[-18:])
    alpha = part_1 - part_2
    cond = alpha < 0
    alpha[cond] = 1
    alpha[~cond] = -1
    return alpha.dropna()
# SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2)
def alpha_155(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    # 1 / (1 + com) = 2 / 13
    part_1 = ewmm(volume_, 5.5)
    # 1 / (1 + com) = 2 / 27
    part_2 = ewmm(volume_, 12.5)
    part_3 = part_1 - part_2
    # 1 / (1 + com) = 2 / 10
    part_4 = ewmm(part_3, 4)
    alpha = part_3.iloc[-1] - part_4.iloc[-1]
    return alpha.dropna()
# (MAX(RANK(DECAYLINEAR(DELTA(VWAP, 5), 3)), RANK(DECAYLINEAR(((DELTA(((OPEN * 0.15) + (LOW*0.85)),2) / ((OPEN * 0.15) + (LOW * 0.85))) * -1), 3))) * -1)
def alpha_156(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    part_1 = func_decaylinear(avg_.iloc[-8:].diff(5).iloc[-3:], 3).iloc[-1]
    cond1 = part_1 == 0
    part_1[cond1] = np.nan
    part_1 = part_1.rank(pct=True)
    temp_1 = (open_.iloc[-5:] * 0.15 + low_.iloc[-5:] * 0.85).diff(2).iloc[-3:] / (open_.iloc[-3:] * 0.15 + low_.iloc[-3:] * 0.85)
    part_2 = func_decaylinear(temp_1, 3).iloc[-1]
    cond2 = part_2 == 0
    part_2[cond2] = np.nan
    part_2 = part_2.rank(pct=True)
    alpha = np.maximum(part_1, part_2) * -1
    return alpha.dropna()
# (MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1 * RANK(DELTA((CLOSE - 1), 5))))), 2), 1)))), 1), 5) +TSRANK(DELAY((-1 * RET), 6), 5))
def alpha_157(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(11) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    temp_1 = ((close_.iloc[-11:] - 1).diff(5).iloc[-6:].rank(axis=1, pct=True) * -1).rank(axis=1, pct=True).rank(axis=1, pct=True)
    temp_2 = rsum(rmin(temp_1, 2).iloc[-5:], 1).iloc[-5:]
    temp_3 = np.log(temp_2).rank(axis=1, pct=True).rank(axis=1, pct=True)
    part_1 = temp_3.min()
    part_2 = (func_ret(code).iloc[-11:] * -1).shift(6).iloc[-5:].rank(axis=0, pct=True).iloc[-1]
    alpha = part_1 + part_2
    return alpha.dropna()
# ((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE
def alpha_158(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    # 1 / (1 + com) = 2 / 15
    sma_close = ewmm(close_, 6.5)
    temp_1 = high_.iloc[-1] - sma_close.iloc[-1]
    temp_2 = low_.iloc[-1] - sma_close.iloc[-1]
    alpha = (temp_1 - temp_2) / close_.iloc[-1]
    return alpha.dropna()
# ((CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),6))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),6)*12*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),12))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),12)*6*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),24))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,D ELAY(CLOSE,1)),24)*6*24)*100/(6*12+6*24+12*24)
def alpha_159(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(25) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    data1 = np.maximum(high_.iloc[-24:], close_.shift().iloc[-24:])
    data2 = np.minimum(low_.iloc[-24:], close_.shift().iloc[-24:])
    x = ((close_.iloc[-1] - (data2.iloc[-6:]).sum()) / ((data1.iloc[-6:]).sum() - (data2.iloc[-6:]).sum())) * 12 * 24
    y = ((close_.iloc[-1] - (data2.iloc[-12:]).sum()) / ((data1.iloc[-12:]).sum() - (data2.iloc[-12:]).sum())) * 6 * 24
    z = ((close_.iloc[-1] - (data2.iloc[-24:]).sum()) / ((data1.iloc[-24:]).sum() - (data2.iloc[-24:]).sum())) * 6 * 24
    alpha = (x + y + z) * 100 / (6 * 12 + 12 * 24 + 6 * 24)
    return alpha.dropna()
# SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
def alpha_160(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    cond1 = close_ <= close_.shift()
    data1 = rstd(close_, 20).iloc[20:]
    data1[~cond1] = 0
    # 1 / (1 + com) = 1 / 20
    alpha = ewmm(data1, 19)
    alpha = alpha.iloc[-1, :]
    cond1 = alpha == 0
    alpha[cond1] = np.nan
    return alpha.dropna()
# MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)
def alpha_161(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(13) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    temp_1 = high_.iloc[-12:] - low_.iloc[-12:]
    temp_2 = abs(close_.shift().iloc[-12:] - high_.iloc[-12:])
    temp_3 = np.maximum(temp_1, temp_2)
    temp_4 = abs(close_.shift().iloc[-12:] - low_.iloc[-12:])
    temp_5 = np.maximum(temp_3, temp_4)
    alpha = temp_5.mean()
    return alpha.dropna()
# (SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))/(MAX(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12)-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12, 1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))
def alpha_162(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    temp = (close_ - close_.shift()).iloc[1:]
    # 1 / (1 + com) = 1 / 12
    data = ewmm(temp.clip(lower=0), 11) / ewmm(abs(temp), 11) * 100
    alpha = (data.iloc[-1] - data.iloc[-12:].min()) / (data.iloc[-12:].max() - data.iloc[-12:].min())
    return alpha.dropna()
# RANK(((((-1 * RET) * MEAN(VOLUME,20)) * VWAP) * (HIGH - CLOSE)))
def alpha_163(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    alpha = -1 * func_ret(code).iloc[-1] * volume_.iloc[-20:].mean() * avg_.iloc[-1] * (high_.iloc[-1] - low_.iloc[-1])
    alpha = alpha.rank(pct=True)
    return alpha.dropna()
# SMA((((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1)-MIN(((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1),12))/(HIGH-LOW)*100,13,2)
def alpha_164(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    cond1 = close_ > close_.shift()
    data1 = 1 / (close_ - close_.shift())
    data1[~cond1] = 1
    # 1 / (1 + com) = 2 / 13
    alpha = ewmm((data1.iloc[12:] - rmin(data1, 12).iloc[12:]) / (high_.iloc[12:] - low_.iloc[12:]) * 100, 5.5)
    alpha = alpha.iloc[-1, :]
    return alpha.dropna()
# MAX(SUMAC(CLOSE-MEAN(CLOSE,48)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,48)))/STD(CLOSE,48)
# -20*(20-1)^1.5*SUM(CLOSE/DELAY(CLOSE,1)-1-MEAN(CLOSE/DELAY(CLOSE,1)-1,20),20)/((20-1)*(20-2)(SUM((CLOSE/DELAY(CLOSE,1),20)^2,20))^1.5)
def alpha_166(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(40) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    part1 = -20 * (19) ** 1.5
    temp = close_.iloc[-39:] / close_.shift().iloc[-39:]
    part2 = (temp.iloc[-20:] - 1 - rmean(temp - 1, 20).iloc[-20:]).sum()
    part3 = 19 * 18 * (temp.iloc[-20:] ** 2).sum() ** (1.5)
    alpha = part1 * part2 / part3
    return alpha.dropna()
# SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)
def alpha_167(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(13) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    data = close_.iloc[-12:] - close_.shift().iloc[-12:]
    alpha = (data.clip(lower=0)).sum()
    return alpha.dropna()
# (-1*VOLUME/MEAN(VOLUME,20))
def alpha_168(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(20) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    alpha = -volume_.iloc[-1] / volume_.iloc[-20:].mean()
    return alpha.dropna()
# SMA(MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),12)-MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1), 26),10,1)
def alpha_169(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    # 1 / (1 + com) = 1 / 9
    temp_1 = ewmm((close_ - close_.shift()).iloc[1:], 8).shift().iloc[1:]
    alpha = ewmm((rmean(temp_1, 12) - rmean(temp_1, 26)).iloc[26:], 9)
    alpha = alpha.iloc[-1, :]
    return alpha.dropna()
# ((((RANK((1 / CLOSE)) * VOLUME) / MEAN(VOLUME,20)) * ((HIGH * RANK((HIGH - CLOSE))) / (SUM(HIGH, 5) / 5))) - RANK((VWAP - DELAY(VWAP, 5))))
def alpha_170(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    part_1 = (1 / close_.iloc[-1]).rank(pct=True) * volume_.iloc[-1] / volume_.iloc[-20:].mean()
    part_2 = (high_.iloc[-1] - close_.iloc[-1]).rank(pct=True) * high_.iloc[-1]
    part_3 = (high_.iloc[-5:]).sum() / 5
    part_4 = (avg_.iloc[-1] - avg_.shift(5).iloc[-1]).rank(pct=True)
    alpha = part_1 * part_2 / part_3 - part_4
    return alpha.dropna()
# ((-1 * ((LOW - CLOSE) * (OPEN^5))) / ((CLOSE - HIGH) * (CLOSE^5)))
def alpha_171(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(1) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    part_1 = (low_.iloc[-1] - close_.iloc[-1]) * (open_.iloc[-1] ** 5) * -1
    part_2 = (close_.iloc[-1] - high_.iloc[-1]) * close_.iloc[-1] ** 5
    alpha = part_1 / part_2
    return alpha.dropna()
# MEAN(ABS(SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0&HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)
def alpha_172(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(20) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    hd = high_.iloc[-19:] - high_.shift().iloc[-19:]
    ld = low_.shift().iloc[-19:] - low_.iloc[-19:]
    tr = np.maximum(
        np.maximum(high_.iloc[-19:] - low_.iloc[-19:], abs(high_.iloc[-19:] - close_.shift().iloc[-19:])),
        abs(low_.iloc[-19:] - close_.shift().iloc[-19:])
    )
    sum_tr = rsum(tr, 14).iloc[-6:]
    cond1 = (ld > 0) & (ld > hd)
    cond2 = (hd > 0) & (hd > ld)
    data1 = ld.copy()
    data1[~cond1] = 0
    data1 = rsum(data1, 14).iloc[-6:]
    data2 = hd.copy()
    data2[~cond2] = 0
    data2 = rsum(data2, 14).iloc[-6:]
    alpha = abs(data1 * 100 / sum_tr - data2 * 100 / sum_tr) / (data1 * 100 / sum_tr + data2 * 100 / sum_tr)
    alpha = alpha.mean() * 100
    return alpha.dropna()
# 3*SMA(CLOSE,13,2)-2*SMA(SMA(CLOSE,13,2),13,2)+SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2);
def alpha_173(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    # 1 / (1 + com) = 2 / 13
    temp_1 = ewmm(close_, 5.5)
    part_1 = temp_1 * 3
    part_2 = ewmm(temp_1, 5.5) * 2
    part_3 = ewmm(ewmm(ewmm(np.log(close_), 5.5), 5.5), 5.5)
    alpha = part_1.iloc[-1] - part_2.iloc[-1] + part_3.iloc[-1]
    return alpha.dropna()
# SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)
def alpha_174(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    cond1 = close_ < close_.shift()
    data1 = rstd(close_, 20)
    data1[cond1] = 0
    # 1 / (1 + com) = 1 / 20
    alpha = ewmm(data1.iloc[20:], 19)
    alpha = alpha.iloc[-1, :]
    return alpha.dropna()
# MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),6)
def alpha_175(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(7) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    temp_1 = np.maximum((high_.iloc[-6:] - low_.iloc[-6:]), abs(close_.shift().iloc[-6:] - high_.iloc[-6:]))
    temp_2 = np.maximum(temp_1, abs(close_.shift().iloc[-6:] - low_.iloc[-6:]))
    alpha = temp_2.mean()
    return alpha.dropna()
# CORR(RANK(((CLOSE - TSMIN(LOW, 12)) / (TSMAX(HIGH, 12) - TSMIN(LOW,12)))), RANK(VOLUME), 6)
def alpha_176(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(17) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    part_1 = (close_.iloc[-6:] - rmin(low_.iloc[-17:], 12).iloc[-6:]) / (rmax(high_.iloc[-17:], 12).iloc[-6:] - rmin(low_.iloc[-17:], 12).iloc[-6:])
    alpha = part_1.rank(axis=1, pct=True).corrwith(volume_.iloc[-6:].rank(axis=1, pct=True))
    return alpha.dropna()
# ((20-HIGHDAY(HIGH,20))/20)*100
def alpha_177(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(20) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    alpha = (20 - func_highday(high_, 20)) / 20 * 100
    return alpha.dropna()
# (CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME
def alpha_178(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(2) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    alpha = (close_.iloc[-1] - close_.shift().iloc[-1]) / close_.shift().iloc[-1] * volume_.iloc[-1]
    return alpha.dropna()
# (RANK(CORR(VWAP, VOLUME, 4)) *RANK(CORR(RANK(LOW), RANK(MEAN(VOLUME,50)), 12)))
def alpha_179(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    avg_ = calculate_avg(code)
    temp_1 = avg_.iloc[-4:].corrwith(volume_.iloc[-4:]).rank(pct=True)
    temp_2 = low_.iloc[-12:].rank(axis=1, pct=True).corrwith(rmean(volume_.iloc[-61:], 50).iloc[-12:].rank(axis=1, pct=True)).rank(pct=True)
    alpha = temp_1 * temp_2
    return alpha.dropna()
# ((MEAN(VOLUME,20) < VOLUME) ? ((-1 * TSRANK(ABS(DELTA(CLOSE, 7)), 60)) * SIGN(DELTA(CLOSE, 7)) : (-1 *VOLUME)))
def alpha_180(code, end_date=None, fq='pre'):
    open_, close_, high_, low_, volume_ = (code[k].tail(67) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    ma = volume_.iloc[-20:].mean()
    cond = ma < volume_.iloc[-1]
    sign = np.sign(close_.iloc[-8:].diff(7).iloc[-1])
    left = (((close_.iloc[-67:].diff(7).abs())
             .iloc[-60:, :].rank(axis=0, pct=True) * (-1))
            .iloc[-1, :] * sign)
    right = volume_.iloc[-1, :] * (-1)
    right[cond] = left[cond]
    alpha = right
    return alpha.dropna()
# SUM(((CLOSE/DELAY(CLOSE,1)-1)-MEAN((CLOSE/DELAY(CLOSE,1)-1),20))-(BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^2,20)/SUM((BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^3)
def alpha_181(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(40) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    bic_, bio_ = get_bench(code)
    temp_1 = close_.iloc[-39:] / close_.shift().iloc[-39:] - 1
    temp_2 = bic_.iloc[-20:] - rmean(bic_.iloc[-39:], 20).iloc[-20:]
    part_1 = (temp_1.iloc[-20:] - rmean(temp_1, 20).iloc[-20:] - temp_2 ** 2).sum()
    part_2 = (temp_2.iloc[-20:] ** 3).sum()
    alpha = part_1 / part_2
    return alpha.dropna()
# COUNT((CLOSE>OPEN & BANCHMARKINDEXCLOSE>BANCHMARKINDEXOPEN)OR(CLOSE<OPEN & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN),20)/20
def alpha_182(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(20) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    bic_, bio_ = get_bench(code)
    cond1 = close_.iloc[-20:] > open_.iloc[-20:]
    cond2 = bic_.iloc[-20:] > bio_.iloc[-20:]
    cond3 = close_.iloc[-20:] < open_.iloc[-20:]
    cond4 = bic_.iloc[-20:] < bio_.iloc[-20:]
    cond5 = (cond1 & cond2) | (cond3 & cond4)
    alpha = cond5.sum()
    return alpha.dropna()
# MAX(SUMAC(CLOSE-MEAN(CLOSE,24)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,24)))/STD(CLOSE,24)
# (RANK(CORR(DELAY((OPEN - CLOSE), 1), CLOSE, 200)) + RANK((OPEN - CLOSE)))
def alpha_184(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    part_1 = (open_.shift().iloc[-200:] - close_.shift().iloc[-200]).corrwith(close_.iloc[-200:]).rank(pct=True)
    part_2 = (open_.iloc[-1] - close_.iloc[-1]).rank(pct=True)
    alpha = part_1 + part_2
    return alpha.dropna()
# RANK((-1 * ((1 - (OPEN / CLOSE))^2)))
def alpha_185(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(1) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    alpha = (((1 - open_.iloc[-1] / close_.iloc[-1]) ** 2) * -1).rank(pct=True)
    return alpha.dropna()
# (MEAN(ABS(SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0&HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)+DELAY(MEAN(ABS(SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0&HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6),6))/2
def alpha_186(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(26) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    hd = high_.iloc[-25:] - high_.shift().iloc[-25:]
    ld = low_.shift().iloc[-25:] - low_.iloc[-25:]
    tr = np.maximum(
        np.maximum(high_.iloc[-25:] - low_.iloc[-25:], abs(high_.iloc[-25:] - close_.shift().iloc[-25:])),
        abs(low_.iloc[-25:] - close_.shift().iloc[-25:])
    )
    sum_tr = rsum(tr, 14).iloc[-12:]
    cond1 = (ld > 0) & (ld > hd)
    cond2 = (hd > 0) & (hd > ld)
    data1 = ld.copy()
    data1[~cond1] = 0
    data1 = rsum(data1, 14).iloc[-12:]
    data2 = hd.copy()
    data2[~cond2] = 0
    data2 = rsum(data2, 14).iloc[-12:]
    alpha = abs(data1 * 100 / sum_tr - data2 * 100 / sum_tr) / (data1 * 100 / sum_tr + data2 * 100 / sum_tr)
    alpha = rmean(alpha, 6).iloc[-7:]
    alpha = (alpha.iloc[-1] + alpha.shift(6).iloc[-1]) / 2 * 100
    return alpha.dropna()
# SUM((OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))),20)
def alpha_187(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(21) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    cond1 = open_.iloc[-20:] <= open_.shift().iloc[-20:]
    data1 = np.maximum(high_.iloc[-20:] - open_.iloc[-20:], (open_.iloc[-20:] - open_.shift().iloc[-20:]))
    data1[cond1] = 0
    alpha = data1.sum()
    return alpha.dropna()
# ((HIGH-LOW–SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100
def alpha_188(code):
    open_, close_, high_, low_, volume_ = (code[k] for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    temp = high_ - low_
    # 1 / (1 + com) = 2 / 11
    part_1 = temp / ewmm(temp, 4.5) - 1
    alpha = part_1 * 100
    alpha = alpha.iloc[-1, :]
    return alpha.dropna()
# MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)
def alpha_189(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(11) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    alpha = abs(close_.iloc[-6:] - rmean(close_.iloc[-11:], 6).iloc[-6:]).mean()
    return alpha.dropna()
# LOG((COUNT(CLOSE/DELAY(CLOSE)-1>((CLOSE/DELAY(CLOSE,19))^(1/20)-1),20)-1)*(SUMIF(((CLOSE/DELAY(C LOSE)-1-(CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)- 1))/((COUNT((CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1),20))*(SUMIF((CLOSE/DELAY(CLOS E)-1-((CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1>(CLOSE/DELAY(CLOSE,19))^(1/20)-1))))
def alpha_190(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(39) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    temp1 = close_.iloc[-20:] / close_.shift().iloc[-20:] - 1
    temp2 = (close_.iloc[-20:] / close_.shift(19).iloc[-20:]) ** (1 / 20) - 1
    part = temp1 - temp2

    alpha = np.log((((part > 0).sum() - 1) * (part.where(part < 0, 0) ** 2).sum()) / ((part < 0).sum() * (part.where(part > 0, 0) ** 2).sum()))
    return alpha.dropna()
# ((CORR(MEAN(VOLUME,20),LOW,5)+((HIGH+LOW)/2))-CLOSE)# 
def alpha_191(code):
    open_, close_, high_, low_, volume_ = (code[k].tail(24) for k in ['open', 'adj_close', 'high', 'low', 'volume'])
    alpha = rmean(volume_.iloc[-24:], 20).iloc[-5:].corrwith(low_.iloc[-5:]) + (high_.iloc[-1] + low_.iloc[-1]) / 2 - close_.iloc[-1]
    return alpha.dropna()