import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import reduce

def calculateFactorReturns(usa_factor, value_weighted=True):
    # Read data
    CD = pd.read_parquet(r'data\SPXConstituentsDaily.parquet')
    CM = pd.read_parquet(r'data\CompleteMapping.parquet')
    # Preprocess CD
    CD = CD.rename(columns={"Date": "date"})
    CD['date'] = pd.to_datetime(CD['date'])
    CD = CD.sort_values(by='date')
    # Preprocess CM
    CM = CM.rename(columns={"PERMNO": "permno"})
    # Setup subtable of CD
    daily_cusip_data = CD[['date', 'CUSIP']].drop_duplicates()
    # Setup subtable of CM
    cusip_securityid_date = CM[['permno', 'CUSIP']].drop_duplicates()
    # Merge table of daily_cusip_data and cusip_securityid_date
    spx_mapping = pd.merge(daily_cusip_data, cusip_securityid_date, on='CUSIP', how='left').dropna()
    spx_mapping['permno'] = spx_mapping['permno'].astype(float)
    spx_mapping = spx_mapping[['date', 'permno', 'CUSIP']].sort_values(by=['date','permno'])
    spx_mapping['period'] = (spx_mapping['date'].dt.to_period('M'))
    
    # calculation start
    factor_returns_list = []
    # Get characteristics names
    char_name = usa_factor.columns[7:]
    # Get the list of permno in each month according to the spx_mapping table
    month_constituents = spx_mapping.groupby('period')['permno'].apply(set).to_dict() 
    usa_factor['period'] = (usa_factor['eom'].dt.to_period('M'))
    
    for i in char_name:
        
        # Extract each factor value
        merged = usa_factor[['period', 'permno', 'ret', 'me', i]].dropna()
        
        # Group factor value into 10 groups
        results = []
        for period, period_data in tqdm(merged.groupby('period'), desc=f"Processing {i}"):
            permnos = list(month_constituents.get(period, set()) & set(period_data['permno'].unique()))
            period_data = period_data[period_data['permno'].isin(permnos)]
            # To make sure there are at least 10 datas in one period
            if len(period_data) < 10:
                results.append({'period': period, i: np.nan})
                continue     
            try:
                factor_uniques=pd.DataFrame(np.sort(period_data[i].unique()), columns=[i])
                factor_uniques = factor_uniques.assign(
                    decile=lambda x: pd.qcut(
                        x[i], 
                        q=10, 
                        labels=False, 
                        duplicates='drop'
                    ) + 1
                )
                period_data = pd.merge(period_data, factor_uniques, on=i, how='left').dropna()
            except ValueError:
                results.append({'period': period, i: np.nan})
                continue
            # Computing grouped rate of return   
            if value_weighted:
                decile_returns = period_data.groupby('decile').apply(lambda g: np.average(g['ret'], weights=g['me']))
            else:
                decile_returns = period_data.groupby('decile')['ret'].mean()    
            # To make sure there are at least two groups
            if len(decile_returns) < 2:
                results.append({'period': period, i: np.nan})
                continue
            # computing long-short portfolio    
            max_decile = decile_returns.index.max()
            min_decile = decile_returns.index.min()
            results.append({
                'period': period,
                i: decile_returns.get(max_decile, 0) - decile_returns.get(min_decile, 0)
            })

        # results
        if results:
            factor_returns = pd.DataFrame(results).set_index('period')
            factor_returns_list.append(factor_returns)
            
    # Collate all results
    return reduce(lambda x,y: x.join(y, how='inner'), factor_returns_list).dropna(how='all').reset_index().sort_values('period')

