import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

class UsaPortfolios55:
    def __init__(self, usa_factor):
        self.usa_factor = usa_factor
    
    def data_preprocess(self):
        # Consist the start time with alpha factor
        start_date = '2004-01-01'
        self.usa_factor = self.usa_factor[self.usa_factor['eom'] >= start_date]
        # Exclude discrete data
        columns_to_exclude = ['ni_inc8q', 'f_score']
        self.usa_factor = self.usa_factor.drop(columns=columns_to_exclude)
        # Deal with infinity and infinity
        self.usa_factor = self.usa_factor.replace([np.inf, -np.inf], [1e10, -1e10])
        self.usa_factor['period'] = (self.usa_factor['eom'].dt.to_period('M')).astype('str')
        # Screen the constituents of sp500
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
        spx_mapping['period'] = (spx_mapping['date'].dt.to_period('M')).astype('str')
        spx_mapping = spx_mapping[['period','permno']].drop_duplicates()
        # Filter usa_factor with sp500 constituents
        self.usa_factor_filter = pd.merge(self.usa_factor, spx_mapping, on=['period','permno'], how='inner')
        columns_to_exclude = ['source_crsp', 'crsp_shrcd', 'crsp_exchcd', 'period']
        self.usa_factor_filter = self.usa_factor_filter.drop(columns=columns_to_exclude)
        
    def portfolio_group(self):     
        # Calculate year average to build the portfoilo
        usa_factor_year = (
            self.usa_factor_filter.groupby(["permno", pd.Grouper(key="eom", freq="Y")])
            .mean()
            .reset_index()
        )
        usa_factor_year['eom'] = usa_factor_year['eom'].dt.to_period('Y').astype('str')
        # Get the characteristics name
        char_name = usa_factor_year.columns[4:]
        # Automatically initialize the nested dictionary
        self.port_dict = defaultdict(lambda: defaultdict(dict))
        # Build portfolios
        for i in char_name:
            subset = usa_factor_year[['eom', 'permno','me', 'ret', i]].dropna()
            for period, period_data in tqdm(subset.groupby('eom'), desc=f"Processing {i}"):
                factor_uniques=pd.DataFrame(np.sort(period_data[i].unique()), columns=[i])
                factor_uniques = factor_uniques.assign(
                    factor_group=lambda x: pd.qcut(
                        x[i], 
                        q=5, 
                        labels=False, 
                        duplicates='drop'
                    ) + 1
                )
                period_data = pd.merge(period_data, factor_uniques, on=i, how='left').dropna()
                cap_uniques=pd.DataFrame(np.sort(period_data['me'].unique()), columns=['me'])
                cap_uniques = cap_uniques.assign(
                    cap_group=lambda x: pd.qcut(
                        x['me'], 
                        q=[0, 0.2, 0.4, 0.6, 0.8, 1],
                        labels=["Low", "LowMedium", "Medium", "MediumHigh", "High"],
                        duplicates='drop'
                    )
                )
                period_data = pd.merge(period_data, cap_uniques, on='me', how='left').dropna()
                # Save diffierent portfolio into dict
                group_combinations = [
                    ('Low', 1, 'low1'),
                    ('Low', 2, 'low2'),
                    ('Low', 3, 'low3'),
                    ('Low', 4, 'low4'),
                    ('Low', 5, 'low5'),
                    ('LowMedium', 1, 'lowmedium1'),
                    ('LowMedium', 2, 'lowmedium2'),
                    ('LowMedium', 3, 'lowmedium3'),
                    ('LowMedium', 4, 'lowmedium4'),
                    ('LowMedium', 5, 'lowmedium5'),
                    ('Medium', 1, 'medium1'),
                    ('Medium', 2, 'medium2'),
                    ('Medium', 3, 'medium3'),
                    ('Medium', 4, 'medium4'),
                    ('Medium', 5, 'medium5'),
                    ('MediumHigh', 1, 'mediumhigh1'),
                    ('MediumHigh', 2, 'mediumhigh2'),
                    ('MediumHigh', 3, 'mediumhigh3'),
                    ('MediumHigh', 4, 'mediumhigh4'),
                    ('MediumHigh', 5, 'mediumhigh5'),
                    ('High', 1, 'high1'),
                    ('High', 2, 'high2'),
                    ('High', 3, 'high3'),
                    ('High', 4, 'high4'),
                    ('High', 5, 'high5')
                ]
                for cap_group, factor_group, key in group_combinations:
                    filtered_data = period_data[(period_data['cap_group'] == cap_group) & (period_data['factor_group'] == factor_group)]
                    # Make sure there are at least 10 securities in one portfolio
                    self.port_dict[i][period][key] = filtered_data['permno'].tolist()
        
    def portfolio_ret(self):   
        # Calculate month average to calculate portfolio return
        usa_factor_month = (
            self.usa_factor_filter.groupby(["permno", pd.Grouper(key="eom", freq="M")])
            .mean()
            .reset_index()
        )
        usa_factor_month['eom'] = usa_factor_month['eom'].dt.to_period('M').astype('str')
        # Get the characteristics name
        char_name = usa_factor_month.columns[4:]
        port_returns_list = []
        # Calculate portfolios return
        for i in char_name:
            results = []
            subset = usa_factor_month[['eom', 'permno','me', 'ret', i]].dropna()
            for period, period_data in tqdm(subset.groupby('eom'), desc=f"Processing {i}"):
                # Using last year average value to build next year portfolio, so need to skip the first year portfolio returns calculating
                lastyear = pd.to_datetime(period, format='%Y-%m')
                lastyear = str(lastyear.to_period('Y') - 1)
                if lastyear == '2003':
                    continue
                groups = ['low1', 'low2', 'low3', 'low4', 'low5', 'lowmedium1', 'lowmedium2','lowmedium3','lowmedium4','lowmedium5','medium1', 'medium2', 'medium3', 'medium4', 'medium5', 'mediumhigh1', 'mediumhigh2', 'mediumhigh3', 'mediumhigh4', 'mediumhigh5', 'high1', 'high2', 'high3', 'high4', 'high5']
                ports = {}
                ret = {}
                for group in groups:
                    ports[group] = period_data[period_data['permno'].isin(self.port_dict[i][lastyear][group])]
                    ret[group] = np.average(ports[group]['ret'], weights=ports[group]['me']) if len(ports[group]) else np.nan
                results.append({
                            'usa_name': i,
                            'period': period,
                            'low1' : ret['low1'],
                            'low2' : ret['low2'],
                            'low3' : ret['low3'],
                            'low4' : ret['low4'],
                            'low5' : ret['low5'],
                            'lowmedium1' : ret['lowmedium1'],
                            'lowmedium2' : ret['lowmedium2'],
                            'lowmedium3' : ret['lowmedium3'],
                            'lowmedium4' : ret['lowmedium4'],
                            'lowmedium5' : ret['lowmedium5'],
                            'medium1' : ret['medium1'],
                            'medium2' : ret['medium2'],
                            'medium3' : ret['medium3'],
                            'medium4' : ret['medium4'],
                            'medium5' : ret['medium5'],
                            'mediumhigh1' : ret['mediumhigh1'],
                            'mediumhigh2' : ret['mediumhigh2'],
                            'mediumhigh3' : ret['mediumhigh3'],
                            'mediumhigh4' : ret['mediumhigh4'],
                            'mediumhigh5' : ret['mediumhigh5'],
                            'high1' : ret['high1'],
                            'high2' : ret['high2'],
                            'high3' : ret['high3'],
                            'high4' : ret['high4'],
                            'high5' : ret['high5']
                        })
            port_returns = pd.DataFrame(results).set_index('period')
            port_returns_list.append(port_returns)
        return pd.concat(port_returns_list).dropna(how='all').reset_index().sort_values(['usa_name','period'])
