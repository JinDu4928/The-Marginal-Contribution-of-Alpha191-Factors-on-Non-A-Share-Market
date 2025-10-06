import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

class Alpha191Portfolios:
    def __init__(self, alpha_factor):
        self.alpha_factor = alpha_factor
    
    def data_preprocess(self):
        # Consist the end time with usa factor
        end_date = '2022-12-31'
        self.alpha_factor = self.alpha_factor[self.alpha_factor['date'] <= end_date]
        # Exclude discrete data
        columns_to_exclude = ['alpha_004', 'alpha_006', 'alpha_053', 'alpha_056', 'alpha_123', 'alpha_148', 'alpha_154']
        self.alpha_factor = self.alpha_factor.drop(columns=columns_to_exclude)
        # Deal with infinity and infinity
        self.alpha_factor = self.alpha_factor.replace([np.inf, -np.inf], [1e10, -1e10])
        # Get 'marekt_cap' and 'totalreturn' data
        CP = pd.read_parquet(r'data\SPXConstituentsPrices.parquet')
        CP = CP[(CP[['bidlow','askhigh','closeprice','volume','openprice','sharesoutstanding']] > 0).all(axis=1)]
        CP['date'] = pd.to_datetime(CP['date'])
        CP = CP.sort_values(by=['securityid', 'date'])
        CP["adjusted_close"]=CP["closeprice"]* CP["adjustmentfactor2"]/CP.groupby('securityid')["adjustmentfactor2"].transform('last')
        first_day_indices = CP.groupby('securityid')['date'].idxmin()
        CP = CP.drop(first_day_indices)
        CP["market_cap"]=CP["adjusted_close"]*CP["sharesoutstanding"]
        subcp = CP[['securityid', 'date', 'market_cap', 'totalreturn']]
        self.alpha_factor = pd.merge(self.alpha_factor, subcp, how='left', on=['securityid', 'date'])
    
    def portfolio_group(self):    
        # Calculate year average to build the portfoilo
        alpha_factor_year = (
            self.alpha_factor.groupby(["securityid", pd.Grouper(key="date", freq="Y")])
            .mean()
            .reset_index()
        )
        alpha_factor_year['date'] = alpha_factor_year['date'].dt.to_period('Y').astype('str')
        # Get the alpha name
        alpha_name = alpha_factor_year.columns[2:163]
        # Automatically initialize the nested dictionary
        self.port_dict = defaultdict(lambda: defaultdict(dict))
        # Build portfolios
        for i in alpha_name:
            subset = alpha_factor_year[['date', 'securityid','market_cap', 'totalreturn', i]].dropna()
            for period, period_data in tqdm(subset.groupby('date'), desc=f"Processing {i}"):
                factor_uniques=pd.DataFrame(np.sort(period_data[i].unique()), columns=[i])
                factor_uniques = factor_uniques.assign(
                    factor_group=lambda x: pd.qcut(
                        x[i], 
                        q=2, 
                        labels=False, 
                        duplicates='drop'
                    ) + 1
                )
                period_data = pd.merge(period_data, factor_uniques, on=i, how='left').dropna()
                cap_uniques=pd.DataFrame(np.sort(period_data['market_cap'].unique()), columns=['market_cap'])
                cap_uniques = cap_uniques.assign(
                    cap_group=lambda x: pd.qcut(
                        x['market_cap'], 
                        q=[0, 0.33, 0.66, 1],
                        labels=["Low", "Medium", "High"],
                        duplicates='drop'
                    )
                )
                period_data = pd.merge(period_data, cap_uniques, on='market_cap', how='left').dropna()
                # Save diffierent portfolio into dict
                group_combinations = [
                    ('Low', 1, 'low1'),
                    ('Low', 2, 'low2'),
                    ('Medium', 1, 'medium1'),
                    ('Medium', 2, 'medium2'),
                    ('High', 1, 'high1'),
                    ('High', 2, 'high2')
                ]
                for cap_group, factor_group, key in group_combinations:
                    filtered_data = period_data[(period_data['cap_group'] == cap_group) & (period_data['factor_group'] == factor_group)]
                    # Make sure there are at least 10 securities in one portfolio
                    if len(filtered_data) >= 10:
                        self.port_dict[i][period][key] = filtered_data['securityid'].tolist()
                    else:
                        self.port_dict[i][period][key] = []
                        print(f"there is a empty list -> i: {i}, period: {period}, key: {key}")
        
    def portfolio_ret(self):
        # Calculate month average to calculate portfolio return
        alpha_factor_month = (
            self.alpha_factor.groupby(["securityid", pd.Grouper(key="date", freq="M")])
            .mean()
            .reset_index()
        )
        alpha_factor_month['date'] = alpha_factor_month['date'].dt.to_period('M').astype('str')
        # Get the alpha name
        alpha_name = alpha_factor_month.columns[2:163]
        port_returns_list = []
        # Calculate portfolios return
        for i in alpha_name:
            results = []
            subset = alpha_factor_month[['date', 'securityid','market_cap', 'totalreturn', i]].dropna()
            for period, period_data in tqdm(subset.groupby('date'), desc=f"Processing {i}"):
                # Using last year average value to build next year portfolio, so need to skip the first year portfolio returns calculating
                lastyear = pd.to_datetime(period, format='%Y-%m')
                lastyear = str(lastyear.to_period('Y') - 1)
                if lastyear == '2003':
                    continue
                groups = ['low1', 'low2', 'medium1', 'medium2', 'high1', 'high2']
                ports = {}
                ret = {}
                for group in groups:
                    ports[group] = period_data[period_data['securityid'].isin(self.port_dict[i][lastyear][group])]
                    ret[group] = np.average(ports[group]['totalreturn'], weights=ports[group]['market_cap']) if len(ports[group]) else np.nan
                results.append({
                            'alpha_name': i,
                            'period': period,
                            'low1' : ret['low1'],
                            'low2' : ret['low2'],
                            'medium1' : ret['medium1'],
                            'medium2' : ret['medium2'],
                            'high1' : ret['high1'],
                            'high2' : ret['high2']
                        })
            port_returns = pd.DataFrame(results).set_index('period')
            port_returns_list.append(port_returns)
        return pd.concat(port_returns_list).dropna(how='all').reset_index().sort_values(['alpha_name','period'])