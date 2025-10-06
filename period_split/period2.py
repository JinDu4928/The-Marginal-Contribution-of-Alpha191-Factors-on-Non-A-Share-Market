import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from Alpha_Formulas import *
import warnings
warnings.filterwarnings("ignore")

class AlphaCalculator:
    def __init__(self, CP, CD, CM, SP):
        self.CP= CP
        self.CD= CD
        self.CM= CM
        self.SP= SP
    

    # Preprocess the data
    def preprocessData(self):
        start_date = '2005-01-02'
        end_date = '2011-01-02'
        
        # Preprocess CP
        self.CP = self.CP[(self.CP[['bidlow','askhigh','closeprice','volume','openprice','sharesoutstanding']] > 0).all(axis=1)]
        self.CP['date'] = pd.to_datetime(self.CP['date'])
        self.CP = self.CP.sort_values(by=['securityid', 'date'])
        
        # Preprocess CD
        self.CD = self.CD.rename(columns={"Date": "date"})
        self.CD['date'] = pd.to_datetime(self.CD['date'])
        self.CD = self.CD.sort_values(by='date')
        
        # Preprocess CM
        self.CM = self.CM.rename(columns={"SecurityID": "securityid"})
        
        # Preprocess SP
        self.SP['date'] = pd.to_datetime(self.SP['date'])
        self.SP = self.SP.sort_values(by='date')

        # Add adjusted close prices column to CP
        self.CP["adjusted_close"]=self.CP["closeprice"]* self.CP["adjustmentfactor2"]/self.CP.groupby('securityid')["adjustmentfactor2"].transform('last')
        first_day_indices = self.CP.groupby('securityid')['date'].idxmin()
        self.CP = self.CP.drop(first_day_indices)
        # Add market cap column to CP
        self.CP["market_cap"]=self.CP["adjusted_close"]*self.CP["sharesoutstanding"]
        # Setup subtable of CP
        self.CP = self.CP[(self.CP['date'] >= start_date) & (self.CP['date'] <= end_date)]
        volume_data = self.CP[['securityid', 'date', 'volume']].drop_duplicates()
        open_data = self.CP[['securityid', 'date', 'openprice']].drop_duplicates()
        close_data = self.CP[['securityid', 'date', 'closeprice']].drop_duplicates()
        low_data = self.CP[['securityid', 'date', 'bidlow']].drop_duplicates()
        high_data = self.CP[['securityid', 'date', 'askhigh']].drop_duplicates()
        adj_close_data = self.CP[['securityid', 'date', 'adjusted_close']].drop_duplicates()
        return_data = self.CP[['securityid', 'date', 'totalreturn']].drop_duplicates()
        market_data = self.CP[['securityid', 'date', 'market_cap']].drop_duplicates()

        # Setup subtable of CD
        daily_cusip_data = self.CD[['date', 'CUSIP']].drop_duplicates()

        # Setup subtable of CM
        cusip_securityid_date = self.CM[['securityid', 'CUSIP']].drop_duplicates()
  
        # Setup subtable of SP
        bench_open_data = self.SP[['securityid', 'date', 'openprice']].drop_duplicates().set_index('date')
        bench_close_data = self.SP[['securityid', 'date', 'closeprice']].drop_duplicates().set_index('date')

        # Table of cusip_securityid mapping
        spx_mapping = pd.merge(daily_cusip_data, cusip_securityid_date, on='CUSIP', how='left').dropna()
        spx_mapping = spx_mapping[['date', 'securityid', 'CUSIP']].sort_values(by=['date','securityid'])
        spx_mapping['securityid'] = spx_mapping['securityid'].astype(int)

        # Save all tables above into an object
        self.sp500 = {
                'volume_data': volume_data,
                'open_data': open_data,
                'close_data': close_data,
                "low_data": low_data,
                "high_data": high_data,
                "adj_close_data": adj_close_data,
                "return_data": return_data,
                'market_data': market_data,
                "spx_mapping": spx_mapping,
                'bench_open_data': bench_open_data,
                'bench_close_data': bench_close_data
        }


    # Calculate the daily alpha191 factor value for each constituent stock in sp500 (according to ConstituentsDaily.parquet to filter daily constituents)
    def calculateAlphaFactors(self):

        # Convert table to wide form for following calculating
        volume_wide = self.sp500["volume_data"].pivot(index='date', columns='securityid', values='volume').sort_index()
        open_wide = self.sp500["open_data"].pivot(index='date', columns='securityid', values='openprice').sort_index()
        low_wide = self.sp500["low_data"].pivot(index='date', columns='securityid', values='bidlow').sort_index()
        high_wide = self.sp500["high_data"].pivot(index='date', columns='securityid', values='askhigh').sort_index()
        adj_close_wide = self.sp500["adj_close_data"].pivot(index='date', columns='securityid', values='adjusted_close').sort_index()

        # Preprocessing constituents mapping
        date_constituents = self.sp500["spx_mapping"].groupby('date')['securityid'].apply(set).to_dict()
        
        # Get dates
        unique_dates = volume_wide.index.sort_values()
        
        # Initialize final table
        self.factor_dict = {}
        valid_alphas = [f"alpha_{j:03d}" for j in range(1, 192) 
                    if f"alpha_{j:03d}" in globals() and callable(globals()[f"alpha_{j:03d}"])]
        for alpha_name in valid_alphas:
            self.factor_dict[alpha_name] = pd.DataFrame(index=unique_dates, columns=volume_wide.columns, dtype=np.float64)

        # Filter table
        for i in tqdm(range(251, len(unique_dates))): 
            # Get a 252-day window
            end_date = unique_dates[i]
            window_dates = unique_dates[i-251:i+1]
            # Get day's constituents
            securityids = list(date_constituents.get(end_date, set()) & set(volume_wide.columns))
            if not securityids:
                continue
        
            # Filter and exclude scurityid with data less than 252 days 
            volume_window = volume_wide.loc[window_dates, securityids].dropna(axis=1, how="any")
            securityids = list(set(volume_window.columns))
            open_window = open_wide.loc[window_dates, securityids]
            high_window = high_wide.loc[window_dates, securityids]
            low_window = low_wide.loc[window_dates, securityids]
            adj_close_window = adj_close_wide.loc[window_dates, securityids]

            # Save all the windowed tables into a dictionary
            data = {
                'volume': volume_window,
                'open': open_window,
                'high': high_window,
                'low': low_window,
                'adj_close': adj_close_window,
                'bench_open': self.sp500['bench_open_data'],
                'bench_close': self.sp500['bench_close_data']
            }

            # Calculate alpha factor values
            for alpha_name in valid_alphas:
                alpha_func = globals()[alpha_name]
                factor_values = alpha_func(data)
                self.factor_dict[alpha_name].loc[end_date, securityids] = factor_values
        
        # Dropna
        for alpha_name in valid_alphas:
            self.factor_dict[alpha_name] = self.factor_dict[alpha_name].dropna(how = "all")
        
        return self.factor_dict
    
current_path = Path()
parent_path = current_path.parent   
CP = pd.read_parquet(parent_path/'data'/'SPXConstituentsPrices.parquet')
CD = pd.read_parquet(parent_path/'data'/'SPXConstituentsDaily.parquet')
CM = pd.read_parquet(parent_path/'data'/'CompleteMapping.parquet')
SP = pd.read_parquet(parent_path/'data'/'SPXPrices.parquet')

calculator = AlphaCalculator(CP, CD, CM, SP)
calculator.preprocessData()
factor_dict = calculator.calculateAlphaFactors()

merged = pd.DataFrame()
for i in range(1, 192):
    name = f"alpha_{i:03d}"
    if name in factor_dict:
        long = factor_dict[name].T.reset_index().melt(id_vars='securityid', var_name='date', value_name=name).dropna()
        merged = pd.merge(merged, long, on=['date', 'securityid'], how='outer') if not merged.empty else long
merged.to_parquet(parent_path/'period_split'/'period2.parquet', index=False)