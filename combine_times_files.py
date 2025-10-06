import pandas as pd

def combine_times_files():
    parquet_files = []
    for i in range(1, 6):
        output_name = fr'period_split\period{i}.parquet'
        parquet_files.append(output_name)

    dfs = [pd.read_parquet(file) for file in parquet_files]
    combined_df = pd.concat(dfs)
    combined_df['securityid'] = combined_df['securityid'].astype('float64')
    combined_df.to_parquet(r'factor_value\alpha_factor.parquet', index=False)