import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

class USACalculator:
    def __init__(self, csv_file, output_dir, chunk_size):
        self.csv_file = csv_file
        self.output_dir = output_dir
        self.chunk_size = chunk_size

    def csv_to_parquet_chunks(self):
        # Create output directory
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set up variables
        chunk_number = 1
        rows = []
        
        with open(self.csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # Get header
            header = next(reader)
            for i, row in enumerate(reader, 1):
                rows.append(row)
                # Write to parquet file when chunk size is reached
                if i % self.chunk_size == 0:
                    df = pd.DataFrame(rows, columns=header)
                    output_file = output_path / f'chunk_{chunk_number}.parquet'
                    df.to_parquet(output_file, index=False)
                    rows = []
                    chunk_number += 1
                    print(f'Created {output_file} with {self.chunk_size} rows')
            
            # Write any remaining rows to a final parquet file
            if rows:
                df = pd.DataFrame(rows, columns=header)
                output_file = output_path / f'chunk_{chunk_number}.parquet'
                df.to_parquet(output_file, index=False)
                print(f'Created final {output_file} with {len(rows)} rows')

    def filter_chunks(self):
        output_path = Path(self.output_dir)
        for i in tqdm(range(1, 15)):
            input_name = f'chunk_{i}.parquet'
            output_name = f'filter_{i}.parquet'
            df = pd.read_parquet(output_path/input_name)
            jkp = pd.read_excel('Factor Details.xlsx')['abr_jkp'].dropna().tolist()
            columns_to_keep = ['eom', 'permno', 'source_crsp', 'crsp_shrcd', 'crsp_exchcd', 'ret', 'me'] + jkp
            df = df[columns_to_keep]
            df = df.replace('', np.nan)
            df = df.astype(float)
            df = df[(df.source_crsp == 1)]
            df = df[(df.crsp_shrcd == 10) | (df.crsp_shrcd == 11)]
            df = df[(df.crsp_exchcd == 1) | (df.crsp_exchcd == 2) | (df.crsp_exchcd == 3)]
            df = df[df.prc > 5]
            df.to_parquet(output_path/output_name)

    def merge_filters(self):
        parquet_files = []
        output_path = Path(self.output_dir)
        for i in range(1, 15):
            output_name = f'filter_{i}.parquet'
            parquet_files.append(output_name)

        dfs = [pd.read_parquet(output_path/file) for file in parquet_files]
        combined_df = pd.concat(dfs)
        combined_df['eom'] = pd.to_datetime(combined_df['eom'], format = "%Y%m%d")
        combined_df.to_parquet(output_path.parent/'factor_value'/'usa_factor.parquet', index=False)