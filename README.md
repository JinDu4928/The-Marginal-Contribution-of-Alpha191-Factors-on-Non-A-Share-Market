# **The Marginal Contribution of Alpha191 Factors on Non-A-Share Market**
This is a short README file for the master thesis titled *The Marginal Contribution of Alpha191 Factors on the Non-A-Share Market*. For a deeper understanding of the code, please check the two files "REPRODUCE.ipynb" and "REPRODUCE_read_only.ipynb". "REPRODUCE_read_only.ipynb" is a read-only file containing saved results, while "REPRODUCE.ipynb" is an empty, modifiable file for reader to experiment with.
## Repository Structure
```plaintext
The Marginal Contribution of Alpha191 Factors on Non-A-Share Market/
├── data/                                # Original datasets
│ ├── CompleteMapping.parquet            # Mapping file for Original datasets
│ ├── SPXConstituentsDaily.parquet       # Daily constituents list of s&p500
│ ├── SPXConstituentsPrices.parquet      # Daily constituents data of s&p500
│ ├── SPXPrices.parquet                  # Daily data of s&p500 index
│ └── usa.csv                            # Jensen dataset original factor value from USA stocks
├── env/                                 # Environment file for runing code
│ ├── environment.yml                    # Required libraries documentation in a Conda environment
│ └── requirements.txt                   # Required libraries documentation in a pip environment
├── factor_returns/                      # Results of calculated factor returns
├── factor_value/                        # Results of calculated factor value
├── period_split/                        # A folder for storing processed s&p500 data
├── portfolios/                          # Results of built portfolios
├── usa_chunks/                          # A folder for storing processed Jensen dataset data
├── Alpha_Formulas.py                    # Alpha191 formulas
├── class_Alpha191Portfolios.py          # A class for building 3×2 Portfolios for Alpha191 factors
├── class_Alpha191Portfolios55.py        # A class for building 5×5 portfolios for Alpha191 factors
├── class_AlphaCalculator.py             # A class for calculating Alpha191 factor value and returns
├── class_DSregression.py                # A class for regression of DS, SS, EN, and PCA
├── class_USACalculator.py               # A class for calculating Jensen dataset factor value
├── class_UsaPortfolios.py               # A class for building 3×2 portfolios for Jensen dataset factors
├── class_UsaPortfolios55.py             # A class for building 5×5 portfolios for Jensen dataset factors
├── combine_times_files.py               # A function for processing s&p500 data
├── download.py                          # A function for download dataset from kaggle.com
├── Factor Details.xlsx                  # Details of Jensen dataset
├── README.md                            # This file
├── REPRODUCE_read_only.ipynb            # Reproduction instructions (read-only version)
├── REPRODUCE.ipynb                      # Reproduction instructions
└── usa_returns.py                       # A function for calculating Jensen dataset factor returns
NOTE: 1. Files not listed in the table above are considered intermediate files generated during computation.
      2. Due to confidentiality reasons, document CompleteMapping.parquet, SPXConstituentsDaily.parquet, SPXConstituentsPrices.parquet, and SPXPrices.parquet can not be provided, please add them all to the folder 'data' to make sure the operation of the process.
```
