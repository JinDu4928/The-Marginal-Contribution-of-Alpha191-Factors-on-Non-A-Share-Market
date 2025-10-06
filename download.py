from kaggle.api.kaggle_api_extended import KaggleApi

def download_usa_csv():

    api = KaggleApi()
    api.authenticate()

    dataset = 'jindu4928/usa-csv'
    download_path = './data/'

    api.dataset_download_files(dataset, path=download_path, unzip=True)
    print("usa.csv dataset download completed. Files have been extracted to the 'data' folder.")

def download_period_data():

    api = KaggleApi()
    api.authenticate()

    dataset = 'jindu4928/period'
    download_path = './period_split/'

    api.dataset_download_files(dataset, path=download_path, unzip=True)
    print("period0 to period5 dataset download completed. Files have been extracted to the 'period_split' folder.")
