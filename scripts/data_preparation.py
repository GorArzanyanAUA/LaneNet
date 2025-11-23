import kagglehub



def download_data():
    # Download latest version
    path = kagglehub.dataset_download("manideep1108/tusimple")

    print("Path to dataset files:", path)


if __name__=="__main__":
    download_data()


