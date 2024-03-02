from pandas import read_csv

def read(size,dataset_path,duplicated_subset):
    df = read_csv(dataset_path)
    df = df[0:size]
    df.drop_duplicates(subset=duplicated_subset, inplace=True)
    df.reset_index(drop = True, inplace = True)
    return df