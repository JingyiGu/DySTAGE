import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

######### read data
data_dir = "./data/raw_asset/raw_data.pkl"
df = pd.read_pickle(data_dir)
print("shape of raw data: ", df.shape)
# df.head()

permno = df["permno"].drop_duplicates() 
date = df["DATE"].drop_duplicates()
date = date.sort_values(ascending=True)

# select date: data from 2000 to 2021
mask = (df['DATE'] >= '2000-1-1') & (df['DATE'] <= '2021-12-31')
df_predictors_after_2000 = df.loc[mask]

# sort by stock and date
df_predictors_after_2000 = df_predictors_after_2000.sort_values(by=['permno', 'DATE'])
print("data shape after 2000:", df_predictors_after_2000.shape) 


############################################################################
################### step 1: missing values ###################
############################################################################

# for each stock, for each feature, fill missing data by forward fill
for i in df_predictors_after_2000.columns:
    df_predictors_after_2000[i] = df_predictors_after_2000.groupby("permno")[i].transform(lambda v: v.ffill())

# delete the stock x with >50% missing features
gg = df_predictors_after_2000.groupby("permno")
d = {}
del_permno = []
# for each stock x and its data g
for x, g in gg:
    l = []
    # i is the feature that all entries are missing
    l = [i for i in g.columns if g[i].isna().mean()==1]
    # dictionary stock: missing features
    d[x] = l
    # if the stock x has 50% features are missing then delete it
    if len(l) > (0.5*df_predictors_after_2000.shape[1]):
        del_permno.append(x)
        
# delete the above stocks
df_predictors_after_2000 = df_predictors_after_2000[~(df_predictors_after_2000.permno.isin(del_permno))]
print("shape of data after deleting stock with 50% missing features:", df_predictors_after_2000.shape)
# delete rows containing either 70% or more than 70% NaN values
perc = 70.0 # N is 70
min_count = int(((100-perc)/100)*df_predictors_after_2000.shape[1] + 1)
df_predictors_after_2000 = df_predictors_after_2000.dropna(axis=0, thresh=min_count)
print(min_count)

# delete columns containing either 50% or more than 50% NaN values
perc = 50.0 
min_count = int(((100-perc)/100)*df_predictors_after_2000.shape[0] + 1)
df_predictors_after_2000 = df_predictors_after_2000.dropna(axis=1, thresh=min_count)
print(min_count)

# str to float
for i in [i for i in df_predictors_after_2000.columns if df_predictors_after_2000[i].dtypes == "object"]:
    df_predictors_after_2000[i] = df_predictors_after_2000[i].str[:1].astype(float)

# for each feature calculate class median and fill with median
gg = df_predictors_after_2000.groupby("DATE")
d = {}
del_permno = []
# for each date x and its data g
for x, g in gg:
    for i in g.columns:
        median = g[i].median()
        df_predictors_after_2000[i][df_predictors_after_2000["DATE"]==x] = \
            df_predictors_after_2000[i][df_predictors_after_2000["DATE"]==x].fillna(median)

path = "./data/preprocess/"
os.makedirs(path, exist_ok=True)
df_predictors_after_2000.to_csv("./data/preprocess/pre-processed.csv", index=False) # (1014381, 169)

############################################################################
################### step 2: filter by time ###################
############################################################################

data_pre_dir = "./data/preprocess/pre-processed.csv"
df_predictors_after_2000 = pd.read_csv(data_pre_dir, parse_dates=["DATE"])

def check_num_stocks(df):
    return len(df.permno.drop_duplicates())

# delete stocks that have less than 15 years data, 
num_month = 12 * 15 # total # months = 264
del_permno = []
for i in df_predictors_after_2000.permno.drop_duplicates():
    if sum(df_predictors_after_2000.permno == i) < num_month:
        del_permno.append(i)
print("# stocks with less than 10 years to be deleted: ", len(del_permno)) # 
df_predictors_after_2000 = df_predictors_after_2000[~(df_predictors_after_2000.permno.isin(del_permno))]
print("# stocks:", check_num_stocks(df_predictors_after_2000))                                                                                           

# check stocks that disconnected 
dct = {}
for i in df_predictors_after_2000.permno.drop_duplicates():
    stock = df_predictors_after_2000[df_predictors_after_2000["permno"] == i]
    start_date, end_date = stock["DATE"].iloc[0], stock["DATE"].iloc[-1]
    diff = (end_date.year * 12 + end_date.month) - (start_date.year * 12 + start_date.month)
    if (diff + 1) != len(stock):
        dct[i] = (diff + 1) - len(stock)


del_permno = list(dct.keys())
df_predictors_after_2000 = df_predictors_after_2000[~(df_predictors_after_2000.permno.isin(del_permno))]
print("# disconnected stocks to be deleted: ", len(del_permno)) # 
                                                                                           
print("# stocks:", check_num_stocks(df_predictors_after_2000))                                                                                           

def check_missing(df):
    print("total number of missing data:", df.isna().sum().sum())
    print("number of missing data per feature:", df.isna().sum())

check_missing(df_predictors_after_2000)

sum(abs(df_predictors_after_2000["exret_rf"]) > 0.2) / len(df_predictors_after_2000)

df_predictors_after_2000.to_csv("./data/preprocess/pre-processed_15.csv", index=False)



############################################################################
################### step 3: explorary analysis on graph ###################
############################################################################


data_dir = "./data/preprocess/pre-processed_15.csv"
df_after_2000 = pd.read_csv(data_dir, parse_dates=["DATE"])

def check_num_stocks(df):
    return len(df.permno.drop_duplicates())

check_num_stocks(df_after_2000)

def node2index(unique_nodes):
    """
    Returns two dictionaries for mapping nodes to indices and vice versa.

    Parameters:
        unique_nodes (array-like): An array-like object containing unique node IDs.

    Returns:
        tuple: A tuple of two dictionaries. The first dictionary maps node IDs to indices,
        and the second dictionary maps indices to node IDs.
    """
    node2index_dct = {}
    index2node_dct = {}
    for i, node in enumerate(unique_nodes):
        node2index_dct[node] = i
        index2node_dct[i] = node
    return node2index_dct, index2node_dct



def get_invalid_nodes(df, unique_nodes, seq_len):
    """
    Finds nodes that have less data than the specified sequence length.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        unique_nodes (array-like): An array-like object containing unique node IDs.
        seq_len (int): The sequence length.

    Returns:
        list: A list of node IDs with less data than the sequence length.
    """
    invalid = []
    for node in unique_nodes:
        if len(df.loc[df["permno"] == node, "exret_rf"].values) < seq_len:
            invalid.append(node)
    return invalid

# Get unique nodes and dates
unique_nodes = df_after_2000["permno"].drop_duplicates().sort_values().values
num_nodes = len(unique_nodes)
unique_dates = df_after_2000["DATE"].drop_duplicates().sort_values().values
num_dates = len(unique_dates)

# Create dictionaries for mapping nodes to indices and vice versa
node2index_dct, index2node_dct = node2index(unique_nodes)

# Print the number of nodes and dates
print(f"Number of nodes: {num_nodes}, number of dates: {num_dates}") # 3317

# Select the relevant columns of the DataFrame
df_return = df_after_2000.loc[:, ["permno", "DATE", "exret_rf"]]

# Check number of invalid nodes over time
seq_len = 3 * 12
invalids = []

for t in range(num_dates - seq_len):
    temp_dates = unique_dates[t: t + seq_len]
    mask = df_return["DATE"].isin(temp_dates)
    df_temp = df_return.loc[mask]
    invalid = get_invalid_nodes(df_temp, unique_nodes, seq_len)
    invalids.append(invalid)

# Calculate the number of invalid nodes for each timestamp
num_invalids = []
for invalid in invalids:
    num_invalids.append(len(invalid))

# Save the num_invalids variable to a file
np.savetxt("./data/preprocess/num_invalids.txt", num_invalids)


# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 2))

# Plot the number of invalid stocks over time
ax1.plot(unique_dates[seq_len:], num_invalids)
ax1.set_title("Number of Invalid Stocks (Less Than 3 Years) Over Time")
ax1.set_xlabel("Date")
ax1.set_ylabel("Number of Invalid Stocks")
ax1.grid(True)

# Save the plot to a file
ax1.figure.savefig("./data/preprocess/invalid-stocks.svg")

# Plot the number of valid stocks over time
ax2.plot(unique_dates[seq_len:], num_nodes - np.array(num_invalids))
ax2.set_title("Number of Valid Stocks over Time (3-Year Period)")
ax2.set_xlabel("Date")
ax2.set_ylabel("Number of Valid Stocks")
ax2.grid(True)

# Save the plot to a file
ax2.figure.savefig("./data/preprocess/valid-stocks.svg")


