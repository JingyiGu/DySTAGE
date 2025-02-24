
import numpy as np
import pandas as pd
import os
import pickle as pkl
import networkx as nx
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from utils.utilities import *

data_dir = "./data/preprocess/pre-processed_15.csv"
df_after_2000 = pd.read_csv(data_dir, parse_dates=["DATE"])

upper_bound = 0.95
lower_bound = 0.05

upper_quantile = np.quantile(df_after_2000.exret_rf, upper_bound)
lower_quantile = np.quantile(df_after_2000.exret_rf, lower_bound)

df_after_2000.exret_rf.loc[df_after_2000.exret_rf > upper_quantile] = upper_quantile
df_after_2000.exret_rf.loc[df_after_2000.exret_rf < lower_quantile] = lower_quantile

############################################################################
################### step 1: build adjacency matrix ###################
############################################################################



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

def get_weighted_adjacency_matrix(df, unique_nodes, seq_len, filter = 0.3):
    """
    Computes the adjacency matrix of the graph.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        unique_nodes (array-like): An array-like object containing unique node IDs.
        seq_len (int): The sequence length.

    Returns:
        numpy.ndarray: The adjacency matrix of the graph.
    """
    # check if stocks have data of whole sequence length
    invalid = get_invalid_nodes(df, unique_nodes, seq_len)
    valid = np.setdiff1d(unique_nodes, invalid, assume_unique=True)
    num_valid = len(valid)
    # remove invalid nodes
    df_cleaned = df[~(df["permno"].isin(invalid))]
    # calculate correlation
    X = np.zeros((num_valid, seq_len))
    adj = np.eye(len(unique_nodes), dtype=float)
    node2index_dct, _ = node2index(unique_nodes)
    # print(node2index_dct)
    for i, node in enumerate(valid):
        X[i] = df_cleaned.loc[df_cleaned["permno"] == node, "exret_rf"].values
    corr = np.corrcoef(X)
    for i in range(num_valid):
        for j in range(i+1, num_valid):
            if abs(corr[i,j]) > filter:
                adj[node2index_dct[valid[i]], node2index_dct[valid[j]]] = corr[i,j]
                adj[node2index_dct[valid[j]], node2index_dct[valid[i]]] = corr[i,j]
    return adj

# Get unique nodes and dates
unique_nodes = df_after_2000["permno"].drop_duplicates().sort_values().values
num_nodes = len(unique_nodes)
unique_dates = df_after_2000["DATE"].drop_duplicates().sort_values().values
num_dates = len(unique_dates)

# Create dictionaries for mapping nodes to indices and vice versa
node2index_dct, index2node_dct = node2index(unique_nodes)

# Print the number of nodes and dates
print(f"Number of nodes: {num_nodes}, number of dates: {num_dates}") # 2151, 264

# Select the relevant columns of the DataFrame
df_return = df_after_2000.loc[:, ["permno", "DATE", "exret_rf"]]

seq_len = 3 * 12
graphs = []

# Generate adjacency matrices for each 3 years period
for t in range(num_dates - seq_len):
    date_range = unique_dates[t : t + seq_len]
    mask = df_return['DATE'].isin(date_range) 
    df_return_temp = df_return.loc[mask]
    adj_matrix = get_weighted_adjacency_matrix(df_return_temp, unique_nodes, seq_len)
    graphs.append(sp.coo_matrix(adj_matrix,dtype=np.float32))

# Save to a pickle file
output_file = './data/asset/graph.pkl'
with open(output_file, 'wb') as f:
    pkl.dump(graphs, f, protocol=4)



############################################################################
################### step 2: build feature matrix ###################
############################################################################

def get_feature(df: pd.DataFrame, valid: list) -> np.ndarray:
    """
    Extracts features from a DataFrame and returns them as a numpy array.

    Args:
        df (pd.DataFrame): The DataFrame containing the features.
        valid (list): A list of valid nodes.

    Returns:
        np.ndarray: A numpy array containing the extracted features.
    """
    # Remove permno and date columns
    num_features = df.shape[1] - 2 
    inputs = np.zeros((num_nodes, num_features))
    # Get indices of valid nodes
    valid_idx = [node2index_dct[i] for i in valid]
    # Extract features (except for permon and date) and assign to inputs array
    inputs[valid_idx] = df.iloc[:, 2:].values
    return inputs


def prepare_data(df) -> list:
    """
    Prepares the data for each time stamp and returns a list of inputs and invalid nodes.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        unique_dates (list): A list of unique dates.
        unique_nodes (list): A list of unique nodes.
        node2index_dct (dict): A dictionary mapping nodes to indices.
        num_nodes (int): The total number of nodes.

    Returns:
        list: A list of inputs and invalid nodes.
    """
    invalids = []
    inputs_list = []
    features = []
    for temp_date in unique_dates:
        mask = df["DATE"] == temp_date
        df_temp = df.loc[mask]
        valid = df_temp["permno"].values
        invalid = np.setdiff1d(unique_nodes, valid, assume_unique=True)
        inputs = get_feature(df_temp, valid)
        invalids.append(invalid)
        inputs_list.append(inputs)
        sparse_matrix = csr_matrix(inputs)
        features.append(sparse_matrix)
    return inputs_list, invalids, features


inputs_list, invalids, features = prepare_data(df_after_2000)
print('shape of features in each time step: ', inputs_list[0].shape)

output_file = './data/asset/features.pkl'

# Save to a pickle file
with open(output_file, 'wb') as f:
    pkl.dump(features, f, protocol=4)




############################################################################
########## step 3: generate shortest path inlcuding node ##########
############################################################################

# edge path
import torch_geometric as tg
from torch_geometric.data import Data
import torch
from torch_geometric.utils.convert import to_networkx
import pickle as pkl
import networkx as nx
import scipy.sparse as sp

def load_graphs(dataset_str):
    """Load graph snapshots given the name of dataset"""
    with open("data/{}/{}".format(dataset_str, "graph.pkl"), "rb") as f:
        graphs = pkl.load(f)
    print("Loaded {} graphs ".format(len(graphs)))
    return graphs

def load_features(dataset_str):
    """Load feature vectors given the name of dataset"""
    with open(f"data/{dataset_str}/features.pkl", "rb") as f:
        features = pkl.load(f)
    return features[36:]

def normalize_adj(adj):
    rowsum = np.array(adj.sum(1))
    r_inv = sp.diags(np.power(rowsum, -1).flatten(), dtype=np.float32)
    adj_normalized = r_inv.dot(adj)
    return adj_normalized

adjs = load_graphs('asset')
features = load_features('asset')
adjs = [normalize_adj(a) for a in adjs]

# shortest path length
shortest_paths = []
for idx in range(len(adjs)):
    x = torch.Tensor(np.array(features[idx].todense()))
    edge_index, edge_weight = tg.utils.from_scipy_sparse_matrix(adjs[idx])
    G = to_networkx(Data(x=x,edge_index=edge_index, edge_weight=edge_weight))
    shortest_path_len = nx.floyd_warshall_numpy(G, weight=None)
    shortest_path_len[shortest_path_len==np.inf] = 10
    shortest_paths.append(shortest_path_len)
    print('finish shortest path at time step', idx)

paths_len = torch.Tensor(shortest_paths)
output_file = './data/asset/shortest_paths.pt'
torch.save(paths_len,output_file)


############################################################################
########## step 5: generate multi-scale corr as edge attribute #############
############################################################################

def get_scale_corr(t, historical_step, filter = 0.3):
    date_range = unique_dates[t - historical_step : t]
    mask = df_return['DATE'].isin(date_range) 
    df_return_temp = df_return.loc[mask]
    corr_matrix = get_weighted_adjacency_matrix(df_return_temp, unique_nodes, historical_step, filter)
    return corr_matrix

scales = [3, 6, 12, 24, 36]
edge_attr = []
seq_len = 3 * 12

# Generate adjacency matrices for each 3 years period
for t in range(seq_len, num_dates):
    edge_attr_scale = []
    adj = get_scale_corr(t, seq_len)
    edge_mask = adj!=0
    for scale in scales:
        corr_scale = get_scale_corr(t, scale, 0)
        c = np.where(edge_mask, corr_scale, 0)
        edge_attr_scale.append(c)
    edge_attr.append(np.stack(edge_attr_scale, axis=2))


# Save to a pickle file
output_file = './data/asset/edge_feat.pkl'
with open(output_file, 'wb') as f:
    pkl.dump(edge_attr, f, protocol=4)
