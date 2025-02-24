# DySTAGE: Dynamic Graph Representation Learning for Asset Pricing via Spatio-Temporal Attention and Graph Encodings

We introduce the source code of our paper "[DySTAGE: Dynamic Graph Representation Learning for Asset Pricing via Spatio-Temporal Attention and Graph Encodings](https://dl.acm.org/doi/pdf/10.1145/3677052.3698680)" (ICAIF 2024). 

![framework](https://github.com/user-attachments/assets/d8b4a272-9853-4668-8b46-fdcadd4d3af6)


DySTAGE is a novel dynamic graph representation learning framework designed to enhance asset pricing prediction by capturing the evolving interrelationships among financial assets. Unlike traditional models that rely on static graphs or time series data, DySTAGE employs dynamic graph structures to represent the temporal and spatial dependencies of assets in financial networks. This enables the model to adapt to changes in asset composition and their correlations over time.

## Key Features
- **Dynamic Graph Construction**: The framework dynamically constructs graphs from time series data to reflect the evolving asset interrelationships, accommodating asset additions and removals over time.
- **Spatio-Temporal Attention**: DySTAGE employs a combination of spatial and temporal attention mechanisms to capture both the topological (structural) and temporal (historical) patterns of asset behavior.
- **Graph Encodings**: Utilizes financial insights to design graph encodings that improve model performance, including importance, spatial, and edge correlation encodings.
- **Performance Optimization**: Demonstrates superior predictive accuracy and portfolio management outcomes over popular benchmarks, offering profitable investment strategies based on asset price prediction.

## Installation

First, ensure you have the required dependencies. You can install them using the command below:
```
pip install -r requirements.txt
```

## Data Preparation

### Data Availability
Due to confidentiality, we cannot provide the dataset used in our paper. However, you can prepare your own dataset and store the raw data in `data/raw_asset/raw_data.pkl`. It should be a dataset representing daily or monthly asset prices. The dataset must include the following columns:
- `DATE`: the timestamp of each record
- `permno`: the unique identifier for each asset
- `exret_rf`: the excess return (risk free) as the target
- Additional feature columns

### Data Processing
Once the raw data is prepared, you can process it using following scripts to construct the graph representation and generate necessary files for model training:
```
python preprocess.py
python build_graph.py
```

### Generated Data Files

- **graph.pkl**: A list of tensors representing the adjacency matrix of graph at each time step. Each matrix has a shape of `(N, N)`, where `N` is the total number of nodes/assets. The element of the matrix is the pearson correlation of excess return between two assets, calculated over past 36 months for monthly data or 61 trading days for daily data.
  - If the node does not exist in a particular time step, all its connections are set to `0`.
- **features.pkl**: A historical list of feature matrices, each of size `(N, F)`, where `F` is the feature dimension. Each feature matrix is stored in csr matrix.     - For non-existent nodes, features are set to `0` at the respective time step.
- **edge_feat.pkl**: A historical list of multi-scale edge attributes, each of size `(N, N, scale)`, where `scale` is the number of scales, 
- **shortest_path.pt**: A torch sensor in the shape of `(N, N, N)`, representing the shortest path between all pairs of nodes across the graph.

### Example Data Structure
Your folder structure should look like this:
```
data/
    {data_name}/
        graph.pkl
        features.pkl
        edge_feat.pkl
        shortest_path.pt
```

## Training
To train DySTAGE, run the following command:
```
python main.py --dataset {data_folder_name}
```
During training, the best model along with predictions and labels will be saved to the `result` directory.

## Citation
If you use this code for your research, please kindly cite our paper:
```
@inproceedings{gu2024dystage,
  title={DySTAGE: Dynamic Graph Representation Learning for Asset Pricing via Spatio-Temporal Attention and Graph Encodings},
  author={Gu, Jingyi and Ye, Junyi and Uddin, Ajim and Wang, Guiling},
  booktitle={Proceedings of the 5th ACM International Conference on AI in Finance},
  pages={388--396},
  year={2024}
}
```
