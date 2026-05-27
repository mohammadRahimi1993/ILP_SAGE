#!/usr/bin/env python
# coding: utf-8

# ### Import Library
import os
import dgl
import torch
import torchmetrics
import transformers
import torcheval
import numpy as np
from collections import defaultdict
from tabulate import tabulate

os.environ['TORCH'] = torch.__version__
os.environ['DGLBACKEND'] = "pytorch"
device = torch.device("cpu")
 
try:
    import dgl
    import dgl.graphbolt as gb
    installed = True
except ImportError as error:
    installed = False
    print(error)

print("DGL installed!" if installed else "DGL not found!")
print("PyTorch Version: ", torch.__version__)
print("TorchMetrics Version: ", torchmetrics.__version__)
print("Transformers Version: ", transformers.__version__)
print("DGL Version: ", dgl.__version__)
print("TorchEval Is: ", torcheval.__version__)


# ### Import Dataset
# ILP_Date_Zip_File = '../Data/Data_InductiveLinkPrediction.zip'
# TLP_Data_Zip_File = '../Data/Data_TransductiveLinkPrediciton.zip'

# !unzip -q {ILP_Date_Zip_File} -d {'../Data'}
# !unzip -q {TLP_Data_Zip_File} -d {'../Data'}

datasets = sorted([folder for folder in os.listdir('./data') if os.path.isdir(os.path.join('./data ', folder))])                                                          
def create_dataset_dict(base_dir:str='./data'):
    datasets = {}
    for dataset_name in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset_name)
        if os.path.isdir(dataset_path):
            datasets[dataset_name] = {
                "train": os.path.join(dataset_path, "train.txt"),
                "valid": os.path.join(dataset_path, "valid.txt"),
                "test":  os.path.join(dataset_path, "test.txt")}
    return datasets

# Save Path Dictionay
ILP_dataset_paths = create_dataset_dict('./data/Data_InductiveLinkPrediction')
TLP_dataset_paths = create_dataset_dict('./data/Data_TransductiveLinkPrediciton')

# Sort Dictionary
ILP_dataset_paths = dict(sorted(ILP_dataset_paths.items()))
TLP_dataset_paths = dict(sorted(TLP_dataset_paths.items()))


# ### Save To Excel File
import os
import pandas as pd

def save_df_to_excel(df: pd.DataFrame, file_path: str, df_name: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if not os.path.exists(file_path):
        df.to_excel(file_path, sheet_name=df_name, index=False)
    else:
        with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name=df_name, index=False)


# ### Comparative Analysis of Persian and English Datasets
import os
import pandas as pd
import networkx as nx
from collections import Counter
from tabulate import tabulate

def load_data(file_path):
    sep = "," if file_path.endswith('.csv') else "\t"
    return pd.read_csv(file_path, sep=sep, header=None, names=["head", "relation", "tail"])

def analyze_graph_metrics(file_path):
    df = load_data(file_path)
    G = nx.MultiDiGraph()
    G.add_edges_from(zip(df["head"], df["tail"], df["relation"]))

    degrees = dict(G.degree())
    counter = Counter(degrees.values())
    avg_deg = sum(degrees.values()) / G.number_of_nodes() if G.number_of_nodes() else 0

    # محاسبه‌ی تعداد سه‌تایی‌ها، موجودیت‌ها و روابط
    num_triples = len(df)
    num_entities = len(set(df["head"]).union(set(df["tail"])))
    num_relations = len(set(df["relation"]))

    return {
        "Triples": num_triples,
        "Entities": num_entities,
        "Relations": num_relations,
        "Deg_1": counter.get(1, 0),
        "Deg_2": counter.get(2, 0),
        "Deg_3": counter.get(3, 0),
        "Avg_Degree": round(avg_deg, 2),
        "Density": round(nx.density(G), 6),
        "Sparsity": round(1 - nx.density(G), 6)
    }

def process_file(file_path, label):
    if os.path.isfile(file_path) and file_path.endswith(('.csv', '.txt')):
        metrics = analyze_graph_metrics(file_path)
        if metrics:
            metrics['Dataset'] = label
            return metrics
    return None

def analyze_all_datasets(all_dirs):
    results = []
    for base_dir in all_dirs:
        for root, _, files in os.walk(base_dir):
            dataset_name = os.path.basename(root)
            for file in files:
                path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()
                label_type = "CSV" if ext == '.csv' else "TXT"
                label = f"{dataset_name}_{os.path.splitext(file)[0]}"
                result = process_file(path, label)
                if result:
                    results.append(result)

    return pd.DataFrame(results)[[
        "Dataset", "Triples", "Entities", "Relations",
        "Deg_1", "Deg_2", "Deg_3", "Avg_Degree", "Density", "Sparsity"
    ]]

# Dataset Paths
all_dirs = [
    "./data/Data_InductiveLinkPrediction",
    "./data/Data_TransductiveLinkPrediciton" ]

# Analyze Tabels
df_result = analyze_all_datasets(all_dirs).sort_values("Dataset")
save_df_to_excel(df_result, "./experiment/analysis_dataset.xlsx", "Analysis_1")


# ### Comparative Analysis of Persian and English Datasets
import os
import pandas as pd
from tabulate import tabulate

def analyze_kg_files(base_dir):
    """Analyze knowledge graph files with better error handling"""
    results = []

    for dataset in sorted(os.listdir(base_dir)):
        dataset_path = os.path.join(base_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue

        stats = {'Dataset': dataset}

        for split in ['train', 'test']:
            for ext in ['.csv', '.txt']:
                file_path = os.path.join(dataset_path, f"{split}{ext}")
                if not os.path.exists(file_path):
                    continue

                try:
                    # Read file with automatic format detection
                    try:
                        # First, try reading with header
                        df = pd.read_csv(file_path)
                        # If required columns are missing, read without header
                        if not all(col in df.columns for col in ['head', 'relation', 'tail']):
                            df = pd.read_csv(file_path, sep='\t' if ext == '.txt' else ',',
                                             header=None, names=['head', 'relation', 'tail'])
                    except:
                        # If error occurs, read without header
                        df = pd.read_csv(file_path, sep='\t' if ext == '.txt' else ',',
                                         header=None, names=['head', 'relation', 'tail'])

                    # Compute basic statistics
                    stats.update({
                        f'{split}_triples': int(len(df)),
                        f'{split}_relations': int(df['relation'].nunique()),
                        f'{split}_entities': int(pd.concat([df['head'], df['tail']]).nunique())
                    })
                    break

                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    continue

        results.append(stats)

    return pd.DataFrame(results)

# Analysis of Inductive Link Prediction Datasets
base_dir = "./data/Data_InductiveLinkPrediction"
df_result = analyze_kg_files(base_dir)
save_df_to_excel(df_result, "./experiment/GraphSAGE_analysis_dataset.xlsx", "Analysis_2")

# Analysis of Transductive Link Prediction Datasets
base_dir = "./data/Data_TransductiveLinkPrediciton"
df_result = analyze_kg_files(base_dir)
save_df_to_excel(df_result, "./experiment/GraphSAGE_analysis_dataset.xlsx", "Analysis_3")       


# ### Create DGL Dataset
import dgl
import torch
import pandas as pd
from dgl.data import DGLDataset

class PersianDGLDataset(DGLDataset):
    def __init__(self, train_file, test_file, seed=42):
        self.train_file = train_file
        self.test_file = test_file
        self.seed = seed
        self.process()
        super().__init__(name="PersianLinkPrediction")

    def process(self):
        # Initialize mappings
        self.entity2id = {}
        self.relation2id = {}
        ent_id, rel_id = 0, 0

        # Process training data
        train_triples = self._load_and_process_file(self.train_file, ent_id, rel_id)
        ent_id, rel_id = len(self.entity2id), len(self.relation2id)

        # Process test data (using same mappings)
        test_triples = self._load_and_process_file(self.test_file, ent_id, rel_id)

        # Build graphs
        self.graphs = {
            "train": self._build_graph(train_triples),
            "test": self._build_graph(test_triples)
        }

    def _load_file(self, file_path):
        """Load file based on its extension"""
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.txt'):
            return pd.read_csv(file_path, sep='\t', header=None,
                             names=['subjectLabel', 'predicateLabel', 'objectLabel'])
        else:
            raise ValueError("Unsupported file format. Only .csv and .txt files are supported.")

    def _load_and_process_file(self, file_path, ent_id_start, rel_id_start):
        """Load and process a single file, updating mappings"""
        triples = []
        df = self._load_file(file_path)

        for _, row in df.iterrows():
            h, r, t = row['subjectLabel'], row['predicateLabel'], row['objectLabel']

            # Update entity mappings
            for ent in [h, t]:
                if ent not in self.entity2id:
                    self.entity2id[ent] = ent_id_start
                    ent_id_start += 1

            # Update relation mappings
            if r not in self.relation2id:
                self.relation2id[r] = rel_id_start
                rel_id_start += 1

            triples.append((
                self.entity2id[h],
                self.relation2id[r],
                self.entity2id[t]))

        return triples

    def _build_graph(self, triples):
        """Build DGL graph from triples"""
        src, rel, dst = zip(*triples)
        src = torch.tensor(src)
        dst = torch.tensor(dst)
        rel = torch.tensor(rel)

        g = dgl.graph((src, dst), num_nodes=len(self.entity2id))
        g.edata["e_type"] = rel
        g.edata["edge_mask"] = torch.ones(g.num_edges(), dtype=torch.bool)
        g.ndata["ntype"] = torch.zeros(g.num_nodes(), dtype=torch.int)
        g.ndata["feat"] = torch.randn(g.num_nodes(), 64)
        return g

    def __getitem__(self, split):
        return self.graphs[split]

    def __len__(self):
        return len(self.graphs)

class GraphBatchDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, pos_graphs, neg_graphs):
        self.graphs = graphs
        self.pos_graphs = pos_graphs
        self.neg_graphs = neg_graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return {
            "graph": self.graphs[idx],
            "pos_graph": self.pos_graphs[idx],
            "neg_graph": self.neg_graphs[idx]}


# ### Generate Positive Graph And Negative Graph
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import dgl
import scipy.sparse as sp
from tabulate import tabulate
import torch

class GraphNegativeSampler:
    def __init__(self, train_graph, test_graph, train_neg_ratio=1.0, test_neg_ratio=1.0):
        self.train_graph = train_graph
        self.test_graph = test_graph
        self.train_neg_ratio = train_neg_ratio
        self.test_neg_ratio = test_neg_ratio
        self.train_pos_g, self.train_neg_g = self._prepare_graphs(train_graph, train_neg_ratio)
        self.test_pos_g, self.test_neg_g = self._prepare_graphs(test_graph, test_neg_ratio)

    def _generate_negative_samples(self, graph):
        u, v = graph.edges()
        adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())),
                          shape=(graph.num_nodes(), graph.num_nodes()))
        return np.where(1 - adj.todense() - np.eye(graph.num_nodes()) != 0)

    def _prepare_graphs(self, graph, ratio):
        return ( self._create_positive_graph(graph),
                 self._create_negative_graph(graph, ratio))

    def _create_positive_graph(self, graph):
        g = dgl.graph(graph.edges(), num_nodes=graph.num_nodes())
        g.edata["e_type"] = graph.edata["e_type"]
        g.ndata.update({k: graph.ndata[k] for k in ["feat", "ntype"]})
        return g

    def _create_negative_graph(self, graph, ratio):
        neg_u, neg_v = self._generate_negative_samples(graph)
        num_samples = int(graph.num_edges() * ratio)
        replace = len(neg_u) < num_samples
        sample_ids = np.random.choice(len(neg_u), num_samples, replace=replace)

        g = dgl.graph((neg_u[sample_ids], neg_v[sample_ids]), num_nodes=graph.num_nodes())
        g.edata["e_type"] = torch.randint(0, graph.edata["e_type"].max().item()+1, (g.num_edges(),))
        g.ndata.update({
            "feat": graph.ndata["feat"],
            "ntype": torch.ones(graph.num_nodes(), dtype=torch.int)})
        return g

    @property
    def training_graphs(self):
        return self.train_pos_g, self.train_neg_g

    @property
    def test_graphs(self):
        return self.test_pos_g, self.test_neg_g


# ### Link Prediction Model
from dgl.nn import SAGEConv
import torch.nn as nn

class ImprovedGraphSAGE(nn.Module):
  def __init__(self, in_feats, h_feats, out_feats, dropout=0.5):
        super(ImprovedGraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, "mean")
        self.conv2 = SAGEConv(h_feats, out_feats, "mean")
        self.dropout = nn.Dropout(dropout)

  def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(g, h)
        return h


import dgl.function as fn
class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            return g.edata["score"][:, 0]


# ### Train method
from torch.utils.data import DataLoader
import itertools
from tqdm import tqdm
import dgl

def train_model(model,
                pred,
                dataloader,
                epochs,
                lr=0.01):

    optimizer = torch.optim.Adam(itertools.chain(model.parameters(),
                                                 pred.parameters()),
                                                 lr=lr)

    all_losses = []
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0.0

        for batch in dataloader:
            batch_graph = batch["graph"]  
            pos_graph = batch["pos_graph"] 
            neg_graph = batch["neg_graph"] 

            # Forward pass
            h = model(batch_graph, batch_graph.ndata["feat"])
            pos_score = pred(pos_graph, h)
            neg_score = pred(neg_graph, h)
            loss = compute_loss(pos_score,neg_score)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            e = epoch
            loss = epoch_loss

        all_losses.append(epoch_loss)

    print(f"\nEpoch: {e}, Loss: {loss:.4f}")
    return h, all_losses

def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)


# ### Train And Evaluation
import torch
import torch.nn.functional as F
import numpy as np
import itertools
from tqdm import tqdm
from dgl.dataloading import GraphDataLoader
from IPython.display import clear_output
from sklearn.metrics import average_precision_score, roc_auc_score
from tabulate import tabulate
from torchmetrics.retrieval import RetrievalMRR, RetrievalHitRate
from sklearn import metrics
import time
import psutil
import psutil, os

def get_current_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

def train_and_evaluate_model(result: dict,
                             dataset_name,
                             ILP_dataset_paths,
                             h_feats=16,
                             out_feats=10,
                             dropout=0.5,
                             epochs=2000,
                             lr=0.001,
                             train_neg_ratio=10,
                             test_neg_ratio=1):

    # Record memory status at the start
    mem_start = get_current_ram_usage()
    start_train = time.time()
    
    # ===== Step 1: Dataset Preparation =====
    graphs = PersianDGLDataset(
        train_file=ILP_dataset_paths[dataset_name]['train'],
        test_file=ILP_dataset_paths[dataset_name]['test'])

    sampler = GraphNegativeSampler(
        graphs['train'], graphs['test'],
        train_neg_ratio=train_neg_ratio,
        test_neg_ratio=test_neg_ratio)

    train_pos_g, train_neg_g = sampler.training_graphs
    test_pos_g, test_neg_g = sampler.test_graphs

    train_dataset = GraphBatchDataset([graphs['train']], [train_pos_g], [train_neg_g])
    train_loader = GraphDataLoader(train_dataset, batch_size=1, collate_fn=lambda x: x[0])

    test_dataset = GraphBatchDataset([graphs['test']], [test_pos_g], [test_neg_g])
    test_loader = GraphDataLoader(test_dataset, batch_size=1, collate_fn=lambda x: x[0])

    # ===== Step 2: Training =====
    def compute_loss(pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([
            torch.ones(pos_score.shape[0]),
            torch.zeros(neg_score.shape[0])
        ])
        return F.binary_cross_entropy_with_logits(scores, labels)

    in_feats = graphs['train'].ndata['feat'].shape[1]
    model = ImprovedGraphSAGE(
        in_feats=in_feats,
        h_feats=h_feats,
        out_feats=out_feats,
        dropout=dropout)

    pred = DotPredictor()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(pred.parameters()),
        lr=lr)
    
    for epoch in tqdm(range(epochs)):
        for batch in train_loader:
            h = model(batch['graph'], batch['graph'].ndata['feat'])
            pos_score = pred(batch['pos_graph'], h)
            neg_score = pred(batch['neg_graph'], h)
            loss = compute_loss(pos_score, neg_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    end_train = time.time()
    start_test = time.time()
    
    # ===== Step 3: Evaluation =====
    pos_scores, pos_labels = [], []
    neg_scores, neg_labels = [], []
    hit1_list, hit3_list, hit10_list = [], [], []

    with torch.no_grad():
        ranks = []
        for batch in test_loader:
            h = model(batch['graph'], batch['graph'].ndata['feat'])
            score_pos = pred(batch['pos_graph'], h).squeeze()
            score_neg = pred(batch['neg_graph'], h).squeeze()

            neg_per_pos = len(score_neg) // len(score_pos)
            pos_scores += score_pos.tolist() if score_pos.dim() > 0 else [score_pos.item()]
            neg_scores += score_neg.tolist() if score_neg.dim() > 0 else [score_neg.item()]
            pos_labels += [1] * len(score_pos) if score_pos.dim() > 0 else [1]
            neg_labels += [0] * len(score_neg) if score_neg.dim() > 0 else [0]

            for i in range(len(score_pos)):
                neg_i = score_neg[i * neg_per_pos : (i + 1) * neg_per_pos]
                rank_i = torch.sum(neg_i > score_pos[i]).item() + 1
                ranks.append(rank_i)
                hit1_list.append(1 if rank_i <= 1 else 0)
                hit3_list.append(1 if rank_i <= 3 else 0)
                hit10_list.append(1 if rank_i <= 10 else 0)

    end_test = time.time()
    
    # Record memory status at the end
    mem_end = get_current_ram_usage()
    
    # Calculate average RAM usage during this process
    avg_ram_usage = (mem_start + mem_end) / 2

    # ===== Step 4: Result Metrics =====
    result[dataset_name] = {
        "AUC": metrics.roc_auc_score(pos_labels + neg_labels, pos_scores + neg_scores),
        "AUC_PR": metrics.average_precision_score(pos_labels + neg_labels, pos_scores + neg_scores),
        "MRR": np.mean(1.0 / np.array(ranks)).item(),
        "Hit1": np.mean(hit1_list),
        "Hit3": np.mean(hit3_list),
        "Hit10": np.mean(hit10_list),
        "train_time": end_train - start_train,
        "test_time": end_test - start_test,
        "ram_gb": avg_ram_usage   # RAM
    }
    return result



from IPython.display import clear_output
from tabulate import tabulate
def display_results_table(result_dict):
    clear_output()
    headers = ['Dataset', 'AUC', 'AUC_PR', 'MRR', 'Hit1', 'Hit3', 'Hit10', 'train_time', 'test_time']
    rows = []
    for name, metrics in result_dict.items():
        row = [name] + [metrics[h] for h in headers[1:]]
        rows.append(row)
    print("\n" + tabulate(rows,
                          headers=headers,
                          tablefmt="fancy_grid",
                          floatfmt=".4f")) 


# Hyperparameter
num_runs = 10
epochs = 2000
# ### Conducting experiments on inductive datasets with a negative sample
all_results = defaultdict(list)
for run in range(num_runs):
    clear_output()
    print("\nConducting experiments on inductive datasets with a negative sample ..")
    print(f"Experiment Iteration: {run + 1}/{num_runs}")
    result = {}

    for name, path in ILP_dataset_paths.items():
            print(f"Training and evaluating the GraphSAGE model on the dataset {name}...")
            result = train_and_evaluate_model(
            result,
            name,
            ILP_dataset_paths,
            h_feats=32,
            out_feats=8,
            dropout=0.5,
            epochs=epochs,
            lr=0.001,
            train_neg_ratio=1,
            test_neg_ratio=1)

    # Store results for this run
    for dataset_name, result_metrics in result.items():
        all_results[dataset_name].append(result_metrics)


final_results = {}
for dataset_name, runs in all_results.items():
    result_metrics = runs[0].keys()  # Get metric names
    dataset_stats = {}
    for metric in result_metrics:
        values = [run[metric] for run in runs]
        dataset_stats[f"{metric}_mean"] = np.mean(values)
        dataset_stats[f"{metric}_std"] = np.std(values)
    final_results[dataset_name] = dataset_stats
    
# Convert Datafram To Dictionary    
df = pd.DataFrame.from_dict(final_results, orient="index")
df.reset_index(inplace=True)
df.rename(columns={"index": "Dataset"}, inplace=True)
save_df_to_excel(df, "./experiment/GraphSAGE_results.xlsx", "inductive_neg1_run")
print('The experiments have been completed, and the results have been saved in the Experiment folder.')


# ### Conducting Experiments on Inductive Datasets with 50 Negative Samples
all_results = defaultdict(list)
for run in range(num_runs):
    clear_output()
    print("\nConducting Experiments on Inductive Datasets with 50 Negative Samples ..")
    print(f"Experiment Iteration: {run + 1}/{num_runs}")
    result = {}

    for name, path in ILP_dataset_paths.items():
            print(f"Training and evaluating the GraphSAGE model on the dataset {name}...")
            result = train_and_evaluate_model(
            result,
            name,
            ILP_dataset_paths,
            h_feats=32,
            out_feats=8,
            dropout=0.5,
            epochs=epochs,
            lr=0.001,
            train_neg_ratio=1,
            test_neg_ratio=50)

    # Store results for this run
    for dataset_name, result_metrics in result.items():
        all_results[dataset_name].append(result_metrics)


final_results = {}
for dataset_name, runs in all_results.items():
    result_metrics = runs[0].keys()  # Get metric names
    dataset_stats = {}
    for metric in result_metrics:
        values = [run[metric] for run in runs]
        dataset_stats[f"{metric}_mean"] = np.mean(values)
        dataset_stats[f"{metric}_std"] = np.std(values)
    final_results[dataset_name] = dataset_stats
    
# Convert Datafram To Dictionary    
df = pd.DataFrame.from_dict(final_results, orient="index")
df.reset_index(inplace=True)
df.rename(columns={"index": "Dataset"}, inplace=True)
save_df_to_excel(df, "./experiment/GraphSAGE_results.xlsx", "inductive_neg50_run")  
print('The experiments have been completed, and the results have been saved in the Experiment folder.')


# ### Conducting Experiments on Transductive Datasets with 1 Negative Sample
all_results = defaultdict(list)
for run in range(num_runs):
    print("\nConducting Experiments on Transductive Datasets with 1 Negative Sample ..")
    print(f"Experiment Iteration: {run + 1}/{num_runs}")
    result = {}
    for name, path in TLP_dataset_paths.items():
            print(f"Training and evaluating the GraphSAGE model on the dataset {name}...")
            result = train_and_evaluate_model(
            result,
            name,
            TLP_dataset_paths,
            h_feats=32,
            out_feats=8,
            dropout=0.5,
            epochs=epochs,
            lr=0.001,
            train_neg_ratio=1,
            test_neg_ratio=1)

    # Store results for this run
    for dataset_name, result_metrics in result.items():
        all_results[dataset_name].append(result_metrics)


final_results = {}
for dataset_name, runs in all_results.items():
    result_metrics = runs[0].keys()  # Get metric names
    dataset_stats = {}
    for metric in result_metrics:
        values = [run[metric] for run in runs]
        dataset_stats[f"{metric}_mean"] = np.mean(values)
        dataset_stats[f"{metric}_std"] = np.std(values)
    final_results[dataset_name] = dataset_stats
    
# Convert Datafram To Dictionary    
df = pd.DataFrame.from_dict(final_results, orient="index")
df.reset_index(inplace=True)
df.rename(columns={"index": "Dataset"}, inplace=True)
save_df_to_excel(df, "./experiment/GraphSAGE_results.xlsx", "transductive_neg1_run")
print('The experiments have been completed, and the results have been saved in the Experiment folder.')


# ### Conducting Experiments on Transductive Datasets with 50 Negative Samples
all_results = defaultdict(list)
for run in range(num_runs):
    clear_output()
    print("\nConducting Experiments on Transductive Datasets with 50 Negative Samples ..")
    print(f"Experiment Iteration: {run + 1}/{num_runs}")
    result = {}

    for name, path in TLP_dataset_paths.items():
            print(f"Training and evaluating the GraphSAGE model on the dataset {name}...")
            result = train_and_evaluate_model(
            result,
            name,
            TLP_dataset_paths,
            h_feats=32,
            out_feats=8,
            dropout=0.5,
            epochs=epochs,
            lr=0.001,
            train_neg_ratio=1,
            test_neg_ratio=50)

    # Store results for this run
    for dataset_name, result_metrics in result.items():
        all_results[dataset_name].append(result_metrics)


final_results = {}
for dataset_name, runs in all_results.items():
    result_metrics = runs[0].keys()  # Get metric names
    dataset_stats = {}
    for metric in result_metrics:
        values = [run[metric] for run in runs]
        dataset_stats[f"{metric}_mean"] = np.mean(values)
        dataset_stats[f"{metric}_std"] = np.std(values)
    final_results[dataset_name] = dataset_stats
    
# Convert Datafram To Dictionary    
df = pd.DataFrame.from_dict(final_results, orient="index")
df.reset_index(inplace=True)
df.rename(columns={"index": "Dataset"}, inplace=True)
save_df_to_excel(df, "./experiment/GraphSAGE_results.xlsx", "transductive_neg50_run")  
print('The experiments have been completed, and the results have been saved in the Experiment folder.')

