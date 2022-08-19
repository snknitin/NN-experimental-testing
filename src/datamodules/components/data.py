import torch
import os
import os.path as osp
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import InMemoryDataset,Dataset
from torch_geometric.data import HeteroData
import pandas as pd
from tqdm import tqdm

from src.utils import transform as tnf
import torch_geometric.transforms as T
import random
import shutil
random.seed(42)
torch.manual_seed(3407)
np.random.seed(0)

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class IdentityEncoder(object):
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)

class YearlyData(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        return ['file1']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        ...

    def get_edges(self, x, y):
        src = np.random.randint(0, x, 3500)
        dest = np.random.randint(0, y, 3500)
        return [src, dest]

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        g1 = HeteroData()
        g1["node1"].x = torch.tensor(np.round(np.random.rand(1250, 6) * 10), dtype=torch.float)
        g1["node2"].x = torch.tensor(np.round(np.random.rand(2000, 6) * 20), dtype=torch.float)
        g1["node3"].x = torch.tensor(np.round(np.random.rand(100, 6) * 10), dtype=torch.float)

        g1['node2', 'to', 'node1'].edge_index = torch.tensor(self.get_edges(2000, 1250), dtype=torch.long)
        g1['node3', 'to', 'node2'].edge_index = torch.tensor(self.get_edges(100, 2000), dtype=torch.long)

        g1['node2', 'to', 'node1'].edge_attr = torch.tensor(np.round(np.random.rand(3500, 6) * 10), dtype=torch.float)
        g1['node3', 'to', 'node2'].edge_attr = torch.tensor(np.round(np.random.rand(3500, 5) * 10), dtype=torch.float)

        g1["node2", "to", "node1"].edge_label = torch.rand((3500, 1))
        g1["node3", "to", "node2"].edge_label = torch.rand((3500, 1))

        node_types, edge_types = g1.metadata()
        for node_type in node_types:
            g1[node_type].num_nodes = g1[node_type].x.size(0)

        data_list.append(g1)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class DailyData(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root

    @property
    def raw_dir(self):
        return os.path.join(self.root,'raw')

    @property
    def raw_file_names(self):
        edge_dir = os.path.join(self.raw_dir,"relations")
        files = sorted(os.listdir(edge_dir))
        return [os.path.join(edge_dir,f) for f in files]

    @property
    def processed_file_names(self):
        return ['data_{0}.pt'.format(x) for x in range(30)]

    def download(self):
        # Download to `self.raw_dir`.
        ...


    def get_scalers(self,edge_cols):
        """
        Scaling the edge attributed based on the whole data for consistency
        we need to return scaler and perform the transform
        """

        def f(i):
            return pd.read_csv(i, usecols=edge_cols, low_memory=True)

        # Read edge_cols from all raw files
        df = pd.concat(map(f, self.raw_file_names))
        scaler = MinMaxScaler()
        scaler.fit(df.to_numpy())
        del df
        return scaler


    def load_full_node_csv(self, featpath, idxpath):
        """
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        node_dir = osp.join(self.raw_dir, "node-features/")
        df = pd.read_csv(os.path.join(node_dir,featpath),header=None)
        map_df = pd.read_csv(os.path.join(node_dir,idxpath))
        mapping = dict(zip(map_df["name"], map_df["index"]))
        x = torch.tensor(df.values, dtype=torch.float)
        return x, mapping

    def get_new_ids(self, src, dst):
        """
        Converting the fulldata index mapping into daily index
        This will require only a subset of nodes from the full files
        Based on that subset, the index mapping of each edge will change to suit it
        """
        # Gets the list of unique idx of src and dst from the given list of the file
        # Convert to tensort to run unique and sort, then convert to lsit for enumerate
        src_extract = torch.tensor(src).unique().numpy().tolist()
        dst_extract = torch.tensor(dst).unique().numpy().tolist()

        # Create a 0-n mapping for daily graph
        new_src_ids = {k: i for i, k in enumerate(src_extract)}
        new_dst_ids = {k: i for i, k in enumerate(dst_extract)}

        # Get updated list of idx as it would be from day graphs nodes
        new_src = [new_src_ids[x] for x in src]
        new_dst = [new_dst_ids[x] for x in dst]

        # Create daily graph edge index
        edge_index = torch.tensor([new_src, new_dst])

        return edge_index, src_extract, dst_extract

    def load_edge_csv(self, edge_file_path, edge_cols, src_index_col, src_mapping,
                      dst_index_col, dst_mapping, encoders=None):
        """
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """
        df = pd.read_csv(edge_file_path)
        # src = [src_mapping[index] for index in df[src_index_col]]
        # dst = [dst_mapping[index] for index in df[dst_index_col]]
        src = []
        dst = []
        for index, row in df.iterrows():
            try:
                s = src_mapping[row[src_index_col]]
                d = dst_mapping[row[dst_index_col]]
            except:
                df.drop(index, inplace=True)
                # print("Missed a key")
                continue
            src.append(s)
            dst.append(d)

        # Updates edge indices for dailygraph based on node index to extract
        edge_index, src_extract, dst_extract = self.get_new_ids(src, dst)

        edge_attr = df[edge_cols]
        edge_attr = torch.tensor(edge_attr.values, dtype=torch.float)

        edge_label = None
        if encoders is not None:
            edge_label = [encoder(df[col]) for col, encoder in encoders.items()]
            edge_label = torch.cat(edge_label, dim=-1)

        return src_extract, dst_extract, edge_index, edge_attr, edge_label


    def process(self):
        print("Creating the data processed files for the first time")
        edge_dir = os.path.join(self.raw_dir, 'relations/')

        node1_feat, node1_mapping = self.load_full_node_csv("ship_feat.csv","ship_mapping.csv")
        node2_feat, node2_mapping = self.load_full_node_csv("cust_feat.csv","cust_mapping.csv")
        node3_feat, node3_mapping = self.load_full_node_csv("prod_feat.csv","prod_mapping.csv")

        cop_edge_cols = ["cop1","cop2","cop3","lab1"]
        sdc_edge_cols = ["sdc1","sdc2","sdc3","sdc4"]

        cop_scaler = self.get_scalers(edge_cols=cop_edge_cols)
        sdc_scaler = self.get_scalers(edge_cols=sdc_edge_cols)

        #sdc_encoder = None
        cop_encoder = None

        sdc_encoder = {'lab2': IdentityEncoder(dtype=torch.long)}
        #cop_encoder = {'net_qty': IdentityEncoder(dtype=torch.long)}

        # Create data object
        idx = 0  # keep track of data files
        file_list = self.raw_file_names
        for file_name in tqdm(file_list):
            edge_file_path = file_name
            data = HeteroData()
            sdc_ship_extract, sdc_cust_extract, sdc_edge_index, sdc_edge_attr, sdc_edge_label = self.load_edge_csv(
                edge_file_path,
                edge_cols=sdc_edge_cols,
                src_index_col="ship_name",
                src_mapping=node1_mapping,
                dst_index_col="cust_name",
                dst_mapping=node2_mapping,
                encoders=sdc_encoder)

            cop_cust_extract, cop_prod_extract, cop_edge_index, cop_edge_attr, cop_edge_label = self.load_edge_csv(
                edge_file_path,
                edge_cols=cop_edge_cols,
                src_index_col="cust_name",
                src_mapping=node2_mapping,
                dst_index_col="prod_name",
                dst_mapping=node3_mapping,
                encoders=cop_encoder)
            # Check if nodes ids to extract are equal from both edges
            # assert(cop_cust_extract==sdc_cust_extract)
            for edge_type in [('node2', 'to', 'node3'), ('node1', 'to', 'node2')]:
                if edge_type == ('node2', 'to', 'node3'):
                    data['node2', 'to', 'node3'].edge_index = cop_edge_index  # [2, num_edges_orders]
                    data['node2', 'to', 'node3'].edge_attr = cop_edge_attr
                    if cop_edge_label is not None:
                        data['node2', 'orders', 'node3'].edge_label = cop_edge_label  # [num_edges,1]
                    data['node2', 'to', 'node3'].edge_scaler = cop_scaler

                else:
                    data['node1', 'to', 'node2'].edge_index = sdc_edge_index  # [2, num_edges_delivered]
                    data['node1', 'to', 'node2'].edge_attr = sdc_edge_attr
                    if sdc_edge_label is not None:
                        data['node1', 'to', 'node2'].edge_label = sdc_edge_label  # [num_edges,1]
                    data['node1', 'to', 'node2'].edge_scaler = sdc_scaler


            data["node1"].x = node1_feat[sdc_ship_extract]
            data["node2"].x = node2_feat[sdc_cust_extract]
            data["node3"].x = node3_feat[cop_prod_extract]

            node_types, edge_types = data.metadata()
            for node_type in node_types:
                data[node_type].num_nodes = data[node_type].x.size(0)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data


if __name__ == '__main__':
    root = osp.join(os.getcwd(), "data/dailyroot")
    proc_path = os.path.join(root, 'processed')
    if os.path.exists(proc_path):
        shutil.rmtree(os.path.join(root,'processed'))

    if os.path.exists(proc_path):
        shutil.rmtree(os.path.join(root,'processed'))

    transform = T.Compose([tnf.ScaleEdges(attrs=["edge_attr"]),
                            T.NormalizeFeatures(attrs=["x", "edge_attr"]),
                            T.ToUndirected(),
                            T.AddSelfLoops(),
                            tnf.RevDelete()])

    data2 = DailyData(root, transform=transform)
    #print(data2.metadata())
    nd2 = data2[1]
    print(nd2.metadata())

