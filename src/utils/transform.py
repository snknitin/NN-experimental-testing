import torch_geometric.transforms as T
from collections import defaultdict
import torch
from torch_geometric.loader import DataLoader

from typing import List, Union

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform



@functional_transform('revlabel_delete')
class RevDelete(BaseTransform):
    r"""Deletes the edge_label on reverse edges.
    Args:
        attrs (List[str]): The names of attributes to consider.
            (default: :obj:`["edge_attr"]`)
    """
    def  __init__(self, attrs: List[str] = ["edge_attr"]):
        self.attrs = attrs

    def __call__(self, data: Union[Data, HeteroData]):
        for edge,store in list(zip(data.edge_types,data.edge_stores)):
            if edge[1].startswith("rev"):
                if "edge_label" in store.keys():
                    del store["edge_label"]
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


@functional_transform('scale_edges')
class ScaleEdges(BaseTransform):
    r"""Column-normalizes the attributes given in :obj:`attrs` to standardize or scale them.

    Mainly useful for edges, if the scalar for the collective data edges is stored
    as data[(node1,to,node2)].edge_scaler = scaler

    where the scaler can be StandardScaler or MinMaxScaler from scikit-learn that
    has been fit to the total dataset

    Args:
        attrs (List[str]): The names of attributes to normalize.
            (default: :obj:`["edge_attr"]`)
    """
    def __init__(self, attrs: List[str] = ["edge_attr"]):
        self.attrs = attrs

    def __call__(self, data: Union[Data, HeteroData]):
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                if "edge_scaler" in store.keys():
                    scaler = store["edge_scaler"]
                    # needs to be a float tensor since this is the edge_attr
                    value = torch.tensor(scaler.transform(value),dtype=torch.float)
                    store[key] = value
                    del store["edge_scaler"]
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


def getfullstats(dailydata):
    """
    Pass the data into a loader to get the full stats
    Do this before Transforms that add reverse edges
    Create a transform for this if you want to limit edge columns
    """
    loader = DataLoader(dailydata, batch_size=len(dailydata))
    data = next(iter(loader))
    edge_stats = defaultdict(dict)
    for edge in data.edge_types:
        e = data[edge].edge_attr
        # get stats
        edge_stats[edge]["mean"] = e.mean(dim=0, keepdim=True)
        edge_stats[edge]["std"] = e.std(dim=0, keepdim=True)
        edge_stats[edge]["min"] = torch.min(e, 0).values
        edge_stats[edge]["max"] = torch.max(e, 0).values
    return edge_stats

def edge_norm_fn(data):
    """
    Normalization of edge_attributes
    """
    edge_stats = getfullstats(data)
    for edge in data[0].edge_types:
        xmean = edge_stats[edge]["mean"]
        xstd = edge_stats[edge]["std"]
        xmin,xmax = edge_stats[edge]["min"],edge_stats[edge]["max"]
        for i in range(len(data)):
            x = data[i][edge].edge_attr
            data[i][edge].edge_attr = (x-xmean)/(xmax-xmin)
    return data
