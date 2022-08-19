from typing import Dict, List, Any, Optional, Tuple

import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from src.utils import transform as tnf
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.typing import EdgeType, NodeType
import pytorch_lightning as pl

from src.datamodules.components.data import DailyData
import hydra
import omegaconf
import pyrootutils


class IdentityEncoder(object):
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)


class GraphDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = "/dailydata",
                 train_val_test_split= [0.80,0.10,0.10],
                 batch_size: int = 64,
                 num_workers: int = 0,
                 pin_memory: bool = False):
        super().__init__()
        self.data_dir = data_dir
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.transform = T.Compose([tnf.ScaleEdges(attrs=["edge_attr"]),
                            T.NormalizeFeatures(attrs=["x", "edge_attr"]),
                            T.ToUndirected(),
                            T.AddSelfLoops(),
                            tnf.RevDelete()])

    def prepare_data(self) -> None:
        # Download logic or first time prep
        # It is not recommended to assign state here (e.g. self.x = y).
        DailyData(self.hparams.data_dir,transform=self.transform)

    def setup(self, stage: Optional[str] = None) -> None:
        # data operations you might want to perform on every GPU

        data = DailyData(self.hparams.data_dir,transform=self.transform)
        self.metadata = data[0].metadata()
        n = (len(data) + 9) // 10
        lengths = self.hparams.train_val_test_split
        if stage=="fit" or stage is None:
            self.train_data = data[:-2 * n]
            self.val_data = data[-2 * n: -n]
        if stage =="test" or stage is None:
            self.test_data = data[-n:]

    def metadata(self) -> Tuple[List[NodeType], List[EdgeType]]:
        node_types = ['node1', 'node2', 'node3']
        edge_types = [('node2', 'to', 'node3'),
                      ('node1', 'to', 'node2'),
                      ('node3', 'rev_to', 'node2'),
                      ('node2', 'rev_to', 'node1')]
        return node_types, edge_types

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_data,batch_size=self.hparams.batch_size,num_workers=self.hparams.num_workers,pin_memory=self.hparams.pin_memory, shuffle=False)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_data,batch_size=self.hparams.batch_size,num_workers=self.hparams.num_workers,pin_memory=self.hparams.pin_memory, shuffle=False)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_data, batch_size=self.hparams.batch_size,num_workers=self.hparams.num_workers,pin_memory=self.hparams.pin_memory, shuffle=False)

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

    # def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
    #     if isinstance(batch, CustomBatch):
    #         # move all tensors in your custom data structure to the device
    #         batch.samples = batch.samples.to(device)
    #         batch.targets = batch.targets.to(device)
    #     elif dataloader_idx == 0:
    #         # skip device transfer for the first dataloader or anything you wish
    #         pass
    #     else:
    #         batch = super().transfer_batch_to_device(data, device, dataloader_idx)
    #     return batch

    # def on_before_batch_transfer(self, batch, dataloader_idx):
    #     batch['x'] = transforms(batch['x'])
    #     return batch


if __name__ == '__main__':
    root = pyrootutils.setup_root(__file__, pythonpath=True)

    cfg = omegaconf.OmegaConf.load(root / "configs"/"datamodule"/ "dailydata.yaml")
    # cfg.data_dir = str(root / cfg.data_dir)
    data = hydra.utils.instantiate(cfg)


    data.prepare_data()
    data.setup()
    print(data.metadata)
    print(next(iter(data.train_dataloader())))