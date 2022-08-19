from typing import Optional, Union, List
import torch
from torch.nn import MSELoss
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
import torch_geometric.transforms as T
import src.utils.transform as tnf
import os
import hydra
import omegaconf
import pyrootutils
from src.utils.metrics import CustomMetrics,metric_collection
from pytorch_lightning.callbacks import Callback,EarlyStopping,ModelCheckpoint,RichProgressBar,ModelSummary



class NetQtyModel(pl.LightningModule):
    def __init__(self,encoder,decoder,optimizer,lr=0.01):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.transform = T.Compose([tnf.ScaleEdges(attrs=["edge_attr"]),
                                    T.NormalizeFeatures(attrs=["x", "edge_attr"]),
                                    T.ToUndirected(),
                                    T.AddSelfLoops(),
                                    tnf.RevDelete()])

        self.encoder = self.hparams.encoder
        self.decoder = self.hparams.decoder
        self.lr = lr


        # self.metrics = {'train':metric_collection,
        #                 'val':metric_collection,
        #                 'test':metric_collection}

        self.metrics = {'train': CustomMetrics(),
                        'val': CustomMetrics(),
                        'test': CustomMetrics()}


    def forward(self, x_dict,edge_index_dict,edge_attr_dict):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict,edge_index_dict,edge_attr_dict)

    def loss_function(self,pred,targets):
        loss = MSELoss()
        return loss(pred,targets)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(self.parameters(),lr=self.lr)
        # cycle momentum needs to be False for Adam to work
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.3, step_size_up=10,
                                                      cycle_momentum=False)
        return [optimizer], [lr_scheduler]

    # compute
    def compute_metrics(self,preds,targets,mode):
        preds = preds.detach().cpu().int()
        targets = targets.detach().cpu().int()
        acc = self.metrics[mode](preds, targets)
        return acc

    # logging
    def logging_step(self,loss,acc,mode) -> None:
        self.log(f"{mode}/step/loss",loss,on_epoch=False,on_step=True,rank_zero_only=True)
        for metric_name, ac in acc.items():
            self.log(f"{mode}/step/{metric_name}",ac,on_epoch=False,on_step=True,rank_zero_only=True)

    def logging_epoch(self, avg_loss,acc, mode):
        self.log(f"{mode}/epoch/loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True, rank_zero_only=True)
        for metric_name,ac in acc.items():
            self.log(f"{mode}/epoch/{metric_name}", ac, on_step=False, on_epoch=True,rank_zero_only=True)
        self.metrics[mode].reset()

    def _shared_step(self, batch, batch_idx, mode):
        preds = self(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
        targets = batch[('node1', 'to', 'node2')].edge_label.flatten().float()
        loss = self.loss_function(preds, targets)
        acc = self.compute_metrics(preds, targets, mode)
        self.logging_step(loss, acc, mode)
        results = {'loss': loss,'preds':preds,'targets':targets}
        return results

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.metrics["val"].reset()

    # Training
    def training_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        mode = "train"
        results = self._shared_step(batch, batch_idx, mode)
        return results

    def validation_step(self,batch, batch_idx, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        mode = "val"
        results = self._shared_step(batch, batch_idx, mode)
        return results

    def test_step(self,batch, batch_idx, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        mode = "test"
        results = self._shared_step(batch, batch_idx, mode)
        return results

    def training_epoch_end(self, training_step_outputs: EPOCH_OUTPUT) -> None:
        mode="train"
        avg_loss = torch.stack([x["loss"] for x in training_step_outputs]).mean()
        all_preds = torch.hstack([x["preds"] for x in training_step_outputs])
        all_targets = torch.hstack([x["targets"] for x in training_step_outputs])
        acc = self.compute_metrics(all_preds,all_targets, mode)
        self.logging_epoch(avg_loss,acc, mode)


    def validation_epoch_end(self, val_step_outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        # [results,results,results ...]
        mode = "val"
        avg_val_loss = torch.stack([x["loss"] for x in val_step_outputs]).mean()
        all_preds = torch.hstack([x["preds"] for x in val_step_outputs])
        all_targets = torch.hstack([x["targets"] for x in val_step_outputs])
        acc = self.compute_metrics(all_preds, all_targets, mode)
        self.logging_epoch(avg_val_loss,acc, mode)
        results = {'progress_bar': {'val_loss':avg_val_loss},
                   'val_loss': avg_val_loss}
        return results

    def test_epoch_end(self, test_step_outputs):
        mode = "test"
        avg_test_loss = torch.stack([x["loss"] for x in test_step_outputs]).mean()
        all_preds = torch.hstack([x["preds"] for x in test_step_outputs])
        all_targets = torch.hstack([x["targets"] for x in test_step_outputs])
        acc = self.compute_metrics(all_preds, all_targets, mode)
        self.logging_epoch(avg_test_loss, acc, mode)
        results = {'progress_bar': {'test_loss': avg_test_loss},
                   'test_loss': avg_test_loss}
        #self.metrics["test"].reset()
        return results


    #
    # def predict_step(self,batch, batch_idx, *args, **kwargs) -> Optional[STEP_OUTPUT]:
    #     preds = self(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
    #     return preds
    #



if __name__=="__main__":
    pl.seed_everything(3407)
    data_dir = os.path.join(os.getcwd(), "../../data/dailyroot")
    root = pyrootutils.setup_root(__file__, pythonpath=True)

    paths_cfg = omegaconf.OmegaConf.load(root / "configs" / "paths" / "default.yaml")
    paths = hydra.utils.instantiate(paths_cfg)


    data_cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "dailydata.yaml")
    #data_cfg.data_dir=data_dir
    data = hydra.utils.instantiate(data_cfg)

    model_cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "net_qty_model.yaml")
    model= hydra.utils.instantiate(model_cfg)

    # callbacks_cfg = omegaconf.OmegaConf.load(root / "configs" / "callbacks" / "default.yaml")
    # callbacks = hydra.utils.instantiate(callbacks_cfg)
    experiment_dir = os.path.join(os.getcwd(), "../../src/outputs")
    callbacks = []
    goldstar_metric = "val/epoch/loss"
    filename_format = "epoch={epoch:04d}-validation.loss={validation/loss:.3f}"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=3,
        filename=filename_format,
        monitor=goldstar_metric,
        mode="min",
        auto_insert_metric_name=False,
        dirpath=experiment_dir,
        every_n_epochs=3,
    )

    summary_callback = pl.callbacks.ModelSummary(max_depth=2)

    callbacks = [summary_callback, checkpoint_callback]

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val/epoch/loss", mode="min", patience=3
    )
    callbacks.append(early_stopping_callback)

    # Enable chkpt , gpu, epochs
    trainer = pl.Trainer(
                         max_steps=1000,max_epochs=25,
                         #accelerator='gpu', devices=1,
                         #log_every_n_steps=5,
                         #check_val_every_n_epoch=3,
                         gradient_clip_val=1.0,
                         deterministic=True,
                         progress_bar_refresh_rate=5,
                         callbacks=callbacks
                         #auto_lr_find=True
                         #overfit_batches=10
                         )
    # Autotune LR
    # lr_finder = trainer.tuner.lr_find(model=model,datamodule=data,max_lr=0.01)
    # model.lr = lr_finder.suggestion()
    # print(model.lr)

    trainer.fit(model=model,datamodule=data)
    # trainer = pl.Trainer(max_steps=1000,max_epochs=500,check_val_every_n_epochs=10,
    #                      auto_lr_find=True,gradient_clip_val=1.0,deterministic=True,
    #                      accumulate_grad_batches=4,sync_batchnorm=True
    #                      )

    # trainer.validate(model)
    # trainer.test(model)
    # trainer.predict(model)

    # print on callback
