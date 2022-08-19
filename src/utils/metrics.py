from typing import Optional
from torchmetrics import Metric
import torch
from torchmetrics import Accuracy,MeanAbsolutePercentageError,SymmetricMeanAbsolutePercentageError
from torchmetrics import MeanAbsoluteError,MeanSquaredError,R2Score,KLDivergence,WeightedMeanAbsolutePercentageError
from torchmetrics import MetricCollection


metric_collection = MetricCollection([
        MeanAbsoluteError(),
        MeanSquaredError(),
        R2Score(multioutput='uniform_average')
        # MeanAbsolutePercentageError(),
        # SymmetricMeanAbsolutePercentageError(),
        # WeightedMeanAbsolutePercentageError()
    ])

class CustomMetrics(Metric):
    # Set to True if the metric is differentiable else set to False
    is_differentiable: Optional[bool] = None

    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = True

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = False

    def __init__(self):
        super().__init__()

        # self.add_state("mae", default=torch.tensor(0), dist_reduce_fx="mean")
        # self.add_state("mse", default=torch.tensor(0), dist_reduce_fx="mean")
        # self.add_state("rmse", default=torch.tensor(0), dist_reduce_fx="mean")
        # self.add_state("r2", default=torch.tensor(0), dist_reduce_fx="mean")


    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        #preds, target = self._input_format(preds, target)
        assert preds.shape == targets.shape

        mae = MeanAbsoluteError()
        mse = MeanSquaredError()
        r2 = R2Score(multioutput='uniform_average')
        smape = SymmetricMeanAbsolutePercentageError()
        kld = KLDivergence()

        self.mae = mae(preds,targets)
        self.mse = mse(preds,targets)
        self.rmse = torch.sqrt(self.mse)
        self.r2 = r2(preds,targets)
        self.smape = smape(preds,targets)
        self.kld = kld(preds[:,None],targets[:,None])


    def compute(self):
        return {'MeanAbsoluteError':self.mae,
                'MeanSquaredError':self.mse,
                'RootMeanSquaredError':self.rmse,
                'R2Score':self.r2,
                'SMAPE' : self.smape,
                'KLDiv':self.kld}


if __name__ == "__main__":
    preds = torch.tensor([28,145,69,84,92,50,39],dtype=torch.float)
    targets = torch.tensor([24,120,73,84,86,38,48],dtype=torch.float)
    metric_collection = MetricCollection([
        MeanAbsoluteError(),
        MeanSquaredError(),
        R2Score(multioutput='uniform_average')
    ])
    metric = CustomMetrics()
    metric(preds,targets)
    # acc = metric_collection(preds,targets)
    acc = metric.compute()
    print(acc)
    metric.reset()

