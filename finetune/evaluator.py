import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm


class Evaluator:
    def __init__(self, params, data_loader):
        self.params      = params
        self.data_loader = data_loader

    def get_metrics_for_regression(self, model):
        model.eval()
        truths, preds = [], []

        for x, y in tqdm(self.data_loader, mininterval=1):
            x, y = x.cuda().float(), y.cuda().float()
            pred = model(x).view(-1, 1)
            y    = y.view(-1, 1)
            if self.params.use_regression_norm:
                pred = pred * self.params.y_std + self.params.y_mean
            truths.extend(y.cpu().squeeze().tolist())
            preds.extend(pred.cpu().squeeze().tolist())

        truths = np.array(truths)
        preds  = np.array(preds)
        return (np.corrcoef(truths, preds)[0, 1],
                r2_score(truths, preds),
                mean_squared_error(truths, preds) ** 0.5)
