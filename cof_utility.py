import torch
import numpy as np

import globals

# COF Utility Class
class COFUtility:
    def __init__(self):
        pass

    @staticmethod
    def accumulated_cost(acquired_set, cost):
        nb_iters = len(acquired_set)
        accumulated_cost = np.zeros(nb_iters)
        cof_ids = acquired_set[:, 0].round().to(dtype=int)
        for i in range(nb_iters):
            cof_id = cof_ids[i]
            if i == 0:
                accumulated_cost[i] = cost[cof_id]
            else:
                accumulated_cost[i] = accumulated_cost[i - 1] + cost[cof_id]
        return accumulated_cost

    @staticmethod
    def build_cost(acquired_set, cost):
        cof_ids = (int(a[0].item()) for a in acquired_set)
        return torch.tensor([cost[cof_id] for cof_id in cof_ids], dtype=globals.PRECISION, device=globals.device).unsqueeze(-1)

    @staticmethod
    def get_y_max(acquired_set, y):
        cof_ids = [a[0].to(dtype=int) for a in acquired_set]
        y_max = np.max([y[cof_id].cpu().item() for cof_id in cof_ids])  # Ensure values are on the CPU
        return y_max
