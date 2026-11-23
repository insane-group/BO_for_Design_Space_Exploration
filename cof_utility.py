# Copyright (c) 2026 NCSRD
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

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
