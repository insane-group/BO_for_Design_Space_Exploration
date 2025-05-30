import torch
import numpy as np

import globals

# COF Processor Class
class COFProcessor:
    def __init__(self):
        pass

    @staticmethod
    def row_to_tensor(row):
        return torch.tensor(row.values, dtype=globals.PRECISION, device=globals.device)

    @staticmethod
    def min_max_normalize(X):
        X_normalized = X.clone()
        for j in range(X.size()[1]):
            min_val = torch.min(X[:, j]).item()
            max_val = torch.max(X[:, j]).item()
            if (max_val - min_val) == 0:
                X_normalized[:, j] = 0
            else:
                X_normalized[:, j] = (X[:, j] - min_val) / (max_val - min_val)
        return X_normalized

    @staticmethod
    def get_initializing_COF(X):
        # Move the mean tensor to CPU before converting to NumPy
        data_center = np.array([X[:, i].mean().cpu().item() for i in range(X.size()[1])])
        return np.argmin(np.linalg.norm(X.cpu().numpy() - data_center, axis=1))


    @staticmethod  
    def diverse_set(X, seed_cof, train_size):
        # Move X to CPU before using NumPy functions
        X_cpu = X.cpu().numpy()
        ids_train = [seed_cof]
        for j in range(train_size - 1):
            dist_to_train_set = np.linalg.norm(X_cpu - X_cpu[ids_train, None, :], axis=2)
            min_dist_to_a_training_pt = np.min(dist_to_train_set, axis=0)
            new_cof_id = np.argmax(min_dist_to_a_training_pt)
            ids_train.append(new_cof_id)
        assert np.size(np.unique(ids_train)) == train_size
        return np.array(ids_train)


    @staticmethod
    def in_acquired_set(cof_id, acquired_set):
        cof_ids = acquired_set[:, 0]
        return torch.any(cof_ids == cof_id)

    @staticmethod
    def build_X_train(acquired_set, X):
        cof_ids = [int(a[0].item()) for a in acquired_set]  # Convert to integer indices
        return X[torch.tensor(cof_ids, device=globals.device, dtype=torch.long), :]

    @staticmethod
    def build_y_train(acquired_set, y):
        cof_ids = [int(a[0].item()) for a in acquired_set]  # Convert to integer indices
        y_train = y[torch.tensor(cof_ids, device=globals.device, dtype=torch.long)].unsqueeze(-1)
        return y_train

    @staticmethod
    def initialize_acquired_set(initializing_COFs):
        return torch.tensor([[cof_id] for cof_id in initializing_COFs], device=globals.device, dtype=globals.PRECISION)