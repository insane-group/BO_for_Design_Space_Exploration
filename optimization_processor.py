import os
import gc
import torch
import numpy as np
from scipy.stats import norm

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms.outcome import Standardize

from gpytorch.mlls import ExactMarginalLogLikelihood

import globals
from cof_processor import COFProcessor
from cof_utility import COFUtility

# Optimization Processor Class
class OptimizationProcessor:
    @staticmethod
    def train_surrogate_model(X_train, y_train):
        X_train = X_train.to(dtype=globals.PRECISION)
        y_train = y_train.to(dtype=globals.PRECISION)

        if globals.GPmodel == "single":
            model = SingleTaskGP(
                X_train,
                y_train,
                outcome_transform=Standardize(m=1)
            )
        else:
            model = SaasFullyBayesianSingleTaskGP(
                X_train,
                y_train,
                outcome_transform=Standardize(m=1)
            )


        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        torch.cuda.empty_cache()
        gc.collect()
        return model

    @staticmethod
    def mu_sigma(model, X):
        f_posterior = model.posterior(X)
        mean = f_posterior.mean.squeeze().detach().cpu()
        std_dev = torch.sqrt(f_posterior.variance.squeeze()).detach().cpu()
        del f_posterior  # Free memory
        return mean, std_dev
    
    @staticmethod
    def EI_hf(model, X, acquired_set, xi, y):
        hf_mu, hf_sigma = OptimizationProcessor.mu_sigma(model, X)

        # Convert PyTorch tensors to NumPy arrays
        hf_mu_np = hf_mu.cpu().numpy()  # Ensure the tensor is moved to CPU
        hf_sigma_np = hf_sigma.cpu().numpy()  # Ensure the tensor is moved to CPU

        # Get y_max
        y_max = COFUtility.get_y_max(acquired_set, y)

        # Calculate EI
        z = (hf_mu_np - y_max) / hf_sigma_np
        explore_term = hf_sigma_np * norm.pdf(z)
        exploit_term = (hf_mu_np - y_max
                        - xi # comment/un-comment to include/exclude the exploitation/exploration term
                        ) * norm.cdf(z)
        ei = explore_term + exploit_term

        # Ensure non-negative values for EI
        return np.maximum(ei, 0)

    @staticmethod
    def acquisition_scores(model, X, acquired_set, xi, y):
        ei = OptimizationProcessor.EI_hf(model, X, acquired_set, xi, y)
        return ei

    @staticmethod
    def run_Bayesian_optimization(nb_iterations, initializing_COFs, batch_size=globals.BATCH_SIZE, verbose=False, stop_threshold=0.0000001, max_iteration=globals.ITER, X=None, y=None):
        local_threshold = globals.next_threshold
        assert nb_iterations > len(initializing_COFs)
        stop_flag = False
        acquired_set = COFProcessor.initialize_acquired_set(initializing_COFs)
        max_expected_improvement_seen = 0
        top_cof_id = torch.argmax(y).item()

        # Create directory to save BO checkpoint files

        bo_save_dir = os.path.join(globals.save_dir, "bo_points")
        os.makedirs(bo_save_dir, exist_ok=True)

        bo_points_dict = {}

        for i in range(globals.ITER):
            print("BO iteration: ", i)
            X_train = COFProcessor.build_X_train(acquired_set, X)
            y_train = COFProcessor.build_y_train(acquired_set, y)

            # Set xi based on the iteration number
            xi = globals.INITIAL_XI if i < globals.XI_THRESHOLD else globals.FINAL_XI

            if verbose:
                print("Initialization - \n")
                print("\tCOF IDs acquired    = ", [acq_[0].item() for acq_ in acquired_set])
                print("\n\tTraining data:\n")
                print("\t\t X train shape = ", X_train.shape)
                print("\t\t y train shape = ", y_train.shape)

            model = OptimizationProcessor.train_surrogate_model(X_train, y_train)

            # After training the model, clear the cache
            del X_train, y_train
            torch.cuda.empty_cache()
            gc.collect()

            the_acquisition_scores = OptimizationProcessor.acquisition_scores(model, X, acquired_set, xi, y)

            for cof_id in acquired_set:
                the_acquisition_scores[int(cof_id)] = - np.inf

            batch_indices = np.argpartition(the_acquisition_scores, -batch_size)[-batch_size:]
            batch_acq = torch.tensor([[idx] for idx in batch_indices], dtype=globals.PRECISION, device=globals.device)
            acquired_set = torch.cat((acquired_set, batch_acq))

            ei_current = OptimizationProcessor.EI_hf(model, X, acquired_set, xi, y)
            total_expected_improvement = np.sum(ei_current)
            max_expected_improvement_seen = max(max_expected_improvement_seen, total_expected_improvement)
            percentage_of_max_expected_improvement = (total_expected_improvement / max_expected_improvement_seen) * 100

            print("Total Expected Improvement:", total_expected_improvement)
            print("Percentage of Max Expected Improvement:", percentage_of_max_expected_improvement)
            print(f"Iteration {i}: Acquired set size = {len(acquired_set)}")

            # Clear the cache after the acquisition step
            del ei_current, total_expected_improvement, batch_acq, model, the_acquisition_scores, percentage_of_max_expected_improvement
            torch.cuda.empty_cache()
            gc.collect()

            # Save the current acquired set every SAVE_INTERVAL iterations
            if len(acquired_set) >= local_threshold:
                current_sample = acquired_set.cpu().numpy()  # shape: (n_samples, 1)
                iter_key = i + 1
                bo_points_dict[iter_key] = current_sample.copy()
                # Save to CSV (one file per checkpoint)
                save_filename = os.path.join(bo_save_dir, f"bo_points_iter_{iter_key}.csv")
                np.savetxt(save_filename, current_sample, delimiter=",", fmt='%d')
                print(f"Saved BO sampled points at iteration {iter_key} to {save_filename}")
                local_threshold += globals.next_threshold

            if i >= max_iteration:
                print(f"Iteration count exceeded {max_iteration}. Stopping optimization.")
                stop_flag = True
                break

            if verbose:
                print("\tacquired COF batch", batch_indices)

        return acquired_set, bo_points_dict