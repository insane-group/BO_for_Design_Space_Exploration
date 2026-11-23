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
from xgboost import XGBRegressor

# XI = 0.01                     # Adjust this value to change the exploration-exploitation balance
ITER = 120                      # Adjust this value to change the number of BO iterations
PRECISION = torch.float64       # Variable to adjusting the precision
INITS = 3                       # Number of initial points from the design space to start the BO
BATCH_SIZE = 5                  # Define the batch size for sampling
TOP = 10                        # Define the number of top COFs to be predicted

next_threshold = 50
nb_iterations  = ITER
save_dir       = None

# Define exploration-exploitation parameter for acquisition function

INITIAL_XI = 0.0                # Initial value for xi
FINAL_XI = 0.0                  # Final value for xi after threshold
XI_THRESHOLD = 100              # Threshold iteration to change xi

# Zero values for INITIAL_XI and FINAL_XI will result in  a default bo behaviour.

GPmodel = "single"

model = XGBRegressor(           # Define the XGBoost regressor
            n_estimators=800,
            max_depth=5,
            eta=0.02,
            subsample=0.75,
            colsample_bytree=0.7,
            reg_lambda=0.6,
            reg_alpha=0.15,
            random_state=61
        )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check for GPU availability
