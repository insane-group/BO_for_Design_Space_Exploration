#!/bin/bash

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

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <environment_name> <conda or mamba>"
    exit 1
fi

echo "Setting up environment."

$2 create --name $1 --file ./requirements.txt -y || { echo "Failed wile creating the mamba environment."; exit 1; }

$2 run -n $1 pip install xgboost                                     || { echo "Failed wile installing var."; exit 1; }

echo "Finished setting up environment."
