<!--
  ~ Copyright (c) 2026 NCSRD
  ~
  ~ Licensed under the Apache License, Version 2.0 (the "License");
  ~ you may not use this file except in compliance with the License.
  ~ You may obtain a copy of the License at
  ~
  ~     http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing, software
  ~ distributed under the License is distributed on an "AS IS" BASIS,
  ~ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ~ See the License for the specific language governing permissions and
  ~ limitations under the License.
  ~
  ~ SPDX-License-Identifier: Apache-2.0
-->

# BO_for_Design_Space_Exploration



## About The Project

This github repo accompanies the paper *Navigating Materials Design Spaces with Efficient Bayesian Optimization: A Case Study in Functionalized
Nanoporous Materials* by providing the code used for the  described experiments.



## Abstract

Machine learning (ML) has the potential to accelerate the discovery of high-performance materials by learning complex structure–property relationships and prioritizing candidates for costly experiments or simulations. However, ML efficiency is often offset by the need for large, high-quality training
datasets, motivating strategies that intelligently select the most informative samples. Here, we formulate the search for top-performing functionalized nanoporous materials (metal–organic and covalent–organic frameworks) as a global optimization problem and apply Bayesian Optimization
(BO) to identify regions of interest and rank candidates with minimal evaluations. We highlight the importance of a proper and efficient initilization scheme of the BO process, and we introduce the idea that BO acquired samples can serve as data to train an XGBoost regression predictive model
that can further enrich the efficient mapping of the region of high performing instances of the design space. Across multiple literature-derived adsorption and diffusion datasets containing thousands of structures, our BO framework identifies 2x- to 3x- more materials of a top-100 or top-10 ranking list, than random-sampling-based ML pipelines, and it achieves significantly higher ranking quality. Moreover, the surrogate enrichment strategy further boosts top-N recovery while maintaining high ranking fidelity. By shifting the evaluation focus from average predictive metrics (e.g., R2, MSE)
to task-specific criteria (e.g., recall@N and nDCG), our approach offers a practical, data-efficient, and computationally accessible route to guide experimental and computational campaigns toward the most promising materials.



## Configuration

For the implementation of the code we have used *mamba* as our package manager , but *conda* should work fine as well. For specific instructions on installing these package managers please refer to the following links:

- **Conda:**   https://docs.conda.io/en/latest/
- **Mamba:** https://github.com/mamba-org/mamba

After installing your selected package manager you can run the ***env_setup.sh*** bash script contained in the repo. This should create a mamba/conda environment containing all the necessary libraries to execute our code.

The bash script expects two command line arguments. The first one is the name that you want to give to the new environment and the second is whether you are using mamba or conda. So a typical run of the script should look like this:

```bash
./env_setup.sh test_environment mamba
```

After the script has finished just activate the environment by running

```bash
mamba activate test_environment
```

and then you are ready to execute the python file.



## Usage

The execution of the program happens through the *main.py* file. It takes two possible command line arguments.

1.  *-t or --target* which defines the target property that the user wants to optimise. By default the target value is set to *nch4*. When this argument is provided the BO experiments are executed.
2.  -p or --path. This arguments provides the path to the results of a BO experiment and calls the compute_metrics function which creates a file with the evaluation of the results as described in the manuscript.

example of execution:

```bash
python ./main.py
```

or

```bash
python ./main.py -t nch4
```

or 

```bash
python ./main.py -p ./COF_CH4_H2_Keskin_NCH4/sampling_results
```



This is a list of all the possible inputs that the -t parameter can get.

| -t or --target parameter |                      Dataset                       | Target Property (Column name) |
| :----------------------: | :------------------------------------------------: | :---------------------------: |
|          *nch4*          | HypoCOF-CH4H2-CH4-1bar-TPOT-Input-B - Original.csv |    COF_CH4_H2_Keskin_NCH4     |
|          *nh2*           | HypoCOF-CH4H2-CH4-1bar-TPOT-Input-B - Original.csv |     COF_CH4_H2_Keskin_NH2     |
|      *del_capacity*      |                   dataset_v1.csv                   |         del_capacity          |
|    *high_uptake_mol*     |                   dataset_v1.csv                   |        highUptake_mol         |
|       *uptake_vol*       |                     mofdb.csv                      |      uptake_vol [g H2/L]      |
|      *uptake_grav*       |                     mofdb.csv                      |      uptake_grav [wt. %]      |
|          *d_o2*          |             MOFdata_O2_H2_uptakes.csv              |             D_o2              |
|         *d_sel*          |             MOFdata_O2_H2_uptakes.csv              |             D_sel             |
|       *co2_uptake*       |                 Merged_Dataset.csv                 | CO2_uptake_1bar_298K (mmol/g) |
|      *selectivity*       |                 Merged_Dataset.csv                 |          Selectivity          |
|    *working_capacity*    |                 Merged_Dataset.csv                 |   Working_Capacity (mmol/g)   |
|      *h2_absorbed*       |                 Merged_Dataset.csv                 | H2_adsorbed_100bar_77K (mg/g) |
|       *c3h8_c3h6*        |                 Merged_Dataset.csv                 | C3H8/C3H6 Selectivity (1Bar)  |
|       *c2h6_c2h4*        |                 Merged_Dataset.csv                 | C2H6/C2H4 Selectivity (1Bar)  |
|      *propane_avg*       |                 Merged_Dataset.csv                 |      propane_avg(mol/kg)      |
|     *propylene_avg*      |                 Merged_Dataset.csv                 |     propylene_avg(mol/kg)     |
|       *ethane_avg*       |                 Merged_Dataset.csv                 |      ethane_avg(mol/kg)       |
|      *ethylene_avg*      |                 Merged_Dataset.csv                 |     ethylene_avg(mol/kg)      |

The parameters for the Bayesian Optimisation are defined in the *globals.py* file and can be modified according to user preferences.

**NOTE** During runtime the warning "InputDataWarning: Data (input features) is not contained to the unit cube. Please consider min-max scaling the input data" appears. This is due to the lack of feature normalisation on the evaluation phase when a Gaussian Process model is trained with the samples selected by the BO, and it does not hinder the execution of the program.



## Results

Depending on the target parameter that the user selects, a folder containing the results is created with a specific name. The mapping of property to directory name is presented in the table below:



|  target parameter  |    Results Directory Name     |
| :----------------: | :---------------------------: |
|       *nch4*       |    COF_CH4_H2_Keskin_NCH4     |
|       *nh2*        |     COF_CH4_H2_Keskin_NH2     |
|   *del_capacity*   |       COF_del_capacity        |
| *high_uptake_mol*  |      COF_high_uptake_mol      |
|    *uptake_vol*    |      Hydrogen_uptake_vol      |
|   *uptake_grav*    |     Hydrogen_uptake_grav      |
|       *d_o2*       |        MOF_O2_N2_d_o2         |
|      *d_sel*       |        MOF_O2_N2_d_sel        |
|    *co2_uptake*    |    Ethyl_propyl_CO2_uptake    |
|   *selectivity*    |   Ethyl_propyl_selectivity    |
| *working_capacity* | Ethyl_propyl_working_capacity |
|   *h2_absorbed*    |   Ethyl_propyl_h2_absorbed    |
|    *c3h8_c3h6*     |    Ethyl_propyl_c3h8_c3h6     |
|    *c2h6_c2h4*     |    Ethyl_propyl_c2h6_c2h4     |
|   *propane_avg*    |   Ethyl_propyl_propane_avg    |
|  *propylene_avg*   |  Ethyl_propyl_propylene_avg   |
|    *ethane_avg*    |    Ethyl_propyl_ethane_avg    |
|   *ethylene_avg*   |   Ethyl_propyl_ethylene_avg   |

Inside there are two folders contained:

- bo_points
- sampling_results

The first directory contains one csv file for each BO iteration, and each file contains the ids of the selected samples. 

The second directory contains one csv file for each iteration. In each file there is a list of the ids of the top-N samples (N is manually set in the globals.py file). For each on of the real top samples we keep the information concerning whether it has been sampled (both in random and bo iterations) and also how high in the ranking of each iteration it has been placed.

In case the program is run with the -p parameter and the evaluation functions are called then a file named *evaluation_metrics_results.csv* will appear in the results directory containing the evaluation metrics for each run of different sample size.



## Data

All the datasets used for our experiments are contained in the *datasets* directory.



## License

This project is licensed under the Apache 2 license. See `LICENSE` for details.



## Contact

If you want to contact us you can reach as at the emails of the authors as mentioned in the manuscript.



## Contributors

 <a href= "https://github.com/Sileonis">Panagiotis Krokidas</a> <br />

 <a href= "https://github.com/vGkatsis">Vassilis Gkatsis </a> <br />

 <a href= "">John Theocharis</a> <br />