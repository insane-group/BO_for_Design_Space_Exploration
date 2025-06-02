import os
import numpy as np
import pandas as pd

import globals

def select_dataset(target_property):

    if target_property == "nch4" or target_property == "nh2":

        if target_property == "nch4":

            df = pd.read_csv('./datasets/HypoCOF-CH4H2-CH4-1bar-TPOT-Input-B - Original.csv')
            y_column = 'NCH4 - 1 bar (mol/kg)'

            globals.save_dir = 'COF_CH4_H2_Keskin_NCH4'
    
        else:    
            df = pd.read_csv('./datasets/HypoCOF-CH4H2-H2-1bar-TPOT-Input-B - Original.csv')
            y_column = 'NH2 - 1 bar (mol/kg)'

            globals.save_dir = 'COF_CH4_H2_Keskin_NH2'

        feature_columns = [  'PLD (Å)', 'LCD (Å)', 'Sacc (m2/gr)', 'Porosity', 'Density (gr/cm3)',
                            '%C', '%F', '%H', '%N', '%O', '%S', '%Si']

    elif target_property == "del_capacity" or target_property == "high_uptake_mol":

        df = pd.read_csv('./datasets/dataset_v1.csv')

        # Drop the 'MOF_no' column as it is no longer needed
        df.drop(columns=['number'], inplace=True)

        feature_columns = [  'dimensions', 'bond_type','voidFraction',
            'supercellVolume', 'density', 'surface_area', 'linkerA', 'linkerB', 
            'net', 'cell_a', 'cell_b', 'cell_c', 'alpha_deg', 'beta_deg' ,'gamma_deg',
            'chemical_formula', 'num_carbon','num_fluorine', 'num_hydrogen', 
            'num_nitrogen', 'num_oxygen', 'num_sulfur','num_silicon', 'vertices', 'edges',
            'genus', 'largest_incl_sphere','largest_free_sphere', 'largest_incl_sphere_along_path']

        if target_property == "del_capacity":
            y_column = 'del_capacity'

            globals.save_dir = 'COF_del_capacity'
        else:
            y_column = 'highUptake_mol'

            globals.save_dir = 'COF_high_uptake_mol'


    elif target_property == "uptake_vol" or target_property == "uptake_grav":

        df = pd.read_csv('./datasets/mofdb.csv')

        threshold = 60

        missing_counts = df.isnull().sum()
        missing_counts
        filtered_columns = missing_counts[missing_counts <= threshold].index
        df_filtered = df[filtered_columns]
        df_filtered
        df = df_filtered[['asa_grav [m²/g]', 'asa_vol [m²/cm³]', 'av_vf', 'pore_volume [cm³/g]', 'density [g/cm³]', 'uptake_grav [wt. %]',
                        'lcd [Å]', 'pld [Å]', 'LFPD [Å]',
                        'cell_volume [Å³]', 'uptake_vol [g H2/L]']]

        feature_columns = ['asa_grav [m²/g]', 'asa_vol [m²/cm³]', 'av_vf', 'pore_volume [cm³/g]', 
                        'density [g/cm³]','lcd [Å]', 'pld [Å]', 'LFPD [Å]','cell_volume [Å³]']

        if target_property == "uptake_vol":
            y_column = 'uptake_vol [g H2/L]'

            globals.save_dir = 'Hydrogen_uptake_vol'

        else:
            y_column = 'uptake_grav [wt. %]'

            globals.save_dir = 'Hydrogen_uptake_grav'

    elif target_property == "d_o2" or target_property == "d_sel":

        df = pd.read_csv('./datasets/MOFdata_O2_H2_uptakes.csv')
        # df['D_sel']= df['SelfdiffusionofO2cm2s']/df['SelfdiffusionofN2cm2s']
        df['D_sel']= np.log10(df['SelfdiffusionofO2cm2s']) - np.log10(df['SelfdiffusionofN2cm2s'])
        df['D_o2'] = np.log10(df['SelfdiffusionofO2cm2s'])
        df['D_n2'] = np.log10(df['SelfdiffusionofN2cm2s'])

        feature_columns = [ 'LCD', 'PLD','LFPD','Volume', 'ASA_m2_g', 'ASA_m2_cm3',
            'NASA_m2_g', 'NASA_m2_cm3', 'AV_VF', 'AV_cm3_g', 'NAV_cm3_g','metal type']

        if target_property == "d_o2":
            y_column = 'D_o2'

            globals.save_dir = 'MOF_O2_N2_d_o2'

        else:
            y_column = 'D_sel'

            globals.save_dir = 'MOF_O2_N2_d_sel'

    elif target_property in ["co2_uptake", "selectivity", "working_capacity", "h2_absorbed", "c3h8_c3h6", "c2h6_c2h4", "propane_avg", "propylene_avg", "ethane_avg", "ethylene_avg"]:

        df = pd.read_csv('./datasets/Merged_Dataset.csv')

        feature_columns = ['POAVF', 'CellV (A^3)', 'total_POV_volumetric','sum-mc_CRY-I-3-all', 'Df_y',
                           'metallic_percentage', 'Di', 'ASA(m2/gram)_1.9', 'O', 'degree_unsaturation', 
                           'C', 'sum-mc_CRY-chi-3-all', 'ASA (m^2/cm^3)','sum-f-lig-Z-0', 'f-lig-T-3', 'Dif', 'Ni']

        if target_property == "co2_uptake":
            y_column = 'CO2_uptake_1bar_298K (mmol/g)'
            globals.save_dir = 'Ethyl_propyl_CO2_uptake'

        elif target_property == "selectivity":
            y_column = 'Selectivity'
            globals.save_dir = 'Ethyl_propyl_selectivity'

        elif target_property == "working_capacity":
            y_column = 'Working_Capacity (mmol/g)'
            globals.save_dir = 'Ethyl_propyl_working_capacity'

        elif target_property == "h2_absorbed":
            y_column = 'H2_adsorbed_100bar_77K (mg/g)'
            globals.save_dir = 'Ethyl_propyl_h2_absorbed'

        elif target_property == "c3h8_c3h6":
            y_column = 'C3H8/C3H6 Selectivity (1Bar)'
            globals.save_dir = 'Ethyl_propyl_c3h8_c3h6'

        elif target_property == "c2h6_c2h4":
            y_column = 'C2H6/C2H4 Selectivity (1Bar)'
            globals.save_dir = 'Ethyl_propyl_c2h6_c2h4'

        elif target_property == "propane_avg":
            y_column = 'propane_avg(mol/kg)'
            globals.save_dir = 'Ethyl_propyl_propane_avg'

        elif target_property == "propylene_avg":
            y_column = 'propylene_avg(mol/kg)'
            globals.save_dir = 'Ethyl_propyl_propylene_avg'

        elif target_property == "ethane_avg":
            y_column = 'ethane_avg(mol/kg)'
            globals.save_dir = 'Ethyl_propyl_ethane_avg'

        else:
            y_column = 'ethylene_avg(mol/kg)'
            globals.save_dir = 'Ethyl_propyl_ethylene_avg'

    else:
        print("The inserted dataset name does not exist. Please selecet on of the following names: \n" \
        "'nch4', 'nh2', 'del_capacity', 'high_uptake_mol', 'co2_uptake', 'selectivity', 'working_capacity', 'h2_absorbed', 'c3h8_c3h6', 'c2h6_c2h4', 'propane_avg', 'propylene_avg', 'ethane_avg', 'ethylene_avg', 'uptake_vol', 'uptake_grav', 'do2', 'd_sel'")
        exit()

    row_count = df.shape[0]
    print("Number of rows in the dataset:", row_count)
    df.dropna(inplace=True)

    row_count = df.shape[0]
    print("Number of rows in the dataset:", row_count)

    # Display the column names
    print("Column names in the file:")
    print(df.columns.tolist())

    # Create save dir if it doesn't exist
    if not os.path.exists(globals.save_dir):
        os.makedirs(globals.save_dir)

    return df, feature_columns, y_column

def calculate_r2_score(y_true, y_pred):
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

def explained_variance(y_true, y_pred):
    var_true = np.var(y_true, ddof=1)
    var_res = np.var(y_true - y_pred, ddof=1)
    r2 = 1 - (var_res / var_true)
    return r2