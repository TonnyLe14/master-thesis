import pandas as pd
import numpy as np

def map_steel_family(df):
    """
    Map steel grades to their respective steel families.
    
    Args:
        df (pd.DataFrame): DataFrame with 'steel_grade' column
    
    Returns:
        pd.DataFrame: DataFrame with added 'steel_family' column
    """
    family_mapping = {
        # AHSS family
        '2F63': 'AHSS', '2F95': 'AHSS', '2Q91': 'AHSS', '3F63': 'AHSS',
        
        # CMn family
        '110E': 'CMn', '110F': 'CMn', '110H': 'CMn', '112H': 'CMn',
        '112L': 'CMn', '114E': 'CMn', '116L': 'CMn', '125C': 'CMn',
        '126C': 'CMn', '126L': 'CMn', '180L': 'CMn', '180N': 'CMn',
        '184L': 'CMn', '184M': 'CMn', '186C': 'CMn', '187L': 'CMn',
        '180G': 'CMn', '111C': 'CMn', '114C': 'CMn', '114H': 'CMn',
        '121L': 'CMn', '1T46': 'CMn', '1T36': 'CMn', '1T86': 'CMn',
        '1S38': 'CMn', '1S42': 'CMn', '1T32': 'CMn', '1T80': 'CMn',
        '1T82': 'CMn', '1T34': 'CMn', '110B': 'CMn', '1P65': 'CMn',
        '184K': 'CMn', '1N47': 'CMn', '1N57': 'CMn', '123L': 'CMn',
        '115H': 'CMn', '1T44': 'CMn', '1T84': 'CMn', '1T94': 'CMn',
        '115E': 'CMn', '1P85': 'CMn',
        
        # HSLA family
        '1N80': 'HSLA', '1N31': 'HSLA', '1N60': 'HSLA', '1N61': 'HSLA',
        '1N81': 'HSLA', '1N91': 'HSLA', '1N84': 'HSLA', '1N64': 'HSLA',
        '1N94': 'HSLA', '1N32': 'HSLA', '1N33': 'HSLA', '1N62': 'HSLA',
        '1N63': 'HSLA', '1N82': 'HSLA', '1N83': 'HSLA', '1N92': 'HSLA',
        '1N93': 'HSLA', '1N36': 'HSLA', '1N37': 'HSLA', '1N66': 'HSLA',
        '1N67': 'HSLA', '1N86': 'HSLA', '1N87': 'HSLA', '1N96': 'HSLA',
        '1N97': 'HSLA', '1N38': 'HSLA', '1N39': 'HSLA', '1N68': 'HSLA',
        '1N69': 'HSLA', '1N88': 'HSLA', '1N98': 'HSLA', '1N99': 'HSLA',
        '3N73': 'HSLA',
        
        # IF family
        '514Z': 'IF', '515M': 'IF', '581G': 'IF', '590Q': 'IF',
        '590Z': 'IF', '591M': 'IF', '594Q': 'IF', '594Z': 'IF',
        '595M': 'IF', '542P': 'IF', '543P': 'IF', '544P': 'IF',
        '545P': 'IF', '540Z': 'IF', '541M': 'IF', '54AE': 'IF',
        '561P': 'IF', '592P': 'IF', '593P': 'IF', '598P': 'IF',
        '599P': 'IF', '59AD': 'IF', '552V': 'IF', '553V': 'IF',
        '589L': 'IF', '59TM': 'IF', '55AV': 'IF'
    }
    
    df_copy = df.copy()
    df_copy['steel_family'] = df_copy['steel_grade'].map(family_mapping)
    return df_copy

def remove_outliers(df):
    """
    Remove outliers from the r_value column based on IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    Q1 = df['r_value'].quantile(0.10)
    Q3 = df['r_value'].quantile(0.90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df['r_value'] >= lower_bound) & (df['r_value'] <= upper_bound)]

def process_steel_data(data_path, description_path, corr_rate, dvl, model_output=True, nan_threshold=10):
    # Read input data
    line_csv = data_path + f'2024-04-04_DVL{dvl}_test_data_for_refit.csv'
    dvl1 = pd.read_csv(line_csv, delimiter=',')
    dvl1 = dvl1[dvl1['r_value'] != 0]

    # Read description data
    desc = pd.read_excel(description_path, sheet_name=3)
    
    if model_output:
        tmp = desc[(desc['model'] != 'n') & (desc['Table Name'] == f"mecomep_dv2{dvl}")
            & ((desc['Input Type'] == 'Actual - Measurement or Count') | 
                (desc['Input Type'] == 'Prediction - Calculation'))]
        
        features_selected = tmp[['Attribute Name', 'Input Type']]
        existing_columns = [col for col in features_selected['Attribute Name'] if col in dvl1.columns]
        dvl1_selected = dvl1[existing_columns + ["rm", "ag", "a80", "n_value", 'r_value', 'steel_grade']]

    else:
        tmp = desc[(desc['model'] != 'n') & (desc['Table Name'] == f"mecomep_dv2{dvl}")
            & (desc['Input Type'] == 'Actual - Measurement or Count')]
        
        features_selected = tmp[['Attribute Name', 'Input Type']]
        existing_columns = [col for col in features_selected['Attribute Name'] if col in dvl1.columns]
        dvl1_selected = dvl1[existing_columns + ['r_value', 'steel_grade']]

    # Handle missing values
    nan_cols = dvl1_selected.isna().sum().sort_values(ascending=False)
    nan_cols = nan_cols[nan_cols > 0]
    nan_percentage = (nan_cols / len(dvl1_selected)) * 100

    # Drop columns where NaN percentage is above the specified threshold
    cols_to_drop = nan_percentage[nan_percentage > nan_threshold].index.tolist()

    # Drop the identified columns
    dvl1_selected = dvl1_selected.drop(cols_to_drop, axis=1)

    # Drop rows with remaining NaN values
    dvl1_selected.dropna(inplace=True)

    # Feature correlation
    corr = dvl1_selected.drop(['steel_grade'], axis=1).corr()['r_value'].abs()
    selected_features = corr[abs(corr) >= corr_rate].index.tolist()
    print(f'Dropped {len(dvl1_selected.columns) - len(selected_features)} columns')

    # Additional filtering
    dvl1_selected = dvl1_selected[selected_features + ['steel_grade']]
    counts = dvl1_selected['steel_grade'].value_counts()
    filtered_values = counts[counts >= 5].index
    dvl1_selected = dvl1_selected[dvl1_selected['steel_grade'].isin(filtered_values)]
    
    # Map steel families and remove outliers
    dvl1_selected = map_steel_family(dvl1_selected)
    dvl1_selected = dvl1_selected.groupby('steel_family').apply(remove_outliers).reset_index(drop=True)
    
    return dvl1_selected