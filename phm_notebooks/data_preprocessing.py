# Imports

import pandas as pd
import os
import glob
from tqdm.notebook import tqdm

# Define the data folder path
data_folder = './raw_data'
# Define the list of cutters with wear data
cutters_with_wear = ['c1', 'c4', 'c6']


# Load wear data
wear_data_frames = []
for cutter in cutters_with_wear:
    wear_file_path = os.path.join(data_folder, cutter, f'{cutter}_wear.csv')
    try:
        df = pd.read_csv(wear_file_path)
        df['cutter'] = cutter
        # The wear file has columns: 'cut', 'flute_1', 'flute_2', 'flute_3'
        df.rename(columns={'flute_1': 'flute1', 'flute_2': 'flute2', 'flute_3': 'flute3'}, inplace=True)
        df['ID'] = df['cutter'] + '_' + df['cut'].astype(str)
        wear_data_frames.append(df)
    except FileNotFoundError:
        print(f"Warning: Wear file not found for {cutter} at {wear_file_path}")

if wear_data_frames:
    wear_df = pd.concat(wear_data_frames, ignore_index=True)
    # Reorder columns to match the request
    wear_df = wear_df[['cutter', 'cut', 'ID', 'flute1', 'flute2', 'flute3']]
    print("wear_df created successfully.")
else:
    print("No wear data was loaded. wear_df is empty.")
    wear_df = pd.DataFrame(columns=['cutter', 'cut', 'ID', 'flute1', 'flute2', 'flute3'])

# Save wear_df to CSV
wear_df.to_csv('./preprocessed_data/wear_df.csv',index=False)


# Define column names for the signal files
# Define column names for the signal files
signal_col_names = [
    'Fx', 'Fy', 'Fz',  # Force in X, Y, Z
    'Vx', 'Vy', 'Vz',  # Vibration in X, Y, Z
    'AE_RMS'           # AE-RMS (V)
]

all_cuts_data = []

# Load signals only for cutters with wear data
cutters_with_wear = ['c1', 'c4', 'c6']

for cutter in cutters_with_wear:
    cutter_number = cutter[1:]  # Extracts the number, e.g., '1' from 'c1'
    
    # The signal files are in a nested directory, e.g., c1/c1/
    signal_files_dir = os.path.join(data_folder, cutter, cutter)
    # Create a glob pattern to find all signal files for the cutter
    glob_pattern = os.path.join(signal_files_dir, f'c_{cutter_number}_*.csv')
    signal_files = sorted(glob.glob(glob_pattern))

    if not signal_files:
        print(f"Warning: No signal files found for {cutter} in {signal_files_dir}")
        continue
    
    print(f"\nProcessing cutter: {cutter}...")
    for file_path in tqdm(signal_files, desc=f"Loading {cutter} signals"):
        filename = os.path.basename(file_path)
        
        # Extract the cut number from the filename (e.g., 'c_1_001.csv' -> 1)
        try:
            cut_number = int(filename.split('_')[-1].replace('.csv', ''))
        except (ValueError, IndexError):
            print(f"Warning: Could not parse cut number from filename: {filename}")
            continue

        # Create the unique ID for the cut
        id_val = f"{cutter}_{cut_number}"
        
        # Read the signal data, which has no header
        try:
            # Read the raw data into a NumPy array for efficiency
            signal_data = np.loadtxt(file_path, delimiter=',')
            
            # Create a dictionary for the current cut
            cut_data = {
                'cut_ID': id_val,
                'cutter': cutter,
                'cut': cut_number
            }
            
            # Add each signal as a separate column with the NumPy array
            for i, col_name in enumerate(signal_col_names):
                cut_data[col_name] = signal_data[:, i]
                
            all_cuts_data.append(cut_data)
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

if all_cuts_data:
    # Create the DataFrame from the list of dictionaries
    signals_DF = pd.DataFrame(all_cuts_data)
    
    # Reorder columns for clarity
    ordered_cols = ['cutter', 'cut', 'cut_ID'] + signal_col_names
    signals_DF = signals_DF[ordered_cols]
    
    print("\nsignals_DF with nested arrays created successfully.")
    display(signals_DF.head())
    
    # Verify the structure by checking the type of a signal column
    print("\nData type of a signal column entry:")
    print(type(signals_DF['Fx'].iloc[0]))
else:
    print("\nNo signal data was loaded. signals_DF is empty.")
    signals_DF = pd.DataFrame(columns=['cutter', 'cut', 'cut_ID'] + signal_col_names)


# Save signals_DF to Parquet
signals_DF.to_parquet('./preprocessed_data/signals_DF.parquet', index=False)