import pickle
import os
import pandas as pd
import numpy as np

cur_dir= os.getcwd()
base_dir=os.path.join(cur_dir,"scarce_merged")


type_name="type4"
dataset_name="ms"

path_list= ["GG-CG","GC-CG","CG-CC","CC-CC"]
path_list= ["CC-CC"]



def load_and_process_results(base_dir, dataset_name, type_name, path_list):
    # Initialize an empty DataFrame with specified columns
    columns = [
        'avg_test_acc', 'std_test_acc', 'avg_test_f1', 'std_test_f1',
        'avg_test_precision', 'std_test_precision', 'avg_test_recall',
        'std_test_recall', 'avg_epoch_time'
    ]
    df = pd.DataFrame(index=path_list, columns=columns)
    
    # Process each path
    for pt in path_list:
        load_path = os.path.join(base_dir, dataset_name, type_name,pt)
        try:
            final_results_file = os.path.join(load_path, os.listdir(load_path)[-1])
        except IndexError:
            continue  # Skip if no files found in the directory

        with open(final_results_file, "rb") as f:
            loaded_results = pickle.load(f)

        # Compute metrics and store them in the DataFrame
        df.loc[pt, 'avg_test_acc'] = np.mean(np.array(loaded_results['test_acc']))
        df.loc[pt, 'std_test_acc'] = np.std(np.array(loaded_results['test_acc']))
        df.loc[pt, 'avg_test_f1'] = np.mean(np.array(loaded_results['test_f1']))
        df.loc[pt, 'std_test_f1'] = np.std(np.array(loaded_results['test_f1']))
        df.loc[pt, 'avg_test_precision'] = np.mean(np.array(loaded_results['test_precision']))
        df.loc[pt, 'std_test_precision'] = np.std(np.array(loaded_results['test_precision']))
        df.loc[pt, 'avg_test_recall'] = np.mean(np.array(loaded_results['test_recall']))
        df.loc[pt, 'std_test_recall'] = np.std(np.array(loaded_results['test_recall']))
        df.loc[pt, 'avg_epoch_time'] = np.mean(loaded_results['avg_epoch_time'])
        print(np.array(loaded_results['test_acc']))
       
    return df


df_results = load_and_process_results(base_dir, dataset_name, type_name, path_list)
print(df_results)

"""
df_results.to_csv(f"results_csv/results_{dataset_name}_{type_name}.csv", index=True) #  Saves results


# Load DataFrame from a CSV file
loaded_df = pd.read_csv('results.csv', index_col=0)  # Ensure the first column is used as the DataFrame index

# Print the loaded DataFrame
print(loaded_df.columns)

"""