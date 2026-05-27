from src.config import Config
from src.data_utils import create_dataset_dict
from src.data_utils import KnowledgeGraphDataset, create_dataloader
from src.training import run_experiment
from src.logger import save_df_to_excel
from collections import defaultdict
import pandas as pd
import numpy as np
import os

print("Initializing experiment datasets...")

# Main settings
cfg = Config()
ILP_dataset_paths = dict(sorted(create_dataset_dict(cfg.DIR_ILP).items()))
TLP_dataset_paths = dict(sorted(create_dataset_dict(cfg.DIR_TLP).items()))

def run_suite(dataset_paths, 
              suite_name, 
              metric, 
              exp_iter=10, 
              neg_ratio_hitk=50, 
              neg_ratio_mrr=1):
    
    all_results = defaultdict(list)
    print(f"\nRunning {suite_name} (Metric: {metric})...")
    result = {}
    
    for it in range(exp_iter):
        print(f"Iteration {it+1}/{exp_iter}")
        for dataset_name, path_dict in dataset_paths.items():
            result = run_experiment( dataset_name=dataset_name,
                                     file_path=path_dict,
                                     fanouts_train=[15, 10, 5],
                                     fanouts_test=[-1, -1],
                                     batch_size_train=1024,
                                     batch_size_test=256,
                                     neg_ratio_train=1,
                                     neg_ratio_hitk=neg_ratio_hitk,
                                     neg_ratio_mrr=neg_ratio_mrr,
                                     metric=metric,
                                     learning_rate=0.01,
                                     epochs=100,
                                     result=result)
            
        # Transfer results of this iteration to the overall list
        for name, res in result.items():
            all_results[name].append(res)
    
    # Final processing: mean and standard deviation over 10 iterations
    final_data = []
    for dataset_name, metrics_list in all_results.items():
        row = {"Dataset": dataset_name}
        # Extract column names (auc, auc_pr, ram_gb, etc.) from the first result
        keys = metrics_list[0].keys()
        for k in keys:
            vals = [m[k] for m in metrics_list]
            row[f"{k}_mean"] = np.mean(vals)
            row[f"{k}_std"] = np.std(vals)
        final_data.append(row)
        
    df = pd.DataFrame(final_data)
 
    # Save to Excel file
    print(df)
    save_df_to_excel(df, os.path.join(cfg.RESULT_DIR, "ILPSAGE_final_results.xlsx"), suite_name)
    print(f"Suite '{suite_name}' completed and saved to {cfg.RESULT_DIR}.")

    
# run_suites    
run_suite(ILP_dataset_paths, "inductive_neg1", "AUC-PR", neg_ratio_mrr=1)
run_suite(ILP_dataset_paths, "inductive_neg50", "Hit@10", neg_ratio_hitk=50)
run_suite(TLP_dataset_paths, "transductive_neg1", "AUC-PR", neg_ratio_mrr=1)
run_suite(TLP_dataset_paths, "transductive_neg50", "Hit@10", neg_ratio_hitk=50)


