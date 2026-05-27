import os
import pandas as pd
import numpy as np
from src.config import Config
from src.data_utils import create_dataset_dict
from src.training import run_experiment
from src.logger import save_df_to_excel

# Initialization and Dataset Loading
cfg = Config()
print("Starting Comprehensive Ablation Study (Train vs Test Fanouts)...")

# Load all Inductive datasets (English and Persian)
ILP_dataset_paths = dict(sorted(create_dataset_dict(cfg.DIR_ILP).items()))

# Define the full list of target inductive datasets
target_datasets = ['nell_v1_ind', 'nell_v2_ind', 'nell_v3_ind', 'nell_v4_ind',
                   'WN18RR_v1_ind', 'WN18RR_v2_ind', 'WN18RR_v3_ind', 'WN18RR_v4_ind',
                   'fb237_v1_ind', 'fb237_v2_ind', 'fb237_v3_ind', 'fb237_v4_ind',
                   'PersianILP-v1_ind' ,'PersianILP-v2_ind', 'PersianILP-v3_ind' ]

# 2. Define Ablation Scenarios for Fanout Configurations
# Each scenario tests the impact of neighborhood depth during training and inference
ablation_scenarios = [{"name": "Light-Weight (2-Layer)", "train": [10, 5], "test": [10, 5]},
                      {"name": "Baseline (3-Layer)", "train": [15, 10, 5], "test": [-1, -1]}, 
                      {"name": "Deep-Train/Full-Test", "train": [20, 15, 10, 5], "test": [-1, -1]},
                      {"name": "Standard-Train/Limited-Test", "train": [15, 10, 5], "test": [15, 10, 5]},
                      {"name": "Heavy-Sampling (5-Layer)", "train": [25, 20, 15, 10, 5], "test": [-1, -1]}]

# Number of runs to ensure statistical significance (Standard for Q1 papers)
num_runs = 3
def run_comprehensive_ablation_Hits10():
    final_results = []

    for dataset_name in target_datasets:
        if dataset_name not in ILP_dataset_paths:
            print(f"Warning: {dataset_name} not found in the specified directory. Skipping...")
            continue
            
        dataset_path = ILP_dataset_paths[dataset_name]
        print(f"\n\n>>> Evaluating Dataset: {dataset_name} <<<")

        for scenario in ablation_scenarios:
            hit_scores = []
            auc_pr_scores = []
            
            print(f"\nScenario: {scenario['name']}")
            print(f"Configuration: Train_Fanout={scenario['train']} | Test_Fanout={scenario['test']}")
            
            for run in range(num_runs):
                print(f"  Iteration {run + 1}/{num_runs}...")
                
                # Execute training and evaluation
                # result={} is passed to ensure independent iteration results
                res_dict = run_experiment( dataset_name=dataset_name,
                                           file_path=dataset_path,
                                           fanouts_train=scenario['train'],
                                           fanouts_test=scenario['test'],
                                           batch_size_train=1024,
                                           batch_size_test=256,
                                           neg_ratio_train=1,
                                           neg_ratio_hitk=50,
                                           neg_ratio_mrr=1,
                                           metric='Hit@10',
                                           learning_rate=0.01,
                                           epochs=50,
                                           result={})

                # Collect metrics from the experiment result
                if dataset_name in res_dict:
                    metrics = res_dict[dataset_name]
                    hit_scores.append(metrics.get('hit_rate_10', 0))
                    auc_pr_scores.append(metrics.get('auc_pr', 0))

            # Store aggregated statistics (Mean and Std Dev)
            final_results.append({
                'Dataset': dataset_name,
                'Scenario': scenario['name'],
                'Train_Fanout': str(scenario['train']),
                'Test_Fanout': str(scenario['test']),
                'Mean_Hit@10': np.mean(hit_scores),
                'Std_Hit@10': np.std(hit_scores),
                'Mean_AUC_PR': np.mean(auc_pr_scores),
                'Std_AUC_PR': np.std(auc_pr_scores)
            })

    # 3. Save Final Results to Excel for Manuscript Documentation
    df = pd.DataFrame(final_results)
    output_file = os.path.join(cfg.RESULT_DIR, "comprehensive_ablation_results.xlsx")
    save_df_to_excel(df, output_file, "Ablation_Fanouts_Hits@10")
    
    print("\n" + "="*60)
    print(f"Ablation Study Completed Successfully.")
    print(f"Results are stored in: {output_file}")
    print("="*60)


# Number of runs to ensure statistical significance (Standard for Q1 papers)
num_runs = 3
def run_comprehensive_ablation_AUCPR():
    final_results = []

    for dataset_name in target_datasets:
        if dataset_name not in ILP_dataset_paths:
            print(f"Warning: {dataset_name} not found in the specified directory. Skipping...")
            continue
            
        dataset_path = ILP_dataset_paths[dataset_name]
        print(f"\n\n>>> Evaluating Dataset: {dataset_name} <<<")

        for scenario in ablation_scenarios:
            hit_scores = []
            auc_pr_scores = []
            
            print(f"\nScenario: {scenario['name']}")
            print(f"Configuration: Train_Fanout={scenario['train']} | Test_Fanout={scenario['test']}")
            
            for run in range(num_runs):
                print(f"  Iteration {run + 1}/{num_runs}...")
                
                # Execute training and evaluation
                # result={} is passed to ensure independent iteration results
                res_dict = run_experiment( dataset_name=dataset_name,
                                           file_path=dataset_path,
                                           fanouts_train=scenario['train'],
                                           fanouts_test=scenario['test'],
                                           batch_size_train=1024,
                                           batch_size_test=256,
                                           neg_ratio_train=1,
                                           neg_ratio_hitk=50,
                                           neg_ratio_mrr=1,
                                           metric='AUC-PR',
                                           learning_rate=0.01,
                                           epochs=50,
                                           result={})

                # Collect metrics from the experiment result
                if dataset_name in res_dict:
                    metrics = res_dict[dataset_name]
                    hit_scores.append(metrics.get('hit_rate_10', 0))
                    auc_pr_scores.append(metrics.get('auc_pr', 0))

            # Store aggregated statistics (Mean and Std Dev)
            final_results.append({
                'Dataset': dataset_name,
                'Scenario': scenario['name'],
                'Train_Fanout': str(scenario['train']),
                'Test_Fanout': str(scenario['test']),
                'Mean_Hit@10': np.mean(hit_scores),
                'Std_Hit@10': np.std(hit_scores),
                'Mean_AUC_PR': np.mean(auc_pr_scores),
                'Std_AUC_PR': np.std(auc_pr_scores)
            })

    # 3. Save Final Results to Excel for Manuscript Documentation
    df = pd.DataFrame(final_results)
    output_file = os.path.join(cfg.RESULT_DIR, "comprehensive_ablation_results.xlsx")
    save_df_to_excel(df, output_file, "Ablation_Fanouts_AUCPR")
    
    print("\n" + "="*60)
    print(f"Ablation Study Completed Successfully.")
    print(f"Results are stored in: {output_file}")
    print("="*60)


run_comprehensive_ablation_Hits10()
run_comprehensive_ablation_AUCPR()