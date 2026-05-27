# main_analysis.py
import os
import pandas as pd
from src.config import Config
from src.logger import save_df_to_excel
from src.analysis_utils import ( get_degrees, 
                                 process_and_group_datasets, 
                                 get_all_degrees )


# 1. Configuration initialization
cfg = Config()
dir_ilp = cfg.DIR_ILP
dir_tlp = cfg.DIR_TLP
output_file = os.path.join(cfg.RESULT_DIR, "analysis_dataset.xlsx")
print("Running Dataset Analysis...")


# 2. Running analyses
# Analysis 1: Degree Analysis
df = get_degrees(dir_ilp).set_index('Dataset').add(get_degrees(dir_tlp).set_index('Dataset'), fill_value=0).reset_index()
save_df_to_excel(df, output_file, "Analysis_1")


# Analysis 2: Grouped Stats
df_ilp = process_and_group_datasets(dir_ilp)
df_tlp = process_and_group_datasets(dir_tlp)
save_df_to_excel(df_ilp, output_file, "Analysis_inductive")
save_df_to_excel(df_tlp, output_file, "Analysis_transductive")


# Analysis 3: All Degrees
result_df = get_all_degrees(dir_ilp, dir_tlp)
save_df_to_excel(result_df, output_file, "degree_counts")
print(f"Analysis saved to {output_file}")