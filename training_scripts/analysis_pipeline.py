"""
This script is made to automate the analysis of the model performance for a batch of models.
You must give a command line argument of the model search folder to be analyzed.

$ python3 search_analysis.py bncnn

"""
import numpy as np
import torch
import pandas as pd
import convstack as cstk
import sys
import os

if __name__ == "__main__":
    start_idx = None
    if len(sys.argv) >= 2:
        try:
            start_idx = int(sys.argv[1])
            grand_folders = sys.argv[2:]
        except:
            grand_folders = sys.argv[1:]
    torch.cuda.empty_cache()
    for grand_folder in grand_folders:
        print("Analyzing", grand_folder)
        df = cstk.analysis.analysis_pipeline(grand_folder, verbose=True)
        df.to_csv(os.path.join(grand_folder,"model_data.csv"),sep="!", index=False, header=True)



