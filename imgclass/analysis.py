import sys
import numpy as np
import torch
from imgclass.models import *
import imgclass.utils as utils
import os
import pandas as pd
import matplotlib.pyplot as plt

def get_model_folders(main_folder):
    """
    Returns a list of paths to the model folders contained within the argued main_folder

    main_folder - str
        path to main folder
    """
    folders = []
    for d, sub_ds, files in os.walk(main_folder):
        for sub_d in sub_ds:
            contents = os.listdir(os.path.join(d,sub_d))
            for content in contents:
                if ".pt" in content:
                    folders.append(sub_d)
                    break
    return sorted(folders,key=lambda x: int(x.split("/")[-1].split("_")[1]))

def extract_hypstxt(path):
    """
    Gets the hyperparameters from the corresponding hyperparams.txt file

    path: str
        path to the txt file
    """
    hyps = dict()
    with open(path, 'r') as hypfile:
        for line in hypfile:
            if ("(" in line and ")" in line) or ":" not in line:
                continue
            splt = line.strip().split(":")
            if len(splt) > 2: 
                continue
            splt[0] = splt[0].strip()
            splt[1] = splt[1].strip()
            hyps[splt[0]] = splt[1]
            if hyps[splt[0]].lower() == "true" or hyps[splt[0]].lower() == "false":
                hyps[splt[0]] = hyps[splt[0]] == "true"
            elif hyps[splt[0]] == "None":
                hyps[splt[0]] = None
            elif splt[0] in {"lr", "l1", 'l2', 'noise', 'bnorm_momentum'}:
                hyps[splt[0]] = float(splt[1])
            elif splt[0] in {"n_epochs", "exp_num", 'batch_size'}:
                hyps[splt[0]] = int(splt[1])
            elif splt[0] in {"img_shape", "chans"} and "," in splt[1]:
                temp = splt[1].replace('[','').replace(']','').split(",")
                hyps[splt[0]] = [int(x.strip()) for x in temp]
    return hyps

def get_hyps(folder):
    try:
        folder = os.path.expanduser(folder)
        path = os.path.join(folder, "hyperparams.json")
        return utils.load_json(path)
    except Exception as e:
        try:
            path = os.path.join(main_dir, folder)
            stream = stream_folder(path)
            hyps = dict()
            for k in stream.keys():
                if "state_dict" not in k and "model_hyps" not in k:
                    hyps[k] = stream[k]
            assert 'lr' in hyps 
            return hyps
        except Exception as ee:
            path = os.path.join(main_dir, folder, "hyperparams.txt")
            return extract_hypstxt(path)

def get_analysis_table(folder, hyps=None):
    """
    Returns a dict that can easily be converted into a dataframe
    """
    table = dict()
    if hyps is None:
        hyps = get_hyps(folder)
    for k,v in hyps.items():
        if "state_dict" not in k and "model_hyps" not in k:
            table[k] = [v]
    return table

def read_model_file(file_name):
    """
    file_name: str
        path to the model save file. The save should contain "model_hyps", "model_state_dict",
        and "model_type" as keys.
    """
    data = torch.load(file_name, map_location="cpu")
    model_type = data['model_type']
    model = globals()[model_type](**data['model_hyps'])
    if "n_shakes" in data and data['n_shakes'] > 1:
        sd = data['model_state_dict']
        for name, param in model.named_parameters():
            if "alpha" in name:
                param.data = sd[name]
    model.load_state_dict(data['model_state_dict'])
    return model

def read_model(folder, ret_metrics=False):
    """
    Recreates model architecture and loads the saved statedict from a model folder

    folder - str
        path to folder that contains model checkpoints
    ret_metrics - bool
        if true, returns the recorded training metric history (i.e. val loss, val acc, etc)
    """
    metrics = dict()
    try:
        _, _, fs = next(os.walk(folder.strip()))
    except Exception as e:
        print(e)
        print("It is likely that folder", folder.strip(),"does not exist")
        assert False
    for i in range(len(fs)+100):
        f = os.path.join(folder.strip(),"epoch_{0}.pt".format(i))
        try:
            with open(f, "rb") as fd:
                data = torch.load(fd, map_location=torch.device("cpu"))
            if ret_metrics:
                for k,v in data.items():
                    if k == "loss" or k == "epoch" or k == "val_loss" or k == "val_acc" or\
                       k == "acc":
                        if k not in metrics:
                            metrics[k] = [v]
                        else:
                            metrics[k].append(v)
        except Exception as e:
            pass

    model_type = data['model_type']
    model = globals()[model_type](**data['model_hyps'])
    if "n_shakes" in data and data['n_shakes'] > 1:
        sd = data['model_state_dict']
        for name, param in model.named_parameters():
            if "alpha" in name:
                param.data = sd[name]
    model.load_state_dict(data['model_state_dict'])
    if ret_metrics:
        return model, metrics
    return model

def get_analysis_figs(folder, metrics):
    assert 'epoch' in metrics
    if 'loss' in metrics and 'val_loss' in metrics:
        fig = plt.figure()
        plt.plot(metrics['epoch'], metrics['loss'],color='k')
        plt.plot(metrics['epoch'], metrics['val_loss'],color='b')
        plt.legend(["train", "validation"])
        plt.title("Loss Curves")
        plt.savefig(os.path.join(folder,'loss_curves.png'))

    if 'acc' in metrics and 'val_acc' in metrics:
        fig = plt.figure()
        plt.plot(metrics['epoch'], metrics['acc'],color='k')
        plt.plot(metrics['epoch'], metrics['val_acc'],color='b')
        plt.legend(["train", "validation"])
        plt.title("Accuracies")
        plt.savefig(os.path.join(folder,'acc_curves.png'))

def analyze_model(folder, verbose=True):
    """
    Calculates model performance on the testset and calculates interneuron correlations.

    folder: str
        the folder full of checkpoints
    """
    hyps = get_hyps(folder)
    table = get_analysis_table(folder, hyps=hyps)

    model,metrics = read_model(folder,ret_metrics=True)
    get_analysis_figs(folder, metrics)

    train_acc, train_loss = metrics['acc'][-1], metrics['loss'][-1]
    table['train_acc'] =  [train_acc]
    table['train_loss'] = [train_loss]
    val_acc, val_loss = metrics['val_acc'][-1], metrics['val_loss'][-1]
    table['val_acc'] = [val_acc]
    table['val_loss'] = [val_loss]
    if verbose:
        print("ValAcc: {:05e}, ValLoss: {:05e}".format(val_acc, val_loss))
    return pd.DataFrame(table)

def analysis_pipeline(main_folder, verbose=True):
    """
    Evaluates model on test set, calculates interneuron correlations, 
    and creates figures.

    main_folder: str
        the folder full of model folders that contain checkpoints
    """
    model_folders = get_model_folders(main_folder)
    csv = 'model_data.csv'
    csv_path = os.path.join(main_folder,csv)
    if os.path.exists(csv_path):
        main_df = pd.read_csv(csv_path, sep="!")
    else:
        main_df = dict()
    for folder in model_folders:
        save_folder = os.path.join(main_folder, folder)
        if "save_folder" in main_df and save_folder in set(main_df['save_folder']):
            if verbose:
                print("Skipping",folder," due to previous record")
            continue
        if verbose:
            print("\n\nAnalyzing", folder)
        
        df = analyze_model(save_folder, verbose=verbose)
        if isinstance(main_df, dict):
            main_df = df
        else:
            main_df = main_df.append(df, sort=True)
    return main_df


