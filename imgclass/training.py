import os
import sys
from time import sleep
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import *
import torch.nn.functional as F
import imgclass.utils as utils
import imgclass.datas as datas
from imgclass.models import *
import time
from tqdm import tqdm
import math
from queue import Queue
import gc
import resource
import json

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device('cpu')

def record_session(model, hyps):
    """
    model: torch nn.Module
        the model to be trained
    hyps: dict
        dict of relevant hyperparameters
    """
    if not os.path.exists(hyps['save_folder']):
        os.mkdir(hyps['save_folder'])
    path = os.path.join(hyps['save_folder'],"hyperparams.txt")
    with open(path,'w') as f:
        f.write(str(model)+'\n')
        for k in sorted(hyps.keys()):
            f.write(str(k) + ": " + str(hyps[k]) + "\n")
    path = os.path.join(hyps['save_folder'],"hyperparams.json")
    with open(path,'w') as f:
        temp_hyps = {k:v for k,v in hyps.items()}
        del temp_hyps['model_class']
        json.dump(temp_hyps, f)

def get_model(hyps):
    """
    hyps: dict
        dict of relevant hyperparameters
    """
    model = hyps['model_class'](**hyps)
    model = model.to(DEVICE)
    return model

def get_optim_objs(hyps, model):
    """
    hyps: dict
        dict of relevant hyperparameters
    model: torch nn.Module
        the model to be trained
    """
    if 'lossfxn' not in hyps:
        hyps['lossfxn'] = "CrossEntropyLoss"
    else:
        loss_fxn = globals()[hyps['lossfxn']]()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyps['lr'],
                                            weight_decay=hyps['l2'])
    b = 'scheduler' in hyps
    b = b and hyps['scheduler'] == "CosineAnnealingLR"
    if b:
        scheduler = globals()[hyps['scheduler']](optimizer,
                                            T_max=hyps['n_epochs'],
                                            eta_min=5e-5)
    else:
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=8)
    return optimizer, scheduler, loss_fxn

def print_train_update(loss, acc, n_loops, i, avg_time):
    s = "Loss: {:.5e} | Acc: {:.5e} | {}/{} | s/iter: {}"
    s = s.format( loss.item(), acc.item(), i, n_loops, avg_time)
    print(s, end="       \r")

def get_data_distrs(hyps):
    dataset = hyps['dataset']
    val_p = hyps['val_p']
    val_loc = hyps['val_loc']
    img_size = hyps['img_shape'][1]
    batch_size = hyps['batch_size']
    n_workers = hyps['n_workers']
    shuffle = hyps['shuffle']
    val_bsize = 1000
    train_data, val_data = datas.get_data_split(dataset, val_p=val_p)

    train_distr = torch.utils.data.DataLoader(train_data,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=n_workers)
    val_distr = torch.utils.data.DataLoader(val_data,
                                            batch_size=val_bsize,
                                            shuffle=False,
                                            num_workers=n_workers)
    n_labels = train_data.n_labels
    return train_distr, val_distr, n_labels


def train(hyps, verbose=False):
    """
    hyps: dict
        all the hyperparameters set by the user
    verbose: bool
        if true, will print status updates during training
    """
    # Initialize miscellaneous parameters 
    torch.cuda.empty_cache()
    batch_size = hyps['batch_size']

    if 'skip_nums' in hyps and hyps['skip_nums'] is not None and\
                              len(hyps['skip_nums']) > 0 and\
                              hyps['exp_num'] in hyps['skip_nums']:

        print("Skipping", hyps['save_folder'])
        results = {"save_folder":hyps['save_folder'], 
                            "Loss":None, "ValAcc":None, 
                            "ValLoss":None, "TestPearson":None}
        return results

    # Get Data and Data Distributers
    train_distr, val_distr, n_labels = get_data_distrs(hyps)

    hyps["n_units"] = n_labels
    model = get_model(hyps)
    print(model)

    record_session(model, hyps)
    train_log = os.path.join(hyps['save_folder'], "training_log.txt")

    # Make optimization objects (lossfxn, optimizer, scheduler)
    optimizer, scheduler, loss_fxn = get_optim_objs(hyps, model)
    if 'gauss_reg' in hyps and hyps['gauss_reg'] > 0:
        gauss_reg = utils.GaussRegularizer(model, [0,6],
                                    std=hyps['gauss_reg'])

    # Training
    for epoch in range(hyps['n_epochs']):
        print("Epoch", epoch, " -- ", hyps['save_folder'])
        n_loops = len(train_distr)
        model.train(mode=True)
        epoch_loss = 0
        epoch_acc = 0
        stats_string = 'Epoch ' + str(epoch) + " -- " +\
                                hyps['save_folder'] + "\n"
        starttime = time.time()

        # Train Loop
        for i,(x,y) in enumerate(train_distr):
            iterstart = time.time()
            optimizer.zero_grad()

            y = y.long().to(DEVICE).squeeze()
            print("y:", y.shape)
            print("x:", x.shape)
            preds = model(x.float().to(DEVICE).contiguous()).squeeze()
            print("preds:", preds.shape)
            loss = loss_fxn(preds.contiguous(), y.contiguous())
            loss.backward()
            optimizer.step()
            argmaxes = torch.argmax(preds, dim=-1)
            acc = (argmaxes.long()==y.long()).float().mean()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            if verbose:
                print_train_update(loss, acc, n_loops, i,
                                    time.time()-iterstart)
            if math.isnan(epoch_loss) or math.isinf(epoch_loss)\
                                    or hyps['exp_name']=="test":
                break
        n_loops = i+1 # Just in case miscalculated

        # Clean Up Train Loop
        avg_loss = epoch_loss/n_loops
        avg_acc = epoch_acc/n_loops
        s = 'Avg Loss: {} | Avg Acc: {} | Time: {}\n'
        stats_string += s.format(avg_loss, avg_acc,
                                time.time()-starttime)
        del x
        del y

        # Validation
        model.eval()
        starttime = time.time()
        with torch.no_grad():
            n_loops = len(val_distr)
            val_loss = 0
            val_acc  = 0
            if verbose:
                print()
                print("Validating")

            # Val Loop
            for i,(x,y) in enumerate(val_distr):
                iterstart = time.time()
                y = y.long().to(DEVICE).squeeze()
                preds = model(x.to(DEVICE)).squeeze()
                loss = loss_fxn(preds, y)
                argmaxes = torch.argmax(preds, dim=-1)
                acc = (argmaxes.long()==y.long()).float().mean()

                val_loss += loss.item()
                val_acc += acc.item()
                if verbose:
                    print_train_update(loss, acc, n_loops, i,
                                        time.time()-iterstart)
                if math.isnan(epoch_loss) or math.isinf(epoch_loss)\
                                        or hyps['exp_name']=="test":
                    break
            print()
            n_loops = i+1 # Just in case miscalculated

            # Validation Evaluation
            val_loss = val_loss/n_loops
            val_acc = val_acc/n_loops
            s = 'Val Loss: {} | Val Acc: {} | Time: {}\n'
            stats_string += s.format(val_loss, val_acc,
                                 time.time()-starttime)

        if 'scheduler' in hyps and hyps['scheduler'] == "CosineAnnealingLR":
            scheduler.step()
        elif 'scheduler' in hyps and hyps['scheduler'] == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        # Save Model Snapshot
        optimizer.zero_grad()
        save_dict = {
            "model_type": hyps['model_type'],
            "hyps":hyps,
            "model_state_dict":model.state_dict(),
            "optim_state_dict":optimizer.state_dict(),
            "loss": avg_loss,
            "acc": avg_acc,
            "epoch":epoch,
            "val_loss":val_loss,
            "val_acc":val_acc
        }
        for k in hyps.keys():
            if k not in save_dict:
                save_dict[k] = hyps[k]
        io.save_checkpoint(save_dict, hyps['save_folder'],
                                            hyps['exp_id'],
                                            del_prev=True)

        # Print Epoch Stats
        gc.collect()
        max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        stats_string += "Memory Used: {:.2f} mb".format(
                                              max_mem_used/1024)+"\n"
        print(stats_string)
        with open(train_log,'a') as f:
            f.write(stats_string+"\n")
        # If loss is nan, training is futile
        if math.isnan(avg_loss) or math.isinf(avg_loss) or\
                                            hyps['exp_name']=="test":
            break

    # Final save
    results = {
                "save_folder":hyps['save_folder'], 
                "Loss":avg_loss, 
                "Acc":avg_acc,
                "ValAcc":val_acc, 
                "ValLoss":val_loss 
                }
    with open(hyps['save_folder'] + "/hyperparams.txt",'a') as f:
        s = " ".join([str(k)+":"+str(results[k]) for k in\
                                        sorted(results.keys())])
        s = "\n" + s + '\n'
        f.write(s)
    return results

def fill_hyper_q(hyps, hyp_ranges, keys, hyper_q, idx=0):
    """
    Recursive function to load each of the hyperparameter combinations 
    onto a queue.

    hyps - dict of hyperparameters created by a HyperParameters object
        type: dict
        keys: name of hyperparameter
        values: value of hyperparameter
    hyp_ranges - dict of ranges for hyperparameters to take over the search
        type: dict
        keys: name of hyperparameters to be searched over
        values: list of values to search over for that hyperparameter
    keys - keys of the hyperparameters to be searched over. Used to
            specify order of keys to search
    train - method that handles training of model. Should return a dict of results.
    hyper_q - Queue to hold all parameter sets
    idx - the index of the current key to be searched over
    """
    # Base call, runs the training and saves the result
    if idx >= len(keys):
        if 'exp_num' not in hyps:
            if 'starting_exp_num' not in hyps: hyps['starting_exp_num'] = 0
            hyps['exp_num'] = hyps['starting_exp_num']
        hyps['save_folder'] = hyps['exp_name'] + "/" + hyps['exp_name'] +"_"+ str(hyps['exp_num']) 
        for k in keys:
            hyps['save_folder'] += "_" + str(k)+str(hyps[k])

        hyps['model_class'] = globals()[hyps['model_type']]

        # Load q
        hyper_q.put([{k:v for k,v in hyps.items()}])
        hyps['exp_num'] += 1

    # Non-base call. Sets a hyperparameter to a new search value and passes down the dict.
    else:
        key = keys[idx]
        for param in hyp_ranges[key]:
            hyps[key] = param
            hyper_q = fill_hyper_q(hyps, hyp_ranges, keys, hyper_q, idx+1)
    return hyper_q

def get_device(visible_devices, cuda_buffer=3000):
    info = tdrutils.get_cuda_info()
    for i,mem_dict in enumerate(info):
        if i in visible_devices and mem_dict['remaining_mem'] >= cuda_buffer:
            return i
    return -1

def hyper_search(hyps, hyp_ranges, keys, device, early_stopping=10, stop_tolerance=.01):
    starttime = time.time()
    # Make results file
    if not os.path.exists(hyps['exp_name']):
        os.mkdir(hyps['exp_name'])
    results_file = hyps['exp_name']+"/results.txt"
    with open(results_file,'a') as f:
        f.write("Hyperparameters:\n")
        for k in hyps.keys():
            if k not in hyp_ranges:
                f.write(str(k) + ": " + str(hyps[k]) + '\n')
        f.write("\nHyperranges:\n")
        for k in hyp_ranges.keys():
            f.write(str(k) + ": [" + ",".join([str(v) for v in hyp_ranges[k]])+']\n')
        f.write('\n')
    
    hyper_q = Queue()
    
    hyper_q = fill_hyper_q(hyps, hyp_ranges, keys, hyper_q, idx=0)
    total_searches = hyper_q.qsize()
    print("n_searches:", total_searches)

    result_count = 0
    print("Starting Hyperloop")
    while not hyper_q.empty():
        print("Searches left:", hyper_q.qsize(),"-- Running Time:", time.time()-starttime)
        print()
        hyperset = hyper_q.get()
        results = train(*hyperset, verbose=True)
        with open(results_file,'a') as f:
            results = " -- ".join([str(k)+":"+str(results[k]) for k in sorted(results.keys())])
            f.write("\n"+results+"\n")

