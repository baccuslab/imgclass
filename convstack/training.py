import os
import sys
from time import sleep
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import convstack.utils as utils
import convstack.datas as datas
from convstack.models import *
import time
from tqdm import tqdm
import math
from queue import Queue
import gc
import resource
import json

DEVICE = torch.device("cuda:0")

def record_session(model, hyps, model_hyps):
    """
    model: torch nn.Module
        the model to be trained
    hyps: dict
        dict of relevant hyperparameters
    model_hyps: dict
        dict of relevant hyperparameters specific to creating the model architecture
    """
    if not os.path.exists(hyps['save_folder']):
        os.mkdir(hyps['save_folder'])
    with open(os.path.join(hyps['save_folder'],"hyperparams.txt"),'w') as f:
        f.write(str(model)+'\n')
        for k in sorted(hyps.keys()):
            f.write(str(k) + ": " + str(hyps[k]) + "\n")
    with open(os.path.join(hyps['save_folder'],"hyperparams.json"),'w') as f:
        temp_hyps = {k:v for k,v in hyps.items()}
        del temp_hyps['model_class']
        json.dump(temp_hyps, f)
    with open(os.path.join(hyps['save_folder'],"model_hyps.json"),'w') as f:
        temp_hyps = {k:v for k,v in model_hyps.items()}
        if 'model_class' in temp_hyps:
            del temp_hyps['model_class']
        json.dump(temp_hyps, f)

def get_model(model_hyps):
    """
    model_hyps: dict
        dict of relevant hyperparameters
    """
    model = model_hyps['model_class'](**model_hyps)
    model = model.to(hyps['device'])
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
    optimizer = torch.optim.Adam(model.parameters(), lr=hyps['lr'], weight_decay=hyps['l2'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.1)
    return optimizer, scheduler, loss_fxn

def print_train_update(loss, acc, n_loops, i):
    s = "Loss: {:.5e} | Acc: {:.5e} | {}/{}".format(loss.item(), acc.item(), i,n_loops)
    print(s, end="       \r")

def train(hyps, model_hyps, verbose=False):
    """
    hyps: dict
        all the hyperparameters set by the user
    model_hyps: dict
        the hyperparameters specific to creating the model architecture
    verbose: bool
        if true, will print status updates during training
    """
    # Initialize miscellaneous parameters 
    torch.cuda.empty_cache()
    hyps['device'] = DEVICE
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
    datapath = hyps['datapath']
    val_p = hyps['val_p']
    val_loc = hyps['val_loc']
    img_size = hyps['img_shape'][1]
    train_data, val_data, label_distribution = datas.train_val_split(datapath, val_p=val_p,
                                                         val_loc=val_loc,img_size=img_size)
    batch_size = hyps['batch_size']
    n_workers = hyps['n_workers']
    shuffle = hyps['shuffle']
    train_distr = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                            shuffle=shuffle, num_workers=n_workers)
    val_distr = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False,
                                                              num_workers=n_workers)

    model_hyps["n_units"] = train_distr.n_labels
    model = get_model(model_hyps)

    record_session(model, hyps, model_hyps)

    # Make optimization objects (lossfxn, optimizer, scheduler)
    optimizer, scheduler, loss_fxn = get_optim_objs(hyps, model, train_data.centers)
    if 'gauss_reg' in hyps and hyps['gauss_reg'] > 0:
        gauss_reg = utils.GaussRegularizer(model, [0,6], std=hyps['gauss_reg'])

    # Training
    for epoch in range(hyps['n_epochs']):
        print("Beginning Epoch", epoch, " -- ", hyps['save_folder'])
        print()
        n_loops = len(train_data)/batch_size
        n_loops = int(n_loops) + int(n_loops==int(n_loops))
        model.train(mode=True)
        epoch_loss = 0
        stats_string = 'Epoch ' + str(epoch) + " -- " + hyps['save_folder'] + "\n"
        starttime = time.time()

        # Train Loop
        for i,(x,y) in enumerate(train_distr):
            optimizer.zero_grad()

            y = y.long().to(DEVICE)
            preds = model(x.to(DEVICE))
            loss = loss_fxn(preds, y)
            if 'gauss_reg' in hyps and hyps['gauss_reg'] > 0:
                loss += hyps['gauss_loss_coef']*gauss_reg.get_loss()
            loss.backward()
            optimizer.step()
            argmaxes = torch.argmax(preds, dim=-1)
            acc = (argmaxes==y).mean()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            if verbose:
                print_train_update(loss, acc, n_loops, i)
            if math.isnan(epoch_loss) or math.isinf(epoch_loss) or hyps['exp_name']=="test":
                break
        n_loops = i+1 # Just in case miscalculated

        # Clean Up Train Loop
        avg_loss = epoch_loss/n_loops
        avg_acc = epoch_acc/n_loops
        stats_string += 'Avg Loss: {} | Avg Acc: {} | Time: {}\n'.format(avg_loss, avg_acc,
                                                                     time.time()-starttime)
        del x
        del y

        # Validation
        model.eval()
        starttime = time.time()
        with torch.no_grad():
            n_loops = len(val_distr)/step_size
            n_loops = int(n_loops) + int(n_loops==int(n_loops))
            if verbose:
                print()
                print("Validating")

            # Val Loop
            for i,(x,y) in enumerate(val_distr):
                y = y.long().to(DEVICE)
                preds = model(x.to(DEVICE))
                loss = loss_fxn(preds, y)
                argmaxes = torch.argmax(preds, dim=-1)
                acc = (argmaxes==y).mean()

                val_loss += loss.item()
                val_acc += acc.item()
                if verbose:
                    print_train_update(loss, acc, n_loops, i)
                if math.isnan(epoch_loss) or math.isinf(epoch_loss) or hyps['exp_name']=="test":
                    break
            n_loops = i+1 # Just in case miscalculated

            # Validation Evaluation
            val_loss = val_loss/n_loops
            val_acc = val_acc/n_loops
            stats_string += 'Val Loss: {} | Val Acc: {} | Time: {}\n'.format(val_loss, val_acc,
                                                                     time.time()-starttime)
            scheduler.step(val_loss)

        # Save Model Snapshot
        optimizer.zero_grad()
        save_dict = {
            "model_type": hyps['model_type'],
            "model_hyps": model_hyps,
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
        utils.save_checkpoint(save_dict, hyps['save_folder'], del_prev=True)

        # Print Epoch Stats
        gc.collect()
        max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        stats_string += "Memory Used: {:.2f} mb".format(max_mem_used / 1024)+"\n"
        print(stats_string)
        # If loss is nan, training is futile
        if math.isnan(avg_loss) or math.isinf(avg_loss) or stop:
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
        s = " ".join([str(k)+":"+str(results[k]) for k in sorted(results.keys())])
        s = "\n" + s + '\n'
        f.write(s)
    return results

def get_model_hyps(hyps):
    model_hyps = {k:v for k,v in hyps.items()}

    fn_args = set(hyps['model_class'].__init__.__code__.co_varnames) 
    if "kwargs" in fn_args:
        fn_args = fn_args | set(TDRModel.__init__.__code__.co_varnames)
    keys = list(model_hyps.keys())
    for k in keys:
        if k not in fn_args:
            del model_hyps[k]
    return model_hyps

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
        model_hyps = get_model_hyps(hyps)
        model_hyps['model_class'] = globals()[hyps['model_type']]

        # Load q
        hyper_q.put([{k:v for k,v in hyps.items()}, {k:v for k,v in model_hyps.items()}])
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

    trainer = Trainer(early_stopping=early_stopping, stop_tolerance=stop_tolerance)

    result_count = 0
    print("Starting Hyperloop")
    while not hyper_q.empty():
        print("Searches left:", hyper_q.qsize(),"-- Running Time:", time.time()-starttime)
        hyperset = hyper_q.get()
        hyperset.append(device)
        results = trainer.train(*hyperset, verbose=True)
        with open(results_file,'a') as f:
            results = " -- ".join([str(k)+":"+str(results[k]) for k in sorted(results.keys())])
            f.write("\n"+results+"\n")

