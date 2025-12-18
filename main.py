import os, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from datautils import load_npy_safely, make_data
from tqdm import tqdm
from utils import save_metrics
from models import TwoHeadNet, baselineModel, conv3SingleHeadNet, conv4SingleHeadNet, conv5SingleHeadNet, conv6SingleHeadNet
import argparse
import random
import pandas as pd

# use this for one-out ablation
def train_loop(
    weights,
    loss_path="loss",
    metr_path="metrics",
    name="twohead",
    epochs=50,
    gpu=0,
    weight_decay=1e-4,
    dataset='denoise',
    model=None,
    seed=42
):
    # added for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

    metr_path=loss_path

    if dataset == 'denoise':
        TRAIN_CLEAN = "data/EOG_EEG_train_output.npy"
        TRAIN_CONT  = "data/EOG_EEG_train_input.npy"
        TEST_CLEAN  = "data/EOG_EEG_test_output.npy"
        TEST_CONT   = "data/EOG_EEG_test_input.npy"
    else:
        raise NotImplementedError

    FS = 256
    EPOCH_S = 2.0
    N_SAMPLES = int(FS * EPOCH_S)

    np.set_printoptions(suppress=True, floatmode="fixed", linewidth=120)

    if gpu == 'cpu':
        device = "cpu"
    else:
        device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}')

    S_tr = load_npy_safely(TRAIN_CLEAN, N_SAMPLES=N_SAMPLES)
    X_tr = load_npy_safely(TRAIN_CONT, N_SAMPLES=N_SAMPLES)
    S_te = load_npy_safely(TEST_CLEAN, N_SAMPLES=N_SAMPLES)
    X_te = load_npy_safely(TEST_CONT, N_SAMPLES=N_SAMPLES)

    print("Train:", X_tr.shape, "Test:", X_te.shape, "(memmap:", isinstance(X_tr, np.memmap), ")")

    ds_trn, ds_val, ds_tst = make_data(X_tr, S_tr, X_te, S_te, seed=seed)

    train_losses, val_losses = [], []

    if model is None:
        if name == "twohead":
            model = TwoHeadNet(ch=24, device=device).to(device)
        elif name in ("baseline", "A0"):
            model = baselineModel(ch=24, device=device).to(device)
        # elif name in ("onehead", "A1.5"):
        #     model = SingleHeadNet(ch=24, device=device).to(device)
        # elif name in ('bnormtwohead', "A5"):
        #     model = bNormTwoHeadNet(ch=24, device=device).to(device)
    
    else:
        model = model.to(device)

    num_epochs = epochs

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'CURRENT PARAMS - {params}')

    os.makedirs(f"plots_seed{seed}", exist_ok=True)
    os.makedirs(f"metrics_new_seed{seed}", exist_ok=True)
    os.makedirs(f"models_folder_new_seed{seed}", exist_ok=True)

    LOSS_GRAPH_PATH = f"plots_seed{seed}/{loss_path}_{model.__class__.__name__}_{dataset}_epochs_{num_epochs}_{params}_params.png"
    METRICS_PATH = f"metrics_new_seed{seed}/{metr_path}_{model.__class__.__name__}_{dataset}_epochs_{num_epochs}_metrics_{params}_params.txt"
    METRICS_PATH_BCI = f"metrics_new_seed{seed}/{metr_path}_{model.__class__.__name__}_{dataset}_epochs_{num_epochs}_bci_metrics_{params}_params.txt"

    if os.path.exists(METRICS_PATH):
        print('skipping')
        return

    optim = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=weight_decay)
    # sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=30 )
    best_val, best_state = 1e9, None

    for epoch in range(1, num_epochs + 1):
        model.train()
        logs = []
        pbar = tqdm(ds_trn, desc=f"Epoch {epoch:02d}", leave=False)
        for batch in pbar:
            optim.zero_grad()
            out = model(batch["x"].to(device)[:, None, :])
            L, d = model.loss(batch, out, weights)
            L.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            logs.append(d["L_clean"])
            pbar.set_postfix(train_loss=np.mean(logs))

        train_loss = float(np.mean(logs))

        model.eval()
        with torch.no_grad():
            vloss = []
            for batch in ds_val:
                out = model(batch["x"].to(device)[:, None, :])
                L, _ = model.loss(batch, out, weights)
                vloss.append(float(L.item()))
            val_loss = float(np.mean(vloss))

        # sched.step()
        # optim.step()
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:02d}  train_clean_MSE={train_loss:.5f}  val_total={val_loss:.5f}")
        if val_loss < best_val - 1e-4:
            best_val, best_state = val_loss, {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(model, f"models_folder_new_seed{seed}/best_{model.__class__.__name__}_{loss_path}_{dataset}_epoch_{num_epochs}_weights_{params}_params.pth")

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(train_losses, 'b-', label="Train Loss")
    ax2.plot(val_losses, 'r-', label="Val Loss")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color='b')
    ax2.set_ylabel("Val Loss", color='r')

    plt.title("Training and Validation Loss")
    fig.tight_layout()
    plt.savefig(LOSS_GRAPH_PATH)
    plt.close()

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f'correct shapes - x: {X_te.shape}, s: {S_te.shape}')
    save_metrics(METRICS_PATH, model, ds_tst, X_te, S_te, weights)
    print('denoise metrics saved.')
    X_gen_te = load_npy_safely("data_mwn/conts_snr_-3_3_test.npy", N_SAMPLES=N_SAMPLES, znorm=True)
    X_gen_te = X_gen_te[np.random.choice(np.arange(X_gen_te.shape[0]), size=4000), :]
    S_gen_te = load_npy_safely("data_mwn/cleans_snr_-3_3_test.npy", N_SAMPLES=N_SAMPLES, znorm=True)
    S_gen_te = S_gen_te[np.random.choice(np.arange(S_gen_te.shape[0]), size=4000), :]
    _, _, ds_gen_te = make_data(X_tr, S_tr, X_gen_te, S_gen_te)
    print(f'shapes - x: {X_gen_te.shape}, s: {S_gen_te.shape}')
    save_metrics(METRICS_PATH_BCI, model, ds_gen_te, X_gen_te, S_gen_te)
    print('bci metrics saved.')

def channelTrainLoop(ch, group_vals, scale_vals, mse_only_w, gpu=0):
    for g in group_vals:
        if g > ch:
            continue
        for s in scale_vals:
            print(f'Training ch{ch}_grp{g}_scale{s}')
            model3 = conv3SingleHeadNet(ch=ch, device=f'cuda:{gpu}', groups=g, scale_up=s)
            train_loop(mse_only_w, loss_path=f'final_3conv_ch{ch}_grp{g}_scale{s}_mse_only', metr_path=f'final_3conv_ch{ch}_grp{g}_scale{s}_mse_only', epochs=50, gpu=gpu, dataset='denoise', model=model3)
            model4 = conv4SingleHeadNet(ch=ch, device=f'cuda:{gpu}', groups=g, scale_up=s)
            train_loop(mse_only_w, loss_path=f'final_4conv_ch{ch}_grp{g}_scale{s}_mse_only', metr_path=f'final_4conv_ch{ch}_grp{g}_scale{s}_mse_only', epochs=50, gpu=gpu, dataset='denoise', model=model4)
            model5 = conv5SingleHeadNet(ch=ch, device=f'cuda:{gpu}', groups=g, scale_up=s)
            train_loop(mse_only_w, loss_path=f'final_5conv_ch{ch}_grp{g}_scale{s}_mse_only', metr_path=f'final_5conv_ch{ch}_grp{g}_scale{s}_mse_only', epochs=50, gpu=gpu, dataset='denoise', model=model5)
            model6 = conv6SingleHeadNet(ch=ch, device=f'cuda:{gpu}', groups=g, scale_up=s)
            train_loop(mse_only_w, loss_path=f'final_6conv_ch{ch}_grp{g}_scale{s}_mse_only', metr_path=f'final_6conv_ch{ch}_grp{g}_scale{s}_mse_only', epochs=50, gpu=gpu, dataset='denoise', model=model6)

 

if __name__=="__main__":
    gpu=5
    seed = 99

    # dont change this, deprecated feature
    mse_only_w = {'L_clean':1.0}

    # channel_vals = [1, 2, 4, 8]
    # group_vals = [1, 2, 4, 8]
    # scale_vals = [1, 2, 4, 8]

    # channelTrainLoop(channel_vals[0], group_vals, scale_vals, mse_only_w, gpu=gpu)
    # channelTrainLoop(channel_vals[1], group_vals, scale_vals, mse_only_w, gpu=gpu)
    # channelTrainLoop(channel_vals[2], group_vals, scale_vals, mse_only_w, gpu=gpu)
    # channelTrainLoop(channel_vals[3], group_vals, scale_vals, mse_only_w, gpu=gpu)

    # to reproduce 143 models
    metr = pd.read_csv('FINAL_COMBINED_METRICS_143.csv')
    metr = metr[['model_type', 'chans', 'groups', 'scale']]

    for idx, row in metr.iterrows():
        model_type = row['model_type']
        ch = row['chans']
        g = row['groups']
        s = row['scale']
        print(f'Training {idx}. {model_type}_ch{ch}_grp{g}_scale{s}')
        if model_type == '3conv':
            model = conv3SingleHeadNet(ch=ch, device=f'cuda:{gpu}', groups=g, scale_up=s)
        elif model_type == '4conv':
            model = conv4SingleHeadNet(ch=ch, device=f'cuda:{gpu}', groups=g, scale_up=s)
        elif model_type == '5conv':
            model = conv5SingleHeadNet(ch=ch, device=f'cuda:{gpu}', groups=g, scale_up=s)
        elif model_type == '6conv':
            model = conv6SingleHeadNet(ch=ch, device=f'cuda:{gpu}', groups=g, scale_up=s)
        else:
            raise
            continue
        
        train_loop(mse_only_w, loss_path=f'seed{seed}_final_{model_type}_ch{ch}_grp{g}_scale{s}_denoise', metr_path=f'seed{seed}_final_{model_type}_ch{ch}_grp{g}_scale{s}_denoise', epochs=50, gpu=gpu, dataset='denoise', model=model, seed=seed)
