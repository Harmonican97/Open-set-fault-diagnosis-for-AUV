import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from phys_protonet import PhysProtoNet, compute_topology_loss, get_haizhe_similarity_matrix, HaizhePhysicsEngine

def load_dataset(data_path="./dataset"):
    file_map = {
        "0_normal.csv": 0,
        "2_pg.csv": 1,
        "3_pdb.csv": 2
    }
    known_data, known_labels = [], []
    
    for file_name, label in file_map.items():
        file_full_path = os.path.join(data_path, file_name)
        if not os.path.exists(file_full_path):
            raise FileNotFoundError(f"File not found: {file_full_path}")
            
        df = pd.read_csv(file_full_path)
        features = df.iloc[:, 1:].values  # Delete time colummn
        known_data.append(features)
        known_labels.append(np.full(len(features), label))
        
    return known_data, known_labels

def preprocess_and_sample(data_list, label_list, seq_len=50, stride=10):
    scaler = StandardScaler()
    all_data = np.vstack(data_list)
    scaler.fit(all_data)
    
    x_frames, y_frames = [], []
    for data, label in zip(data_list, label_list):
        std_data = scaler.transform(data)
        for i in range(0, len(std_data) - seq_len, stride):
            x_frames.append(std_data[i : i + seq_len].T) # (C, L)
            y_frames.append(label[i])
            
    X = torch.tensor(np.array(x_frames), dtype=torch.float32)
    Y = torch.tensor(np.array(y_frames), dtype=torch.long)
    
    class_counts = np.bincount(Y.numpy())
    class_weights = 1. / class_counts
    weights = class_weights[Y.numpy()]
    sampler = WeightedRandomSampler(weights, len(weights))
    
    loader = DataLoader(TensorDataset(X, Y), batch_size=32, sampler=sampler)
    return loader, scaler

def train_phys_protonet(model, train_loader, epochs=200):
    device = next(model.parameters()).device
    
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=epochs)
    
    num_k = model.prototypes.shape[0] 
    sim_mat = get_haizhe_similarity_matrix(num_k).to(device)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            z, distances = model(inputs)
            
            # 1. cls loss
            l_cls = F.cross_entropy(-distances, labels)
            
            # 2. md loss
            l_v15 = model.compute_v15_loss(z, distances, labels, margin=20.0)
            
            # 3. topo loss
            l_topo = compute_topology_loss(model.prototypes, sim_mat)
            
            total_loss = l_cls + 1.2 * l_v15 + 0.3 * l_topo
            
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += total_loss.item()
            pbar.set_postfix({"Loss": f"{total_loss.item():.3f}", "LR": f"{optimizer.param_groups[0]['lr']:.6f}"})

    model.eval()
    with torch.no_grad():
        all_z, all_l = [], []
        for inputs, labels in train_loader:
            z, _ = model(inputs.to(device))
            all_z.append(z)
            all_l.append(labels)
        all_z = torch.cat(all_z)
        all_l = torch.cat(all_l)
        
        for c in range(3):
            z_c = all_z[all_l == c]
            if len(z_c) > 1:
                model.running_var[c] = torch.var(z_c, dim=0) + 1e-5
    
def main():
    DATA_DIR = "./dataset"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. data load
    raw_data, raw_labels = load_dataset(DATA_DIR)

    # 2. pre-processing
    train_loader, scaler = preprocess_and_sample(raw_data, raw_labels)
    
    # 3. Initialization
    model = PhysProtoNet(input_dim=16, feature_dim=128, num_known_classes=3).to(device)
    
    # 4. train
    train_phys_protonet(model, train_loader, epochs=200)
    
    # 5. save
    torch.save({
        'model_state': model.state_dict(),
        'scaler': scaler
    }, "haizhe_phys_protonet.pth")

if __name__ == "__main__":
    main()