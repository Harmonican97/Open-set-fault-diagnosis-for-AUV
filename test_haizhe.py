import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset

from phys_protonet import PhysProtoNet, HaizhePhysicsEngine


def draw_cm(model_name, cm, class_names):
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percentage = (cm.astype('float') / row_sums) * 100
    cm_percentage = np.round(cm_percentage, 2)

    annot = np.empty_like(cm_percentage, dtype=object)
    for i in range(cm_percentage.shape[0]):
        for j in range(cm_percentage.shape[1]):
            if cm_percentage[i, j] == 0:
                annot[i, j] = "0%"
            else:
                annot[i, j] = "{:.2f}%".format(cm_percentage[i, j])

    # Visualize confusion matrix
    mpl.rc('font', family='Times New Roman', weight='bold')
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'

    fig_cm = plt.figure(figsize=(10, 8))
    
    sns.heatmap(cm_percentage, 
            annot=annot, 
            fmt="", 
            cmap="Blues", 
            xticklabels=class_names, 
            yticklabels=class_names,
            annot_kws={'size': 16, 'fontweight': 'bold'})
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted", fontsize=18, fontweight='bold')
    plt.ylabel("True", fontsize=18, fontweight='bold')
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    save_dir = './result_statistics/' + model_name + '_confusion_matrix.png'

    fig_cm.savefig(save_dir)
    plt.close(fig_cm)

    return

def plot_latent_space_new(z_feats, y_true, class_names=None, save_path="v15_final_tsne.png", model_name='Phys-ProtoNet'):
    
    unique_labels = np.unique(y_true)
    if class_names is None:
        class_names = [f"Fault {int(i)}" for i in unique_labels]
    
    tsne = TSNE(n_components=2, perplexity=45, init='pca', random_state=42)
    embedded = tsne.fit_transform(z_feats)
    
    mpl.rc('font', family='Times New Roman', weight='bold')
    
    plt.figure(figsize=(12, 10))
    
    for i, class_name in enumerate(class_names):
        idx = y_true == unique_labels[i]
        plt.scatter(
            embedded[idx, 0], 
            embedded[idx, 1], 
            label=class_name,
            alpha=0.7,
            edgecolors='w',
            s=60
        )
    
    plt.title(f't-SNE Visualization of {model_name} Features', fontsize=24, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=22, fontweight='bold')
    plt.ylabel('t-SNE Dimension 2', fontsize=22, fontweight='bold')
    
    plt.xticks(fontsize=18, fontweight='bold')
    plt.yticks(fontsize=18, fontweight='bold')
    
    plt.legend(title="Fault Types", title_fontsize='14', fontsize=14, loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved to: {save_path}")
    

def visualize_features_tsne(features, labels, class_names=None, save_path=None, model_name='models'):
    """
    Visualize high-dimensional features using t-SNE
    """
    if class_names is None:
        class_names = [f"Fault {i}" for i in range(len(np.unique(labels)))]
    
    # Apply t-SNE dimensionality reduction
    print("Applying t-SNE dimensionality reduction in " + model_name + "...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=45)
    features_tsne = tsne.fit_transform(features)
    
    mpl.rc('font', family='Times New Roman', weight='bold')
    # Plot the t-SNE visualization
    visualziation_tsne = plt.figure(figsize=(12, 10))
    for i, class_name in enumerate(class_names):
        idx = labels == i
        plt.scatter(
            features_tsne[idx, 0], 
            features_tsne[idx, 1], 
            label=class_name,
            alpha=0.7,
            edgecolors='w'
        )
    
    plt.legend(title="Fault Types", title_fontsize='14', fontsize=14)
    plt.title('t-SNE Visualization of ' + model_name + ' Features', fontsize=24, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=22, fontweight='bold')
    plt.ylabel('t-SNE Dimension 2', fontsize=22, fontweight='bold')
    plt.xticks(fontsize=18, fontweight='bold')
    plt.yticks(fontsize=18, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.close(visualziation_tsne)
    
    return features_tsne

MASS_EFF = 6.5737 
B_MINUS_G = 3.86

def create_loaders(data_path, scaler, seq_len=50):

    known_files = {"0_normal.csv": 0, "2_pg.csv": 1, "3_pdb.csv": 2}
    all_files = {**known_files, "1_ad.csv": -1, "4_pds.csv": -1}
    
    def get_tensor_data(file_dict):
        x_list, y_list = [], []
        for fname, label in file_dict.items():
            path = os.path.join(data_path, fname)
            if os.path.exists(path):
                df = pd.read_csv(path)
                data = scaler.transform(df.iloc[:, 1:].values)
                for i in range(0, len(data) - seq_len, 20):
                    x_list.append(data[i:i+seq_len].T)
                    y_list.append(label)
        return torch.tensor(np.array(x_list), dtype=torch.float32), torch.tensor(np.array(y_list), dtype=torch.long)

    x_train, y_train = get_tensor_data(known_files)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=False)
    x_test, y_test = get_tensor_data(all_files)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=32, shuffle=False)
    return train_loader, test_loader

def calibrate_thresholds_v15(model, train_loader, device):
    model.eval()
    all_mds, all_resids = [], []
    with torch.no_grad():
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            # physics residual
            res_z = HaizhePhysicsEngine.get_batch_residuals(inputs)[:, 12]
            # MD
            z, dists = model(inputs)
            _, preds = torch.min(dists, dim=1)
            
            batch_vars = model.running_var[preds]
            diff = z - model.prototypes[preds]
            mahalanobis_d = torch.sum((diff**2) / batch_vars, dim=1)
            
            all_mds.append(mahalanobis_d.cpu().numpy())
            all_resids.append(res_z.cpu().numpy())
            
    md_th = np.percentile(np.concatenate(all_mds), 95)
    p_th = np.percentile(np.concatenate(all_resids), 90) 
    return md_th, p_th

def run_evaluation(model, test_loader, md_th, p_th, device):
    model.eval()
    raw_results = []
    all_z = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            res_z = HaizhePhysicsEngine.get_batch_residuals(inputs)[:, 12].cpu().numpy()
            z, dists = model(inputs)
            _, preds = torch.min(dists, dim=1)
            
            # MD calculation
            batch_vars = model.running_var[preds]
            diff = z - model.prototypes[preds]
            md_vals = torch.sum((diff**2) / batch_vars, dim=1).cpu().numpy()
            
            all_z.append(z.cpu().numpy())
            for i in range(len(labels)):
                raw_results.append({
                    'true': labels[i].item(), 'pred': preds[i].item(),
                    'md': md_vals[i], 'resid': res_z[i]
                })

    y_true, y_pred = [], []
    win_size = 11
    for i in range(len(raw_results)):
        window = raw_results[max(0, i-5):min(len(raw_results), i+6)]
        avg_md = sum(r['md'] for r in window) / len(window)
        avg_r_ratio = sum(r['resid'] for r in window) / (len(window) * p_th)
        
        if avg_md > md_th or (avg_md > md_th * 0.75 and avg_r_ratio > 1.2):
            final_cls = -1
        else:
            cls_votes = [r['pred'] for r in window]
            final_cls = max(set(cls_votes), key=cls_votes.count)
            
        y_true.append(raw_results[i]['true'])
        y_pred.append(final_cls)

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"\n[Phys-ProtoNet]Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    
    # plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)

    class_name = ['Unknown', 'Normal', 'PG', 'PDB']
    draw_cm(model_name="Phys-ProtoNet", cm=cm, class_names=class_name)
    
    return y_true, y_pred, np.concatenate(all_z)

def plot_v15_latent_space(z_feats, y_true):

    tsne = TSNE(n_components=2, perplexity=45, init='pca', random_state=42)
    embedded = tsne.fit_transform(z_feats)
    
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=embedded[:,0], y=embedded[:,1], hue=y_true, palette="bright", style=y_true)
    plt.title("Phys-ProtoNet Latent Space Visualization (Variance Penalty Enhanced)")
    plt.savefig("final_tsne.png")

def main():
    DATA_PATH = "./dataset"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load("haizhe_phys_protonet.pth", map_location=device)
    model = PhysProtoNet(input_dim=16, feature_dim=128, num_known_classes=3).to(device)
    model.load_state_dict(checkpoint['model_state'])
    scaler = checkpoint['scaler']
    
    train_loader, test_loader = create_loaders(DATA_PATH, scaler)
    
    md_th, p_th = calibrate_thresholds_v15(model, train_loader, device)
    
    y_t, y_p, z_feats = run_evaluation(model, test_loader, md_th, p_th, device)
    
    class_name = ['Unknown', 'Normal', 'PG', 'PDB']
    plot_latent_space_new(z_feats=z_feats, y_true=y_t, class_names=class_name)

if __name__ == "__main__":
    main()