import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ============================================================
# 경로 설정
# ============================================================
SCRIPT_DIR   = Path(__file__).parent.resolve()
MASTER_CSV   = SCRIPT_DIR / r"C:\Users\김하연\Desktop\CT classifier\csv\data_splits\master_df_5k.csv"
FEATURE_DIR  = SCRIPT_DIR / r"C:\Users\김하연\Desktop\CT classifier\features_undersampled_x2"
MERGED_NPZ   = SCRIPT_DIR / "features_under_x2_all.npz"
TSNE_IMG     = SCRIPT_DIR / "tsne_embeddings_under_x2.png"
MODEL_SAVE   = SCRIPT_DIR / "ich_classifier_under_x2.pt"

SUBTYPE_COLS = ['any', 'epidural', 'intraparenchymal',
                'intraventricular', 'subarachnoid', 'subdural']
SCAN_LABEL_COLS = [f'scan_{c}' for c in SUBTYPE_COLS]

# ============================================================
# GPU 설정
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=" * 60)
print("STEP 3: Feature 통합 + t-SNE + Classifier 학습")
print(f"  디바이스: {device}")
if device.type == 'cuda':
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("=" * 60 + "\n")

# ============================================================
# 1. 개별 .npy → 하나의 .npz로 통합
# ============================================================
print("[1/5] 개별 .npy 파일 통합 중...")

df = pd.read_csv(MASTER_CSV)

# scan 단위 고유 목록 (label + split 포함)
df_scan = df.drop_duplicates(subset='series_uid')[
    ['series_uid'] + SCAN_LABEL_COLS + ['split']
].reset_index(drop=True)

embeddings  = []
series_uids = []
labels      = []
splits      = []
missing     = []

for _, row in tqdm(df_scan.iterrows(), total=len(df_scan),
                   desc="  통합", ncols=80):
    npy_path = FEATURE_DIR / f"{row['series_uid']}.npy"

    if not npy_path.exists():
        missing.append(row['series_uid'])
        continue

    emb = np.load(npy_path)                        # (512,)
    lbl = row[SCAN_LABEL_COLS].values.astype(np.float32)  # (6,)

    embeddings.append(emb)
    series_uids.append(row['series_uid'])
    labels.append(lbl)
    splits.append(row['split'])

embeddings  = np.stack(embeddings,  axis=0)   # (N, 512)
labels      = np.stack(labels,      axis=0)   # (N, 6)
series_uids = np.array(series_uids)           # (N,)
splits      = np.array(splits)                # (N,)

# 통합 파일 저장
np.savez(
    MERGED_NPZ,
    embeddings  = embeddings,
    series_uids = series_uids,
    labels      = labels,
    splits      = splits
)

print(f"  → 통합 완료: {len(embeddings):,}개 scan")
print(f"  → embeddings shape : {embeddings.shape}")
print(f"  → labels shape     : {labels.shape}")
print(f"  → 누락된 scan      : {len(missing):,}개")
print(f"  → 저장: {MERGED_NPZ}\n")

# split별 확인
for sp in ['train', 'val', 'test']:
    mask = splits == sp
    pos  = labels[mask][:, 0].sum()  # any=1 기준
    print(f"  {sp:<6}: {mask.sum():>5,}개 scan  |  출혈 {int(pos):>4,}개 "
          f"({100*pos/mask.sum():.1f}%)")
print()

# ============================================================
# 2. t-SNE 시각화
# ============================================================
print("[2/5] t-SNE 시각화 중...")
from sklearn.manifold import TSNE

# 전체 데이터로 t-SNE (너무 많으면 샘플링)
MAX_TSNE = 3000
if len(embeddings) > MAX_TSNE:
    idx = np.random.choice(len(embeddings), MAX_TSNE, replace=False)
    emb_tsne = embeddings[idx]
    lbl_tsne = labels[idx, 0]  # any 기준
    print(f"  → {MAX_TSNE:,}개 샘플링 후 t-SNE 수행")
else:
    emb_tsne = embeddings
    lbl_tsne = labels[:, 0]

with tqdm(total=1, desc="  t-SNE 계산", ncols=80) as pbar:
    tsne   = TSNE(n_components=2, random_state=42, perplexity=30)
    coords = tsne.fit_transform(emb_tsne)   # (N, 2)
    pbar.update(1)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 출혈 유무 (any)
colors = ['steelblue' if l == 0 else 'crimson' for l in lbl_tsne]
axes[0].scatter(coords[:, 0], coords[:, 1], c=colors, s=8, alpha=0.6)
axes[0].set_title('t-SNE: 출혈 유무 (파랑=정상 / 빨강=출혈)')
axes[0].set_xlabel('t-SNE 1')
axes[0].set_ylabel('t-SNE 2')

# subtype별 색상 (train 데이터만)
if len(embeddings) > MAX_TSNE:
    lbl_sub = labels[idx]
else:
    lbl_sub = labels

subtype_colors = {
    'normal'           : 'lightgray',
    'epidural'         : 'gold',
    'intraparenchymal' : 'darkorange',
    'intraventricular' : 'green',
    'subarachnoid'     : 'purple',
    'subdural'         : 'crimson',
}
color_map = []
for l in lbl_sub:
    if l[0] == 0:
        color_map.append('lightgray')
    elif l[1] == 1:
        color_map.append('gold')
    elif l[2] == 1:
        color_map.append('darkorange')
    elif l[3] == 1:
        color_map.append('green')
    elif l[4] == 1:
        color_map.append('purple')
    else:
        color_map.append('crimson')

axes[1].scatter(coords[:, 0], coords[:, 1], c=color_map, s=8, alpha=0.6)
axes[1].set_title('t-SNE: Subtype별')
for name, color in subtype_colors.items():
    axes[1].scatter([], [], c=color, label=name, s=40)
axes[1].legend(loc='best', fontsize=8)
axes[1].set_xlabel('t-SNE 1')

plt.tight_layout()
plt.savefig(TSNE_IMG, dpi=150, bbox_inches='tight')
plt.close()
print(f"  → 저장: {TSNE_IMG}\n")

# ============================================================
# 3. Dataset / DataLoader 정의
# ============================================================
print("[3/5] Dataset / DataLoader 구성 중...")

class ICHEmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.X = torch.tensor(embeddings, dtype=torch.float32)
        self.y = torch.tensor(labels,     dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_mask = splits == 'train'
val_mask   = splits == 'val'
test_mask  = splits == 'test'

train_ds = ICHEmbeddingDataset(embeddings[train_mask], labels[train_mask])
val_ds   = ICHEmbeddingDataset(embeddings[val_mask],   labels[val_mask])
test_ds  = ICHEmbeddingDataset(embeddings[test_mask],  labels[test_mask])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)

print(f"  → train: {len(train_ds):,} / val: {len(val_ds):,} / test: {len(test_ds):,}\n")

# ============================================================
# 4. MLP Classifier 정의 및 학습
# ============================================================
print("[4/5] Classifier 학습 중...")

class ICHClassifier(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)   # logits (N, 6)

model_clf  = ICHClassifier().to(device)
optimizer  = torch.optim.Adam(model_clf.parameters(), lr=1e-4, weight_decay=1e-3)
# val loss 기준으로 10 epoch 개선 없으면 lr 절반으로 자동 감소
scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=True
)

# ── Class weight 계산 (train set 기준) ──
# pos_weight[i] = (음성 수) / (양성 수) per subtype
# → minority class(epidural 등)에 더 큰 패널티 부여
train_labels = torch.tensor(labels[splits == 'train'], dtype=torch.float32)
n_train      = len(train_labels)
pos_counts   = train_labels.sum(dim=0).clamp(min=1)       # (6,) 양성 수
neg_counts   = (n_train - pos_counts).clamp(min=1)        # (6,) 음성 수
# sqrt로 완화: epidural 13.22 → ~3.6 (loss 안정화)
pos_weight   = torch.sqrt(neg_counts / pos_counts).to(device)  # (6,) weight

print(f"  Class weights (pos_weight):")
for i, (col, w) in enumerate(zip(SUBTYPE_COLS, pos_weight.cpu().tolist())):
    print(f"    {col:<22}: {w:.2f}")
print()

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

NUM_EPOCHS    = 100
SMOOTHING     = 0.1    # label smoothing 강도
ES_PATIENCE   = 20     # early stopping: 20 epoch 개선 없으면 중단

best_val_auc  = 0.0
es_counter    = 0      # early stopping 카운터
history = {'train_loss': [], 'val_loss': [], 'val_auc': []}

for epoch in tqdm(range(NUM_EPOCHS), desc="  학습", ncols=80):

    # ── Train ──
    model_clf.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits   = model_clf(X_batch)
        # label smoothing: 1→0.9, 0→0.1
        y_smooth = y_batch * (1 - SMOOTHING) + 0.5 * SMOOTHING
        loss     = criterion(logits, y_smooth)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # ── Validation ──
    model_clf.eval()
    val_loss  = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model_clf(X_batch)
            loss   = criterion(logits, y_batch)
            val_loss  += loss.item()
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
    val_loss  /= len(val_loader)
    all_preds  = np.concatenate(all_preds,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # AUC per class
    aucs = []
    for i in range(all_labels.shape[1]):
        if all_labels[:, i].sum() > 0:
            aucs.append(roc_auc_score(all_labels[:, i], all_preds[:, i]))
    val_auc = np.mean(aucs)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_auc'].append(val_auc)

    # ReduceLROnPlateau: val_loss 기준으로 lr 자동 조절
    scheduler.step(val_loss)

    # Best model 저장 + Early Stopping
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        es_counter   = 0
        torch.save(model_clf.state_dict(), MODEL_SAVE)
    else:
        es_counter += 1
        if es_counter >= ES_PATIENCE:
            tqdm.write(f"  ⏹ Early Stopping: {ES_PATIENCE} epoch 동안 개선 없음 "
                       f"(epoch {epoch+1})")
            break

    # 10 epoch마다 출력 (현재 lr 포함)
    if (epoch + 1) % 10 == 0:
        cur_lr = optimizer.param_groups[0]["lr"]
        tqdm.write(f"  Epoch {epoch+1:>3} | "
                   f"train_loss={train_loss:.4f} | "
                   f"val_loss={val_loss:.4f} | "
                   f"val_AUC={val_auc:.4f} | "
                   f"lr={cur_lr:.2e} | "
                   f"ES={es_counter}/{ES_PATIENCE}")

print(f"\n  → Best val AUC: {best_val_auc:.4f}")
print(f"  → 모델 저장: {MODEL_SAVE}\n")

# ============================================================
# 5. Test 평가
# ============================================================
print("[5/5] Test set 최종 평가...")

model_clf.load_state_dict(torch.load(MODEL_SAVE, map_location=device))
model_clf.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in tqdm(test_loader, desc="  평가", ncols=80):
        X_batch = X_batch.to(device)
        logits  = model_clf(X_batch)
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(y_batch.numpy())

all_preds  = np.concatenate(all_preds,  axis=0)
all_labels = np.concatenate(all_labels, axis=0)

print(f"\n  {'subtype':<22} {'AUC':>6}")
print("  " + "-" * 30)
aucs = []
for i, col in enumerate(SUBTYPE_COLS):
    if all_labels[:, i].sum() > 0:
        auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
        aucs.append(auc)
        print(f"  {col:<22} {auc:.4f}")
    else:
        print(f"  {col:<22}  N/A (양성 없음)")

mean_auc = np.mean(aucs)
print(f"\n  Mean AUC: {mean_auc:.4f}")

# 학습 곡선 저장
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history['train_loss'], label='train')
axes[0].plot(history['val_loss'],   label='val')
axes[0].set_title('Loss Curve')
axes[0].set_xlabel('Epoch')
axes[0].legend()

axes[1].plot(history['val_auc'])
axes[1].set_title('Val AUC Curve')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('AUC')

plt.tight_layout()
curve_path = SCRIPT_DIR / "training_curve.png"
plt.savefig(curve_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  → 학습 곡선 저장: {curve_path}")
print("\n✅ STEP 3 완료")
print(f"   저장 파일:")
print(f"     features_all.npz    → 통합 embedding")
print(f"     tsne_embeddings.png → t-SNE 시각화")
print(f"     ich_classifier.pt   → 학습된 모델")
print(f"     training_curve.png  → 학습 곡선")