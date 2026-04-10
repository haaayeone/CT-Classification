import pandas as pd
import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

# ============================================================
# 경로 설정
# ============================================================
SCRIPT_DIR  = Path(__file__).parent.resolve()
MASTER_CSV  = SCRIPT_DIR / r"C:\Users\김하연\Desktop\CT classifier\csv\data_splits\master_df_5k.csv"
FEATURE_DIR = SCRIPT_DIR / "features"
FEATURE_DIR.mkdir(exist_ok=True)

# ============================================================
# GPU 설정
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=" * 60)
print("STEP 2: CT-FM Feature Extraction")
print(f"  디바이스: {device}")
if device.type == 'cuda':
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("=" * 60 + "\n")

# ============================================================
# 패키지 설치 확인
# ============================================================
try:
    from lighter_zoo import SegResEncoder
    from monai.transforms import (
        EnsureType, Orientation,
        ScaleIntensityRange, CropForeground, Compose
    )
except ImportError:
    print("⚠️  필요한 패키지가 없습니다. 아래 명령어로 설치하세요:")
    print("    pip install lighter_zoo monai pydicom")
    exit(1)

# ============================================================
# CT-FM 공식 전처리 함수 (DICOM → tensor)
# ============================================================
def dicom_to_hu(dcm):
    """DICOM pixel array → HU 변환"""
    pixel     = dcm.pixel_array.astype(np.float32)
    slope     = float(getattr(dcm, 'RescaleSlope',     1))
    intercept = float(getattr(dcm, 'RescaleIntercept', 0))
    return pixel * slope + intercept

def load_scan_volume(filepaths_sorted):
    """
    z축 정렬된 filepath 리스트 → 전처리된 tensor
    CT-FM 공식 전처리: HU [-1024, 2048] → [0, 1]
    반환: (1, 1, D, H, W) tensor
    """
    slices = []
    h_ref, w_ref = None, None

    for fp in filepaths_sorted:
        try:
            dcm = pydicom.dcmread(fp)
            hu  = dicom_to_hu(dcm)

            # 첫 slice 기준으로 H, W 크기 저장
            if h_ref is None:
                h_ref, w_ref = hu.shape

            # 크기가 다른 slice는 resize
            if hu.shape != (h_ref, w_ref):
                hu_tensor = torch.tensor(hu).unsqueeze(0).unsqueeze(0)
                hu = F.interpolate(
                    hu_tensor, size=(h_ref, w_ref),
                    mode='bilinear', align_corners=False
                ).squeeze().numpy()

            slices.append(hu)

        except Exception:
            # 손상된 slice → 이전 slice 복제 또는 0으로 채움
            if slices:
                slices.append(slices[-1].copy())
            elif h_ref:
                slices.append(np.zeros((h_ref, w_ref), dtype=np.float32))
            continue

    if not slices:
        return None

    # (D, H, W) 스택
    volume = np.stack(slices, axis=0).astype(np.float32)

    # CT-FM 공식 전처리: HU [-1024, 2048] → [0, 1] clip & scale
    a_min, a_max = -1024.0, 2048.0
    volume = np.clip(volume, a_min, a_max)
    volume = (volume - a_min) / (a_max - a_min)   # [0, 1]

    # tensor 변환: (1, 1, D, H, W)
    vol_tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # D: 원본 slice 수 유지 (14~55장), H/W만 160으로 통일
    # → z축 정보 왜곡 없음, CT-FM Fully Convolutional이라 D 유연
    original_d = volume.shape[0]
    vol_resized = F.interpolate(
        vol_tensor,
        size=(original_d, 256, 256),
        mode='trilinear',
        align_corners=False
    )
    return vol_resized  # (1, 1, original_d, 256, 256)

# ============================================================
# CT-FM 모델 로드 (HuggingFace에서 자동 다운로드)
# ============================================================
print("[1/4] CT-FM 모델 로드 중 (최초 실행 시 자동 다운로드)...")
with tqdm(total=1, desc="  로드", ncols=80) as pbar:
    model = SegResEncoder.from_pretrained("project-lighter/ct_fm_feature_extractor")
    model = model.to(device)
    model.eval()
    pbar.update(1)
print("  → CT-FM 로드 완료 ✅\n")

# ============================================================
# master_df 로드 및 scan groupby
# ============================================================
print("[2/4] master_df 로드 중...")
with tqdm(total=1, desc="  로드", ncols=80) as pbar:
    df = pd.read_csv(MASTER_CSV)
    pbar.update(1)

total_scans = df['series_uid'].nunique()
print(f"  → 총 scan 수: {total_scans:,}")

# 이미 추출 완료된 scan 확인 (중단 후 재시작 대비)
done_scans  = {f.stem for f in FEATURE_DIR.glob("*.npy")}
remain_scans = total_scans - len(done_scans)
print(f"  → 완료된 scan: {len(done_scans):,}개 (스킵)")
print(f"  → 남은 scan:   {remain_scans:,}개\n")

# ============================================================
# Feature Extraction
# ============================================================
print("[3/4] Feature Extraction 시작...")
failed_scans = []
scan_groups  = df.sort_values('z').groupby('series_uid')

with torch.no_grad():
    for series_uid, group in tqdm(
        scan_groups,
        total=total_scans,
        desc="  추출",
        ncols=80
    ):
        # 이미 추출된 scan 스킵
        if series_uid in done_scans:
            continue

        # z축 정렬된 filepath 리스트
        filepaths = group.sort_values('z')['filepath'].tolist()

        # 볼륨 로드 및 전처리
        volume = load_scan_volume(filepaths)
        if volume is None:
            failed_scans.append(series_uid)
            continue

        try:
            volume = volume.to(device)

            # CT-FM forward
            # model(x) → list of feature maps, [-1]이 가장 깊은 레이어
            output = model(volume)[-1]   # (1, 512, D', H', W')

            # Global average pooling → (512,)
            embedding = F.adaptive_avg_pool3d(output, 1).squeeze().cpu().numpy()

            # series_uid로 저장
            np.save(FEATURE_DIR / f"{series_uid}.npy", embedding)

        except Exception as e:
            failed_scans.append(series_uid)
            tqdm.write(f"  ⚠️  실패: {series_uid} | {e}")
            continue

# ============================================================
# 완료 보고
# ============================================================
print("\n[4/4] 결과 확인...")
extracted = list(FEATURE_DIR.glob("*.npy"))
print(f"  → 추출 완료: {len(extracted):,} / {total_scans:,} scans")
print(f"  → 실패한 scan: {len(failed_scans):,}개")

if failed_scans:
    fail_log = SCRIPT_DIR / "failed_scans.txt"
    with open(fail_log, 'w') as f:
        f.write('\n'.join(failed_scans))
    print(f"  → 실패 목록 저장: {fail_log}")

# 샘플 embedding shape 확인
sample = list(FEATURE_DIR.glob("*.npy"))
if sample:
    emb = np.load(sample[0])
    print(f"\n  샘플 embedding shape: {emb.shape}")  # (512,) 이어야 함
    print(f"  mean={emb.mean():.4f}, std={emb.std():.4f}")

print(f"\n  저장 위치: {FEATURE_DIR}")
print(f"  파일 형식: {{series_uid}}.npy  →  shape (512,)")
print("\n✅ STEP 2 완료 → 다음: STEP 3 (Classifier 학습)")