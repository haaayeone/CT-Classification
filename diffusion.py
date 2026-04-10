import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset


# --- Configuration -----------------------------------------------------------

# Data Paths
ROOT_4X = "/home/ali/Feature_space_diffusion_model/amsr_feature_space_training/FasterMRI-main/Features_log_GAP/4x_with_model"
ROOT_FULLY = "/home/ali/Feature_space_diffusion_model/amsr_feature_space_training/FasterMRI-main/Features_log_GAP/Fully_sampled"

# Training Hyperparameters
BATCH_SIZE = 64
EPOCHS = 100000
INPUT_DIM = 512
TIMESTEPS = 1000
LEARNING_RATE = 2e-4  # A slightly lower LR can improve stability

# Normalization Mode
# "global_-1to1" is generally recommended for consistency across all samples.
NORMALIZE_MODE = "global_-1to1"

# Logging and Weight Directories
LOG_DIR = "./logs"
WEIGHT_DIR = "./weights"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(WEIGHT_DIR, exist_ok=True)


# --- Data Handling -----------------------------------------------------------

def compute_global_min_max(dir_4x, dir_fully, cache_file="global_minmax.json"):
    """Caches or computes the global min/max values across all data splits."""
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            print(f"Loading cached global stats from {cache_file}")
            return json.load(f)

    min_val, max_val = float("inf"), float("-inf")
    pbar = tqdm.tqdm(desc="Computing Global Min/Max")
    file_count = 0
    for split in ["Train", "Val", "Test"]:
        for dir_path in [os.path.join(dir_4x, split), os.path.join(dir_fully, split)]:
            if not os.path.exists(dir_path):
                continue
            files = [f for f in os.listdir(dir_path) if f.endswith(".npy")]
            file_count += len(files)
            pbar.total = file_count
            for file in files:
                arr = np.load(os.path.join(dir_path, file))
                min_val = min(min_val, arr.min())
                max_val = max(max_val, arr.max())
                pbar.update(1)
    pbar.close()

    stats = {"min": float(min_val), "max": float(max_val)}
    with open(cache_file, "w") as f:
        json.dump(stats, f)
    print(f"Computed and cached global stats: {stats}")
    return stats


# Pre-compute global statistics if requested
_GLOBAL_STATS = (
    compute_global_min_max(ROOT_4X, ROOT_FULLY)
    if NORMALIZE_MODE.startswith("global")
    else None
)


class FeaturePairDataset(Dataset):
    """
    Loads pairs of (undersampled, fully-sampled) feature vectors.
    Returns: (target, condition) -> (fully-sampled, undersampled)
    """

    def __init__(self, root_4x, root_fully, split="Train"):
        self.undersampled_dir = os.path.join(root_4x, split)
        self.fullysampled_dir = os.path.join(root_fully, split)

        self.filenames = sorted(
            [
                f
                for f in os.listdir(self.undersampled_dir)
                if f.endswith(".npy")
                and os.path.exists(os.path.join(self.fullysampled_dir, f))
            ]
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        # Condition is the undersampled features, Target is the fully-sampled
        cond_features = np.load(
            os.path.join(self.undersampled_dir, filename)
        ).astype(np.float32)
        target_features = np.load(
            os.path.join(self.fullysampled_dir, filename)
        ).astype(np.float32)

        # --- Normalization ---
        if NORMALIZE_MODE.startswith("sample"):
            sample_min = min(cond_features.min(), target_features.min())
            sample_max = max(cond_features.max(), target_features.max())
            range_val = sample_max - sample_min
            if range_val < 1e-6:
                cond_features = np.zeros_like(cond_features)
                target_features = np.zeros_like(target_features)
            else:
                cond_features = (cond_features - sample_min) / (range_val + 1e-8)
                target_features = (target_features - sample_min) / (
                    range_val + 1e-8
                )
        elif NORMALIZE_MODE.startswith("global"):
            gmin, gmax = _GLOBAL_STATS["min"], _GLOBAL_STATS["max"]
            range_val = gmax - gmin
            cond_features = (cond_features - gmin) / (range_val + 1e-8)
            target_features = (target_features - gmin) / (range_val + 1e-8)
        else:
            raise ValueError(f"Unknown NORMALIZE_MODE '{NORMALIZE_MODE}'")

        # Map to [-1, 1] if requested
        if NORMALIZE_MODE.endswith("-1to1"):
            cond_features = 2.0 * cond_features - 1.0
            target_features = 2.0 * target_features - 1.0

        return torch.from_numpy(target_features), torch.from_numpy(cond_features)


# --- Diffusion Components ----------------------------------------------------


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embeddings."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


def get_cosine_schedule_with_warmup(
    optimizer, warmup_steps, total_steps, lr_min=0.0
):
    """Cosine learning rate scheduler with a linear warmup."""

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min + (1 - lr_min) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def get_diffusion_variables(timesteps=1000, s=0.008, device="cpu"):
    """
    Pre-computes the variables for the diffusion process based on a cosine schedule.
    This is more efficient than calculating them on the fly.
    """
    steps = torch.arange(timesteps + 1, dtype=torch.float64, device=device)
    f = lambda t: torch.cos(((t / timesteps + s) / (1 + s)) * math.pi * 0.5) ** 2
    alphas_cumprod = f(steps) / f(torch.tensor(0.0, device=device))

    # Ensure numerical stability
    alphas_cumprod = torch.clamp(alphas_cumprod, 0.0001, 0.9999)

    # Betas are derived from alphas_cumprod and have length `timesteps`
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clamp(betas, 0.0001, 0.9999)

    alphas = 1.0 - betas

    # These are used in the DDPM sampling formula
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # Calculation for posterior variance
    posterior_variance = (
        betas * (1.0 - alphas_cumprod_prev[1:]) / (1.0 - alphas_cumprod[1:])
    )
    posterior_log_variance_clipped = torch.log(
        torch.clamp(posterior_variance, min=1e-20)
    )

    # Return a dictionary of all required variables, ensuring they all have length `timesteps`
    return {
        "betas": betas.float(),
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod[1:].float(),
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod[1:].float(),
        "sqrt_recip_alphas": sqrt_recip_alphas.float(),
        "posterior_log_variance_clipped": posterior_log_variance_clipped.float(),
    }


# --- Network Architecture ----------------------------------------------------


class ResidualBlock(nn.Module):
    """
    A robust residual block with FiLM conditioning, GELU activation, and LayerNorm.
    """

    def __init__(self, dim, time_dim, cond_dim=None, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Main path
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)

        # Time embedding projection
        self.time_proj = nn.Linear(time_dim, dim * 4)

        # FiLM (Feature-wise Linear Modulation) for conditioning
        if cond_dim is not None:
            self.film = nn.Linear(
                cond_dim, dim * 8
            )  # gamma and beta for the widened dim
        else:
            self.film = None

    def forward(self, x, t_emb, cond=None):
        res = x
        h = self.norm1(x)
        h = self.fc1(h) + self.time_proj(t_emb)
        h = self.act(h)

        if cond is not None and self.film is not None:
            # Generate scale (gamma) and shift (beta) from condition
            gamma, beta = self.film(cond).chunk(2, dim=-1)
            h = (1 + gamma) * h + beta  # Apply FiLM

        h = self.dropout(h)
        h = self.fc2(h)

        return h + res  # Residual connection


class RefinedMultiScaleUNet1D(nn.Module):
    """
    A refined multi-scale UNet architecture for 1D feature diffusion.

    Key Improvements:
    - Skip connections are concatenated and projected, a more powerful fusion method.
    - Uses a consistent and robust `ResidualBlock`.
    - Corrected decoder logic to ensure tensor shapes are compatible.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dims: tuple[int, ...] = (256, 512, 1024),
        blocks_per_level: int = 2,
        time_dim: int = 256,
        cond_dim: int | None = None,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        levels = len(hidden_dims)

        # --- Time Embedding ---
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # --- Input Projection ---
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])

        # --- Encoder ---
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(levels):
            dim_in = hidden_dims[i]
            dim_out = hidden_dims[i + 1] if i < levels - 1 else hidden_dims[-1]

            self.enc_blocks.append(
                nn.ModuleList(
                    [
                        ResidualBlock(dim_in, time_dim, cond_dim)
                        for _ in range(blocks_per_level)
                    ]
                )
            )

            if i < levels - 1:
                self.downs.append(nn.Linear(dim_in, dim_out))

        # --- Bottleneck ---
        self.mid_blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dims[-1], time_dim, cond_dim)
                for _ in range(blocks_per_level)
            ]
        )

        # --- Decoder (Corrected Logic) ---
        self.dec_blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.skip_projs = nn.ModuleList()

        for i in reversed(range(levels)):
            dim_from_up = hidden_dims[i]
            dim_from_skip = hidden_dims[i]
            dim_target = hidden_dims[i - 1] if i > 0 else hidden_dims[0]

            # Skip projection fuses the tensor from the skip connection and the up-sampled path
            self.skip_projs.append(
                nn.Linear(dim_from_up + dim_from_skip, dim_from_up)
            )

            # Residual blocks operate on the fused tensor
            self.dec_blocks.append(
                nn.ModuleList(
                    [
                        ResidualBlock(dim_from_up, time_dim, cond_dim)
                        for _ in range(blocks_per_level)
                    ]
                )
            )

            # Upsampling layer prepares the tensor for the next decoder level
            if i > 0:
                self.ups.append(nn.Linear(dim_from_up, dim_target))

        # --- Output Projection ---
        self.output_norm = nn.LayerNorm(hidden_dims[0])
        self.output_proj = nn.Linear(hidden_dims[0], input_dim)

    def forward(self, x, t, cond=None):
        t_emb = self.time_mlp(t)
        x = self.input_proj(x)
        skips = []

        # === Encoder ===
        for i, blocks in enumerate(self.enc_blocks):
            for blk in blocks:
                x = blk(x, t_emb, cond)
            skips.append(x)
            if i < len(self.downs):
                x = self.downs[i](x)

        # === Bottleneck ===
        for blk in self.mid_blocks:
            x = blk(x, t_emb, cond)

        # === Decoder (Corrected Logic) ===
        skips = skips[::-1]  # Reverse for easy popping
        for i, (blocks, skip_proj) in enumerate(zip(self.dec_blocks, self.skip_projs)):
            skip = skips[i]
            x = torch.cat([x, skip], dim=-1)
            x = skip_proj(x)

            for blk in blocks:
                x = blk(x, t_emb, cond)

            if i < len(self.ups):
                x = self.ups[i](x)

        x = self.output_norm(x)
        return self.output_proj(x)


# --- Training & Evaluation -------------------------------------------------


def q_sample(x_0, t, diff_vars):
    """Forward diffusion process: noise the data."""
    noise = torch.randn_like(x_0)
    # Get pre-computed values for the given timesteps t
    sqrt_alpha_cumprod_t = diff_vars["sqrt_alphas_cumprod"][t].view(-1, 1)
    sqrt_one_minus_alpha_cumprod_t = diff_vars["sqrt_one_minus_alphas_cumprod"][
        t
    ].view(-1, 1)

    xt = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
    return xt, noise


def diffusion_loss(model, x_0, cond, t, diff_vars):
    """Calculate the MSE loss between the predicted noise and the actual noise."""
    x_t, noise = q_sample(x_0, t, diff_vars)
    predicted_noise = model(x_t, t, cond)
    return F.mse_loss(predicted_noise, noise)


@torch.no_grad()
def p_sample_loop(model, cond, diff_vars, device, timesteps=TIMESTEPS):
    """DDPM sampling loop to generate data from noise."""
    model.eval()
    x = torch.randn_like(cond).to(device)  # Start with pure noise

    for i in tqdm.tqdm(
        reversed(range(timesteps)), desc="DDPM Sampling", total=timesteps, leave=False
    ):
        t = torch.full((x.size(0),), i, device=device, dtype=torch.long)

        # Predict noise
        pred_noise = model(x, t, cond)

        # Get pre-computed variables
        sqrt_recip_alphas_t = diff_vars["sqrt_recip_alphas"][t].view(-1, 1)
        betas_t = diff_vars["betas"][t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = diff_vars[
            "sqrt_one_minus_alphas_cumprod"
        ][t].view(-1, 1)

        # Denoise using the DDPM formula
        x = sqrt_recip_alphas_t * (
            x - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if i > 0:
            # Add noise back in (stochastic step)
            posterior_log_variance_t = diff_vars["posterior_log_variance_clipped"][
                t
            ].view(-1, 1)
            noise = torch.randn_like(x)
            # The 0.5 removes the log from the log-variance
            x += torch.exp(0.5 * posterior_log_variance_t) * noise

    return torch.clamp(x, -1.0, 1.0)  # Clamp to valid [-1, 1] range


@torch.no_grad()
def evaluate_reconstruction(model, loader, diff_vars, device, timesteps):
    """Evaluate the model by running the full reconstruction pipeline."""
    total_mse = 0
    model.eval()

    for x_0, cond in tqdm.tqdm(loader, desc="Validating"):
        x_0 = x_0.to(device)
        cond = cond.to(device)

        x_rec = p_sample_loop(model, cond, diff_vars, device, timesteps)

        if torch.isnan(x_rec).any() or torch.isinf(x_rec).any():
            print("Warning: Invalid values (NaN/Inf) in reconstruction. Skipping batch.")
            continue

        total_mse += F.mse_loss(x_rec, x_0, reduction="sum").item()

    return total_mse / len(loader.dataset)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- DataLoaders ---
    train_dataset = FeaturePairDataset(ROOT_4X, ROOT_FULLY, split="Train")
    val_dataset = FeaturePairDataset(ROOT_4X, ROOT_FULLY, split="Val")
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True
    )

    # --- Model & Optimizer ---
    model = RefinedMultiScaleUNet1D(
        input_dim=INPUT_DIM,
        cond_dim=INPUT_DIM,
        hidden_dims=(256, 512, 512),  # Example dimensions
        time_dim=256,
    )

    # Note: For multi-GPU, DistributedDataParallel is preferred over DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Schedulers & Diffusion Vars ---
    total_steps = EPOCHS * len(train_loader)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps=500, total_steps=total_steps
    )
    diffusion_vars = get_diffusion_variables(timesteps=TIMESTEPS, device=device)

    # --- Training Loop ---
    best_val_mse = float("inf")
    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for step, (x_0, cond) in enumerate(pbar):
            x_0, cond = x_0.to(device), cond.to(device)
            t = torch.randint(0, TIMESTEPS, (x_0.size(0),), device=device).long()

            optimizer.zero_grad()
            loss = diffusion_loss(model, x_0, cond, t, diffusion_vars)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()

            pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        # --- Validation & Model Saving (only every 100 epochs) ---
        if (epoch + 1) % 100 == 0:
            val_mse = evaluate_reconstruction(
                model, val_loader, diffusion_vars, device, timesteps=TIMESTEPS
            )
            print(f"\nEpoch {epoch+1} | Validation MSE: {val_mse:.6f}\n")

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_model_path = os.path.join(WEIGHT_DIR, "best_model.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f"🎉 New best model saved with Val MSE: {best_val_mse:.6f}")

        # Save a checkpoint of the last model
        torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, "last_model.pth"))


if __name__ == "__main__":
    train()