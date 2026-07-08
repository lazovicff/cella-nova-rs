#!/usr/bin/env python3
"""
Flow Lenia Training — PyTorch GPU (batched)

All per-kernel and per-channel operations are batched for GPU efficiency.
Autograd handles the backward pass automatically.

Usage:
    python train/train_lenia.py
    python train/train_lenia.py --eval trained_kernels.bin
"""

import argparse
import math
import random
import struct
import time

import numpy as np
import torch
import torch.nn.functional as F

# =========================================================================
# Constants
# =========================================================================

GRID_SIZE = 512
NUM_CHANNELS = 3
NUM_KERNELS = 9
NUM_STEPS = 40
DT = 0.2
DD = 5
SIGMA = 0.65

C0 = [0, 0, 0, 1, 1, 1, 2, 2, 2]
C1 = [[0, 1, 6], [2, 3, 4], [5, 7, 8]]

GLOBAL_R = 42.0
RADII = [0.8, 0.6, 1.0, 0.7, 0.5, 0.9, 0.65, 0.55, 0.85]
A_FLAT = [
    0.0,
    0.6,
    0.0,
    0.0,
    0.5,
    0.0,
    0.0,
    0.7,
    0.0,
    0.0,
    0.55,
    0.0,
    0.0,
    0.45,
    0.0,
    0.0,
    0.65,
    0.0,
    0.0,
    0.5,
    0.0,
    0.0,
    0.6,
    0.0,
    0.0,
    0.55,
    0.0,
]
W_FLAT = [
    0.08,
    0.06,
    0.01,
    0.07,
    0.05,
    0.01,
    0.09,
    0.07,
    0.01,
    0.08,
    0.06,
    0.01,
    0.07,
    0.05,
    0.01,
    0.09,
    0.07,
    0.01,
    0.08,
    0.06,
    0.01,
    0.07,
    0.05,
    0.01,
    0.08,
    0.06,
    0.01,
]
B_FLAT = [
    0.8,
    -0.3,
    0.0,
    0.7,
    -0.25,
    0.0,
    0.9,
    -0.35,
    0.0,
    0.75,
    -0.3,
    0.0,
    0.65,
    -0.2,
    0.0,
    0.85,
    -0.35,
    0.0,
    0.7,
    -0.25,
    0.0,
    0.6,
    -0.2,
    0.0,
    0.8,
    -0.3,
    0.0,
]

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


# =========================================================================
# Sobel kernels
# =========================================================================

SOBEL_X = torch.tensor(
    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device
)
SOBEL_Y = torch.tensor(
    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device
)


# =========================================================================
# PyTorch Flow Lenia (fully batched)
# =========================================================================


class FlowLeniaTorch(torch.nn.Module):
    """PyTorch Flow Lenia with fully batched GPU operations."""

    def __init__(
        self,
        grid_size,
        num_channels,
        num_kernels,
        c0,
        c1,
        dt=0.2,
        dd=5,
        sigma=0.65,
        basal_rate=0.0,
        kinetic_cost=0.0,
    ):
        super().__init__()
        self.size = grid_size
        self.nc = num_channels
        self.nk = num_kernels
        self.dt = dt
        self.dd = dd
        self.sigma = sigma
        self.basal_rate = basal_rate
        self.kinetic_cost = kinetic_cost

        # Channel mapping buffers
        self.register_buffer("c0_idx", torch.tensor(c0, dtype=torch.long))
        # Channel aggregate weight matrix: [nc, nk] where M[c,k]=1 if k contributes to c
        agg = torch.zeros(num_channels, num_kernels)
        for c, ks in enumerate(c1):
            agg[c, ks] = 1.0
        self.register_buffer("channel_agg_weight", agg)

        # Growth params (fixed: μ=0, σ=5, h=1)
        self.register_buffer("kernel_m", torch.zeros(num_kernels, 1, 1))
        self.register_buffer("kernel_s", torch.full((num_kernels, 1, 1), 5.0))
        self.register_buffer("kernel_h", torch.ones(num_kernels, 1, 1))

        # Trainable kernel FFT weights: [nk, H, W, 2] (real, imag)
        self.kernels_fft = torch.nn.Parameter(
            torch.zeros(
                num_kernels, grid_size, grid_size, 2, dtype=torch.float32, device=device
            )
        )

        # Sobel kernels as conv2d weights: [1, 1, 3, 3]
        self.register_buffer("sobel_x", SOBEL_X.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", SOBEL_Y.view(1, 1, 3, 3))

    def generate_kernels(self, global_r, radii, a, w, b):
        """Generate initial kernel FFT weights."""
        kernels_np = np.zeros((self.nk, self.size, self.size), dtype=np.complex64)
        for k in range(self.nk):
            kernel_real = np.zeros((self.size, self.size), dtype=np.float64)
            mid = self.size // 2
            for i in range(self.size):
                for j in range(self.size):
                    dx = i - mid
                    dy = j - mid
                    dist = math.sqrt(dx * dx + dy * dy)
                    d_scaled = dist / ((global_r + 15.0) * radii[k])
                    sig = 0.5 * (math.tanh((-d_scaled + 1.0) * 5.0) + 1.0)
                    ker_val = 0.0
                    for p in range(3):
                        diff = d_scaled - a[k * 3 + p]
                        ker_val += b[k * 3 + p] * math.exp(
                            -(diff * diff) / w[k * 3 + p]
                        )
                    kernel_real[i, j] = sig * ker_val
            total = np.sum(kernel_real)
            if total > 0.0:
                kernel_real /= total
            kernel_real = np.fft.ifftshift(kernel_real)
            kernels_np[k] = np.fft.fft2(kernel_real).astype(np.complex64)

        kt = torch.from_numpy(np.stack([kernels_np.real, kernels_np.imag], axis=-1))
        self.kernels_fft.data.copy_(kt.to(device))

    # ------------------------------------------------------------------
    # Forward pass (all batched)
    # ------------------------------------------------------------------

    def _growth(self, x):
        """Batched growth: x [nk, H, W], params broadcast [nk, 1, 1]."""
        diff = x - self.kernel_m
        g = torch.exp(-(diff * diff) / (2.0 * self.kernel_s * self.kernel_s))
        return (2.0 * g - 1.0) * self.kernel_h

    def _sobel(self, field):
        """Batched Sobel gradient. field: [C, H, W]."""
        C, H, W = field.shape
        pad = 1
        field_pad = F.pad(field, (pad, pad, pad, pad), mode="circular").unsqueeze(
            0
        )  # [1, C, H+2, W+2]
        kx = self.sobel_x.expand(C, 1, 3, 3)  # [C, 1, 3, 3]
        ky = self.sobel_y.expand(C, 1, 3, 3)
        gx = F.conv2d(field_pad, kx, groups=C).squeeze(0)  # [C, H, W]
        gy = F.conv2d(field_pad, ky, groups=C).squeeze(0)
        return gx, gy

    def _reintegrate(self, channels, flow_x, flow_y):
        """Gaussian-weighted neighborhood sum advection (matches Rust GPU).

        Uses F.unfold to batch all neighbor contributions in a single pass.
        The tent function has support of ~1.15px, so effective range is [-2, 2]
        (25 neighbors) rather than the full [-dd, dd] (121 neighbors).

        channels: [nc, H, W]
        flow_x, flow_y: [nc, H, W]
        """
        nc, H, W = channels.shape
        dd = self.dd
        sigma = self.sigma
        dt = self.dt
        ma = dd - sigma
        max_sz = min(1.0, 2.0 * sigma)
        area_norm = 4.0 * sigma * sigma

        fx = flow_x.clamp(-ma, ma)
        fy = flow_y.clamp(-ma, ma)

        # Effective neighborhood: tent support is ~1.15px, so [-2, 2] suffices
        eff_dd = 2
        K = (2 * eff_dd + 1) ** 2  # 25

        # Extract all patches at once via unfold
        # [nc, 1, H, W] -> unfold -> [nc, K, H*W]
        ch_patches = F.unfold(
            channels.unsqueeze(1), kernel_size=2 * eff_dd + 1, padding=eff_dd
        )
        fx_patches = F.unfold(
            fx.unsqueeze(1), kernel_size=2 * eff_dd + 1, padding=eff_dd
        )
        fy_patches = F.unfold(
            fy.unsqueeze(1), kernel_size=2 * eff_dd + 1, padding=eff_dd
        )

        # Offset grid for all K positions
        offsets = torch.arange(-eff_dd, eff_dd + 1, device=channels.device)
        grid_y, grid_x = torch.meshgrid(offsets, offsets, indexing="ij")
        dx_all = grid_x.reshape(1, K, 1).float()  # [1, K, 1]
        dy_all = grid_y.reshape(1, K, 1).float()

        # Compute tent weights for all K positions at once
        dpx = torch.abs(dx_all + fx_patches * dt)  # [nc, K, H*W]
        dpy = torch.abs(dy_all + fy_patches * dt)
        sz_x = torch.clamp(0.5 - dpx + sigma, 0.0, max_sz)
        sz_y = torch.clamp(0.5 - dpy + sigma, 0.0, max_sz)
        area = (sz_x * sz_y) / area_norm  # [nc, K, H*W]

        # Mask: only contribute if channel value > 0 (matches Rust GPU)
        mask = (ch_patches > 0.0).float()
        weighted = ch_patches * area * mask  # [nc, K, H*W]

        # Sum over all K neighbor positions
        new_ch = weighted.sum(dim=1).reshape(nc, H, W)  # [nc, H, W]

        # Metabolic costs (using flow at the current pixel, not neighbor)
        flow_mag = torch.sqrt(fx**2 + fy**2 + 1e-8)
        new_ch = (
            new_ch * (1.0 - self.basal_rate * dt) - self.kinetic_cost * flow_mag * dt
        )
        new_ch = new_ch.clamp(min=0.0)

        return new_ch

    def _step(self, ch):
        """Single timestep. Extracted for gradient checkpointing."""
        H, W = self.size, self.size

        # --- Batched per-kernel convolution ---
        src = ch[self.c0_idx]  # [nk, H, W]
        kfft = torch.view_as_complex(self.kernels_fft)  # [nk, H, W]
        conv_fft = torch.fft.fft2(src) * kfft  # [nk, H, W]
        conv = torch.fft.ifft2(conv_fft).real / (H * W)  # [nk, H, W]

        # Batched growth
        u = self._growth(conv)  # [nk, H, W]

        # --- Batched channel aggregate ---
        u_channel = (self.channel_agg_weight @ u.reshape(self.nk, -1)).reshape(
            self.nc, H, W
        )

        # --- Sum channels ---
        sum_a = ch.sum(dim=0)  # [H, W]

        # --- Batched Sobel ---
        nabla_ux, nabla_uy = self._sobel(u_channel)  # [nc, H, W]
        nabla_ax, nabla_ay = self._sobel(sum_a.unsqueeze(0))  # [1, H, W]
        nabla_ax, nabla_ay = nabla_ax[0], nabla_ay[0]  # [H, W]

        # --- Flow field ---
        alpha = torch.clamp((ch / self.nc) ** 2, 0.0, 1.0)  # [nc, H, W]
        flow_x = nabla_ux * (1.0 - alpha) - nabla_ax * alpha
        flow_y = nabla_uy * (1.0 - alpha) - nabla_ay * alpha

        # --- Batched reintegration ---
        ch = self._reintegrate(ch, flow_x, flow_y)

        return ch

    def forward(self, channels, num_steps):
        """Run forward pass for num_steps with gradient checkpointing.

        Checkpointing discards intermediate activations between timesteps
        to avoid OOM on long BPTT sequences. They are recomputed during
        the backward pass.

        Args:
            channels: [nc, H, W] initial state.
            num_steps: number of timesteps.

        Returns:
            [nc, H, W] final state.
        """
        ch = channels

        for _ in range(num_steps):
            ch = torch.utils.checkpoint.checkpoint(self._step, ch, use_reentrant=False)

        return ch


# =========================================================================
# Helpers
# =========================================================================


def center_of_mass(state, width):
    total = width * width
    ch0 = state[:total]
    cx = 0.0
    cy = 0.0
    s = 0.0
    for i in range(total):
        v = ch0[i]
        if v > 0.001:
            x = i % width
            y = i // width
            cx += x * v
            cy += y * v
            s += v
    if s > 0.0:
        return (cx / s, cy / s)
    return (0.0, 0.0)


def make_target_shape(radius=18.0):
    t = np.zeros((32, 32), dtype=np.float64)
    for i in range(32):
        for j in range(32):
            dx = i - 16.0
            dy = j - 16.0
            dist = math.sqrt(dx * dx + dy * dy) / radius
            if dist < 1.0:
                t[i, j] = 0.1
            if dist < 0.5:
                t[i, j] = 0.9
    return t


def place_target(target_shape, grid_size, cx, cy):
    t = np.zeros(grid_size * grid_size, dtype=np.float64)
    for i in range(32):
        for j in range(32):
            px = cx + i - 16
            py = cy + j - 16
            if 0 <= px < grid_size and 0 <= py < grid_size:
                t[py * grid_size + px] = target_shape[j, i]
    return t


def make_init_patch(grid_size, patch_size=32):
    ch0 = np.zeros(grid_size * grid_size, dtype=np.float64)
    cx = grid_size // 2
    cy = grid_size // 2
    x0 = cx - patch_size // 2
    y0 = cy - patch_size // 2
    for dy in range(patch_size):
        for dx in range(patch_size):
            px = x0 + dx
            py = y0 + dy
            if 0 <= px < grid_size and 0 <= py < grid_size:
                ch0[py * grid_size + px] = random.random()
    return ch0


def save_kernels(filename, model, num_kernels, grid_size):
    all_kernels = []
    for k in range(num_kernels):
        kfft = torch.view_as_complex(model.kernels_fft[k])
        kflat = kfft.detach().cpu().numpy().ravel()
        for v in kflat:
            all_kernels.append(v.real)
            all_kernels.append(v.imag)

    total = grid_size * grid_size
    header = struct.pack("III", num_kernels, grid_size, total)
    data = struct.pack(f"{len(all_kernels)}f", *all_kernels)
    with open(filename, "wb") as f:
        f.write(header)
        f.write(data)
    print(f"Saved {num_kernels} kernels ({total} elements each) to {filename}")


def load_kernels(filename, num_kernels, grid_size):
    with open(filename, "rb") as f:
        header = f.read(12)
        nk, gs, total = struct.unpack("III", header)
        assert nk == num_kernels, f"File has {nk} kernels, expected {num_kernels}"
        assert gs == grid_size, f"File has grid {gs}, expected {grid_size}"
        data = struct.unpack(f"{nk * total * 2}f", f.read())

    kernels = []
    for k in range(num_kernels):
        start = k * total * 2
        end = start + total * 2
        kdata = np.array(data[start:end], dtype=np.float32)
        kernels.append(kdata[0::2] + 1j * kdata[1::2])
    print(f"Loaded {num_kernels} kernels from {filename}")
    return kernels


# =========================================================================
# Training
# =========================================================================


def run_training(args):
    print("=" * 60)
    print(f"Flow Lenia Training — PyTorch GPU (batched) [{device}]")
    print("=" * 60)

    grid_size = args.grid_size
    num_channels = args.num_channels
    num_kernels = args.num_kernels
    num_steps = args.num_steps
    lr = args.lr
    num_trials = args.trials
    steps_per_stage = args.steps_per_stage

    model = FlowLeniaTorch(
        grid_size,
        num_channels,
        num_kernels,
        C0,
        C1,
        DT,
        DD,
        SIGMA,
        0.0,
        0.0,
    ).to(device)

    model.generate_kernels(GLOBAL_R, RADII, A_FLAT, W_FLAT, B_FLAT)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    init_patch = make_init_patch(grid_size, args.patch_size)
    center = grid_size // 2
    target_shape = make_target_shape(18.0)

    stage_distances = [0.0, 8.0, 16.0, 32.0, 48.0, 64.0]

    angle = random.random() * 2.0 * math.pi
    dir_x = math.cos(angle)
    dir_y = math.sin(angle)
    print(f"Direction: ({dir_x:.2f}, {dir_y:.2f})")
    print(f"Stages: {len(stage_distances)} distances, {steps_per_stage} steps each")
    print(f"Trials: {num_trials}, LR: {lr}, Steps: {num_steps}")
    print()

    best_disp = 0.0

    for trial in range(num_trials):
        print(f"\n=== Trial {trial} ===")

        with torch.no_grad():
            noise = torch.randn_like(model.kernels_fft) * 0.01
            model.kernels_fft.add_(noise)

        for stage_idx, distance in enumerate(stage_distances):
            tcx = int(round(center + dir_x * distance))
            tcy = int(round(center + dir_y * distance))
            tcx = max(16, min(grid_size - 16, tcx))
            tcy = max(16, min(grid_size - 16, tcy))
            target_flat = place_target(target_shape, grid_size, tcx, tcy)
            target_t = torch.from_numpy(
                target_flat.reshape(grid_size, grid_size).astype(np.float32)
            ).to(device)

            for step in range(steps_per_stage):
                t0 = time.time()

                # Reset state
                init_t = torch.from_numpy(
                    init_patch.reshape(grid_size, grid_size).astype(np.float32)
                ).to(device)
                channels = torch.zeros(
                    num_channels, grid_size, grid_size, device=device
                )
                channels[0] = init_t

                # Forward + loss + backward
                optimizer.zero_grad()
                final = model(channels, num_steps)
                loss = F.mse_loss(final[0], target_t)
                loss.backward()
                optimizer.step()

                elapsed = time.time() - t0

                if step % 5 == 0:
                    grad_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            grad_norm += p.grad.norm().item() ** 2
                    grad_norm = math.sqrt(grad_norm)

                    state = final.detach().cpu().numpy().ravel()
                    com_x, com_y = center_of_mass(state, grid_size)
                    total_mass = float(final[0].sum().item())
                    disp = math.sqrt((com_x - center) ** 2 + (com_y - center) ** 2)

                    print(
                        f"  T{trial} s{step:3d}: loss={loss.item():.6f} "
                        f"|g|={grad_norm:.1f} "
                        f"com=({com_x:.0f},{com_y:.0f}) "
                        f"disp={disp:.1f} "
                        f"sum={total_mass:.1f} "
                        f"{elapsed * 1000:.0f}ms"
                    )

        # Evaluate trial
        with torch.no_grad():
            init_t = torch.from_numpy(
                init_patch.reshape(grid_size, grid_size).astype(np.float32)
            ).to(device)
            channels = torch.zeros(num_channels, grid_size, grid_size, device=device)
            channels[0] = init_t
            final = model(channels, num_steps)
            state = final.detach().cpu().numpy().ravel()
            com_x, com_y = center_of_mass(state, grid_size)
            disp = math.sqrt((com_x - center) ** 2 + (com_y - center) ** 2)
            total_mass = float(final[0].sum().item())
            print(
                f"  => Trial {trial} final: "
                f"com=({com_x:.0f},{com_y:.0f}) "
                f"disp={disp:.1f} sum={total_mass:.1f}"
            )

            if disp > best_disp:
                best_disp = disp
                print("  => New best!")
                save_kernels(args.output, model, num_kernels, grid_size)

    print(f"\n=== Best displacement: {best_disp:.1f} ===")


# =========================================================================
# Evaluation
# =========================================================================


def run_eval(args):
    grid_size = args.grid_size
    num_channels = args.num_channels
    num_kernels = args.num_kernels
    num_steps = args.num_steps
    center = grid_size // 2

    kernels = load_kernels(args.eval, num_kernels, grid_size)

    model = FlowLeniaTorch(
        grid_size,
        num_channels,
        num_kernels,
        C0,
        C1,
        DT,
        DD,
        SIGMA,
        0.0,
        0.0,
    ).to(device)

    with torch.no_grad():
        for k in range(num_kernels):
            kt = torch.from_numpy(
                np.stack([kernels[k].real, kernels[k].imag], axis=-1)
            ).to(device)
            model.kernels_fft[k] = kt

    init_patch = make_init_patch(grid_size, args.patch_size)

    with torch.no_grad():
        init_t = torch.from_numpy(
            init_patch.reshape(grid_size, grid_size).astype(np.float32)
        ).to(device)
        channels = torch.zeros(num_channels, grid_size, grid_size, device=device)
        channels[0] = init_t

        for step in range(num_steps):
            channels = model(channels, 1)
            if step % 10 == 0 or step == num_steps - 1:
                state = channels.detach().cpu().numpy().ravel()
                com_x, com_y = center_of_mass(state, grid_size)
                total_mass = float(channels[0].sum().item())
                disp = math.sqrt((com_x - center) ** 2 + (com_y - center) ** 2)
                print(
                    f"  Step {step:3d}: com=({com_x:.0f},{com_y:.0f}) "
                    f"disp={disp:.1f} sum={total_mass:.1f}"
                )

        state = channels.detach().cpu().numpy().ravel()
        com_x, com_y = center_of_mass(state, grid_size)
        disp = math.sqrt((com_x - center) ** 2 + (com_y - center) ** 2)
        print(f"\nFinal: com=({com_x:.0f},{com_y:.0f}) disp={disp:.1f}")


# =========================================================================
# Main
# =========================================================================


def main():
    parser = argparse.ArgumentParser(description="Flow Lenia Training (PyTorch GPU)")
    parser.add_argument(
        "--eval",
        type=str,
        default=None,
        help="Load trained kernels from file and evaluate",
    )
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE)
    parser.add_argument("--num-channels", type=int, default=NUM_CHANNELS)
    parser.add_argument("--num-kernels", type=int, default=NUM_KERNELS)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--steps-per-stage", type=int, default=100)
    parser.add_argument("--output", type=str, default="train/trained_kernels.bin")
    args = parser.parse_args()

    if args.eval:
        run_eval(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
