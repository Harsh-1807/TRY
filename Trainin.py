# -*- coding: utf-8 -*-
  """
  Stage 2: Train CorrDiff UNet  (Flow-Matching Corrector  OR  CorrDiff Residual)
  ===============================================================================
  DUAL-MODE TRAINER  ──  plug-and-play, 4 × H100, anti-overfit
  
  CHANGES vs previous version
  ────────────────────────────
    * DDP OPTIMIZED : Added DistributedSampler for 4x faster epochs.
    * METRIC SYNC   : Global metric all_reduce to properly calculate sharded validation.
    * QDM ISOLATION : QDM strictly computes on Rank 0 using the full val-split, with barrier sync.
    * LOSS UPGRADE  : Nuclear PCC Push & Bridge-Killer Sparsity.
  """
  
  import os
  import math
  import time
  import argparse
  import traceback
  import warnings
  from copy import deepcopy
  warnings.filterwarnings("ignore", category=UserWarning)
  
  import numpy as np
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  import torch.distributed as dist
  from torch.utils.data import DataLoader, Subset
  from torch.utils.data.distributed import DistributedSampler
  from torch.amp import GradScaler, autocast
  from torch.optim import AdamW
  from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
  
  try:
      import albumentations as A
      HAS_ALB = True
  except ImportError:
      HAS_ALB = False
      warnings.warn("albumentations not found. pip install albumentations opencv-python-headless")
  
  from Dataset import UpscaleDataset
  from Network import CorrDiffRegressor, UNet, FlowMatching, PhysicsGuide, QDM
  
  # ══════════════════════════════════════════════════════════════════════════════
  # 0.  MODE TOGGLE
  # ══════════════════════════════════════════════════════════════════════════════
  TRAIN_MODE = os.environ.get("TRAIN_MODE", "corrdiff_residual")
  assert TRAIN_MODE in ("flow_matching", "corrdiff_residual"), \
      f"Bad TRAIN_MODE={TRAIN_MODE!r}"
  
  SIGMA_DATA = 0.1925  
  
  # ══════════════════════════════════════════════════════════════════════════════
  # 1.  PATHS
  # ══════════════════════════════════════════════════════════════════════════════
  RF_PATH  = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/data/RF_1975to2023.nc"
  ORO_PATH = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/data/oro.nc"
  D2M_PATH = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/data/era5_aligned_to_rf.nc"
  REG_CKPT = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/Variance/checkpoints/regressor/regressor_best.pth"
  CKPT_DIR = "checkpoints/v11_nrb3/"
  
  # ══════════════════════════════════════════════════════════════════════════════
  # 2.  HYPER-PARAMETERS
  # ══════════════════════════════════════════════════════════════════════════════
  BATCH        = 16
  ACCUM_STEPS  = 2
  LR           = 1e-4     
  MIN_LR       = LR * 0.1
  EPOCHS       = 1000      
  PATIENCE     = 200
  T_COND       = 5
  PRECIP_CH    = 0
  BASE_CH      = 256
  CHANNEL_MULT = (1, 2, 2, 4)
  NRB          = 3        
  DROPOUT      = 0.01
  FM_STEPS     = 15
  CFG_SCALE    = 1.5
  P_CFG_DROP   = 0.10
  WEIGHT_DECAY = 1e-3
  GRAD_CLIP    = 1.0
  EMA_DECAY    = 0.999
  N_ENS        = 2
  REG_IN_CH    = 2
  REG_D2M_CH   = 1
  D2M_CH       = 1
  UNET_D2M_CH  = 1
  UNET_VAR_MAP_CH = 1
  TOPO_CH      = 3
  GLOBAL_DIM   = 2
  UNET_IN_CH   = 1 + 1 + T_COND
  DS_FACTOR    = 4
  
  # ══════════════════════════════════════════════════════════════════════════════
  # 3.  EDM NOISE SCHEDULE
  # ══════════════════════════════════════════════════════════════════════════════
  def build_edm_schedule(n_steps, sigma_min=0.002, sigma_data=SIGMA_DATA, rho=7.0):
      sigma_max = 2.0 * sigma_data
      steps = torch.arange(n_steps, dtype=torch.float32) / max(n_steps - 1, 1)
      return (sigma_max**(1/rho) + steps*(sigma_min**(1/rho) - sigma_max**(1/rho)))**rho
  
  _EDM_CPU = build_edm_schedule(FM_STEPS)
  
  # ══════════════════════════════════════════════════════════════════════════════
  # 4.  EMA
  # ══════════════════════════════════════════════════════════════════════════════
  class EMA:
      def __init__(self, model, decay=EMA_DECAY):
          self.decay  = decay
          self.shadow = {k: v.clone().detach().float() for k, v in model.state_dict().items()}
  
      @torch.no_grad()
      def update(self, model):
          for k, v in model.state_dict().items():
              self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1-self.decay)
  
      def apply_to(self, model):
          model.load_state_dict({k: v.to(next(model.parameters()).device)
                                 for k, v in self.shadow.items()})
  
      def restore(self, model, state):
          model.load_state_dict(state)
  
  # ══════════════════════════════════════════════════════════════════════════════
  # 5.  AUGMENTATION
  # ══════════════════════════════════════════════════════════════════════════════
  def _make_aug_pipeline():
      if not HAS_ALB:
          return None
      return A.Compose([
          A.HorizontalFlip(p=0.5),
          A.VerticalFlip(p=0.5),
          A.RandomRotate90(p=0.5),
          A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4),
          A.GaussNoise(var_limit=(1e-5, 5e-4), p=0.25),
          A.RandomGamma(gamma_limit=(70, 150), p=0.3),
      ], additional_targets={"topo": "image", "d2m":  "image"})
  _AUG_PIPELINE = _make_aug_pipeline()
  
  def augment_sample(fp_t, topo_t, d2m_t, aug_prob=0.5):
      def _rederive_coarse(fp):
          return F.avg_pool2d(fp.unsqueeze(0), kernel_size=DS_FACTOR, stride=DS_FACTOR).squeeze(0)
  
      if _AUG_PIPELINE is None or np.random.rand() > aug_prob:
          return fp_t, topo_t, _rederive_coarse(fp_t), d2m_t
  
      fp_np   = fp_t.squeeze(0).numpy().astype(np.float32)
      topo_np = topo_t.squeeze(0).numpy().astype(np.float32)
      d2m_np  = (d2m_t.squeeze(0).numpy().astype(np.float32) if d2m_t is not None else np.zeros_like(fp_np))
  
      result   = _AUG_PIPELINE(image=fp_np, topo=topo_np, d2m=d2m_np)
      fp_aug   = torch.from_numpy(result["image"]).unsqueeze(0)
      topo_aug = torch.from_numpy(result["topo"]).unsqueeze(0)
      d2m_aug  = (torch.from_numpy(result["d2m"]).unsqueeze(0) if d2m_t is not None else None)
      coarse_aug = _rederive_coarse(fp_aug)
      return fp_aug, topo_aug, coarse_aug, d2m_aug
  
  # ══════════════════════════════════════════════════════════════════════════════
  # 6.  TOPO HELPERS
  # ══════════════════════════════════════════════════════════════════════════════
  def compute_slope_aspect(elev, global_elev_max=8600.0, global_slope_max=1.5):
      kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=elev.device).view(1, 1, 3, 3)
      ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=elev.device).view(1, 1, 3, 3)
      
      e = elev.float()
      dx = F.conv2d(e, kx, padding=1)
      dy = F.conv2d(e, ky, padding=1)
      slope = torch.sqrt(dx**2 + dy**2 + 1e-8)
      aspect = torch.atan2(dy, dx)
      
      def global_norm(t, g_min, g_max):
          return 2 * (t - g_min) / (g_max - g_min + 1e-8) - 1
  
      return torch.cat([
          global_norm(e, 0.0, global_elev_max), 
          global_norm(slope, 0.0, global_slope_max), 
          aspect / math.pi 
      ], dim=1)
  
  def expand_topo(topo_1ch):
      return torch.cat([compute_slope_aspect(topo_1ch[i:i+1]) for i in range(topo_1ch.shape[0])], dim=0)
  
  def build_coarse_input(coarse, var_map):
      Hc, Wc = coarse.shape[-2], coarse.shape[-1]
      return torch.cat([coarse, F.adaptive_avg_pool2d(var_map, (Hc, Wc))], dim=1)
  
  # ══════════════════════════════════════════════════════════════════════════════
  # 7.  TEMPORAL CONDITIONING
  # ══════════════════════════════════════════════════════════════════════════════
  def build_temporal_cond(batch, dev, n_frames=T_COND):
      if "tc_frames" in batch:
          tc = batch["tc_frames"].to(dev, non_blocking=True)
          B, T, Hc, Wc = tc.shape
          tc_reshaped = tc.view(B * T, 1, Hc, Wc)
          tc_up = F.interpolate(tc_reshaped, scale_factor=4, mode='bilinear', align_corners=False)
          return tc_up.view(B, T, tc_up.shape[-2], tc_up.shape[-1])
  
      coarse    = batch["coarse"].to(dev, non_blocking=True)
      coarse_up = F.interpolate(coarse, scale_factor=4, mode='bilinear', align_corners=False)
      return coarse_up.expand(-1, n_frames, -1, -1)
  
  # ══════════════════════════════════════════════════════════════════════════════
  # 8.  AUXILIARY LOSSES & METRICS
  # ══════════════════════════════════════════════════════════════════════════════
  def nuclear_pcc_push(pred, target):
      """
      Structural Phase Alignment via Log-Cosh PCC.
      Pushes harder as the correlation improves to overcome the 0.85 plateau.
      """
      p_mu = pred.mean(dim=[-1, -2], keepdim=True)
      t_mu = target.mean(dim=[-1, -2], keepdim=True)
      p_c = pred - p_mu
      t_c = target - t_mu
      
      cov = (p_c * t_c).sum(dim=[-1, -2])
      p_var = (p_c**2).sum(dim=[-1, -2])
      t_var = (t_c**2).sum(dim=[-1, -2])
      
      var_penalty = torch.abs(torch.log(p_var + 1e-6) - torch.log(t_var + 1e-6)).mean()
      
      pcc = cov / (torch.sqrt(p_var * t_var) + 1e-8)
      error = 1.0 - pcc.mean()
      
      # Steepen the Log-Cosh gradient even more
      return torch.log(torch.cosh(error * 5.0)) + (0.1 * var_penalty)
  
  def precipitation_weighted_mae(pred, target, min_snr_weight, alpha=1.5):
      """
      Weights the absolute error by the intensity of the target rainfall.
      alpha > 0 forces the model to care exponentially more about heavy extremes.
      """
      # Create a weight map based on true rainfall intensity
      # +1.0 ensures background (0.0) regions still get standard MAE gradients
      precip_weights = (torch.clamp(target.float(), min=0.0) ** alpha) + 1.0
      
      mae = torch.abs(pred.float() - target.float())
      
      # Apply both the diffusion SNR weighting and the physical precipitation weighting
      return (mae * precip_weights * min_snr_weight).mean()
  
  def soft_dry_penalty(pred, target):
      """
      A gentle alternative to the 'bridge-killer'. 
      Asymmetrically penalizes false-positive drizzle on genuinely dry pixels.
      """
      dry_mask = (target <= 1e-4).float()
      # Only penalize predictions that exceed a trace amount (e.g., 0.1 mm)
      false_rain = torch.relu(pred.float() - 0.1) 
      return (false_rain * dry_mask).mean()
  
  def mass_conservation_loss(pred, target):
      """
      Ensures the total water volume is roughly conserved across the domain.
      """
      pred_total = pred.float().mean(dim=[-1, -2])
      target_total = target.float().mean(dim=[-1, -2])
      return torch.abs(pred_total - target_total).mean()
  
  def hybrid_sigma_loss(pred, target, sigma, epoch=0):
      # 1. Delay the PCC push so the model learns structural baseline first
      lambda_pcc = 0.0 if epoch < 10 else min(1.0 + (epoch / 100.0), 6.0) 
      lambda_spectral = 0.10
      lambda_dry = 0.05   # Tuning parameter for the false-drizzle penalty
      lambda_mass = 0.2  # Tuning parameter for total volume conservation
      
      eps = 1e-6
      snr = (SIGMA_DATA ** 2) / (sigma ** 2 + eps)
      min_snr_weight = torch.clamp(snr, max=5.0)
      
      # Upgraded Spatial Loss: Focuses on extremes
      spatial_loss = precipitation_weighted_mae(pred, target, min_snr_weight, alpha=1.5)
      
      # 2. Phase Alignment (Your Nuclear PCC)
      t_var = torch.var(target.float(), dim=[-1, -2])
      valid_pcc_mask = (t_var > 1e-4).float().view(-1, 1, 1)
      
      if valid_pcc_mask.sum() > 0:
          pcc_loss = nuclear_pcc_push(pred.float() * valid_pcc_mask, target.float() * valid_pcc_mask)
      else:
          pcc_loss = torch.tensor(0.0, device=pred.device)
      
      # 3. Spectral Loss
      fft_pred = torch.fft.rfft2(pred.float()).abs()
      fft_target = torch.fft.rfft2(target.float()).abs()
      spectral_loss = torch.abs(fft_pred - fft_target).mean()
      
      # 4. Physical / Domain Constraints
      dry_loss = soft_dry_penalty(pred, target)
      mass_loss = mass_conservation_loss(pred, target)
  
      return spatial_loss + (lambda_spectral * spectral_loss) + (lambda_pcc * pcc_loss) + (lambda_dry * dry_loss) + (lambda_mass * mass_loss)
  
  @torch.no_grad()
  def weighted_pcc(pred, target, lat_w=None):
      B = pred.shape[0]
      p = pred.float().view(B, -1); t = target.float().view(B, -1)
      if lat_w is not None:
          w = lat_w.view(1, -1).expand_as(p)
          w = w / w.sum(dim=1, keepdim=True)
          pm = (p * w).sum(dim=1, keepdim=True); tm = (t * w).sum(dim=1, keepdim=True)
          pm_c = p - pm; tm_c = t - tm
          r = (pm_c * tm_c * w).sum(dim=1) / (torch.sqrt((pm_c**2 * w).sum(dim=1) * (tm_c**2 * w).sum(dim=1)) + 1e-8)
      else:
          pm = p.mean(dim=1, keepdim=True); tm = t.mean(dim=1, keepdim=True)
          pm_c = p - pm; tm_c = t - tm
          p_l2 = torch.sqrt((pm_c**2).sum(dim=1))
          t_l2 = torch.sqrt((tm_c**2).sum(dim=1))
          r = (pm_c * tm_c).sum(dim=1) / (p_l2 * t_l2 + 1e-8)
      return r.mean().item()
      
  @torch.no_grad()
  def raw_weighted_pcc(pred_log, target_log, lat_w=None):
      pred_raw = torch.clamp(torch.expm1(pred_log.float()), min=0.0)
      target_raw = torch.clamp(torch.expm1(target_log.float()), min=0.0)
      return weighted_pcc(pred_raw, target_raw, lat_w)
  
  @torch.no_grad()
  def crps_ensemble(samples, target):
      N = samples.shape[0]
      if N == 1:
          return (samples[0] - target).abs().mean().item()
      mae = (samples - target.unsqueeze(0)).abs().mean(0)
      pair = (samples.unsqueeze(0) - samples.unsqueeze(1)).abs()
      return (mae - 0.5 / N / (N - 1) * pair.sum([0, 1])).mean().item()
  
  @torch.no_grad()
  def psd_tail_ratio(pred, target, hff=0.3):
      P=torch.fft.rfft2(pred.float()).abs(); T=torch.fft.rfft2(target.float()).abs()
      c=int((1-hff)*P.shape[-1])
      return (P[...,c:].mean()/(T[...,c:].mean()+1e-8)).item()
  
  @torch.no_grad()
  def fractions_skill_score(pred, target, threshold=0.5, window=5):
      p_bin = (pred > threshold).float()
      t_bin = (target > threshold).float()
      p_frac = F.avg_pool2d(p_bin, kernel_size=window, stride=1, padding=window//2)
      t_frac = F.avg_pool2d(t_bin, kernel_size=window, stride=1, padding=window//2)
      mse = ((p_frac - t_frac)**2).mean(dim=[-1, -2])
      ref = (p_frac**2).mean(dim=[-1, -2]) + (t_frac**2).mean(dim=[-1, -2])
      fss = 1.0 - mse / (ref + 1e-8)
      return fss.mean().item()
  
  # ══════════════════════════════════════════════════════════════════════════════
  # 9.  DDP SETUP
  # ══════════════════════════════════════════════════════════════════════════════
  def setup():
      rank=int(os.environ.get("RANK",0)); ws=int(os.environ.get("WORLD_SIZE",1))
      lr_=int(os.environ.get("LOCAL_RANK",0))
      if ws>1: dist.init_process_group("nccl"); torch.cuda.set_device(lr_)
      torch.backends.cudnn.benchmark = True
      torch.backends.cuda.matmul.allow_tf32 = True   
      torch.backends.cudnn.allow_tf32 = True
      return rank,ws,lr_,torch.device(f"cuda:{lr_}" if torch.cuda.is_available() else "cpu")
  
  def ar(t, ws):
      if ws>1 and dist.is_initialized(): dist.all_reduce(t,op=dist.ReduceOp.SUM); t/=ws
      return t
  
  # ══════════════════════════════════════════════════════════════════════════════
  # 10.  REGRESSOR LOADER
  # ══════════════════════════════════════════════════════════════════════════════
  def load_regressor(ckpt_path, dev):
      ck=torch.load(ckpt_path, map_location=dev)
      reg=CorrDiffRegressor(
          in_channels=ck.get("reg_in_channels", REG_IN_CH), out_channels=1,
          base_channels=64, channel_mult=(1,2,4), num_blocks=2,
          global_dim=GLOBAL_DIM, topo_channels=TOPO_CH,
          d2m_channels=ck.get("d2m_channels", REG_D2M_CH),   
          use_d2m=ck.get("use_d2m", True),
      ).to(dev)
      state={k.replace("module.",""):v for k,v in ck["model_state_dict"].items()}
      reg.load_state_dict(state); reg.eval()
      for p in reg.parameters(): p.requires_grad_(False)
      return reg
  
  # ══════════════════════════════════════════════════════════════════════════════
  # 11.  DDIM SAMPLER  (V2)
  # ══════════════════════════════════════════════════════════════════════════════
  @torch.no_grad()
  def ddim_sample(raw_model, mu, tc, tp, gf, d2m, var_map, edm_schedule, dev):
      B = mu.shape[0]
      sigmas = edm_schedule.to(dev)
      
      # Initialize exactly at the schedule's starting noise level
      x_t = torch.randn_like(mu) * sigmas[0]
  
      for i, sigma_cur in enumerate(sigmas):
          s_cur  = sigma_cur.view(1, 1, 1, 1)
          c_in   = 1. / torch.sqrt(s_cur**2 + SIGMA_DATA**2)
          c_out  = s_cur * SIGMA_DATA / torch.sqrt(s_cur**2 + SIGMA_DATA**2)
          c_skip = SIGMA_DATA**2 / (s_cur**2 + SIGMA_DATA**2)
          c_n    = (sigma_cur.log() / 4).expand(B)
          x_t_scaled = x_t * c_in
          x_in = torch.cat([c_in * x_t, mu, tc], dim=1)
          D_pred = raw_model(x_in, c_n, topo=tp, global_features=gf, d2m=d2m, var_map=var_map, T=T_COND)
          x0_hat = c_skip * x_t[:, :1] + c_out * D_pred
          
          if i < len(sigmas) - 1:
              sigma_next = sigmas[i + 1].view(1, 1, 1, 1)
              x_t = x0_hat + sigma_next * (x_t - x0_hat) / s_cur.clamp(min=1e-8)
          else:
              x_t = x0_hat
      return x_t
  
  # ══════════════════════════════════════════════════════════════════════════════
  # 12.  TRAINING LOOP
  # ══════════════════════════════════════════════════════════════════════════════
  def train():
      rank,ws,lr_,dev=setup()
      os.makedirs(CKPT_DIR, exist_ok=True)
  
      SAVE       = os.path.join(CKPT_DIR,f"unet_{TRAIN_MODE}_nrb{NRB}_sigma{SIGMA_DATA:.3f}_best.pth")
      LATEST     = os.path.join(CKPT_DIR,f"unet_{TRAIN_MODE}_nrb{NRB}_sigma{SIGMA_DATA:.3f}_latest.pth")
      QDM_BEST   = os.path.join(CKPT_DIR,f"qdm_{TRAIN_MODE}_nrb{NRB}_best.pth")
      QDM_PATH = os.path.join(CKPT_DIR,f"qdm_{TRAIN_MODE}_nrb{NRB}_latest.pth")
  
      edm_schedule=_EDM_CPU.clone()
  
      ds =UpscaleDataset(RF_PATH,ORO_PATH,d2m_file=D2M_PATH, split="train",normalize=True,device="cpu")
      n  =len(ds); trn=int(0.70*n); van=int(0.10*n)
  
      # SHARDED DATASETS
      train_sub = Subset(ds, range(0, trn))
      val_sub   = Subset(ds, range(trn, trn+van))
      
      train_sampler = DistributedSampler(train_sub, num_replicas=ws, rank=rank, shuffle=True, drop_last=True) if ws > 1 else None
      val_sampler   = DistributedSampler(val_sub, num_replicas=ws, rank=rank, shuffle=False) if ws > 1 else None
  
      _loader_kwargs = dict(num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2)
      
      trl = DataLoader(train_sub, BATCH, sampler=train_sampler, shuffle=(train_sampler is None), drop_last=True, **_loader_kwargs)
      val = DataLoader(val_sub, BATCH, sampler=val_sampler, shuffle=False, **_loader_kwargs)
  
      reg=load_regressor(REG_CKPT,dev)
      if rank==0: print(f"[Stage-1] Regressor loaded (frozen)")
  
      if TRAIN_MODE=="flow_matching":
          fm=FlowMatching(n_steps=FM_STEPS,cfg_scale=CFG_SCALE); loss_label="L_flow  "
      else:
          fm=None; loss_label="L_denoise"
  
      model=UNet(
          in_channels=UNET_IN_CH, out_channels=1, base_channels=BASE_CH,
          channel_mult=CHANNEL_MULT, num_res_blocks=NRB, dropout=DROPOUT,
          global_dim=GLOBAL_DIM, use_bottleneck_attention=True,
          topo_channels=TOPO_CH, use_d2m=True,
          d2m_channels=UNET_D2M_CH,          
          use_var_map=True,                  
          var_map_channels=UNET_VAR_MAP_CH,  
          temporal_frames=T_COND,
      ).to(dev)
  
      if ws>1:
          model=nn.parallel.DistributedDataParallel(model,device_ids=[lr_],find_unused_parameters=False)
      raw=model.module if ws>1 else model
  
      ema   =EMA(raw,decay=EMA_DECAY)
      opt   =AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY,betas=(0.9,0.999))
      scaler=GradScaler(device=dev.type)
      sched =CosineAnnealingWarmRestarts(opt,T_0=50,T_mult=1,eta_min=MIN_LR)
  
      start = 0
      best_val_loss = float('inf')
      best_val_pcc = -1.0      
      no_improve = 0
  
      if os.path.exists(SAVE):
          ck=torch.load(SAVE,map_location=dev)
          try:
              raw.load_state_dict({k.replace("module.",""):v for k,v in ck["model_state_dict"].items()})
              opt.load_state_dict(ck["optimizer_state_dict"])
              start=ck.get("epoch",0)+1
              best_val_loss=ck.get("val_loss", float('inf'))
              best_val_pcc=ck.get("best_val_pcc", -1.0)
              no_improve=ck.get("no_improve",0)
              if "ema_shadow" in ck:
                  ema.shadow={k:v.to(dev) for k,v in ck["ema_shadow"].items()}
              if rank==0: print(f"[RESUME BEST] ep={start}  best_val_pcc={best_val_pcc:.4f}")
          except RuntimeError as e:
              if rank==0: print(f"[RESUME ABORTED] Model shapes differ (likely NRB change), starting fresh: {e}")
  
      if rank==0:
          np_=sum(p.numel() for p in model.parameters() if p.requires_grad)
          print(f"[MODEL]  UNet {np_/1e6:.2f}M  mode={TRAIN_MODE}  sigma_data={SIGMA_DATA}")
          print(f"[COND]   UNET_D2M_CH={UNET_D2M_CH}  UNET_VAR_MAP_CH={UNET_VAR_MAP_CH}  T_COND={T_COND}")
          print(f"[OPTIM]  LR={LR}  ACCUM={ACCUM_STEPS}  eff_batch={BATCH*ACCUM_STEPS*ws}")
          print(f"[PERF]   DDP Enabled (ws={ws})  persistent_workers=True  prefetch_factor=2")
          hdr=(f"{'Ep':>5}|{loss_label:>10}|{'ValLoss':>8}|{'wPCC':>7}|{'CRPS':>8}|{'PSD_r':>7}|{'FSS':>6}|{'LR':>9}")
          print(hdr); print("-"*len(hdr))
  
      lat_w=None
  
      for ep in range(start, EPOCHS):
          if train_sampler is not None:
              train_sampler.set_epoch(ep)
  
          model.train()
          t0=time.time(); sum_ml=nb=0.
          opt.zero_grad(set_to_none=True)
          _opt_steps=0   
  
          for step, b in enumerate(trl, 1):
              try:
                  fp       = b["fine"].to(dev, non_blocking=True)[:,PRECIP_CH:PRECIP_CH+1]
                  topo_1ch = b["topo"].to(dev, non_blocking=True)
                  xi_raw   = b["coarse"].to(dev, non_blocking=True)
                  d2m      = b["d2m"].to(dev, non_blocking=True) if "d2m" in b else None
                  var_map  = b["var_map"].to(dev, non_blocking=True)
                  gf       = torch.stack([b["doy"],b["hour"]],1).float().to(dev, non_blocking=True)
                  tc       = build_temporal_cond(b, dev)
  
                  if torch.rand(1).item() < 0.5:
                      fp = fp.flip(-1); topo_1ch = topo_1ch.flip(-1)
                      if d2m is not None: d2m = d2m.flip(-1)
                      var_map = var_map.flip(-1)
                  if torch.rand(1).item() < 0.5:
                      fp = fp.flip(-2); topo_1ch = topo_1ch.flip(-2)
                      if d2m is not None: d2m = d2m.flip(-2)
                      var_map = var_map.flip(-2)
  
                  xi_raw = F.avg_pool2d(fp, kernel_size=DS_FACTOR, stride=DS_FACTOR)
                  tp  = expand_topo(topo_1ch)
                  xi  = build_coarse_input(xi_raw, var_map)
  
                  with torch.no_grad(): mu = reg(xi, topo=tp, global_features=gf, d2m=d2m)
  
                  residual = fp - mu
                  cfg_drop = (torch.rand(fp.shape[0], device=dev) < P_CFG_DROP)
  
                  if TRAIN_MODE == "flow_matching":
                      x_t, t_vec, v_star = fm.get_train_sample(residual)
                      x_in    = torch.cat([x_t, mu, tc], dim=1)
                      with autocast(device_type=dev.type, dtype=torch.bfloat16):
                          v_pred   = model(x_in, t_vec, topo=tp, global_features=gf, cfg_drop=cfg_drop, d2m=d2m, var_map=var_map, T=T_COND)
                          loss     = fm.loss(v_pred, v_star) / ACCUM_STEPS
  
                  else:
                      idx     = torch.randint(0, len(edm_schedule), (fp.shape[0],))
                      sigma_t = edm_schedule[idx].to(dev).view(-1,1,1,1)
                      eps     = torch.randn_like(residual)
                      x_t     = residual + sigma_t*eps
                      c_in    = 1./torch.sqrt(sigma_t**2 + SIGMA_DATA**2)
                      c_out   = sigma_t*SIGMA_DATA/torch.sqrt(sigma_t**2 + SIGMA_DATA**2)
                      c_skip  = SIGMA_DATA**2/(sigma_t**2 + SIGMA_DATA**2)
                      c_n     = (sigma_t.log()/4).view(fp.shape[0])
                      x_in = torch.cat([c_in * x_t, mu, tc], dim=1)
                      
                      with autocast(device_type=dev.type, dtype=torch.bfloat16):
                          D_pred = model(x_in, c_n, topo=tp, global_features=gf, cfg_drop=cfg_drop, d2m=d2m, var_map=var_map, T=T_COND)
                          x0_pred = c_skip*x_t[:,:1] + c_out*D_pred
                          loss = hybrid_sigma_loss(x0_pred, residual, sigma_t, epoch=ep) / ACCUM_STEPS
  
                  scaler.scale(loss).backward()
  
                  if step % ACCUM_STEPS == 0:
                      scaler.unscale_(opt)
                      nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                      if not torch.isfinite(loss):
                          print(f"[{rank}] Detected NaN loss, skipping batch")
                          opt.zero_grad(set_to_none=True)
                          continue 
                      scaler.step(opt); scaler.update()
                      opt.zero_grad(set_to_none=True)
                      ema.update(raw)   
                      _opt_steps += 1
  
                  with torch.no_grad(): sum_ml+=loss.item()*ACCUM_STEPS; nb+=1
  
              except Exception:
                  if rank==0: traceback.print_exc()
                  raise
  
          sched.step()
          te_ml=ar(torch.tensor(sum_ml/max(nb,1),device=dev),ws).item()
          lr_now=opt.param_groups[0]["lr"]
  
          # ── VALIDATION (EMA weights & DDP Metrics) ────────────────────────
          live_state=deepcopy(raw.state_dict()); ema.apply_to(raw); model.eval()
          
          # [v_loss, vn, pcc_sum, raw_pcc_sum, crps_sum, psd_sum, fss_sum]
          v_metrics = torch.zeros(7, device=dev) 
          vb = torch.tensor(0, device=dev, dtype=torch.long)
  
          do_heavy_eval = (ep + 1) % 5 == 0
  
          with torch.no_grad():
              for b in val:
                  fp      = b["fine"].to(dev, non_blocking=True)[:,PRECIP_CH:PRECIP_CH+1]
                  tp      = expand_topo(b["topo"].to(dev, non_blocking=True))
                  gf      = torch.stack([b["doy"],b["hour"]],1).float().to(dev, non_blocking=True)
                  xi_raw  = b["coarse"].to(dev, non_blocking=True)
                  var_map = b["var_map"].to(dev, non_blocking=True)
                  d2m     = b["d2m"].to(dev, non_blocking=True) if "d2m" in b else None
                  tc      = build_temporal_cond(b, dev)
                  xi      = build_coarse_input(xi_raw, var_map)
  
                  with autocast(device_type=dev.type, dtype=torch.bfloat16):
                      mu = reg(xi, topo=tp, global_features=gf, d2m=d2m)
  
                  residual=fp-mu
  
                  if TRAIN_MODE=="flow_matching":
                      x_t,t_vec,v_star=fm.get_train_sample(residual)
                      x_in = torch.cat([c_in * x_t, mu, tc], dim=1)
                      with autocast(device_type=dev.type, dtype=torch.bfloat16):
                          v_pred=model(x_in,t_vec,topo=tp,global_features=gf, d2m=d2m,var_map=var_map,T=T_COND)
                          l_val=fm.loss(v_pred,v_star)
                  else:
                      idx=torch.randint(0,len(edm_schedule),(fp.shape[0],))
                      sigma_t=edm_schedule[idx].to(dev).view(-1,1,1,1)
                      eps=torch.randn_like(residual); x_t=residual+sigma_t*eps
                      c_in=1./torch.sqrt(sigma_t**2+SIGMA_DATA**2)
                      c_out=sigma_t*SIGMA_DATA/torch.sqrt(sigma_t**2+SIGMA_DATA**2)
                      c_skip=SIGMA_DATA**2/(sigma_t**2+SIGMA_DATA**2)
                      c_n=(sigma_t.log()/4).view(fp.shape[0])
                      x_in = torch.cat([c_in * x_t, mu, tc], dim=1)
                      with autocast(device_type=dev.type, dtype=torch.bfloat16):
                          D_pred = model(x_in, c_n, topo=tp, global_features=gf, d2m=d2m, var_map=var_map, T=T_COND)
                          x0_pred=c_skip*x_t[:,:1]+c_out*D_pred
                          l_val= hybrid_sigma_loss(x0_pred, residual, sigma_t, epoch=ep)
                  v_metrics[0] += l_val * fp.shape[0]
                  v_metrics[1] += fp.shape[0]
  
                  if do_heavy_eval:
                      samples=[]
                      for _ in range(N_ENS):
                          if TRAIN_MODE=="flow_matching":
                              x_cond=torch.cat([mu,tc],dim=1)
                              s=fm.sample(raw,x_cond,topo=tp,global_features=gf, d2m=d2m,var_map=var_map,cfg_scale=CFG_SCALE,T=T_COND)+mu
                          else:
                              s=ddim_sample(raw,mu,tc,tp,gf,d2m,var_map,edm_schedule,dev)+mu
                          s=PhysicsGuide.apply(s,xi_raw,enforce_mass=True,enforce_dry=True)
                          samples.append(s)
  
                      samples_t = torch.stack(samples) # Define it here!
                      samples_phys = [torch.expm1(s.clamp(min=0.0)) for s in samples]
                      mean_phys = torch.stack(samples_phys).mean(0)
                      best_s = torch.log1p(mean_phys)
                      
                      v_metrics[2] += weighted_pcc(best_s, fp, lat_w)
                      v_metrics[3] += raw_weighted_pcc(best_s, fp, lat_w)
                      v_metrics[4] += crps_ensemble(samples_t, fp)
                      v_metrics[5] += psd_tail_ratio(best_s, fp)
                      v_metrics[6] += fractions_skill_score(best_s, fp, threshold=0.5, window=5)
                      vb += 1
  
          ema.restore(raw,live_state); model.train()
  
          # DDP GLOBAL SYNCHRONIZATION
          if ws>1 and dist.is_initialized():
              dist.all_reduce(v_metrics, op=dist.ReduceOp.SUM)
              dist.all_reduce(vb, op=dist.ReduceOp.SUM)
  
          vw = (v_metrics[0] / v_metrics[1].clamp(min=1)).item()
          el = time.time()-t0
          raw_pcc = -1.0 
          is_new_best = False
  
          if do_heavy_eval:
              wpcc    = (v_metrics[2] / vb.clamp(min=1)).item()
              raw_pcc = (v_metrics[3] / vb.clamp(min=1)).item()
              crps_v  = (v_metrics[4] / vb.clamp(min=1)).item()
              psd_r   = (v_metrics[5] / vb.clamp(min=1)).item()
              fss_v   = (v_metrics[6] / vb.clamp(min=1)).item()
              
              if rank==0:
                  star=" ★" if raw_pcc > best_val_pcc else "" 
                  print(f"{ep:>5}|{te_ml:>10.5f}|{vw:>8.4f}| L-PCC:{wpcc:>6.4f} | R-PCC:{raw_pcc:>6.4f} |"
                        f" CRPS:{crps_v:>6.4f}|{psd_r:>7.3f}|{fss_v:>6.3f}|{lr_now:>9.2e}  [{el:.0f}s]{star}")
  
                  ck_base={
                      "epoch":ep,"model_state_dict":raw.state_dict(),
                      "optimizer_state_dict":opt.state_dict(),"ema_shadow":ema.shadow,
                      "val_loss":vw,"wpcc":wpcc,"best_val_pcc": max(best_val_pcc, raw_pcc),"crps":crps_v,"psd_tail_ratio":psd_r,
                      "no_improve":no_improve,"train_mode":TRAIN_MODE,"sigma_data":SIGMA_DATA,
                      "unet_in_channels":UNET_IN_CH,"t_cond":T_COND,
                      "d2m_channels":UNET_D2M_CH,"var_map_channels":UNET_VAR_MAP_CH,
                      "reg_d2m_channels":REG_D2M_CH,
                  }
                  torch.save(ck_base, LATEST)
  
                  if raw_pcc > best_val_pcc:
                      best_val_pcc = raw_pcc
                      is_new_best = True
                      no_improve = 0
                      torch.save(ck_base, SAVE)
                      print(f"  ★ BEST R-PCC={raw_pcc:.4f}  val_loss={vw:.4f}  CRPS={crps_v:.4f}  PSD_r={psd_r:.3f}  FSS={fss_v:.3f}")
                      if (ep+1)%50==0:
                          torch.save(ck_base, os.path.join(CKPT_DIR,f"unet_{TRAIN_MODE}_nrb{NRB}_ep{ep+1:04d}_rpcc{raw_pcc:.4f}.pth"))
                  else:
                      no_improve += 2
          else:
              if rank==0:
                  print(f"{ep:>5}|{te_ml:>10.5f}|{vw:>8.4f}| ------ FAST EPOCH ------ |{lr_now:>9.2e}  [{el:.0f}s]")
  
          # ── ISOLATED QDM CALIBRATION (Rank 0 ONLY) ────────────────────────
          if (ep + 1) % 5 == 0 or raw_pcc > best_val_pcc:   
              if rank == 0:
                  print(f"\n[QDM] Starting isolated calibration on rank 0 (Full validation split)...")
                  # We spin up a fresh UN-SHARDED dataloader for QDM so it fits the true global distribution
                  val_qdm_loader = DataLoader(val_sub, BATCH, shuffle=False, num_workers=4, pin_memory=True)
                  
                  _qdm_latest = QDM(n_quantiles=500)
                  _all_pred, _all_obs = [], []
                  _live2 = deepcopy(raw.state_dict())
                  ema.apply_to(raw); model.eval()
                  
                  with torch.no_grad():
                      for _b in val_qdm_loader:
                          _fp     = _b["fine"].to(dev, non_blocking=True)[:,PRECIP_CH:PRECIP_CH+1]
                          _tp     = expand_topo(_b["topo"].to(dev, non_blocking=True))
                          _gf     = torch.stack([_b["doy"],_b["hour"]],1).float().to(dev, non_blocking=True)
                          _xi_raw = _b["coarse"].to(dev, non_blocking=True)
                          _vm     = _b["var_map"].to(dev, non_blocking=True)
                          _d2m    = _b["d2m"].to(dev, non_blocking=True) if "d2m" in _b else None
                          _tc     = build_temporal_cond(_b, dev)
                          _xi     = build_coarse_input(_xi_raw, _vm)
                          _mu     = reg(_xi, topo=_tp, global_features=_gf, d2m=_d2m)
                          
                          if fm is not None:
                              _xc = torch.cat([_mu, _tc], dim=1)
                              _s  = fm.sample(raw, _xc, topo=_tp, global_features=_gf, d2m=_d2m, var_map=_vm, cfg_scale=CFG_SCALE, T=T_COND) + _mu
                          else:
                              _s  = ddim_sample(raw, _mu, _tc, _tp, _gf, _d2m, _vm, edm_schedule, dev) + _mu
                              
                          _s = PhysicsGuide.apply(_s, _xi_raw, enforce_mass=True, enforce_dry=False)
                          _all_pred.append(_s.cpu()); _all_obs.append(_fp.cpu())
                          
                  ema.restore(raw, _live2); model.train()
                  _qdm_latest.fit(torch.cat(_all_pred), torch.cat(_all_obs))
                  _qdm_latest.save(QDM_PATH)
                  print(f"[QDM] Calibration complete & saved to {QDM_PATH}\n")
                  if is_new_best:
                      _qdm_latest.save(QDM_BEST)
                      print(f"[QDM] ★ BEST calibration saved to {QDM_BEST}\n")
                  else:
                      print()
              
              # CRITICAL: GPUs 1, 2, and 3 MUST wait here for Rank 0 to finish QDM
              if ws > 1 and dist.is_initialized():
                  dist.barrier()
  
          if no_improve>=PATIENCE:
              if rank==0: print(f"\n⚠  Early stop ep={ep+1}  best_val_pcc={best_val_pcc:.4f}")
              break
  
      if ws>1 and dist.is_initialized(): dist.destroy_process_group()
  
  # ══════════════════════════════════════════════════════════════════════════════
  # 13.  INFERENCE
  # ══════════════════════════════════════════════════════════════════════════════
  class CorrDiffInference:
      def __init__(self, reg_ckpt, unet_ckpt, qdm_ckpt, device, cfg_scale=CFG_SCALE, n_ens=8):
          self.dev=device; self.cfg_scale=cfg_scale; self.n_ens=n_ens
          self.reg=load_regressor(reg_ckpt,device)
          ck=torch.load(unet_ckpt,map_location=device)
          tc=ck.get("t_cond",T_COND); self.tc=tc
          self.train_mode=ck.get("train_mode","flow_matching")
          self.fm=FlowMatching(n_steps=FM_STEPS,cfg_scale=cfg_scale)
          self.edm_sched=build_edm_schedule(FM_STEPS,sigma_data=ck.get("sigma_data",SIGMA_DATA))
          self.unet=UNet(
              in_channels=ck.get("unet_in_channels",UNET_IN_CH),out_channels=1,
              base_channels=BASE_CH,channel_mult=CHANNEL_MULT,num_res_blocks=NRB,dropout=0.,
              global_dim=GLOBAL_DIM,topo_channels=TOPO_CH,use_d2m=True,
              d2m_channels=ck.get("d2m_channels",UNET_D2M_CH),
              use_var_map=True,
              var_map_channels=ck.get("var_map_channels",UNET_VAR_MAP_CH),
              temporal_frames=tc,
          ).to(device)
          state=({k:v.to(device) for k,v in ck["ema_shadow"].items()} if "ema_shadow" in ck
                 else {k.replace("module.",""):v for k,v in ck["model_state_dict"].items()})
          self.unet.load_state_dict(state); self.unet.eval()
          self.qdm=QDM.load(qdm_ckpt) if qdm_ckpt and os.path.exists(qdm_ckpt) else None
  
      @torch.no_grad()
      def predict(self, coarse, var_map, topo, d2m, doy, hour, tc_frames=None):
          dev=self.dev
          coarse=coarse.to(dev); var_map=var_map.to(dev)
          topo=topo.to(dev); d2m=d2m.to(dev)
          gf=torch.stack([doy.to(dev),hour.to(dev)],dim=1).float()
          xi=build_coarse_input(coarse,var_map); tp=expand_topo(topo)
          if tc_frames is None:
              coarse_up=F.interpolate(coarse,scale_factor=4,mode='bilinear',align_corners=False)
              tc_frames=coarse_up.expand(-1,self.tc,-1,-1).to(dev)
          else:
              tc_frames=tc_frames.to(dev)
  
          mu=self.reg(xi,topo=tp,global_features=gf,d2m=d2m)
  
          samples=[]
          for _ in range(self.n_ens):
              if self.train_mode=="flow_matching":
                  x_cond=torch.cat([mu,tc_frames],dim=1)
                  s=self.fm.sample(self.unet,x_cond,topo=tp,global_features=gf, d2m=d2m,var_map=var_map,cfg_scale=self.cfg_scale,T=self.tc)+mu
              else:
                  s=ddim_sample(self.unet,mu,tc_frames,tp,gf,d2m,var_map, self.edm_sched,dev)+mu
              s=PhysicsGuide.apply(s,coarse,enforce_mass=True,enforce_dry=True)
              if self.qdm is not None: s=self.qdm.apply(s)
              samples.append(s)
          samples=torch.stack(samples)
          return {"mean":samples.mean(0),"std":samples.std(0),"samples":samples,"mu":mu}
  
  # ══════════════════════════════════════════════════════════════════════════════
  # ENTRY POINT
  # ══════════════════════════════════════════════════════════════════════════════
  if __name__=="__main__":
      parser=argparse.ArgumentParser()
      parser.add_argument("--mode",    default=None,choices=["flow_matching","corrdiff_residual"])
      parser.add_argument("--epochs",  type=int,   default=None)
      parser.add_argument("--batch",   type=int,   default=None)
      parser.add_argument("--lr",      type=float, default=None)
      parser.add_argument("--dropout", type=float, default=None)
      parser.add_argument("--patience",type=int,   default=None)
      args=parser.parse_args()
      if args.mode     is not None and "TRAIN_MODE" not in os.environ: TRAIN_MODE=args.mode
      if args.epochs   is not None: EPOCHS=args.epochs
      if args.batch    is not None: BATCH=args.batch
      if args.lr       is not None: LR=args.lr
      if args.dropout  is not None: DROPOUT=args.dropout
      if args.patience is not None: PATIENCE=args.patience
      train()
