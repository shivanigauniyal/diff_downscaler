# Lint as: python3
"""Training NCSN++ on ANT domain multivariate data (pr + tas) with sub-VP SDE."""
import ml_collections
from ml_downscaling_emulator.configs.subvpsde.cordex_ml_mv_cncsnpp_continuous_defaults import get_config as get_default_configs


def get_config():
  config = get_default_configs()

  # training
  training = config.training
  training.n_epochs = 2000
  # Single-GPU config (confirmed working, job 43708040: ~13.6 samples/s).
  # 4-GPU nn.DataParallel (batch_size=16) was tried but hit repeated GPU-fault/
  # hang issues (jobs 43714182, 43904412) - reverted to single-GPU until a
  # proper DDP refactor is done.
  training.batch_size = 4
  # bf16 autocast on the forward/loss pass to cut memory use (attn_resolutions=(83,)
  # runs full self-attention on 664x664 target - see CUDA OOM debugging notes)
  training.amp = True
  # Patch-based training: random-crop both target and (upsampled) predictor to
  # this size each step instead of training on the full 664x664 domain. Cuts
  # per-step activation memory (lets batch_size go back up) and speeds up
  # training; at 11km/pixel a 256px patch still spans ~2800km, larger than
  # typical synoptic-scale systems, so this should retain most useful context.
  # Set to 0 to disable and train on the full image_size again.
  training.random_crop_size = 256

  # data
  data = config.data
  data.dataset_name = 'ANT_historical/ant_historical'
  data.predictor_variables = (
      "hus250", "hus500", "hus850",
      "ta250",  "ta500",  "ta850",
      "ua250",  "ua500",  "ua850",
      "va250",  "va500",  "va850",
      "zg250",  "zg500",  "zg850",
  )
  data.target_variables = ("target_pr", "target_tas")
  # orography (0 over ocean, ~2300m mean over the ice sheet) so the model has
  # a direct high-res signal to distinguish ocean from land/ice
  data.static_variables = ("surface_altitude",)
  # Target grid: 664x664 (11km ANT), predictor grid: 60x75 (100km ANT)
  data.image_size = 664
  data.predictor_image_size = 60
  # clip target_pr to 0 before sqrt (regridding leaves tiny negative noise
  # that produces NaN under the default sqrturrecen transform's raw sqrt);
  # override target_tas with standardise+recentre
  data.target_transform_overrides = ml_collections.ConfigDict()
  data.target_transform_overrides.target_pr = "clip0;sqrt;ur;recen"
  data.target_transform_overrides.target_tas = "mm;recen"

  # model: attention at coarsest resolution (664 / 2^3 = 83)
  model = config.model
  model.attn_resolutions = (83,)
  # Use memory-efficient attention (torch scaled_dot_product_attention) instead
  # of the legacy explicit einsum+softmax attention (AttnBlockpp), which OOMs
  # at this resolution because it materializes the full 83x83 x 83x83
  # attention matrix. Set back to 'ddpm' to restore the legacy block if needed.
  model.attention_type = 'efficient'

  return config
