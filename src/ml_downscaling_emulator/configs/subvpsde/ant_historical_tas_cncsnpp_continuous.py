# Lint as: python3
"""Training NCSN++ on ANT domain temperature only with sub-VP SDE."""
import ml_collections
from ml_downscaling_emulator.configs.subvpsde.ant_historical_mv_cncsnpp_continuous import get_config as get_default_configs


def get_config():
  config = get_default_configs()

  # training
  # GH200 HBM is larger than the A100s the batch_size=4 mv config was tuned
  # for, and there's one fewer target channel here, so try a bigger batch;
  # re-tune down if this OOMs.
  config.training.batch_size = 8

  # data
  data = config.data
  data.target_variables = ("target_tas",)
  data.target_transform_key = "mm;recen"
  # single target now, so drop the inherited pr+tas override dict
  data.target_transform_overrides = ml_collections.ConfigDict()

  return config
