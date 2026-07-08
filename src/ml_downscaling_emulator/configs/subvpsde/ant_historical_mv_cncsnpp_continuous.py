# Lint as: python3
"""Training NCSN++ on ANT domain multivariate data (pr + tas) with sub-VP SDE."""
import ml_collections
from ml_downscaling_emulator.configs.subvpsde.cordex_ml_mv_cncsnpp_continuous_defaults import get_config as get_default_configs


def get_config():
  config = get_default_configs()

  # training
  training = config.training
  training.n_epochs = 2000

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
  data.static_variables = ()
  # Target grid: 664x664 (11km ANT), predictor grid: 60x75 (100km ANT)
  data.image_size = 664
  data.predictor_image_size = 60
  # target_pr uses default sqrturrecen; override target_tas with standardise+recentre
  data.target_transform_overrides = ml_collections.ConfigDict()
  data.target_transform_overrides.target_tas = "mm;recen"

  # model: attention at coarsest resolution (664 / 2^3 = 83)
  model = config.model
  model.attn_resolutions = (83,)

  return config
