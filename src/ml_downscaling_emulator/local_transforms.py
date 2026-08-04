"""Local target-transform fixes not available in the installed mlde_utils package.

`mlde_utils.transforms.ClipT` is a no-op on the forward `transform()` call (it
only clips on `invert()`), since it assumes precipitation data is already
non-negative. In practice, regridded `target_pr` data contains a small
fraction (~0.9%) of tiny (~1e-9) negative values from interpolation noise.
Feeding these into a sqrt/root transform (`x ** 0.5`) produces NaNs, which
poison the training loss.

`ClipToZeroT` actually clips negative values to zero on the forward
transform, so it can be placed *before* a sqrt/root transform in a composed
target transform key, e.g. "clip0;sqrt;ur;recen" instead of "sqrturrecen".
"""

from mlde_utils.transforms import register_transform


@register_transform(name="clip0")
class ClipToZeroT:
    def __init__(self, variables):
        self.variables = variables

    def fit(self, target_ds, model_src_ds=None):
        return self

    def transform(self, ds):
        for var in self.variables:
            ds[var] = ds[var].clip(min=0.0)
        return ds

    def invert(self, ds):
        for var in self.variables:
            ds[var] = ds[var].clip(min=0.0)
        return ds
