## 0.2.2
# Fixes
Incorrect comparison for modifier merge strategy

## 0.2.1
# Fixes
SigmmoidCDF peak value < self.sigma_start

## 0.2.0
### Breaking
Rename `ScheduleCommon.num_train_timesteps` -> `ScheduleCommon.base_timesteps`

Change `diffusers.parse_diffusers_config` to take config as dict instead of **config

Moved customization properties from all pytorch.noise classes into their own property structs,
guarded by static type analysis. Should allow easily configuring the rng while guaranteeing state drop.

Replace Flow() schedule with FlowShift() schedule modifier.

Updated `parse_diffusers_config` and `from_diffusers_config` to properly handle multiple modifiers

Removed Linear.sigma_end

### Additions
Linear() can now adjust presented `.subnormal` property

Added diffusers class -> sampler map. Passing `sampler` to diffuser config parsing is now optional.

More diffuserse config mappings

SigmaCDF() schedule

`scripts/plot_schedules.py` for visualising noise schedules

### Fixes
Linear() and derivatives now respect sigma_start properly

Brownian now takes steps in proper order. Old behavior available via `.reverse` prop

## 0.1.1
Remove `TensorNoiseCommon.device` field.

No major version as it was already a broken field.

## 0.1.0
Initial release with core features
