## 0.2.0
### Breaking
Rename `ScheduleCommon.num_train_timesteps` -> `ScheduleCommon.base_timesteps`

Change `diffusers.parse_diffusers_config` to take config as dict instead of **config

Moved customization properties from all pytorch.noise classes into their own property structs,
guarded by static type analysis. Should allow easily configuring the rng while guaranteeing state drop.

### Additions
Linear() can now adjust presented `.subnormal` property

Added diffusers class -> sampler map. Passing `sampler` to diffuser config parsing is now optional.

More diffuserse config mappings

### Fixes
Linear() and derivatives now respect sigma_start and sigma_end properly

Brownian now takes steps in proper order. Old behavior available via `.reverse` prop

## 0.1.1
Remove `TensorNoiseCommon.device` field.

No major version as it was already a broken field.

## 0.1.0
Initial release with core features
