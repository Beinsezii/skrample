## 0.2.0
### Breaking
Rename `ScheduleCommon.num_train_timesteps` -> `ScheduleCommon.base_timesteps`

Change `diffusers.parse_diffusers_config` to take config as dict instead of **config

Moved customization properties from all pytorch.noise classes into their own property structs,
guarded by static type analysis. Should allow easily configuring the rng while guaranteeing state drop.

Brownian was previously incorectly stepping in reverse. This was remedied, with reverse being made a new prop

...

## 0.1.1
Remove `TensorNoiseCommon.device` field.

No major version as it was already a broken field.

## 0.1.0
Initial release with core features
