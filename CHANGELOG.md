## 0.4.2
### Additions
Unlocked Adams order

Algorithm performance improvements

### Fixes
Fix `MergeStrategy` possibly not fully deduplicating lists

## 0.4.1
### Fixes
Add `SkrampleWrapperScheduler.set_begin_index` for diffusers 0.34 compat

## 0.4.0
### Additions
Hyper schedule modifier

`SkrampleWrapperSchedule.allow_dynamic`

## 0.3.0
### Breaking
Made SkrampleSchedule and SkrampleSampler frozen and therefore hashable

Refactored IPNDM -> Adams
  - Old defaults are equivalent to `Adams(order=4)`
  - Alrogithm updated for correctness, seeds may change

Refactored `subnormal: bool` -> `sigma_transform: SigmaTransform`
  - Exists in same places as `subnormal` was, including being provided by the schedules.
  - Besides the signature, usage is the same

Removed SkrampleSampler.predictor
  - Prediction functions now in `skrample.common`
  - SkrampleWrapperScheduler now has separate `predictor` field
  - SkrampleSampler.sample signature changed `output: T` -> `prediction: T`

Changed `HighOrderSampler.min_order/.max_order` into static methods rather than properties.

`HighOrderSampler.effective_order` now always uses `1` as minimum.

Removed FlowShift.mu

### Additions
ScheduleModifier has multiple new helper functions for working with frozen

scheduling.schedule_lru() for cached schedule retrieval.
Overhead of SkrampleWrapperSchedule.step() is reduced by 95% when using a highly complex schedule class

SPC sampler - simple predictor/corrector using midpoints

Updated `scripts/plot_schedules.py` -> `scripts/plot_skrample.py` for plotting samplers as well

Support for `return_dict=True`

`UniP` standalone sampler

## 0.2.3
### Fixes
Wrapper seed generation failing on non-contiguous inputs

## 0.2.2
### Fixes
Incorrect comparison for modifier merge strategy

## 0.2.1
### Fixes
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
