import itertools

import pytest
from diffusers.configuration_utils import ConfigMixin
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.schedulers.scheduling_dpmsolver_sde import DPMSolverSDEScheduler
from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_ipndm import IPNDMScheduler
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from testing_common import FLOW_CONFIG, SCALED_CONFIG

from skrample.diffusers import SkrampleWrapperScheduler
from skrample.sampling.models import DiffusionModel, FlowModel, NoiseModel, VelocityModel
from skrample.sampling.structured import DPM, Adams, Euler, UniPC
from skrample.scheduling import Beta, Exponential, FlowShift, Karras, Linear, Scaled, ScheduleModifier, SubSchedule

EPSILON = NoiseModel()
FLOW = FlowModel()
VELOCITY = VelocityModel()


def assert_wrapper(wrapper: SkrampleWrapperScheduler, scheduler: ConfigMixin) -> None:
    a, b = wrapper, SkrampleWrapperScheduler.from_diffusers_config(scheduler)
    a.fake_config = b.fake_config
    assert a.sampler == b.sampler  # individual asserts for complex structs first for easier debugging
    assert a.schedule == b.schedule
    assert a == b


@pytest.mark.parametrize(
    (
        "modifiers",
        "stochasticity",
        "model_transform",
        "order",
    ),
    itertools.product(
        [
            ("lower_order_final", None),  # dummy flag always true
            ("use_karras_sigmas", Karras),
            ("use_exponential_sigmas", Exponential),
            ("use_beta_sigmas", Beta),
        ],
        [("dpmsolver", False), ("dpmsolver++", False), ("sde-dpmsolver", True), ("sde-dpmsolver++", True)],
        [("epsilon", EPSILON), ("v_prediction", VELOCITY)],
        range(1, 4),
    ),
)
def test_dpm(
    modifiers: tuple[str, type[SubSchedule] | None],
    stochasticity: tuple[str, bool],
    model_transform: tuple[str, DiffusionModel],
    order: int,
) -> None:
    flag, mod = modifiers
    algo, noise = stochasticity
    dfpred, skpred = model_transform
    assert_wrapper(
        SkrampleWrapperScheduler(
            DPM(stochasticity=noise, order=order),
            mod(Scaled()) if mod else Scaled(),
            skpred,
        ),
        DPMSolverMultistepScheduler.from_config(
            SCALED_CONFIG
            | {
                "prediction_type": dfpred,
                "solver_order": order,
                "algorithm_type": algo,
                "final_sigmas_type": "sigma_min",  # for non ++ to not err
                flag: True,
            }
        ),
    )


def test_dpm_flow() -> None:
    assert_wrapper(
        SkrampleWrapperScheduler(DPM(order=2), FlowShift(Linear()), FLOW),
        DPMSolverMultistepScheduler.from_config(FLOW_CONFIG),
    )


def test_euler() -> None:
    assert_wrapper(
        SkrampleWrapperScheduler(Euler(), Scaled()),
        EulerDiscreteScheduler.from_config(SCALED_CONFIG),
    )


def test_euler_a() -> None:
    assert_wrapper(
        SkrampleWrapperScheduler(DPM(order=1, stochasticity=True), Scaled()),
        EulerAncestralDiscreteScheduler.from_config(SCALED_CONFIG),
    )


def test_euler_flow() -> None:
    assert_wrapper(
        SkrampleWrapperScheduler(Euler(), FlowShift(Linear()), FLOW),
        FlowMatchEulerDiscreteScheduler.from_config(FLOW_CONFIG),
    )


def test_ipndm() -> None:
    assert_wrapper(
        SkrampleWrapperScheduler(Adams(order=4), Scaled()),
        IPNDMScheduler.from_config(SCALED_CONFIG),
    )


def test_unipc() -> None:
    assert_wrapper(
        SkrampleWrapperScheduler(UniPC(order=2), Scaled()),
        UniPCMultistepScheduler.from_config(SCALED_CONFIG),
    )


def test_unipc_flow() -> None:
    assert_wrapper(
        SkrampleWrapperScheduler(UniPC(order=2), FlowShift(Linear()), FLOW),
        UniPCMultistepScheduler.from_config(FLOW_CONFIG),
    )


def test_dpmsde() -> None:
    assert_wrapper(
        SkrampleWrapperScheduler(DPM(order=1, stochasticity=True), Scaled()),
        DPMSolverSDEScheduler.from_config(SCALED_CONFIG),
    )


def test_ddim() -> None:
    assert_wrapper(
        SkrampleWrapperScheduler(Euler(), Scaled()),
        DDIMScheduler.from_config(SCALED_CONFIG),
    )


def test_ddpm() -> None:
    assert_wrapper(
        SkrampleWrapperScheduler(DPM(order=1, stochasticity=True), Scaled()),
        DDPMScheduler.from_config(SCALED_CONFIG),
    )


@pytest.mark.parametrize(
    ("karras", "exp", "beta", "subschedule"),
    [
        # https://github.com/huggingface/diffusers/blob/2d0110f8182d18834d5039b19232e5761023b5f6/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py#L441-L468
        (True, True, True, Karras),
        (False, True, True, Exponential),
        (True, False, True, Karras),
        (True, True, False, Karras),
        (True, False, False, Karras),
        (False, True, False, Exponential),
        (False, False, True, Beta),
        (False, False, False, None),
    ],
)
def test_subschedule_mro_vp(karras: bool, exp: bool, beta: bool, subschedule: type[SubSchedule] | None) -> None:
    scheduler: DPMSolverMultistepScheduler = DPMSolverMultistepScheduler.from_config(SCALED_CONFIG)
    scheduler._internal_dict = dict(  # bypass validation checks
        scheduler.config
        | {
            "use_karras_sigmas": karras,
            "use_exponential_sigmas": exp,
            "use_beta_sigmas": beta,
            "use_flow_sigmas": False,
            "flow_shift": 3,
        }
    )
    assert_wrapper(
        SkrampleWrapperScheduler(DPM(), Scaled() if subschedule is None else subschedule(Scaled())),
        scheduler,
    )


@pytest.mark.parametrize(
    ("karras", "exp", "beta", "subschedule"),
    [
        # Different than VP due to manual override in parse_diffusers_config
        (True, True, True, FlowShift),
        (False, True, True, FlowShift),
        (True, False, True, FlowShift),
        (True, True, False, FlowShift),
        (True, False, False, FlowShift),
        (False, True, False, FlowShift),
        (False, False, True, Beta),
        (False, False, False, FlowShift),
    ],
)
def test_subschedule_mro_fm(
    karras: bool,
    exp: bool,
    beta: bool,
    subschedule: type[SubSchedule | ScheduleModifier],
) -> None:
    scheduler: DPMSolverMultistepScheduler = DPMSolverMultistepScheduler.from_config(FLOW_CONFIG)
    scheduler._internal_dict = dict(  # bypass validation checks
        scheduler.config
        | {
            "use_karras_sigmas": karras,
            "use_exponential_sigmas": exp,
            "use_beta_sigmas": beta,
            "use_flow_sigmas": True,
            "flow_shift": 3,
        }
    )
    assert_wrapper(SkrampleWrapperScheduler(DPM(), subschedule(Linear()), FlowModel()), scheduler)
