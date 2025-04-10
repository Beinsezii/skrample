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
from skrample.sampling import DPM, EPSILON, FLOW, IPNDM, VELOCITY, Euler, UniPC
from skrample.scheduling import Beta, Exponential, FlowShift, Karras, Linear, Scaled


def check_wrapper(wrapper: SkrampleWrapperScheduler, scheduler: ConfigMixin, params: list[str] = []) -> None:
    a, b = wrapper, SkrampleWrapperScheduler.from_diffusers_config(scheduler)
    a.fake_config = b.fake_config
    assert a == b, " | ".join([type(scheduler).__name__] + [str(p) for p in params])


def test_dpm() -> None:
    for flag, mod in [
        ("lower_order_final", None),  # dummy flag always true
        ("use_karras_sigmas", Karras),
        ("use_exponential_sigmas", Exponential),
        ("use_beta_sigmas", Beta),
    ]:
        for algo, noise in [
            ("dpmsolver", False),
            ("dpmsolver++", False),
            ("sde-dpmsolver", True),
            ("sde-dpmsolver++", True),
        ]:
            for uniform, spacing in [(False, "leading"), (True, "trailing")]:
                for skpred, dfpred in [(EPSILON, "epsilon"), (VELOCITY, "v_prediction")]:
                    for order in range(1, 4):
                        check_wrapper(
                            SkrampleWrapperScheduler(
                                DPM(predictor=skpred, add_noise=noise, order=order),
                                mod(Scaled(uniform=uniform)) if mod else Scaled(uniform=uniform),
                            ),
                            DPMSolverMultistepScheduler.from_config(
                                SCALED_CONFIG
                                | {
                                    "prediction_type": dfpred,
                                    "solver_order": order,
                                    "timestep_spacing": spacing,
                                    "algorithm_type": algo,
                                    "final_sigmas_type": "sigma_min",  # for non ++ to not err
                                    flag: True,
                                }
                            ),
                            [flag, algo, spacing, dfpred, f"o{order}"],
                        )

    check_wrapper(
        SkrampleWrapperScheduler(DPM(predictor=FLOW, order=2), FlowShift(Linear())),
        DPMSolverMultistepScheduler.from_config(FLOW_CONFIG),
    )


def test_euler() -> None:
    check_wrapper(
        SkrampleWrapperScheduler(Euler(), Scaled(uniform=False)),
        EulerDiscreteScheduler.from_config(SCALED_CONFIG),
    )
    check_wrapper(
        SkrampleWrapperScheduler(Euler(add_noise=True), Scaled(uniform=False)),
        EulerAncestralDiscreteScheduler.from_config(SCALED_CONFIG),
    )
    check_wrapper(
        SkrampleWrapperScheduler(Euler(predictor=FLOW), FlowShift(Linear())),
        FlowMatchEulerDiscreteScheduler.from_config(FLOW_CONFIG),
    )
    check_wrapper(
        SkrampleWrapperScheduler(Euler(predictor=FLOW), Beta(FlowShift(Linear()))),
        FlowMatchEulerDiscreteScheduler.from_config(FLOW_CONFIG | {"use_beta_sigmas": True}),
    )


def test_ipndm() -> None:
    check_wrapper(
        SkrampleWrapperScheduler(IPNDM(), Scaled(uniform=False)),
        IPNDMScheduler.from_config(SCALED_CONFIG),
    )


def test_unipc() -> None:
    check_wrapper(
        SkrampleWrapperScheduler(UniPC(order=2), Scaled(uniform=False)),
        UniPCMultistepScheduler.from_config(SCALED_CONFIG),
    )
    check_wrapper(
        SkrampleWrapperScheduler(UniPC(predictor=FLOW, order=2), FlowShift(Linear())),
        UniPCMultistepScheduler.from_config(FLOW_CONFIG),
    )


def test_alias() -> None:
    check_wrapper(
        SkrampleWrapperScheduler(DPM(add_noise=True), Scaled(uniform=False)),
        DPMSolverSDEScheduler.from_config(SCALED_CONFIG),
    )
    check_wrapper(
        SkrampleWrapperScheduler(Euler(), Scaled(uniform=False)),
        DDIMScheduler.from_config(SCALED_CONFIG),
    )
    check_wrapper(
        SkrampleWrapperScheduler(Euler(add_noise=True), Scaled(uniform=False)),
        DDPMScheduler.from_config(SCALED_CONFIG),
    )
