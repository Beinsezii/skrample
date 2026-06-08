import PIL.Image
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from skrample.pytorch.noise import Colored

with torch.inference_mode():
    device = torch.device("cuda")
    dtype = torch.float32

    size: int = 1024
    exponents: list[float] = [-(2**3), -(2**2), -(2**1), 0, 2**-1, 2**0, 2**0.5]

    size = round(size / 8) * 8

    aekl: AutoencoderKL = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="vae",
    ).to(device=device, dtype=dtype)  # type: ignore # ???

    batches = torch.randn(len(exponents), 4, size // 8, size // 8, device=device, dtype=dtype)

    colors = [[Colored.colorize_noise(t, e) for t in batches] for e in exponents]

    canvas = PIL.Image.new("RGB", (size * len(exponents), size * len(exponents)))

    for y, batches in enumerate(colors):
        for x, latent in enumerate(batches):
            decoded = aekl.decode(latent.unsqueeze(0) / aekl.config["scaling_factor"]).sample[0]  # pyright: ignore [reportAttributeAccessIssue]
            im = PIL.Image.fromarray(
                ((decoded + 1) * (255 / 2)).clamp(0, 255).permute(1, 2, 0).to(device="cpu", dtype=torch.uint8).numpy()
            )

            canvas.paste(im, (x * size, y * size))

    canvas.save("colored_noise_grid.png")
