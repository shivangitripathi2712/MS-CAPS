from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from PIL import Image

from diffusers import (
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
)

try:
    from diffusers import FluxPipeline
    FLUX_AVAILABLE = True
except Exception:
    FluxPipeline = None
    FLUX_AVAILABLE = False

try:
    from diffusers import FluxImg2ImgPipeline
    FLUX_IMG2IMG_AVAILABLE = True
except Exception:
    FluxImg2ImgPipeline = None
    FLUX_IMG2IMG_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Device / dtype / backend helpers
# --------------------------------------------------------------------------- #
def resolve_device(device: str = "auto") -> str:
    """Resolve a device string safely (auto -> cuda:0 or cpu)."""
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[T2I] CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        if ":" in device:
            gpu_id = int(device.split(":")[-1])
            if gpu_id >= torch.cuda.device_count():
                print(
                    f"[T2I] Requested {device}, but only "
                    f"{torch.cuda.device_count()} CUDA device(s) found. "
                    "Falling back to cuda:0."
                )
                return "cuda:0"
    return device


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    """Resolve torch dtype safely (cpu -> float32 regardless of request)."""
    if device == "cpu":
        return torch.float32
    dtype = dtype.lower()
    if dtype in ("float16", "fp16", "half"):
        return torch.float16
    if dtype in ("bfloat16", "bf16"):
        return torch.bfloat16
    return torch.float32


def infer_backend(model_name: str, backend: str = "auto") -> str:
    """Decide whether to use SDXL or FLUX based on model name."""
    backend = backend.lower().strip()
    if backend in {"sdxl", "flux"}:
        return backend
    model_lower = model_name.lower()
    if "flux" in model_lower or "black-forest-labs" in model_lower:
        return "flux"
    return "sdxl"


def _round_to_multiple(value: int, multiple: int = 16) -> int:
    """FLUX requires H/W divisible by 16 (latent packing). Round down."""
    rounded = (value // multiple) * multiple
    return max(multiple, rounded)


# --------------------------------------------------------------------------- #
# Reconstructor
# --------------------------------------------------------------------------- #
class T2IReconstructor:
    """
    Text-to-image reconstructor (decoder G) for the KG-RD framework.

    Parameters
    ----------
    model_name : str
        HF model name or local path.
    backend : str
        "auto", "sdxl", or "flux".
    device : str
        "auto", "cpu", "cuda:0", "cuda:1", ...
    dtype : str
        "float16", "bfloat16", or "float32". FLUX.1-dev expects bfloat16.
    num_inference_steps : int
        Denoising steps. Must match between baseline and candidates.
    guidance_scale : float
        FLUX.1-dev: ~3.5 (distilled guidance). SDXL: ~7.5.
    image_size : int
        Square output side. Rounded down to a multiple of 16 for FLUX.
    negative_prompt : Optional[str]
        SDXL only. FLUX.1-dev uses distilled guidance and ignores it.
    max_sequence_length : int
        FLUX T5 token budget. 512 routes long captions through T5 instead of
        truncating at CLIP's 77. Captions longer than this are truncated and
        will trigger a warning (see ``check_prompt_length``).
    img2img_mode : bool
        ABLATION ONLY. Enables FluxImg2ImgPipeline. RD-invalidating — see
        module docstring. Off by default; forced off for non-FLUX backends or
        if FluxImg2ImgPipeline is unavailable.
    img2img_strength : float
        Init-image preservation in [0, 1]; only used when img2img_mode=True.
    """

    def __init__(
        self,
        model_name: str = "black-forest-labs/FLUX.1-dev",
        backend: str = "auto",
        device: str = "auto",
        dtype: str = "bfloat16",
        num_inference_steps: int = 20,
        guidance_scale: float = 3.5,
        image_size: int = 512,
        negative_prompt: Optional[str] = None,
        max_sequence_length: int = 512,
        img2img_mode: bool = False,
        img2img_strength: float = 0.7,
    ):
        self.model_name = model_name
        self.backend = infer_backend(model_name=model_name, backend=backend)
        self.device = resolve_device(device)
        self.torch_dtype = resolve_dtype(dtype, self.device)

        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.negative_prompt = negative_prompt
        self.max_sequence_length = int(max_sequence_length)

        # FLUX requires H/W divisible by 16.
        requested = int(image_size)
        self.image_size = _round_to_multiple(requested, 16)
        if self.image_size != requested:
            print(
                f"[T2I] image_size {requested} not divisible by 16; "
                f"using {self.image_size}."
            )

        # ---- img2img guardrails (ablation only) ----
        self.img2img_mode = bool(img2img_mode)
        self.img2img_strength = float(img2img_strength)
        if not (0.0 <= self.img2img_strength <= 1.0):
            raise ValueError(
                f"img2img_strength must be in [0, 1], got {self.img2img_strength}"
            )
        if self.img2img_mode and self.backend != "flux":
            print(f"[T2I] img2img requested but backend='{self.backend}'. Disabling.")
            self.img2img_mode = False
        if self.img2img_mode and not FLUX_IMG2IMG_AVAILABLE:
            print("[T2I] FluxImg2ImgPipeline unavailable (pip install -U diffusers). "
                  "Disabling img2img.")
            self.img2img_mode = False
        if self.img2img_mode:
            print("[T2I] *** WARNING: img2img mode is RD-INVALIDATING. ***")
            print("[T2I] *** Distortion no longer isolates caption information. ***")
            print(f"[T2I] *** Ablation use only (strength={self.img2img_strength}). ***")

        print(
            f"[T2I] backend='{self.backend}', model='{model_name}', "
            f"device='{self.device}', dtype='{self.torch_dtype}', "
            f"steps={self.num_inference_steps}, guidance={self.guidance_scale}, "
            f"size={self.image_size}, mode={'img2img' if self.img2img_mode else 'text-only'}"
        )

        if self.backend == "flux":
            self._load_flux()
        elif self.backend == "sdxl":
            self._load_sdxl()
        else:
            raise ValueError(f"Unsupported T2I backend: {self.backend}")

        # Quiet per-image progress bars for clean batch logs.
        try:
            self.pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

        print("[T2I] Model loaded successfully.")

    # ----------------------------------------------------------------- #
    # Loaders
    # ----------------------------------------------------------------- #
    def _load_sdxl(self) -> None:
        pipe_kwargs = {"torch_dtype": self.torch_dtype, "use_safetensors": True}
        if self.torch_dtype == torch.float16 and self.device.startswith("cuda"):
            pipe_kwargs["variant"] = "fp16"
        self.pipe = StableDiffusionXLPipeline.from_pretrained(self.model_name, **pipe_kwargs)
        self.pipe.to(self.device)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_attention_slicing()
        self._enable_vae_memory_savers()

    def _load_flux(self) -> None:
        if not FLUX_AVAILABLE:
            raise ImportError(
                "FluxPipeline unavailable. Run: "
                "pip install -U diffusers transformers accelerate sentencepiece protobuf"
            )
        cls = FluxImg2ImgPipeline if self.img2img_mode else FluxPipeline
        self.pipe = cls.from_pretrained(self.model_name, torch_dtype=self.torch_dtype)
        # A6000 (48GB) fits bf16 FLUX (~33GB) fully — keep on-device, no CPU
        # offload (offload trades large speed loss for memory we don't need).
        self.pipe.to(self.device)
        self._enable_vae_memory_savers()

    def _enable_vae_memory_savers(self) -> None:
        """VAE slicing + tiling: cheap, helps avoid VAE OOM at higher res."""
        if not self.device.startswith("cuda"):
            return
        for fn in ("enable_vae_slicing", "enable_vae_tiling"):
            try:
                getattr(self.pipe, fn)()
            except Exception:
                pass

    # ----------------------------------------------------------------- #
    # Prompt handling
    # ----------------------------------------------------------------- #
    def build_reconstruction_prompt(self, caption: str) -> str:
        """
        Use the caption verbatim as the prompt. No instruction prefix: a prefix
        would consume T5 budget and bias generation away from caption content.
        """
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError("[T2I] Caption must be a non-empty string.")
        return caption.strip()

    def check_prompt_length(self, caption: str) -> Tuple[int, bool]:
        """
        Return (t5_token_count, truncated). Warns if the caption exceeds the
        FLUX T5 budget, because silent truncation drops caption information and
        biases D_0 / D(c'_k). FLUX only; returns (-1, False) for SDXL or if no
        T5 tokenizer is present.
        """
        tok = getattr(self.pipe, "tokenizer_2", None)  # T5 tokenizer on FLUX
        if tok is None:
            return -1, False
        ids = tok(caption, truncation=False, return_tensors=None)["input_ids"]
        n = len(ids)
        truncated = n > self.max_sequence_length
        if truncated:
            print(
                f"[T2I] WARNING: caption is {n} T5 tokens > "
                f"max_sequence_length={self.max_sequence_length}; it will be "
                "TRUNCATED. Distortion for this caption is computed on a "
                "shortened prompt."
            )
        return n, truncated

    def _prepare_init_image(self, init_image: Image.Image) -> Image.Image:
        if not isinstance(init_image, Image.Image):
            raise TypeError(f"[T2I] init_image must be a PIL.Image, got {type(init_image)}")
        init_image = init_image.convert("RGB")
        if init_image.size != (self.image_size, self.image_size):
            init_image = init_image.resize(
                (self.image_size, self.image_size), resample=Image.BICUBIC
            )
        return init_image

    # ----------------------------------------------------------------- #
    # Generation
    # ----------------------------------------------------------------- #
    @torch.no_grad()
    def generate(
        self,
        caption: str,
        seed: int = 42,
        output_path: Optional[Union[str, Path]] = None,
        init_image: Optional[Image.Image] = None,
    ) -> Image.Image:
        """Generate one reconstructed image from a caption (deterministic per seed)."""
        prompt = self.build_reconstruction_prompt(caption)

        if self.img2img_mode and init_image is None:
            raise ValueError("[T2I] img2img_mode enabled but no init_image provided.")

        gen_device = self.device if self.device.startswith("cuda") else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(int(seed))

        if self.backend == "flux":
            self.check_prompt_length(prompt)
            if self.img2img_mode:
                result = self.pipe(
                    prompt=prompt,
                    image=self._prepare_init_image(init_image),
                    strength=self.img2img_strength,
                    height=self.image_size,
                    width=self.image_size,
                    guidance_scale=self.guidance_scale,
                    num_inference_steps=self.num_inference_steps,
                    max_sequence_length=self.max_sequence_length,
                    generator=generator,
                )
            else:
                result = self.pipe(
                    prompt,
                    height=self.image_size,
                    width=self.image_size,
                    guidance_scale=self.guidance_scale,
                    num_inference_steps=self.num_inference_steps,
                    max_sequence_length=self.max_sequence_length,
                    generator=generator,
                )
        else:  # SDXL, text-only
            result = self.pipe(
                prompt=prompt,
                negative_prompt=self.negative_prompt,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                height=self.image_size,
                width=self.image_size,
                generator=generator,
            )

        image = result.images[0].convert("RGB")
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
        return image

    @torch.no_grad()
    def generate_multi_seed(
        self,
        caption: str,
        seeds: Optional[List[int]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        prefix: str = "recon",
        init_image: Optional[Image.Image] = None,
    ) -> List[Image.Image]:
        """Generate one image per seed from the same caption."""
        seeds = seeds or [42]
        out_dir = Path(output_dir) if output_dir is not None else None
        images: List[Image.Image] = []
        for seed in seeds:
            output_path = (out_dir / f"{prefix}_seed{seed}.png") if out_dir else None
            images.append(
                self.generate(
                    caption=caption, seed=seed,
                    output_path=output_path, init_image=init_image,
                )
            )
        return images

    def reconstruct(
        self,
        caption: str,
        seeds: Optional[List[int]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        prefix: str = "recon",
        init_image: Optional[Image.Image] = None,
    ) -> List[Image.Image]:
        """Main entry point used by the pipeline. Returns one image per seed."""
        return self.generate_multi_seed(
            caption=caption, seeds=seeds, output_dir=output_dir,
            prefix=prefix, init_image=init_image,
        )

    def clear_memory(self) -> None:
        """Clear CUDA cache after large reconstruction batches."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----------------------------------------------------------------- #
    # Config constructor
    # ----------------------------------------------------------------- #
    @classmethod
    def from_config(cls, config: dict) -> "T2IReconstructor":
        """Build from the project YAML ``t2i`` block."""
        t = config.get("t2i", {})
        return cls(
            model_name=t.get("model", "black-forest-labs/FLUX.1-dev"),
            backend=t.get("backend", "auto"),
            device=t.get("device", "auto"),
            dtype=t.get("dtype", "bfloat16"),
            num_inference_steps=t.get("num_inference_steps", 20),
            guidance_scale=t.get("guidance_scale", 3.5),
            image_size=t.get("image_size", 512),
            max_sequence_length=t.get("max_sequence_length", 512),
            img2img_mode=t.get("img2img_mode", False),
            img2img_strength=t.get("img2img_strength", 0.7),
        )

# =========================================================================== #
# Structure-conditioned reconstruction (FLUX ControlNet)  --  OPTION B
# =========================================================================== #
#
# Why this is RD-VALID (unlike img2img):
#   The structure map (canny edges OR depth) is extracted ONCE from the ORIGINAL
#   image I, via set_structure_reference(I), and the SAME map conditions the
#   baseline G(c) and every candidate G(c'_k). Structure is therefore a fixed
#   side-channel identical across all captions -- it advantages no caption. The
#   caption must still supply all CONTENT (colors, vehicle types, visible text,
#   attributes, described orientation), so Q_img continues to measure caption
#   informativeness. Composition/layout no longer drifts between reconstructions.
#
#   Contrast with img2img, which feeds the original's PIXELS (incl. content) into
#   every reconstruction and so inflates Q_img regardless of caption quality.
#
# Usage (handled by the pipeline):
#   recon = StructureConditionedReconstructor(...)
#   recon.set_structure_reference(original_image)   # once per sample
#   recon.reconstruct(caption=c,  ...)              # baseline, uses cached map
#   recon.reconstruct(caption=ck, ...)              # candidates, same map

try:
    from diffusers import FluxControlNetModel, FluxControlNetPipeline
    FLUX_CONTROLNET_AVAILABLE = True
except Exception:
    FluxControlNetModel = None
    FluxControlNetPipeline = None
    FLUX_CONTROLNET_AVAILABLE = False

# FLUX.1-dev ControlNet-Union-Pro control_mode codes.
_UNION_CONTROL_MODES = {"canny": 0, "tile": 1, "depth": 2, "blur": 3, "pose": 4, "gray": 5}

DEFAULT_CONTROLNET_MODEL = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro"
DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


class StructureConditionedReconstructor(T2IReconstructor):
    """
    FLUX ControlNet reconstructor. Shares structure (edges/depth) from the
    original image across baseline and all candidates so layout is preserved
    while the caption still drives content. Same public API as T2IReconstructor
    plus ``set_structure_reference(image)``.
    """

    def __init__(
        self,
        control_type: str = "canny",                # "canny" | "depth"
        controlnet_model: str = DEFAULT_CONTROLNET_MODEL,
        controlnet_conditioning_scale: float = 0.6,
        control_mode: Optional[int] = None,         # None -> auto for Union models
        depth_model: str = DEFAULT_DEPTH_MODEL,
        canny_low: int = 100,
        canny_high: int = 200,
        **kwargs,
    ):
        self.control_type = str(control_type).lower().strip()
        if self.control_type not in ("canny", "depth"):
            raise ValueError(f"control_type must be 'canny' or 'depth', got {control_type!r}")
        self.controlnet_model = controlnet_model
        self.controlnet_conditioning_scale = float(controlnet_conditioning_scale)
        self.depth_model = depth_model
        self.canny_low = int(canny_low)
        self.canny_high = int(canny_high)
        self._control_image: Optional[Image.Image] = None
        self._depth_pipe = None

        # Auto-pick control_mode for Union models (dedicated models ignore it).
        if control_mode is None and "union" in str(controlnet_model).lower():
            control_mode = _UNION_CONTROL_MODES.get(self.control_type)
        self.control_mode = control_mode

        kwargs["img2img_mode"] = False  # structure conditioning is not img2img
        super().__init__(**kwargs)

    # ----- override the FLUX loader to build a ControlNet pipeline -----
    def _load_flux(self) -> None:
        if not FLUX_CONTROLNET_AVAILABLE:
            raise ImportError(
                "FluxControlNetPipeline unavailable. Run: pip install -U diffusers"
            )
        print(f"[T2I-Struct] Loading ControlNet ({self.control_type}): {self.controlnet_model}")
        controlnet = FluxControlNetModel.from_pretrained(
            self.controlnet_model, torch_dtype=self.torch_dtype)
        self.pipe = FluxControlNetPipeline.from_pretrained(
            self.model_name, controlnet=controlnet, torch_dtype=self.torch_dtype)
        self.pipe.to(self.device)
        self._enable_vae_memory_savers()
        print(f"[T2I-Struct] ControlNet pipeline ready "
              f"(scale={self.controlnet_conditioning_scale}, mode={self.control_mode}).")

    # ----- structure map extraction (once per sample) -----
    def set_structure_reference(self, image: Image.Image) -> None:
        """Extract and cache the control map from the ORIGINAL image I."""
        self._control_image = self._make_control_image(image)

    def _make_control_image(self, image: Image.Image) -> Image.Image:
        img = image.convert("RGB").resize((self.image_size, self.image_size))
        if self.control_type == "canny":
            try:
                import cv2
            except Exception as exc:
                raise ImportError(
                    "control_type='canny' needs opencv: pip install opencv-python-headless"
                ) from exc
            arr = np.array(img)
            edges = cv2.Canny(arr, self.canny_low, self.canny_high)
            edges = np.stack([edges, edges, edges], axis=-1)
            return Image.fromarray(edges)
        # depth
        if self._depth_pipe is None:
            from transformers import pipeline as hf_pipeline
            dev = _gpu_index_safe(self.device)
            self._depth_pipe = hf_pipeline("depth-estimation", model=self.depth_model, device=dev)
        depth = self._depth_pipe(img)["depth"]
        return depth.convert("RGB").resize((self.image_size, self.image_size))

    # ----- override single-image generation to feed the control map -----
    @torch.no_grad()
    def generate(
        self,
        caption: str,
        seed: int = 42,
        output_path: Optional[Union[str, Path]] = None,
        init_image: Optional[Image.Image] = None,   # ignored; structure is cached
    ) -> Image.Image:
        if self._control_image is None:
            raise RuntimeError(
                "[T2I-Struct] set_structure_reference(original_image) must be called "
                "before reconstruct()/generate()."
            )
        prompt = self.build_reconstruction_prompt(caption)
        self.check_prompt_length(prompt)
        gen_device = self.device if self.device.startswith("cuda") else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(int(seed))

        kw = dict(
            prompt=prompt,
            control_image=self._control_image,
            controlnet_conditioning_scale=self.controlnet_conditioning_scale,
            height=self.image_size, width=self.image_size,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            max_sequence_length=self.max_sequence_length,
            generator=generator,
        )
        if self.control_mode is not None:
            kw["control_mode"] = int(self.control_mode)

        image = self.pipe(**kw).images[0].convert("RGB")
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
        return image

    @classmethod
    def from_config(cls, config: dict) -> "StructureConditionedReconstructor":
        t = config.get("t2i", {})
        sc = t.get("structure_conditioning", {}) or {}
        return cls(
            control_type=sc.get("control_type", "canny"),
            controlnet_model=sc.get("controlnet_model", DEFAULT_CONTROLNET_MODEL),
            controlnet_conditioning_scale=sc.get("controlnet_conditioning_scale", 0.6),
            control_mode=sc.get("control_mode", None),
            depth_model=sc.get("depth_model", DEFAULT_DEPTH_MODEL),
            canny_low=sc.get("canny_low", 100),
            canny_high=sc.get("canny_high", 200),
            model_name=t.get("model", "black-forest-labs/FLUX.1-dev"),
            backend=t.get("backend", "auto"),
            device=t.get("device", "auto"),
            dtype=t.get("dtype", "bfloat16"),
            num_inference_steps=t.get("num_inference_steps", 20),
            guidance_scale=t.get("guidance_scale", 3.5),
            image_size=t.get("image_size", 512),
            max_sequence_length=t.get("max_sequence_length", 512),
        )


def _gpu_index_safe(device: str) -> int:
    """HF pipeline device index: GPU id for cuda, -1 for cpu."""
    if isinstance(device, str) and device.startswith("cuda"):
        return int(device.split(":")[-1]) if ":" in device else 0
    return -1


# --------------------------------------------------------------------------- #
# Factory: pick text-only vs structure-conditioned from config
# --------------------------------------------------------------------------- #
def build_reconstructor(config: dict):
    """
    Return the reconstructor selected by config.

    Structure conditioning is used when EITHER:
        t2i.mode == "controlnet"
        t2i.structure_conditioning.enabled == true
    Otherwise the canonical text-only T2IReconstructor is returned.
    """
    t = config.get("t2i", {})
    sc = t.get("structure_conditioning", {}) or {}
    use_struct = (str(t.get("mode", "text_only")).lower() == "controlnet"
                  or bool(sc.get("enabled", False)))
    if use_struct:
        print("[T2I] Mode: STRUCTURE-CONDITIONED (FLUX ControlNet).")
        return StructureConditionedReconstructor.from_config(config)
    print("[T2I] Mode: TEXT-ONLY (canonical).")
    return T2IReconstructor.from_config(config)

