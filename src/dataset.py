"""Local (image, dense caption) dataset loader.

Each caption is a ``.txt`` file whose stem matches an image file name.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset


# Default sub-paths under ``data_dir`` for this dataset's layout.
_DEFAULT_IMAGES_SUBDIR = "vehicle_image_dataset/indian_vehicle_images/indian_vehicle_images"
_DEFAULT_CAPTIONS_SUBDIR = "vehicle_image_dataset/generated_captions"
_DEFAULT_IMAGE_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png")

class VehicleCaptionDataset(Dataset):
    """
    PyTorch dataset over local (image, dense-caption) pairs.

    Parameters
    ----------
    data_dir : str
        Root directory containing ``vehicle_image_dataset/``.
    images_subdir : str
        Path to the image folder, relative to ``data_dir``.
    captions_subdir : str
        Path to the caption-``.txt`` folder, relative to ``data_dir``.
    max_samples : Optional[int]
        If given, keep only the first ``max_samples`` *valid* samples
        (after the caption-length filter). ``None`` keeps all.
    min_caption_tokens : int
        Skip captions with fewer than this many whitespace tokens.
    image_exts : Sequence[str]
        Image extensions to try when resolving an image for a caption stem.
    """

    def __init__(
        self,
        data_dir: str = "./inputs",
        images_subdir: str = _DEFAULT_IMAGES_SUBDIR,
        captions_subdir: str = _DEFAULT_CAPTIONS_SUBDIR,
        max_samples: Optional[int] = None,
        min_caption_tokens: int = 30,
        image_exts: Sequence[str] = _DEFAULT_IMAGE_EXTS,
    ):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / images_subdir
        self.captions_dir = self.data_dir / captions_subdir
        self.min_caption_tokens = int(min_caption_tokens)
        self.image_exts = tuple(e.lower() for e in image_exts)

        if not self.captions_dir.is_dir():
            raise FileNotFoundError(f"Caption directory not found: {self.captions_dir}")
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.images_dir}")

        print(
            f"[VehicleCaptionDataset] images_dir='{self.images_dir}'\n"
            f"[VehicleCaptionDataset] captions_dir='{self.captions_dir}'"
        )

        self.samples: List[Dict[str, Any]] = self._build_index(max_samples)

        if not self.samples:
            raise RuntimeError(
                "No usable samples found. Check directory paths and "
                f"min_caption_tokens={self.min_caption_tokens}."
            )

        print(
            f"[VehicleCaptionDataset] Loaded {len(self.samples)} usable samples "
            f"(min_caption_tokens={self.min_caption_tokens})."
        )

    # ------------------------------------------------------------------ #
    # Index construction
    # ------------------------------------------------------------------ #
    def _build_index(self, max_samples: Optional[int]) -> List[Dict[str, Any]]:
        """Pair each caption ``.txt`` with its image by stem, then filter."""
        caption_paths = sorted(self.captions_dir.glob("*.txt"))
        if not caption_paths:
            raise RuntimeError(f"No .txt caption files in {self.captions_dir}")

        samples: List[Dict[str, Any]] = []
        n_missing_image = 0
        n_empty = 0
        n_too_short = 0

        for cap_path in caption_paths:
            stem = cap_path.stem

            image_path = self._resolve_image(stem)
            if image_path is None:
                n_missing_image += 1
                continue

            caption = cap_path.read_text(encoding="utf-8", errors="replace").strip()
            if not caption:
                n_empty += 1
                continue

            token_count = len(caption.split())
            if token_count < self.min_caption_tokens:
                n_too_short += 1
                continue

            samples.append(
                {
                    "index": len(samples),
                    "image_path": image_path,
                    "caption": caption,
                    "image_id": stem,
                    "token_count": token_count,
                }
            )

            if max_samples is not None and len(samples) >= max_samples:
                break

        skipped = n_missing_image + n_empty + n_too_short
        if skipped:
            print(
                f"[VehicleCaptionDataset] Skipped {skipped} caption(s): "
                f"{n_missing_image} missing image, {n_empty} empty, "
                f"{n_too_short} below min_caption_tokens."
            )
        return samples

    def _resolve_image(self, stem: str) -> Optional[Path]:
        """Return the image path matching ``stem``, trying each extension."""
        for ext in self.image_exts:
            candidate = self.images_dir / f"{stem}{ext}"
            if candidate.is_file():
                return candidate
        return None

    # ------------------------------------------------------------------ #
    # Image loading
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_rgb(image_path: Path) -> Image.Image:
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    # ------------------------------------------------------------------ #
    # Dataset protocol
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        return {
            "image": self._load_rgb(sample["image_path"]),
            "caption": sample["caption"],
            "image_id": sample["image_id"],
            "index": sample["index"],
            "token_count": sample["token_count"],
        }

    # ------------------------------------------------------------------ #
    # Convenience constructor from the project YAML config
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "VehicleCaptionDataset":
        """
        Build a dataset from the parsed YAML config.

        Reads ``config['dataset']`` for ``max_samples`` / ``min_caption_tokens``
        and an optional ``config['dataset']['local']`` block for path overrides:

            dataset:
              max_samples: null
              min_caption_tokens: 30
              local:
                data_dir: "./inputs"
                images_subdir: "..."
                captions_subdir: "..."
        """
        ds_cfg = config.get("dataset", {})
        local = ds_cfg.get("local", {})
        return cls(
            data_dir=local.get("data_dir", "./inputs"),
            images_subdir=local.get("images_subdir", _DEFAULT_IMAGES_SUBDIR),
            captions_subdir=local.get("captions_subdir", _DEFAULT_CAPTIONS_SUBDIR),
            max_samples=ds_cfg.get("max_samples"),
            min_caption_tokens=ds_cfg.get("min_caption_tokens", 30),
        )


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """DataLoader collate: PIL images cannot be stacked, so return parallel lists."""
    return {
        "images": [item["image"] for item in batch],
        "captions": [item["caption"] for item in batch],
        "image_ids": [item["image_id"] for item in batch],
        "indices": [item["index"] for item in batch],
        "token_counts": [item["token_count"] for item in batch],
    }
