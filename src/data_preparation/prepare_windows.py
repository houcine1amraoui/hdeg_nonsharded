from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.utils.get_folders_utils import get_processed_folder

SPLITS = (
    "train",
    "val",
    "actor2_test",
    "actor1_test",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct supervised sliding-window pairs from the "
            "normalized preprocessing outputs."
        )
    )

    parser.add_argument(
        "--processed_dir",
        type=str,
        required=True,
        help=(
            "Directory containing arrays.npz, timestamps.npz, "
            "and devices.json produced by main_preprocess.py."
        ),
    )

    parser.add_argument(
        "--window_size",
        type=int,
        required=True,
        help="Observation window length W.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory in which the windowed split files are saved. "
            "Defaults to <processed_dir>/windows."
        ),
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help=(
            "Splits to process. By default all available splits "
            "are processed."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing window files.",
    )

    return parser.parse_args()


def load_npz_array(
    archive: np.lib.npyio.NpzFile,
    key: str,
    archive_name: str,
) -> np.ndarray:
    if key not in archive.files:
        raise KeyError(
            f"Split '{key}' was not found in {archive_name}. "
            f"Available entries: {archive.files}"
        )

    return archive[key]


def validate_devices(
    processed_dir: Path,
    num_devices: int,
) -> Optional[list]:
    devices_path = processed_dir / "devices.json"

    if not devices_path.is_file():
        print(
            "[WARNING] devices.json was not found. "
            "Device-count validation will be skipped."
        )
        return None

    with devices_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        devices = json.load(file)

    if not isinstance(devices, list):
        raise ValueError(
            "devices.json must contain a JSON list."
        )

    if len(devices) != num_devices:
        raise ValueError(
            "Device-count mismatch: "
            f"devices.json contains {len(devices)} devices, "
            f"but the data contains {num_devices} columns."
        )

    return devices


def validate_data_array(
    data: np.ndarray,
    split: str,
) -> None:
    if data.ndim != 2:
        raise ValueError(
            f"Split '{split}' must have shape (T, N), "
            f"but received {data.shape}."
        )

    num_timesteps, num_devices = data.shape

    if num_timesteps <= 0:
        raise ValueError(
            f"Split '{split}' contains no observations."
        )

    if num_devices <= 0:
        raise ValueError(
            f"Split '{split}' contains no devices."
        )

    if not np.issubdtype(data.dtype, np.floating):
        raise TypeError(
            f"Split '{split}' must contain floating-point values. "
            f"Received dtype {data.dtype}."
        )

    if not np.isfinite(data).all():
        raise ValueError(
            f"Split '{split}' contains NaN or infinite values."
        )


def validate_timestamps(
    timestamps: np.ndarray,
    data: np.ndarray,
    split: str,
) -> None:
    if timestamps.ndim != 1:
        raise ValueError(
            f"Timestamps for split '{split}' must be one-dimensional. "
            f"Received shape {timestamps.shape}."
        )

    if len(timestamps) != len(data):
        raise ValueError(
            f"Timestamp/data length mismatch for split '{split}': "
            f"{len(timestamps)} timestamps vs "
            f"{len(data)} observations."
        )


def construct_window_pairs(
    data: np.ndarray,
    timestamps: np.ndarray,
    window_size: int,
) -> Dict[str, np.ndarray]:
    """
    Construct supervised temporal window/target pairs.

    Given:

        data.shape = (T, N)

    produce:

        X.shape = (T-W, W, N)
        y.shape = (T-W, W, N)

    where:

        X[i] = data[i : i + W]
        y[i] = data[i + 1 : i + W + 1]

    Timestamp alignment:

        window_start_timestamps[i] = timestamps[i]
        target_timestamps[i]       = timestamps[i + 1 : i + W + 1]
    """

    num_timesteps, num_devices = data.shape

    num_samples = num_timesteps - window_size

    if num_samples <= 0:
        raise ValueError(
            "The split does not contain enough observations for "
            f"window_size={window_size}. "
            f"Received T={num_timesteps}."
        )

    # -------------------------------------------------------------
    # Construct the observation windows.
    #
    # Shape:
    #
    #   (T-W, W, N)
    #
    # The returned array is explicitly copied so that the saved
    # artifact owns contiguous data rather than depending on a
    # stride-based view.
    # -------------------------------------------------------------
    X = np.lib.stride_tricks.sliding_window_view(
        data,
        window_shape=window_size,
        axis=0,
    )

    # sliding_window_view with axis=0 produces:
    #
    #   (T-W+1, N, W)
    #
    # for this two-dimensional input.
    #
    # We need:
    #
    #   (T-W+1, W, N)
    #
    # and then remove the final window because that window has no
    # following target.
    X = np.transpose(
        X,
        (0, 2, 1),
    )

    # Consecutive views form the current and one-step-shifted windows.
    y = X[1:num_samples + 1].copy()
    X = X[:num_samples].copy()

    # -------------------------------------------------------------
    # Target:
    #
    # y[i] = data[i + 1 : i + W + 1]
    #
    # Shape:
    #
    #   (T-W, W, N)
    # -------------------------------------------------------------
    # y was copied from the shifted window views above.

    # -------------------------------------------------------------
    # Timestamp alignment
    # -------------------------------------------------------------
    window_start_timestamps = timestamps[
        :num_samples
    ].copy()

    target_timestamps = np.lib.stride_tricks.sliding_window_view(
        timestamps,
        window_shape=window_size,
    )[1:num_samples + 1].copy()

    # -------------------------------------------------------------
    # Final shape assertions
    # -------------------------------------------------------------
    expected_x_shape = (
        num_samples,
        window_size,
        num_devices,
    )

    expected_y_shape = (
        num_samples,
        window_size,
        num_devices,
    )

    if X.shape != expected_x_shape:
        raise RuntimeError(
            "Unexpected window tensor shape. "
            f"Expected {expected_x_shape}, "
            f"received {X.shape}."
        )

    if y.shape != expected_y_shape:
        raise RuntimeError(
            "Unexpected target tensor shape. "
            f"Expected {expected_y_shape}, "
            f"received {y.shape}."
        )

    if len(window_start_timestamps) != num_samples:
        raise RuntimeError(
            "Unexpected number of window-start timestamps."
        )

    if target_timestamps.shape != (num_samples, window_size):
        raise RuntimeError(
            "Unexpected target timestamp shape; expected (T-W, W)."
        )

    return {
        "X": X,
        "y": y,
        "window_start_timestamps": (
            window_start_timestamps
        ),
        "target_timestamps": target_timestamps,
    }


def verify_window_alignment(
    data: np.ndarray,
    windows: Dict[str, np.ndarray],
    window_size: int,
) -> None:
    """
    Verify the semantic relationship between each window and target.

    This is intentionally performed on the generated artifact so that
    an indexing error cannot silently propagate into DBRL training.
    """

    X = windows["X"]
    y = windows["y"]

    num_samples = X.shape[0]

    if num_samples == 0:
        raise RuntimeError(
            "Cannot verify an empty window dataset."
        )

    # Verify the first sample.
    if not np.array_equal(
        X[0],
        data[:window_size],
    ):
        raise RuntimeError(
            "First observation window is incorrectly aligned."
        )

    if not np.array_equal(
        y[0],
        data[1:window_size + 1],
    ):
        raise RuntimeError(
            "First target is incorrectly aligned."
        )

    # Verify the last sample.
    last_start = num_samples - 1

    if not np.array_equal(
        X[last_start],
        data[
            last_start:
            last_start + window_size
        ],
    ):
        raise RuntimeError(
            "Last observation window is incorrectly aligned."
        )

    if not np.array_equal(
        y[last_start],
        data[
            last_start + 1:
            last_start + window_size + 1
        ],
    ):
        raise RuntimeError(
            "Last target is incorrectly aligned."
        )

    # Verify a middle sample when possible.
    if num_samples >= 3:
        middle = num_samples // 2

        if not np.array_equal(
            X[middle],
            data[
                middle:
                middle + window_size
            ],
        ):
            raise RuntimeError(
                "Middle observation window is incorrectly aligned."
            )

        if not np.array_equal(
            y[middle],
            data[
                middle + 1:
                middle + window_size + 1
            ],
        ):
            raise RuntimeError(
                "Middle target is incorrectly aligned."
            )


def save_window_split(
    output_path: Path,
    split: str,
    windows: Dict[str, np.ndarray],
    window_size: int,
    num_devices: int,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists:\n{output_path}\n\n"
            "Use --overwrite if you want to replace it."
        )

    np.savez_compressed(
        output_path,
        X=windows["X"],
        y=windows["y"],
        window_start_timestamps=(
            windows["window_start_timestamps"]
        ),
        target_timestamps=(
            windows["target_timestamps"]
        ),
        window_size=np.asarray(
            window_size,
            dtype=np.int64,
        ),
        num_devices=np.asarray(
            num_devices,
            dtype=np.int64,
        ),
    )


def process_split(
    split: str,
    arrays: np.lib.npyio.NpzFile,
    timestamps_archive: np.lib.npyio.NpzFile,
    output_dir: Path,
    window_size: int,
    overwrite: bool,
) -> None:
    print()
    print("=" * 70)
    print(f"Processing split: {split}")
    print("=" * 70)

    data = load_npz_array(
        arrays,
        split,
        "arrays.npz",
    )

    timestamps = load_npz_array(
        timestamps_archive,
        split,
        "timestamps.npz",
    )

    validate_data_array(
        data,
        split,
    )

    validate_timestamps(
        timestamps,
        data,
        split,
    )

    num_timesteps, num_devices = data.shape

    print(
        f"Input shape          : {data.shape}"
    )
    print(
        f"Input dtype          : {data.dtype}"
    )
    print(
        f"Number of timestamps : {len(timestamps)}"
    )

    windows = construct_window_pairs(
        data=data,
        timestamps=timestamps,
        window_size=window_size,
    )

    verify_window_alignment(
        data=data,
        windows=windows,
        window_size=window_size,
    )

    output_path = Path(
         f"{output_dir}/{split}.npz"
    )

    save_window_split(
        output_path=output_path,
        split=split,
        windows=windows,
        window_size=window_size,
        num_devices=num_devices,
        overwrite=overwrite,
    )

    print()
    print("Generated window dataset")
    print(
        f"  X shape              : "
        f"{windows['X'].shape}"
    )
    print(
        f"  y shape              : "
        f"{windows['y'].shape}"
    )
    print(
        f"  window timestamps    : "
        f"{windows['window_start_timestamps'].shape}"
    )
    print(
        f"  target timestamps    : "
        f"{windows['target_timestamps'].shape}"
    )
    print(
        f"  output               : "
        f"{output_path}"
    )

    # Additional numerical verification.
    if not np.isfinite(
        windows["X"]
    ).all():
        raise RuntimeError(
            f"Generated X for split '{split}' "
            "contains NaN or infinite values."
        )

    if not np.isfinite(
        windows["y"]
    ).all():
        raise RuntimeError(
            f"Generated y for split '{split}' "
            "contains NaN or infinite values."
        )

    print(
        "[PASS] Window construction and alignment verified."
    )


def main_prepare_windows(config) -> None:
    processed_data_folder = get_processed_folder(config)
    output_dir = f"{processed_data_folder}/windows"
    arrays = np.load(f"{processed_data_folder}/arrays.npz")
    timestamps_archive = np.load(f"{processed_data_folder}/timestamps.npz")
    window_size = config["preprocessing"]["window_size"]
    # ---------------------------------------------------------
    # Construct each requested split.
    # ---------------------------------------------------------

    splits = ["train", "val", "actor2_test", "actor1_test"]

    for split in splits:
        process_split(
            split=split,
            arrays=arrays,
            timestamps_archive=timestamps_archive,
            output_dir=output_dir,
            window_size=window_size,
            overwrite=True,
        )

    print()
    print("=" * 70)
    print("Window preparation completed successfully.")
    print("=" * 70)
    print(
        f"Window files saved under:\n{processed_data_folder}"
    )


if __name__ == "__main__":
    main_prepare_windows()