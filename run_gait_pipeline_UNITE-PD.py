"""

UNITE-PD pipeline for APDM Opal5 .h5 files.

Processing logic:
- Require Tel Aviv JSON files containing detected IC/TC events.
- Use Tel Aviv IC/TC events as the source of gait events.
- Estimate min_vel boundaries only for gaitmap trajectory compatibility.
- Apply fixed APDM mounting rotations, gravity alignment, and optional yaw orientation correction.
- Reconstruct stride-level trajectories and compute gaitmap spatial parameters.
- Compute gaitmap temporal parameters internally only for outlier exclusion.
- Apply external-style stride-time and arc-length outlier exclusion.
- Compute lumbar/shank turning and freezing metrics.
- Save compact per-trial and cohort-level Excel outputs.

Author - Nicholas D'Cruz

"""
from __future__ import annotations

import math
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import h5py
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

from scipy.spatial.transform import Rotation as R
from scipy.signal import welch, resample_poly, butter, sosfiltfilt
from math import gcd

from mobgap.turning import TdElGohary
from mobgap.data_transform import ButterworthFilter
from mobgap.utils.conversions import to_body_frame

from gaitmap.preprocessing import sensor_alignment
from gaitmap.preprocessing.sensor_alignment import PcaAlignment
from gaitmap.utils.rotations import flip_dataset, rotation_from_angle, rotate_dataset
from gaitmap.stride_segmentation import BarthDtw
from gaitmap.utils.coordinate_conversion import convert_to_fbf
from gaitmap.event_detection import HerzerEventDetection
from gaitmap.trajectory_reconstruction import (
    StrideLevelTrajectory,
    MadgwickRtsKalman,
    ForwardBackwardIntegration)

from gaitmap.parameters import TemporalParameterCalculation, SpatialParameterCalculation
from gaitmap.zupt_detection import StrideEventZuptDetector
from gaitmap.utils.consts import SF_ACC, SF_GYR
from gaitmap.utils.datatype_helper import SensorData, is_sensor_data
from types import SimpleNamespace

import warnings
import json
SUPPRESS_GAITMAP_WARNINGS = True

if SUPPRESS_GAITMAP_WARNINGS:
    warnings.filterwarnings("ignore", module="gaitmap.*")
    warnings.filterwarnings("ignore", module="gaitmap_mad.*")

from pandas.errors import PerformanceWarning
warnings.filterwarnings("ignore", category=PerformanceWarning)

# =============================================================================
# Settings
# =============================================================================

DIRECTORY_PATH = Path(r"C:\Users\u0111219\Documents\Elsa\rawdata")
OUTPUT_FOLDER_NAME = "processed_gait_UNITEPD"

SAMPLING_RATE_HZ = 128
EXTERNAL_EVENT_JSON_SUFFIX = ".json"
H5_FILE_SUFFIXES = ("uncued.h5", "external.h5", "internal.h5")

# Trial window used as the supervised walking bout.
TRIAL_START_S = 3
TRIAL_END_S = 123

#TURNING DETECTION
TURN_SMOOTHING_CUTOFF_HZ = 1.0
TURN_MIN_PEAK_ANGLE_VELOCITY_DPS = 15.0
TURN_LOWER_THRESHOLD_VELOCITY_DPS = 10.0
TURN_MIN_GAP_BETWEEN_TURNS_S = 0.05
TURN_ALLOWED_TURN_DURATION_S = (0.5, 6.0)
TURN_ALLOWED_TURN_ANGLE_DEG = (45.0, np.inf)

# Spatial reconstruction settings
TRAJ_USE_MAGNETOMETER = False
TRAJ_ZUPT_HALF_REGION_SIZE_S = 0.05
TRAJ_MADGWICK_BETA = 0.105
TRAJ_VELOCITY_ERROR_VARIANCE = 0.01

# External-event-to-gaitmap conversion settings
EXTERNAL_EVENT_MATCH_ANCHOR = "ic"

# Min velocity estimation
EXTERNAL_MIN_VEL_MIN_GAP_S = 0.35
EXTERNAL_MIN_VEL_BEFORE_TC_GAP_S = 0.03
EXTERNAL_MIN_VEL_AFTER_IC_GAP_S = 0.05
EXTERNAL_TERMINAL_MIN_VEL_SEARCH_AFTER_IC_S = 0.90
EXTERNAL_FORCE_KEEP_ROWS_WITH_FALLBACK_MINVEL = True
EXTERNAL_MIN_VEL_SEARCH_WIN_SIZE_MS = 100.0

EXCLUDE_SIMPLE_OUTLIER_STRIDES = True
# Match external MATLAB clean-CV logic:
# isoutlier(StrideTimeVector, 'mean', 'thresholdfactor', 2)
OUTLIER_USE_EXTERNAL_MEAN_SD_RULE = True
OUTLIER_STRIDE_TIME_SD_FACTOR = 2.0
OUTLIER_STRIDE_TIME_SD_SCOPE = "combined"  # external script pools stride times

# Optional additional exclusions
OUTLIER_EXCLUDE_STRIDE_TIME_GT_MAX = True
OUTLIER_STRIDE_TIME_MAX_S = 2.0
OUTLIER_EXCLUDE_ARC_LENGTH_GT_MAX = True
OUTLIER_ARC_LENGTH_MAX_M = 3.0

# Automatic sensor orientation selection
AUTO_SELECT_ORIENTATION = True
SAVE_AUTO_ORIENTATION_QC = True
AUTO_ORIENTATION_MIN_STRIDES_PER_SIDE = 10
AUTO_ORIENTATION_SCORE_MARGIN_TO_SWITCH = 25.0

AUTO_ORIENTATION_CANDIDATES = [
    (0, 0),
    (180, 180),
    (90, 90),
    (-90, -90),
    (90, -90),
    (-90, 90),
    (180, 0),
    (0, 180),
    (90, 0),
    (-90, 0),
    (0, 90),
    (0, -90),
]

# Walking-only PCA fallback for automatic orientation selection
AUTO_TRY_PCA_ALIGNMENT_FALLBACK = True
AUTO_PCA_ALIGNMENT_MAX_SAMPLES_PER_SENSOR = 200_000
AUTO_PCA_ALIGNMENT_TARGET_AXIS = "y"
AUTO_PCA_ALIGNMENT_PLANE_AXIS = ("gyr_x", "gyr_y")

# Optional manual yaw overrides.
# Manual non-zero overrides take precedence over automatic orientation selection.
ORIENTATION_YAW_OVERRIDES = {
    # Example:
    # "sub-xxx_ses-xx_task-xxx": {
    #     "sensor_left": 180.0,
    #     "sensor_right": 0.0,
    # },
}


# =============================================================================
# 2. GENERAL HELPERS
# =============================================================================

def _decode_attr(value: Any) -> Any:
    """Decode h5py attributes that may be bytes, numpy bytes, or plain strings."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8")
    return value


def _safe_name(value: str) -> str:
    """Remove whitespace from APDM sensor labels, matching MATLAB char(label(~isspace(label)))."""
    return "".join(str(value).strip().split())


def _ensure_2d_signal_orientation(arr: np.ndarray, expected_channels: int) -> np.ndarray:
    """
    Return signal as channels x samples and ensure the result is writable.

    MATLAB h5read often gave arrays used as:
        Opal.Sensor.acc(1:3, :)

    h5py may return either channels x samples or samples x channels.
    This helper normalizes to channels x samples and returns a writable copy.
    """
    arr = np.array(arr, dtype=float, copy=True)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")

    if arr.shape[0] == expected_channels:
        return np.array(arr, dtype=float, copy=True)

    if arr.shape[1] == expected_channels:
        return np.array(arr.T, dtype=float, copy=True)

    raise ValueError(
        f"Could not infer orientation for array shape {arr.shape}; "
        f"expected one dimension to be {expected_channels}."
    )


def _nanmean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.all(np.isnan(x)):
        return np.nan
    return float(np.nanmean(x))


def _nanstd_matlab(x: np.ndarray) -> float:
    """
    MATLAB std(x) uses sample standard deviation with N-1 normalization.
    This matches std(..., 'omitnan') behavior for vectors.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size <= 1:
        return np.nan
    return float(np.std(x, ddof=1))


def _cv_matlab(x: np.ndarray, use_abs_mean: bool = False) -> float:
    mean_x = _nanmean(x)
    sd_x = _nanstd_matlab(x)
    if np.isnan(mean_x) or mean_x == 0:
        return np.nan
    denominator = abs(mean_x) if use_abs_mean else mean_x
    return float(sd_x / denominator * 100)


def _asym_matlab(left_mean: float, right_mean: float, use_abs_denominator: bool = False) -> float:
    """
    Match MATLAB code:
        abs(L_av - R_av) / mean(L_av + R_av) * 100

    Since L_av + R_av is scalar, mean(L_av + R_av) == L_av + R_av.
    For gm_angle_tc, MATLAB used abs(mean(L_av + R_av)).
    """
    denom = left_mean + right_mean
    if use_abs_denominator:
        denom = abs(denom)
    if denom == 0 or np.isnan(denom):
        return np.nan
    return float(abs(left_mean - right_mean) / denom * 100)


def _summarize_lr_exact(
    left: np.ndarray,
    right: np.ndarray,
    prefix: str,
    use_abs_cv: bool = False,
    use_abs_asym_denominator: bool = False,
) -> Dict[str, float]:
    """
    Create exact MATLAB-style variable names:
        prefix_L_av, prefix_L_sd, prefix_L_cv,
        prefix_R_av, ...
        prefix_C_av, prefix_C_sd, prefix_C_cv, prefix_C_as
    """
    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    combined = np.concatenate([left, right])

    left_av = _nanmean(left)
    left_sd = _nanstd_matlab(left)
    left_cv = _cv_matlab(left, use_abs_mean=use_abs_cv)

    right_av = _nanmean(right)
    right_sd = _nanstd_matlab(right)
    right_cv = _cv_matlab(right, use_abs_mean=use_abs_cv)

    combined_av = _nanmean(combined)
    combined_sd = _nanstd_matlab(combined)
    combined_cv = _cv_matlab(combined, use_abs_mean=use_abs_cv)

    combined_as = _asym_matlab(
        left_av,
        right_av,
        use_abs_denominator=use_abs_asym_denominator,
    )

    return {
        f"{prefix}_L_av": left_av,
        f"{prefix}_L_sd": left_sd,
        f"{prefix}_L_cv": left_cv,
        f"{prefix}_R_av": right_av,
        f"{prefix}_R_sd": right_sd,
        f"{prefix}_R_cv": right_cv,
        f"{prefix}_C_av": combined_av,
        f"{prefix}_C_sd": combined_sd,
        f"{prefix}_C_cv": combined_cv,
        f"{prefix}_C_as": combined_as,
    }


def _extract_existing_column(df: pd.DataFrame, possible_names: List[str]) -> np.ndarray:
    """Extract a column by trying several possible column names."""
    if df is None or df.empty:
        return np.array([], dtype=float)

    for name in possible_names:
        if name in df.columns:
            return df[name].to_numpy(dtype=float)

    return np.array([], dtype=float)


def _get_s_id_index_values(df: pd.DataFrame) -> pd.Index:
    """Return s_id values whether they are an index level or a simple index."""
    if isinstance(df.index, pd.MultiIndex) and "s_id" in df.index.names:
        return df.index.get_level_values("s_id")
    return df.index


def _filter_by_s_ids(df: pd.DataFrame, valid_s_ids: List[Any]) -> pd.DataFrame:
    """Filter DataFrame rows by s_id, supporting MultiIndex or regular index."""
    if df is None or df.empty or len(valid_s_ids) == 0:
        return df.iloc[0:0].copy()

    s_ids = _get_s_id_index_values(df)
    return df.loc[s_ids.isin(valid_s_ids)].copy()

def build_retained_event_list_for_trajectory(
    segments: List[Tuple[int, int]],
    event_list: Dict[str, pd.DataFrame],
    remove_boundary_strides: bool = True,
    number_boundary_strides: int = 1,
) -> Dict[str, pd.DataFrame]:
    """
    Build retained event lists for trajectory reconstruction using the same
    combined chronological boundary trimming as final segment filtering.

    This ensures trajectory reconstruction, spatial parameters, gaitmap
    summaries, and MATLAB-style summaries use the same boundary-trim logic.
    """
    retained_parts = {
        "sensor_left": [],
        "sensor_right": [],
    }

    for seg_start, seg_end in segments:
        (
            _valid_before,
            valid_after,
            _n_before,
            _n_after,
        ) = get_combined_boundary_trimmed_s_ids_for_segment(
            event_list=event_list,
            seg_start=seg_start,
            seg_end=seg_end,
            remove_boundary_strides=remove_boundary_strides,
            number_boundary_strides=number_boundary_strides,
        )

        for sensor in ["sensor_left", "sensor_right"]:
            retained_segment_events = _filter_by_s_ids(
                event_list[sensor],
                valid_after[sensor],
            )

            if retained_segment_events is not None and not retained_segment_events.empty:
                retained_parts[sensor].append(retained_segment_events)

    retained_event_list = {}

    for sensor in ["sensor_left", "sensor_right"]:
        if retained_parts[sensor]:
            retained_event_list[sensor] = pd.concat(
                retained_parts[sensor],
                axis=0,
            )
        else:
            retained_event_list[sensor] = event_list[sensor].iloc[0:0].copy()

    return retained_event_list

def get_retained_s_ids_from_segment_outputs(segment_outputs):
    retained = {
        "sensor_left": set(),
        "sensor_right": set(),
    }

    for bout in segment_outputs:
        for sensor in ["sensor_left", "sensor_right"]:
            events = bout["filtered_events"][sensor]

            if events is None or events.empty:
                continue

            if isinstance(events.index, pd.MultiIndex) and "s_id" in events.index.names:
                s_ids = events.index.get_level_values("s_id")
            else:
                s_ids = events.index

            retained[sensor].update(list(s_ids))

    return retained

def _cdiff(x: np.ndarray) -> np.ndarray:
    """
    Approximation of MATLAB cdiff used for jerk.

    Uses central difference for interior points and first-order edges.
    """
    x = np.asarray(x, dtype=float)

    if x.size < 2:
        return np.zeros_like(x)

    out = np.zeros_like(x)
    out[1:-1] = (x[2:] - x[:-2]) / 2
    out[0] = x[1] - x[0]
    out[-1] = x[-1] - x[-2]

    return out


def _resample_matlab_style(x: np.ndarray, new_fs: int, old_fs: int) -> np.ndarray:
    """
    Approximate MATLAB:
        resample(x, new_fs, old_fs)

    Uses scipy.signal.resample_poly with reduced integer ratio.
    """
    x = np.asarray(x, dtype=float)

    common = gcd(int(new_fs), int(old_fs))
    up = int(new_fs // common)
    down = int(old_fs // common)

    return resample_poly(x, up, down)


def _pwelch_matlab_style(
    x: np.ndarray,
    window_length: int,
    nfft: int,
    fs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate MATLAB:
        pwelch(x, window_length, [], nfft, fs)

    MATLAB uses a Hamming window when window is given as a scalar.
    SciPy default is Hann, so we explicitly use Hamming here.
    """
    x = np.asarray(x, dtype=float)

    f, pxx = welch(
        x,
        fs=fs,
        window="hamming",
        nperseg=window_length,
        noverlap=window_length // 2,
        nfft=nfft,
        detrend="constant",
        scaling="density",
    )

    return pxx, f


def _freq_idx(freqs: np.ndarray, target: float) -> int:
    """
    Return index closest to target frequency.

    MATLAB used find(F == target). With nfft=fs*10, resolution is 0.1 Hz,
    so 0.5, 3, 8, and 12 should exist exactly, but nearest index is safer.
    """
    freqs = np.asarray(freqs, dtype=float)
    return int(np.argmin(np.abs(freqs - target)))


def _safe_band_ratio(
    pxx_norm: np.ndarray,
    idx_num_start: int,
    idx_num_end: int,
    idx_den_start: int,
    idx_den_end: int,
) -> float:
    """
    Match MATLAB structure:
        sum(Pxx_N(ii:ii1)).^2 / sum(Pxx_N(ii2:ii)).^2

    MATLAB indexing is inclusive. Python slicing end is exclusive,
    so we use idx_end + 1.
    """
    numerator = np.sum(pxx_norm[idx_num_start:idx_num_end + 1]) ** 2
    denominator = np.sum(pxx_norm[idx_den_start:idx_den_end + 1]) ** 2

    if denominator == 0:
        return np.nan

    return float(numerator / denominator)


def _normalise_psd(pxx: np.ndarray) -> np.ndarray:
    """
    Match MATLAB:
        Pxx_N = Pxx ./ sum(Pxx)
    """
    pxx = np.asarray(pxx, dtype=float)
    total = np.sum(pxx)

    if total == 0:
        return np.full_like(pxx, np.nan, dtype=float)

    return pxx / total


def _count_large_positive_local_maxima(
    y: np.ndarray,
    smoothing_kernel_samples: int = 5,
    min_peak_height_deg_s: float = 180,
    min_distance_samples: int = 64,
) -> Tuple[int, float]:
    """
    Count large positive local maxima in gyr_y.

    Returns:
        n_positive_peaks
        median_positive_peak_height
    """
    y_sm = _moving_average(y, smoothing_kernel_samples)

    pos_idx, _ = _find_local_extrema(y_sm)

    if pos_idx.size == 0:
        return 0, np.nan

    pos_idx = pos_idx[y_sm[pos_idx] >= min_peak_height_deg_s]
    pos_idx = _suppress_nearby_peaks(pos_idx, min_distance_samples)

    if pos_idx.size == 0:
        return 0, np.nan

    return int(pos_idx.size), float(np.nanmedian(y_sm[pos_idx]))

def apply_extra_yaw_rotation(
    dataset: SensorData,
    left_deg: float = 0.0,
    right_deg: float = 0.0,
) -> SensorData:
    """
    Apply additional z-axis yaw rotations to left/right sensors.

    This is useful when the initial fixed sensor rotations are correct for gravity
    but AP/ML axes remain wrong.
    """
    dataset=make_sensor_data_writable(dataset)
    rotations = {
        "sensor_left": R.from_euler("z", left_deg, degrees=True),
        "sensor_right": R.from_euler("z", right_deg, degrees=True),
    }

    return rotate_dataset(dataset, rotations)

def apply_file_orientation_yaw_override(
    dataset: SensorData,
    base_name: str,
    overrides: Dict[str, Dict[str, float]],
) -> Tuple[SensorData, Dict[str, float]]:
    """
    Apply file-specific yaw override if one exists.

    Example:
        ORIENTATION_YAW_OVERRIDES = {
            "sub-bel030_ses-t0_task-external": {
                "sensor_left": 90,
                "sensor_right": -90,
            }
        }
    """
    if base_name not in overrides:
        return dataset, {}

    left_deg = overrides[base_name].get("sensor_left", 0.0)
    right_deg = overrides[base_name].get("sensor_right", 0.0)

    corrected = apply_extra_yaw_rotation(
        dataset,
        left_deg=left_deg,
        right_deg=right_deg,
    )

    return corrected, {
        "sensor_left": left_deg,
        "sensor_right": right_deg,
    }

def _get_retained_s_ids_from_event_list(event_list: Dict[str, pd.DataFrame]) -> Dict[str, set]:
    retained = {
        "sensor_left": set(),
        "sensor_right": set(),
    }

    for sensor in ["sensor_left", "sensor_right"]:
        if sensor not in event_list:
            continue

        df = event_list[sensor]

        if df is None or df.empty:
            continue

        if isinstance(df.index, pd.MultiIndex) and "s_id" in df.index.names:
            s_ids = df.index.get_level_values("s_id")
        else:
            s_ids = df.index

        retained[sensor].update(list(s_ids))

    return retained

def _trajectory_heading_normalized_stats(
    trajectory: StrideLevelTrajectory,
    retained_s_ids: Dict[str, set],
) -> Dict[str, float]:
    """
    Compute heading-normalized trajectory spread metrics.

    These are used only for candidate scoring.
    """
    stats = {}

    for sensor in ["sensor_left", "sensor_right"]:
        if sensor not in trajectory.position_:
            stats[f"{sensor}_n_traj"] = 0
            stats[f"{sensor}_lat_iqr"] = np.nan
            stats[f"{sensor}_lat_max_abs"] = np.nan
            stats[f"{sensor}_dx_median"] = np.nan
            continue

        pos = trajectory.position_[sensor]

        if not all(col in pos.columns for col in ["pos_x", "pos_y"]):
            stats[f"{sensor}_n_traj"] = 0
            stats[f"{sensor}_lat_iqr"] = np.nan
            stats[f"{sensor}_lat_max_abs"] = np.nan
            stats[f"{sensor}_dx_median"] = np.nan
            continue

        all_y = []
        dxs = []

        for s_id in retained_s_ids.get(sensor, []):
            if isinstance(pos.index, pd.MultiIndex) and "s_id" in pos.index.names:
                stride_pos = pos.loc[pos.index.get_level_values("s_id") == s_id]
            else:
                if s_id not in pos.index:
                    continue
                stride_pos = pos.loc[[s_id]]

            if stride_pos.empty or len(stride_pos) < 2:
                continue

            x = stride_pos["pos_x"].to_numpy(dtype=float)
            y = stride_pos["pos_y"].to_numpy(dtype=float)

            x = x - x[0]
            y = y - y[0]

            dx = x[-1]
            dy = y[-1]

            if not np.isfinite(dx) or not np.isfinite(dy):
                continue

            angle = np.arctan2(dy, dx)

            rot = np.array([
                [np.cos(-angle), -np.sin(-angle)],
                [np.sin(-angle),  np.cos(-angle)],
            ])

            xy_rot = rot @ np.vstack([x, y])

            all_y.extend(xy_rot[1, :].tolist())
            dxs.append(xy_rot[0, -1])

        all_y = np.asarray(all_y, dtype=float)
        dxs = np.asarray(dxs, dtype=float)

        stats[f"{sensor}_n_traj"] = len(dxs)

        if len(all_y) == 0:
            stats[f"{sensor}_lat_iqr"] = np.nan
            stats[f"{sensor}_lat_max_abs"] = np.nan
        else:
            stats[f"{sensor}_lat_iqr"] = float(
                np.nanpercentile(all_y, 75) - np.nanpercentile(all_y, 25)
            )
            stats[f"{sensor}_lat_max_abs"] = float(np.nanmax(np.abs(all_y)))

        stats[f"{sensor}_dx_median"] = float(np.nanmedian(dxs)) if len(dxs) else np.nan

    return stats

def _segment_mask_from_segments(n_samples: int, segments: List[Tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(n_samples, dtype=bool)

    for start, end in segments:
        start = int(max(0, start))
        end = int(min(n_samples - 1, end))

        if end >= start:
            mask[start:end + 1] = True

    return mask

def _count_large_positive_gyr_y_peaks_for_dataset(
    dataset: SensorData,
    segments: List[Tuple[int, int]],
    min_peak_height_deg_s: float = 180,
    min_distance_samples: int = 64,
) -> Dict[str, int]:
    """
    Count large positive local maxima in gyr_y for both sensors,
    restricted to straight-walking segments.
    """
    n_samples = len(dataset["sensor_left"])
    mask = _segment_mask_from_segments(n_samples, segments)

    counts = {}

    for sensor in ["sensor_left", "sensor_right"]:
        y = dataset[sensor]["gyr_y"].to_numpy(dtype=float)
        y_eval = y[mask]

        y_sm = _moving_average(y_eval, 5)

        pos_idx, _ = _find_local_extrema(y_sm)

        if pos_idx.size == 0:
            counts[sensor] = 0
            continue

        pos_idx = pos_idx[y_sm[pos_idx] >= min_peak_height_deg_s]
        pos_idx = _suppress_nearby_peaks(pos_idx, min_distance_samples)

        counts[sensor] = int(len(pos_idx))

    return counts

def evaluate_orientation_candidate(
    dataset_candidate: SensorData,
    segments: List[Tuple[int, int]],
    sampling_rate_hz: int,
) -> Dict[str, Any]:
    """
    Evaluate one orientation candidate without saving outputs.

    Higher score is better.
    """
    result = {
    "status": "failed",
    "score": -1e9,

    "n_left": np.nan,
    "n_right": np.nan,
    "n_total": np.nan,

    "stride_length_median": np.nan,
    "stride_length_cv": np.nan,

    "positive_peak_count_left": np.nan,
    "positive_peak_count_right": np.nan,

    "raw_y_abs_left": np.nan,
    "raw_y_abs_right": np.nan,
    "raw_y_sep_lr": np.nan,
    "raw_heading_diff_lr": np.nan,

    "lat_iqr_left": np.nan,
    "lat_iqr_right": np.nan,
    "lat_max_left": np.nan,
    "lat_max_right": np.nan,

    "error": "",
}

    try:
        bf_data = convert_to_fbf(
            dataset_candidate,
            left_like="sensor_left",
            right_like="sensor_right",
        )

        dtw = BarthDtw()
        dtw.max_cost = 4.5
        dtw.max_template_stretch_ms = 32
        dtw = dtw.segment(
            data=bf_data,
            sampling_rate_hz=sampling_rate_hz,
        )

        ed = HerzerEventDetection()
        ed = ed.detect(
            data=bf_data,
            stride_list=dtw.stride_list_,
            sampling_rate_hz=sampling_rate_hz,
        )

        retained_event_list = build_retained_event_list_for_trajectory(
            segments=segments,
            event_list=ed.min_vel_event_list_,
            remove_boundary_strides=False,
            number_boundary_strides=0,
        )


        n_left = len(retained_event_list["sensor_left"])
        n_right = len(retained_event_list["sensor_right"])
        n_total = n_left + n_right

        if n_left < 3 or n_right < 3:
            result.update({
                "status": "too_few_retained_strides",
                "n_left": n_left,
                "n_right": n_right,
                "n_total": n_total,
                "score": -10000 + n_total,
            })
            return result

        ori_method = MadgwickRtsKalman(
            use_magnetometer=TRAJ_USE_MAGNETOMETER,
            madgwick_beta=TRAJ_MADGWICK_BETA,
            velocity_error_variance=TRAJ_VELOCITY_ERROR_VARIANCE,
            zupt_detector=StrideEventZuptDetector(
                half_region_size_s=TRAJ_ZUPT_HALF_REGION_SIZE_S
            ),
        )

        pos_method = ForwardBackwardIntegration(
            gravity=[0, 0, 9.81],
            level_assumption=True,
        )

        trajectory = StrideLevelTrajectory(
            ori_method=ori_method,
            pos_method=pos_method,
        ).estimate(
            data=dataset_candidate,
            stride_event_list=retained_event_list,
            sampling_rate_hz=sampling_rate_hz,
        )

        spatial = SpatialParameterCalculation()
        spatial = spatial.calculate(
            stride_event_list=retained_event_list,
            positions=trajectory.position_,
            orientations=trajectory.orientation_,
            sampling_rate_hz=sampling_rate_hz,
        )

        retained_s_ids = _get_retained_s_ids_from_event_list(retained_event_list)

        traj_stats = _trajectory_heading_normalized_stats(
            trajectory=trajectory,
            retained_s_ids=retained_s_ids,
        )
        
        raw_traj_stats = _trajectory_raw_endpoint_stats(
            trajectory=trajectory,
            retained_s_ids=retained_s_ids,
        )
        positive_peak_counts = _count_large_positive_gyr_y_peaks_for_dataset(
            dataset=dataset_candidate,
            segments=segments,
            min_peak_height_deg_s=180,
            min_distance_samples=64,
        )

        # Extract stride lengths.
        sl_left = _extract_existing_column(
            spatial.parameters_["sensor_left"],
            ["stride_length", "stride length [m]"],
        )
        sl_right = _extract_existing_column(
            spatial.parameters_["sensor_right"],
            ["stride_length", "stride length [m]"],
        )

        sl_all = np.concatenate([sl_left, sl_right])

        stride_length_median = float(np.nanmedian(sl_all)) if len(sl_all) else np.nan
        stride_length_cv = _cv_matlab(sl_all) if len(sl_all) else np.nan

        lat_iqr_left = traj_stats.get("sensor_left_lat_iqr", np.nan)
        lat_iqr_right = traj_stats.get("sensor_right_lat_iqr", np.nan)
        lat_max_left = traj_stats.get("sensor_left_lat_max_abs", np.nan)
        lat_max_right = traj_stats.get("sensor_right_lat_max_abs", np.nan)
        
        raw_y_abs_left = raw_traj_stats.get("sensor_left_raw_endpoint_y_abs_median", np.nan)
        raw_y_abs_right = raw_traj_stats.get("sensor_right_raw_endpoint_y_abs_median", np.nan)
        raw_y_sep_lr = raw_traj_stats.get("raw_left_right_endpoint_y_separation", np.nan)
        
        raw_heading_diff_lr = raw_traj_stats.get("raw_left_right_heading_difference_deg", np.nan)
        raw_heading_iqr_left = raw_traj_stats.get("sensor_left_raw_heading_deg_iqr", np.nan)
        raw_heading_iqr_right = raw_traj_stats.get("sensor_right_raw_heading_deg_iqr", np.nan)

        pos_peak_total = (
            positive_peak_counts.get("sensor_left", 0)
            + positive_peak_counts.get("sensor_right", 0)
        )

        # -------------------------
        # Heuristic scoring
        # -------------------------
        score = 0.0

        # Reward enough retained strides.
        score += min(n_total, 80) * 5
        score += min(n_left, n_right) * 10

        # Penalize persistent positive gyr_y peaks.
        score -= pos_peak_total * 3

        # Penalize implausible stride length.
        if np.isfinite(stride_length_median):
            if stride_length_median < 0.5 or stride_length_median > 2.5:
                score -= 500
        else:
            score -= 500

        # Penalize high stride-length CV.
        if np.isfinite(stride_length_cv):
            score -= max(0, stride_length_cv - 15) * 10
        else:
            score -= 200

        # Penalize heading-normalized lateral spread.
        for val in [lat_iqr_left, lat_iqr_right]:
            if np.isfinite(val):
                score -= max(0, val - 0.15) * 300
            else:
                score -= 100

        for val in [lat_max_left, lat_max_right]:
            if np.isfinite(val):
                score -= max(0, val - 0.35) * 200
            else:
                score -= 100
        # Penalize raw lateral endpoint displacement.
        # This catches cases where heading-normalized plots look okay,
        # but raw trajectories diverge laterally.
        for val in [raw_y_abs_left, raw_y_abs_right]:
            if np.isfinite(val):
                score -= max(0, val - 0.20) * 500
            else:
                score -= 100
        
        # Strongly penalize left-right raw endpoint separation.
        # In the bad plot, this can be ~1.0 m, which should be heavily penalized.
        if np.isfinite(raw_y_sep_lr):
            score -= max(0, raw_y_sep_lr - 0.35) * 700
        else:
            score -= 100
        
        # Penalize large left-right heading mismatch.
        if np.isfinite(raw_heading_diff_lr):
            score -= max(0, raw_heading_diff_lr - 20) * 15
        
        # Penalize high raw heading variability within each side.
        for val in [raw_heading_iqr_left, raw_heading_iqr_right]:
            if np.isfinite(val):
                score -= max(0, val - 20) * 10
                
        result.update({
        "status": "ok",
        "score": float(score),
        "n_left": n_left,
        "n_right": n_right,
        "n_total": n_total,
        "stride_length_median": stride_length_median,
        "stride_length_cv": stride_length_cv,
        "positive_peak_count_left": positive_peak_counts.get("sensor_left", np.nan),
        "positive_peak_count_right": positive_peak_counts.get("sensor_right", np.nan),
        "lat_iqr_left": lat_iqr_left,
        "lat_iqr_right": lat_iqr_right,
        "lat_max_left": lat_max_left,
        "lat_max_right": lat_max_right,
    
        "raw_y_abs_left": raw_y_abs_left,
        "raw_y_abs_right": raw_y_abs_right,
        "raw_y_sep_lr": raw_y_sep_lr,
        "raw_heading_diff_lr": raw_heading_diff_lr,
        "raw_heading_iqr_left": raw_heading_iqr_left,
        "raw_heading_iqr_right": raw_heading_iqr_right,
        })


        return result

    except Exception as exc:
        result.update({
            "status": "error",
            "error": str(exc),
            "score": -1e9,
        })
        return result

def auto_select_orientation_for_processing(
    dataset_sf_fixed: SensorData,
    segments: List[Tuple[int, int]],
    sampling_rate_hz: int,
    candidates: List[Tuple[float, float]],
    min_strides_per_side: int = 3,
    score_margin_to_switch: float = 25.0,
) -> Tuple[SensorData, Dict[str, Any], pd.DataFrame]:
    """
    Automatically select the best extra yaw orientation for processing.

    The selector evaluates candidate z-axis rotations using the same downstream
    gait-processing logic:
        - convert_to_fbf
        - BarthDtw
        - HerzerEventDetection
        - retained stride list
        - trajectory/spatial QC when possible

    The selected candidate is applied to dataset_sf_fixed and returned.

    A non-zero candidate is only selected if it improves the score over baseline
    by at least score_margin_to_switch. This prevents unnecessary orientation
    changes when 0/0 is already acceptable.
    """
    rows = []

    for left_yaw, right_yaw in candidates:
        candidate_data = apply_extra_yaw_rotation(
            dataset_sf_fixed,
            left_deg=left_yaw,
            right_deg=right_yaw,
        )

        metrics = evaluate_orientation_candidate(
            dataset_candidate=candidate_data,
            segments=segments,
            sampling_rate_hz=sampling_rate_hz,
        )

        row = {
            "candidate_left_yaw_deg": float(left_yaw),
            "candidate_right_yaw_deg": float(right_yaw),
        }
        row.update(metrics)
        rows.append(row)

    qc_df = pd.DataFrame(rows)

    if qc_df.empty:
        selected_info = {
            "selected_left_yaw_deg": 0.0,
            "selected_right_yaw_deg": 0.0,
            "selection_reason": "no_candidates_evaluated",
            "selected_score": np.nan,
            "baseline_score": np.nan,
        }

        return dataset_sf_fixed, selected_info, qc_df

    # Find baseline row.
    baseline_mask = (
        (qc_df["candidate_left_yaw_deg"] == 0.0)
        & (qc_df["candidate_right_yaw_deg"] == 0.0)
    )

    if baseline_mask.any():
        baseline_row = qc_df.loc[baseline_mask].iloc[0]
        baseline_score = float(baseline_row.get("score", -1e9))
    else:
        baseline_score = -1e9

    # Only allow candidates with enough retained strides.
    valid = qc_df.copy()

    valid = valid[
        (valid["status"] == "ok")
        & (valid["n_left"] >= min_strides_per_side)
        & (valid["n_right"] >= min_strides_per_side)
    ].copy()

    if valid.empty:
        selected_left = 0.0
        selected_right = 0.0
        selected_score = baseline_score
        selection_reason = "no_valid_nonbaseline_candidate"
    else:
        valid = valid.sort_values("score", ascending=False).reset_index(drop=True)
        best = valid.iloc[0]

        best_left = float(best["candidate_left_yaw_deg"])
        best_right = float(best["candidate_right_yaw_deg"])
        best_score = float(best["score"])

        # Accept non-zero only if meaningfully better than baseline.
        if (best_left, best_right) == (0.0, 0.0):
            selected_left = best_left
            selected_right = best_right
            selected_score = best_score
            selection_reason = "baseline_best"
        elif best_score >= baseline_score + score_margin_to_switch:
            selected_left = best_left
            selected_right = best_right
            selected_score = best_score
            selection_reason = "nonzero_candidate_improved_score"
        else:
            selected_left = 0.0
            selected_right = 0.0
            selected_score = baseline_score
            selection_reason = "baseline_retained_due_to_small_score_margin"

    selected_data = apply_extra_yaw_rotation(
        dataset_sf_fixed,
        left_deg=selected_left,
        right_deg=selected_right,
    )

    selected_info = {
        "selected_left_yaw_deg": selected_left,
        "selected_right_yaw_deg": selected_right,
        "selection_reason": selection_reason,
        "selected_score": selected_score,
        "baseline_score": baseline_score,
        "score_margin_to_switch": score_margin_to_switch,
    }

    return selected_data, selected_info, qc_df

def _trajectory_raw_endpoint_stats(
    trajectory: StrideLevelTrajectory,
    retained_s_ids: Dict[str, set],
) -> Dict[str, float]:
    """
    Compute raw, non-heading-normalized trajectory endpoint statistics.

    This catches cases where left and right trajectories diverge strongly
    in opposite lateral directions, even if heading-normalized trajectories
    look acceptable.
    """
    stats = {}

    endpoint_y = {}
    endpoint_x = {}
    heading_deg = {}

    for sensor in ["sensor_left", "sensor_right"]:
        endpoint_y[sensor] = []
        endpoint_x[sensor] = []
        heading_deg[sensor] = []

        if sensor not in trajectory.position_:
            continue

        pos = trajectory.position_[sensor]

        if not all(col in pos.columns for col in ["pos_x", "pos_y"]):
            continue

        for s_id in retained_s_ids.get(sensor, []):
            if isinstance(pos.index, pd.MultiIndex) and "s_id" in pos.index.names:
                stride_pos = pos.loc[pos.index.get_level_values("s_id") == s_id]
            else:
                if s_id not in pos.index:
                    continue
                stride_pos = pos.loc[[s_id]]

            if stride_pos.empty or len(stride_pos) < 2:
                continue

            x = stride_pos["pos_x"].to_numpy(dtype=float)
            y = stride_pos["pos_y"].to_numpy(dtype=float)

            dx = x[-1] - x[0]
            dy = y[-1] - y[0]

            endpoint_x[sensor].append(dx)
            endpoint_y[sensor].append(dy)
            heading_deg[sensor].append(np.degrees(np.arctan2(dy, dx)))

    for sensor in ["sensor_left", "sensor_right"]:
        y_arr = np.asarray(endpoint_y[sensor], dtype=float)
        x_arr = np.asarray(endpoint_x[sensor], dtype=float)
        h_arr = np.asarray(heading_deg[sensor], dtype=float)

        prefix = sensor

        stats[f"{prefix}_raw_endpoint_y_median"] = float(np.nanmedian(y_arr)) if len(y_arr) else np.nan
        stats[f"{prefix}_raw_endpoint_y_abs_median"] = float(np.nanmedian(np.abs(y_arr))) if len(y_arr) else np.nan
        stats[f"{prefix}_raw_endpoint_y_iqr"] = (
            float(np.nanpercentile(y_arr, 75) - np.nanpercentile(y_arr, 25))
            if len(y_arr)
            else np.nan
        )

        stats[f"{prefix}_raw_endpoint_x_median"] = float(np.nanmedian(x_arr)) if len(x_arr) else np.nan
        stats[f"{prefix}_raw_heading_deg_median"] = float(np.nanmedian(h_arr)) if len(h_arr) else np.nan
        stats[f"{prefix}_raw_heading_deg_iqr"] = (
            float(np.nanpercentile(h_arr, 75) - np.nanpercentile(h_arr, 25))
            if len(h_arr)
            else np.nan
        )

    left_y = stats.get("sensor_left_raw_endpoint_y_median", np.nan)
    right_y = stats.get("sensor_right_raw_endpoint_y_median", np.nan)

    left_h = stats.get("sensor_left_raw_heading_deg_median", np.nan)
    right_h = stats.get("sensor_right_raw_heading_deg_median", np.nan)

    if np.isfinite(left_y) and np.isfinite(right_y):
        stats["raw_left_right_endpoint_y_separation"] = float(abs(left_y - right_y))
    else:
        stats["raw_left_right_endpoint_y_separation"] = np.nan

    if np.isfinite(left_h) and np.isfinite(right_h):
        stats["raw_left_right_heading_difference_deg"] = float(abs(left_h - right_h))
    else:
        stats["raw_left_right_heading_difference_deg"] = np.nan

    return stats

def _lowpass_filter_1d(
    x: np.ndarray,
    sampling_rate_hz: float,
    cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase Butterworth low-pass filter for one 1D signal.
    """
    x = np.asarray(x, dtype=float)

    if x.size < 10:
        return x.copy()

    nyq = sampling_rate_hz / 2.0

    if cutoff_hz >= nyq:
        return x.copy()

    sos = butter(
        order,
        cutoff_hz,
        btype="lowpass",
        fs=sampling_rate_hz,
        output="sos",
    )

    return sosfiltfilt(sos, x)

def _remove_s_ids_from_df(df: pd.DataFrame, bad_s_ids: set) -> pd.DataFrame:
    """
    Remove rows whose s_id is in bad_s_ids.
    Supports regular index and MultiIndex with level 's_id'.
    """
    if df is None or df.empty or len(bad_s_ids) == 0:
        return df.copy()

    s_ids = _get_s_id_index_values(df)
    return df.loc[~s_ids.isin(bad_s_ids)].copy()

def _series_by_s_id(
    df: pd.DataFrame,
    possible_columns: List[str],
) -> pd.Series:
    """
    Extract one column as a Series indexed by s_id.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    selected_col = None

    for col in possible_columns:
        if col in df.columns:
            selected_col = col
            break

    if selected_col is None:
        return pd.Series(dtype=float)

    values = df[selected_col].astype(float).copy()

    if isinstance(df.index, pd.MultiIndex) and "s_id" in df.index.names:
        values.index = df.index.get_level_values("s_id")
    else:
        values.index = df.index

    return values


def detect_external_style_outlier_s_ids(
    temporal_parameters: Dict[str, pd.DataFrame],
    spatial_parameters: Dict[str, pd.DataFrame],
    stride_time_sd_factor: float = 2.0,
    stride_time_sd_scope: str = "combined",
    exclude_stride_time_gt_max: bool = True,
    stride_time_max_s: float = 2.0,
    exclude_arc_length_gt_max: bool = True,
    arc_length_max_m: float = 3.0,
) -> Tuple[Dict[str, set], pd.DataFrame]:
    """
    Detect outlier strides using the external MATLAB-style clean-CV rule:

        outliers = isoutlier(StrideTimeVector, 'mean', 'thresholdfactor', 2)

    This corresponds to flagging stride times outside mean ± factor*SD.

    Optional extra rules:
        - stride_time > stride_time_max_s
        - arc_length > arc_length_max_m

    The external script pools stride times across bouts before applying the
    clean-CV outlier rule, so the recommended scope is "combined".
    """
    sensors = ["sensor_left", "sensor_right"]

    if stride_time_sd_scope not in ["combined", "per_sensor"]:
        raise ValueError("stride_time_sd_scope must be 'combined' or 'per_sensor'.")

    bad_s_ids_by_sensor = {
        "sensor_left": set(),
        "sensor_right": set(),
    }

    rows = []

    for sensor in sensors:
        temporal = temporal_parameters.get(sensor, pd.DataFrame())
        spatial = spatial_parameters.get(sensor, pd.DataFrame())

        stride_time_s = _series_by_s_id(
            temporal,
            ["stride_time", "stride time [s]", "stride_time_s"],
        )

        arc_length_m = _series_by_s_id(
            spatial,
            ["arc_length", "arc length [m]", "arc_length_m"],
        )

        all_s_ids = set(stride_time_s.index.tolist()) | set(arc_length_m.index.tolist())

        for s_id in all_s_ids:
            rows.append(
                {
                    "sensor": sensor,
                    "s_id": s_id,
                    "stride_time_s": stride_time_s.get(s_id, np.nan),
                    "arc_length_m": arc_length_m.get(s_id, np.nan),
                    "bad_stride_time_mean_sd": False,
                    "bad_stride_time_gt_max": False,
                    "bad_arc_length_gt_max": False,
                    "bad_any": False,
                    "reason": "",
                }
            )

    qc_df = pd.DataFrame(rows)

    if qc_df.empty:
        return bad_s_ids_by_sensor, qc_df

    qc_df["stride_time_sd_mean_used_s"] = np.nan
    qc_df["stride_time_sd_sd_used_s"] = np.nan
    qc_df["stride_time_sd_upper_threshold_s"] = np.nan
    qc_df["stride_time_sd_lower_threshold_s"] = np.nan

    # ------------------------------------------------------------
    # External MATLAB-style mean ± 2SD rule.
    # ------------------------------------------------------------
    if stride_time_sd_scope == "combined":
        valid = np.isfinite(qc_df["stride_time_s"].to_numpy(dtype=float))
        values = qc_df.loc[valid, "stride_time_s"].to_numpy(dtype=float)

        if len(values) > 1:
            mean_st = float(np.nanmean(values))
            sd_st = float(np.nanstd(values, ddof=1))

            upper = mean_st + stride_time_sd_factor * sd_st
            lower = mean_st - stride_time_sd_factor * sd_st

            bad_sd = (
                np.isfinite(qc_df["stride_time_s"])
                & (
                    (qc_df["stride_time_s"] > upper)
                    | (qc_df["stride_time_s"] < lower)
                )
            )

            qc_df.loc[bad_sd, "bad_stride_time_mean_sd"] = True
            qc_df["stride_time_sd_mean_used_s"] = mean_st
            qc_df["stride_time_sd_sd_used_s"] = sd_st
            qc_df["stride_time_sd_upper_threshold_s"] = upper
            qc_df["stride_time_sd_lower_threshold_s"] = lower

    else:
        for sensor in sensors:
            sensor_mask = qc_df["sensor"] == sensor
            valid = sensor_mask & np.isfinite(qc_df["stride_time_s"])
            values = qc_df.loc[valid, "stride_time_s"].to_numpy(dtype=float)

            if len(values) <= 1:
                continue

            mean_st = float(np.nanmean(values))
            sd_st = float(np.nanstd(values, ddof=1))

            upper = mean_st + stride_time_sd_factor * sd_st
            lower = mean_st - stride_time_sd_factor * sd_st

            bad_sd = (
                sensor_mask
                & np.isfinite(qc_df["stride_time_s"])
                & (
                    (qc_df["stride_time_s"] > upper)
                    | (qc_df["stride_time_s"] < lower)
                )
            )

            qc_df.loc[bad_sd, "bad_stride_time_mean_sd"] = True
            qc_df.loc[sensor_mask, "stride_time_sd_mean_used_s"] = mean_st
            qc_df.loc[sensor_mask, "stride_time_sd_sd_used_s"] = sd_st
            qc_df.loc[sensor_mask, "stride_time_sd_upper_threshold_s"] = upper
            qc_df.loc[sensor_mask, "stride_time_sd_lower_threshold_s"] = lower

    # ------------------------------------------------------------
    # Optional extra rule: stride_time > 2 s.
    # ------------------------------------------------------------
    if exclude_stride_time_gt_max:
        bad_gt_max = (
            np.isfinite(qc_df["stride_time_s"])
            & (qc_df["stride_time_s"] > stride_time_max_s)
        )

        qc_df.loc[bad_gt_max, "bad_stride_time_gt_max"] = True

    # ------------------------------------------------------------
    # Optional extra rule: arc_length > 3 m.
    # ------------------------------------------------------------
    if exclude_arc_length_gt_max:
        bad_arc = (
            np.isfinite(qc_df["arc_length_m"])
            & (qc_df["arc_length_m"] > arc_length_max_m)
        )

        qc_df.loc[bad_arc, "bad_arc_length_gt_max"] = True

    # ------------------------------------------------------------
    # Final bad flag.
    # ------------------------------------------------------------
    qc_df["bad_any"] = (
        qc_df["bad_stride_time_mean_sd"]
        | qc_df["bad_stride_time_gt_max"]
        | qc_df["bad_arc_length_gt_max"]
    )

    def _reason(row):
        reasons = []

        if row["bad_stride_time_mean_sd"]:
            reasons.append("stride_time_mean_sd")

        if row["bad_stride_time_gt_max"]:
            reasons.append("stride_time_gt_max")

        if row["bad_arc_length_gt_max"]:
            reasons.append("arc_length_gt_max")

        return ";".join(reasons)

    qc_df["reason"] = qc_df.apply(_reason, axis=1)

    for sensor in sensors:
        bad_s_ids = qc_df.loc[
            (qc_df["sensor"] == sensor) & (qc_df["bad_any"]),
            "s_id",
        ].tolist()

        bad_s_ids_by_sensor[sensor] = set(bad_s_ids)

    return bad_s_ids_by_sensor, qc_df

def remove_bad_s_ids_from_outputs(
    event_list: Dict[str, pd.DataFrame],
    temporal_parameters: Dict[str, pd.DataFrame],
    spatial_parameters: Dict[str, pd.DataFrame],
    bad_s_ids_by_sensor: Dict[str, set],
) -> Tuple[
    Dict[str, pd.DataFrame],
    Dict[str, pd.DataFrame],
    Dict[str, pd.DataFrame],
]:
    """
    Remove bad stride IDs from event, temporal, and spatial outputs.
    """
    event_clean = {}
    temporal_clean = {}
    spatial_clean = {}

    for sensor in ["sensor_left", "sensor_right"]:
        bad_s_ids = bad_s_ids_by_sensor.get(sensor, set())

        event_clean[sensor] = _remove_s_ids_from_df(
            event_list[sensor],
            bad_s_ids,
        )

        temporal_clean[sensor] = _remove_s_ids_from_df(
            temporal_parameters[sensor],
            bad_s_ids,
        )

        spatial_clean[sensor] = _remove_s_ids_from_df(
            spatial_parameters[sensor],
            bad_s_ids,
        )

    return event_clean, temporal_clean, spatial_clean

def _has_nonzero_yaw_override(
    base_name: str,
    overrides: Dict[str, Dict[str, float]],
) -> bool:
    if base_name not in overrides:
        return False

    left = float(overrides[base_name].get("sensor_left", 0.0))
    right = float(overrides[base_name].get("sensor_right", 0.0))

    return not (left == 0.0 and right == 0.0)

def apply_walking_pca_alignment_fallback(
    dataset_sf_aligned_to_gravity: SensorData,
    segments: List[Tuple[int, int]],
    max_samples_per_sensor: int = 200_000,
    target_axis: str = "y",
    pca_plane_axis: Tuple[str, str] = ("gyr_x", "gyr_y"),
) -> Tuple[SensorData, Dict[str, Any]]:
    """
    Estimate horizontal walking-axis alignment from straight-walking segments
    using gaitmap PcaAlignment, then apply the resulting rotation to the full
    dataset.

    This is intended as an automatic fallback when the fixed mounting rotations
    do not generalize to a recording.
    """
    walking_data = build_walking_only_dataset_from_segments(
        dataset=dataset_sf_aligned_to_gravity,
        segments=segments,
        max_samples_per_sensor=max_samples_per_sensor,
    )

    if (
        walking_data["sensor_left"].empty
        or walking_data["sensor_right"].empty
    ):
        return dataset_sf_aligned_to_gravity, {
            "pca_alignment_applied": False,
            "pca_alignment_reason": "no_walking_data",
        }

    try:
        pca_alignment = PcaAlignment(
            target_axis=target_axis,
            pca_plane_axis=pca_plane_axis,
        )

        pca_alignment = pca_alignment.align(walking_data)

        dataset_pca_aligned = rotate_dataset(
            dataset_sf_aligned_to_gravity,
            pca_alignment.rotation_,
        )

        return dataset_pca_aligned, {
            "pca_alignment_applied": True,
            "pca_alignment_reason": "success",
        }

    except Exception as exc:
        return dataset_sf_aligned_to_gravity, {
            "pca_alignment_applied": False,
            "pca_alignment_reason": f"error: {exc}",
        }

def auto_select_best_orientation_pipeline(
    dataset_sf_aligned_to_gravity: SensorData,
    segments: List[Tuple[int, int]],
    sampling_rate_hz: int,
    candidates: List[Tuple[float, float]],
) -> Tuple[SensorData, Dict[str, Any], pd.DataFrame]:
    """
    Select between:
        A. default fixed-mounted orientation
        B. walking-only PCA fallback orientation

    For each base orientation, robust negative-y correction is applied,
    then extra yaw candidates are evaluated.

    Returns the selected dataset and full QC table.
    """
    pipeline_rows = []

    candidate_datasets = []

    # ------------------------------------------------------------
    # A. Default orientation path
    # ------------------------------------------------------------
    default_data, default_orientation_qc = enforce_negative_y_peaks_robust(
        dataset=dataset_sf_aligned_to_gravity,
        segments=segments,
        positive_percentile=95,
        negative_percentile=5,
        flip_ratio_threshold=1.05,
        min_peak_amplitude_deg_s=100,
        flip_mode="preserve_z_rotation",
    )

    candidate_datasets.append(
        {
            "base_method": "default_fixed_rotation",
            "dataset": default_data,
            "base_info": {
                "pca_alignment_applied": False,
                "pca_alignment_reason": "not_used",
            },
            "orientation_qc": default_orientation_qc,
        }
    )

    # ------------------------------------------------------------
    # B. PCA fallback path
    # ------------------------------------------------------------
    if AUTO_TRY_PCA_ALIGNMENT_FALLBACK:
        pca_data, pca_info = apply_walking_pca_alignment_fallback(
            dataset_sf_aligned_to_gravity=dataset_sf_aligned_to_gravity,
            segments=segments,
            max_samples_per_sensor=AUTO_PCA_ALIGNMENT_MAX_SAMPLES_PER_SENSOR,
            target_axis=AUTO_PCA_ALIGNMENT_TARGET_AXIS,
            pca_plane_axis=AUTO_PCA_ALIGNMENT_PLANE_AXIS,
        )

        pca_data, pca_orientation_qc = enforce_negative_y_peaks_robust(
            dataset=pca_data,
            segments=segments,
            positive_percentile=95,
            negative_percentile=5,
            flip_ratio_threshold=1.05,
            min_peak_amplitude_deg_s=100,
            flip_mode="preserve_z_rotation",
        )

        candidate_datasets.append(
            {
                "base_method": "walking_pca_alignment",
                "dataset": pca_data,
                "base_info": pca_info,
                "orientation_qc": pca_orientation_qc,
            }
        )

    # ------------------------------------------------------------
    # Evaluate yaw candidates for each base dataset
    # ------------------------------------------------------------
    best_dataset = None
    best_info = None
    best_score = -1e9

    for candidate_base in candidate_datasets:
        base_method = candidate_base["base_method"]
        base_dataset = candidate_base["dataset"]

        selected_data, selected_info, yaw_qc_df = auto_select_orientation_for_processing(
            dataset_sf_fixed=base_dataset,
            segments=segments,
            sampling_rate_hz=sampling_rate_hz,
            candidates=candidates,
            min_strides_per_side=AUTO_ORIENTATION_MIN_STRIDES_PER_SIDE,
            score_margin_to_switch=0.0,
        )

        if yaw_qc_df is not None and not yaw_qc_df.empty:
            yaw_qc_df = yaw_qc_df.copy()
            yaw_qc_df.insert(0, "base_method", base_method)

            for key, value in candidate_base["base_info"].items():
                yaw_qc_df[key] = value

            pipeline_rows.append(yaw_qc_df)

        selected_score = selected_info.get("selected_score", -1e9)

        if selected_score is None or not np.isfinite(selected_score):
            selected_score = -1e9

        if selected_score > best_score:
            best_score = selected_score
            best_dataset = selected_data
            best_info = {
                **selected_info,
                "selected_base_method": base_method,
                **candidate_base["base_info"],
                "orientation_qc": candidate_base["orientation_qc"],
            }

    if best_dataset is None:
        best_dataset = default_data
        best_info = {
            "selected_base_method": "default_fixed_rotation",
            "selected_left_yaw_deg": 0.0,
            "selected_right_yaw_deg": 0.0,
            "selection_reason": "all_orientation_pipelines_failed",
            "selected_score": np.nan,
            "baseline_score": np.nan,
            "pca_alignment_applied": False,
            "pca_alignment_reason": "not_selected",
            "orientation_qc": default_orientation_qc,
        }

    if pipeline_rows:
        full_qc_df = pd.concat(pipeline_rows, ignore_index=True)
    else:
        full_qc_df = pd.DataFrame()

    return best_dataset, best_info, full_qc_df

def build_walking_only_dataset_from_segments(
    dataset: SensorData,
    segments: List[Tuple[int, int]],
    max_samples_per_sensor: int = 200_000,
) -> Dict[str, pd.DataFrame]:
    """
    Build a walking-only dataset from straight-walking segments.

    This is used to estimate automatic orientation/alignment from actual gait
    rather than from the whole recording.

    Parameters
    ----------
    dataset
        Multi-sensor dataset containing "sensor_left" and "sensor_right".
        This can be a dictionary-style SensorData or a column MultiIndex DataFrame
        that supports dataset["sensor_left"] and dataset["sensor_right"] access.

    segments
        List of straight-walking segments as sample index pairs:
            [(start, end), ...]

    max_samples_per_sensor
        Maximum number of samples to include per sensor.

    Returns
    -------
    walking_data
        Dictionary:
            {
                "sensor_left": walking-only DataFrame,
                "sensor_right": walking-only DataFrame,
            }
    """
    walking_parts = {
        "sensor_left": [],
        "sensor_right": [],
    }

    for sensor in ["sensor_left", "sensor_right"]:
        total = 0

        for start, end in segments:
            if total >= max_samples_per_sensor:
                break

            start = int(max(0, start))
            end = int(min(len(dataset[sensor]) - 1, end))

            if end <= start:
                continue

            seg = dataset[sensor].iloc[start:end + 1].copy()

            remaining = max_samples_per_sensor - total

            if len(seg) > remaining:
                seg = seg.iloc[:remaining].copy()

            walking_parts[sensor].append(seg)
            total += len(seg)

    walking_data = {}

    for sensor in ["sensor_left", "sensor_right"]:
        if walking_parts[sensor]:
            walking_data[sensor] = pd.concat(
                walking_parts[sensor],
                axis=0,
                ignore_index=True,
            )
        else:
            walking_data[sensor] = dataset[sensor].iloc[0:0].copy()

    return walking_data

def _check_consecutive_s_ids(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Check whether the retained stride IDs in a DataFrame are consecutive.

    Returns
    -------
    is_consecutive
        True if s_ids are consecutive or if there are too few rows to check.

    reason
        Text reason for QC/debugging.
    """
    if df is None or df.empty:
        return False, "empty"

    s_ids = _get_s_id_index_values(df)

    if len(s_ids) <= 1:
        return True, "too_few_s_ids_to_check"

    try:
        s_ids_int = np.asarray(s_ids, dtype=int)
    except Exception:
        return True, "non_numeric_s_ids_not_checked"

    diffs = np.diff(s_ids_int)

    if np.all(diffs == 1):
        return True, "consecutive"

    gap_positions = np.where(diffs != 1)[0]
    gap_text = ";".join(
        [
            f"{s_ids_int[i]}->{s_ids_int[i + 1]}"
            for i in gap_positions
        ]
    )

    return False, f"non_consecutive_s_ids:{gap_text}"

def load_external_event_bouts_json(json_path: Path) -> List[Dict[str, Any]]:
    """
    Load external/collaborator event output JSON.

    Expected structure:
        [
            {
                "filtered_events": {
                    "sensor_left": {"ic": [...], "tc": [...]},
                    "sensor_right": {"ic": [...], "tc": [...]}
                },
                ...
            },
            ...
        ]

    Event timings are assumed to be absolute sample indices.
    """
    json_path = Path(json_path)

    with open(json_path, "r") as f:
        data = json.load(f)

    return data

def summarize_retained_stride_counts_from_segment_outputs(
    segment_outputs: List[Dict[str, Any]],
) -> Dict[str, int]:
    """
    Count final retained event rows after:
        - turn/segment filtering
        - boundary trimming
        - outlier exclusion

    This reflects the events that remain in segment_outputs and therefore
    contribute to final gaitmap-summary rows.
    """
    counts = {
        "sensor_left_rows": 0,
        "sensor_right_rows": 0,
        "combined_rows": 0,
        "sensor_left_unique_s_ids": 0,
        "sensor_right_unique_s_ids": 0,
        "combined_unique_s_ids": 0,
        "combined_before_boundary_trim": 0,
        "combined_after_boundary_trim": 0,
    }

    retained_s_ids = {
        "sensor_left": set(),
        "sensor_right": set(),
    }

    for bout in segment_outputs:
        counts["combined_before_boundary_trim"] += int(
            bout.get("n_combined_event_strides_before_boundary_trim", 0)
        )

        counts["combined_after_boundary_trim"] += int(
            bout.get("n_combined_event_strides_after_boundary_trim", 0)
        )

        for sensor in ["sensor_left", "sensor_right"]:
            events = bout["filtered_events"][sensor]

            if events is None or events.empty:
                continue

            n_rows = len(events)

            if sensor == "sensor_left":
                counts["sensor_left_rows"] += n_rows
            else:
                counts["sensor_right_rows"] += n_rows

            s_ids = _get_s_id_index_values(events)
            retained_s_ids[sensor].update(list(s_ids))

    counts["combined_rows"] = (
        counts["sensor_left_rows"]
        + counts["sensor_right_rows"]
    )

    counts["sensor_left_unique_s_ids"] = len(retained_s_ids["sensor_left"])
    counts["sensor_right_unique_s_ids"] = len(retained_s_ids["sensor_right"])
    counts["combined_unique_s_ids"] = (
        counts["sensor_left_unique_s_ids"]
        + counts["sensor_right_unique_s_ids"]
    )

    return counts




def _events_to_external_style_sequence(
    events_left: pd.DataFrame,
    events_right: pd.DataFrame,
    event_column: str = "ic",
) -> pd.DataFrame:
    """
    Build a chronological L/R event sequence from left/right event DataFrames.

    This supports reproducing the external toolbox's one-leg stride-time count:
    if a bout starts with L, count L-R-L intervals; if it starts with R, count
    R-L-R intervals.
    """
    parts = []

    for events, side in [(events_left, "L"), (events_right, "R")]:
        if events is None or events.empty or event_column not in events.columns:
            continue
        tmp = pd.DataFrame(
            {
                "sample": events[event_column].to_numpy(dtype=float),
                "side": side,
            }
        )
        tmp = tmp[np.isfinite(tmp["sample"])]
        if not tmp.empty:
            parts.append(tmp)

    if not parts:
        return pd.DataFrame(columns=["sample", "side"])

    seq = pd.concat(parts, ignore_index=True)
    seq = seq.sort_values("sample", kind="mergesort").reset_index(drop=True)
    return seq


def external_style_stride_times_from_event_sequence(seq: pd.DataFrame, sampling_rate_hz: int) -> np.ndarray:
    """
    Replicate the external toolbox counting idea for stride-time observations.

    The external MATLAB code uses only one alternating side per bout: if the
    bout starts with L, it identifies L-R-L; if it starts with R, it identifies
    R-L-R. It then removes stride intervals where the stride-location gap is
    greater than 2.
    """
    if seq is None or seq.empty or len(seq) < 3:
        return np.array([], dtype=float)

    sides = "".join(seq["side"].astype(str).tolist())
    start_side = sides[0]
    pattern = "LRL" if start_side == "L" else "RLR"

    stride_locations = [
        i for i in range(0, len(sides) - 2)
        if sides[i:i + 3] == pattern
    ]

    if len(stride_locations) == 0:
        return np.array([], dtype=float)

    # External script appends StrideLocations(end)+2 to close the final stride.
    final_location = stride_locations[-1] + 2
    if final_location < len(seq):
        stride_locations = stride_locations + [final_location]

    if len(stride_locations) < 2:
        return np.array([], dtype=float)

    samples = seq["sample"].to_numpy(dtype=float)
    stride_times = []

    for i in range(len(stride_locations) - 1):
        # External line: StrideTime(diff(StrideLocations)>2)=[]
        if (stride_locations[i + 1] - stride_locations[i]) > 2:
            continue
        stride_times.append(
            (samples[stride_locations[i + 1]] - samples[stride_locations[i]]) / sampling_rate_hz
        )

    return np.asarray(stride_times, dtype=float)



def _external_style_clean_stride_time_vector(
    stride_times: np.ndarray,
    stride_time_sd_factor: float = OUTLIER_STRIDE_TIME_SD_FACTOR,
    exclude_stride_time_gt_max: bool = OUTLIER_EXCLUDE_STRIDE_TIME_GT_MAX,
    stride_time_max_s: float = OUTLIER_STRIDE_TIME_MAX_S,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Apply the same stride-time-only outlier logic to an external-style
    stride-time vector.

    This mirrors the trial-level external MATLAB clean-CV concept:
        isoutlier(StrideTimeVector, 'mean', 'thresholdfactor', 2)

    and optionally applies the pipeline's additional stride-time > max rule.
    Arc-length filtering is not available for raw external JSON stride times.
    """
    arr = np.asarray(stride_times, dtype=float).reshape(-1)
    finite_mask = np.isfinite(arr)

    bad_mean_sd = np.zeros(arr.shape, dtype=bool)
    bad_gt_max = np.zeros(arr.shape, dtype=bool)

    finite_values = arr[finite_mask]

    mean_st = np.nan
    sd_st = np.nan
    lower = np.nan
    upper = np.nan

    if finite_values.size > 1:
        mean_st = float(np.nanmean(finite_values))
        sd_st = float(np.nanstd(finite_values, ddof=1))
        lower = mean_st - stride_time_sd_factor * sd_st
        upper = mean_st + stride_time_sd_factor * sd_st

        bad_mean_sd = (
            finite_mask
            & ((arr < lower) | (arr > upper))
        )

    if exclude_stride_time_gt_max:
        bad_gt_max = finite_mask & (arr > stride_time_max_s)

    bad_any = bad_mean_sd | bad_gt_max
    clean = arr[finite_mask & ~bad_any]

    qc = {
        "stride_time_clean_mean_used_s": mean_st,
        "stride_time_clean_sd_used_s": sd_st,
        "stride_time_clean_lower_threshold_s": lower,
        "stride_time_clean_upper_threshold_s": upper,
        "n_stride_time_raw": int(np.sum(finite_mask)),
        "n_stride_time_clean": int(clean.size),
        "n_stride_time_outliers_total": int(np.sum(bad_any)),
        "n_stride_time_outliers_mean_sd": int(np.sum(bad_mean_sd)),
        "n_stride_time_outliers_gt_max": int(np.sum(bad_gt_max)),
    }

    return clean, qc

def filter_outputs_by_external_direct_event_list(
    external_bouts: List[Dict[str, Any]],
    event_list: Dict[str, pd.DataFrame],
    spatial_parameters: Dict[str, pd.DataFrame],
    temporal_parameters: Dict[str, pd.DataFrame],
) -> List[Dict[str, Any]]:
    """
    Build segment_outputs directly from an external-derived min_vel_event_list_.

    No event matching is performed. Rows are assigned to bouts using the
    `external_bout_id` column generated by external_events_to_min_vel_event_list.
    """
    segment_outputs = []
    sensors = ["sensor_left", "sensor_right"]

    for external_bout_id, bout in enumerate(external_bouts, start=1):
        filtered_events = {}
        filtered_spatial = {}
        filtered_temporal = {}
        counts = {}

        samples = []
        for sensor in sensors:
            ext_events = _external_events_to_df(bout, sensor)
            if ext_events is not None and not ext_events.empty:
                if "ic" in ext_events.columns:
                    samples.extend(ext_events["ic"].dropna().tolist())
                if "tc" in ext_events.columns:
                    samples.extend(ext_events["tc"].dropna().tolist())

        if len(samples) > 0:
            segment_start = int(np.nanmin(samples))
            segment_end = int(np.nanmax(samples))
        else:
            segment_start = np.nan
            segment_end = np.nan

        for sensor in sensors:
            events = event_list.get(sensor, pd.DataFrame())

            if events is None or events.empty or "external_bout_id" not in events.columns:
                valid_s_ids = []
            else:
                valid_s_ids = events.index[
                    events["external_bout_id"].astype(float) == float(external_bout_id)
                ].tolist()

            filtered_events[sensor] = _filter_by_s_ids(
                event_list.get(sensor, pd.DataFrame()),
                valid_s_ids,
            )
            filtered_spatial[sensor] = _filter_by_s_ids(
                spatial_parameters.get(sensor, pd.DataFrame()),
                valid_s_ids,
            )
            filtered_temporal[sensor] = _filter_by_s_ids(
                temporal_parameters.get(sensor, pd.DataFrame()),
                valid_s_ids,
            )

            ext_events = _external_events_to_df(bout, sensor)
            if ext_events is not None and not ext_events.empty and EXTERNAL_EVENT_MATCH_ANCHOR in ext_events.columns:
                n_external_events = int(ext_events[EXTERNAL_EVENT_MATCH_ANCHOR].notna().sum())
            else:
                n_external_events = 0

            counts[sensor] = {
                "n_external_events": n_external_events,
                "n_external_direct_minvel_rows": len(valid_s_ids),
                "n_filtered_event_rows": len(filtered_events[sensor]),
                "n_filtered_temporal_rows": len(filtered_temporal[sensor]),
                "n_filtered_spatial_rows": len(filtered_spatial[sensor]),
            }

        segment_outputs.append(
            {
                "bout_id": external_bout_id,
                "segment_start": segment_start,
                "segment_end": segment_end,
                "filtered_events": filtered_events,
                "filtered_spatial_paras": filtered_spatial,
                "filtered_temporal_paras": filtered_temporal,
                "counts": counts,
                "boundary_trim_mode": "external_events_direct_estimated_min_vel",
                "number_boundary_strides": 0,
                "n_combined_event_strides_before_boundary_trim": (
                    counts["sensor_left"]["n_external_events"]
                    + counts["sensor_right"]["n_external_events"]
                ),
                "n_combined_event_strides_after_boundary_trim": (
                    counts["sensor_left"]["n_external_direct_minvel_rows"]
                    + counts["sensor_right"]["n_external_direct_minvel_rows"]
                ),
            }
        )

    return segment_outputs


def estimate_min_vel_in_window(
    sensor_df: pd.DataFrame,
    window_start: float,
    window_end: float,
    sampling_rate_hz: int,
    min_vel_search_win_size_ms: float = 100.0,
) -> float:
    """
    Estimate min_vel as the minimum gyroscope-energy sample in a fixed window.
    """
    if not np.isfinite(window_start) or not np.isfinite(window_end):
        return np.nan

    n_samples = len(sensor_df)

    lo = int(max(0, round(window_start)))
    hi = int(min(n_samples - 1, round(window_end)))

    if hi <= lo:
        return np.nan

    gyr_cols = [
        c for c in ["gyr_ml", "gyr_pa", "gyr_si", "gyr_x", "gyr_y", "gyr_z"]
        if c in sensor_df.columns
    ]

    if len(gyr_cols) == 0:
        return np.nan

    gyr = sensor_df[gyr_cols].iloc[lo:hi + 1].to_numpy(dtype=float)
    energy = np.nansum(gyr ** 2, axis=1)

    if energy.size == 0 or np.all(~np.isfinite(energy)):
        return np.nan

    win = int(round(min_vel_search_win_size_ms / 1000.0 * sampling_rate_hz))
    win = max(3, win)

    if energy.size < win:
        return float(lo + int(np.nanargmin(energy)))

    kernel = np.ones(win, dtype=float) / win
    energy_smooth = np.convolve(energy, kernel, mode="valid")

    if energy_smooth.size == 0 or np.all(~np.isfinite(energy_smooth)):
        return np.nan

    local_min = int(np.nanargmin(energy_smooth))
    min_vel_local = local_min + win // 2

    return float(lo + min_vel_local)

def _robust_normalize_score(x: np.ndarray) -> np.ndarray:
    """
    Robustly normalize a 1D score vector so lower values remain lower.
    """
    x = np.asarray(x, dtype=float)

    if x.size == 0 or np.all(~np.isfinite(x)):
        return np.full_like(x, np.nan, dtype=float)

    med = np.nanmedian(x)
    iqr = np.nanpercentile(x, 75) - np.nanpercentile(x, 25)

    if not np.isfinite(iqr) or iqr <= 1e-9:
        iqr = np.nanstd(x)

    if not np.isfinite(iqr) or iqr <= 1e-9:
        iqr = 1.0

    return (x - med) / iqr


def estimate_min_vel_combined_signal_in_window(
    sensor_df: pd.DataFrame,
    window_start: float,
    window_end: float,
    sampling_rate_hz: int,
    min_vel_search_win_size_ms: float = 100.0,
    gyro_weight: float = 1.0,
    acc_jerk_weight: float = 0.75,
    acc_norm_weight: float = 0.25,
) -> float:
    """
    Estimate min_vel using combined gyro + acceleration stability.

    The selected sample is the centre of the window where the combined score
    is lowest:

        score = low gyro energy
              + low acceleration jerk/change
              + low acceleration-norm variability

    This is more robust than gyro-only when the gyro has multiple nearby minima.
    """
    if not np.isfinite(window_start) or not np.isfinite(window_end):
        return np.nan

    n_samples = len(sensor_df)

    lo = int(max(0, round(window_start)))
    hi = int(min(n_samples - 1, round(window_end)))

    if hi <= lo:
        return np.nan

    gyr_cols = [
        c for c in ["gyr_ml", "gyr_pa", "gyr_si", "gyr_x", "gyr_y", "gyr_z"]
        if c in sensor_df.columns
    ]

    acc_cols = [
        c for c in ["acc_ml", "acc_pa", "acc_si", "acc_x", "acc_y", "acc_z"]
        if c in sensor_df.columns
    ]

    if len(gyr_cols) == 0:
        return np.nan

    gyr = sensor_df[gyr_cols].iloc[lo:hi + 1].to_numpy(dtype=float)
    gyro_energy = np.nansum(gyr ** 2, axis=1)

    # ------------------------------------------------------------
    # Acceleration jerk/change score.
    # ------------------------------------------------------------
    if len(acc_cols) > 0:
        acc = sensor_df[acc_cols].iloc[lo:hi + 1].to_numpy(dtype=float)

        acc_diff = np.diff(acc, axis=0, prepend=acc[[0], :])
        acc_jerk_energy = np.nansum(acc_diff ** 2, axis=1)

        acc_norm = np.linalg.norm(acc, axis=1)
        acc_norm_diff = np.abs(
            np.diff(acc_norm, prepend=acc_norm[0])
        )
    else:
        acc_jerk_energy = np.zeros_like(gyro_energy)
        acc_norm_diff = np.zeros_like(gyro_energy)

    if gyro_energy.size == 0 or np.all(~np.isfinite(gyro_energy)):
        return np.nan

    win = int(round(min_vel_search_win_size_ms / 1000.0 * sampling_rate_hz))
    win = max(3, win)

    def smooth_valid(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)

        if x.size < win:
            return x

        kernel = np.ones(win, dtype=float) / win
        return np.convolve(x, kernel, mode="valid")

    gyro_s = smooth_valid(gyro_energy)
    acc_jerk_s = smooth_valid(acc_jerk_energy)
    acc_norm_s = smooth_valid(acc_norm_diff)

    min_len = min(len(gyro_s), len(acc_jerk_s), len(acc_norm_s))

    if min_len == 0:
        return np.nan

    gyro_s = gyro_s[:min_len]
    acc_jerk_s = acc_jerk_s[:min_len]
    acc_norm_s = acc_norm_s[:min_len]

    score = (
        gyro_weight * _robust_normalize_score(gyro_s)
        + acc_jerk_weight * _robust_normalize_score(acc_jerk_s)
        + acc_norm_weight * _robust_normalize_score(acc_norm_s)
    )

    if score.size == 0 or np.all(~np.isfinite(score)):
        return np.nan

    local = int(np.nanargmin(score))

    if gyro_energy.size >= win:
        local = local + win // 2

    return float(lo + local)


def estimate_min_vel_boundary_combined_or_fallback(
    sensor_df: pd.DataFrame,
    window_start: float,
    window_end: float,
    fallback_sample: float,
    sampling_rate_hz: int,
    min_vel_search_win_size_ms: float = 100.0,
) -> Tuple[float, str]:
    """
    Estimate min_vel boundary using combined gyro+acc signal.

    If that fails, use fallback_sample so external rows are not dropped.
    """
    candidate = estimate_min_vel_combined_signal_in_window(
        sensor_df=sensor_df,
        window_start=window_start,
        window_end=window_end,
        sampling_rate_hz=sampling_rate_hz,
        min_vel_search_win_size_ms=min_vel_search_win_size_ms,
    )

    if np.isfinite(candidate):
        return float(candidate), "combined_gyro_acc_minimum"

    if np.isfinite(fallback_sample):
        return float(fallback_sample), "fallback_boundary_sample"

    return np.nan, "failed_no_fallback"

def build_min_vel_boundaries_for_external_bout(
    ext_events: pd.DataFrame,
    sensor_df: pd.DataFrame,
    sampling_rate_hz: int,
    min_vel_search_win_size_ms: float = 100.0,
    first_boundary_before_tc_s: float = 0.25,
    first_boundary_search_back_s: float = 0.60,
    internal_after_prev_ic_s: float = 0.05,
    internal_before_current_tc_s: float = 0.03,
    terminal_after_last_ic_s: float = 0.45,
    terminal_search_after_last_ic_s: float = 0.90,
    min_boundary_gap_s: float = 0.02,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build N+1 min_vel boundaries for N external IC/TC events.

    For N external rows:
        row i start = boundaries[i]
        row i end   = boundaries[i + 1]

    Internal boundary i is searched between:
        IC_{i-1} and TC_i

    This means there is one min_vel boundary in the stance interval between
    consecutive external steps.
    """
    ext = ext_events.copy()
    ext = ext.sort_values("ic").reset_index(drop=True)

    n_events = len(ext)

    boundaries = np.full(n_events + 1, np.nan, dtype=float)
    reasons = [""] * (n_events + 1)

    if n_events == 0:
        return boundaries, reasons

    tc = ext["tc"].to_numpy(dtype=float)
    ic = ext["ic"].to_numpy(dtype=float)

    # ------------------------------------------------------------
    # Boundary 0: before first TC.
    # ------------------------------------------------------------
    first_tc = tc[0]

    boundaries[0], reasons[0] = estimate_min_vel_boundary_combined_or_fallback(
        sensor_df=sensor_df,
        window_start=first_tc - first_boundary_search_back_s * sampling_rate_hz,
        window_end=first_tc - internal_before_current_tc_s * sampling_rate_hz,
        fallback_sample=first_tc - first_boundary_before_tc_s * sampling_rate_hz,
        sampling_rate_hz=sampling_rate_hz,
        min_vel_search_win_size_ms=min_vel_search_win_size_ms,
    )

    # ------------------------------------------------------------
    # Internal boundaries:
    # one min_vel between previous IC and current TC.
    # ------------------------------------------------------------
    for i in range(1, n_events):
        previous_ic = ic[i - 1]
        current_tc = tc[i]

        window_start = previous_ic + internal_after_prev_ic_s * sampling_rate_hz
        window_end = current_tc - internal_before_current_tc_s * sampling_rate_hz

        fallback = previous_ic + 0.50 * (current_tc - previous_ic)

        boundaries[i], reasons[i] = estimate_min_vel_boundary_combined_or_fallback(
            sensor_df=sensor_df,
            window_start=window_start,
            window_end=window_end,
            fallback_sample=fallback,
            sampling_rate_hz=sampling_rate_hz,
            min_vel_search_win_size_ms=min_vel_search_win_size_ms,
        )

    # ------------------------------------------------------------
    # Terminal boundary after last IC.
    # This retains the final external stride.
    # ------------------------------------------------------------
    last_ic = ic[-1]

    boundaries[-1], reasons[-1] = estimate_min_vel_boundary_combined_or_fallback(
        sensor_df=sensor_df,
        window_start=last_ic + internal_after_prev_ic_s * sampling_rate_hz,
        window_end=last_ic + terminal_search_after_last_ic_s * sampling_rate_hz,
        fallback_sample=last_ic + terminal_after_last_ic_s * sampling_rate_hz,
        sampling_rate_hz=sampling_rate_hz,
        min_vel_search_win_size_ms=min_vel_search_win_size_ms,
    )

    # ------------------------------------------------------------
    # Make boundaries finite and strictly increasing.
    # ------------------------------------------------------------
    min_gap_samples = max(1, int(round(min_boundary_gap_s * sampling_rate_hz)))

    for i in range(len(boundaries)):
        if not np.isfinite(boundaries[i]):
            if i == 0:
                boundaries[i] = max(0, tc[0] - first_boundary_before_tc_s * sampling_rate_hz)
            else:
                boundaries[i] = boundaries[i - 1] + min_gap_samples

            reasons[i] = "monotonic_fallback_from_nan"

        if i > 0 and boundaries[i] <= boundaries[i - 1]:
            boundaries[i] = boundaries[i - 1] + min_gap_samples
            reasons[i] = "monotonic_nudged_forward"

    return boundaries, reasons

def external_events_to_min_vel_event_list(
    external_bouts: List[Dict[str, Any]],
    bf_data: SensorData,
    sampling_rate_hz: int,
    min_vel_search_win_size_ms: float = 100.0,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Convert external JSON IC/TC events directly into a gaitmap-compatible
    min_vel_event_list_.

    This version uses boundary-based min_vel construction:

        N external IC/TC rows
        -> N + 1 min_vel boundaries
        -> N gaitmap-compatible rows

    This prevents dropping the final stride and avoids per-row min_vel failures
    reducing the number of retained external events.
    """
    sensors = ["sensor_left", "sensor_right"]

    if external_bouts is None:
        raise ValueError(
            "external_events_to_min_vel_event_list received external_bouts=None."
        )

    if isinstance(external_bouts, dict):
        external_bouts = [external_bouts]

    event_parts = {sensor: [] for sensor in sensors}
    qc_rows = []
    next_s_id = {sensor: 0 for sensor in sensors}

    for external_bout_id, bout in enumerate(external_bouts, start=1):
        for sensor in sensors:
            ext_events = _external_events_to_df(bout, sensor)

            if ext_events is None or ext_events.empty:
                qc_rows.append(
                    {
                        "sensor": sensor,
                        "external_bout_id": external_bout_id,
                        "n_external_events": 0,
                        "n_output_minvel_strides": 0,
                        "n_rows_dropped_after_boundary_construction": 0,
                        "reason": "no_external_events_for_sensor",
                    }
                )
                continue

            if "ic" not in ext_events.columns or "tc" not in ext_events.columns:
                qc_rows.append(
                    {
                        "sensor": sensor,
                        "external_bout_id": external_bout_id,
                        "n_external_events": 0,
                        "n_output_minvel_strides": 0,
                        "n_rows_dropped_after_boundary_construction": 0,
                        "reason": "missing_ic_or_tc_columns",
                    }
                )
                continue

            # ------------------------------------------------------------------
            # Keep only external IC/TC and preserve event order by IC.
            # ------------------------------------------------------------------
            ext_events = ext_events[["ic", "tc"]].copy()

            ext_events["ic"] = pd.to_numeric(ext_events["ic"], errors="coerce")
            ext_events["tc"] = pd.to_numeric(ext_events["tc"], errors="coerce")

            ext_events = ext_events.dropna(subset=["ic", "tc"], how="any")
            ext_events = ext_events.sort_values("ic").reset_index(drop=True)

            n_external_events = len(ext_events)

            if n_external_events == 0:
                qc_rows.append(
                    {
                        "sensor": sensor,
                        "external_bout_id": external_bout_id,
                        "n_external_events": 0,
                        "n_output_minvel_strides": 0,
                        "n_rows_dropped_after_boundary_construction": 0,
                        "reason": "no_valid_external_ic_tc_events",
                    }
                )
                continue

            sensor_df = bf_data[sensor]

            # Preserve true external event numbering within this bout/sensor.
            ext_events["external_event_idx"] = np.arange(
                1,
                n_external_events + 1,
                dtype=int,
            )

            # ------------------------------------------------------------------
            # Build N+1 min_vel boundaries for N external rows.
            # ------------------------------------------------------------------
            boundaries, boundary_reasons = build_min_vel_boundaries_for_external_bout(
                ext_events=ext_events,
                sensor_df=sensor_df,
                sampling_rate_hz=sampling_rate_hz,
                min_vel_search_win_size_ms=min_vel_search_win_size_ms,
            )

            out = ext_events.copy()

            # One output row per external event.
            out["start"] = boundaries[:-1]
            out["end"] = boundaries[1:]
            out["min_vel"] = out["start"]

            out["min_vel_boundary_reason_start"] = boundary_reasons[:-1]
            out["min_vel_boundary_reason_end"] = boundary_reasons[1:]

            out["pre_ic"] = out["ic"].shift(1)
            out["external_bout_id"] = external_bout_id

            # ------------------------------------------------------------------
            # Keep only rows that are truly impossible.
            # This should almost never drop rows now.
            # ------------------------------------------------------------------
            valid = (
                np.isfinite(out["start"])
                & np.isfinite(out["end"])
                & np.isfinite(out["ic"])
                & np.isfinite(out["tc"])
                & (out["end"] > out["start"])
            )

            n_before = len(out)
            out = out.loc[valid].copy()
            n_after = len(out)

            if out.empty:
                qc_rows.append(
                    {
                        "sensor": sensor,
                        "external_bout_id": external_bout_id,
                        "n_external_events": n_before,
                        "n_output_minvel_strides": 0,
                        "n_rows_dropped_after_boundary_construction": n_before,
                        "reason": "all_rows_invalid_after_boundary_construction",
                    }
                )
                continue

            for col in ["start", "end", "ic", "tc", "min_vel"]:
                out[col] = out[col].round().astype(int)

            # Boolean/numeric metadata only. String columns will be stripped
            # before gaitmap trajectory calculation.
            out["boundary_start_code"] = [
                1 if reason == "energy_minimum" else 2
                for reason in out["min_vel_boundary_reason_start"].tolist()
            ]

            out["boundary_end_code"] = [
                1 if reason == "energy_minimum" else 2
                for reason in out["min_vel_boundary_reason_end"].tolist()
            ]

            n_rows = len(out)

            out.index = np.arange(
                next_s_id[sensor],
                next_s_id[sensor] + n_rows,
                dtype=int,
            )
            out.index.name = "s_id"
            next_s_id[sensor] += n_rows

            event_parts[sensor].append(out)

            qc_rows.append(
                {
                    "sensor": sensor,
                    "external_bout_id": external_bout_id,
                    "n_external_events": n_before,
                    "n_output_minvel_strides": n_after,
                    "n_rows_dropped_after_boundary_construction": n_before - n_after,
                    "reason": "ok",
                }
            )

    # ----------------------------------------------------------------------
    # Concatenate per-sensor event parts.
    # ----------------------------------------------------------------------
    event_list = {}

    for sensor in sensors:
        if len(event_parts[sensor]) > 0:
            df = pd.concat(event_parts[sensor], axis=0)
            df.index.name = "s_id"
            event_list[sensor] = df
        else:
            empty = pd.DataFrame(
                columns=[
                    "start",
                    "end",
                    "ic",
                    "tc",
                    "min_vel",
                    "pre_ic",
                    "external_bout_id",
                    "external_event_idx",
                    "min_vel_boundary_reason_start",
                    "min_vel_boundary_reason_end",
                    "boundary_start_code",
                    "boundary_end_code",
                ]
            )
            empty.index.name = "s_id"
            event_list[sensor] = empty

    qc_df = pd.DataFrame(qc_rows)

    return event_list, qc_df

def strip_event_list_for_gaitmap_calculations(
    event_list: Dict[str, pd.DataFrame],
    n_samples: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Convert a full external-derived event list into a gaitmap-safe event list.

    Gaitmap trajectory/spatial code requires only numeric event columns and
    expects event samples to lie inside the min_vel stride interval:

        start <= tc < end
        start <= ic < end

    If IC or TC lies exactly on the end boundary, this function nudges end
    forward by one sample. This prevents errors such as:

        KeyError: '[(16, 128)] not in index'

    because that happens when the relative IC/TC index equals the stride length.
    """
    required_cols = ["start", "end", "ic", "tc", "min_vel"]
    optional_cols = ["pre_ic"]

    stripped = {}

    for sensor in ["sensor_left", "sensor_right"]:
        df = event_list.get(sensor, pd.DataFrame())

        if df is None or df.empty:
            empty = pd.DataFrame(columns=required_cols + optional_cols)
            empty.index.name = "s_id"
            stripped[sensor] = empty
            continue

        available_cols = [
            c for c in required_cols + optional_cols
            if c in df.columns
        ]

        out = df[available_cols].copy()

        # Convert all gaitmap-facing columns to numeric.
        for col in available_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        # Required columns must exist and be finite.
        out = out.dropna(subset=required_cols, how="any").copy()

        if out.empty:
            empty = pd.DataFrame(columns=required_cols + optional_cols)
            empty.index.name = "s_id"
            stripped[sensor] = empty
            continue

        # Round to integer sample indices.
        for col in required_cols:
            out[col] = out[col].round().astype(int)

        if "pre_ic" in out.columns:
            out["pre_ic"] = pd.to_numeric(out["pre_ic"], errors="coerce")

        # ------------------------------------------------------------
        # Make stride boundaries safe for gaitmap.
        # ------------------------------------------------------------
        for idx, row in out.iterrows():
            start = int(row["start"])
            end = int(row["end"])
            ic = int(row["ic"])
            tc = int(row["tc"])

            # Ensure start is before both TC and IC.
            earliest_event = min(ic, tc)

            if start > earliest_event:
                start = earliest_event - 1

            # Ensure end is after both TC and IC.
            # Important: gaitmap uses relative indexing, so ic == end is invalid.
            latest_event = max(ic, tc)

            if end <= latest_event:
                end = latest_event + 1

            # Boundaries must be valid.
            if n_samples is not None:
                start = max(0, min(start, n_samples - 2))
                end = max(start + 1, min(end, n_samples - 1))
            else:
                start = max(0, start)
                end = max(start + 1, end)

            out.at[idx, "start"] = start
            out.at[idx, "end"] = end
            out.at[idx, "min_vel"] = start

        # Final validity check.
        valid = (
            np.isfinite(out["start"])
            & np.isfinite(out["end"])
            & np.isfinite(out["ic"])
            & np.isfinite(out["tc"])
            & np.isfinite(out["min_vel"])
            & (out["end"] > out["start"])
            & (out["ic"] >= out["start"])
            & (out["ic"] < out["end"])
            & (out["tc"] >= out["start"])
            & (out["tc"] < out["end"])
        )

        out = out.loc[valid].copy()

        out.index.name = "s_id"
        stripped[sensor] = out

    return stripped

def _external_events_to_df(
    external_bout: Dict[str, Any],
    sensor: str,
) -> pd.DataFrame:
    """
    Convert one external JSON bout/sensor into a DataFrame with ic/tc columns.
    """
    sensor_events = external_bout.get("filtered_events", {}).get(sensor, {})

    ic = np.asarray(sensor_events.get("ic", []), dtype=float)
    tc = np.asarray(sensor_events.get("tc", []), dtype=float)

    n = max(len(ic), len(tc))

    if n == 0:
        return pd.DataFrame(columns=["ic", "tc"])

    ic_pad = np.full(n, np.nan, dtype=float)
    tc_pad = np.full(n, np.nan, dtype=float)

    ic_pad[:len(ic)] = ic
    tc_pad[:len(tc)] = tc

    return pd.DataFrame({"ic": ic_pad, "tc": tc_pad})


def _event_sample_range_from_external_bout(
    external_bout: Dict[str, Any],
) -> Tuple[float, float]:
    """
    Get min/max sample from all external IC/TC events in one external bout.
    """
    samples = []

    for sensor in ["sensor_left", "sensor_right"]:
        events = _external_events_to_df(external_bout, sensor)

        if events is None or events.empty:
            continue

        samples.extend(events["ic"].dropna().tolist())
        samples.extend(events["tc"].dropna().tolist())

    if len(samples) == 0:
        return np.nan, np.nan

    return float(np.nanmin(samples)), float(np.nanmax(samples))

# =============================================================================
# 3. OPAL5 H5 READER
# =============================================================================

def read_opal5_data(filename: Path) -> Dict[str, Any]:
    """
    Python translation of MATLAB readOpal5Data.

    MATLAB logic:
        fileFormat = h5readatt(filename,'/','FileFormatVersion');
        sensors = h5info(filename,'/Sensors');
        orientation = h5info(filename,'/Processed');
        Opal.SampleRate = h5readatt(first sensor Configuration, 'Sample Rate');

        for each sensor:
            label = h5readatt(..., 'Label 0')
            label = deblank(label)
            Opal.(label_no_spaces).acc = h5read(..., '/Accelerometer')
            Opal.(label_no_spaces).gyro = h5read(..., '/Gyroscope')
            Opal.(label_no_spaces).mag = h5read(..., '/Magnetometer')
            Opal.(label_no_spaces).orient = h5read(..., corresponding Processed Orientation)
    """
    filename = Path(filename)

    with h5py.File(filename, "r") as f:
        file_format = _decode_attr(f.attrs.get("FileFormatVersion", np.nan))
        try:
            file_format = float(file_format)
        except Exception:
            raise ValueError(f"Could not read FileFormatVersion from {filename}")

        if file_format < 5:
            raise ValueError(f"{filename} is FileFormatVersion {file_format}; this reader expects version 5+")

        if "/Sensors" not in f:
            raise ValueError(f"No /Sensors group found in {filename}")

        if "/Processed" not in f:
            raise ValueError(f"No /Processed group found in {filename}")

        sensor_group_names = list(f["/Sensors"].keys())
        processed_group_names = list(f["/Processed"].keys())

        if len(sensor_group_names) == 0:
            raise ValueError(f"No sensor groups found under /Sensors in {filename}")

        first_sensor_name = sensor_group_names[0]
        sample_rate = _decode_attr(
            f[f"/Sensors/{first_sensor_name}/Configuration"].attrs["Sample Rate"]
        )
        sample_rate = float(sample_rate)

        opal: Dict[str, Any] = {
            "SampleRate": sample_rate,
            "Time": None,
            "acc_units": None,
            "gyro_units": None,
            "sensors": {},
        }

        for i, sensor_name in enumerate(sensor_group_names):
            sensor_path = f"/Sensors/{sensor_name}"
            cfg_path = f"{sensor_path}/Configuration"

            label = _decode_attr(f[cfg_path].attrs["Label 0"])
            label_clean = _safe_name(label)

            acc = _ensure_2d_signal_orientation(
                np.asarray(f[f"{sensor_path}/Accelerometer"]),
                expected_channels=3,
            )
            gyro = _ensure_2d_signal_orientation(
                np.asarray(f[f"{sensor_path}/Gyroscope"]),
                expected_channels=3,
            )
            mag = _ensure_2d_signal_orientation(
                np.asarray(f[f"{sensor_path}/Magnetometer"]),
                expected_channels=3,
            )

            orient = None
            if i < len(processed_group_names):
                processed_name = processed_group_names[i]
                orient_path = f"/Processed/{processed_name}/Orientation"
                if orient_path in f:
                    orient = _ensure_2d_signal_orientation(
                        np.asarray(f[orient_path]),
                        expected_channels=4,
                    )

            if orient is None:
                raise ValueError(
                    f"Could not find orientation for sensor {label_clean} in {filename}. "
                    "Your MATLAB reader had an AHRS fallback commented out; this Python version requires Orientation."
                )

            opal["sensors"][label_clean] = {
                "acc": acc,
                "gyro": gyro,
                "mag": mag,
                "orient": orient,
                "monitorLabel": label,
            }

        first_sensor_path = f"/Sensors/{first_sensor_name}"

        if f"{first_sensor_path}/Time" in f:
            time1 = np.asarray(f[f"{first_sensor_path}/Time"])
            time1 = np.asarray(time1).reshape(-1)
            opal["Time"] = ((time1 - time1[0]).astype(float) / 1_000_000.0)

        if f"{first_sensor_path}/Accelerometer" in f:
            opal["acc_units"] = _decode_attr(
                f[f"{first_sensor_path}/Accelerometer"].attrs.get("Units", "")
            )

        if f"{first_sensor_path}/Gyroscope" in f:
            opal["gyro_units"] = _decode_attr(
                f[f"{first_sensor_path}/Gyroscope"].attrs.get("Units", "")
            )

    sensors = opal["sensors"]

    # MATLAB-equivalent ankle-to-shin remapping.
    if "RightAnkle" in sensors and "RightShin" not in sensors:
        sensors["RightShin"] = sensors.pop("RightAnkle")

    if "LeftAnkle" in sensors and "LeftShin" not in sensors:
        sensors["LeftShin"] = sensors.pop("LeftAnkle")

    return opal


def _sensor_to_dataframe(sensor: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Convert one Opal sensor dict to gaitmap/mobgap columns.

    Ensures all arrays are writable copies to avoid:
        buffer source array is read-only
    errors on some systems.
    """
    acc = np.array(sensor["acc"], dtype=float, copy=True)
    gyro = np.array(sensor["gyro"], dtype=float, copy=True) * 180.0 / np.pi
    mag = np.array(sensor["mag"], dtype=float, copy=True)
    orient = np.array(sensor["orient"], dtype=float, copy=True)

    n = min(acc.shape[1], gyro.shape[1], mag.shape[1], orient.shape[1])

    arr = np.vstack([
        acc[:, :n],
        gyro[:, :n],
        mag[:, :n],
        orient[:, :n],
    ]).T

    arr = np.array(arr, dtype=float, copy=True)

    df = pd.DataFrame(
        arr,
        columns=[
            "acc_x", "acc_y", "acc_z",
            "gyr_x", "gyr_y", "gyr_z",
            "mag_x", "mag_y", "mag_z",
            "q_x", "q_y", "q_z", "q_w",
        ],
    )

    return df.copy(deep=True)

def make_sensor_data_writable(dataset: SensorData) -> SensorData:
    """
    Force all arrays inside a SensorData object to be writable copies.

    Supports:
        - dict[str, DataFrame]
        - MultiIndex-column DataFrame
        - single DataFrame
    """
    if isinstance(dataset, dict):
        out = {}

        for sensor, df in dataset.items():
            out[sensor] = pd.DataFrame(
                np.array(df.to_numpy(dtype=float), dtype=float, copy=True),
                columns=df.columns,
                index=df.index,
            )

        return out

    if isinstance(dataset, pd.DataFrame):
        return pd.DataFrame(
            np.array(dataset.to_numpy(dtype=float), dtype=float, copy=True),
            columns=dataset.columns,
            index=dataset.index,
        )

    return dataset

def build_trial_dataframes_from_opal(opal: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build:
    - gaitmap-like foot data with MultiIndex columns:
        sensor_left / sensor_right
    - mobgap-like lumbar data:
        LowerBack DataFrame
    """
    sensors = opal["sensors"]

    required = ["Lumbar", "LeftShin", "RightShin"]
    missing = [name for name in required if name not in sensors]
    if missing:
        raise ValueError(f"Missing required sensors: {missing}. Available sensors: {list(sensors.keys())}")

    lower_back = _sensor_to_dataframe(sensors["Lumbar"])
    sensor_left = _sensor_to_dataframe(sensors["LeftShin"])
    sensor_right = _sensor_to_dataframe(sensors["RightShin"])

    common = min(len(lower_back), len(sensor_left), len(sensor_right))
    lower_back = lower_back.iloc[:common].reset_index(drop=True)
    sensor_left = sensor_left.iloc[:common].reset_index(drop=True)
    sensor_right = sensor_right.iloc[:common].reset_index(drop=True)

    gaitmap_data = pd.concat(
        {
            "sensor_left": sensor_left,
            "sensor_right": sensor_right,
        },
        axis=1,
    )

    return gaitmap_data, lower_back


# =============================================================================
# 4. CUSTOM SENSOR ALIGNMENT UTILITIES FROM YOUR SCRIPT
# =============================================================================

def _moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k is None or k <= 1:
        return x
    kernel = np.ones(int(k), dtype=float)
    kernel /= kernel.sum()
    return np.convolve(x, kernel, mode="same")


def _check_required_columns(df: pd.DataFrame):
    required = set((*SF_GYR, *SF_ACC))
    if not required.issubset(df.columns):
        missing = required.difference(df.columns)
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Expected SF_GYR={SF_GYR} and SF_ACC={SF_ACC}."
        )


def _find_local_extrema(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return indices of local maxima and minima using sign changes of the first difference.
    Max at i: dx[i-1] > 0 and dx[i] < 0.
    Min at i: dx[i-1] < 0 and dx[i] > 0.
    """
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    n = x.shape[0]
    if n < 3:
        return np.array([], dtype=int), np.array([], dtype=int)

    dx = np.diff(x)
    sign_prev = np.sign(dx[:-1])
    sign_next = np.sign(dx[1:])
    idx = np.arange(1, n - 1)
    maxima = idx[(sign_prev > 0) & (sign_next < 0)]
    minima = idx[(sign_prev < 0) & (sign_next > 0)]
    return maxima, minima


def _peak_prominence_mask(
    x: np.ndarray,
    idx: np.ndarray,
    half_window: int,
    min_prominence: float,
) -> np.ndarray:
    """
    Simple prominence filter:
    baseline as local median in [i-half_window, i+half_window].
    Keep peaks with |x[i]-baseline| >= min_prominence.
    """
    if idx.size == 0:
        return np.zeros(0, dtype=bool)

    n = x.shape[0]
    hw = max(1, int(half_window)) if half_window is not None else 1
    keep = np.zeros(idx.size, dtype=bool)

    for k, i in enumerate(idx):
        lo = max(0, i - hw)
        hi = min(n, i + hw + 1)
        baseline = np.median(x[lo:hi])
        prominence = abs(x[i] - baseline)
        keep[k] = prominence >= float(min_prominence)

    return keep


def _suppress_nearby_peaks(idx: np.ndarray, min_distance: int) -> np.ndarray:
    """Keep peaks at least min_distance samples apart using a greedy rule."""
    if idx.size == 0 or not min_distance or min_distance <= 1:
        return idx

    kept = []
    last = -10**9
    for i in idx:
        if i - last >= min_distance:
            kept.append(i)
            last = i

    return np.asarray(kept, dtype=int)


def _dominant_y_peak_polarity(
    y: np.ndarray,
    smoothing_kernel_samples: int = 5,
    min_prominence_deg_s: float = 3.0,
    half_window_for_prominence: int = 10,
    min_distance_samples: int = 0,
) -> str:
    """
    Determine whether positive or negative peaks dominate in y.
    Returns: 'positive', 'negative', or 'none'.
    """
    y_sm = _moving_average(y, smoothing_kernel_samples)

    pos_idx, neg_idx = _find_local_extrema(y_sm)

    pos_keep = _peak_prominence_mask(
        y_sm, pos_idx, half_window_for_prominence, min_prominence_deg_s
    )
    neg_keep = _peak_prominence_mask(
        y_sm, neg_idx, half_window_for_prominence, min_prominence_deg_s
    )

    pos_idx = pos_idx[pos_keep]
    neg_idx = neg_idx[neg_keep]

    pos_idx = _suppress_nearby_peaks(pos_idx, min_distance_samples)
    neg_idx = _suppress_nearby_peaks(neg_idx, min_distance_samples)

    pos_energy = float(np.sum(np.maximum(0.0, y_sm[pos_idx]))) if pos_idx.size else 0.0
    neg_energy = float(np.sum(np.maximum(0.0, -y_sm[neg_idx]))) if neg_idx.size else 0.0

    if pos_energy == 0.0 and neg_energy == 0.0:
        return "none"

    return "positive" if pos_energy >= neg_energy else "negative"


def _Rz_pi() -> np.ndarray:
    """Proper rotation that preserves Z and flips X and Y."""
    return np.diag([-1.0, -1.0, 1.0])


def _flip_y_only_improper(arr: np.ndarray) -> np.ndarray:
    """Flip only Y using an improper transform."""
    m = np.diag([1.0, -1.0, 1.0])
    return arr @ m.T


def _apply_flip_y(
    gyr: np.ndarray,
    acc: np.ndarray,
    mode: str = "preserve_z_rotation",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    mode:
    - preserve_z_rotation: rotate 180 degrees about Z; flips X and Y, keeps Z.
    - flip_y_only: multiply Y by -1 only.
    """
    mode = mode.lower()

    if mode == "preserve_z_rotation":
        rot = _Rz_pi()
        return gyr @ rot.T, acc @ rot.T

    if mode == "flip_y_only":
        return _flip_y_only_improper(gyr), _flip_y_only_improper(acc)

    raise ValueError("mode must be 'preserve_z_rotation' or 'flip_y_only'.")


def enforce_negative_y_peaks_single(
    df: pd.DataFrame,
    smoothing_kernel_samples: int = 5,
    min_prominence_deg_s: float = 3.0,
    half_window_for_prominence: int = 10,
    min_distance_samples: int = 0,
    flip_mode: str = "preserve_z_rotation",
) -> pd.DataFrame:
    """
    Ensure gyr_y peaks are predominantly negative.
    If positive peaks dominate, flip according to flip_mode.
    """
    _check_required_columns(df)
    out = df.copy()

    gyr = out[list(SF_GYR)].to_numpy()
    acc = out[list(SF_ACC)].to_numpy()
    gy = gyr[:, 1]

    polarity = _dominant_y_peak_polarity(
        gy,
        smoothing_kernel_samples=smoothing_kernel_samples,
        min_prominence_deg_s=min_prominence_deg_s,
        half_window_for_prominence=half_window_for_prominence,
        min_distance_samples=min_distance_samples,
    )

    if polarity == "positive":
        gyr_new, acc_new = _apply_flip_y(gyr, acc, mode=flip_mode)
        out.loc[:, list(SF_GYR)] = gyr_new
        out.loc[:, list(SF_ACC)] = acc_new

    return out


def enforce_negative_y_peaks(
    dataset: SensorData,
    smoothing_kernel_samples: int = 5,
    min_prominence_deg_s: float = 3.0,
    half_window_for_prominence: int = 10,
    min_distance_samples: int = 0,
    flip_mode: str = "preserve_z_rotation",
) -> SensorData:
    """
    Wrapper supporting:
    - Single DataFrame
    - Dict[str, DataFrame]
    - Row MultiIndex DataFrame
    - Column MultiIndex DataFrame
    """
    ds_type = is_sensor_data(dataset)

    if ds_type == "single":
        return enforce_negative_y_peaks_single(
            dataset,
            smoothing_kernel_samples,
            min_prominence_deg_s,
            half_window_for_prominence,
            min_distance_samples,
            flip_mode,
        )

    if isinstance(dataset, dict):
        return {
            name: enforce_negative_y_peaks_single(
                df,
                smoothing_kernel_samples,
                min_prominence_deg_s,
                half_window_for_prominence,
                min_distance_samples,
                flip_mode,
            )
            for name, df in dataset.items()
        }

    df = dataset.copy()

    if isinstance(df.index, pd.MultiIndex):
        out = df.copy()
        for sensor_name, block in df.groupby(level=0, sort=False):
            block_no_sensor = block.droplevel(level=0)
            new_block = enforce_negative_y_peaks_single(
                block_no_sensor,
                smoothing_kernel_samples,
                min_prominence_deg_s,
                half_window_for_prominence,
                min_distance_samples,
                flip_mode,
            )
            out.loc[(sensor_name, new_block.index), list(SF_GYR)] = new_block[list(SF_GYR)].values
            out.loc[(sensor_name, new_block.index), list(SF_ACC)] = new_block[list(SF_ACC)].values
        return out

    if isinstance(df.columns, pd.MultiIndex):
        sensors = sorted(set(df.columns.get_level_values(0)))
        out = df.copy()
        for sensor in sensors:
            sub = df.loc[:, sensor]
            new_sub = enforce_negative_y_peaks_single(
                sub,
                smoothing_kernel_samples,
                min_prominence_deg_s,
                half_window_for_prominence,
                min_distance_samples,
                flip_mode,
            )
            for sig in (*SF_GYR, *SF_ACC):
                out[(sensor, sig)] = new_sub[sig]
        return out

    raise TypeError(
        "Multi-sensor input detected, but DataFrame is neither row MultiIndex nor column MultiIndex."
    )

def enforce_negative_y_peaks_robust_single(
    df: pd.DataFrame,
    sample_mask: Optional[np.ndarray] = None,
    positive_percentile: float = 95,
    negative_percentile: float = 5,
    flip_ratio_threshold: float = 1.05,
    min_peak_amplitude_deg_s: float = 100,
    flip_mode: str = "preserve_z_rotation",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Robustly ensure that dominant gait-related gyr_y peaks are negative.

    This function compares high positive and high negative amplitudes in gyr_y.
    If positive peaks are clearly larger than negative peaks, the sensor is flipped.

    Parameters
    ----------
    df
        Single-sensor DataFrame with acc_x/y/z and gyr_x/y/z columns.
    sample_mask
        Optional boolean mask selecting samples used to decide orientation.
        This can be restricted to the walking part of the trial.
    positive_percentile
        Percentile used to estimate dominant positive peak amplitude.
    negative_percentile
        Percentile used to estimate dominant negative peak amplitude.
    flip_ratio_threshold
        Flip if positive_peak / negative_peak_abs exceeds this threshold.
    min_peak_amplitude_deg_s
        Do not flip if the signal is too small to make a reliable decision.
    flip_mode
        'preserve_z_rotation' is recommended and matches your existing logic.

    Returns
    -------
    corrected_df, qc
    """
    _check_required_columns(df)

    out = df.copy()

    y = out["gyr_y"].to_numpy(dtype=float)

    if sample_mask is not None:
        sample_mask = np.asarray(sample_mask, dtype=bool)
        y_eval = y[sample_mask]
    else:
        y_eval = y

    y_eval = y_eval[np.isfinite(y_eval)]

    qc = {
        "positive_peak_estimate": np.nan,
        "negative_peak_abs_estimate": np.nan,
        "peak_ratio_positive_to_negative": np.nan,
        "flipped": False,
        "reason": "",
    }

    if y_eval.size < 10:
        qc["reason"] = "too_few_samples"
        return out, qc

    positive_peak = np.nanpercentile(y_eval, positive_percentile)
    negative_peak_abs = abs(np.nanpercentile(y_eval, negative_percentile))

    qc["positive_peak_estimate"] = float(positive_peak)
    qc["negative_peak_abs_estimate"] = float(negative_peak_abs)

    if negative_peak_abs == 0:
        ratio = np.inf
    else:
        ratio = positive_peak / negative_peak_abs

    qc["peak_ratio_positive_to_negative"] = float(ratio)

    dominant_amplitude = max(abs(positive_peak), abs(negative_peak_abs))

    if dominant_amplitude < min_peak_amplitude_deg_s:
        qc["reason"] = "signal_amplitude_too_low"
        return out, qc

    # First rule: original percentile-based check
    flip_by_percentile = (
        positive_peak > 0
        and ratio > flip_ratio_threshold
    )
    
    # Second rule: morphology-based fallback
    # This catches cases where negative troughs are large but the repeated
    # gait-defining local maxima are still positive.
    n_large_positive_peaks, median_positive_peak_height = _count_large_positive_local_maxima(
        y_eval,
        smoothing_kernel_samples=5,
        min_peak_height_deg_s=180,
        min_distance_samples=64,
    )
    
    qc["n_large_positive_local_maxima"] = n_large_positive_peaks
    qc["median_large_positive_peak_height"] = median_positive_peak_height
    
    flip_by_positive_local_maxima = (
        n_large_positive_peaks >= 5
        and np.isfinite(median_positive_peak_height)
        and median_positive_peak_height >= 180
    )
    
    if flip_by_percentile or flip_by_positive_local_maxima:
        gyr = out[list(SF_GYR)].to_numpy()
        acc = out[list(SF_ACC)].to_numpy()
    
        gyr_new, acc_new = _apply_flip_y(
            gyr=gyr,
            acc=acc,
            mode=flip_mode,
        )
    
        out.loc[:, list(SF_GYR)] = gyr_new
        out.loc[:, list(SF_ACC)] = acc_new
    
        qc["flipped"] = True
    
        if flip_by_percentile:
            qc["reason"] = "positive_peaks_dominated_percentile"
        else:
            qc["reason"] = "large_positive_local_maxima_detected"
    
    else:
        qc["reason"] = "negative_peaks_ok_or_ambiguous"

    return out, qc

def enforce_negative_y_peaks_robust(
    dataset: SensorData,
    segments: Optional[List[Tuple[int, int]]] = None,
    positive_percentile: float = 95,
    negative_percentile: float = 5,
    flip_ratio_threshold: float = 1.05,
    min_peak_amplitude_deg_s: float = 100,
    flip_mode: str = "preserve_z_rotation",
) -> Tuple[SensorData, Dict[str, Dict[str, float]]]:
    """
    Apply robust negative gyr_y peak enforcement to each sensor.

    If segments are provided, only samples within those segments are used to
    decide whether to flip. The flip itself is applied to the full signal.
    """
    qc = {}

    # Build sample mask from straight-walking segments if available.
    sample_mask = None

    if segments is not None and len(segments) > 0:
        if isinstance(dataset, dict):
            n_samples = len(next(iter(dataset.values())))
        elif isinstance(dataset.columns, pd.MultiIndex):
            n_samples = len(dataset)
        else:
            n_samples = len(dataset)

        sample_mask = np.zeros(n_samples, dtype=bool)

        for start, end in segments:
            start = int(max(0, start))
            end = int(min(n_samples - 1, end))
            if end > start:
                sample_mask[start:end + 1] = True

    # Dict format
    if isinstance(dataset, dict):
        out = {}
        for sensor_name, df in dataset.items():
            corrected, sensor_qc = enforce_negative_y_peaks_robust_single(
                df,
                sample_mask=sample_mask,
                positive_percentile=positive_percentile,
                negative_percentile=negative_percentile,
                flip_ratio_threshold=flip_ratio_threshold,
                min_peak_amplitude_deg_s=min_peak_amplitude_deg_s,
                flip_mode=flip_mode,
            )
            out[sensor_name] = corrected
            qc[sensor_name] = sensor_qc

        return out, qc

    # Column MultiIndex format, which is what your gaitmap data usually uses.
    if isinstance(dataset, pd.DataFrame) and isinstance(dataset.columns, pd.MultiIndex):
        out = dataset.copy()
        sensors = sorted(set(dataset.columns.get_level_values(0)))

        for sensor_name in sensors:
            sub = dataset.loc[:, sensor_name]

            corrected, sensor_qc = enforce_negative_y_peaks_robust_single(
                sub,
                sample_mask=sample_mask,
                positive_percentile=positive_percentile,
                negative_percentile=negative_percentile,
                flip_ratio_threshold=flip_ratio_threshold,
                min_peak_amplitude_deg_s=min_peak_amplitude_deg_s,
                flip_mode=flip_mode,
            )

            for col in corrected.columns:
                out[(sensor_name, col)] = corrected[col].to_numpy()

            qc[sensor_name] = sensor_qc

        return out, qc

    # Single sensor fallback.
    corrected, sensor_qc = enforce_negative_y_peaks_robust_single(
        dataset,
        sample_mask=sample_mask,
        positive_percentile=positive_percentile,
        negative_percentile=negative_percentile,
        flip_ratio_threshold=flip_ratio_threshold,
        min_peak_amplitude_deg_s=min_peak_amplitude_deg_s,
        flip_mode=flip_mode,
    )

    qc["single_sensor"] = sensor_qc
    return corrected, qc

def get_combined_boundary_trimmed_s_ids_for_segment(
    event_list: Dict[str, pd.DataFrame],
    seg_start: int,
    seg_end: int,
    remove_boundary_strides: bool = True,
    number_boundary_strides: int = 1,
) -> Tuple[Dict[str, List[Any]], Dict[str, List[Any]], int, int]:
    """
    For one straight-walking segment, find valid left/right strides and apply
    combined chronological boundary trimming.

    Valid stride rule:
        ic >= seg_start and tc <= seg_end

    Boundary trimming:
        combine left and right valid strides, sort by IC, then remove
        number_boundary_strides from the beginning and from the end of the
        combined sequence.

    Returns
    -------
    valid_s_ids_before_trim_by_sensor
        Dict with valid s_ids before boundary trimming.

    valid_s_ids_after_trim_by_sensor
        Dict with valid s_ids after combined boundary trimming.

    n_combined_before_trim
        Number of valid combined left/right stride rows before trimming.

    n_combined_after_trim
        Number of valid combined left/right stride rows after trimming.
    """
    sensors = ["sensor_left", "sensor_right"]

    number_boundary_strides = int(max(0, number_boundary_strides))

    valid_s_ids_before_trim_by_sensor = {
        "sensor_left": [],
        "sensor_right": [],
    }

    combined_valid_rows = []

    for sensor in sensors:
        events = event_list[sensor]

        if events is None or events.empty:
            continue

        for s_id, row in events.iterrows():
            ic = row["ic"]
            tc = row["tc"]

            if ic >= seg_start and tc <= seg_end:
                valid_s_ids_before_trim_by_sensor[sensor].append(s_id)

                combined_valid_rows.append(
                    {
                        "sensor": sensor,
                        "s_id": s_id,
                        "ic": float(ic),
                        "tc": float(tc),
                    }
                )

    combined_valid_rows = sorted(
        combined_valid_rows,
        key=lambda x: x["ic"],
    )

    n_combined_before_trim = len(combined_valid_rows)

    if remove_boundary_strides and number_boundary_strides > 0:
        if n_combined_before_trim > 2 * number_boundary_strides:
            combined_after_trim = combined_valid_rows[
                number_boundary_strides:-number_boundary_strides
            ]
        else:
            combined_after_trim = []
    else:
        combined_after_trim = combined_valid_rows

    valid_s_ids_after_trim_by_sensor = {
        "sensor_left": [],
        "sensor_right": [],
    }

    for row in combined_after_trim:
        valid_s_ids_after_trim_by_sensor[row["sensor"]].append(row["s_id"])

    n_combined_after_trim = len(combined_after_trim)

    return (
        valid_s_ids_before_trim_by_sensor,
        valid_s_ids_after_trim_by_sensor,
        n_combined_before_trim,
        n_combined_after_trim,
    )


# =============================================================================
# 6. STRAIGHT-WALKING SEGMENT LOGIC
# =============================================================================

def compute_straight_segments_from_turns(
    turn_list: pd.DataFrame,
    n_samples: int,
    sampling_rate_hz: int,
    trial_start_s: float,
    trial_end_s: Optional[float],
) -> List[Tuple[int, int]]:
    """
    Match your current version:
        turns = turning_detector.turn_list_.sort_values('start')[['start', 'end']].to_numpy()
        trial_start = (3*sampling_rate_hz)
        trial_end = (123*sampling_rate_hz)

    If trial_end_s is None, use n_samples - trial_start.
    """
    trial_start = int(trial_start_s * sampling_rate_hz)

    if trial_end_s is None:
        trial_end = int(n_samples - trial_start)
    else:
        trial_end = int(trial_end_s * sampling_rate_hz)
        trial_end = min(trial_end, n_samples - 1)

    trial_start = max(0, min(trial_start, n_samples - 1))
    trial_end = max(trial_start + 1, min(trial_end, n_samples - 1))

    if turn_list is None or len(turn_list) == 0:
        return [(trial_start, trial_end)]

    turns = turn_list.sort_values("start")[["start", "end"]].to_numpy()

    segments: List[Tuple[int, int]] = []
    current_start = trial_start

    for start, end in turns:
        start = int(start)
        end = int(end)

        if end < trial_start or start > trial_end:
            continue

        start = max(start, trial_start)
        end = min(end, trial_end)

        if current_start < start:
            segments.append((current_start, start - 1))

        current_start = max(current_start, end + 1)

    if current_start < trial_end:
        segments.append((current_start, trial_end))

    return segments


def filter_outputs_for_segments(
    segments: List[Tuple[int, int]],
    event_list: Dict[str, pd.DataFrame],
    spatial_parameters: Dict[str, pd.DataFrame],
    temporal_parameters: Dict[str, pd.DataFrame],
    remove_boundary_strides: bool = True,
    number_boundary_strides: int = 1,
) -> List[Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Filter event, temporal, and spatial outputs to straight-walking segments.

    For each non-turn segment:
        valid_s_ids are strides where:
            ic >= segment_start and tc <= segment_end

    Boundary trimming:
        left and right valid strides are combined and sorted by IC timing.
        number_boundary_strides are removed from the beginning and from the
        end of this combined chronological sequence.

    This means NUMBER_BOUNDARY_STRIDES = 1 removes 2 total stride rows per
    straight segment, not 2 per sensor.
    """
    segment_outputs = []

    sensors = ["sensor_left", "sensor_right"]

    for bout_idx, (seg_start, seg_end) in enumerate(segments, start=1):
        filtered_events = {}
        filtered_spatial = {}
        filtered_temporal = {}
        counts = {}

        (
            valid_s_ids_before_trim_by_sensor,
            valid_s_ids_after_trim_by_sensor,
            n_combined_before_trim,
            n_combined_after_trim,
        ) = get_combined_boundary_trimmed_s_ids_for_segment(
            event_list=event_list,
            seg_start=seg_start,
            seg_end=seg_end,
            remove_boundary_strides=remove_boundary_strides,
            number_boundary_strides=number_boundary_strides,
        )

        for sensor in sensors:
            valid_s_ids_before_trim = valid_s_ids_before_trim_by_sensor[sensor]
            valid_s_ids_after_trim = valid_s_ids_after_trim_by_sensor[sensor]

            filtered_events[sensor] = _filter_by_s_ids(
                event_list[sensor],
                valid_s_ids_after_trim,
            )

            filtered_spatial[sensor] = _filter_by_s_ids(
                spatial_parameters[sensor],
                valid_s_ids_after_trim,
            )

            filtered_temporal[sensor] = _filter_by_s_ids(
                temporal_parameters[sensor],
                valid_s_ids_after_trim,
            )

            counts[sensor] = {
                "n_event_strides_before_boundary_trim": len(valid_s_ids_before_trim),
                "n_event_strides_after_boundary_trim": len(valid_s_ids_after_trim),
                "n_filtered_event_rows": len(filtered_events[sensor]),
                "n_filtered_temporal_rows": len(filtered_temporal[sensor]),
                "n_filtered_spatial_rows": len(filtered_spatial[sensor]),
            }

        segment_outputs.append(
            {
                "bout_id": bout_idx,
                "segment_start": seg_start,
                "segment_end": seg_end,
                "filtered_events": filtered_events,
                "filtered_spatial_paras": filtered_spatial,
                "filtered_temporal_paras": filtered_temporal,
                "counts": counts,
                "boundary_trim_mode": "combined_chronological",
                "number_boundary_strides": number_boundary_strides,
                "n_combined_event_strides_before_boundary_trim": n_combined_before_trim,
                "n_combined_event_strides_after_boundary_trim": n_combined_after_trim,
            }
        )

    return segment_outputs

# =============================================================================
# 7. MATLAB SPATIOTEMPORAL METRICS PORT
# =============================================================================

def _clean_event_vectors_like_matlab(
    lhs: np.ndarray,
    lto: np.ndarray,
    rhs: np.ndarray,
    rto: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Directly port the MATLAB event cleaning while-loops:

        while rto(1)<lhs(1); rto(1)=[]; end
        while lto(1)<rhs(1); lto(1)=[]; end
        while rhs(end)>lto(end); rhs(end)=[]; end
        while lhs(end)>rto(end); lhs(end)=[]; end
        while sum(lhs<rhs(1))>1 ; lhs(1)=[]; end
        while sum(rhs<lhs(1))>1 ; rhs(1)=[]; end
    """
    lhs = np.asarray(lhs, dtype=float).reshape(-1)
    lto = np.asarray(lto, dtype=float).reshape(-1)
    rhs = np.asarray(rhs, dtype=float).reshape(-1)
    rto = np.asarray(rto, dtype=float).reshape(-1)

    while len(rto) > 0 and len(lhs) > 0 and rto[0] < lhs[0]:
        rto = rto[1:]

    while len(lto) > 0 and len(rhs) > 0 and lto[0] < rhs[0]:
        lto = lto[1:]

    while len(rhs) > 0 and len(lto) > 0 and rhs[-1] > lto[-1]:
        rhs = rhs[:-1]

    while len(lhs) > 0 and len(rto) > 0 and lhs[-1] > rto[-1]:
        lhs = lhs[:-1]

    while len(lhs) > 0 and len(rhs) > 0 and np.sum(lhs < rhs[0]) > 1:
        lhs = lhs[1:]

    while len(rhs) > 0 and len(lhs) > 0 and np.sum(rhs < lhs[0]) > 1:
        rhs = rhs[1:]

    return lhs, lto, rhs, rto

def _append_array(accumulator: List[float], values: np.ndarray):
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size > 0:
        accumulator.extend(values.tolist())

def _compute_bout_metrics_like_matlab(
    events_left: pd.DataFrame,
    events_right: pd.DataFrame,
    temporal_left: pd.DataFrame,
    temporal_right: pd.DataFrame,
    spatial_left: pd.DataFrame,
    spatial_right: pd.DataFrame,
    sampling_rate_hz: int,
    events_already_matlab_cleaned: bool = False,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Compute MATLAB-style event-derived metrics for one straight-walking segment.

    A segment is only used for MATLAB-style processing if both left and right
    sides have at least 2 retained event rows after boundary trimming.
    """
    if events_left is None or events_right is None:
        return None

    if events_left.empty or events_right.empty:
        return None

    if len(events_left) < 2:
        return None

    if len(events_right) < 2:
        return None
    
    left_s_ids_ok, _ = _check_consecutive_s_ids(events_left)
    right_s_ids_ok, _ = _check_consecutive_s_ids(events_right)
    
    if not left_s_ids_ok or not right_s_ids_ok:
        return None

    lhs = events_left["ic"].to_numpy(dtype=float)
    lto = events_left["tc"].to_numpy(dtype=float)
    rhs = events_right["ic"].to_numpy(dtype=float)
    rto = events_right["tc"].to_numpy(dtype=float)

    if not (len(lhs) >= 2 and len(rto) >= 2 and len(rhs) >= 2 and len(lto) >= 2):
        return None
    
    if not events_already_matlab_cleaned:
        lhs, lto, rhs, rto = _clean_event_vectors_like_matlab(lhs, lto, rhs, rto)

    if not (len(lhs) >= 2 and len(rto) >= 2 and len(rhs) >= 2 and len(lto) >= 2):
        return None

    fs = sampling_rate_hz

    strtime_L = np.abs(np.diff(lhs)) / fs
    strtime_R = np.abs(np.diff(rhs)) / fs

    numberL = len(lhs)
    numberR = len(rhs)
    number = min(numberL, numberR)

    # Step time logic.
    if abs(lhs[0]) > abs(rhs[0]):
        steptime_R = np.array(
            [abs(lhs[k] - rhs[k]) / fs for k in range(number)],
            dtype=float,
        )
    else:
        steptime_R = np.array(
            [abs(lhs[k + 1] - rhs[k]) / fs for k in range(max(0, number - 1))],
            dtype=float,
        )

    if abs(lhs[0]) > abs(rhs[0]):
        steptime_L = np.array(
            [abs(rhs[l + 1] - lhs[l]) / fs for l in range(max(0, number - 1))],
            dtype=float,
        )
    else:
        steptime_L = np.array(
            [abs(rhs[l] - lhs[l]) / fs for l in range(number)],
            dtype=float,
        )

    # Stance time logic.
    if abs(lhs[0]) > abs(rhs[0]):
        Lstance = np.array(
            [abs(lto[a + 1] - lhs[a]) / fs for a in range(max(0, number - 1)) if a + 1 < len(lto)],
            dtype=float,
        )
    else:
        Lstance = np.array(
            [abs(lto[a] - lhs[a]) / fs for a in range(min(number, len(lto), len(lhs)))],
            dtype=float,
        )

    if abs(rhs[0]) > abs(lhs[0]):
        Rstance = np.array(
            [abs(rto[b + 1] - rhs[b]) / fs for b in range(max(0, number - 1)) if b + 1 < len(rto)],
            dtype=float,
        )
    else:
        Rstance = np.array(
            [abs(rto[b] - rhs[b]) / fs for b in range(min(number, len(rto), len(rhs)))],
            dtype=float,
        )

    # Percent stance.
    n = min(len(strtime_L), len(Lstance))
    perc_stanceTL = np.array(
        [(Lstance[c] / strtime_L[c]) * 100 for c in range(n)],
        dtype=float,
    )

    n = min(len(strtime_R), len(Rstance))
    perc_stanceTR = np.array(
        [(Rstance[c] / strtime_R[c]) * 100 for c in range(n)],
        dtype=float,
    )

    # Swing time logic.
    if abs(lhs[0]) < abs(rhs[0]):
        Lswing = np.array(
            [abs(lhs[m + 1] - lto[m]) / fs for m in range(max(0, number - 1)) if m + 1 < len(lhs) and m < len(lto)],
            dtype=float,
        )
    else:
        Lswing = np.array(
            [abs(lhs[m] - lto[m]) / fs for m in range(min(number, len(lhs), len(lto)))],
            dtype=float,
        )

    if abs(rhs[0]) < abs(lhs[0]):
        Rswing = np.array(
            [abs(rhs[nn + 1] - rto[nn]) / fs for nn in range(max(0, number - 1)) if nn + 1 < len(rhs) and nn < len(rto)],
            dtype=float,
        )
    else:
        Rswing = np.array(
            [abs(rhs[nn] - rto[nn]) / fs for nn in range(min(number, len(rhs), len(rto)))],
            dtype=float,
        )

    # Percent swing.
    n = min(len(strtime_L), len(Lswing))
    perc_swingTL = np.array(
        [(Lswing[o] / strtime_L[o]) * 100 for o in range(n)],
        dtype=float,
    )

    n = min(len(strtime_R), len(Rswing))
    perc_swingTR = np.array(
        [(Rswing[o] / strtime_R[o]) * 100 for o in range(n)],
        dtype=float,
    )

    # Double support.
    n_ds = min(number, len(lto), len(rhs), len(rto), len(lhs))
    DoubleSupportR = np.array(
        [abs(lto[p] - rhs[p]) / fs for p in range(n_ds)],
        dtype=float,
    )
    DoubleSupportL = np.array(
        [abs(rto[q] - lhs[q]) / fs for q in range(n_ds)],
        dtype=float,
    )

    n = min(len(strtime_L), len(DoubleSupportL))
    perc_dsL = np.array(
        [(DoubleSupportL[r] / strtime_L[r]) * 100 for r in range(n)],
        dtype=float,
    )

    n = min(len(strtime_R), len(DoubleSupportR))
    perc_dsR = np.array(
        [(DoubleSupportR[r] / strtime_R[r]) * 100 for r in range(n)],
        dtype=float,
    )

    # Cadence conditional logic.
    cadence = np.nan
    if numberL > numberR:
        denom = (lhs[numberL - 1] - lhs[0]) / fs
        cadence = ((numberR + numberL) / denom) * 60 if denom != 0 else np.nan
    elif numberR > numberL:
        denom = (rhs[numberR - 1] - rhs[0]) / fs
        cadence = ((numberR + numberL) / denom) * 60 if denom != 0 else np.nan
    elif numberR == numberL and abs(lhs[0]) < abs(rhs[0]):
        denom = (rhs[numberR - 1] - lhs[0]) / fs
        cadence = ((numberR + numberL) / denom) * 60 if denom != 0 else np.nan
    elif numberR == numberL and abs(lhs[0]) > abs(rhs[0]):
        denom = (lhs[numberR - 1] - rhs[0]) / fs
        cadence = ((numberR + numberL) / denom) * 60 if denom != 0 else np.nan

    # GA conditional logic.
    if len(Lswing) > len(Rswing):
        ga_len = len(Rswing)
    else:
        ga_len = len(Lswing)

    GA = np.full(ga_len, np.nan, dtype=float)
    for w in range(ga_len):
        if Lswing[w] > Rswing[w]:
            if Lswing[w] != 0:
                GA[w] = 100 * abs(math.log(Rswing[w] / Lswing[w]))
        else:
            if Rswing[w] != 0:
                GA[w] = 100 * abs(math.log(Lswing[w] / Rswing[w]))

    # PCI phi conditional logic.
    if lhs[0] > rhs[0]:
        phi_len = max(0, min(len(rhs) - 1, len(lhs)))
        phi = np.full(phi_len, np.nan, dtype=float)
        for y in range(phi_len):
            denom = rhs[y + 1] - rhs[y]
            phi[y] = 360 * (lhs[y] - rhs[y]) / denom if denom != 0 else np.nan
    else:
        phi_len = max(0, min(len(lhs) - 1, len(rhs)))
        phi = np.full(phi_len, np.nan, dtype=float)
        for y in range(phi_len):
            denom = lhs[y + 1] - lhs[y]
            phi[y] = 360 * (rhs[y] - lhs[y]) / denom if denom != 0 else np.nan

    
    gm_gaitspeed_L = _extract_existing_column(
        spatial_left,
        ["gait_velocity", "gait velocity [m/s]"])
    gm_gaitspeed_R = _extract_existing_column(
        spatial_right,
        ["gait_velocity", "gait velocity [m/s]"],)
    gait_speed_ms = _nanmean(np.concatenate([gm_gaitspeed_L, gm_gaitspeed_R]))


    # GSR conditional logic from MATLAB.
    GSR = np.nan
    if not np.isnan(gait_speed_ms) and gait_speed_ms != 0:
        if numberL > numberR:
            denom = (lhs[numberL - 1] - lhs[0]) / fs
            GSR = (((numberL + numberR) / 2) / denom) / gait_speed_ms if denom != 0 else np.nan
        elif numberR > numberL:
            # MATLAB line used rhs(numberL,1), which is likely a typo but we preserve logic safely.
            idx = min(numberL, len(rhs)) - 1
            denom = (rhs[idx] - rhs[0]) / fs
            GSR = (((numberL + numberR) / 2) / denom) / gait_speed_ms if denom != 0 else np.nan
        elif numberR == numberL and abs(lhs[0]) < abs(rhs[0]):
            denom = (rhs[numberL - 1] - lhs[0]) / fs
            GSR = (((numberL + numberR) / 2) / denom) / gait_speed_ms if denom != 0 else np.nan
        elif numberR == numberL and abs(lhs[0]) > abs(rhs[0]):
            denom = (lhs[numberL - 1] - rhs[0]) / fs
            GSR = (((numberL + numberR) / 2) / denom) / gait_speed_ms if denom != 0 else np.nan

    return {
    "strtime_L": strtime_L,
    "strtime_R": strtime_R,
    "steptime_L": steptime_L,
    "steptime_R": steptime_R,
    "stancetime_L": Lstance,
    "stancetime_R": Rstance,
    "swingtime_L": Lswing,
    "swingtime_R": Rswing,
    "dsuptime_L": DoubleSupportL,
    "dsuptime_R": DoubleSupportR,
    "perc_stancetime_L": perc_stanceTL,
    "perc_stancetime_R": perc_stanceTR,
    "perc_swingtime_L": perc_swingTL,
    "perc_swingtime_R": perc_swingTR,
    "perc_dsuptime_L": perc_dsL,
    "perc_dsuptime_R": perc_dsR,
    "cadence": np.array([cadence], dtype=float),
    "GA": GA,
    "GSR": np.array([GSR], dtype=float),
    "phi": phi,
    }

def _append_gaitmap_metrics_from_filtered_outputs(
    accum: Dict[str, List[float]],
    temporal_left: pd.DataFrame,
    temporal_right: pd.DataFrame,
    spatial_left: pd.DataFrame,
    spatial_right: pd.DataFrame,
):
    """
    Append gaitmap temporal/spatial parameters from filtered rows directly.

    This is independent of whether a straight segment has enough bilateral
    events for MATLAB-style event pairing.
    """
    gm_strtime_L = _extract_existing_column(temporal_left, ["stride_time", "stride time [s]"])
    gm_strtime_R = _extract_existing_column(temporal_right, ["stride_time", "stride time [s]"])

    gm_stancetime_L = _extract_existing_column(temporal_left, ["stance_time", "stance time [s]"])
    gm_stancetime_R = _extract_existing_column(temporal_right, ["stance_time", "stance time [s]"])

    gm_swingtime_L = _extract_existing_column(temporal_left, ["swing_time", "swing time [s]"])
    gm_swingtime_R = _extract_existing_column(temporal_right, ["swing_time", "swing time [s]"])

    gm_strlen_L = _extract_existing_column(spatial_left, ["stride_length", "stride length [m]"])
    gm_strlen_R = _extract_existing_column(spatial_right, ["stride_length", "stride length [m]"])

    gm_gaitspeed_L = _extract_existing_column(spatial_left, ["gait_velocity", "gait velocity [m/s]"])
    gm_gaitspeed_R = _extract_existing_column(spatial_right, ["gait_velocity", "gait velocity [m/s]"])

    gm_lat_exc_L = _extract_existing_column(spatial_left, ["max_lateral_excursion", "max lateral excursion [m]"])
    gm_lat_exc_R = _extract_existing_column(spatial_right, ["max_lateral_excursion", "max lateral excursion [m]"])

    gm_sens_lift_L = _extract_existing_column(spatial_left, ["max_sensor_lift", "max sensor lift [m]"])
    gm_sens_lift_R = _extract_existing_column(spatial_right, ["max_sensor_lift", "max sensor lift [m]"])

    gm_angle_ic_L = _extract_existing_column(spatial_left, ["ic_angle", "ic angle [deg]"])
    gm_angle_ic_R = _extract_existing_column(spatial_right, ["ic_angle", "ic angle [deg]"])

    gm_angle_tc_L = _extract_existing_column(spatial_left, ["tc_angle", "tc angle [deg]"])
    gm_angle_tc_R = _extract_existing_column(spatial_right, ["tc_angle", "tc angle [deg]"])

    gm_arclen_L = _extract_existing_column(spatial_left, ["arc_length", "arc length [m]"])
    gm_arclen_R = _extract_existing_column(spatial_right, ["arc_length", "arc length [m]"])

    _append_array(accum["gm_strtime_L"], gm_strtime_L)
    _append_array(accum["gm_strtime_R"], gm_strtime_R)

    _append_array(accum["gm_stancetime_L"], gm_stancetime_L)
    _append_array(accum["gm_stancetime_R"], gm_stancetime_R)

    _append_array(accum["gm_swingtime_L"], gm_swingtime_L)
    _append_array(accum["gm_swingtime_R"], gm_swingtime_R)

    _append_array(accum["gm_strlen_L"], gm_strlen_L)
    _append_array(accum["gm_strlen_R"], gm_strlen_R)

    _append_array(accum["gm_gaitspeed_L"], gm_gaitspeed_L)
    _append_array(accum["gm_gaitspeed_R"], gm_gaitspeed_R)

    _append_array(accum["gm_lat_exc_L"], gm_lat_exc_L)
    _append_array(accum["gm_lat_exc_R"], gm_lat_exc_R)

    _append_array(accum["gm_sens_lift_L"], gm_sens_lift_L)
    _append_array(accum["gm_sens_lift_R"], gm_sens_lift_R)

    _append_array(accum["gm_angle_ic_L"], gm_angle_ic_L)
    _append_array(accum["gm_angle_ic_R"], gm_angle_ic_R)

    _append_array(accum["gm_angle_tc_L"], gm_angle_tc_L)
    _append_array(accum["gm_angle_tc_R"], gm_angle_tc_R)

    _append_array(accum["gm_arclen_L"], gm_arclen_L)
    _append_array(accum["gm_arclen_R"], gm_arclen_R)

def summarize_filtered_segments_like_matlab(
    segment_outputs: List[Dict[str, Any]],
    sampling_rate_hz: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pool all straight-walking bouts per trial 

    Returns:
        summary_df: one-row DataFrame with summary variables.
        bout_level_df: optional useful QC table per retained straight segment.
    """
    
    gaitmap_accum = {
        "gm_strtime_L": [],
        "gm_strtime_R": [],
        "gm_stancetime_L": [],
        "gm_stancetime_R": [],
        "gm_swingtime_L": [],
        "gm_swingtime_R": [],
        "gm_strlen_L": [],
        "gm_strlen_R": [],
        "gm_gaitspeed_L": [],
        "gm_gaitspeed_R": [],
        "gm_lat_exc_L": [],
        "gm_lat_exc_R": [],
        "gm_sens_lift_L": [],
        "gm_sens_lift_R": [],
        "gm_angle_ic_L": [],
        "gm_angle_ic_R": [],
        "gm_angle_tc_L": [],
        "gm_angle_tc_R": [],
        "gm_arclen_L": [],
        "gm_arclen_R": [],
    }
    
    # ------------------------------------------------------------
    # First: pool gaitmap parameters from all retained rows after
    # boundary trimming, independent of MATLAB bilateral validity.
    # ------------------------------------------------------------
    for bout in segment_outputs:
        _append_gaitmap_metrics_from_filtered_outputs(
            accum=gaitmap_accum,
            temporal_left=bout["filtered_temporal_paras"]["sensor_left"],
            temporal_right=bout["filtered_temporal_paras"]["sensor_right"],
            spatial_left=bout["filtered_spatial_paras"]["sensor_left"],
            spatial_right=bout["filtered_spatial_paras"]["sensor_right"],
        )
            
    bout_rows = []

    for bout in segment_outputs:

        counts = bout.get("counts", {})

        left_counts = counts.get("sensor_left", {})
        right_counts = counts.get("sensor_right", {})
                
        bout_rows.append(
            {
                "bout_id": bout["bout_id"],
                "segment_start": bout["segment_start"],
                "segment_end": bout["segment_end"],
        
                "n_left_filtered_temporal_rows": left_counts.get(
                    "n_filtered_temporal_rows", np.nan
                ),
                "n_right_filtered_temporal_rows": right_counts.get(
                    "n_filtered_temporal_rows", np.nan
                ),
        
                "n_left_filtered_spatial_rows": left_counts.get(
                    "n_filtered_spatial_rows", np.nan
                ),
                "n_right_filtered_spatial_rows": right_counts.get(
                    "n_filtered_spatial_rows", np.nan
                ),
            }
        )     
    
    bout_level_df = pd.DataFrame(bout_rows)
    result: Dict[str, float] = {}
    if not bout_level_df.empty:
        result["n_bouts_total"] = len(bout_level_df)
    else:
        result["n_bouts_total"] = 0
    
    # gaitmap temporal.
    result.update(_summarize_lr_exact(np.array(gaitmap_accum["gm_strtime_L"]), np.array(gaitmap_accum["gm_strtime_R"]), "gm_strtime"))
    result.update(_summarize_lr_exact(np.array(gaitmap_accum["gm_stancetime_L"]), np.array(gaitmap_accum["gm_stancetime_R"]), "gm_stancetime"))
    result.update(_summarize_lr_exact(np.array(gaitmap_accum["gm_swingtime_L"]), np.array(gaitmap_accum["gm_swingtime_R"]), "gm_swingtime"))

    # gaitmap spatial.
    result.update(_summarize_lr_exact(np.array(gaitmap_accum["gm_strlen_L"]), np.array(gaitmap_accum["gm_strlen_R"]), "gm_strlen"))
    result.update(_summarize_lr_exact(np.array(gaitmap_accum["gm_gaitspeed_L"]), np.array(gaitmap_accum["gm_gaitspeed_R"]), "gm_gaitspeed"))
    result.update(_summarize_lr_exact(np.array(gaitmap_accum["gm_lat_exc_L"]), np.array(gaitmap_accum["gm_lat_exc_R"]), "gm_lat_exc"))
    result.update(_summarize_lr_exact(np.array(gaitmap_accum["gm_sens_lift_L"]), np.array(gaitmap_accum["gm_sens_lift_R"]), "gm_sens_lift"))
    result.update(_summarize_lr_exact(np.array(gaitmap_accum["gm_angle_ic_L"]), np.array(gaitmap_accum["gm_angle_ic_R"]), "gm_angle_ic"))

    # used abs denominator and abs mean for gm_angle_tc CV.
    result.update(
        _summarize_lr_exact(
            np.array(gaitmap_accum["gm_angle_tc_L"]),
            np.array(gaitmap_accum["gm_angle_tc_R"]),
            "gm_angle_tc",
            use_abs_cv=True,
            use_abs_asym_denominator=True,
        )
    )

    result.update(_summarize_lr_exact(np.array(gaitmap_accum["gm_arclen_L"]), np.array(gaitmap_accum["gm_arclen_R"]), "gm_arclen"))


    
    summary_df = pd.DataFrame([result])

    return summary_df, bout_level_df

# =============================================================================
# 8. TURN AND FOG PROCESSING
# =============================================================================

def analyze_360_turns_and_fog_metrics(
    lumbar_body_frame: pd.DataFrame,
    shank_right_df: pd.DataFrame,
    shank_left_df: pd.DataFrame,
    turn_list: pd.DataFrame,
    sample_rate: int,
) -> Dict[str, float]:
    """
    Python translation of Analyze_Turns.

    This function calculates:
    - turn metrics from lumbar/sacrum angular velocity
    - freezing/time-frequency metrics from right and left shank acceleration

    Input expectations
    ------------------
    lumbar_body_frame:
        Body-frame lumbar dataframe from mobgap.to_body_frame(lumbar_data).
        Expected to contain 'gyr_is'. This is used as Sacrum row 4.

    shank_right_df, shank_left_df:
        Reoriented shank sensor dataframes.
        Expected acceleration columns: acc_x, acc_y, acc_z.

        Mapping follows MATLAB:
            ShankR(1,:) = vertical acceleration
            ShankR(2,:) = ML acceleration
            ShankR(3,:) = AP acceleration

        In this implementation:
            vertical = acc_x
            ML       = acc_y
            AP       = acc_z

        If your axis mapping differs, change the three lines where
        shank arrays are constructed below.

    turn_list:
        turning_detector.turn_list_ with 'start' and 'end' columns.

    sample_rate:
        Original sampling rate, e.g. 128 Hz.

    Returns
    -------
    Dictionary with fields named like MATLAB result struct.
    """
    resS = {}

    # ---------------------------------------------------------------------
    # Construct MATLAB-like matrices
    # ---------------------------------------------------------------------
    # MATLAB:
    #   Sacrum(4,:) is angular velocity used for turns
    #
    # Python:
    #   use lumbar body-frame gyr_is as the fourth row.
    if "gyr_is" not in lumbar_body_frame.columns:
        raise ValueError(
            "lumbar_body_frame must contain 'gyr_is'. "
            "Check output of mobgap.utils.conversions.to_body_frame."
        )

    n_samples = len(lumbar_body_frame)

    sacrum_gyr_is = lumbar_body_frame["gyr_is"].to_numpy(dtype=float)

    # Placeholder rows for Sacrum(1:3,:) because this MATLAB function only
    # uses Sacrum(2,:) for jerk and Sacrum(4,:) for turn velocity.
    # For Sacrum(2,:), use lumbar ML acceleration if available.
    if "acc_ml" in lumbar_body_frame.columns:
        sacrum_acc_ml = lumbar_body_frame["acc_ml"].to_numpy(dtype=float)
    elif "acc_y" in lumbar_body_frame.columns:
        sacrum_acc_ml = lumbar_body_frame["acc_y"].to_numpy(dtype=float)
    else:
        sacrum_acc_ml = np.zeros(n_samples, dtype=float)

    Sacrum = np.vstack([
        np.zeros(n_samples),
        sacrum_acc_ml,
        np.zeros(n_samples),
        sacrum_gyr_is,
    ])

    # MATLAB:
    #   ShankR(1,:) vertical
    #   ShankR(2,:) ML
    #   ShankR(3,:) AP
    #
    # Current mapping:
    #   acc_x -> vertical
    #   acc_y -> ML
    #   acc_z -> AP
    #
    # Adjust here if your alignment convention differs.
    ShankR = np.vstack([
        shank_right_df["acc_x"].to_numpy(dtype=float),
        shank_right_df["acc_y"].to_numpy(dtype=float),
        shank_right_df["acc_z"].to_numpy(dtype=float),
    ])

    ShankL = np.vstack([
        shank_left_df["acc_x"].to_numpy(dtype=float),
        shank_left_df["acc_y"].to_numpy(dtype=float),
        shank_left_df["acc_z"].to_numpy(dtype=float),
    ])


    # ---------------------------------------------------------------------
    # Turn beginning and ending
    # ---------------------------------------------------------------------
    if turn_list is not None and len(turn_list) > 0:
        beginning = turn_list["start"].to_numpy(dtype=int)
        ending = turn_list["end"].to_numpy(dtype=int)
    else:
        beginning = np.array([], dtype=int)
        ending = np.array([], dtype=int)

    # MATLAB removes everything from the first too-short turn onward.
    if beginning.size > 0:
        p_beginning = beginning.copy()
        p_ending = ending.copy()
        pduration = (p_ending - p_beginning) / sample_rate

        for ff in range(len(pduration)):
            if pduration[ff] < 0.1:
                beginning = p_beginning[:ff]
                ending = p_ending[:ff]
                break
            else:
                beginning = p_beginning
                ending = p_ending

    # ---------------------------------------------------------------------
    # Turn metrics
    # ---------------------------------------------------------------------
    if beginning.size > 0:
        resS["TOTturns"] = len(beginning)

        duration = (ending - beginning) / sample_rate

        resS["AVEduration"] = float(np.mean(duration))
        resS["CVduration"] = _cv_matlab(duration)
        resS["MEDduration"] = float(np.median(duration))

        # Peak speed
        peakspeed = []
        for j in range(len(beginning)):
            start = int(max(0, beginning[j]))
            end = int(min(n_samples - 1, ending[j]))
            peakspeed.append(np.max(np.abs(Sacrum[3, start:end + 1])))

        peakspeed = np.asarray(peakspeed, dtype=float)

        resS["AVEpeakspeed"] = float(np.mean(peakspeed))
        resS["CVpeakspeed"] = _cv_matlab(peakspeed)
        resS["MEDpeakspeed"] = float(np.median(peakspeed))

        # Mean speed
        meanspeed = []
        for j in range(len(beginning)):
            start = int(max(0, beginning[j]))
            end = int(min(n_samples - 1, ending[j]))
            meanspeed.append(np.mean(np.abs(Sacrum[3, start:end + 1])))

        meanspeed = np.asarray(meanspeed, dtype=float)

        resS["AVEmeanspeed"] = float(np.mean(meanspeed))
        resS["CVmeanspeed"] = _cv_matlab(meanspeed)
        resS["MEDmeanspeed"] = float(np.median(meanspeed))

        # Turn angle
        angle = []
        for j in range(len(beginning)):
            start = int(max(0, beginning[j]))
            end = int(min(n_samples - 1, ending[j]))
            angle.append(np.trapezoid(np.abs(Sacrum[3, start:end + 1])) / sample_rate)

        angle = np.asarray(angle, dtype=float)

        resS["AVEangle"] = float(np.mean(angle))
        resS["CVangle"] = _cv_matlab(angle)
        resS["MEDangle"] = float(np.median(angle))
        resS["MAXangle"] = float(np.max(angle))
        resS["MINangle"] = float(np.min(angle))

        # ML jerkiness
        jerk = []
        for j in range(len(beginning)):
            start = int(max(0, beginning[j]))
            end = int(min(n_samples - 1, ending[j]))

            Ajerk = (_cdiff(Sacrum[1, start:end + 1]) ** 2) * (sample_rate / 2)
            jerk.append(np.sqrt(0.5 * np.trapezoid(Ajerk / sample_rate)))

        jerk = np.asarray(jerk, dtype=float)

        resS["AVEjerk"] = float(np.mean(jerk))
        resS["CVjerk"] = _cv_matlab(jerk)
        resS["MEDjerk"] = float(np.median(jerk))
        resS["MAXjerk"] = float(np.max(jerk))
        resS["MINjerk"] = float(np.min(jerk))

    else:
        # Match expectation: still calculate freezing metrics even if no turns.
        resS["TOTturns"] = 0
        resS["Rturn"] = 0
        resS["Lturn"] = 0

    # ---------------------------------------------------------------------
    # Percentage time frozen
    # MATLAB:
    #   fc2 = 200
    #   x = resample(ShankR(3,:),fc2,sampleRate)
    #   y = resample(ShankL(3,:),fc2,sampleRate)
    # ---------------------------------------------------------------------
    fc2 = 200

    x = _resample_matlab_style(ShankR[2, :], fc2, sample_rate)
    y = _resample_matlab_style(ShankL[2, :], fc2, sample_rate)

    starts = np.arange(0, len(x) - fc2 - 1, fc2, dtype=int)

    Ratio_x = []
    Ratio_y = []

    for k in range(max(0, len(starts) - 1)):
        seg_x = x[starts[k]:starts[k + 1]]
        seg_y = y[starts[k]:starts[k + 1]]

        Pxx, _ = _pwelch_matlab_style(seg_x, fc2, fc2 * 10, fc2)
        Pxx_x = _normalise_psd(Pxx)

        Pyy, Fxx = _pwelch_matlab_style(seg_y, fc2, fc2 * 10, fc2)
        Pxx_y = _normalise_psd(Pyy)

        LF = _freq_idx(Fxx, 3)
        HF = _freq_idx(Fxx, 8)
        LLF = _freq_idx(Fxx, 0.5)

        Ratio_x.append(_safe_band_ratio(Pxx_x, LF, HF, LLF, LF))
        Ratio_y.append(_safe_band_ratio(Pxx_y, LF, HF, LLF, LF))

    Ratio_x = np.asarray(Ratio_x, dtype=float)
    Ratio_y = np.asarray(Ratio_y, dtype=float)

    if Ratio_x.size > 0:
        percF = ((Ratio_x > 2.5) | (Ratio_y > 2.5)).astype(int)
        resS["FoGtime"] = float((100 * np.sum(percF == 1)) / len(percF))
    else:
        resS["FoGtime"] = np.nan

    # ---------------------------------------------------------------------
    # Freezing ratios
    # MATLAB:
    #   resample to 50 Hz
    # ---------------------------------------------------------------------
    fs_fog = 50

    acc_R_ap = _resample_matlab_style(ShankR[2, :], fs_fog, sample_rate)
    acc_R_ml = _resample_matlab_style(ShankR[1, :], fs_fog, sample_rate)
    acc_R_v = _resample_matlab_style(ShankR[0, :], fs_fog, sample_rate)

    acc_L_ap = _resample_matlab_style(ShankL[2, :], fs_fog, sample_rate)
    acc_L_ml = _resample_matlab_style(ShankL[1, :], fs_fog, sample_rate)
    acc_L_v = _resample_matlab_style(ShankL[0, :], fs_fog, sample_rate)

    nperseg = 100
    nfft = fs_fog * 10

    PrA, _ = _pwelch_matlab_style(acc_R_ap, nperseg, nfft, fs_fog)
    PrM, _ = _pwelch_matlab_style(acc_R_ml, nperseg, nfft, fs_fog)
    PrV, _ = _pwelch_matlab_style(acc_R_v, nperseg, nfft, fs_fog)

    PlA, _ = _pwelch_matlab_style(acc_L_ap, nperseg, nfft, fs_fog)
    PlM, _ = _pwelch_matlab_style(acc_L_ml, nperseg, nfft, fs_fog)
    PlV, F = _pwelch_matlab_style(acc_L_v, nperseg, nfft, fs_fog)

    PrA_N = _normalise_psd(PrA)
    PrM_N = _normalise_psd(PrM)
    PrV_N = _normalise_psd(PrV)

    PlA_N = _normalise_psd(PlA)
    PlM_N = _normalise_psd(PlM)
    PlV_N = _normalise_psd(PlV)

    ii = _freq_idx(F, 3)
    ii1 = _freq_idx(F, 8)
    ii2 = _freq_idx(F, 0.5)

    # Original ratios: 3-8 Hz divided by 0.5-3 Hz
    resS["Ratio_rA"] = _safe_band_ratio(PrA_N, ii, ii1, ii2, ii)
    resS["Ratio_rM"] = _safe_band_ratio(PrM_N, ii, ii1, ii2, ii)
    resS["Ratio_rV"] = _safe_band_ratio(PrV_N, ii, ii1, ii2, ii)

    resS["Ratio_lA"] = _safe_band_ratio(PlA_N, ii, ii1, ii2, ii)
    resS["Ratio_lM"] = _safe_band_ratio(PlM_N, ii, ii1, ii2, ii)
    resS["Ratio_lV"] = _safe_band_ratio(PlV_N, ii, ii1, ii2, ii)

    # Mean and max from right and left leg
    resS["FOGRatio_A_mean"] = float(np.mean([resS["Ratio_rA"], resS["Ratio_lA"]]))
    resS["FOGRatio_A_max"] = float(np.max([resS["Ratio_rA"], resS["Ratio_lA"]]))

    resS["FOGRatio_V_mean"] = float(np.mean([resS["Ratio_rV"], resS["Ratio_lV"]]))
    resS["FOGRatio_V_max"] = float(np.max([resS["Ratio_rV"], resS["Ratio_lV"]]))

    resS["FOGRatio_M_mean"] = float(np.mean([resS["Ratio_rM"], resS["Ratio_lM"]]))
    resS["FOGRatio_M_max"] = float(np.max([resS["Ratio_rM"], resS["Ratio_lM"]]))

    return resS

# =============================================================================
# 9. PER-FILE PROCESSING
# =============================================================================

def process_one_h5_file(h5_file: Path, root: Path, output_root: Path) -> Optional[Path]:
    """
    Process one .h5 file and write per-trial Excel.
    """
    h5_file = Path(h5_file)

    relative_parent = h5_file.parent.relative_to(root)
    output_dir = output_root / relative_parent
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = h5_file.stem
    out_excel = output_dir / f"{base_name}_summary.xlsx"

    if out_excel.exists():
        print(f"Skipping existing output: {out_excel}")
        return out_excel

    external_json_path = h5_file.with_suffix(EXTERNAL_EVENT_JSON_SUFFIX)
    if not external_json_path.exists():
        print(f"Skipping {h5_file.name}: no external event JSON found ({external_json_path.name}).")
        return None

    print(f"Processing: {h5_file}")

    opal = read_opal5_data(h5_file)
    gaitmap_data, lumbar_data = build_trial_dataframes_from_opal(opal)
    gaitmap_data=make_sensor_data_writable(gaitmap_data)
    lumbar_data=lumbar_data.copy(deep=True)

    # Original foot sensor rotations.
    rot_left = rotation_from_angle(np.array([1, 0, 0]), np.deg2rad(-90)) * rotation_from_angle(
        np.array([0, 0, 1]), np.deg2rad(90)
    )

    rot_right = rotation_from_angle(np.array([1, 0, 0]), np.deg2rad(90)) * rotation_from_angle(
        np.array([0, 0, 1]), np.deg2rad(-90)
    )

    rotations = {
        "sensor_left": rot_left,
        "sensor_right": rot_right,
    }

    dataset_sf = flip_dataset(gaitmap_data, rotations)
    dataset_sf = make_sensor_data_writable(dataset_sf)

    dataset_sf_aligned_to_gravity, _ = sensor_alignment.align_dataset_to_gravity_min_vel(
        dataset_sf,
        SAMPLING_RATE_HZ,
        gyr_min_vel_th_deg_s=7,
    )
    dataset_sf_aligned_to_gravity=make_sensor_data_writable(dataset_sf_aligned_to_gravity)
    
    dataset_sf_fixed = enforce_negative_y_peaks(
        dataset=dataset_sf_aligned_to_gravity,
        smoothing_kernel_samples=5,
        min_prominence_deg_s=200,
        half_window_for_prominence=20,
        min_distance_samples=128,
        flip_mode="preserve_z_rotation",
    )

    # Turning detection only, using lumbar data.
    imu_data = to_body_frame(lumbar_data)

    turning_detector = TdElGohary(
        smoothing_filter=ButterworthFilter(
            order=4,
            cutoff_freq_hz=TURN_SMOOTHING_CUTOFF_HZ,
            filter_type="lowpass",
            zero_phase=True,
        ),
        min_peak_angle_velocity_dps=TURN_MIN_PEAK_ANGLE_VELOCITY_DPS,
        lower_threshold_velocity_dps=TURN_LOWER_THRESHOLD_VELOCITY_DPS,
        min_gap_between_turns_s=TURN_MIN_GAP_BETWEEN_TURNS_S,
        allowed_turn_duration_s=TURN_ALLOWED_TURN_DURATION_S,
        allowed_turn_angle_deg=TURN_ALLOWED_TURN_ANGLE_DEG,
    )
        
    turning_detector.detect(imu_data, sampling_rate_hz=SAMPLING_RATE_HZ)

    # Straight walking segments: whole trial minus turns.
    n_samples = len(dataset_sf_fixed["sensor_left"])
    segments = compute_straight_segments_from_turns(
        turn_list=turning_detector.turn_list_,
        n_samples=n_samples,
        sampling_rate_hz=SAMPLING_RATE_HZ,
        trial_start_s=TRIAL_START_S,
        trial_end_s=TRIAL_END_S,
    )
    
    dataset_sf_fixed, orientation_qc = enforce_negative_y_peaks_robust(
    dataset=dataset_sf_fixed,
    segments=segments,
    positive_percentile=95,
    negative_percentile=5,
    flip_ratio_threshold=1.05,
    min_peak_amplitude_deg_s=100,
    flip_mode="preserve_z_rotation",
    )   

    auto_orientation_info = {
        "selected_left_yaw_deg": 0.0,
        "selected_right_yaw_deg": 0.0,
        "selection_reason": "auto_selection_disabled",
        "selected_score": np.nan,
        "baseline_score": np.nan,
        "score_margin_to_switch": AUTO_ORIENTATION_SCORE_MARGIN_TO_SWITCH,
    }
    
    auto_orientation_qc_df = pd.DataFrame()
    yaw_override_applied = {}
    
    # Manual non-zero overrides still win, but zero-degree placeholder overrides
    # do not suppress automatic selection.
    if _has_nonzero_yaw_override(base_name, ORIENTATION_YAW_OVERRIDES):
        dataset_sf_fixed, yaw_override_applied = apply_file_orientation_yaw_override(
            dataset=dataset_sf_fixed,
            base_name=base_name,
            overrides=ORIENTATION_YAW_OVERRIDES,
        )
    
        auto_orientation_info = {
            "selected_left_yaw_deg": yaw_override_applied.get("sensor_left", 0.0),
            "selected_right_yaw_deg": yaw_override_applied.get("sensor_right", 0.0),
            "selection_reason": "manual_nonzero_override",
            "selected_score": np.nan,
            "baseline_score": np.nan,
            "score_margin_to_switch": AUTO_ORIENTATION_SCORE_MARGIN_TO_SWITCH,
        }
    
    elif AUTO_SELECT_ORIENTATION:
        dataset_sf_fixed, auto_orientation_info, auto_orientation_qc_df = (
        auto_select_best_orientation_pipeline(
            dataset_sf_aligned_to_gravity=dataset_sf_aligned_to_gravity,
            segments=segments,
            sampling_rate_hz=SAMPLING_RATE_HZ,
            candidates=AUTO_ORIENTATION_CANDIDATES,
        )
    )  
        
    turn_fog_metrics = analyze_360_turns_and_fog_metrics(
        lumbar_body_frame=imu_data,
        shank_right_df=dataset_sf_fixed["sensor_right"],
        shank_left_df=dataset_sf_fixed["sensor_left"],
        turn_list=turning_detector.turn_list_,
        sample_rate=SAMPLING_RATE_HZ,
    )
    
    # Convert foot data to foot body frame.
    bf_data = convert_to_fbf(
        dataset_sf_fixed,
        left_like="sensor_left",
        right_like="sensor_right",
    )

    # -------------------------------------------------------------------------
    # External-direct event mode
    # -------------------------------------------------------------------------
    # This script does not run DTW, Herzer, Rampp, or event matching. The
    # external JSON IC/TC events are used directly. The only estimated event is
    # min_vel, which is required by gaitmap trajectory reconstruction.
    external_json_path = h5_file.with_suffix(EXTERNAL_EVENT_JSON_SUFFIX)
    external_bouts = load_external_event_bouts_json(external_json_path)

    if external_bouts is None:
        print(f"Skipping {h5_file.name}: external JSON loaded as None.")
        return None

    if isinstance(external_bouts, dict):
        external_bouts = [external_bouts]

    if len(external_bouts) == 0:
        print(f"Skipping {h5_file.name}: external JSON contains no bouts.")
        return None

    print("Using external IC/TC events directly; estimating min_vel only.")

    retained_event_list_for_trajectory, external_minvel_qc_df = (
        external_events_to_min_vel_event_list(
            external_bouts=external_bouts,
            bf_data=bf_data,
            sampling_rate_hz=SAMPLING_RATE_HZ,
            min_vel_search_win_size_ms=EXTERNAL_MIN_VEL_SEARCH_WIN_SIZE_MS,
        )
    )

    ed = SimpleNamespace(min_vel_event_list_=retained_event_list_for_trajectory)

    print(
        "External-direct min_vel event rows:",
        f"left={len(retained_event_list_for_trajectory['sensor_left'])}, "
        f"right={len(retained_event_list_for_trajectory['sensor_right'])}",
    )

    # Trajectory reconstruction.
    ori_method = MadgwickRtsKalman(
        use_magnetometer=TRAJ_USE_MAGNETOMETER,
        madgwick_beta=TRAJ_MADGWICK_BETA,
        velocity_error_variance=TRAJ_VELOCITY_ERROR_VARIANCE,
        zupt_detector=StrideEventZuptDetector(half_region_size_s=TRAJ_ZUPT_HALF_REGION_SIZE_S),
    )

    pos_method = ForwardBackwardIntegration(
        gravity=[0, 0, 9.81],
        level_assumption=True,
    )

    trajectory = StrideLevelTrajectory(
        ori_method=ori_method,
        pos_method=pos_method,
    )

    stride_event_list_for_gaitmap = strip_event_list_for_gaitmap_calculations(
        retained_event_list_for_trajectory,
        n_samples=len(dataset_sf_fixed["sensor_left"]),
    )
    
    trajectory = trajectory.estimate(
        data=dataset_sf_fixed,
        stride_event_list=stride_event_list_for_gaitmap,
        sampling_rate_hz=SAMPLING_RATE_HZ,
    )
    
    # Temporal and spatial parameters.
    temporal_paras = TemporalParameterCalculation()
    temporal_paras = temporal_paras.calculate(
        stride_event_list=stride_event_list_for_gaitmap,
        sampling_rate_hz=SAMPLING_RATE_HZ,
    )
    
    spatial_paras = SpatialParameterCalculation()
    spatial_paras = spatial_paras.calculate(
        stride_event_list=stride_event_list_for_gaitmap,
        positions=trajectory.position_,
        orientations=trajectory.orientation_,
        sampling_rate_hz=SAMPLING_RATE_HZ,
    )

    bad_outlier_s_ids_by_sensor, simple_outlier_qc_df = (
        detect_external_style_outlier_s_ids(
            temporal_parameters=temporal_paras.parameters_,
            spatial_parameters=spatial_paras.parameters_,
            stride_time_sd_factor=OUTLIER_STRIDE_TIME_SD_FACTOR,
            stride_time_sd_scope=OUTLIER_STRIDE_TIME_SD_SCOPE,
            exclude_stride_time_gt_max=OUTLIER_EXCLUDE_STRIDE_TIME_GT_MAX,
            stride_time_max_s=OUTLIER_STRIDE_TIME_MAX_S,
            exclude_arc_length_gt_max=OUTLIER_EXCLUDE_ARC_LENGTH_GT_MAX,
            arc_length_max_m=OUTLIER_ARC_LENGTH_MAX_M,
        )
    )
    
    if EXCLUDE_SIMPLE_OUTLIER_STRIDES:
        event_list_for_summary, temporal_parameters_for_summary, spatial_parameters_for_summary = (
            remove_bad_s_ids_from_outputs(
                event_list=retained_event_list_for_trajectory,
                temporal_parameters=temporal_paras.parameters_,
                spatial_parameters=spatial_paras.parameters_,
                bad_s_ids_by_sensor=bad_outlier_s_ids_by_sensor,
            )
        )
    else:
        event_list_for_summary = retained_event_list_for_trajectory
        temporal_parameters_for_summary = temporal_paras.parameters_
        spatial_parameters_for_summary = spatial_paras.parameters_

    # Gaitmap summary outputs are grouped by external bouts. external JSON 
    # represents the externally retained event set.
    segment_outputs = filter_outputs_by_external_direct_event_list(
        external_bouts=external_bouts,
        event_list=event_list_for_summary,
        spatial_parameters=spatial_parameters_for_summary,
        temporal_parameters=temporal_parameters_for_summary,
    )
    
    final_stride_counts = summarize_retained_stride_counts_from_segment_outputs(
        segment_outputs
    )
    
    print(
        "Final retained gaitmap strides:",
        f"left={final_stride_counts['sensor_left_rows']}, "
        f"right={final_stride_counts['sensor_right_rows']}, "
        f"total={final_stride_counts['combined_rows']}"
    )
        
    summary_df, bout_qc_df = summarize_filtered_segments_like_matlab(
        segment_outputs,
        sampling_rate_hz=SAMPLING_RATE_HZ)
    summary_df = summary_df.copy()

    for key, value in turn_fog_metrics.items():
        summary_df[key] = value
        
    summary_df["n_events_left"] = len(ed.min_vel_event_list_["sensor_left"])
    summary_df["n_events_right"] = len(ed.min_vel_event_list_["sensor_right"])
    summary_df["n_events_total"] = (summary_df["n_events_left"].iloc[0]
        + summary_df["n_events_right"].iloc[0])

    summary_df["auto_orientation_selected_base_method"] = auto_orientation_info.get("selected_base_method", "")
    summary_df["auto_orientation_selected_left_yaw_deg"] = auto_orientation_info.get("selected_left_yaw_deg", 0.0)
    summary_df["auto_orientation_selected_right_yaw_deg"] = auto_orientation_info.get("selected_right_yaw_deg", 0.0)
    
    summary_df["orientation_yaw_manual_override_applied"] = bool(yaw_override_applied)
    summary_df["orientation_yaw_manual_override_left_deg"] = yaw_override_applied.get("sensor_left", 0.0)
    summary_df["orientation_yaw_manual_override_right_deg"] = yaw_override_applied.get("sensor_right", 0.0)
        
    summary_df["simple_outlier_exclusion_enabled"] = EXCLUDE_SIMPLE_OUTLIER_STRIDES
    summary_df["outlier_external_mean_sd_rule_enabled"] = OUTLIER_USE_EXTERNAL_MEAN_SD_RULE
    summary_df["outlier_stride_time_sd_factor"] = OUTLIER_STRIDE_TIME_SD_FACTOR
    summary_df["outlier_stride_time_sd_scope"] = OUTLIER_STRIDE_TIME_SD_SCOPE
    summary_df["outlier_exclude_stride_time_gt_max"] = OUTLIER_EXCLUDE_STRIDE_TIME_GT_MAX
    summary_df["outlier_stride_time_max_s"] = OUTLIER_STRIDE_TIME_MAX_S
    summary_df["outlier_exclude_arc_length_gt_max"] = OUTLIER_EXCLUDE_ARC_LENGTH_GT_MAX
    summary_df["outlier_arc_length_max_m"] = OUTLIER_ARC_LENGTH_MAX_M
    summary_df["n_simple_outlier_strides_left"] = len(bad_outlier_s_ids_by_sensor.get("sensor_left", set()))
    summary_df["n_simple_outlier_strides_right"] = len(bad_outlier_s_ids_by_sensor.get("sensor_right", set()))
    summary_df["n_simple_outlier_strides_total"] = (
        summary_df["n_simple_outlier_strides_left"].iloc[0]
        + summary_df["n_simple_outlier_strides_right"].iloc[0])
 
    # Add identifiers.
    rel_parts = list(relative_parent.parts)
    summary_df.insert(0, "source_file", h5_file.name)
    summary_df.insert(1, "relative_folder", str(relative_parent))

    # Optional extraction of subject/session from folder structure.
    # If your structure is root/sub-xxx/ses-xxx/APDM/file.h5, these will work.
    subject = next((p for p in rel_parts if p.startswith("sub-")), "")
    session = next((p for p in rel_parts if p.startswith("ses-")), "")
    base_name = h5_file.stem
    if "uncued" in base_name:
        condition = "uncued"
    elif "external" in base_name:
        condition = "external"
    elif "internal" in base_name:
        condition = "internal"
    
    summary_df.insert(0, "subject", subject)
    summary_df.insert(1, "session", session)
    summary_df.insert(2, "condition", condition)

    summary_df.to_excel(out_excel, index=False, engine="openpyxl")

    print(f"Saved trial summary: {out_excel}")

    return out_excel


# =============================================================================
# 9. COHORT OUTPUT
# =============================================================================

def combine_cohort_level_excel(output_root: Path) -> Optional[Path]:
    """
    Combine all per-trial summary files into one cohort-level Excel file.
    """
    output_root = Path(output_root)
    summary_files = sorted(output_root.rglob("*_summary.xlsx"))

    # Exclude existing cohort summaries if rerun.
    summary_files = [
        f for f in summary_files
        if f.name not in {"cohort_summary.xlsx"}
    ]

    if len(summary_files) == 0:
        print("No per-trial summary files found for cohort summary.")
        return None

    dfs = []
    for file in summary_files:
        try:
            df = pd.read_excel(file, engine="openpyxl")
            df.insert(0, "summary_file", str(file.relative_to(output_root)))
            dfs.append(df)
        except Exception as exc:
            print(f"Could not read summary file {file}: {exc}")

    if len(dfs) == 0:
        print("No readable summary files found for cohort summary.")
        return None

    cohort_df = pd.concat(dfs, ignore_index=True)

    cohort_path = output_root / "cohort_summary.xlsx"
    cohort_df.to_excel(cohort_path, index=False, engine="openpyxl")

    print(f"Saved cohort summary: {cohort_path}")
    return cohort_path


# =============================================================================
# 10. MAIN RUNNER
# =============================================================================

def find_h5_files(root: Path, output_folder_name: str) -> List[Path]:
    """
    Find eligible .h5 files under root while excluding the processed output folder.

    Only process files ending with:
        - uncued.h5
        - external.h5
        - internal.h5
    """
    root = Path(root)

    allowed_suffixes = (
        "uncued.h5",
        "external.h5",
        "internal.h5",
    )

    files = []

    for file in sorted(root.rglob("*.h5")):
        if output_folder_name in file.parts:
            continue

        file_name_lower = file.name.lower()

        if file_name_lower.endswith(allowed_suffixes):
            files.append(file)

    return files


def run_pipeline():
    root = Path(DIRECTORY_PATH)
    output_root = root / OUTPUT_FOLDER_NAME
    output_root.mkdir(parents=True, exist_ok=True)

    h5_files = find_h5_files(root, OUTPUT_FOLDER_NAME)

    print(f"Found {len(h5_files)} .h5 files.")

    error_rows = []
    processed_outputs = []

    for h5_file in h5_files:
        try:
            out_file = process_one_h5_file(h5_file, root, output_root)

            if out_file is not None:
                processed_outputs.append(out_file)

        except Exception as exc:
            print(f"Error processing {h5_file}: {exc}")
            traceback.print_exc()

            error_rows.append(
                {
                    "file": str(h5_file),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    if error_rows:
        error_df = pd.DataFrame(error_rows)
        error_path = output_root / "processing_errors.xlsx"
        error_df.to_excel(error_path, index=False, engine="openpyxl")
        print(f"Saved processing errors: {error_path}")

    combine_cohort_level_excel(output_root)

    print("Pipeline finished.")


if __name__ == "__main__":
    run_pipeline()
