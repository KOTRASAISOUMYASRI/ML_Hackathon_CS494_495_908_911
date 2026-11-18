"""Validate GPS readings and report per-trip data quality."""
import numpy as np


def clean_track(track):
    """Drop invalid and non-increasing readings, retaining raw observation IDs."""
    finite = np.isfinite(track[["longitude", "latitude", "timestamp"]]).all(axis=1)
    valid = finite & track.longitude.abs().le(180) & track.latitude.abs().le(90)
    measurements = track.loc[valid].copy()
    ordered = measurements.timestamp.gt(
        measurements.timestamp.cummax().shift(fill_value=-np.inf)
    )
    cleaned = measurements.loc[ordered].reset_index(drop=True)
    return cleaned


def summarize_track(track, cleaned, *, gap_seconds=30):
    """Return JSON-safe diagnostics; deltas describe the retained readings."""
    deltas = cleaned.timestamp.diff().dropna()
    coords = {}
    for name in ("longitude", "latitude"):
        coords[f"min_{name}"] = float(cleaned[name].min()) if len(cleaned) else None
        coords[f"max_{name}"] = float(cleaned[name].max()) if len(cleaned) else None
    return {
        "raw_observations": len(track),
        "valid_observations": len(cleaned),
        "removed_observations": len(track) - len(cleaned),
        "missing_or_invalid_fields": {
            column: int((~np.isfinite(track[column])).sum())
            for column in ("longitude", "latitude", "timestamp")
        },
        "out_of_range_coordinates": int(
            (track.longitude.abs().gt(180) | track.latitude.abs().gt(90)).sum()
        ),
        "non_increasing_raw_steps": int(track.timestamp.diff().le(0).sum()),
        "median_interval_s": float(deltas.median()) if len(deltas) else None,
        "max_interval_s": float(deltas.max()) if len(deltas) else None,
        "non_1s_steps": int(deltas.ne(1).sum()),
        "large_gaps": int(deltas.gt(gap_seconds).sum()),
        "duration_s": float(deltas.sum()),
        **coords,
    }
