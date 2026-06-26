"""eeg_common.py — single source of truth for the Unicorn EEG pipeline.

Hardware: g.tec Unicorn Hybrid Black.
  - 8 EEG channels sampled at 250 Hz.
  - Standard Unicorn montage order (THIS is the canonical channel order used
    everywhere in the pipeline): Fz, C3, Cz, C4, Pz, PO7, Oz, PO8.
  - The Unicorn LSL app streams 17 channels total
    (8 EEG + accX/Y/Z + gyroX/Y/Z + counter + battery + validation).
    The 8 EEG channels are ALWAYS the first 8 columns/rows of the stream.
  - The Unicorn Recorder writes CSV (comma-separated, with header):
    8 EEG columns first, then the extra sensor/meta columns.

Task: subject plays "Canabalt" (endless runner, SPACEBAR = jump). We detect
spacebar presses from EEG via binary classification:
  - press class (label 1): epoch centred on a keypress event.
  - rest  class (label 0): epoch centred on a random no-press baseline time.
This combines a motor/readiness potential (C3/Cz/C4) with a visual response
(Pz/Oz). Those are the informative channels.

This module defines constants, the in-memory DATA CONTRACT, and tiny helpers.
It performs NO I/O and does NO heavy computation. Other modules import from
here so that shapes, channel order, and constants always match.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Core acquisition constants
# --------------------------------------------------------------------------- #

FS: int = 250          # sampling rate in Hz
EEG_CH: int = 8        # number of EEG channels we keep/use

# Canonical Unicorn montage order. INDEX in this list == row index in the
# raw EEG buffer and == channel axis index in epochs. Do not reorder.
CHANNEL_NAMES: list[str] = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]

assert len(CHANNEL_NAMES) == EEG_CH

# Number of channels the Unicorn LSL stream actually carries; EEG is first 8.
UNICORN_LSL_TOTAL_CH: int = 17

# --------------------------------------------------------------------------- #
# Frequency bands (Hz). Used for band-power features.
# --------------------------------------------------------------------------- #

BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 50.0),
}

# --------------------------------------------------------------------------- #
# Preprocessing constants (MNE-Python is used downstream).
# --------------------------------------------------------------------------- #

BANDPASS: Tuple[float, float] = (0.1, 50.0)   # Hz (l_freq, h_freq)
NOTCH_FREQ: float = 50.0                       # Hz mains notch

# --------------------------------------------------------------------------- #
# Epoching constants (seconds, relative to the event).
# --------------------------------------------------------------------------- #

TMIN: float = -0.2                       # epoch start, 200 ms before event
TMAX: float = 0.8                        # epoch end, 800 ms after event
BASELINE: Tuple[float, float] = (-0.2, 0.0)  # baseline correction window

# Number of samples in one epoch given FS, TMIN, TMAX.
# Inclusive of both endpoints, matching MNE's epoch length convention.
N_TIMES: int = int(round((TMAX - TMIN) * FS)) + 1   # = 251 at 250 Hz

# Event label convention (MNE-style integer codes).
LABEL_REST: int = 0      # no-press baseline
LABEL_SPACE: int = 1     # spacebar press

# --------------------------------------------------------------------------- #
# DATA CONTRACT (in-memory). Every module must conform to these shapes/dtypes.
# --------------------------------------------------------------------------- #
#
# raw EEG buffer:
#     data        : float32 ndarray, shape (EEG_CH=8, n_samples)
#                   row i corresponds to CHANNEL_NAMES[i]. Units: microvolts.
#     timestamps  : float64 ndarray, shape (n_samples,)
#                   monotonic time in SECONDS, one per column of `data`.
#                   Same clock used to log keypress event times.
#
# events:
#     events      : int ndarray, shape (n_events, 2) = [sample_index, label]
#                   sample_index indexes the columns of the raw buffer.
#                   label is LABEL_SPACE (1) or LABEL_REST (0).
#                   (This is the MNE-style events array minus the middle
#                    "previous value" column, which is unused here.)
#
# epochs:
#     epochs      : float32 ndarray, shape (n_epochs, EEG_CH=8, N_TIMES)
#                   axis 0 = epoch, axis 1 = channel (CHANNEL_NAMES order),
#                   axis 2 = time samples from TMIN..TMAX inclusive.
#     labels      : int ndarray, shape (n_epochs,) of LABEL_SPACE / LABEL_REST.
#
# --------------------------------------------------------------------------- #

RAW_DTYPE = np.float32
TIME_DTYPE = np.float64
EPOCH_DTYPE = np.float32


# --------------------------------------------------------------------------- #
# Tiny helpers
# --------------------------------------------------------------------------- #

def channel_index(name: str) -> int:
    """Return the row/channel-axis index of an EEG channel by name.

    Parameters
    ----------
    name : str
        Channel name, e.g. "Cz". Case-sensitive against CHANNEL_NAMES.

    Returns
    -------
    int
        Index into CHANNEL_NAMES / row of the raw buffer / channel axis of
        epochs.

    Raises
    ------
    ValueError
        If `name` is not one of the 8 Unicorn EEG channels.
    """
    try:
        return CHANNEL_NAMES.index(name)
    except ValueError as exc:  # re-raise with a clearer message
        raise ValueError(
            f"{name!r} is not a Unicorn EEG channel; valid: {CHANNEL_NAMES}"
        ) from exc


def make_info(sfreq: float = FS):
    """Build and return an mne.Info for the 8-channel Unicorn EEG montage.

    The info uses CHANNEL_NAMES (in canonical order), channel type 'eeg', and
    the standard 10-20 montage so that topographic / source plotting works.

    Parameters
    ----------
    sfreq : float, default FS (250)
        Sampling frequency in Hz.

    Returns
    -------
    mne.Info
        Info object describing the 8 EEG channels.

    Notes
    -----
    Imports MNE lazily so that this module stays importable (and the data
    contract usable) in environments where MNE is not installed.
    """
    import mne  # local import: keep module import cheap / optional

    info = mne.create_info(
        ch_names=list(CHANNEL_NAMES),
        sfreq=float(sfreq),
        ch_types="eeg",
    )
    # Attach standard 10-20 positions; all 8 Unicorn labels exist in it.
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, match_case=False, on_missing="warn")
    return info


__all__ = [
    "FS",
    "EEG_CH",
    "CHANNEL_NAMES",
    "UNICORN_LSL_TOTAL_CH",
    "BANDS",
    "BANDPASS",
    "NOTCH_FREQ",
    "TMIN",
    "TMAX",
    "BASELINE",
    "N_TIMES",
    "LABEL_REST",
    "LABEL_SPACE",
    "RAW_DTYPE",
    "TIME_DTYPE",
    "EPOCH_DTYPE",
    "channel_index",
    "make_info",
]
