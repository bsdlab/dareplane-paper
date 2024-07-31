import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks

from paper_plots_aux import (apply_default_styles, find_matching,
                             plot_around_time_points, shifted_match_dt,
                             t_match, xdf_to_data_dict)

cfgs = {
    "regular": dict(
        handle_clock_resets=True,
        dejitter_timestamps=True,
    ),
    "irregular": dict(
        handle_clock_resets=False,
        dejitter_timestamps=False,
    ),
}


selected_streams = [
    {"name": "control_signal"},
    {"name": "decoder_output"},
    {"name": "AODataStream"},
    {"name": "ao_cmd"},
]

# COLORS = [
#     "rgba(111,124,145,1)",
#     "rgba(67,162,202,1)",
#     "rgba(186,228,188,1)",
#     "rgba(123,204,196,1)",
#     "#aaa",
# ]

COLORS = px.colors.qualitative.Bold


def find_th_crossing(
    x: np.ndarray,
    th: float,
    horizon: int,
    precission: float = 30,
    direction: str = "up",
) -> np.ndarray:
    """
    Find the index of the first crossing of a threshold
    """

    if direction == "up":
        th_idx = np.where((x - th < precission) & (x - th > 0))[0]
    elif direction == "down":
        th_idx = np.where((x - th > -precission) & (x - th < 0))[0]

    diff = np.diff(th_idx)

    # take only those were at least one horizon is between
    last_idx = np.where(diff > horizon)[0]
    first_idx = np.hstack([[0], last_idx + 1])

    first_th_idx = th_idx[first_idx]

    # ensure that crossings is in the right direction
    idx_msk = []
    for idx in first_th_idx:
        if direction == "up":
            idx_msk.append(
                (x[idx - horizon : idx].mean() < th)
                & (x[idx : idx + horizon].mean() > th)
            )
        elif direction == "down":

            idx_msk.append(
                (x[idx - horizon : idx].mean() > th)
                & (x[idx : idx + horizon].mean() < th)
            )

    midx = first_th_idx[idx_msk]

    return midx


def calc_deltas(d: dict, for_src: str = "arduino") -> dict:
    steps = {
        "source_decorder": calculate_delta_passthrough,
        "decoder_control": calculate_passthrough_controller_delay,
        "control_stim": calculate_controller_controll_signal,
        "stim_osci": calculate_arduino_to_osci_b,
    }

    steps_ct = {
        "source_decorder": calculate_delta_passthrough_ct,
        "decoder_control": calculate_passthrough_controller_delay_ct,
        "control_stim": calculate_controller_to_ct_bic,
        "stim_osci": calculate_ct_cmd_to_ct_ch,
    }
    steps_ao = {
        "source_decorder": calculate_delta_passthrough_ao,
        "decoder_control": calculate_passthrough_controller_delay_ao,
        "control_stim": calculate_controller_controll_signal_ao,
        "stim_osci": calculate_ao_cmd_to_ao_ch2,
        "stim_osci_dejitter": calculate_ao_cmd_to_ao_ch2_with_jitter_correction,
    }
    fmap = {"arduino": steps, "ct": steps_ct, "ao": steps_ao}

    ret_d = {k: v(d) for k, v in fmap[for_src].items()}

    return ret_d


def calculate_delta_passthrough(d: dict) -> list[float]:
    """
    Calculate the delay between actual curve and passthrough, calculate this
    on the non stretched data as this will give information about the actual
    jitter
    """

    sir = d["PICOSTREAM_ireg"]
    ddir = d["decoder_output_ireg"]

    tdiffs = np.diff(sir["ts"])
    tdiffd = np.diff(ddir["ts"])

    idx_new_s = np.where(tdiffs > 0.003)[0] + 1
    idx_new_d = np.where(tdiffd > 0.003)[0] + 1
    ta = sir["ts"][idx_new_s]
    tb = ddir["ts"][idx_new_d]
    xa = sir["x"][idx_new_s][:, 0]
    xb = ddir["x"][idx_new_d][:, 0]

    dts = shifted_match_dt(ta, tb, xa, xb)

    print(f"Source LSL -> decoder_output: {np.quantile(dts, [.05, .5, .95])}")

    return dts


def calculate_delta_passthrough_ct(d: dict) -> list[float]:
    """
    Calculate the delay between actual curve and passthrough, calculate this
    on the non stretched data as this will give information about the actual
    jitter
    """

    sir = d["ct_bic_ireg"]
    ddir = d["decoder_output_ireg"]
    idx_ch = 2

    tdiffs = np.diff(sir["ts"])
    tdiffd = np.diff(ddir["ts"])

    idx_new_s = np.where(tdiffs > 0.003)[0] + 1
    idx_new_d = np.where(tdiffd > 0.003)[0] + 1

    ta = sir["ts"][idx_new_s]
    tb = ddir["ts"][idx_new_d]
    xa = sir["x"][idx_new_s][:, idx_ch]
    xb = ddir["x"][idx_new_d][:, 0]

    dts = shifted_match_dt(ta, tb, xa, xb)
    dts = dts[dts > 0]

    print(f"Source LSL -> decoder_output: {np.quantile(dts, [.05, .5, .95])}")

    return dts


def calculate_passthrough_controller_delay(d: dict) -> list[float]:
    sir = d["decoder_output_ireg"]
    ddir = d["control_signal_ireg"]

    # if ddir["x"].max() == 1.5:
    #     ddir["x"] *= 100

    th_pass_idx = find_th_crossing(
        sir["x"],
        100,
        horizon=5,
        precission=500,
        direction="up",  # precission=300 as jump sized in original osci signal are quite high
    )

    th_pass_idx_ctr = find_th_crossing(
        x=ddir["x"].flatten(), th=130, horizon=1, precission=30, direction="up"
    )

    ix1, ix2 = find_matching(
        a=sir["ts"][th_pass_idx],
        b=ddir["ts"][th_pass_idx_ctr],
        tol=0.1,
        first="a",
    )

    dts = ddir["ts"][th_pass_idx_ctr[ix2]] - sir["ts"][th_pass_idx[ix1]]
    dts.max()
    len(dts[dts > 0.01])  # 6 occurences

    # # just for testing
    # n = 1_000_000
    # msk = np.hstack([np.arange(n), (-1 * np.arange(n)[1:])[::-1]])
    # tmin = min([sir["ts"].min(), ddir["ts"].min()])
    # fig = go.Figure()
    # fig = fig.add_scatter(x=sir["ts"][:n] - tmin, y=sir["x"][:n].flatten())
    # fig = fig.add_scatter(x=ddir["ts"][:n] - tmin, y=ddir["x"][:n].flatten())
    # fig = fig.add_scatter(
    #     x=sir["ts"][th_pass_idx] - tmin,
    #     y=sir["x"][th_pass_idx].flatten(),
    #     mode="markers",
    #     marker_color="#222",
    # )
    # fig = fig.add_scatter(
    #     x=ddir["ts"][th_pass_idx_ctr] - tmin,
    #     y=ddir["x"][th_pass_idx_ctr].flatten(),
    #     mode="markers",
    #     marker_color="#3f3",
    # )
    #
    # fig = fig.add_scatter(
    #     x=ddir["ts"][th_pass_idx_ctr[ix2][dts > 0.01]] - tmin,
    #     y=ddir["x"][th_pass_idx_ctr[ix2][dts > 0.01]].flatten(),
    #     mode="markers",
    #     marker_color="#f33",
    #     marker_size=10,
    # )
    #
    # fig.show()
    #
    # After visual inspection, one can see that the 6 occurences for dts > 0.01
    # are all problems of the heuristics --> drop them
    dts = dts[dts < 0.01]

    # consider only causal estimations
    # dts = dt[dt > 0]
    #

    print(f"Decoder -> control module: {np.quantile(dts, [.05, .5, .95])}")

    return dts


def calculate_passthrough_controller_delay_ct(d: dict) -> list[float]:

    sir = d["decoder_output_ireg"]
    ddir = d["control_signal_ireg"]

    th_pass_idx = find_th_crossing(
        sir["x"],
        100,
        horizon=5,
        precission=500,
        direction="up",  # precission=500 as jump sized in original osci signal are quite high
    )

    # needs a very small horizon to also cover small steps
    th_pass_idx_ctr = find_th_crossing(
        ddir["x"].flatten(), 120, horizon=1, precission=50, direction="up"
    )

    ix1, ix2 = t_match(sir["ts"][th_pass_idx], ddir["ts"][th_pass_idx_ctr])

    dt = ddir["ts"][th_pass_idx_ctr[ix2]] - sir["ts"][th_pass_idx[ix1]]

    # just for testing

    # n = 1_000_000
    # tmin = min([sir["ts"].min(), ddir["ts"].min()])
    # fig = go.Figure()
    # fig = fig.add_scatter(x=sir["ts"][:n] - tmin, y=sir["x"][:n].flatten())
    # fig = fig.add_scatter(x=ddir["ts"][:n] - tmin, y=ddir["x"][:n].flatten())
    # fig = fig.add_scatter(
    #     x=sir["ts"][th_pass_idx] - tmin,
    #     y=sir["x"][th_pass_idx].flatten(),
    #     mode="markers",
    #     marker_color="#222",
    # )
    # fig = fig.add_scatter(
    #     x=ddir["ts"][th_pass_idx_ctr] - tmin,
    #     y=ddir["x"][th_pass_idx_ctr].flatten(),
    #     mode="markers",
    #     marker_color="#3f3",
    # )
    # fig = fig.add_scatter(
    #     x=ddir["ts"][th_pass_idx_ctr[ix2][dt > 0.01]] - tmin,
    #     y=ddir["x"][th_pass_idx_ctr[ix2][dt > 0.01]].flatten(),
    #     mode="markers",
    #     marker_color="#f33",
    # )
    #
    # fig.show()

    # plot_around_time_points(d, ddir["ts"][th_pass_idx_ctr[ix2]][dt > 0.01])

    # consider only causal estimations, others are an issue of the heuristic
    dts = dt[dt > 0]

    print(f"Decoder -> control module: {np.quantile(dts, [.05, .5, .95])}")

    return dts


def calculate_controller_controll_signal(d: dict) -> list[float]:

    sir = d["control_signal_ireg"]
    ddir = d["arduino_cmd_ireg"].copy()

    psir = np.arange(len(sir["x"]) - 1)[np.diff(sir["x"].flatten()) > 48] + 1
    pdir = np.arange(len(ddir["x"]) - 1)[np.diff(ddir["x"].flatten()) > 48] + 1

    # tmin = min([sir["ts"].min(), ddir["ts"].min()])

    # fig = go.Figure()
    # fig = fig.add_scatter(x=sir["ts"] - tmin, y=sir["x"].flatten())
    # fig = fig.add_scatter(
    #     x=ddir["ts"] - tmin, y=ddir["x"].flatten(), line_color="#f33"
    # )
    # fig = fig.add_scatter(
    #     x=sir["ts"][psir] - tmin,
    #     y=sir["x"][psir].flatten(),
    #     mode="markers",
    #     marker_color="#3f3",
    # )
    # fig = fig.add_scatter(
    #     x=ddir["ts"][pdir] - tmin,
    #     y=ddir["x"][pdir].flatten(),
    #     mode="markers",
    #     marker_color="#3ff",
    # )
    #
    # fig.show()
    #
    # manual adjustments
    # Here the control_signal is longer than the arduino cmds which were stopped earlier
    # Alignment is very good (verified visually) -> just start from the first very good match
    istart = np.argmin(
        [np.abs(sir["ts"][i] - ddir["ts"][pdir[0]]) for i in psir]
    )

    dts = [
        ddir["ts"][ip] - sir["ts"][isx] for ip, isx in zip(pdir, psir[istart:])
    ]
    print(
        f"Control module -> arduino input: {np.quantile(dts, [.05, .5, .95])}"
    )

    return dts


def calculate_controller_to_ct_bic(d: dict) -> list[float]:
    sir = d["control_signal_ireg"]
    ddir = d["CTBicControl_ireg"].copy()

    psir = np.arange(len(sir["x"]) - 1)[np.diff(sir["x"].flatten()) > 48] + 1

    # For the CorTec module, we logged string marker < maybe align before redoing?
    # pdir = np.arange(len(ddir["x"]) - 1)[np.diff(ddir["x"].flatten()) > 48] + 1
    pdir = np.where(ddir["x"] == "firing_callback")[0]

    tmin = min([sir["ts"].min(), ddir["ts"].min()])

    # fig = go.Figure()
    # fig = fig.add_scatter(x=sir["ts"] - tmin, y=sir["x"].flatten())
    # fig = fig.add_scatter(
    #     x=sir["ts"][psir] - tmin,
    #     y=sir["x"][psir].flatten(),
    #     mode="markers",
    #     marker_color="#3f3",
    # )
    # for idx in pdir:
    #     fig.add_vline(
    #         x=ddir["ts"][idx] - tmin,
    #         line_color="#f33",
    #         line_dash="dash",
    #         opacity=0.5,
    #     )
    #
    # fig.show()

    # manual adjustments
    #
    # NOTE: Not every control peak was unvoking a callback firing (there was)
    # a grace period of 1.5ms as otherwise the EvalKit stopped responding
    # after a few seconds
    # --> For each sending -> get the preceding control increase
    sidx = [
        psir[np.argmin(np.abs(sir["ts"][psir] - ddir["ts"][idx]))]
        for idx in pdir
    ]
    dts = ddir["ts"][pdir] - sir["ts"][sidx]

    print(
        f"Control module -> CT API input: {np.quantile(dts, [.05, .5, .95])}"
    )

    return dts


def calculate_arduino_to_osci_b(d: dict) -> list[float]:
    sir = d["arduino_cmd_ireg"]
    ddir = d["PICOSTREAM_ireg"]

    # Use this crossing criterion as for the cmd stream, the signal is square
    # enough and for the arduino, it seems a better proxi for reaction
    # as the peak itself has some with, an the pin definitly started reacting
    # once it is above the threshold
    psir = np.arange(len(sir["x"]) - 1)[np.diff(sir["x"].flatten()) > 30] + 1
    pdir = np.arange(len(ddir["x"]) - 1)[np.diff(ddir["x"][:, 1]) > 8000] + 1

    t1 = sir["ts"][psir]
    t2 = ddir["ts"][pdir]
    # plt.plot(sir["ts"], sir["x"].flatten() * 100)
    # plt.plot(ddir["ts"], ddir["x"][:, 1], "r", alpha=0.5)
    # plt.plot(t2, ddir["x"][pdir, 1], "g.", alpha=0.5)
    # plt.plot(t1, sir["x"][psir] * 100, "b.", alpha=0.5)
    # plt.show()

    #
    # assure the sir comes first
    msk = np.where(t2 > t1[0])[0]

    # tolerance of 200ms seems ok, as otherwise false positives start to appear
    # checked with plotting down below
    idx1, idx2 = find_matching(a=t1, b=t2[msk], tol=0.2, first="a")
    # idx1, idx2 = t_match(t1, t2[msk])

    # Assert that selections are unique
    assert np.unique(idx1).shape[0] == len(idx1), "x1 not unique"
    assert np.unique(idx2).shape[0] == len(idx2), "x2 not unique"

    dts = t2[msk][idx2] - t1[idx1]

    # plt.plot(sir["ts"], sir["x"].flatten() * 100)
    # plt.plot(ddir["ts"], ddir["x"][:, 1], "r", alpha=0.5)
    # plt.plot(t2[msk][idx2], ddir["x"][pdir[msk][idx2], 1], "g.", alpha=0.5)
    # plt.plot(t1[idx1], sir["x"][psir[idx1]] * 100, "b.", alpha=0.5)
    # plt.plot(t1[idx1][dts < 0], sir["x"][psir[idx1][dts < 0]] * 100, "ko")
    #
    # # this should not appear < would not be causal
    # plt.plot(
    #     t2[msk][idx2][dts < 0],
    #     ddir["x"][pdir[msk][idx2][dts < 0], 0],
    #     "o",
    #     color="#f3f",
    # )
    #
    # # check large differences
    # plt.plot(
    #     t2[msk][idx2][dts > 0.02],
    #     ddir["x"][pdir[msk][idx2][dts > 0.02], 1],
    #     "o",
    #     color="#fa0",
    # )

    plt.show()
    #
    # Still very few spurious indices left...
    print(f"Arduino -> osci in: {np.quantile(dts, [.05, .5, .95])}")

    return dts


def calculate_ct_cmd_to_ct_ch(d: dict) -> list[float]:
    # NOTE: See comments in calculate_controller_controll_signal

    # sir = d["control_signal_ireg"]
    sir = d["CTBicControl_ireg"]
    ddir = d["ct_bic_ireg"]

    # Use this crossing criterion as for the cmd stream, the signal is square
    # enough and for the arduino, it seems a better proxi for reaction
    # as the peak itself has some with, an the pin definitly started reacting
    # once it is above the threshold
    # psir = np.arange(len(sir["x"]) - 1)[np.diff(sir["x"].flatten()) > 48] + 1
    psir = np.where(sir["x"] == "firing_callback")[0]
    pdir = np.arange(len(ddir["x"]) - 1)[np.diff(ddir["x"][:, 1]) > 2000] + 1

    t1 = sir["ts"][psir]
    t2 = ddir["ts"][pdir]
    # plt.plot(sir["ts"], sir["x"].flatten())
    # plt.plot(ddir["ts"], ddir["x"][:, 1], "r", alpha=0.5)
    # plt.plot(t2, ddir["x"][pdir, 1], "g.", alpha=0.5)
    # plt.vlines(
    #     t1, ymin=ddir["x"].min(), ymax=ddir["x"].max(), color="b", alpha=0.5
    # )
    # plt.show()

    # assure the sir comes first
    msk = np.where(t2 > t1[0])[0]

    idx1, idx2 = find_matching(
        t1, t2[msk], tol=0.2
    )  # same tolerance as for Arduin

    # Assert that selections are unique
    assert np.unique(idx1).shape[0] == len(idx1), "x1 not unique"
    assert np.unique(idx2).shape[0] == len(idx2), "x2 not unique"

    dts = t2[msk][idx2] - t1[idx1]
    # plt.plot(sir["ts"], sir["x"].flatten() * 100)
    # plt.plot(ddir["ts"], ddir["x"][:, 1], "r", alpha=0.5)
    # plt.plot(t2[msk][idx2], ddir["x"][pdir[msk][idx2], 1], "g.", alpha=0.5)
    # plt.plot(t1[idx1], sir["x"][psir[idx1]] * 100, "b.", alpha=0.5)
    # plt.plot(t1[idx1][dts < 0], sir["x"][psir[idx1][dts < 0]] * 100, "ko")
    # plt.plot(
    #     t2[msk][idx2][dts < 0],
    #     ddir["x"][pdir[msk][idx2][dts < 0]],
    #     "o",
    #     color="#f3f",
    # )
    # plt.show()
    #
    # Still very few spurious indices left...
    print(f"CT API command -> CT data: {np.quantile(dts, [.05, .5, .95])}")

    return dts[dts > 0]


def calculate_delta_passthrough_ao(d: dict) -> list[float]:
    """
    Calculate the delay between actual curve and passthrough, calculate this
    on the non stretched data as this will give information about the actual
    jitter
    """

    sir = d["AODataStream_ireg"]
    ddir = d["decoder_output_ireg"]

    tdiffs = np.diff(sir["ts"])
    tdiffd = np.diff(ddir["ts"])

    # just get the first sample of a chunk -> leads to ~20k comparison points
    idx_new_s = np.where(tdiffs > 0.005)[0] + 1
    idx_new_d = (np.where(tdiffd > 0.005)[0] + 1)[1:]

    # Check alignment, we would expect a good match ~1ms
    min_idx = min([len(idx_new_s), len(idx_new_d)])
    dt = sir["ts"][idx_new_s][:min_idx] - ddir["ts"][idx_new_d][:min_idx]

    n_check = 100_000
    if np.abs(dt[:n_check]).max() > 0.005:
        # try shifting by one
        idx_new_d = idx_new_d[1:]
        min_idx = min([len(idx_new_s), len(idx_new_d)])
        dt = sir["ts"][idx_new_s][:min_idx] - ddir["ts"][idx_new_d][:min_idx]
        assert (
            np.abs(dt[:n_check]).max() > 0.005
        ), "Alignment could be off, please double check"

    # idx_ch = 8
    # # just for testing
    # n_samples = 100_000
    # fig = go.Figure()
    # fig = fig.add_scatter(
    #     x=sir["ts"].flatten()[:n_samples],
    #     y=sir["x"][:n_samples, idx_ch].flatten(),
    # )
    # fig = fig.add_scatter(
    #     x=ddir["ts"].flatten()[:n_samples],
    #     y=ddir["x"].flatten()[:n_samples],
    #     line_color="#f33",
    # )
    #
    # mskd = idx_new_d < n_samples
    # msks = idx_new_s < n_samples
    # fig = fig.add_scatter(
    #     x=ddir["ts"][idx_new_d[mskd]],
    #     y=ddir["x"][idx_new_d[mskd]].flatten(),
    #     mode="markers",
    #     marker_color="#1a1",
    # )
    #
    # fig = fig.add_scatter(
    #     x=sir["ts"][idx_new_s[msks]],
    #     y=sir["x"][idx_new_s[msks], idx_ch],
    #     mode="markers",
    #     marker_color="#18a",
    # )
    # fig.show()
    #

    #
    # assert psir.shape == pdir.shape
    idxs, idxd = find_matching(
        sir["ts"][idx_new_s], ddir["ts"][idx_new_d], tol=0.2
    )  # Same tolerance as for Arduino

    dts = ddir["ts"][idx_new_d[idxd]] - sir["ts"][idx_new_s[idxs]]
    len(dts)
    np.asarray(dts).max()
    #
    # # Plot only the very large differences --> they seem to be artifacts of the
    # # heuristics
    # t_max = 100
    # smsk = sir["ts"].flatten() < t_max
    # dmsk = ddir["ts"].flatten() < t_max
    # fig = go.Figure()
    # fig = fig.add_scatter(
    #     x=sir["ts"].flatten()[smsk],
    #     y=sir["x"][smsk, 8].flatten(),
    # )
    # fig = fig.add_scatter(
    #     x=ddir["ts"].flatten()[dmsk],
    #     y=ddir["x"].flatten()[dmsk],
    #     line_color="#3f3",
    # )
    # # points for large dist
    # s_idx = idx_new_s[idxs][dts > 0.01]
    # d_idx = idx_new_d[idxd][dts > 0.01]
    #
    # fig = fig.add_scatter(
    #     x=ddir["ts"][d_idx],
    #     y=ddir["x"][d_idx].flatten(),
    #     mode="markers",
    #     marker_color="#f33",
    #     marker_size=10,
    # )
    #
    # fig = fig.add_scatter(
    #     x=sir["ts"][s_idx],
    #     y=sir["x"][s_idx, 8],
    #     mode="markers",
    #     marker_color="#222",
    #     marker_size=10,
    # )
    # fig.show()
    #
    print(f"Source LSL -> decoder_output: {np.quantile(dts, [.05, .5, .95])}")

    return dts


def calculate_passthrough_controller_delay_ao(d: dict) -> list[float]:
    sir = d["decoder_output_ireg"]
    ddir = d["control_signal_ireg"]

    if ddir["x"].max() == 1.5:
        ddir["x"] *= 100

    th_pass_idx = find_th_crossing(
        sir["x"], 15000, horizon=100, precission=550, direction="up"
    )

    th_pass_idx_ctr = find_th_crossing(
        ddir["x"].flatten(), 120, horizon=10, precission=50, direction="up"
    )

    # just for testing

    # n = 1_000_000
    # msk = np.hstack([np.arange(n), (-1 * np.arange(n)[1:])[::-1]])
    # tmin = min([sir["ts"].min(), ddir["ts"].min()])
    # fig = FigureResampler(go.Figure())
    # fig.add_trace(
    #     go.Scatter(name="decoder", mode="markers+lines"),
    #     hf_x=sir["ts"] - tmin,
    #     hf_y=sir["x"].flatten(),
    # )
    # fig.add_trace(
    #     go.Scatter(name="control", mode="markers+lines"),
    #     hf_x=ddir["ts"] - tmin,
    #     hf_y=ddir["x"].flatten() * 100,
    # )
    # fig.add_trace(
    #     go.Scatter(mode="markers", name="decoder_pass"),
    #     hf_x=sir["ts"][th_pass_idx] - tmin,
    #     hf_y=sir["x"][th_pass_idx].flatten(),
    # )
    # fig.add_trace(
    #     go.Scatter(mode="markers", name="control_pass"),
    #     hf_x=ddir["ts"][th_pass_idx_ctr] - tmin,
    #     hf_y=ddir["x"][th_pass_idx_ctr].flatten() * 100,
    # )
    # fig.update_layout(height=1200)
    # fig.show_dash()
    #
    # plt.plot(sir["ts"][:n] - tmin, sir["x"][:n].flatten())
    # plt.plot(ddir["ts"][:n] - tmin, ddir["x"][:n].flatten() * 100)
    # plt.plot(
    #     sir["ts"][th_pass_idx] - tmin, sir["x"][th_pass_idx].flatten(), "g."
    # )
    # plt.plot(
    #     ddir["ts"][th_pass_idx_ctr] - tmin,
    #     ddir["x"][th_pass_idx_ctr].flatten() * 100,
    #     "b.",
    # )
    # plt.show()

    # find corresponding high values in the control signal
    # Compare always against closest value
    # ctr will have more idx
    # >> take the decoder output and find the closest next control response within 1s

    idx, idx_ctrl = find_matching(
        sir["ts"][th_pass_idx], ddir["ts"][th_pass_idx_ctr], tol=0.2
    )  # Same tolerance as for Arduino

    dts = [
        ddir["ts"][ip] - sir["ts"][isx]
        for ip, isx in zip(th_pass_idx_ctr[idx_ctrl], th_pass_idx[idx])
    ]

    print(f"Decoder -> control module: {np.quantile(dts, [.05, .5, .95])}")

    # plot a single example
    # n = 200  # around
    # mx_i = np.argmax(dts)
    # mx_i = len(dts) - 9
    # iis = th_pass_idx[idx][mx_i]
    # ic = th_pass_idx_ctr[idx_ctrl][mx_i]
    # fig = go.Figure()
    # sps = slice(iis - n, n + iis)
    # spc = slice(ic - n, n + ic)
    # fig.add_scatter(x=sir["ts"][sps], y=sir["x"][sps].flatten())
    # fig.add_scatter(x=ddir["ts"][spc], y=ddir["x"][spc].flatten() * 100)
    # fig.add_scatter(x=[sir["ts"][iis]], y=sir["x"][iis].flatten())
    # fig.add_scatter(x=[ddir["ts"][ic]], y=ddir["x"][ic].flatten() * 100)
    # fig.show()

    return dts


def calculate_controller_controll_signal_ao(d: dict) -> list[float]:
    sir = d["control_signal_ireg"]
    ddir = d["ao_cmd_ireg"].copy()

    psir = np.arange(len(sir["x"]) - 1)[np.diff(sir["x"].flatten()) > 48] + 1
    pdir = np.arange(len(ddir["x"]) - 1)[np.diff(ddir["x"].flatten()) > 48] + 1

    tmin = min([sir["ts"].min(), ddir["ts"].min()])

    # fig = go.Figure()
    # fig = fig.add_scatter(x=sir["ts"] - tmin, y=sir["x"].flatten())
    # fig = fig.add_scatter(
    #     x=ddir["ts"] - tmin, y=ddir["x"].flatten(), line_color="#f33"
    # )
    # fig = fig.add_scatter(
    #     x=sir["ts"][psir] - tmin,
    #     y=sir["x"][psir].flatten(),
    #     mode="markers",
    #     marker_color="#3f3",
    # )
    # fig = fig.add_scatter(
    #     x=ddir["ts"][pdir] - tmin,
    #     y=ddir["x"][pdir].flatten(),
    #     mode="markers",
    #     marker_color="#3ff",
    # )
    #
    # fig.show()
    #
    # manual adjustments
    # Here the control_signal is longer than the arduino cmds which were stopped earlier
    # Alignment is very good (verified visually) -> just start from the first very good match
    istart = np.argmin(
        [np.abs(sir["ts"][i] - ddir["ts"][pdir[0]]) for i in psir]
    )

    dts = [
        ddir["ts"][ip] - sir["ts"][isx] for ip, isx in zip(pdir, psir[istart:])
    ]
    print(
        f"Control module -> arduino input: {np.quantile(dts, [.05, .5, .95])}"
    )

    return dts


def calculate_ao_cmd_to_ao_ch2(d: dict) -> list[float]:
    sir = d["ao_cmd_ireg"]
    ddir = d["AODataStream_ireg"]

    # For the AO the actual computation is a bit more complex: We know that
    # data is appearing in chunks every ~5ms. So we should go to the end of
    # the block before the appearance of the stim pulse and then extrapolate
    # from there given the number of samples.
    psir = np.arange(len(sir["x"]) - 1)[np.diff(sir["x"].flatten()) > 48] + 1

    pdir = find_peaks(
        -ddir["x"][:, 1],
        height=10000,
        prominence=1000,
        distance=2500,
    )[0]

    # pdir = np.arange(len(ddir["x"]) - 1)[np.diff(ddir["x"][:, 1]) < -10000] + 1

    t1 = sir["ts"][psir]
    t2 = ddir["ts"][pdir]
    # plt.plot(sir["ts"], sir["x"].flatten() * 100)
    # plt.plot(ddir["ts"], ddir["x"][:, 1], "r", alpha=0.5)
    # plt.plot(t2, ddir["x"][pdir, 1], "g.", alpha=0.5)
    # plt.plot(t1, sir["x"][psir] * 100, "b.", alpha=0.5)
    # plt.show()

    # assure the sir comes first
    msk = np.where(t2 > t1[0])[0]

    idx1, idx2 = find_matching(
        t1, t2[msk], tol=0.2
    )  # Same tolerance as for Arduino

    # Assert that selections are unique
    assert np.unique(idx1).shape[0] == len(idx1), "x1 not unique"
    assert np.unique(idx2).shape[0] == len(idx2), "x2 not unique"

    dts = t2[msk][idx2] - t1[idx1]
    # plt.plot(sir["ts"], sir["x"].flatten() * 100)
    # plt.plot(ddir["ts"], ddir["x"][:, 1], "r", alpha=0.5)
    # plt.plot(t2[msk][idx2], ddir["x"][pdir[msk][idx2], 1], "g.", alpha=0.5)
    # plt.plot(t1[idx1], sir["x"][psir[idx1]] * 100, "b.", alpha=0.5)
    # plt.plot(t1[idx1][dts < 0], sir["x"][psir[idx1][dts < 0]] * 100, "ko")
    # plt.plot(
    #     t2[msk][idx2][dts < 0],
    #     ddir["x"][pdir[msk][idx2][dts < 0]],
    #     "o",
    #     color="#f3f",
    # )
    # plt.show()
    #
    # Still very few spurious indices left...
    print(f"AO command module -> AO data: {np.quantile(dts, [.05, .5, .95])}")

    return dts[dts > 0]


def calculate_ao_cmd_to_ao_ch2_with_jitter_correction(d: dict) -> list[float]:
    sir = d["ao_cmd_ireg"]
    ddir = d["AODataStream_ireg"]

    # For the AO the actual computation is a bit more complex: We know that
    # data is appearing in chunks every ~5ms. So we should go to the end of
    # the block before the appearance of the stim pulse and then extrapolate
    # from there given the number of samples.
    psir = np.arange(len(sir["x"]) - 1)[np.diff(sir["x"].flatten()) > 48] + 1

    pdir = find_peaks(
        -ddir["x"][:, 1],
        height=10000,
        prominence=1000,
        distance=2500,
    )[0]

    # get array of chunk numbers
    aux = np.zeros(ddir["ts"].shape[0])
    aux[1:][np.diff(ddir["ts"]) > 0.005] = 1
    chunk = np.cumsum(aux)

    # for each time point grab the distance to the last chunk switch and the
    # time at the last chunk end
    p_chunks = chunk[pdir]

    chunk_last_idx = [np.where(chunk == p - 1)[0][-1] for p in p_chunks]

    last_pre_chunk_to_peak = pdir - chunk_last_idx
    dt = last_pre_chunk_to_peak / 5500  # actual sampling rate of the AO

    t1 = sir["ts"][psir]
    t2 = ddir["ts"][chunk_last_idx] + dt
    # plt.plot(sir["ts"], sir["x"].flatten() * 100)
    # plt.plot(ddir["ts"], ddir["x"][:, 1], "r", alpha=0.5)
    # plt.plot(t2, ddir["x"][pdir, 1], "g.", alpha=0.5)
    # plt.plot(t1, sir["x"][psir] * 100, "b.", alpha=0.5)
    # plt.show()

    # assure the sir comes first
    msk = np.where(t2 > t1[0])[0]

    idx1, idx2 = find_matching(t1, t2[msk], tol=0.2)  # same as for Arduino

    # Assert that selections are unique
    assert np.unique(idx1).shape[0] == len(idx1), "x1 not unique"
    assert np.unique(idx2).shape[0] == len(idx2), "x2 not unique"

    dts = t2[msk][idx2] - t1[idx1]
    # plt.plot(sir["ts"], sir["x"].flatten() * 100)
    # plt.plot(ddir["ts"], ddir["x"][:, 1], "r", alpha=0.5)
    # plt.plot(t2[msk][idx2], ddir["x"][pdir[msk][idx2], 1], "g.", alpha=0.5)
    # plt.plot(t1[idx1], sir["x"][psir[idx1]] * 100, "b.", alpha=0.5)
    # plt.plot(t1[idx1][dts < 0], sir["x"][psir[idx1][dts < 0]] * 100, "ko")
    # plt.plot(
    #     t2[msk][idx2][dts < 0],
    #     ddir["x"][pdir[msk][idx2][dts < 0]],
    #     "o",
    #     color="#f3f",
    # )
    # plt.show()
    #
    # Still very few spurious indices left...
    print(f"AO command module -> AO data: {np.quantile(dts, [.05, .5, .95])}")

    return dts[dts > 0]


def load_or_create_pickle(file: Path) -> dict:
    if not file.parent.exists():
        file.parent.mkdir(parents=True)

    if file.exists():
        d = pickle.load(open(file, "rb"))
    else:
        pfx_map = {
            "AO_test": "ao",
            "Arduino_test": "arduino",
            "CT_test": "ct",
        }

        xdf = Path(f"./data/{file.stem}.xdf")

        # Note: for the AO xdf, the labeling of the nominal sampling rate
        #       is wrong. It should be 5.5KHz
        data = xdf_to_data_dict(xdf, cfgs=cfgs, tmax_s=300)

        d = calc_deltas(data, for_src=pfx_map[file.stem])
        pickle.dump(d, open(file, "wb"))

    return d


def get_compare_statistics() -> pd.DataFrame:
    ao_file = Path("results/AO_test.p")
    ct_file = Path("results/CT_test.p")
    arduino_file = Path("results/Arduino_test.p")

    ao = load_or_create_pickle(ao_file)
    ct = load_or_create_pickle(ct_file)
    arduino = load_or_create_pickle(arduino_file)

    data = []
    for k, d, f in zip(
        ["ao", "ct", "arduino"],
        [ao, ct, arduino],
        [ao_file, ct_file, arduino_file],
    ):
        for dk, dv in d.items():
            data.append(
                pd.DataFrame(
                    {
                        "source": [k] * len(dv),
                        "delta_type": [dk] * len(dv),
                        "file": [f] * len(dv),
                        "dt_s": dv,
                    }
                )
            )

    df = pd.concat(data)

    # drop clear outliers
    # TODO: check how these outliers come into the data
    df = df[(df["dt_s"] < 1) & (df["dt_s"] > 0)]

    return df


def q1(x):
    return x.quantile(0.01)


# 90th Percentile
def q99(x):
    return x.quantile(0.99)


def create_stats_table(df: pd.DataFrame) -> str:
    # work in ms
    dw = df.copy()
    dw.dt_s = dw.dt_s * 1000

    dg = (
        dw.groupby(["source", "delta_type"], as_index="False")
        .agg({"dt_s": ["mean", "min", "max", "median", q1, q99, "count"]})
        .reset_index()
    )
    dg["source"] = dg.source.map(
        {"ao": "Neuro Omega", "ct": "CorTec EvalKit", "arduino": "Arduino Uno"}
    )
    dg["source"] = pd.Categorical(
        dg["source"], ["Arduino Uno", "CorTec EvalKit", "Neuro Omega"]
    )

    dg["delta_type"] = pd.Categorical(
        dg["delta_type"],
        ["source_decorder", "decoder_control", "control_stim", "stim_osci"],
    )
    dg = dg.sort_values(["source", "delta_type"])

    delta_str = r"$\Delta$"
    dg.delta_type = dg.delta_type.map(
        {
            "control_stim": f"{delta_str}CT",
            "decoder_control": f"{delta_str}DC",
            "source_decorder": f"{delta_str}SD",
            "stim_osci": f"{delta_str}TS",
        }
    )

    # dp = pd.pivot_table(
    #     dg,
    #     values=["dt_s"],
    #     index=["delta_type"],
    #     columns=["source"],
    #     aggfunc="first",
    # )
    # dp.columns = dp.columns.reorder_levels([2, 0, 1]).droplevel(1)
    #
    # dp.sort_index(axis=1, level=[0], inplace=True)
    #
    # sort = ["count", "mean [ms]", "median [ms]", "min [ms]", "max [ms]"]
    # sources = dg.source.unique()
    # idx = [(s, m) for s in sources for m in sort]
    #
    # dp = dp.loc[:, idx]
    #

    dg.columns = dg.columns.droplevel(0)
    sort = [
        "count",
        "mean",
        "median",
        "min",
        "max",
        "q1",
        "q99",
    ]
    dg = dg.rename(
        columns=dict(
            zip(["count", "mean", "median", "min", "max", "q1", "q99"], sort)
        )
    )
    # drop counts for now
    dg = dg.drop("count", axis=1)

    cols = dg.columns.tolist()
    cols[0] = "Test type"
    cols[1] = "Difference"
    dg.columns = cols
    dg.set_index(["Test type", "Difference"], inplace=True)

    cols = dg.columns.tolist()
    for i in range(0, len(cols)):
        cols[i] = ("", cols[i])
    cols[0] = ("Time [ms]", cols[0][1])

    dg.columns = pd.MultiIndex.from_tuples(cols)

    ltx = dg.to_latex(
        formatters={"count": int},
        float_format="{:0.3f}".format,
        caption="Performance comparison of bench top system tests",
        label="tab:performance_comparison",
    )
    print(ltx)

    # some help for faster replacement in the latex file
    print("".join([f"{{{v:.3f}}}" for v in dg.iloc[3, 1:].values]))
    print("".join([f"{{{v:.3f}}}" for v in dg.iloc[7, 1:].values]))
    print("".join([f"{{{v:.3f}}}" for v in dg.iloc[11, 1:].values]))

    print(
        f"$\\Delta_{{q99}}$SD + $\\Delta_{{q99}}$DC + $\\Delta_{{q99}}$CT = \\qty{{{dg.iloc[:3, -1].sum():.3f}}}{{ms}}"
    )

    print(
        f"$\\Delta_{{q99}}$SD + $\\Delta_{{q99}}$DC + $\\Delta_{{q99}}$CT = \\qty{{{dg.iloc[4:7, -1].sum():.3f}}}{{ms}}"
    )

    print(
        f"$\\Delta_{{q99}}$SD + $\\Delta_{{q99}}$DC + $\\Delta_{{q99}}$CT = \\qty{{{dg.iloc[8:11, -1].sum():.3f}}}{{ms}}"
    )

    return ltx


def plot_example_trace_and_box(df: pd.DataFrame) -> go.Figure:
    xdf = "./data/CT_test.xdf"  # lots of noise on the Oscilloscope signal
    # d = xdf_to_data_dict(xdf, cfgs=cfgs, tmax_s=200)
    global cfgs
    d = xdf_to_data_dict(
        xdf, cfgs=cfgs, tmax_s=300
    )  # tmax is only relevant for the sample trace, not for the box plot stats

    d = crop_to_window(d, 73.0, 77.2)

    # subplots single col span and three separate
    fig = make_subplots(
        rows=3,
        cols=3,
        specs=[
            [{"colspan": 2}, None, {}],
            [{"colspan": 3}, None, None],
            [{}, {}, {}],
        ],
        row_heights=[0.2, 0.2, 0.7],
        vertical_spacing=0.08,
        subplot_titles=[
            "<b>(A)</b> Signal and response example - BIC-EvalKit",
            "",
            "",  # zoomed even further
            "<b>(B)</b> Arduino",
            "<b>(C)</b> BIC-EvalKit",
            "<b>(D)</b> Neuro Omega",
        ],
    )
    fig = sample_examples_trace_plot(d, fig)

    # use smaller data slices to reduce figure size
    fig = zoomed_in_examples_trace_plot_TS(d, fig, tmin=4.105, tmax=4.16)
    fig = zoomed_in_examples_trace_plot_others(d, fig, tmin=4.1, tmax=4.2)
    fig = fig.update_layout(height=2200)

    df = get_compare_statistics()
    fig = add_stats_box_plots(fig, df)

    fig = apply_default_styles(fig)
    for ann in fig.layout.annotations:
        ann.font.size = 24

    fig = fig.update_layout(
        width=1100,
        height=1400,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255, 255, 255, 0.9)",
        ),
    )

    return fig


def crop_to_window(
    d: dict, tstart_s: float, tend_s: float, reset_time: bool = True
) -> dict:
    """
    Crop the dictionary to the time window
    """
    for k, v in d.items():
        msk = (v["ts"] > tstart_s) & (v["ts"] < (tend_s))
        d[k]["ts"] = v["ts"][msk]
        d[k]["x"] = v["x"][msk]

    # reset times
    if reset_time:
        mint = min([v["ts"].min() for v in d.values()])
        for k, v in d.items():
            d[k]["ts"] -= mint

    return d


def add_stats_box_plots(fig: go.Figure, df: pd.DataFrame) -> go.Figure:

    dw = df.copy()

    delta_str = r"Δ"
    dw.delta_type = dw.delta_type.map(
        {
            "control_stim": f"{delta_str}CT",
            "decoder_control": f"{delta_str}DC",
            "source_decorder": f"{delta_str}SD",
            "stim_osci": f"{delta_str}TS",
        }
    )
    cmap = {
        "arduino": "#636efa",
        "ct": "#00cc96",
        "ao": "#ffa15a",
    }

    for i, (src, name) in enumerate(
        zip(["arduino", "ct", "ao"], ["Arduino", "EvalKit", "Neuro Omega"])
    ):

        ds = dw[dw["source"] == src]
        fig.add_trace(
            go.Box(
                x=ds["delta_type"],
                y=ds["dt_s"],
                name=name,
                boxmean=True,
                boxpoints="all",
                jitter=0.5,
                pointpos=-1.8,
                marker_color=cmap[src],
                marker=dict(opacity=0.5),
                showlegend=False,
            ),
            row=3,
            col=i + 1,
        )

    fig = fig.update_yaxes(
        title_text="",
        type="log",
        minor=dict(showgrid=True),
        row=3,
        range=[np.log10(0.00005), np.log10(0.3)],
    )

    fig = fig.update_yaxes(
        title_text="Time for update [s]",
        type="log",
        minor=dict(showgrid=True),
        row=3,
        col=1,
        range=[np.log10(0.00005), np.log10(0.3)],
    )

    # fig = fig.update_xaxes(title_text="Difference", row=2, col=1)
    return fig


def sample_examples_trace_plot(d: dict, fig: go.Figure) -> go.Figure:
    """Add the sample trace plot"""
    # signal_stream = "PICOSTREAM_ireg"
    signal_stream = "ct_bic_ireg"

    fig = fig.add_scatter(
        x=d[signal_stream]["ts"],
        y=d[signal_stream]["x"][:, 2],
        line_color=COLORS[0],
        name="Ch 3 - 'signal'",
        mode="lines",
        row=1,
        col=1,
    )

    fig = fig.add_scatter(
        x=d[signal_stream]["ts"],
        y=d[signal_stream]["x"][:, 1],
        line_color=COLORS[1],
        name="Ch 1 - stim artifcat",
        mode="lines",
        row=1,
        col=1,
    )

    fig = fig.add_scatter(
        x=d["decoder_output_ireg"]["ts"],
        y=d["decoder_output_ireg"]["x"][:, 0],
        line_color=COLORS[2],
        name="decoded",
        mode="lines",
        row=1,
        col=1,
    )

    fig = fig.add_scatter(
        x=d["control_signal_ireg"]["ts"],
        y=d["control_signal_ireg"]["x"][:, 0],
        name="control",
        mode="lines",
        line_color=COLORS[3],
        row=1,
        col=1,
    )

    # create a timeseries to reflect the CT triggers (we used markers)
    xt = d["CTBicControl_ireg"]["ts"]
    xt_cb = xt[np.where(d["CTBicControl_ireg"]["x"] == "firing_callback")[0]]

    fig = fig.add_scatter(
        x=xt_cb,
        y=np.ones(len(xt_cb)) * 100,
        name="stim trigger",
        mode="markers",
        marker=dict(size=10, color=COLORS[4]),
        row=1,
        col=1,
    )

    fig = fig.add_scatter(
        x=[d["ct_bic_ireg"]["ts"][0], d["ct_bic_ireg"]["ts"][-1]],
        # y=[15_000, 15_000],
        y=[100, 100],
        name="threshold",
        mode="lines",
        line_color="#333",
        line_dash="dash",
        opacity=0.5,
        row=1,
        col=1,
    )

    fig = fig.update_yaxes(title_text="Amplitude [AU]", row=1, col=1)
    fig = fig.update_xaxes(title_text="Time [s]", row=1, col=1)

    return fig


def plot_segment_pico(
    d: dict,
    fig: go.Figure,
    row: int,
    col: int,
    tmin: float,
    tmax: float,
    mode: str = "lines",
) -> go.Figure:

    bic_msk = (d["PICOSTREAM_ireg"]["ts"] > tmin) & (
        d["PICOSTREAM_ireg"]["ts"] < tmax
    )
    dec_msk = (d["decoder_output_ireg"]["ts"] > tmin) & (
        d["decoder_output_ireg"]["ts"] < tmax
    )
    ctrl_msk = (d["control_signal_ireg"]["ts"] > tmin) & (
        d["control_signal_ireg"]["ts"] < tmax
    )
    trg_msk = (d["CTBicControl_ireg"]["ts"] > tmin) & (
        d["CTBicControl_ireg"]["ts"] < tmax
    )
    showlegend = False

    fig = fig.add_scatter(
        x=d["PICOSTREAM_ireg"]["ts"][bic_msk],
        y=d["PICOSTREAM_ireg"]["x"][bic_msk, 2],
        line_color=COLORS[0],
        marker_color=COLORS[0],
        name="Ch 3 - 'signal'",
        mode=mode,
        showlegend=showlegend,
        row=row,
        col=col,
    )

    fig = fig.add_scatter(
        x=d["PICOSTREAM_ireg"]["ts"][bic_msk],
        y=d["PICOSTREAM_ireg"]["x"][bic_msk, 0],
        name="Ch 1 - stim artifcat",
        mode=mode,
        line_color=COLORS[1],
        marker_color=COLORS[1],
        showlegend=showlegend,
        row=row,
        col=col,
    )

    fig = fig.add_scatter(
        x=d["decoder_output_ireg"]["ts"][dec_msk],
        y=d["decoder_output_ireg"]["x"][dec_msk, 0],
        name="decoded",
        line_color=COLORS[2],
        marker_color=COLORS[2],
        mode=mode,
        showlegend=showlegend,
        row=row,
        col=col,
    )

    fig = fig.add_scatter(
        x=d["control_signal_ireg"]["ts"][ctrl_msk],
        y=d["control_signal_ireg"]["x"][ctrl_msk, 0],
        name="control",
        mode=mode,
        line_color=COLORS[3],
        marker_color=COLORS[3],
        showlegend=showlegend,
        row=row,
        col=col,
    )

    # create a timeseries to reflect the CT triggers (we used markers)
    xt = d["CTBicControl_ireg"]["ts"][trg_msk]
    xt_cb = xt[
        np.where(d["CTBicControl_ireg"]["x"][trg_msk] == "firing_callback")[0]
    ]

    fig = fig.add_scatter(
        x=xt_cb,
        y=np.ones(len(xt_cb)) * 100,
        name="stim trigger",
        mode="markers",
        marker=dict(size=10, color=COLORS[4]),
        row=row,
        col=col,
        showlegend=showlegend,
    )

    fig = fig.add_scatter(
        x=[d["PICOSTREAM_ireg"]["ts"][0], d["PICOSTREAM_ireg"]["ts"][-1]],
        y=[100, 100],
        name="threshold",
        mode="lines",
        line_color="#333",
        line_dash="dash",
        opacity=0.5,
        row=row,
        col=col,
        showlegend=showlegend,
    )

    return fig


def plot_segment(
    d: dict,
    fig: go.Figure,
    row: int,
    col: int,
    tmin: float,
    tmax: float,
    mode: str = "lines",
) -> go.Figure:

    bic_msk = (d["ct_bic_ireg"]["ts"] > tmin) & (d["ct_bic_ireg"]["ts"] < tmax)
    dec_msk = (d["decoder_output_ireg"]["ts"] > tmin) & (
        d["decoder_output_ireg"]["ts"] < tmax
    )
    ctrl_msk = (d["control_signal_ireg"]["ts"] > tmin) & (
        d["control_signal_ireg"]["ts"] < tmax
    )
    trg_msk = (d["CTBicControl_ireg"]["ts"] > tmin) & (
        d["CTBicControl_ireg"]["ts"] < tmax
    )
    showlegend = False

    fig = fig.add_scatter(
        x=d["ct_bic_ireg"]["ts"][bic_msk],
        y=d["ct_bic_ireg"]["x"][bic_msk, 2],
        line_color=COLORS[0],
        marker_color=COLORS[0],
        name="Ch 3 - 'signal'",
        mode=mode,
        showlegend=showlegend,
        row=row,
        col=col,
    )

    fig = fig.add_scatter(
        x=d["ct_bic_ireg"]["ts"][bic_msk],
        y=d["ct_bic_ireg"]["x"][bic_msk, 0],
        name="Ch 1 - stim artifcat",
        mode=mode,
        line_color=COLORS[1],
        marker_color=COLORS[1],
        showlegend=showlegend,
        row=row,
        col=col,
    )

    fig = fig.add_scatter(
        x=d["decoder_output_ireg"]["ts"][dec_msk],
        y=d["decoder_output_ireg"]["x"][dec_msk, 0],
        name="decoded",
        line_color=COLORS[2],
        marker_color=COLORS[2],
        mode=mode,
        showlegend=showlegend,
        row=row,
        col=col,
    )

    fig = fig.add_scatter(
        x=d["control_signal_ireg"]["ts"][ctrl_msk],
        y=d["control_signal_ireg"]["x"][ctrl_msk, 0],
        name="control",
        mode=mode,
        line_color=COLORS[3],
        marker_color=COLORS[3],
        showlegend=showlegend,
        row=row,
        col=col,
    )

    # create a timeseries to reflect the CT triggers (we used markers)
    xt = d["CTBicControl_ireg"]["ts"][trg_msk]
    xt_cb = xt[
        np.where(d["CTBicControl_ireg"]["x"][trg_msk] == "firing_callback")[0]
    ]

    fig = fig.add_scatter(
        x=xt_cb,
        y=np.ones(len(xt_cb)) * 100,
        name="stim trigger",
        mode="markers",
        marker=dict(size=10, color=COLORS[4]),
        row=row,
        col=col,
        showlegend=showlegend,
    )

    fig = fig.add_scatter(
        x=[d["ct_bic_ireg"]["ts"][0], d["ct_bic_ireg"]["ts"][-1]],
        y=[100, 100],
        name="threshold",
        mode="lines",
        line_color="#333",
        line_dash="dash",
        opacity=0.5,
        row=row,
        col=col,
        showlegend=showlegend,
    )

    return fig


def zoomed_in_examples_trace_plot_TS(
    d: dict, fig: go.Figure, tmin: float, tmax: float
) -> go.Figure:
    """Add the sample trace plot"""

    fig = plot_segment(
        d, fig, row=1, col=3, tmin=tmin, tmax=tmax, mode="lines"
    )
    # fig = plot_segment_pico(
    #     d, fig, row=1, col=3, tmin=tmin, tmax=tmax, mode="lines"
    # )

    fig = fig.update_yaxes(
        title_text="",
        row=1,
        col=3,
        range=[-200, 300],
        tickmode="array",
        tickvals=[-200, -100, 0, 100, 200, 300],
        ticktext=["", "-100", "0", "100", "200", ""],
    )
    fig = fig.update_xaxes(
        title_text="Time [s]",
        row=1,
        col=3,
        range=[tmin, tmax],
        tickmode="array",
        # tickvals=[2.385, 2.39, 2.395, 2.4, 2.405],
        # ticktext=["", 2.39, 2.395, 2.4, ""],
    )

    return fig


def zoomed_in_examples_trace_plot_others(
    d: dict, fig: go.Figure, tmin: float, tmax: float
) -> go.Figure:
    """Add the sample trace plot"""

    fig = plot_segment(d, fig, row=2, col=1, tmin=tmin, tmax=tmax)

    # fig = plot_segment_pico(
    #     d, fig, row=2, col=1, tmin=tmin, tmax=tmax, mode="lines+markers"
    # )

    fig = fig.update_yaxes(
        title_text="",
        row=2,
        col=1,
        range=[0, 200],
        tickmode="array",
        tickvals=[0, 50, 100, 150, 200],
        ticktext=["", 50, "", 150, ""],
        # ticktext=["14.8k", "15k", "15.2k"],
    )
    fig = fig.update_xaxes(
        title_text="Time [s]",
        row=2,
        col=1,
        range=[4.114, 4.117],
        tickmode="array",
        tickvals=[
            4.114,
            4.1145,
            4.115,
            4.1155,
            4.116,
            4.1165,
            4.117,
        ],
        ticktext=[
            "",
            4.1145,
            4.115,
            4.1155,
            4.116,
            4.1165,
            "",
        ],
    )

    return fig


if __name__ == "__main__":
    df = get_compare_statistics()

    # print AO normal vs dejittered (own adjustment)
    df[df.source == "ao"].groupby("delta_type")["dt_s"].describe()

    # the box plots are calculated with the plain data only (more conservative)
    fig = plot_example_trace_and_box(
        df[~df.delta_type.str.contains("dejittered")]
    )
    fig.show()

    # test plot
    print(create_stats_table(df))
