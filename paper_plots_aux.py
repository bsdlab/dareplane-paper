from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyxdf
from plotly.express.colors import sample_colorscale
from plotly.graph_objs import _box, _violin
from plotly.subplots import make_subplots
from scipy import stats


class ModeNotImplementedError(ValueError):
    pass


def add_box_significance_indicator(
    fig: go.Figure,
    same_color_only: bool = False,
    xval_pairs: list[tuple] | None = None,
    color_pairs: list[tuple] | None = None,
    stat_func: Callable = stats.ttest_ind,
    p_quantiles: tuple = (0.05, 0.01),
    x_offset_inc: float = 0.13,
    only_significant: bool = True,
) -> go.Figure:
    """
    Add significance indicators between box or violin plots

    Parameters
    ----------
    fig : go.Figure
        the figure to add the indicators to
    same_color_only: bool (True)
        only calculate significance between the same colors (legendgroups)
    xval_pairs: list[tuple] | None (None)
        specify pairs to consider for the significance calculation, if None,
        all combinations will be considered
    color_pairs: list[tuple] | None (None)
        specify colors to consider for the significance calculation, if None,
        all combinations will be considered.
        Only used if same_color_only == False.
    sig_func: Callable (scipy.stats.ttest_ind)
        the significance function to consider
    p_quantiles: tuple[float, float] ((0.05, 0.01))
        the quantiles to be considered for labeling with `*`, `**`, etc.
    x_offset_inc: float (0.05)
        basic offset between the legendgroups as this value cannot be retrieved
        from the traces...
    only_significant: bool (True)
        only show significant indicators, if False, all indicators will be shown
        with `ns` for non-significant

    Returns
    -------
    fig : go.Figure
        the figure with significance indicators added
    """

    # Consider only box and violin plots as distributions
    dists = [
        elm
        for elm in fig.data
        if isinstance(elm, _box.Box) or isinstance(elm, _violin.Violin)
    ]

    # extend to single distributions (one for each x value)
    xmap = get_map_xcat_to_linspace(fig)
    imap = {v: k for k, v in xmap.items()}

    fig = make_xaxis_numeric(fig)

    sdists = pd.DataFrame(
        [
            {
                "lgrp": (
                    elm["legendgroup"] if elm["legendgroup"] is not None else 1
                ),
                "x": xval,
                "xlabel": imap[xval],
                "y": elm["y"][
                    np.asarray(elm["x"]) == xval
                ],  # filter on x for different color pairs
            }
            for elm in dists
            for xval in np.unique(elm["x"])
        ]
    )
    # Make sure the x axis is reflected as numeric as we cannot draw lines
    # with offsets otherwise

    if same_color_only:
        color_pairs = [(cp, cp) for cp in sdists.lgrp.unique()]
    elif color_pairs is None:
        # get all pairs
        color_pairs = [
            (cp1, cp2)
            for i, cp1 in enumerate(sdists.lgrp.unique())
            for cp2 in sdists.lgrp.unique()[i:]
        ]

    dstats = compute_stats(sdists, xval_pairs, color_pairs, stat_func)

    # space occupied in min max range by each line
    line_width_frac = 0.05

    ymin = min([e.min() for e in sdists.y])
    ymax = max([e.max() for e in sdists.y])
    dy = ymax - ymin
    # draw the indicator lines
    yline = ymin - dy * line_width_frac
    for rowi, (c1, c2, x1, x2, (stat, pval), n1, n2) in dstats.iterrows():
        x1_offset = get_x_offset(fig, c1, x_offset_inc)
        x2_offset = get_x_offset(fig, c2, x_offset_inc)
        x1p = xmap[x1] + x1_offset
        x2p = xmap[x2] + x2_offset
        xmid = x1p + (x2p - x1p) / 2

        msk = [pval < pq for pq in p_quantiles]
        if not any(msk):
            sig_label = "ns<br>"  # add the <br> to offset position upwards
            if only_significant:
                continue

        elif all(msk):
            sig_label = "*" * len(msk)
        else:
            # get the first False
            sig_label = "*" * msk.index(False)

        # the line
        fig.add_trace(
            go.Scatter(
                x=[x1p, x2p],
                y=[yline, yline],
                mode="lines+markers",
                marker={"size": 10, "symbol": "line-ns", "line_width": 2},
                line_color="#555555",
                line_dash="dot",
                showlegend=False,
                hoverinfo="skip",  # disable hover
            )
        )

        # Marker for hover
        hovertemplate = (
            f"<b>{x1}</b> vs. <b>{x2}</b><br>"
            f"<b>test function</b>: {stat_func.__name__}"
            f"<br><b>N-dist1</b>: {n1}<br><b>N-dist2</b>: {n2}<br>"
            f"<b>statistic</b>: {stat}<br><b>pval</b>: {pval}<extra></extra>"
        )

        fig.add_trace(
            go.Scatter(
                x=[xmid],
                y=[yline],
                mode="text",
                text=[sig_label],
                showlegend=False,
                name=sig_label,
                marker_line_width=2,
                marker_size=10,
                hovertemplate=hovertemplate,
            )
        )

        # Offset next line
        yline -= dy * line_width_frac

    return fig


def get_num_x_pos(fig: go.Figure, xkey: str) -> float:
    """Get the numeric position for an x value on a categorical x axis"""
    xmap = get_map_xcat_to_linspace(fig)
    return xmap[xkey]


def get_x_offset(fig: go.Figure, cg_key: str, x_offset_inc: float) -> float:
    """Compute the x axis offset for a given color group"""

    # via dict to preserver order
    cgrps = list(
        dict.fromkeys(
            [trc.offsetgroup for trc in fig.data if "offsetgroup" in trc]
        )
    )

    if len(cgrps) == 0 or len(cgrps) == 1:
        return 0
    else:
        extend = (len(cgrps) - 1) / 2
        offsets = np.linspace(-extend, extend, len(cgrps)) * (
            x_offset_inc / extend
        )
        offsetmap = {k: v for k, v in zip(cgrps, offsets)}

        return offsetmap[cg_key]


def compute_stats(
    sdists: pd.DataFrame,
    xval_pairs: list[tuple] | None,
    color_pairs: list[tuple],
    stat_func: Callable,
) -> pd.DataFrame:
    """
    Compute a data frame storing the test statistic for
    color1, color2, x1, x2 comparison tuples
    """
    recs = []
    for cp1, cp2 in color_pairs:
        if xval_pairs is not None:
            wxval_pairs = xval_pairs
        else:
            # Build unique pairs accross the color groups
            if cp1 == cp2:
                uxvals = sdists[(sdists.lgrp == cp1)].xlabel.unique()
                wxval_pairs = [
                    (x1, x2)
                    for i, x1 in enumerate(uxvals)
                    for x2 in uxvals[i + 1 :]
                ]
            else:
                # consider all unique
                wxval_pairs = [
                    (x1, x2)
                    for x1 in sdists[(sdists.lgrp == cp1)].xlabel.unique()
                    for x2 in sdists[(sdists.lgrp == cp2)].xlabel.unique()
                ]

        for x1, x2 in wxval_pairs:
            if x1 != x2 or cp1 != cp2:
                dist1 = sdists[(sdists.lgrp == cp1) & (sdists.xlabel == x1)]
                dist2 = sdists[(sdists.lgrp == cp2) & (sdists.xlabel == x2)]

                if True:  # dist1.shape[0] >= 1 and dist2.shape[0] >= 1:
                    # print(f" >> {dist1=}, {dist2=}, {x1=}, {x2=}, {cp1=}, "
                    # f"{cp2=}")
                    recs.append(
                        {
                            "color1": cp1,
                            "color2": cp2,
                            "x1": x1,
                            "x2": x2,
                            "stat": stat_func(
                                dist1.y.iloc[0], dist2.y.iloc[0]
                            ),
                            "n1": len(dist1.y.iloc[0]),
                            "n2": len(dist2.y.iloc[0]),
                        }
                    )

    return pd.DataFrame(recs)


def apply_default_styles(
    fig: go.Figure,
    row: int | None = None,
    col: int | None = None,
    xzero: bool = True,
    yzero: bool = True,
    showgrid: bool = True,
    ygrid: bool = True,
    xgrid: bool = True,
    gridoptions: dict | None = dict(
        gridcolor="#444444", gridwidth=1, griddash="dot"
    ),  # options for dash are 'solid', 'dot', 'dash', 'longdash', 'dashdot',
    # or 'longdashdot'
    gridoptions_y: dict | None = None,
    gridoptions_x: dict | None = None,
) -> go.Figure:
    fig.update_xaxes(
        showgrid=showgrid,
        gridcolor="#444444",
        linecolor="#444444",
        col=col,
        row=row,
    )
    if xzero:
        fig.update_xaxes(zerolinecolor="#444444", row=row, col=col)

    fig.update_yaxes(
        showgrid=showgrid,
        gridcolor="#444444",
        zerolinecolor="#444444",
        linecolor="#444444",
        row=row,
        col=col,
    )
    if yzero:
        fig.update_yaxes(zerolinecolor="#444444", row=row, col=col)

    # Narrow margins large ticks for better readability
    tickfontsize = 20
    fig.update_layout(
        font=dict(size=tickfontsize),
        margin=dict(l=40, r=5, t=40, b=40),
        title=dict(x=0.5, xanchor="center"),
    )

    # clean background
    fig.update_layout(
        plot_bgcolor="#ffffff",  # 'rgba(0,0,0,0)',   # transparent bg
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_size=16),
    )

    # grid
    if showgrid:
        ygrid = True
        xgrid = True
    if gridoptions:
        gridoptions_x = gridoptions
        gridoptions_y = gridoptions

    if ygrid:
        fig.update_yaxes(
            **gridoptions_y,
            row=row,
            col=col,
        )

    if xgrid:
        fig.update_xaxes(
            **gridoptions_x,
            row=row,
            col=col,
        )

    return fig


def get_dareplane_colors() -> list[str]:
    return [
        "#0868ac",  # blue
        "#43a2ca",  # light blue
        "#7bccc4",  # green
        "#bae4bc",  # light green
        "#f0f9e8",  # lightest green
    ]


def make_xaxis_numeric(fig: go.Figure) -> go.Figure:
    xmap = get_map_xcat_to_linspace(fig)

    if fig.layout.xaxis["tickvals"] and list(xmap.values()) == list(
        fig.layout.xaxis["tickvals"]
    ):
        # nothing to do
        pass
    else:
        # replace x for each trace
        for trc in fig.data:
            trc.x = np.asarray([xmap[x] for x in trc.x])

        # make axis labels reflect the categories again
        fig.update_layout(
            xaxis=dict(
                tickmode="array",
                tickvals=list(xmap.values()),
                ticktext=list(xmap.keys()),
            )
        )

    return fig


def get_map_xcat_to_linspace(
    fig: go.Figure, xmin: int = 0, xmax: int = 1
) -> dict:
    """
    Create a dictionary mapping xlabels to a linspace from `xmin` to `xmax`.
    Two approaches need to be considers:
    1) Either all boxes are created with x-values (e.g. by
    plotly express) or 2) iteratively by adding traces potentially not carrying
    x values. In the latter case, we use the xaxis from the layout to
    sort the traces by their names.

    If xaxis is already numeric and labels are there, just use them instead.
    """

    # check if already numeric with labels
    xaxis = fig.layout.xaxis
    if (
        xaxis["ticktext"]
        and ["tickvals"]
        and all([isinstance(v, (int, float)) for v in xaxis["tickvals"]])
    ):
        d = {k: v for k, v in zip(xaxis["ticktext"], xaxis["tickvals"])}
    else:
        if all([trc.x is None for trc in fig.data]):
            for trc in fig.data:
                if trc.x is None:
                    trc.x = [trc.name] * len(trc.y)

        xgrps = list(
            dict.fromkeys(
                [u for trc in fig.data for u in np.unique(trc.x) if "x" in trc]
            )
        )
        xpos = np.linspace(0, 1, len(xgrps))

        d = {k: v for k, v in zip(xgrps, xpos)}

    return d


def scale_fig_for_paper(fig: go.Figure) -> go.Figure:
    fig = fig.update_layout(
        width=1100,
    )
    for annot in fig.layout.annotations:
        annot.font.size = 20

    return fig


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


def xdf_to_data_dict(
    xdf: Path, cfgs: dict = cfgs, tmax_s: float = 10_000
) -> dict:
    dreg = pyxdf.load_xdf(xdf, **cfgs["regular"])

    direg = pyxdf.load_xdf(xdf, **cfgs["irregular"])

    d = {}
    for k, source in {"reg": dreg[0], "ireg": direg[0]}.items():
        for s in source:
            try:
                if len(s["time_series"]) == 0:
                    # keep the empty stream values -> just to have the name in the
                    # dict keys
                    d[s["info"]["name"][0] + f"_{k}"] = {
                        "x": [],
                        "ts": [],
                        "esrate": s["info"]["effective_srate"],
                        "created_at": s["info"]["created_at"],
                    }
                else:
                    msk = (s["time_stamps"] - s["time_stamps"][0]) < tmax_s
                    d[s["info"]["name"][0] + f"_{k}"] = {
                        "x": (
                            np.asarray(s["time_series"]).flatten()
                            if len(s["time_series"]) == 1
                            or isinstance(s["time_series"][0][0], str)
                            else s["time_series"]
                        )[msk],
                        "ts": s["time_stamps"].flatten()[msk],
                        "esrate": s["info"]["effective_srate"],
                        "created_at": s["info"]["created_at"],
                    }
            except Exception as e:
                print(f"Error in {s['info']['name'][0]}: {e}")

    min_ts = min([min(v["ts"]) for v in d.values() if len(v["ts"]) >= 1])
    for v in d.values():
        v["ts"] -= min_ts

    return d


def find_matching(
    a: np.ndarray, b: np.ndarray, tol: float = 1.0, first: str = "a"
) -> tuple[np.ndarray, np.ndarray]:
    """Find closest matching indeces respecting a tolerance, also assume there is exactly one match"""
    if first == "a":
        x1 = a
        x2 = b
    else:
        x1 = b
        x2 = a

    idx1 = []
    idx2 = []
    for i1, t1 in enumerate(x1):
        d = x2 - t1
        causal_msk = d > 0  # ensure that chronological order is kept
        if any(causal_msk):
            j = np.argmin(d[causal_msk]) + np.where(causal_msk)[0][0]

            if (
                np.abs(x2[j] - t1) < tol
                and j not in idx2  # have match only once!
                and i1 not in idx1
            ):

                idx1.append(i1)
                idx2.append(j)

    return np.array(idx1), np.array(idx2)


def shifted_match_dt(
    ta: np.ndarray, tb: np.ndarray, xa: np.ndarray, xb: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Get matching indices for the test points if values match"""

    t1 = ta if ta[0] < tb[0] else tb
    x1 = xa if ta[0] < tb[0] else xb
    t2 = tb if ta[0] < tb[0] else ta
    x2 = xb if ta[0] < tb[0] else xa

    ix1, ix2 = t_match(t1, t2)

    vmatch = x1[ix1] == x2[ix2]

    dt = t2[ix2[vmatch]] - t1[ix1[vmatch]]

    tidx = np.where(dt < 0)[0]
    if (dt >= 0).any():
        print(
            "WARNIGN: Negative time differences detected at "
            f"t1={t1[ix1[vmatch]][tidx]}"
        )

    return dt


def t_match(t1: np.ndarray, t2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Aux function to get matching indeces of two arrays matching 1 vs 2"""
    tmap = abs(t1[:, None] - t2[None, :]).argmin(axis=-1)
    uv, c = np.unique(tmap, return_counts=True)
    idups = np.where(c > 1)[0]
    t1map = np.ones(t1.shape)
    for id in idups:
        idx = np.where(tmap == uv[id])[0]
        t1map[idx[:-1]] = -1
    # was achieved
    ix1 = np.where(t1map != -1)[0]
    ix2 = uv
    return ix1, ix2


def plot_around_time_points(
    d: dict,
    time_points: list[int],
    t_window_s: float = 1,
    facet_col_wrap: int = 4,
    key_filter: str = "ireg",
    scalings: dict = {},
    channels: dict = {},
) -> go.Figure:

    nplots = len(time_points)

    nrows = (nplots - 1) // facet_col_wrap + 1
    ncols = min(nplots, facet_col_wrap)

    fig = make_subplots(rows=nrows, cols=ncols)

    tmsks = []
    for tp in time_points:
        tmsks.append(
            {
                k: np.logical_and(
                    v["ts"] > tp - t_window_s, v["ts"] < tp + t_window_s
                )
                for k, v in d.items()
            }
        )

    sel_keys = [k for k in d.keys() if key_filter in k]
    cmap = dict(zip(sel_keys, sample_colorscale("jet", len(sel_keys))))

    # only plot numeric values
    for i, (tp, tmsk) in enumerate(zip(time_points, tmsks)):
        fig.add_trace(
            go.Scatter(
                x=[tp, tp],
                y=[-1, 1],
                mode="markers",
                marker=dict(color="red"),
                showlegend=False,
            ),
            row=i // ncols + 1,
            col=i % ncols + 1,
        )
        for k in sel_keys:
            v = d[k]
            scale = scalings.get(k, 1)
            chans = channels.get(k, [0])

            if v["x"].ndim < 2 or not np.issubdtype(
                v["x"][:, 0].dtype, np.number
            ):
                continue

            for ch_i in chans:
                fig.add_trace(
                    go.Scatter(
                        x=v["ts"][tmsk[k]],
                        y=v["x"][tmsk[k]][:, ch_i] * scale,
                        line=dict(color=cmap[k]),
                        name=k + "_" + str(ch_i),
                        legendgroup=k,
                        showlegend=i == 0,
                    ),
                    row=i // ncols + 1,
                    col=i % ncols + 1,
                )

    fig.update_layout(
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        )
    )
    fig = apply_default_styles(fig)

    return fig
