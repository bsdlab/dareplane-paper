import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from plotly.subplots import make_subplots
from scipy import stats

from paper_plots_aux import (add_box_significance_indicator,
                             apply_default_styles, scale_fig_for_paper)

ON_COLOR = "#ac082c"
OFF_COLOR = "#0868ac"
ON_CL_COLOR = "#7bccc4"


# Auxiliary function
def create_box_and_significance(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    cmap = {"ON": ON_COLOR, "OFF": OFF_COLOR, "aDBS": ON_CL_COLOR}

    xlabels = []
    for gk, dg in df.groupby(["session", "stim"]):
        name = f"{gk[0]} {gk[1]}<br>(N={len(dg)})"
        fig.add_trace(
            go.Box(
                y=dg["final_label"],
                name=name,
                marker=dict(color=cmap[gk[1]]),
                boxmean=True,
                showlegend=False,
                boxpoints="all",
            ),
        )
        xlabels.append(name)

    xpairs = [
        ("day_2 OFF<br>(N=69)", "day_2 ON<br>(N=69)"),
        ("day_3 OFF<br>(N=43)", "day_3 ON<br>(N=70)"),
        ("day_4 OFF<br>(N=15)", "day_4 ON<br>(N=34)"),
        ("day_4 ON<br>(N=34)", "day_4 aDBS<br>(N=70)"),
        ("day_4 OFF<br>(N=15)", "day_4 aDBS<br>(N=70)"),
    ]

    fig = add_box_significance_indicator(
        fig,
        xval_pairs=xpairs,
        only_significant=False,
        x_offset_inc=0.3,
        stat_func=stats.mannwhitneyu,
    )

    fig = apply_default_styles(fig, xzero=False, yzero=False)
    return fig


def plot_behavioral_scores_fig9(df: pd.DataFrame):
    fig = make_subplots(
        2,
        2,
        subplot_titles=(
            "<b>(A)</b> Day 2 LDA CopyDraw projections",
            "<b>(B)</b> 6-fold chrono. CV",
            "<b>(C)</b> CopyDraw data projected with day 3 LDA model",
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.15,
        row_heights=[0.4, 0.6],
        column_widths=[0.75, 0.25],
        specs=[[{}, {}], [{"colspan": 2}, {}]],
    )

    df.stim = (
        df.stim.str.replace("on", "ON")
        .str.replace("off", "OFF")
        .str.replace("ON_cl", "aDBS")
    )

    # ---------- scatters
    d2_msk = df["session"] == "day_2"
    fig = fig.add_trace(
        go.Scatter(
            x=df.loc[d2_msk & (df.stim == "OFF"), "startTStamp"],
            y=df.loc[d2_msk & (df.stim == "OFF"), "final_label"],
            mode="markers",
            marker=dict(color=OFF_COLOR),
            name="Day 2 OFF",
        )
    )

    fig = fig.add_trace(
        go.Scatter(
            x=df.loc[d2_msk & (df.stim == "ON"), "startTStamp"],
            y=df.loc[d2_msk & (df.stim == "ON"), "final_label"],
            mode="markers",
            marker=dict(color=ON_COLOR),
            name="Day 2 ON",
        )
    )
    fig = fig.update_xaxes(
        title_text="Time [min]", row=1, col=1, range=[0, 100]
    )
    fig = fig.update_yaxes(
        title_text="LDA dec. func. [AU]",
        col=1,
    )
    fig = fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.74,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(245, 245, 245, 0.9)",
        ),
    )

    # ----- AUCROC
    dr = pd.read_hdf("./data/lda_cross_val_day2.hdf")
    dr["session"] = "day 2"
    dr2 = pd.read_hdf("./data/lda_cross_val_day3.hdf")
    dr2["session"] = "day 3"

    dr = pd.concat([dr, dr2])

    dr = dr[
        (dr.src_data == "normal_detrended")
        & (dr.proj == "decision_function")
        & (dr.train_data == "normal_detrended")
        & (dr.split != "all")
    ]

    dr["stim_val"] = dr.stim.map({"off": 0, "on": 1})
    dr["y_bin"] = dr.y.map(lambda x: 1 if x > 0 else 0)
    dr["acc"] = (dr.stim_val == dr.y_bin).astype(int)

    dg = dr.groupby(["session", "split"])["acc"].mean().reset_index()

    colors = {
        "day 2": "#333",
        "day 3": "#333",
    }
    for sess, dgg in dg.groupby("session"):
        fig = fig.add_trace(
            go.Box(
                y=dgg["acc"],
                name=f"{sess}",
                marker=dict(color=colors[sess]),
                boxmean=True,
                showlegend=False,
                boxpoints="all",
            ),
            row=1,
            col=2,
        )

    fig = fig.update_yaxes(
        title_text="Accuracy", tickformat=".0%", row=1, col=2
    )

    # ---------- boxplots

    # Have ON and OFF spelled capital
    bfig = create_box_and_significance(df)

    # have the traces for significance also display p-vals
    for tr in bfig.data:
        if tr.name == "**":
            pval = eval(
                re.search(r"pval</b>: ([^:^<]+)", tr.hovertemplate).group(1)
            )
            tr.text = (f"** (p={pval:.2e})<br>",)

    for i, tr in enumerate(bfig.data):
        # add the last to the second legend
        if tr.name and "day_4" in tr.name:
            tr.legend = "legend2"
            tr.name = "stim " + re.search(r"([^\s]+)<br>", tr.name).group(1)
            tr.showlegend = True

        fig.add_trace(tr, row=2, col=1)

    fig = fig.update_xaxes(
        ticktext=bfig.layout.xaxis["ticktext"],
        tickvals=bfig.layout.xaxis["tickvals"],
        row=2,
        col=1,
    )

    # First two annotations are subplot titles
    for i, annot in enumerate(fig.layout.annotations[:3]):
        annot.xanchor = "left"
        annot.x = 0.75 if "(B)" in annot.text else 0
        annot.y = annot.y + 0.01

    fig = fig.update_xaxes(tickangle=-45, row=2, col=1)
    fig = apply_default_styles(fig)
    fig = fig.update_xaxes(zeroline=False, row=2, col=1)
    fig = fig.update_yaxes(zeroline=False, row=2, col=1)
    fig = scale_fig_for_paper(fig)

    fig = fig.update_layout(
        legend2=dict(
            y=0.45,
            x=0.01,
            xanchor="left",
            bgcolor="rgba(245, 245, 245, 0.9)",
        ),
        height=1200,
    )

    fig.show()


def plot_model_compare_fig10():
    m1 = joblib.load("./data/model_day3.joblib")
    m2 = joblib.load("./data/model_day4.joblib")

    r1 = m1.named_steps["Regressor"].coef_
    r2 = m2.named_steps["Regressor"].coef_

    bands = [
        "[4-8]Hz",
        "[8-13]Hz",
        "[13-20]Hz",
        "[20-30]Hz",
        "[30-45]Hz",
        "[55-70]Hz",
    ]
    chans = ["ECoG_1", "ECoG_2", "ECoG_3", "ECoG_4"]
    bcs = np.asarray([f"{b} {c}" for c in chans for b in bands])

    # bcs reflects to input of the powers -> now sort to have bands combined
    barr = np.asarray([e.split(" ")[0] for e in bcs])
    idx = np.hstack([np.where(barr == b) for b in bands])[0]

    btitle = [b if b != "[4-8]Hz" else "<b>(A)</b> [4-8]Hz" for b in bands]
    fig = make_subplots(
        2,
        len(bands),
        subplot_titles=btitle
        + [
            "<b>(B)</b> Day 3 regression",
            "",
            "",
            "<b>(C)</b> Day 4 regression with day 3 model",
            "",
            "",
        ],
        shared_yaxes=True,
        specs=[
            [{}, {}, {}, {}, {}, {}],
            [
                {"colspan": 3},
                {},
                {},
                {"colspan": 3},
                {},
                {},
            ],
        ],
    )

    legendgrp = "grp1"
    for ic, b in enumerate(bands):
        nch = len(chans)
        showlegend = ic == 0
        x = [e.split(" ")[1] for e in bcs[idx][ic * nch : (ic + 1) * nch]]
        fig = fig.add_bar(
            x=x,
            y=r1[idx][ic * nch : (ic + 1) * nch],
            name="Day 3",
            marker_color=px.colors.qualitative.Plotly[0],
            legendgroup=legendgrp,
            showlegend=showlegend,
            row=1,
            col=ic + 1,
        )
        fig = fig.add_bar(
            x=x,
            y=r2[idx][ic * nch : (ic + 1) * nch],
            name="Day 4",
            marker_color=px.colors.qualitative.Plotly[2],
            legendgroup=legendgrp,
            showlegend=showlegend,
            row=1,
            col=ic + 1,
        )
    # Add the scatters and regression details
    fig = add_scatters(fig)

    fig = apply_default_styles(fig, xzero=False, yzero=False)
    fig = fig.update_yaxes(title="Ridge regression weight", row=1, col=1)
    fig = fig.update_xaxes(tickangle=90, row=1)
    fig = fig.update_xaxes(zerolinecolor="#444444", row=2)

    # color the backgrounds
    fig = fig.update_layout(
        width=1100,
        height=800,
    )

    for annot in fig.layout.annotations:
        annot.font.size = 20

    fig = fig.update_layout(
        legend=dict(
            y=0.73,
            x=1,
            xanchor="right",
            bgcolor="rgba(245, 245, 245, 0.9)",
        ),
    )

    fig.show()


def add_scatters(fig: go.Figure) -> go.Figure:
    dfs = []
    for i, f in enumerate(
        ["./data/wip_day3_ridge.hdf", "./data/wip_day4_ridge.hdf"]
    ):
        df = pd.read_hdf(f)
        df["session"] = f"day {i + 3}"
        dfs.append(df)

    df = pd.concat(dfs)

    # use the standard ols trendline from plotly express for brevity
    for i, (day, dg) in enumerate(df.groupby("session")):
        showlegend = True
        color = px.colors.qualitative.Plotly[i * 2]
        legend = f"legend{i+2}"
        fig.add_trace(
            go.Scatter(
                x=dg.ypred,
                y=dg.ytrue,
                mode="markers",
                marker=dict(color=color),
                showlegend=False,
            ),
            row=2,
            col=i * 3 + 1,
        )
        # flag outlier -> 3 stds away from the mean
        outl_msk = dg.ypred > dg.ypred.mean() + 3 * dg.ypred.std()

        fig.add_trace(
            go.Scatter(
                x=dg[outl_msk].ypred,
                y=dg[outl_msk].ytrue,
                mode="markers",
                name="outlier",
                marker=dict(
                    color="rgba(0, 0, 0, 0)",
                    size=8,
                    line=dict(color="#f33", width=2),
                ),
                showlegend=showlegend,
                legend=legend,
            ),
            row=2,
            col=i * 3 + 1,
        )

        # add trendline and confidence intervals
        model = sm.OLS(
            dg.ytrue[~outl_msk], sm.add_constant(dg.ypred[~outl_msk])
        ).fit()
        xspan = np.linspace(
            dg.ypred[~outl_msk].min(), dg.ypred[~outl_msk].max(), 100
        )
        new_data = sm.add_constant(xspan)
        result = model.get_prediction(new_data)
        dsumm = result.summary_frame()

        print(model.summary())

        fig.add_trace(
            go.Scatter(
                x=xspan,
                y=dsumm["mean"],
                mode="lines",
                line=dict(color="black"),
                name="OLS trendline",
                showlegend=showlegend,
                legend=legend,
            ),
            row=2,
            col=i * 3 + 1,
        )

        # the confidence intervals
        fig.add_trace(
            go.Scatter(
                x=xspan,
                y=dsumm["mean_ci_lower"],
                mode="lines",
                line=dict(color=color, width=1),
                showlegend=False,
                legend=legend,
            ),
            row=2,
            col=i * 3 + 1,
        )

        fig.add_trace(
            go.Scatter(
                x=xspan,
                y=dsumm["mean_ci_upper"],
                mode="lines",
                line=dict(color=color, width=1),
                name="Mean CI",
                fill="tonexty",
                showlegend=showlegend,
                legend=legend,
            ),
            row=2,
            col=i * 3 + 1,
        )
    fig = fig.update_yaxes(title="y_true", row=2, col=1)
    fig = fig.update_xaxes(title="y_pred", row=2)

    # position the legend
    fig = fig.update_layout(
        legend2=dict(
            y=0.23,
            x=0.48,
            xanchor="right",
            bgcolor="rgba(245, 245, 245, 0.9)",
        ),
        legend3=dict(
            y=0.23,
            x=1,
            xanchor="right",
            bgcolor="rgba(245, 245, 245, 0.9)",
        ),
    )

    return fig


def print_correlation_and_chance():
    dfs = []
    for i, f in enumerate(
        ["./data/wip_day3_ridge.hdf", "./data/wip_day4_ridge.hdf"]
    ):
        df = pd.read_hdf(f)
        df["session"] = f"day {i + 3}"
        dfs.append(df)

    df = pd.concat(dfs)

    mean_pearsonr = df.groupby("session")["pearsonr"].mean()

    # for the chance level
    boot_day3_meanr = np.load(
        "./data/bootstrap_n500_mean_r_day3_model_day3.npy"
    )
    boot_day4_meanr = np.load(
        "./data/bootstrap_n500_mean_r_day4_model_day3.npy"
    )
    cl_day3 = np.quantile(boot_day3_meanr, 0.95)
    cl_day4 = np.quantile(boot_day4_meanr, 0.95)

    print(f"{mean_pearsonr['day 3']=:.3f} [chance: {cl_day3:.3f}]")
    print(f"{mean_pearsonr['day 4']=:.3f} [chance: {cl_day4:.3f}]")


if __name__ == "__main__":
    df = pd.read_hdf("./data/behavioral_data_closed_loop.hdf")
    plot_model_compare_fig10()
