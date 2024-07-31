from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from paper_plots_aux import apply_default_styles, xdf_to_data_dict

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


def load_all_exp_data():
    xdfs = dict(
        NeuroOmega=Path("./data/AO_test.xdf"),
        EvalKit=Path("./data/CT_test.xdf"),
        ArduinoUno=Path("./data/Arduino_test.xdf"),
    )

    stream_map = {
        "NeuroOmega": "AODataStream",
        "EvalKit": "ct_bic",
        "ArduinoUno": "PICOSTREAM",
    }

    data = []
    for k, xdf in xdfs.items():
        d = xdf_to_data_dict(xdf, cfgs=cfgs, tmax_s=300)
        prefix = stream_map[k]

        df = pd.DataFrame(
            {
                "ts": d[f"{prefix}_ireg"]["ts"],
                "x": d[f"{prefix}_ireg"]["x"][:, 0],
            }
        )
        df["dejitter"] = False

        dfd = pd.DataFrame(
            {
                "ts": d[f"{prefix}_reg"]["ts"],
                "x": d[f"{prefix}_reg"]["x"][:, 0],
            }
        )

        dfd["dejitter"] = True
        da = pd.concat([df, dfd], axis=0)
        da["src"] = k
        data.append(da)

    df = pd.concat(data, axis=0)

    return df


def check_suitable_thresholds_for_chunking(df: pd.DataFrame):
    # For NeuroOmega 0.005 seems good ~ chunks about 14ms appart
    # EvalKit almost provides continuous stream, potential chunking ~1ms
    # PICOSTREAM cut off ~1.5ms
    # --> choose the same chunking criterion for all -> 1ms
    fig = px.scatter(
        df[(df.ts < 20) & (~df.dejitter)], x="ts", y="x", facet_row="src"
    )
    fig = apply_default_styles(fig)
    fig.show()


def plot_data_chunk_stats():
    df = load_all_exp_data()
    df.src = df.src.map(
        {
            "NeuroOmega": "Neuro Omega",
            "EvalKit": "BIC-EvalKit",
            "ArduinoUno": "Arduino Uno",
        }
    )

    dp = pd.concat(
        [
            dg.iloc[:100_000]
            for _, dg in df[(df.dejitter == False)].groupby("src")
        ]
    )

    fig = px.scatter(
        dp,
        x="ts",
        y="x",
        color="src",
        facet_col="dejitter",
    )
    fig.show()

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{}, {}], [{"colspan": 2}, None]],
        vertical_spacing=0.15,
        horizontal_spacing=0.2,
        subplot_titles=[
            "<b>(A)</b> Time between chunks",
            "<b>(B)</b> # samples per chunk",
            "<b>(C)</b> Example",
        ],
    )
    # t_example = [7.8, 7.9]
    t_example = [7.7, 7.8]

    stats = {}
    for i, (gk, dg) in enumerate(df.groupby(["src", "dejitter"])):
        if ~gk[1]:
            print(i)
            aux = np.zeros(dg["ts"].shape[0])
            aux[1:][np.diff(dg["ts"]) > 0.001] = 1
            chunk = np.cumsum(aux)
            dg["chunk"] = chunk

            # time between last sample of junk and first of next
            inter_chunk_times = (
                dg.loc[dg.chunk.drop_duplicates().index, "ts"][1:].to_numpy()
                - dg.loc[dg.chunk.drop_duplicates(keep="last").index, "ts"][
                    :-1
                ].to_numpy()
            )

            print(f"{inter_chunk_times.mean()=}, {inter_chunk_times.std()=}")
            fig.add_trace(
                go.Histogram(
                    x=inter_chunk_times,
                    name=gk[0],
                    histnorm="percent",
                    nbinsx=20,
                    marker_color=px.colors.qualitative.Plotly[i],
                ),
                row=1,
                col=1,
            )

            # Adding the counts
            cnt = dg.groupby("chunk")["x"].count()

            stats[gk] = {
                "cnt": {
                    "mean": cnt.mean(),
                    "std": cnt.std(),
                    "min": cnt.min(),
                    "max": cnt.max(),
                    "q01": np.quantile(cnt, 0.01),
                    "q99": np.quantile(cnt, 0.99),
                },
                "dt": {
                    "mean": inter_chunk_times.mean(),
                    "std": inter_chunk_times.std(),
                    "min": inter_chunk_times.min(),
                    "max": inter_chunk_times.max(),
                    "q01": np.quantile(inter_chunk_times, 0.01),
                    "q99": np.quantile(inter_chunk_times, 0.99),
                },
            }

            fig.add_trace(
                go.Histogram(
                    x=cnt.values,
                    name=gk[0] + "_counts",
                    histnorm="percent",
                    nbinsx=20,
                    marker_color=px.colors.qualitative.Plotly[i],
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

            # add the example trace
            tm = (dg["ts"] > t_example[0]) & (dg["ts"] < t_example[1])
            fig.add_trace(
                go.Scatter(
                    x=dg[tm].ts - t_example[0],
                    y=(dg[tm].x - dg[tm].x.mean()) / dg[tm].x.std(),
                    name=gk[0],
                    mode="lines+markers" if ~gk[1] else "lines",
                    marker_color=px.colors.qualitative.Plotly[i],
                    line_color="rgba(30, 30, 30, 0.2)",
                    showlegend=True if ~gk[1] else False,
                ),
                row=2,
                col=1,
            )

    fig = fig.update_xaxes(title="dt [s]", row=1, col=1)
    fig = fig.update_yaxes(title="% of chunks", row=1, col=1)
    fig = fig.update_yaxes(title="% of samples", row=1, col=2)
    fig = fig.update_xaxes(title="# samples", row=1, col=2)
    fig = fig.update_xaxes(title="time [s]", row=2, col=1)
    fig = fig.update_yaxes(title="z-scaled signal [uV]", row=2, col=1)

    fig = apply_default_styles(fig)
    fig = fig.update_layout(
        width=1100,
        height=800,
        legend=dict(
            yanchor="top",
            y=0.45,
            xanchor="right",
            x=1,
            bgcolor="rgba(245, 245, 245, 0.9)",
        ),
    )
    for annot in fig.layout.annotations:
        annot.font.size = 20

    fig.show()

    # stats overview to latex - table 3
    ds = []
    for (k, _), v in stats.items():
        da = pd.DataFrame(
            {
                "mean": [v["cnt"]["mean"], v["dt"]["mean"]],
                "std": [v["cnt"]["std"], v["dt"]["std"]],
                "min": [v["cnt"]["min"], v["dt"]["min"]],
                "max": [v["cnt"]["max"], v["dt"]["max"]],
                "q01": [v["cnt"]["q01"], v["dt"]["q01"]],
                "q99": [v["cnt"]["q99"], v["dt"]["q99"]],
            },
            index=["sample count", "dt_chunk"],
        )
        da["src"] = k
        ds.append(da)

    ds = pd.concat(ds, axis=0)
    ds = ds.reset_index().rename(columns={"index": "measure"})
    ds = ds[
        ["measure", "src", "mean", "std", "min", "max", "q01", "q99"]
    ].sort_values(["measure", "src"])

    ltx = ds.to_latex(
        formatters={"count": int},
        float_format="{:0.3f}".format,
        caption="Chunking of data",
        label="subtab:chunk_comparison",
    )
    print(ltx)
