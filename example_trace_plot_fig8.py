from pathlib import Path

import plotly.graph_objects as go
import pyxdf
from plotly.subplots import make_subplots

from paper_plots_aux import (apply_default_styles, get_dareplane_colors,
                             scale_fig_for_paper)

ON_COLOR = "#ac082c"
OFF_COLOR = "#0868ac"
ON_CL_COLOR = "#7bccc4"


def load_xdf(
    file: Path, all_from_zero: bool = False, dejitter_timestamps: bool = False
) -> dict:

    xdf_data, _ = pyxdf.load_xdf(
        file,
        dejitter_timestamps=dejitter_timestamps,
        # verbose=False,
    )
    stream_names = [s["info"]["name"][0] for s in xdf_data]

    # select just a few streams for the quick inspection
    selected_streams = [
        "AODataStream",
        "AOFctCallStream",
        "bollinger_bands",
        "decoded_ecog",
        "BollingerControlMarkerStream",
    ]

    data = {}
    for stream in selected_streams:
        s = xdf_data[stream_names.index(stream)]
        data[stream] = {"t": s["time_stamps"], "x": s["time_series"]}

    if all_from_zero:
        pass
        for d in data.values():
            if len(d["t"]) > 0:
                d["t"] -= d["t"][0]
    else:
        tmin = min([d["t"][0] for d in data.values() if len(d["t"]) > 0])
        for d in data.values():
            d["t"] -= tmin

    return data


def plot_closed_loop_example():
    file = Path("./data/block7_clcopydraw_on.xdf")

    # Using dejittered timestamps as this is only for visualization
    # and not for extracting precise timings. Due to compuational load,
    # the time-stamps of LSL are not correct anymore -> data should not be
    # loaded with dejittered timestamps if these segments are to be investigated
    data = load_xdf(file, all_from_zero=False, dejitter_timestamps=False)

    fig = make_subplots(
        5,
        1,
        shared_xaxes=True,
        vertical_spacing=0.0,
        row_heights=[0.1, 0.1, 0.1, 0.1, 0.6],
    )
    colors = get_dareplane_colors()

    times = data["AODataStream"]["t"]
    tmin = 12
    tmax = 35
    tmsk = (times > tmin) & (times < tmax)
    # tmsk = (times > 25) & (times < 35)

    for i, name in enumerate(["ECoG_1", "ECoG_2", "ECoG_3", "ECoG_4"]):
        fig.add_trace(
            go.Scatter(
                x=times[
                    tmsk
                ],  # np.linspace(times[tmsk][0], times[tmsk][-1], len(tmsk)),
                y=data["AODataStream"]["x"][tmsk, 16 + i],
                name=name,
                line_color=colors[i],
                opacity=0.7,
            ),
            row=1 + i,
            col=1,
        )

    times = data["bollinger_bands"]["t"]
    tmsk = (times > tmin) & (times < tmax)
    xbb = data["bollinger_bands"]["x"]

    fig = fig.add_trace(
        go.Scatter(
            x=times[tmsk],
            y=xbb[tmsk, 0],
            name="bollinger bands",
            line_dash="dash",
            line_color="black",
            opacity=0.3,
        ),
        row=5,
        col=1,
    )

    fig = fig.add_trace(
        go.Scatter(
            x=times[tmsk],
            y=xbb[tmsk, 2],
            name="bollinger bands",
            line_dash="dash",
            line_color="black",
            opacity=0.3,
            showlegend=False,
        ),
        row=5,
        col=1,
    )

    fig = fig.add_trace(
        go.Scatter(
            x=times[tmsk],
            y=xbb[tmsk, 1],
            name="controller input",
            line_color="black",
        ),
        row=5,
        col=1,
    )

    # mark trigger momements
    off_msk = xbb[tmsk, 1] > xbb[tmsk, 0]
    on_msk = xbb[tmsk, 1] < xbb[tmsk, 2]

    fig = fig.add_trace(
        go.Scatter(
            x=times[tmsk][on_msk],
            y=xbb[tmsk, 1][on_msk],
            text=[f"ON_{i}" for i in range(sum(on_msk))],
            name="ON trigger",
            marker_color=ON_COLOR,
            mode="markers",
            marker_size=8,
        ),
        row=5,
        col=1,
    )

    fig = fig.add_trace(
        go.Scatter(
            x=times[tmsk][off_msk],
            y=xbb[tmsk, 1][off_msk],
            text=[f"OFF_{i}" for i in range(sum(on_msk))],
            name="OFF trigger",
            marker_color=OFF_COLOR,
            mode="markers",
            marker_size=8,
        ),
        row=5,
        col=1,
    )

    # visually inspect after which marker started a grace period (needs to
    # change if time window is changed)
    gp_starts = [
        times[tmsk][on_msk][0],
        times[tmsk][off_msk][2],
        times[tmsk][on_msk][22],
        times[tmsk][off_msk][8],
        times[tmsk][on_msk][41],
        times[tmsk][off_msk][15],
    ]
    dt_gp = 2
    ydomain = fig.layout.yaxis5.domain
    showlegend = True
    for gpstart in gp_starts:
        fig.add_shape(
            type="rect",
            xref="x",
            yref="paper",
            x0=gpstart,
            y0=ydomain[0],
            x1=gpstart + dt_gp,
            y1=ydomain[1],
            line=dict(
                color="rgba(0,0,0,0)",
                width=3,
            ),
            showlegend=showlegend,
            name="grace period",
            fillcolor="#333",
            opacity=0.2,
            layer="below",
        )
        showlegend = False

    fig = apply_default_styles(fig)
    fig = fig.update_yaxes(
        title=dict(text="Potential [μV]"),
        row=3,
        col=1,
    )
    fig = fig.update_yaxes(
        title_text="Control signal [AU]",
        row=5,
        col=1,
        range=[-40, 40],
        # tickvals=[-40, -20, 0, 20, 40],
        tickvals=[-20, 0, 20, 45],
    )

    fig = fig.update_xaxes(range=[tmin, tmax])

    for i in range(1, 5):
        fig = fig.update_yaxes(
            range=[-800, 2000],
            row=i,
            col=1,
            # tickvals=[-500, 0, 500, 1000, 1500],
            tickvals=[0, 1000],
        )

    fig = fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(245, 245, 245, 0.9)",
        ),
        height=800,
        title_text="Sample traces of ECoG triggered aDBS",
    )

    fig = fig.update_xaxes(
        title_text="Time [s]",
        row=5,
        col=1,
        zeroline=False,
    )

    fig = scale_fig_for_paper(fig)

    fig.show()

    # fig.write_image(
    #     "./closed_loop_example_traces.svg"
    # )
