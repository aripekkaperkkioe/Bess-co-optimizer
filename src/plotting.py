import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_simulation_results(
    history_timestamps,
    history_da_price,
    history_id_price,
    history_res_price,
    history_da_sched,
    history_res,
    history_realized_res,
    history_id_trades,
    history_soc,
    history_emin_rec,
    history_emax_rec,
    history_cumulative_profit,
    reserve_gate_history,
    da_gate_history,
    E_cap: float,
    output_path: str = "outputs/bess_simulation.png"
):
    """Generates and saves the 5-panel BESS rolling-horizon co-optimization dashboard."""

    if not history_timestamps:
        print("--- Plotting Warning: No history data to plot. ---")
        return

    # Define relative height ratios for the 5 subplots:
    # [Market Prices, Reserve Prices, Power Dispatch, Battery SoC, Cash Balance]
    height_ratios = [0.8, 0.8, 1.5, 0.8, 0.8]

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(14, 18),
        sharex=True,
        dpi=100,
        gridspec_kw={"height_ratios": height_ratios},
    )

    # Subplot 0: Market Prices
    axes[0].plot(
        history_timestamps, history_da_price, label="DA Price", color="tab:blue", lw=1.2
    )
    axes[0].plot(
        history_timestamps, history_id_price, label="ID Price (Simulated)", color="tab:orange", alpha=0.9, lw=1.2
    )
    axes[0].set_ylabel("€/MWh")
    axes[0].set_title("Market Prices (DA vs. Simulated ID)")
    axes[0].grid(True)
    axes[0].margins(y=0.05)

    # Subplot 1: Ancillary Reserve Market Prices
    axes[1].plot(
        history_timestamps, history_res_price, color="purple", lw=1.2
    )
    axes[1].set_ylabel("€/MW/h")
    axes[1].set_title("Ancillary Market Prices")
    axes[1].grid(True)
    axes[1].margins(y=0.05)

    # Subplot 2: Combined Power Dispatch
    axes[2].plot(
        history_timestamps, history_da_sched, label="DA Schedule", color="tab:green", lw=1.2
    )
    axes[2].plot(
        history_timestamps, history_res, label="Reserved Ancillaries", color="tab:purple", lw=1.2
    )
    axes[2].plot(
        history_timestamps, history_realized_res, label="Activated Power", color="purple", linestyle="--", alpha=0.8, lw=1.0
    )
    axes[2].plot(
        history_timestamps, history_id_trades, label="ID Trades", color="tab:red", lw=1.2
    )
    axes[2].axhline(y=0.0, color="k", linestyle=":", alpha=0.5)
    axes[2].set_ylabel("MW")
    axes[2].set_title("Power Dispatch")
    axes[2].grid(True)
    axes[2].set_ylim(-25, 25)

    # Subplot 3: Battery SoC Trajectory & Robust Envelope Shading
    axes[3].plot(
        history_timestamps, history_soc, label="Realized SoC", color="tab:brown", lw=1.5
    )
    axes[3].fill_between(
        history_timestamps,
        history_emin_rec,
        history_emax_rec,
        color="tab:brown",
        alpha=0.2,
        label=r"Robust Envelope",
    )
    axes[3].axhline(y=E_cap, color="r", linestyle="--", label=f"Max Cap")
    axes[3].axhline(y=0.0, color="k", linestyle="--", label="Min Cap")
    axes[3].set_ylabel("MWh")
    axes[3].set_title("Battery State of Charge")
    axes[3].grid(True)
    axes[3].set_ylim(-2, E_cap + 5)

    # Subplot 4: Cumulative Cash Balance
    axes[4].plot(
        history_timestamps, history_cumulative_profit, color="darkgreen", lw=1.5
    )
    axes[4].set_ylabel("€")
    axes[4].set_xlabel("Time")
    axes[4].set_title("Cumulative Cash Balance")
    axes[4].grid(True)
    axes[4].margins(y=0.1)

    # Plot Reserve Gate Closures (08:00 AM) across all subplots
    active_res_gc = [
        ts for ts in reserve_gate_history if history_timestamps[0] <= ts <= history_timestamps[-1]
    ]
    for idx, gc_time in enumerate(active_res_gc):
        gc_num = mdates.date2num(gc_time)
        for ax_i, ax in enumerate(axes):
            label_text = "Reserve auction" if (idx == 0 and ax_i == 0) else ""
            ax.axvline(x=gc_num, color="purple", linestyle="--", alpha=0.7, lw=1.0, label=label_text)

    # Plot DA Auction Closures (12:00 PM) across all subplots
    active_da_gc = [
        ts for ts in da_gate_history if history_timestamps[0] <= ts <= history_timestamps[-1]
    ]
    for idx, gc_time in enumerate(active_da_gc):
        gc_num = mdates.date2num(gc_time)
        for ax_i, ax in enumerate(axes):
            label_text = "DA Auction" if (idx == 0 and ax_i == 0) else ""
            ax.axvline(x=gc_num, color="tab:blue", linestyle=":", alpha=0.8, lw=1.2, label=label_text)

    # Format x-axis timestamps
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

    # Place legends cleanly outside to the right for all subplots
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(
                handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=0.9, fontsize=8
            )


    # Explicitly pin margins and layout
    fig.subplots_adjust(left=0.07, right=0.85, hspace=0.35, top=0.91, bottom=0.05)

    # Save and show
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight", pad_inches=0.15)
    print(f"Simulation plot saved successfully as '{output_path}'.")
    plt.show()