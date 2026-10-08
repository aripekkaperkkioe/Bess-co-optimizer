"""Main execution script for the BESS Rolling-Horizon Simulation.

This module coordinates the rolling-horizon optimization framework using Pyomo
and HiGHS. It handles market data ingestion, stochastic price scenario
generation, gate closure tracking (DA and reserves), and final performance
dashboard visualization.

Usage:
    python main.py
"""

import json
import logging
import numpy as np
import pandas as pd

from src.AncID_simulator import AncID_simulator
from src.RobustEnvelope import robust_envelopes
from src.BessOptimizer import BessOptimizer
from src.plotting import plot_simulation_results

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
# Suppress INFO logs from Pyomo's APPSI HiGHS wrapper
logging.getLogger("pyomo.contrib.appsi").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def run_rolling_simulation():

  # ==========================================
  # TUNABLE CONFIGURATION PARAMETERS
  # ==========================================
  # 1. Battery Specifications
  P_MAX = 20.0  # Maximum power rating (MW)
  E_CAP = 40.0  # Maximum energy capacity (MWh)
  ETA_CH = 0.96  # Charging efficiency
  ETA_DIS = 0.96  # Discharging efficiency
  initial_soc = 10.0  # Starting State of Charge (MWh)

  # 2. Simulation & Horizon Settings
  HORIZON_STEPS = 180  # Lookahead window (45 hours = 180 * 15-min steps), should be long enough to cover the next day before 8 am gate closure of Ancillary services
  MAX_SIMULATION_STEPS = 96 * 2  # Total simulation steps to run (e.g., 3 days)
  STEPS_PER_DAY = 96  # Number of 15-minute intervals per day (only works with this)

  # 3. Market, Data & Volatility Settings
  CONSTANT_RES_PRICE = 20.0  # Mock ancillary reserve price (€/MWh)
  SIGMA = 5.0  # Volatility parameter for OU price simulation
  TARGET_START_DATE = pd.Timestamp("2025-5-01 00:00:00") #Test data is the whole year 2015
  DATA_CACHE_PATH = "data/market_data_cache.json"

  # 4. Scenario Settings
  NUM_SCENARIOS = 2*1 # Number of stochastic ID & ancillary scenario paths. Must be an even number

  # ==========================================
  # Load market data cache #
  # ==========================================

  try:
    with open(DATA_CACHE_PATH, "r") as f:
      market_data = json.load(f)
  except FileNotFoundError:
    logger.error(f"Cache file '{DATA_CACHE_PATH}' not found.")
    return

  full_da_curve = np.array(market_data["da_curve"])
  full_timestamps = pd.to_datetime(market_data["timestamps"], utc=True)

  # Ensure timezone awareness matches if necessary
  if full_timestamps.tz is not None and TARGET_START_DATE.tz is None:
    target_start = TARGET_START_DATE.tz_localize(full_timestamps.tz)
  else:
    target_start = TARGET_START_DATE

  # 1. Find the starting index for your target date
  start_mask = full_timestamps >= target_start
  if not start_mask.any():
    logger.error(
        f"Target start date {TARGET_START_DATE} is beyond the range of cached"
        " timestamps."
    )
    return

  start_idx = int(np.argmax(start_mask))

  # 2. Cut the data cache to the simulation windows
  BUFFER_STEPS = 3 * STEPS_PER_DAY
  required_length = MAX_SIMULATION_STEPS + HORIZON_STEPS + BUFFER_STEPS
  end_idx = min(start_idx + required_length, len(full_timestamps))

  raw_da_curve = full_da_curve[start_idx:end_idx]
  master_timeline = full_timestamps[start_idx:end_idx]

  # Create artificial reserve price spike to check if the optimizer reacts aggressively
  res_price_array = np.full(len(raw_da_curve), CONSTANT_RES_PRICE)
  day_2_start = STEPS_PER_DAY
  spike_start = day_2_start + 16
  spike_end = day_2_start + 32
  res_price_array[spike_start:spike_end] = 400.0

  market_df = pd.DataFrame(
      {"da_price": raw_da_curve, "res_price": res_price_array},
      index=master_timeline,
  )

  # ==========================================
    # Rolling Horizon
  # ==========================================

  print("--- Starting BESS Rolling-Horizon Simulation (Pyomo + HiGHS) ---")

  total_simulation_steps = MAX_SIMULATION_STEPS
  horizon_hours = HORIZON_STEPS * 0.25
  print(
      f"Simulation configured to run for {total_simulation_steps} steps"
      f" ({total_simulation_steps / STEPS_PER_DAY:.1f} days) with a"
      f" {horizon_hours:.1f}h horizon."
  )

  # Initialize Global Registries
  global_sched_registry = pd.Series(0.0, index=master_timeline)
  global_res_registry = pd.Series(0.0, index=master_timeline)

  history_timestamps = []
  history_soc = []
  history_da_price = []
  history_id_price = []
  history_da_sched = []
  history_res = []
  history_realized_res = []
  history_id_trades = []
  history_cumulative_profit = []
  history_res_price = []  # Log reserve prices separately
  cumulative_profit = 0.0

  current_soc = initial_soc
  p_live = raw_da_curve[0]

  # Initialize Pyomo-Highs Optimizer
  bess = BessOptimizer(
      P_max=P_MAX,
      E_cap=E_CAP,
      eta_ch=ETA_CH,
      eta_dis=ETA_DIS,
      horizon_steps=HORIZON_STEPS,
      num_scenarios=NUM_SCENARIOS,
      dt=0.25,
  )

  # Containers for Gate Tracking & Logging
  step_increment = 1
  reserve_gate_history = []
  da_gate_history = []

  for global_t in range(0, total_simulation_steps, step_increment):
    current_time = master_timeline[global_t]

    window_df = market_df.iloc[global_t : global_t + HORIZON_STEPS]
    da_window = window_df["da_price"].to_numpy()
    res_window = window_df["res_price"].to_numpy()

    # Prepare fixed schedules from global registries
    fixed_sched_data = np.array(
        [global_sched_registry[ts] for ts in window_df.index]
    )
    fixed_res_data = np.array(
        [global_res_registry[ts] for ts in window_df.index]
    )

    # Run Scenario Generator
    scenarios = AncID_simulator(
        da_baseline=da_window,
        p_live=p_live,
        num_id_paths=NUM_SCENARIOS,
        sigma=SIGMA,
    )
    for sc in scenarios:
      sc["initial_soc"] = current_soc

    # Run Pyomo + HiGHS optimizer solve
    solve_output = bess.solve(
        horizon_steps=HORIZON_STEPS,
        current_timestamp=current_time,
        fixed_sched_data=fixed_sched_data,
        fixed_res_data=fixed_res_data,
        scenarios=scenarios,
        da_prices=da_window,
        res_prices=res_window,
        timestamps=window_df.index,
    )

    # Check optimization status for Pyomo/HiGHS
    if solve_output is None:
      raise RuntimeError(
          f"Optimization failed or returned non-optimal status at global step"
          f" {global_t} ({current_time})."
      )

    # Save the rolling horizon step and prepare the next one
    (
        model,
        P_sched,
        P_res,
        P_ch,
        P_dis,
        ID_ch,
        ID_dis,
        E,
        opt_emin,
        opt_emax,
    ) = solve_output

    is_res_gate_time = (current_time.hour == 8) and (current_time.minute == 0)
    is_da_gate_time = (current_time.hour == 12) and (current_time.minute == 0)

    # 1. 8:00 AM Reserve Gate Closure
    if is_res_gate_time:
      delivery_date = current_time.date() + pd.Timedelta(days=1)
      print(
          f"\n========================================\n"
          f"🔒 8:00 AM RESERVE GATE CLOSURE\n"
          f"Current Time: {current_time.strftime('%m-%d %H:%M')} | Locking Delivery Date:"
          f" {delivery_date}\n"
          f"----------------------------------------"
      )
      for t, ts in enumerate(window_df.index):
        if ts.date() == delivery_date:
          res_val = float(P_res[t])
          global_res_registry[ts] = res_val
      reserve_gate_history.append(current_time)

    # 2. 12:00 PM DA Gate Closure
    if is_da_gate_time:
      delivery_date = current_time.date() + pd.Timedelta(days=1)
      print(
          f"\n========================================\n"
          f"🔒 12:00 PM DA AUCTION CLOSURE\n"
          f"Current Time: {current_time.strftime('%m-%d %H:%M')} | Locking Delivery Date:"
          f" {delivery_date}\n"
          f"----------------------------------------"
      )
      for t, ts in enumerate(window_df.index):
        if ts.date() == delivery_date:
          sched_val = float(P_sched[t])
          global_sched_registry[ts] = sched_val
      da_gate_history.append(current_time)

    # --- Step Logging ---
    da_p = da_window[0]
    id_p = scenarios[0]["id_prices"][0]
    da_s = P_sched[0]
    res_v = P_res[0]
    res_p = res_window[0]

    print(
        f"[Step {global_t} | {current_time.strftime('%m-%d %H:%M')}] SoC: {E[0, 0]:.2f} MWh"
        f" | Powers (15 min): Capacity {P_res[0]/4:.2f}, DA {P_sched[0]/4:.2f},"
        f" ID {(ID_dis[0,0] - ID_ch[0,0])/4:.2f}"
        f" | Next Envelope [{opt_emin[1]:.2f}, {opt_emax[1]:.2f}] MWh"
    )

    alpha_0 = scenarios[0]["ancillary_activations"][0]
    realized_res_v = alpha_0 * res_v

    id_trade = ID_dis[0, 0] - ID_ch[0, 0]
    id_ch_val = ID_ch[0, 0]
    id_dis_val = ID_dis[0, 0]
    id_spread_coef = scenarios[0].get("id_spread_coef", 0.5)

    dt = 0.25
    market_rev = (da_s * da_p + res_v * res_window[0] + id_trade * id_p) * dt
    id_cost = id_spread_coef * (id_ch_val + id_dis_val) * dt
    step_cash_flow = market_rev - id_cost
    cumulative_profit += step_cash_flow

    current_soc = E[0, 1]
    p_live = scenarios[0]["id_prices"][1]

    history_timestamps.append(current_time)
    history_soc.append(current_soc)
    history_da_price.append(da_p)
    history_id_price.append(id_p)
    history_da_sched.append(da_s)
    history_res.append(res_v)
    history_realized_res.append(realized_res_v)
    history_id_trades.append(id_trade)
    history_cumulative_profit.append(cumulative_profit)
    history_res_price.append(res_p)

  print("\n--- Simulation Complete ---")

  # ==========================================
  # POST-PROCESS & PLOTTING
  # ==========================================

  print("\n--- Generating Simulation Dashboard ---")
  full_sched_arr = global_sched_registry.to_numpy()
  full_res_arr = global_res_registry.to_numpy()

  global_emin_full, global_emax_full = robust_envelopes(
      fixed_sched=full_sched_arr,
      fixed_res=full_res_arr,
      P_max=P_MAX,
      E_cap=E_CAP,
      eta_ch=ETA_CH,
      eta_dis=ETA_DIS,
      dt=0.25,
  )

  history_emin_rec = []
  history_emax_rec = []
  for ts in history_timestamps:
    idx = np.where(master_timeline == ts)[0][0]
    history_emin_rec.append(global_emin_full[idx + 1])
    history_emax_rec.append(global_emax_full[idx + 1])

  # Call the plotting function
  plot_simulation_results(
      history_timestamps=history_timestamps,
      history_da_price=history_da_price,
      history_id_price=history_id_price,
      history_res_price=history_res_price,
      history_da_sched=history_da_sched,
      history_res=history_res,
      history_realized_res=history_realized_res,
      history_id_trades=history_id_trades,
      history_soc=history_soc,
      history_emin_rec=history_emin_rec,
      history_emax_rec=history_emax_rec,
      history_cumulative_profit=history_cumulative_profit,
      reserve_gate_history=reserve_gate_history,
      da_gate_history=da_gate_history,
      E_cap=E_CAP,
  )


if __name__ == "__main__":
  run_rolling_simulation()