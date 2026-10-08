# Pyomo & HiGHS Rolling-Horizon BESS Simulation

A robust rolling-horizon optimization framework for Battery Energy Storage Systems (BESS) co-optimizing **Day-Ahead (DA)** energy markets, **Intraday (ID)** continuous trading, and **Ancillary Reserve Services** under stochastic market volatility.

---

## Key Features

* **Rolling-Horizon Optimization:** Solves multi-hour lookahead windows using **Pyomo** coupled with the high-performance **HiGHS** MILP solver.
* **Market Gate Closures:** Automatically enforces rigid daily market gate closure timelines (8:00 AM for reserves and 12:00 PM for Day-Ahead auctions).
* **Robust Envelope Guarantees:** Implements backward-propagated robust boundaries to guarantee physical safety against worst-case ancillary activations.
* **Stochastic Scenario Generation:** Models correlated price volatility and ancillary shock paths using Ornstein-Uhlenbeck processes (`AncID_simulator`).
* **Visual Dashboard:** Generates complete post-simulation plotting suites tracking state-of-charge, robust envelopes, market prices, and cumulative cash flows.

---

## Project Structure

```text
├── data/
│   └── market_data_cache.json    # Cached historical market curves & timestamps
├── docs/
│   └── BESS.pdf                  # Detailed mathematical formulation & derivations
├── src/
│   ├── __init__.py
│   ├── AncID_simulator.py        # Stochastic ID & ancillary scenario generator
│   ├── BessOptimizer.py          # Pyomo optimization model with HiGHS wrapper
│   ├── RobustEnvelope.py         # Robust boundary calculation functions
│   └── plotting.py               # Dashboard plotting utilities
├── main.py                       # Main execution script
└── README.md
```

---

## Installation & Prerequisites

### 1. Requirements
* **Python 3.10+**
* **HiGHS Solver** (accessible via Pyomo's APPSI interface or system PATH)

### 2. Install Python Dependencies
Run the following command in your terminal to install the required libraries:

```bash
pip install pyomo numpy pandas matplotlib scipy
```

*(Note: Ensure the HiGHS binary is installed and discoverable by Pyomo/APPSI in your environment).*

---

## Running the Simulation

To execute the rolling-horizon simulation and generate the performance dashboard, run:

```bash
python main.py
```

---

## Configuration

You can easily tweak technical parameters at the top of `main.py`:
* **`P_MAX` / `E_CAP`**: Battery power rating (MW) and energy capacity (MWh).
* **`ETA_CH` / `ETA_DIS`**: Charging and discharging efficiencies.
* **`HORIZON_STEPS`**: Optimization lookahead window size (default: 180 steps / 45 hours).
* **`SIGMA`**: Price volatility scaling factor for scenario generation.

---

## Documentation

For mathematical formulations and model specifications, check the [BESS PDF Documentation](docs/BESS.pdf).