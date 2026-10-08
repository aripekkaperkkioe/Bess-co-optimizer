import time
import numpy as np
import pandas as pd
import pyomo.environ as pyo


class BessOptimizer:
  """Pyomo-based BESS Co-Optimization Optimizer using HiGHS

  Features robust energy envelopes, block consistency, dynamic gate closures, and full
  internal shifted warm-starts.
  """

  def __init__(
      self,
      P_max: float,
      E_cap: float,
      eta_ch: float,
      eta_dis: float,
      horizon_steps: int = 208,
      num_scenarios: int = 2,
      dt: float = 0.25,
  ):
    self.P_max = P_max
    self.E_cap = E_cap
    self.eta_ch = eta_ch
    self.eta_dis = eta_dis
    self.horizon_steps = horizon_steps
    self.num_scenarios = num_scenarios
    self.dt = dt

    # Track previous solution for full warm-starts
    self.prev_solution = None

    # --- Initialize Solver ---
    self.solver = pyo.SolverFactory("appsi_highs")
    self.solver.config.mip_gap = 0.08
    self.solver.config.load_solution = True
    self.solver.config.warmstart = True

    # Initialize Pyomo Model
    self.model = pyo.ConcreteModel("BESS_Pyomo")
    m = self.model
    H = self.horizon_steps
    S = self.num_scenarios

    # Sets
    m.TIME = range(H)
    m.TIME_PLUS = range(H + 1)
    m.SCENARIOS = range(S)

    # --- 1. Create Variables ---
    m.P_sched = pyo.Var(
        m.TIME, bounds=(-P_max, P_max), initialize=0.0, name="P_sched"
    )
    m.R_res = pyo.Var(m.TIME, bounds=(0.0, P_max), initialize=0.0, name="R_res")

    m.P_ch = pyo.Var(
        m.SCENARIOS, m.TIME, bounds=(0.0, P_max), initialize=0.0, name="P_ch"
    )
    m.P_dis = pyo.Var(
        m.SCENARIOS, m.TIME, bounds=(0.0, P_max), initialize=0.0, name="P_dis"
    )
    m.u_var = pyo.Var(
        m.SCENARIOS,
        m.TIME,
        domain=pyo.Binary,
        initialize=0,
        name="u_var",
    )

    # ID and SoC variables
    m.ID_ch = pyo.Var(
        m.SCENARIOS,
        m.TIME,
        bounds=(0.0, P_max),
        initialize=0.0,
        name="ID_ch",
    )
    m.ID_dis = pyo.Var(
        m.SCENARIOS,
        m.TIME,
        bounds=(0.0, P_max),
        initialize=0.0,
        name="ID_dis",
    )
    m.E = pyo.Var(
        m.SCENARIOS,
        m.TIME_PLUS,
        bounds=(0.0, E_cap),
        initialize=0.0,
        name="E",
    )

    # Envelope Variables
    m.P_ub = pyo.Var(
        m.TIME, bounds=(-P_max, P_max), initialize=0.0, name="P_ub"
    )
    m.P_lb = pyo.Var(
        m.TIME, bounds=(-P_max, P_max), initialize=0.0, name="P_lb"
    )
    m.E_min_rec = pyo.Var(
        m.TIME_PLUS, bounds=(0.0, E_cap), initialize=0.0, name="E_min_rec"
    )
    m.E_max_rec = pyo.Var(
        m.TIME_PLUS,
        bounds=(0.0, E_cap),
        initialize=E_cap,
        name="E_max_rec",
    )

    # Linearization Variables for the envelopes
    m.p_dis1 = pyo.Var(
        m.TIME, bounds=(0.0, None), initialize=0.0, name="p_pos1"
    )
    m.p_ch1 = pyo.Var(m.TIME, bounds=(0.0, None), initialize=0.0, name="p_neg1")
    m.p_dis2 = pyo.Var(
        m.TIME, bounds=(0.0, None), initialize=0.0, name="p_pos2"
    )
    m.p_ch2 = pyo.Var(m.TIME, bounds=(0.0, None), initialize=0.0, name="p_neg2")
    m.p_dis3 = pyo.Var(
        m.TIME, bounds=(0.0, None), initialize=0.0, name="p_pos3"
    )
    m.p_ch3 = pyo.Var(m.TIME, bounds=(0.0, None), initialize=0.0, name="p_neg3")
    m.p_dis4 = pyo.Var(
        m.TIME, bounds=(0.0, None), initialize=0.0, name="p_pos4"
    )
    m.p_ch4 = pyo.Var(m.TIME, bounds=(0.0, None), initialize=0.0, name="p_neg4")

    # Linearization variables for the envelope transitions
    m.p_dis5= pyo.Var(m.SCENARIOS,
        m.TIME, bounds=(0.0, None), initialize=0.0, name="p_pos5")
    m.p_ch5 = pyo.Var(m.SCENARIOS,
        m.TIME, bounds=(0.0, None), initialize=0.0, name="p_neg5")
    m.p_dis6 = pyo.Var(m.SCENARIOS,
        m.TIME, bounds=(0.0, None), initialize=0.0, name="p_pos5")
    m.p_ch6 = pyo.Var(m.SCENARIOS,
        m.TIME, bounds=(0.0, None), initialize=0.0, name="p_neg6")

    # Binary Variables for the envelopes
    m.z1 = pyo.Var(m.TIME, domain=pyo.Binary, initialize=0, name="z_bin1")
    m.z2 = pyo.Var(m.TIME, domain=pyo.Binary, initialize=0, name="z_bin2")
    m.z3 = pyo.Var(m.TIME, domain=pyo.Binary, initialize=0, name="z_bin3")
    m.z4 = pyo.Var(m.TIME, domain=pyo.Binary, initialize=0, name="z_bin4")

    # Binary variables for envelope transitions E
    m.z5 = pyo.Var(m.SCENARIOS,
        m.TIME, domain=pyo.Binary, initialize=0, name="z_bin5")
    m.z6 = pyo.Var(m.SCENARIOS,
        m.TIME, domain=pyo.Binary, initialize=0, name="z_bin6")

    # --- 2. Build Static Base Constraints Once ---
    self._build_static_constraints()

  def _build_static_constraints(self):
    m = self.model
    H = self.horizon_steps
    S = self.num_scenarios
    dt = self.dt
    P_max = self.P_max

    # --- Constraints ---
    m.terminal_block = pyo.Constraint(expr=m.R_res[H - 1] == 0.0)

    m.pub_lower = pyo.Constraint(
        m.TIME,
        rule=lambda model, t: model.P_ub[t]
        >= -P_max + model.R_res[t] - model.P_sched[t],
    )
    m.pub_upper = pyo.Constraint(
        m.TIME,
        rule=lambda model, t: model.P_ub[t]
        <= P_max - model.R_res[t] - model.P_sched[t],
    )
    m.plb_lower = pyo.Constraint(
        m.TIME,
        rule=lambda model, t: model.P_lb[t]
        >= -P_max + model.R_res[t] - model.P_sched[t],
    )
    m.plb_upper = pyo.Constraint(
        m.TIME,
        rule=lambda model, t: model.P_lb[t]
        <= P_max - model.R_res[t] - model.P_sched[t],
    )

    M_val = 1.0 * P_max

    # Linearization: Upper Bound Envelopes
    m.lin1 = pyo.Constraint(
        m.TIME,
        rule=lambda model, t: model.p_dis1[t] - model.p_ch1[t]
        == -model.R_res[t] + model.P_ub[t] + model.P_sched[t],
    )
    m.lin1_dis = pyo.Constraint(
        m.TIME, rule=lambda model, t: model.p_dis1[t] <= M_val * model.z1[t]
    )
    m.lin1_ch = pyo.Constraint(
        m.TIME, rule=lambda model, t: model.p_ch1[t] <= M_val * (1 - model.z1[t])
    )

    m.lin2 = pyo.Constraint(
        m.TIME,
        rule=lambda model, t: model.p_dis2[t] - model.p_ch2[t]
        == model.R_res[t] + model.P_ub[t] + model.P_sched[t],
    )
    m.lin2_dis = pyo.Constraint(
        m.TIME, rule=lambda model, t: model.p_dis2[t] <= M_val * model.z2[t]
    )
    m.lin2_ch = pyo.Constraint(
        m.TIME, rule=lambda model, t: model.p_ch2[t] <= M_val * (1 - model.z2[t])
    )

    # Linearization: Lower Bound Envelopes
    m.lin3 = pyo.Constraint(
        m.TIME,
        rule=lambda model, t: model.p_dis3[t] - model.p_ch3[t]
        == -model.R_res[t] + model.P_lb[t] + model.P_sched[t],
    )
    m.lin3_dis = pyo.Constraint(
        m.TIME, rule=lambda model, t: model.p_dis3[t] <= M_val * model.z3[t]
    )
    m.lin3_ch = pyo.Constraint(
        m.TIME, rule=lambda model, t: model.p_ch3[t] <= M_val * (1 - model.z3[t])
    )
    m.lin4 = pyo.Constraint(
        m.TIME,
        rule=lambda model, t: model.p_dis4[t] - model.p_ch4[t]
        == model.R_res[t] + model.P_lb[t] + model.P_sched[t],
    )
    m.lin4_dis = pyo.Constraint(
        m.TIME, rule=lambda model, t: model.p_dis4[t] <= M_val * model.z4[t]
    )
    m.lin4_ch = pyo.Constraint(
        m.TIME, rule=lambda model, t: model.p_ch4[t] <= M_val * (1 - model.z4[t])
    )

    #Linearization: envelope transitions for E

    m.lin5 = pyo.Constraint(
        m.SCENARIOS,
        m.TIME,
        rule=lambda model, s, t:  model.p_dis5[s,t] - model.p_ch5[s,t]
        == -model.R_res[t] + model.ID_dis[s,t]-model.ID_ch[s,t] + model.P_sched[t],
    )
    m.lin5_dis = pyo.Constraint(
        m.SCENARIOS,
        m.TIME,
        rule=lambda model, s, t: model.p_dis5[s,t] <= M_val * model.z5[s,t]
    )
    m.lin5_ch = pyo.Constraint(
        m.SCENARIOS,
        m.TIME,
        rule=lambda model, s, t:  model.p_ch5[s,t] <= M_val * (1 - model.z5[s,t])
    )


    m.lin6 = pyo.Constraint(
        m.SCENARIOS,
        m.TIME,
        rule=lambda model, s, t: model.p_dis6[s,t] - model.p_ch6[s,t]
        == model.R_res[t] + model.ID_dis[s,t]-model.ID_ch[s,t] + model.P_sched[t],
    )
    m.lin6_dis = pyo.Constraint(
        m.SCENARIOS,
        m.TIME,
        rule=lambda model, s, t: model.p_dis6[s,t] <= M_val * model.z6[s,t])
    m.lin6_ch = pyo.Constraint(
        m.SCENARIOS,
        m.TIME,
        rule=lambda model, s, t: model.p_ch6[s,t] <= M_val * (1 - model.z6[s,t]))

    # Envelope transitions
    m.rec_emax_to_emax = pyo.Constraint(
        m.TIME,
        rule=lambda model, t: (
            model.E_max_rec[t]
            + (model.p_ch1[t] * self.eta_ch - model.p_dis1[t] / self.eta_dis)
            * dt
            <= model.E_max_rec[t + 1]
        ),
    )
    m.rec_emax_to_emin = pyo.Constraint(
        m.TIME,
        rule=lambda model, t: (
            model.E_max_rec[t]
            + (model.p_ch2[t] * self.eta_ch - model.p_dis2[t] / self.eta_dis)
            * dt
            >= model.E_min_rec[t + 1]
        ),
    )
    m.rec_emin_to_emax = pyo.Constraint(
        m.TIME,
        rule=lambda model, t: (
            model.E_min_rec[t]
            + (model.p_ch3[t] * self.eta_ch - model.p_dis3[t] / self.eta_dis)
            * dt
            <= model.E_max_rec[t + 1]
        ),
    )
    m.rec_emin_to_emin = pyo.Constraint(
        m.TIME,
        rule=lambda model, t: (
            model.E_min_rec[t]
            + (model.p_ch4[t] * self.eta_ch - model.p_dis4[t] / self.eta_dis)
            * dt
            >= model.E_min_rec[t + 1]
        ),
    )

    # Envelope bounds for transition E

    m.E_emax = pyo.Constraint(
        m.SCENARIOS,
        m.TIME,
        rule=lambda model, s, t: (
                model.E[s,t]
                + (model.p_ch5[s,t] * self.eta_ch - model.p_dis5[s,t] / self.eta_dis)
                * dt
                <= model.E_max_rec[t+1]
        ),
    )

    m.E_emin = pyo.Constraint(
        m.SCENARIOS,
        m.TIME,
        rule = lambda model, s, t: (
                model.E[s,t]
                + (model.p_ch6[s,t] * self.eta_ch - model.p_dis6[s,t] / self.eta_dis)
                * dt
                >= model.E_min_rec[t+1]
        ),
    )

    # Mutual Exclusion for Charging & Discharging
    m.vec_excl_dis = pyo.Constraint(
        m.SCENARIOS,
        m.TIME,
        rule=lambda model, s, t: model.P_dis[s, t] <= P_max * model.u_var[s, t],
    )
    m.vec_excl_ch = pyo.Constraint(
        m.SCENARIOS,
        m.TIME,
        rule=lambda model, s, t: model.P_ch[s, t]
        <= P_max * (1 - model.u_var[s, t]),
    )

    # Non-Anticipativity Constraints at the root node
    m.nac_root_id_ch = pyo.Constraint(
        range(1, S),
        rule=lambda model, s: model.ID_ch[s, 0] == model.ID_ch[0, 0],
    )
    m.nac_root_id_dis = pyo.Constraint(
        range(1, S),
        rule=lambda model, s: model.ID_dis[s, 0] == model.ID_dis[0, 0],
    )

    # State of Charge Dynamics Across Scenarios
    m.vectorized_soc_trans = pyo.Constraint(
        m.SCENARIOS,
        m.TIME,
        rule=lambda model, s, t: model.E[s, t + 1]
        == model.E[s, t]
        + (
            self.eta_ch * model.P_ch[s, t]
            - (1.0 / self.eta_dis) * model.P_dis[s, t]
        )
        * dt,
    )

    # Robust Envelope Bounds with Buffer
    eps_buffer = 0.01
    m.scen_bound_emin = pyo.Constraint(
        m.SCENARIOS,
        m.TIME_PLUS,
        rule=lambda model, s, t: model.E[s, t]
        >= model.E_min_rec[t] + eps_buffer,
    )

    m.scen_bound_emax = pyo.Constraint(
        m.SCENARIOS,
        m.TIME_PLUS,
        rule=lambda model, s, t: model.E[s, t]
        <= model.E_max_rec[t] - eps_buffer,
    )


  def solve(
      self,
      horizon_steps: int,
      current_timestamp: pd.Timestamp,
      fixed_sched_data: np.ndarray,
      fixed_res_data: np.ndarray,
      scenarios: list,
      da_prices: np.ndarray,
      res_prices: np.ndarray,
      dt: float = 0.25,
      timestamps: pd.DatetimeIndex = None,
  ):
    t_start = time.time()
    m = self.model
    S = self.num_scenarios
    H = self.horizon_steps

    if timestamps is None or len(timestamps) < H:
      raise ValueError(
          f"BESSOptimizerPyomo requires 'timestamps' of length at least {H}."
      )

    # --- 1. RESET DEFAULT BOUNDS FIRST ---
    for t in range(H):
      m.P_sched[t].setlb(-self.P_max)
      m.P_sched[t].setub(self.P_max)
      m.R_res[t].setlb(0.0)
      m.R_res[t].setub(self.P_max)

    # --- 1. APPLY FULL INTERNAL WARM START  ---
    if self.prev_solution is not None:
      try:
        shifted_sched = np.roll(self.prev_solution["P_sched"], -1)
        shifted_sched[-1] = shifted_sched[-2]
        shifted_sched = np.clip(shifted_sched, -self.P_max, self.P_max)

        shifted_res = np.roll(self.prev_solution["R_res"], -1)
        shifted_res[-1] = shifted_res[-2]
        shifted_res = np.clip(shifted_res, 0.0, self.P_max)

        for t in range(H):
          m.P_sched[t].set_value(float(shifted_sched[t]))
          m.R_res[t].set_value(float(shifted_res[t]))

        # Scenario binaries
        for s in range(S):
          if s < len(self.prev_solution["u_var"]):
            shifted_u = np.roll(self.prev_solution["u_var"][s], -1)
            shifted_u[-1] = shifted_u[-2]
            for t in range(H):
              m.u_var[s, t].set_value(int(shifted_u[t]))

        # Linearization binaries z1 through z4
        for k in range(1, 5):
          key = f"z{k}"
          if key in self.prev_solution:
            shifted_z = np.roll(self.prev_solution[key], -1)
            shifted_z[-1] = shifted_z[-2]
            for t in range(H):
              getattr(m, key)[t].set_value(int(shifted_z[t]))

        # Envelope records (continuous state warm-start, clipped)
        for env_key in ["E_min_rec", "E_max_rec"]:
          if env_key in self.prev_solution:
            shifted_env = np.roll(self.prev_solution[env_key], -1)
            shifted_env[-1] = shifted_env[-2]
            shifted_env = np.clip(shifted_env, 0.0, self.E_cap)
            for t in range(H + 1):
              getattr(m, env_key)[t].set_value(float(shifted_env[t]))

      except Exception as e:
        print(f"--- Internal warm-start skipped: {e} ---")

    # --- 2. Update Market Locks ---
    da_locked_mask = [False] * H
    res_locked_mask = [False] * H

    for t in range(H):
      t_time = timestamps[t]
      t_date = t_time.date()

      res_gate_deadline = pd.Timestamp.combine(
          t_date - pd.Timedelta(days=1), pd.Timestamp("08:00:00").time()
      )
      da_gate_deadline = pd.Timestamp.combine(
          t_date - pd.Timedelta(days=1), pd.Timestamp("12:00:00").time()
      )

      if current_timestamp.tz is not None:
        if res_gate_deadline.tz is None:
          res_gate_deadline = res_gate_deadline.tz_localize(
              current_timestamp.tz
          )
        if da_gate_deadline.tz is None:
          da_gate_deadline = da_gate_deadline.tz_localize(current_timestamp.tz)

      if current_timestamp > da_gate_deadline:
        da_locked_mask[t] = True
        val = float(fixed_sched_data[t])
        m.P_sched[t].setlb(val)
        m.P_sched[t].setub(val)

      if current_timestamp > res_gate_deadline:
        res_locked_mask[t] = True
        val = float(fixed_res_data[t])
        m.R_res[t].setlb(val)
        m.R_res[t].setub(val)

    # --- 3. Refresh Dynamic Block Consistency Constraints ---
    if hasattr(self, "dynamic_block_constrs") and self.dynamic_block_constrs:
      for c_name in list(self.dynamic_block_constrs.keys()):
        m.del_component(c_name)

    new_block_constrs = {}
    for t in range(H - 1):
      # Reserve block consistency (e.g. 4-hour FCR reserve blocks)
      if timestamps[t].hour // 4 == timestamps[t + 1].hour // 4:
        if not (res_locked_mask[t] and res_locked_mask[t + 1]):
          c_name = f"res_block_{t}"
          new_block_constrs[c_name] = pyo.Constraint(
              expr=m.R_res[t] == m.R_res[t + 1]
          )
          m.add_component(c_name, new_block_constrs[c_name])

    self.dynamic_block_constrs = new_block_constrs

    # --- 4. Update Initial States of Charge ---
    init_socs = np.array(
        [sc.get("initial_soc", 0.5 * self.E_cap) for sc in scenarios]
    )

    if hasattr(m, "init_soc_constr"):
      m.del_component("init_soc_constr")

    m.init_soc_constr = pyo.Constraint(
        m.SCENARIOS,
        rule=lambda model, s: model.E[s, 0] == init_socs[s],
    )

    # --- 5. Update Dynamic Scenario Constraints ---
    if hasattr(self, "dynamic_scenario_constrs") and self.dynamic_scenario_constrs:
      for c_name in list(self.dynamic_scenario_constrs.keys()):
        m.del_component(c_name)

    alphas_mat = np.array([sc["ancillary_activations"] for sc in scenarios])
    self.dynamic_scenario_constrs = {}

    for s in range(S):
      for t in range(H):
        c1_name = f"power_balance_{s}_{t}"
        c1 = pyo.Constraint(
            expr=m.P_dis[s, t] - m.P_ch[s, t]
            == m.P_sched[t]
            + (m.ID_dis[s, t] - m.ID_ch[s, t])
            + alphas_mat[s, t] * m.R_res[t]
        )
        m.add_component(c1_name, c1)
        self.dynamic_scenario_constrs[c1_name] = c1

        c2_name = f"id_bound_lower_{s}_{t}"
        c2 = pyo.Constraint(
            expr=m.ID_dis[s, t] - m.ID_ch[s, t]
            >= -self.P_max - m.P_sched[t] + m.R_res[t]
        )
        m.add_component(c2_name, c2)
        self.dynamic_scenario_constrs[c2_name] = c2

        c3_name = f"id_bound_upper_{s}_{t}"
        c3 = pyo.Constraint(
            expr=m.ID_dis[s, t] - m.ID_ch[s, t]
            <= self.P_max - m.P_sched[t] - m.R_res[t]
        )
        m.add_component(c3_name, c3)
        self.dynamic_scenario_constrs[c3_name] = c3

    # --- 5. Update Objective Function Coefficients ---
    da_prices_arr = np.array(da_prices)
    res_prices_arr = np.array(res_prices)
    id_prices_mat = np.array([sc["id_prices"] for sc in scenarios])
    probs = np.array([sc.get("probability", 1.0 / S) for sc in scenarios])
    deg_coeffs = np.array([sc.get("deg_cost_coef", 10) for sc in scenarios])
    spread_coeffs = np.array(
        [sc.get("id_spread_coef", 0.5) for sc in scenarios]
    )

    terminal_price = (
        float(np.percentile(da_prices, 75)) * 0.3 if len(da_prices) > 0 else 0.0
    )

    da_reserve_revenue = sum(
        (m.P_sched[t] * da_prices_arr[t] + m.R_res[t] * res_prices_arr[t]) * dt
        for t in range(H)
    )

    scen_net_values_expr = 0
    for s in range(S):
      scen_id_rev = sum(
          (m.ID_dis[s, t] - m.ID_ch[s, t]) * id_prices_mat[s, t] * dt
          for t in range(H)
      )
      scen_deg_cost = sum(
          deg_coeffs[s] * (m.P_ch[s, t] + m.P_dis[s, t]) * dt for t in range(H)
      )
      scen_spread_cost = sum(
          spread_coeffs[s] * (m.ID_ch[s, t] + m.ID_dis[s, t]) * dt
          for t in range(H)
      )
      scen_salvage = terminal_price * m.E[s, H]

      scen_net_values_expr += probs[s] * (
          scen_id_rev - scen_deg_cost - scen_spread_cost + scen_salvage
      )

    eps_reg = 0.5
    envelope_expansion_incentive = eps_reg * sum(
        m.E_max_rec[t] - m.E_min_rec[t] for t in range(H)
    )

    if hasattr(m, "obj"):
      m.del_component("obj")

    m.obj = pyo.Objective(
        expr=da_reserve_revenue
        + scen_net_values_expr
        + envelope_expansion_incentive,
        sense=pyo.maximize,
    )

    print(
        f"--- Python Build/Update Time: {time.time() - t_start:.4f} seconds ---"
    )

    # --- 6. Solve Model with HiGHS ---
    t_solve = time.time()
    results = self.solver.solve(m, tee=False)
    print(f"--- HiGHS Solve Time: {time.time() - t_solve:.4f} seconds ---")

    if not pyo.check_optimal_termination(results):
      print(f"--- Optimizer Warning: Non-Optimal Status ---")
      return m, None, None, None, None, None, None, None, None, None

    # Extract Results Arrays
    P_sched_val = np.array([pyo.value(m.P_sched[t]) for t in range(H)])
    P_res_val = np.array([pyo.value(m.R_res[t]) for t in range(H)])
    P_ch_val = np.array([pyo.value(m.P_ch[0, t]) for t in range(H)])
    P_dis_val = np.array([pyo.value(m.P_dis[0, t]) for t in range(H)])

    ID_ch_val = np.zeros((S, H))
    ID_dis_val = np.zeros((S, H))
    E_val = np.zeros((S, H + 1))

    for s in range(S):
      for t in range(H):
        ID_ch_val[s, t] = pyo.value(m.ID_ch[s, t])
        ID_dis_val[s, t] = pyo.value(m.ID_dis[s, t])
        E_val[s, t] = pyo.value(m.E[s, t])
      E_val[s, H] = pyo.value(m.E[s, H])

    opt_emin = np.array([pyo.value(m.E_min_rec[t]) for t in range(H + 1)])
    opt_emax = np.array([pyo.value(m.E_max_rec[t]) for t in range(H + 1)])

    # --- 7. Save State Internally for Next Step Warm-Start ---
    u_var_matrix = np.zeros((S, H), dtype=float)
    for s in range(S):
      for t in range(H):
        val = pyo.value(m.u_var[s, t])
        u_var_matrix[s, t] = float(int(val)) if val is not None else 0.0

    z_dict = {}
    for k in range(1, 5):
      z_vals = []
      for t in range(H):
        val = pyo.value(getattr(m, f"z{k}")[t])
        z_vals.append(float(int(val)) if val is not None else 0.0)
      z_dict[f"z{k}"] = np.array(z_vals, dtype=float)

    self.prev_solution = {
        "P_sched": P_sched_val.astype(float),
        "R_res": P_res_val.astype(float),
        "u_var": u_var_matrix,
        **z_dict,
        "E_min_rec": opt_emin.astype(float),
        "E_max_rec": opt_emax.astype(float),
    }

    return (
        m,
        P_sched_val,
        P_res_val,
        np.clip(P_ch_val, 0.0, self.P_max),
        np.clip(P_dis_val, 0.0, self.P_max),
        np.clip(ID_ch_val, 0.0, None),
        np.clip(ID_dis_val, 0.0, None),
        np.clip(E_val, 0.0, self.E_cap),
        np.clip(opt_emin[:H], 0.0, self.E_cap),
        np.clip(opt_emax[:H], 0.0, self.E_cap),
    )