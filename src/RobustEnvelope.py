import numpy as np
import pyomo.environ as pyo


def robust_envelopes(
    fixed_sched: np.ndarray,
    fixed_res: np.ndarray,
    P_max: float,
    E_cap: float,
    eta_ch: float,
    eta_dis: float,
    dt: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
  """Computes and returns theoretical robust envelopes
  for given fixed capacity and DA schedules.
  """
  H = len(fixed_sched)

  # Build concrete model
  m = pyo.ConcreteModel("Envelope_Standalone")
  m.TIME = range(H)
  m.TIME_PLUS = range(H + 1)

  # --- Variables ---
  m.P_sched = pyo.Var(
      m.TIME, bounds=(-P_max, P_max), initialize=0.0, name="P_sched"
  )
  m.R_res = pyo.Var(m.TIME, bounds=(0.0, P_max), initialize=0.0, name="R_res")

  m.P_ub = pyo.Var(m.TIME, bounds=(-P_max, P_max), initialize=0.0, name="P_ub")
  m.P_lb = pyo.Var(m.TIME, bounds=(-P_max, P_max), initialize=0.0, name="P_lb")

  m.E_min_rec = pyo.Var(
      m.TIME_PLUS, bounds=(0.0, E_cap), initialize=0.0, name="E_min_rec"
  )
  m.E_max_rec = pyo.Var(
      m.TIME_PLUS, bounds=(0.0, E_cap), initialize=E_cap, name="E_max_rec"
  )

  # Linearization variables and binaries
  for i in range(1, 5):
    m.add_component(
        f"p_dis{i}", pyo.Var(m.TIME, bounds=(0.0, None), initialize=0.0)
    )
    m.add_component(
        f"p_ch{i}", pyo.Var(m.TIME, bounds=(0.0, None), initialize=0.0)
    )
    m.add_component(f"z{i}", pyo.Var(m.TIME, domain=pyo.Binary, initialize=0))

  # --- Lock Schedules---
  for t in range(H):
    m.P_sched[t].setlb(float(fixed_sched[t]))
    m.P_sched[t].setub(float(fixed_sched[t]))
    m.R_res[t].setlb(float(fixed_res[t]))
    m.R_res[t].setub(float(fixed_res[t]))

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

  # Upper Bound Envelopes
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

  # Lower Bound Envelopes
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

  m.rec_emax_to_emax = pyo.Constraint(
      m.TIME,
      rule=lambda model, t: (
          model.E_max_rec[t]
          + (model.p_ch1[t] * eta_ch - model.p_dis1[t] / eta_dis) * dt
          <= model.E_max_rec[t + 1]
      ),
  )
  m.rec_emax_to_emin = pyo.Constraint(
      m.TIME,
      rule=lambda model, t: (
          model.E_max_rec[t]
          + (model.p_ch2[t] * eta_ch - model.p_dis2[t] / eta_dis) * dt
          >= model.E_min_rec[t + 1]
      ),
  )
  m.rec_emin_to_emax = pyo.Constraint(
      m.TIME,
      rule=lambda model, t: (
          model.E_min_rec[t]
          + (model.p_ch3[t] * eta_ch - model.p_dis3[t] / eta_dis) * dt
          <= model.E_max_rec[t + 1]
      ),
  )
  m.rec_emin_to_emin = pyo.Constraint(
      m.TIME,
      rule=lambda model, t: (
          model.E_min_rec[t]
          + (model.p_ch4[t] * eta_ch - model.p_dis4[t] / eta_dis) * dt
          >= model.E_min_rec[t + 1]
      ),
  )

  # Maximize envelope width
  m.obj = pyo.Objective(
      expr=sum(m.E_max_rec[t] - m.E_min_rec[t] for t in range(H)),
      sense=pyo.maximize,
  )

  # --- Solve ---
  solver = pyo.SolverFactory("appsi_highs")
  solver.config.mip_gap = 0.01
  solver.config.load_solution = True

  results = solver.solve(m, tee=False)

  if not pyo.check_optimal_termination(results):
    print("--- Robust Envelope Warning: Non-Optimal Status ---")
    return None, None

  opt_emin = np.array([pyo.value(m.E_min_rec[t]) for t in range(H + 1)])
  opt_emax = np.array([pyo.value(m.E_max_rec[t]) for t in range(H + 1)])

  return opt_emin, opt_emax