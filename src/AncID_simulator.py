from typing import Optional
import numpy as np



def AncID_simulator(
    da_baseline: np.ndarray,
    p_live: float,
    num_id_paths: int = 4,
    theta: float = 0.75,
    sigma: float = 1.0,
    dt: float = 0.25,
    df: float = 5.0,  # Degrees of freedom for Student-t
    seed: Optional[int] = None,
) -> list:
  """Generates stochastic price scenario fans using an Ornstein-Uhlenbeck process
  centered around the DA baseline.
  """
  if num_id_paths % 2 != 0:
    raise ValueError(
        "num_id_paths must be an even integer when using antithetic sampling."
    )

  if df <= 2:
    raise ValueError(
        "Degrees of freedom (df) for Student-t must be greater than 2 to"
        " maintain finite variance."
    )

  if seed is not None:
    np.random.seed(seed)

  n_steps = len(da_baseline)
  scenarios_matrix = np.zeros((num_id_paths, n_steps))
  half_scen = num_id_paths // 2

  # Initial deviations at t=0 anchored to current ID price
  dev = np.full(num_id_paths, p_live - da_baseline[0])
  scenarios_matrix[:, 0] = p_live

  # Precompute OU constants
  decay = np.exp(-theta * dt)
  var_factor = sigma * np.sqrt((1 - np.exp(-2 * theta * dt)) / (2 * theta))
  t_scale_factor = np.sqrt((df - 2.0) / df)

  # Generate price path fans via OU process
  for t in range(1, n_steps):
    raw_t = np.random.standard_t(df, size=half_scen)
    z_half = raw_t * t_scale_factor
    z_antithetic = np.concatenate([z_half, -z_half])

    dev = dev * decay + var_factor * z_antithetic
    scenarios_matrix[:, t] = da_baseline[t] + dev

  scenarios = []
  prob_random_single = 1.0 / num_id_paths

  for i in range(num_id_paths):
    id_prices = scenarios_matrix[i]

    # Calculate relative (percentage) price deviation from DA baseline
    price_deviation = id_prices - da_baseline
    safe_baseline = np.maximum(np.abs(da_baseline), 1.0)
    relative_deviation = price_deviation / safe_baseline

    # Scale relative deviation for ancillary activations
    scaled_activation = relative_deviation * 2.0
    random_noise = np.random.normal(0.0, 1, size=n_steps)

    # Strictly capped between -1.0 and 1.0
    random_activations = np.clip(scaled_activation + random_noise, -1.0, 1.0)

    scenarios.append({
        "price_path_id": i,
        "probability": prob_random_single,
        "id_prices": id_prices,
        "ancillary_activations": random_activations,
    })

  return scenarios