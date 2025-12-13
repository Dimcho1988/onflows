# cs_modulator.py
import numpy as np


def tau_of_delta(delta_v, tmin, k, q):
    """
    τ(Δv) = τ_min + k * (Δv)^q

    Parameters
    ----------
    delta_v : array-like or float
        Δv in km/h (we use max(delta_v, 0))
    tmin : float
        τ_min [s]
    k : float
        k parameter
    q : float
        q exponent

    Returns
    -------
    np.ndarray or float
        τ in seconds (same shape as delta_v)
    """
    delta_v = np.maximum(np.asarray(delta_v, dtype=float), 0.0)
    tmin = float(tmin)
    k = float(k)
    q = float(q)
    return tmin + k * (delta_v ** q)


def predict_t90_for_reference(CS, ref_percent, tmin, k, q):
    """
    For constant speed v = ref_percent * CS:
      t90 ≈ 2.303 * τ(Δv_ref)

    Returns
    -------
    dv_ref : float
    tau_ref : float
    t90_est : float
    """
    CS = float(CS)
    ref_percent = float(ref_percent)
    dv_ref = (ref_percent / 100.0 - 1.0) * CS
    tau_ref = float(tau_of_delta(dv_ref, tmin, k, q))
    t90_est = 2.303 * tau_ref
    return dv_ref, tau_ref, t90_est


def calibrate_k_for_target_t90(CS, ref_percent, tmin, q, target_t90):
    """
    Find k so that t90 at v = ref_percent * CS equals target_t90:
      t90 = 2.303 * (tmin + k * (Δv_ref)^q)
    """
    CS = float(CS)
    ref_percent = float(ref_percent)
    tmin = float(tmin)
    q = float(q)
    target_t90 = float(target_t90)

    dv_ref = (ref_percent / 100.0 - 1.0) * CS
    if dv_ref <= 0:
        return 0.0

    dvq = (dv_ref ** q) if dv_ref > 0 else 0.0
    tau_ref_needed = target_t90 / 2.303

    if dvq <= 0:
        return 0.0

    k_needed = (tau_ref_needed - tmin) / dvq
    return float(max(0.0, k_needed))


def apply_cs_modulation(v, dt, CS, tau_min, k_par, q_par, gamma):
    """
    CS modulation model.

    Inputs
    ------
    v  : np.array
        speed [km/h]
    dt : np.array
        time step per sample [s]
    CS, tau_min, k_par, q_par, gamma : float
        model params

    Returns dict
    ------------
    v_mod         : np.array  (v + gamma*r)
    delta_v_plus  : np.array  max(v - CS, 0)
    r             : np.array  "lift" [km/h]
    tau_s         : np.array  effective tau [s]
    """
    v = np.asarray(v, dtype=float)
    dt = np.asarray(dt, dtype=float)

    if v.shape != dt.shape:
        raise ValueError("v и dt трябва да са с еднаква дължина.")

    CS = float(CS)
    tau_min = float(tau_min)
    k_par = float(k_par)
    q_par = float(q_par)
    gamma = float(gamma)

    # safety: non-negative dt
    dt = np.maximum(dt, 1e-6)

    delta_v_plus = np.maximum(v - CS, 0.0)

    A = np.zeros_like(v)          # accumulated "debt"
    r = np.zeros_like(v)          # lift
    tau_series = np.zeros_like(v)

    tau_last = max(tau_min, 1e-6)

    for i in range(len(v)):
        dvp = delta_v_plus[i]

        if dvp > 0:
            tau_i = float(tau_of_delta(dvp, tau_min, k_par, q_par))
            tau_last = max(tau_i, 1e-6)
        else:
            tau_i = tau_last

        tau_series[i] = tau_i

        decay = float(np.exp(-dt[i] / tau_i))
        A_prev = 0.0 if i == 0 else float(A[i - 1])

        A[i] = A_prev * decay + dvp * dt[i]

        r[i] = (1.0 - decay) * A[i] / dt[i] if dt[i] > 0 else 0.0

    v_mod = v + gamma * r

    return {
        "v_mod": v_mod,
        "delta_v_plus": delta_v_plus,
        "r": r,
        "tau_s": tau_series,
    }
