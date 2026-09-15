"""ecalc: per-kernel GPU energy database study.

Energy model:
    E_kernel = E_dyn(kernel) + P_static * t(kernel)
    E_model  = sum_k E_dyn(k) + P_static * T_model
"""

__version__ = "0.1.0"
