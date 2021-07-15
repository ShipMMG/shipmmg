#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_kt.

- pytest code of shipmmg/kt.py
"""

import numpy as np

from shipmmg.kt import simulate_kt, zigzag_test_kt


def kt_sim_result(test_kt_params):
    """Check shimmg.kt.simulate()."""
    duration = 50
    num_of_sampling = 1000
    time_list = np.linspace(0.00, duration, num_of_sampling)
    Ts = 50.0
    δ_list = 10 * np.pi / 180 * np.sin(2.0 * np.pi / Ts * time_list)
    result = simulate_kt(test_kt_params, time_list, δ_list, 0.0)
    assert result.success


def test_zigzag_test_kt(test_kt_params, tmpdir):
    """Check shimmg.kt.zigzag_test_kt()."""
    target_δ_rad = 20.0 * np.pi / 180.0
    target_ψ_rad_deviation = 20.0 * np.pi / 180.0
    duration = 500
    num_of_sampling = 50000
    time_list = np.linspace(0.00, duration, num_of_sampling)
    δ_list, r_list = zigzag_test_kt(
        test_kt_params,
        target_δ_rad,
        target_ψ_rad_deviation,
        time_list,
        δ_rad_rate=10.0 * np.pi / 180,
    )
