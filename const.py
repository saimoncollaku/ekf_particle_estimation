import numpy as np


class EstimatorConstant:
    def __init__(self):
        # Parameter range
        self.l_lb = 1.0
        self.l_ub = 1.5

        # Process noise
        self.sigma_beta = 0.2  # default: 0.2, range: [0.1, 1]
        self.sigma_uc = 0.2    # default: 0.2, range: [0.1, 1]

        # Measurement noise
        self.sigma_GPS = 8  # default: 8, range: [4, 25]
        self.sigma_tau = 1
        self.sigma_psi = 30 / 180 * np.pi

        # Initial state
        self.start_radius_bound = 5  # R  default: 5, range: [2, 15]
        self.start_heading_bound = np.pi / 4  # psi_bar
        self.start_velocity_bound = 10  # tau_bar

        # Time 
        # DO NOT CHANGE
        self.N = 2000
        self.Ts = 0.01
