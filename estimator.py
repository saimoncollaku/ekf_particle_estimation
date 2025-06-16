from const import EstimatorConstant
import numpy as np
from typing import Tuple


import numpy as np
from scipy.linalg import cho_factor, cho_solve
from typing import Tuple

class EKF:
    """
    Optimized Extended Kalman Filter for vehicle state estimation.

    Args:
        estimator_constant : EstimatorConstant
            Constants known to the estimator.
    """

    def __init__(self, estimator_constant: EstimatorConstant):
        self.constant = estimator_constant

    def initialize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize the estimator with mean and covariance of the initial state.
        
        Returns:
            xm : Initial state mean vector [p_x, p_y, psi, tau, l]
            Pm : Initial state covariance matrix
        """
        est_const = self.constant
        # Initialize state mean with sensible defaults
        xm = np.array([
            0.0,
            0.0,
            0.0,
            est_const.start_velocity_bound / 2,
            (est_const.l_lb + est_const.l_ub) / 2
        ])
        # Compute variances for covariance matrix
        R = est_const.start_radius_bound
        var_px = var_py = (R ** 2) / 4
        var_psi = (est_const.start_heading_bound ** 2) / 3
        var_tau = (est_const.start_velocity_bound ** 2) / 12
        var_l = ((est_const.l_ub - est_const.l_lb) ** 2) / 12
        Pm = np.diag([var_px, var_py, var_psi, var_tau, var_l])
        return xm, Pm

    def estimate(
        self,
        xm_prev: np.ndarray,
        Pm_prev: np.ndarray,
        inputs: np.ndarray,
        measurement: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform EKF estimation step.
        
        Args:
            xm_prev : Previous state mean
            Pm_prev : Previous state covariance
            inputs : Control inputs [u_delta, u_c]
            measurement : Sensor measurements [z_px, z_py, z_psi, z_tau]
            
        Returns:
            xm : Updated state mean
            Pm : Updated state covariance
        """
        # Unpack inputs and constants
        u_delta_prev, u_c_prev = inputs
        px_prev, py_prev, psi_prev, tau_prev, l_prev = xm_prev
        Ts = self.constant.Ts
        
        # Precompute beta and trig terms
        beta_prev = np.arctan(0.5 * np.tan(u_delta_prev))
        angle = psi_prev + beta_prev
        sin_angle, cos_angle = np.sin(angle), np.cos(angle)
        sin_beta = np.sin(beta_prev)
        
        # State prediction
        px_pred = px_prev + tau_prev * cos_angle * Ts
        py_pred = py_prev + tau_prev * sin_angle * Ts
        psi_pred = psi_prev + (tau_prev / l_prev) * sin_beta * Ts
        tau_pred = tau_prev + u_c_prev * Ts
        l_pred = l_prev
        x_pred = np.array([px_pred, py_pred, psi_pred, tau_pred, l_pred])
        
        # Jacobian of process model (F)
        F = np.eye(5)
        F[0, 2] = -tau_prev * sin_angle * Ts
        F[0, 3] = cos_angle * Ts
        F[1, 2] = tau_prev * cos_angle * Ts
        F[1, 3] = sin_angle * Ts
        F[2, 3] = (sin_beta / l_prev) * Ts
        F[2, 4] = -(tau_prev * sin_beta) / (l_prev ** 2) * Ts
        
        # Process noise covariance (Q)
        Q = np.zeros((5, 5))
        Q[2, 2] = (self.constant.sigma_beta ** 2) * (Ts ** 2)
        Q[3, 3] = (self.constant.sigma_uc ** 2) * (Ts ** 2)
        
        # Predict covariance and enforce symmetry
        P_pred = F @ Pm_prev @ F.T + Q
        P_pred = (P_pred + P_pred.T) * 0.5
        
        # Measurement update if measurements available
        available = ~np.isnan(measurement)
        # Measurement model matrices
        H_full = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0]
        ])
        R_full = np.diag([
            self.constant.sigma_GPS ** 2,
            self.constant.sigma_GPS ** 2,
            self.constant.sigma_psi ** 2,
            self.constant.sigma_tau ** 2
        ])
        
        # Select available measurements
        H_avail = H_full[available]
        R_avail = R_full[np.ix_(available, available)]
        
        # computing kalman gain
        S = H_avail @ P_pred @ H_avail.T + R_avail
        A = H_avail @ P_pred
        L, lower = cho_factor(S)
        X = cho_solve((L, lower), A)
        K = X.T
        
        # updating state and covariance
        y = measurement[available] - H_avail @ x_pred
        x_updated = x_pred + K @ y
        P_updated = (np.eye(5) - K @ H_avail) @ P_pred
        P_updated = (P_updated + P_updated.T) * 0.5 # enforcing symmetry of P
        
        return x_updated, P_updated


class PF:
    """
    Optimized Particle Filter class

    Args:
        estimator_constant : EstimatorConstant
            Constants known to the estimator.
        noise : str
            Type of noise, either "Gaussian" or "Non-Gaussian".
    """
    def __init__(
            self,
            estimator_constant,
            noise: str,
    ):
        self.constant = estimator_constant
        self.num_particles = 1000  # you should fine tune this parameter
        self.roughening_factor = 1e-7
        self.noise = noise 
        
        # precomputing useful stuff
        self.sqrt3 = np.sqrt(3)
        self.sqrt3_sigma_beta = self.sqrt3 * self.constant.sigma_beta
        self.sqrt3_sigma_uc = self.sqrt3 * self.constant.sigma_uc
        self.sqrt3_sigma_GPS = self.sqrt3 * self.constant.sigma_GPS
        self.sqrt3_sigma_psi = self.sqrt3 * self.constant.sigma_psi
        self.sqrt3_sigma_tau = self.sqrt3 * self.constant.sigma_tau
        self.gauss_norm_GPS = 1 / (self.constant.sigma_GPS * np.sqrt(2 * np.pi))
        self.gauss_norm_psi = 1 / (self.constant.sigma_psi * np.sqrt(2 * np.pi))
        self.gauss_norm_tau = 1 / (self.constant.sigma_tau * np.sqrt(2 * np.pi))

    def initialize(self) -> np.ndarray:
        """
        Initialize the estimator with the particles.

        Returns:
            particles: np.ndarray, dim: (num_states, num_particles)
                The particles corresponding to the initial state estimate.
        """
        est_const = self.constant
        num_particles = self.num_particles

        # Vectorized initialization
        theta = np.random.uniform(0, 2 * np.pi, num_particles)
        radius = est_const.start_radius_bound * np.sqrt(np.random.uniform(0, 1, num_particles))
        px0 = radius * np.cos(theta)
        py0 = radius * np.sin(theta)
        psi0 = np.random.uniform(-est_const.start_heading_bound, est_const.start_heading_bound, num_particles)
        tau0 = np.random.uniform(0, est_const.start_velocity_bound, num_particles)
        l0 = np.random.uniform(est_const.l_lb, est_const.l_ub, num_particles)

        return np.vstack([px0, py0, psi0, tau0, l0])

    def estimate(
            self,
            particles: np.ndarray,
            inputs: np.ndarray,
            measurement: np.ndarray,
    ) -> np.ndarray:
        """
        Optimized state estimation using particle filter.

        Args:
            particles : np.ndarray, dim: (num_states, num_particles)
                Previous particles (k-1).
            inputs : np.ndarray, dim: (num_inputs,)
                System inputs u(k-1) = [u_delta, u_c].
            measurement : np.ndarray, dim: (num_measurement,)
                Sensor measurements z(k) = [z_px, z_py, z_psi, z_tau].

        Returns:
            posteriors : np.ndarray, dim: (num_states, num_particles)
                Posterior particles at time step k.
        """
        u_delta_prev, u_c_prev = inputs
        num_particles = particles.shape[1]
        Ts = self.constant.Ts

        # sampling process noise
        if self.noise == "Non-Gaussian":
            v_beta = np.random.uniform(-self.sqrt3_sigma_beta, self.sqrt3_sigma_beta, num_particles)
            v_uc = np.random.uniform(-self.sqrt3_sigma_uc, self.sqrt3_sigma_uc, num_particles)
        else:
            v_beta = np.random.normal(0, self.constant.sigma_beta, num_particles)
            v_uc = np.random.normal(0, self.constant.sigma_uc, num_particles)

        beta_prev = np.arctan(0.5 * np.tan(u_delta_prev))

        # predicting particles
        px_prev, py_prev, psi_prev, tau_prev, l_prev = particles
        cos_psi_beta = np.cos(psi_prev + beta_prev)
        sin_psi_beta = np.sin(psi_prev + beta_prev)
        sin_beta = np.sin(beta_prev)
        px_pred = px_prev + tau_prev * cos_psi_beta * Ts
        py_pred = py_prev + tau_prev * sin_psi_beta * Ts
        psi_pred = psi_prev + (tau_prev / l_prev * sin_beta) * Ts + v_beta * Ts
        tau_pred = tau_prev + (u_c_prev + v_uc) * Ts
        l_pred = l_prev + np.random.normal(0, 0.01, num_particles) # helping l for faster convergence
        predicted_particles = np.vstack([px_pred, py_pred, psi_pred, tau_pred, l_pred])


        available = ~np.isnan(measurement)
        log_weights = np.zeros(num_particles)
        
        # computing weights for gps
        if available[0] or available[1]:
            sigma = self.constant.sigma_GPS
            for i in (0, 1):
                if available[i]:
                    diff = measurement[i] - predicted_particles[i]
                    if self.noise == "Non-Gaussian":
                        term1 = np.exp(-2 * ((diff - (self.sqrt3_sigma_GPS/2)) / sigma)**2)
                        term2 = np.exp(-2 * ((diff + (self.sqrt3_sigma_GPS/2)) / sigma)**2)
                        log_weights += np.log((term1 + term2) * self.gauss_norm_GPS + 1e-10)
                    else:
                        log_weights += -0.5 * (diff**2) / (sigma**2) + np.log(self.gauss_norm_GPS)
        
        # computing weight for compass
        sigma = self.constant.sigma_psi
        diff = (measurement[2] - psi_pred + np.pi) % (2 * np.pi) - np.pi
        if self.noise == "Non-Gaussian":
            in_range = (diff >= -self.sqrt3_sigma_psi) & (diff <= self.sqrt3_sigma_psi)
            log_weights += np.log(in_range / (2 * self.sqrt3_sigma_psi) + 1e-10)
        else:
            log_weights += -0.5 * (diff**2) / (sigma**2) + np.log(self.gauss_norm_psi)
        
        # computing weight for tachometer
        sigma = self.constant.sigma_tau
        diff = measurement[3] - predicted_particles[3]
        if self.noise == "Non-Gaussian":
            in_range = (diff >= -self.sqrt3_sigma_tau) & (diff <= self.sqrt3_sigma_tau)
            log_weights += np.log(in_range / (2 * self.sqrt3_sigma_tau) + 1e-10)
        else:
            log_weights += -0.5 * (diff**2) / (sigma**2) + np.log(self.gauss_norm_tau)
        
        # normalizing weights
        max_log = np.max(log_weights)
        weights = np.exp(log_weights - max_log)
        weights_sum = np.sum(weights)
        weights = weights / weights_sum

        # here I use systematic resaampling for efficiency
        cumsum = np.cumsum(weights)
        step = 1.0 / num_particles
        u = (np.arange(num_particles) + np.random.uniform(0, step)) * step
        indices = np.searchsorted(cumsum, u)
        posteriors = predicted_particles[:, indices]

        # roughening
        d, N = posteriors.shape
        E = np.ptp(posteriors, axis=1)
        sigmas = self.roughening_factor * E * (N ** (-1/d))
        posteriors += np.random.normal(0, sigmas[:, None], (d, N))
        
        # clipping l
        posteriors[4] = np.clip(posteriors[4], self.constant.l_lb, self.constant.l_ub)

        return posteriors