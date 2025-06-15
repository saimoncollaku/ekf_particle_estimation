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
        if np.any(available):
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
            
            # Compute Kalman gain with robust inversion
            S = H_avail @ P_pred @ H_avail.T + R_avail
            A = H_avail @ P_pred
            
            # Cholesky decomposition for efficient solve
            L, lower = cho_factor(S)
            X = cho_solve((L, lower), A)
            K = X.T
            
            # Update state and covariance
            y = measurement[available] - H_avail @ x_pred
            x_updated = x_pred + K @ y
            P_updated = (np.eye(5) - K @ H_avail) @ P_pred
            P_updated = (P_updated + P_updated.T) * 0.5  # Enforce symmetry
        else:
            x_updated, P_updated = x_pred, P_pred
        
        return x_updated, P_updated


class PF:
    """
    Particle Filter class

    Args:
        estimator_constant : EstimatorConstant
            Constants known to the estimator.
        noise : str
            Type of noise, either "Gaussian" or "Non-Gaussian".
    """
    def __init__(
            self,
            estimator_constant: EstimatorConstant,
            noise: str,
    ):
        self.constant = estimator_constant
        self.num_particles = 800  # you should fine tune this parameter
        if noise == "Gaussian" or noise == "Non-Gaussian":
            self.noise = noise
        else:
            raise ValueError(
                "Noise type not supported, should be either Gaussian or "
                "Non-Gaussian!"
            )

    def initialize(self) -> np.ndarray:
        """
        Initialize the estimator with the particles.

        Returns:
            particles: np.ndarray, dim: (num_states, num_particles)
                The particles corresponding to the initial state estimate. The
                order of states is given by x = [p_x, p_y, psi, tau, l].
        """
        est_const = self.constant
        num_particles = self.num_particles

        # Initialize position uniformly within a circle of radius R
        theta = np.random.uniform(0, 2 * np.pi, num_particles)
        radius = est_const.start_radius_bound * np.sqrt(np.random.uniform(0, 1, num_particles))
        px0 = radius * np.cos(theta)
        py0 = radius * np.sin(theta)

        # Initialize heading uniformly
        psi0 = np.random.uniform(
            -est_const.start_heading_bound,
            est_const.start_heading_bound,
            num_particles
        )

        # Initialize velocity uniformly
        tau0 = np.random.uniform(0, est_const.start_velocity_bound, num_particles)

        # Initialize parameter l uniformly
        l0 = np.random.uniform(est_const.l_lb, est_const.l_ub, num_particles)

        particles = np.vstack([px0, py0, psi0, tau0, l0])
        return particles

    def estimate(
            self,
            particles: np.ndarray,
            inputs: np.ndarray,
            measurement: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate the state of the vehicle.

        Args:
            particles : np.ndarray, dim: (num_states, num_particles)
                The posteriors of the particles of the previous time step k-1.
                The order of states is given by x = [p_x, p_y, psi, tau, l].
            inputs : np.ndarray, dim: (num_inputs,)
                System inputs from time step k-1, u(k-1). The order of the
                inputs is given by u = [u_delta, u_c].
            measurement : np.ndarray, dim: (num_measurement,)
                Sensor measurements from time step k, z(k). The order of the
                measurements is given by z = [z_px, z_py, z_psi, z_tau].

        Returns:
            posteriors : np.ndarray, dim: (num_states, num_particles)
                The posterior particles at time step k. The order of states is
                given by x = [p_x, p_y, psi, tau, l].
        """
        u_delta_prev, u_c_prev = inputs
        num_particles = particles.shape[1]
        Ts = self.constant.Ts

        # Sample process noise
        if self.noise == "Non-Gaussian":
            v_beta = np.random.uniform(
                -np.sqrt(3) * self.constant.sigma_beta,
                np.sqrt(3) * self.constant.sigma_beta,
                num_particles
            )
            v_uc = np.random.uniform(
                -np.sqrt(3) * self.constant.sigma_uc,
                np.sqrt(3) * self.constant.sigma_uc,
                num_particles
            )
        else:
            v_beta = np.random.normal(0, self.constant.sigma_beta, num_particles)
            v_uc = np.random.normal(0, self.constant.sigma_uc, num_particles)

        # Compute beta_prev
        beta_prev = np.arctan(0.5 * np.tan(u_delta_prev))

        # Extract previous states
        px_prev, py_prev, psi_prev, tau_prev, l_prev = particles

        # Predict state
        px_pred = px_prev + tau_prev * np.cos(psi_prev + beta_prev) * Ts
        py_pred = py_prev + tau_prev * np.sin(psi_prev + beta_prev) * Ts
        psi_pred = psi_prev + (tau_prev / l_prev * np.sin(beta_prev)) * Ts + v_beta * Ts
        tau_pred = tau_prev + (u_c_prev + v_uc) * Ts
        l_pred = l_prev + np.random.normal(0, 0.003, num_particles) # !

        predicted_particles = np.vstack([px_pred, py_pred, psi_pred, tau_pred, l_pred])

        # Compute weights
        available = ~np.isnan(measurement)
        if not np.any(available):
            weights = np.ones(num_particles) / num_particles
        else:
            log_weights = np.zeros(num_particles)
            for i, m in enumerate(available):
                if not m:
                    continue
                z = measurement[i]
                if i in [0, 1]:  # GPS (px or py)
                    sigma = self.constant.sigma_GPS
                    diff = z - predicted_particles[i, :]
                    if self.noise == "Non-Gaussian":
                        term1 = np.exp(-0.5 * ((diff - (np.sqrt(3)*sigma/2)) / (sigma/2))**2)
                        term2 = np.exp(-0.5 * ((diff + (np.sqrt(3)*sigma/2)) / (sigma/2))**2)
                        likelihood = (term1 + term2) / (sigma * np.sqrt(2 * np.pi))
                    else:
                        likelihood = np.exp(-0.5 * (diff**2) / sigma**2) / (sigma * np.sqrt(2 * np.pi))
                    log_weights += np.log(likelihood + 1e-10)
                elif i == 2:  # Compass (psi)
                    sigma = self.constant.sigma_psi
                    diff = (z - psi_pred + np.pi) % (2 * np.pi) - np.pi
                    # diff = z - predicted_particles[2, :]
                    if self.noise == "Non-Gaussian":
                        a, b = -np.sqrt(3)*sigma, np.sqrt(3)*sigma
                        in_range = (diff >= a) & (diff <= b)
                        likelihood = in_range.astype(float) / (b - a)
                    else:
                        likelihood = np.exp(-0.5 * (diff**2) / sigma**2) / (sigma * np.sqrt(2 * np.pi))
                    log_weights += np.log(likelihood + 1e-10)
                elif i == 3:  # Tachometer (tau)
                    sigma = self.constant.sigma_tau
                    diff = z - predicted_particles[3, :]
                    if self.noise == "Non-Gaussian":
                        a, b = -np.sqrt(3)*sigma, np.sqrt(3)*sigma
                        in_range = (diff >= a) & (diff <= b)
                        likelihood = in_range.astype(float) / (b - a)
                    else:
                        likelihood = np.exp(-0.5 * (diff**2) / sigma**2) / (sigma * np.sqrt(2 * np.pi))
                    log_weights += np.log(likelihood + 1e-10)
            # Normalize weights
            max_log = np.max(log_weights)
            weights = np.exp(log_weights - max_log)
            weights_sum = np.sum(weights)
            if weights_sum == 0:
                weights = np.ones(num_particles) / num_particles
            else:
                weights /= weights_sum

        # Resample particles
        indices = np.random.choice(num_particles, num_particles, p=weights)
        posteriors = predicted_particles[:, indices]

        return posteriors
