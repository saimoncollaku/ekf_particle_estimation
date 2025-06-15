from const import EstimatorConstant
from simulator import Simulator
from estimator import EKF, PF

import numpy as np
import matplotlib.pyplot as plt
import time


def evaluate(
        estimates_ekf: np.ndarray,
        estimates_pf: np.ndarray,
        simulator,
):
    """
    Calculate the RMSE of all estimates.
    """
    ekf_implimented = np.all(np.logical_not(np.isnan(estimates_ekf)))
    pf_implimented = np.all(np.logical_not(np.isnan(estimates_pf)))

    if ekf_implimented:
        rmse = np.sqrt(np.mean((estimates_ekf - simulator.states) ** 2, axis=1))
        print(f"RMSE of EKF for {simulator.noise} noise:")
        print(f"  position in x: {rmse[0]}")
        print(f"  position in y: {rmse[1]}")
        print(f"  heading psi: {rmse[2]}")
        print(f"  velocity tau: {rmse[3]}")
        print(f"  parameter l: {rmse[4]}")
    
    if pf_implimented:
        rmse = np.sqrt(
            np.mean((np.mean(estimates_pf, axis=2).T - simulator.states) ** 2,
                    axis=1))
        print(f"RMSE of PF for {simulator.noise} noise:")
        print(f"  position in x: {rmse[0]}")
        print(f"  position in y: {rmse[1]}")
        print(f"  heading psi: {rmse[2]}")
        print(f"  velocity tau: {rmse[3]}")
        print(f"  parameter l: {rmse[4]}")


def plot(
        measurements: np.ndarray,
        estimates_ekf: np.ndarray,
        estimates_pf: np.ndarray,
        simulator,
):
    """
    Plot the measurements and estimates.
    """
    ekf_implimented = np.all(np.logical_not(np.isnan(estimates_ekf)))
    pf_implimented = np.all(np.logical_not(np.isnan(estimates_pf)))

    t = np.arange(0, simulator.constant.N)

    num_particles = estimates_pf.shape[2]
    num_show = min(num_particles, 10)
    tpf = np.tile(
        np.arange(0, simulator.constant.N), (num_show, 1)).T.reshape(-1)

    labels = ["$p_{x_c}$", "$p_{y_c}$", "$\\psi$", "$\\tau$"]

    for i in range(4):
        ax = plt.subplot(3, 2, i+1)
        ax.set_ylabel(labels[i])
        plt.plot(
            t, measurements[i, :], label="measurements", alpha=0.5, c='g',
        )
        if ekf_implimented:
            plt.plot(t, estimates_ekf[i, :], label="EKF", alpha=0.7, c='r')
        if pf_implimented:
            plt.scatter(
                tpf, estimates_pf[:, i, :num_show].reshape(1, -1), label="PF",
                alpha=0.1, c='b', s=1,
            )
        plt.legend()

    ax5 = plt.subplot(3, 2, 5)
    ax5.set_ylabel("$l$")
    if ekf_implimented:
        plt.plot(t, estimates_ekf[4, :], label="EKF", alpha=0.7, c='r')
        plt.legend()
    if pf_implimented:
        plt.scatter(
            tpf, estimates_pf[:, 4, :num_show].reshape(1, -1), label="PF",
            alpha=0.1, c='b', s=1,
        )
        plt.legend()

    plt.show()


if __name__ == "__main__":
    # Set random seed
    np.random.seed()

    # State definition
    num_states = 5  # [x, y, psi, tau, l]
    noise_type = "Gaussian"  # "Gaussian" or "Non-Gaussian"

    # Import constants
    est_const = EstimatorConstant()

    # Create simulator and estimator
    simulator = Simulator(noise=noise_type)
    ekf = EKF(est_const)
    pf = PF(est_const, noise=noise_type)

    # Run the
    print("Simulating system...")
    simulator.update_const(est_const)
    simulator.simulate()

    states = simulator.states
    inputs = simulator.inputs
    measurements = simulator.measurements

    # EKF containers
    estimates_ekf = np.zeros((num_states, est_const.N))
    variances_ekf = np.zeros((est_const.N, num_states, num_states))

    # PF containers
    estimates_pf = np.zeros((est_const.N, num_states, pf.num_particles))

    # Initialize the estimator
    estimates_ekf[:, 0], variances_ekf[0, :, :] = ekf.initialize()
    estimates_pf[0, :, :] = pf.initialize()

    # Run the EKF
    print("Running EKF...")
    start = time.time()
    for k in range(1, est_const.N):
        # EKF
        estimates_ekf[:, k], variances_ekf[k, :, :] = ekf.estimate(
            estimates_ekf[:, k - 1].copy(),
            variances_ekf[k - 1, :, :].copy(),
            inputs[:, k - 1].copy(),
            measurements[:, k].copy(),
        )
    end = time.time()
    print(f"EKF average computation time {(end - start) / (est_const.N - 1)}")

    # Run the PF
    print("Running PF...")
    start = time.time()
    for k in range(1, est_const.N):
        # PF
        estimates_pf[k, :, :] = pf.estimate(
            estimates_pf[k - 1, :, :].copy(),
            inputs[:, k - 1].copy(),
            measurements[:, k].copy(),
        )
    end = time.time()
    print(f"PF average computation time {(end - start) / (est_const.N - 1)}")

    # Evaluate the estimates
    evaluate(estimates_ekf, estimates_pf, simulator)

    # Plot the results
    plot(measurements, estimates_ekf, estimates_pf, simulator)
