from params import init_point_pendulum, number_of_points_pendulum, t_pend, b, c,\
                   init_point_lv, number_of_points_lv, t_lv, alpha, beta, gamma, delta, num_runs

from objects import add_gaussian_noise, pendulum_equations, pendulum_solved, \
                    lv_equations, lv_solved

from lv_mcmc import lv_slice, lv_metropolis, lv_nuts
from lv_smc import lv_smc
from lv_ukf import lv_ukf
from pendulum_mcmc import pend_slice, pend_metropolis, pend_nuts
from pendulum_smc import pend_smc
from pendulum_ukf import pend_ukf

import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import freeze_support, Pool
import arviz as az
import pymc as pm


theta_pend = [b, c]
theta_lv = [alpha, beta, gamma, delta]

algorithms = [
    lv_slice, lv_metropolis, lv_nuts, lv_smc, lv_ukf,
    pend_slice, pend_metropolis, pend_nuts, pend_smc, pend_ukf
]


# Pendulum
true_state_pend = pendulum_solved(None, theta_pend)
observed_pend = add_gaussian_noise(true_state_pend, number_of_points_pendulum)

# Lotka-Volterry
true_state_lv = lv_solved(None, theta_lv)
observed_lv = add_gaussian_noise(true_state_lv, number_of_points_lv)


def run_algorithm(algorithm, true_state, observed_data, theta, num_runs):
    rmse_list = []
    predictions_list = []
    trace_list = []
    sampling_time_list = []

    for _ in range(num_runs):
        noisy_data = add_gaussian_noise(true_state,
                                        number_of_points_pendulum if "pend" in algorithm.__name__ else number_of_points_lv)

        if "pend" in algorithm.__name__:
            predictions, sampling_time, trace = algorithm(noisy_data, true_state, pendulum_equations, pendulum_solved, t_pend,
                                                   theta, init_point_pendulum)
        else:
            predictions, sampling_time, trace = algorithm(noisy_data, true_state, lv_equations, lv_solved, t_lv, theta,
                                                   init_point_lv)

        rmse = np.sqrt(np.mean((predictions - noisy_data) ** 2))
        rmse_list.append(rmse)

        predictions_list.append(predictions)
        trace_list.append(trace)
        sampling_time_list.append(sampling_time)

    mean_rmse = np.mean(rmse_list)
    mean_predictions = np.mean(np.array(predictions_list), axis=0)
    mean_trace = np.mean(np.array(trace_list), axis=0)
    mean_sampling_time = np.mean(
        sampling_time_list) if sampling_time_list else None

    return mean_rmse, mean_predictions, mean_sampling_time, mean_trace


if __name__ == '__main__':
    freeze_support()

    true_state_pend = pendulum_solved(None, theta_pend)
    observed_pend = add_gaussian_noise(true_state_pend, number_of_points_pendulum)
    true_state_lv = lv_solved(None, theta_lv)
    observed_lv = add_gaussian_noise(true_state_lv, number_of_points_lv)

    file_created = False

    with Pool() as pool:
        results = pool.starmap(run_algorithm, [
            (algorithm, true_state_pend, observed_pend, theta_pend, num_runs) if "pend" in algorithm.__name__ else (
            algorithm, true_state_lv, observed_lv, theta_lv, num_runs) for algorithm in algorithms])

    for i, result in enumerate(results):
        mean_rmse, mean_predictions, mean_sampling_time, mean_trace = result
        algorithm_name = algorithms[i].__name__

        if mean_trace is not None:
            if 'pend' in algorithm_name:
                az.plot_trace(mean_trace)
                plt.savefig(f'pend_trace_{algorithm_name}.pdf')
                az.plot_posterior(mean_trace)
                plt.savefig(f'pend_posterior_{algorithm_name}.pdf')
            else:
                az.plot_trace(mean_trace)
                plt.savefig(f'lv_trace_{algorithm_name}.pdf')
                az.plot_posterior(mean_trace)
                plt.savefig(f'lv_posterior_{algorithm_name}.pdf')

        if "pend" in algorithm_name:
            fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
            ax1.plot(observed_pend[:, 0], 'x', label='Pomiary')
            ax1.plot(true_state_pend[:, 0], 'b', label='Modelowe wartości')
            ax1.plot(t_pend, mean_predictions[:, 0], label=f"Estymata - algorytm {algorithm_name}")
            ax1.legend(loc='upper right')
            ax1.set_ylabel('Kąt odchylenia [rad]')

            ax2.plot(observed_pend[:, 1], 'x', label='Pomiary')
            ax2.plot(true_state_pend[:, 1], 'g', label='Modelowe wartości')
            ax2.plot(t_pend, mean_predictions[:, 0], label=f'Estymata - algorytm {algorithm_name}')

            plt.xlabel('Czas [s]')
            ax2.legend(loc='upper right')
            ax2.set_ylabel('Prędkość kątowa [rad/s]')
            plt.savefig(f'obj1_{algorithm_name}_predictions.pdf')
        else:
            fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
            ax1.plot(observed_lv[:, 0], 'x', label='Pomiary')
            ax1.plot(true_state_lv[:, 0], 'b', label='Modelowe wartości')
            ax1.plot(t_lv, mean_predictions[:, 0], label=f"Estymata - algorytm {algorithm_name}")
            ax1.legend(loc='upper right')
            ax1.set_ylabel('Liczba ofiar')

            ax2.plot(observed_lv[:, 1], 'x', label='Pomiary')
            ax2.plot(true_state_lv[:, 1], 'g', label='Modelowe wartości')
            ax2.plot(t_lv, mean_predictions[:, 0], label=f'Estymata - algorytm {algorithm_name}')

            plt.xlabel('Czas [tygodnie]')
            ax2.legend(loc='upper right')
            ax2.set_ylabel('Liczba drapieżników')
            plt.savefig(f'obj2_{algorithm_name}_predictions.pdf')

        print(f"{algorithm_name} - RMSE: {mean_rmse:.4f}")
        print(f"Mean Predictions: {mean_predictions}")
        print(f"Mean Observed: {observed_pend if 'pend' in algorithm_name else observed_lv}")
        if mean_sampling_time is not None:
            print(f"Mean Sampling Time: {mean_sampling_time:.4f}")

        if not file_created:
            with open("rmse_t.txt", "w") as file:
                file.write(str(algorithm_name) + ": RMSE = " + str(mean_rmse) + "   |   średni czas:" + "{mean_sampling_time]" + "\n")
            file_created = True
        else:
            with open("rmse_t.txt", "a") as file:
                file.write(str(algorithm_name) + ": RMSE = " + str(mean_rmse) + "   |   średni czas:" + "{mean_sampling_time]" + "\n")

