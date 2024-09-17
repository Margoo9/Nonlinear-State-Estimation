import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from scipy.integrate import odeint
from sklearn.metrics import mean_squared_error
import time
from multiprocessing import freeze_support


# np.random.seed(42)


def main():
    start_time = time.time()

    with pm.Model() as model_lv:

        a = pm.Uniform("b", lower=-10, upper=10)
        b = pm.Uniform("c", lower=-10, upper=10)

        sim = pm.Simulator("sim", pendulum_solved, params=(a, b), epsilon=1, observed=observed_pend)
        idata_lv = pm.sample_smc()

    sampling_time = time.time() - start_time

    az.plot_trace(idata_lv)
    plt.savefig('obj1_trace_smc.pdf')
    plt.show()
    az.plot_posterior(idata_lv)
    plt.savefig('obj1_posterior_smc.pdf')
    plt.show()
    posterior = idata_lv.posterior.stack(samples=("draw", "chain"))
    predictions = pendulum_solved(None, posterior["b"].mean(), posterior["c"].mean())
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
    ax1.plot(observed_pend[:, 0], 'x', label='Pomiary')
    ax1.plot(true_state_pend[:, 0], 'b', label='Modelowe wartości')
    ax1.plot(predictions[:, 0], label='Estymata SMC')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Kąt odchylenia')

    ax2.plot(observed_pend[:, 1], 'x', label='Pomiary')
    ax2.plot(true_state_pend[:, 1], 'g', label='Modelowe wartości')
    ax2.plot(predictions[:, 1], label='Estymata SMC')

    plt.xlabel('Czas')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Prędkość kątowa')
    plt.savefig('obj1_smc_predictions.pdf')

    # plt.show()

    rms = mean_squared_error(true_state_pend, predictions, squared=False)
    print(f'RMSE of pendulum predictions using SMC: {rms}')
    print(f'Sampling time of pendulum using SMC is: {sampling_time}')

    return observed_pend, predictions, rms


if __name__ == '__main__':
    freeze_support()
    # main()

    num_runs = 10
    all_observed = []
    all_predictions = []
    all_rmse = []

    for _ in range(num_runs):
        observed, predictions, rms = main()
        all_observed.append(observed)
        all_predictions.append(predictions)
        all_rmse.append(rms)

    # Calculate averages
    average_observed = np.mean(all_observed, axis=0)
    average_predictions = np.mean(all_predictions, axis=0)
    average_rmse = np.mean(all_rmse)

    # Plot average results
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
    ax1.plot(average_observed[:, 0], 'x', label='Średnie Pomiary')
    ax1.plot(average_predictions[:, 0], label='Średnie Estymaty SMC')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Kąt odchylenia')

    ax2.plot(average_observed[:, 1], 'x', label='Średnie Pomiary')
    ax2.plot(average_predictions[:, 1], label='Średnie Estymaty SMC')

    plt.xlabel('Czas')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Prędkość kątowa')
    plt.savefig('average_smc_predictions.pdf')
    plt.show()

    print(f'Average RMSE of pendulum predictions using SMC: {average_rmse}')
