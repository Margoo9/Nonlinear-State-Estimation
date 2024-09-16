from multiprocessing import freeze_support

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import time
from scipy.integrate import odeint
from sklearn.metrics import mean_squared_error


np.random.seed(42)


def main():
    # Definition of parameters
    # a = 1.5
    # b = 1.2
    # c = 1.3
    # d = 0.9

    a = 0.9
    b = 0.5
    c = 0.75
    d = 0.25

    # initial population of rabbits and foxes
    X0 = [15.0, 7.0]
    # size of data
    size = 100
    # size = 128
    # time lapse
    time_lv = 32
    # time = 100
    t = np.linspace(0, time_lv, size)


    # Lotka - Volterra equation
    def dX_dt(X, t, a, b, c, d):
        """Return the growth rate of fox and rabbit populations."""

        return np.array([a * X[0] - b * X[0] * X[1], -c * X[1] + d * b * X[0] * X[1]])


    # simulator function
    def competition_model(rng, a, b, c, d, size=None):
        return odeint(dX_dt, y0=X0, t=t, rtol=0.01, args=(a, b, c, d))

    # function for generating noisy data to be used as observed data.
    def add_noise(a, b, c, d):
        noise = np.random.normal(size=(size, 2))
        simulated = competition_model(None, a, b, c, d) + noise
        return simulated
    # plotting observed data.
    observed = add_noise(a, b, c, d)
    true_state = competition_model(None, a, b, c, d)
    # _, ax = plt.subplots(figsize=(12, 4))
    # ax.plot(t, observed[:, 0], "x", label="prey")
    # ax.plot(t, observed[:, 1], "x", label="predator")
    # ax.set_xlabel("time")
    # ax.set_ylabel("population")
    # ax.set_title("Observed data")
    # ax.legend()
    # plt.show()

    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
    ax1.plot(observed[:, 0], 'x', label='Pomiary')
    ax1.plot(true_state[:, 0], 'b', label='Modelowe wartości')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Liczba ofiar')

    ax2.plot(observed[:, 1], 'x', label='Pomiary')
    ax2.plot(true_state[:, 1], 'g', label='Modelowe wartości')

    plt.xlabel('Czas')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Liczba drapieżników')
    plt.savefig('obj2_smc_data.pdf')

    start_time = time.time()

    with pm.Model() as model_lv:
        alpha = pm.HalfNormal("alpha", 1.0)
        beta = pm.HalfNormal("beta", 0.5)
        gamma = pm.HalfNormal("gamma", 1.0)
        delta = pm.HalfNormal("delta", 0.5)

        sim = pm.Simulator("sim", competition_model, params=(alpha, beta, gamma, delta), epsilon=1, observed=observed)

        idata_lv = pm.sample_smc()

    sampling_time = time.time() - start_time

    az.plot_trace(idata_lv)
    plt.savefig('obj2_trace_smc.pdf')
    plt.show()
    az.plot_posterior(idata_lv)
    plt.savefig('obj2_posterior_smc.pdf')
    plt.show()

    posterior = idata_lv.posterior.stack(samples=("draw", "chain"))
    predictions = competition_model(None, posterior["alpha"].mean(), posterior["beta"].mean(), posterior["gamma"].mean(), posterior["delta"].mean())
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
    ax1.plot(observed[:, 0], 'x', label='Pomiary')
    ax1.plot(true_state[:, 0], 'b', label='Modelowe wartości')
    ax1.plot(predictions[:, 0], label='Estymata SMC')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Liczba ofiar')

    ax2.plot(observed[:, 1], 'x', label='Pomiary')
    ax2.plot(true_state[:, 1], 'g', label='Modelowe wartości')
    ax2.plot(predictions[:, 1], label='Estymata SMC')

    plt.xlabel('Czas')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Liczba drapieżników')
    plt.savefig('obj2_smc_predictions.pdf')

    plt.show()

    rms = mean_squared_error(true_state, predictions, squared=False)
    print(f'RMSE of lotka-volterra predictions using SMC: {rms}')
    print(f'Sampling time of lotka-volterra using SMC is: {sampling_time}')

    _, ax = plt.subplots(figsize=(14, 6))
    posterior = idata_lv.posterior.stack(samples=("draw", "chain"))
    ax.plot(observed[:, 0], "o", label="prey", c="C0", mec="k")
    ax.plot(observed[:, 1], "o", label="predator", c="C1", mec="k")

    ax.plot(true_state[:, 0], label="prey", c="purple")
    ax.plot(true_state[:, 1], label="predator", c="green")

    ax.plot(competition_model(None, posterior["alpha"].mean(), posterior["beta"].mean(), posterior["gamma"].mean(), posterior["delta"].mean()))
    # for i in np.random.randint(0, size, 75):
    #     sim = competition_model(None, posterior["a"][i], posterior["b"][i])
    #     ax.plot(sim[:, 0], alpha=0.1, c="C0")
    #     ax.plot(sim[:, 1], alpha=0.1, c="C1")
    ax.set_xlabel("time")
    ax.set_ylabel("population")
    ax.legend()
    plt.show()

    # az.plot_trace(idata_lv, kind="rank_vlines")
    # plt.show()
    # az.plot_posterior(idata_lv)
    # plt.show()
    # # plot results
    # _, ax = plt.subplots(figsize=(14, 6))
    # posterior = idata_lv.posterior.stack(samples=("draw", "chain"))
    # ax.plot(t, observed[:, 0], "o", label="prey", c="C0", mec="k")
    # ax.plot(t, observed[:, 1], "o", label="predator", c="C1", mec="k")
    # ax.plot(t, competition_model(None, posterior["a"].mean(), posterior["b"].mean()), linewidth=3)
    # for i in np.random.randint(0, size, 75):
    #     sim = competition_model(None, posterior["a"][i], posterior["b"][i])
    #     ax.plot(t, sim[:, 0], alpha=0.1, c="C0")
    #     ax.plot(t, sim[:, 1], alpha=0.1, c="C1")
    # ax.set_xlabel("time")
    # ax.set_ylabel("population")
    # ax.legend()
    # plt.show()


if __name__ == '__main__':
    freeze_support()
    main()
