import arviz as az
import matplotlib.pyplot as plt
from multiprocessing import freeze_support
import pymc as pm
from sklearn.metrics import mean_squared_error
import time
from scipy.integrate import odeint
import numpy as np


# np.random.seed(42)

def lv_smc(observed_lv, true_state_lv, lv_equations_old, lv_solved_old, t_lv, theta_lv, init_point_lv):
    start_time = time.time()

    def lv_equations(X, t, theta_lv):
        alpha_lv, beta_lv, gamma_lv, delta_lv = theta_lv
        return np.array(
            [alpha_lv * X[0] - beta_lv * X[0] * X[1], -gamma_lv * X[1] + delta_lv * beta_lv * X[0] * X[1]]).flatten()

    def lv_solved(rng, alpha, beta, gamma, delta, size=None, *args):
        theta_lv = (alpha, beta, gamma, delta)
        y0 = np.array(init_point_lv).flatten()
        sol = odeint(lv_equations, y0=y0, t=t_lv, args=(theta_lv,))

        if sol.ndim == 1:
            sol = sol.reshape(-1, len(y0))

        return sol

    with pm.Model() as model_lv:
        alpha = pm.HalfNormal("alpha", 1.0)
        beta = pm.HalfNormal("beta", 0.5)
        gamma = pm.HalfNormal("gamma", 1.0)
        delta = pm.HalfNormal("delta", 0.5)

        sim = pm.Simulator("sim", lv_solved, params=(alpha, beta, gamma, delta), epsilon=1, observed=observed_lv)

        idata_lv = pm.sample_smc()

    sampling_time = time.time() - start_time

    az.plot_trace(idata_lv)
    plt.savefig('obj2_trace_smc.pdf')
    # plt.show()
    az.plot_posterior(idata_lv)
    plt.savefig('obj2_posterior_smc.pdf')
    # plt.show()

    posterior = idata_lv.posterior.stack(samples=("draw", "chain"))
    predictions = lv_solved(None, posterior["alpha"].mean(), posterior["beta"].mean(), posterior["gamma"].mean(),
                            posterior["delta"].mean())
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
    ax1.plot(observed_lv[:, 0], 'x', label='Pomiary')
    ax1.plot(true_state_lv[:, 0], 'b', label='Modelowe wartości')
    ax1.plot(predictions[:, 0], label='Estymata SMC')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Liczba ofiar')

    ax2.plot(observed_lv[:, 1], 'x', label='Pomiary')
    ax2.plot(true_state_lv[:, 1], 'g', label='Modelowe wartości')
    ax2.plot(predictions[:, 1], label='Estymata SMC')

    plt.xlabel('Czas [tygodnie]')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Liczba drapieżników')
    plt.savefig('obj2_smc_predictions.pdf')

    # plt.show()

    rms = mean_squared_error(true_state_lv, predictions, squared=False)
    print(f'RMSE of lotka-volterra predictions using SMC: {rms}')
    print(f'Sampling time of lotka-volterra using SMC is: {sampling_time}')
    return predictions, sampling_time, idata_lv
