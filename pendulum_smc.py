import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from sklearn.metrics import mean_squared_error
import time
from multiprocessing import freeze_support


# np.random.seed(42)


def pend_smc(observed_pend, true_state_pend, pendulum_equations, pendulum_solved, t_pend, theta_pend, init_point_pendulum):
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

    # return observed_pend, predictions, rms
    return predictions, sampling_time
