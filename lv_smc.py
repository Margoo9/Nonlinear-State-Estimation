import arviz as az
import matplotlib.pyplot as plt
from multiprocessing import freeze_support
import pymc as pm
from sklearn.metrics import mean_squared_error
import time


# np.random.seed(42)


def lv_smc(observed_lv, true_state_lv, lv_equations, lv_solved, t_lv, theta_lv, init_point_lv):
    start_time = time.time()

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
    predictions = lv_solved(None, posterior["alpha"].mean(), posterior["beta"].mean(), posterior["gamma"].mean(), posterior["delta"].mean())
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
    ax1.plot(observed_lv[:, 0], 'x', label='Pomiary')
    ax1.plot(true_state_lv[:, 0], 'b', label='Modelowe wartości')
    ax1.plot(predictions[:, 0], label='Estymata SMC')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Liczba ofiar')

    ax2.plot(observed_lv[:, 1], 'x', label='Pomiary')
    ax2.plot(true_state_lv[:, 1], 'g', label='Modelowe wartości')
    ax2.plot(predictions[:, 1], label='Estymata SMC')

    plt.xlabel('Czas')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Liczba drapieżników')
    plt.savefig('obj2_smc_predictions.pdf')

    # plt.show()

    rms = mean_squared_error(true_state_lv, predictions, squared=False)
    print(f'RMSE of lotka-volterra predictions using SMC: {rms}')
    print(f'Sampling time of lotka-volterra using SMC is: {sampling_time}')
    return predictions, sampling_time
