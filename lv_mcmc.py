import arviz as az
import matplotlib.pyplot as plt
from multiprocessing import freeze_support
import pymc as pm
from pymc.ode import DifferentialEquation
from pytensor.compile.ops import as_op
import pytensor.tensor as pt
from scipy.integrate import odeint
from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error
import time


# np.random.seed(42)


def lv_slice(observed_lv, true_state_lv, lv_equations, lv_solved, t_lv, theta_lv, init_point_lv):
    start_time = time.time()

    def ode_model_resid(theta_lv):
        return (
                observed_lv - odeint(func=lv_equations, y0=theta_lv[-2:], t=t_lv, args=(theta_lv,))
        ).flatten()

    results_lv = least_squares(ode_model_resid, x0=theta_lv)

    @as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
    def pytensor_forward_model_matrix(theta_lv):
        return odeint(func=lv_equations, y0=init_point_lv, t=t_lv, rtol=0.01, args=(theta_lv,))

    theta = results_lv.x

    with pm.Model() as model:
        alpha = pm.TruncatedNormal("alpha", mu=theta[0], sigma=0.1, lower=0, initval=theta[0])
        beta = pm.TruncatedNormal("beta", mu=theta[1], sigma=0.01, lower=0, initval=theta[1])
        gamma = pm.TruncatedNormal("gamma", mu=theta[2], sigma=0.1, lower=0, initval=theta[2])
        delta = pm.TruncatedNormal("delta", mu=theta[3], sigma=0.01, lower=0, initval=theta[3])

        sigma = pm.HalfNormal("sigma", 10)

        ode_solution = pytensor_forward_model_matrix(
            pm.math.stack([alpha, beta, gamma, delta])
        )

        pm.Normal("Y_obs", mu=ode_solution, sigma=sigma, observed=observed_lv)

    pm.model_to_graphviz(model=model)
    # plt.show()

    with model:
        trace_slice = pm.sample(step=[pm.Slice()], tune=20000, draws=20000)

    sampling_time = time.time() - start_time

    az.plot_trace(trace_slice)
    plt.savefig('obj2_trace_slice.pdf')
    # plt.show()
    az.plot_posterior(trace_slice)
    plt.savefig('obj2_posterior_slice.pdf')
    # plt.show()

    posterior = trace_slice.posterior.stack(samples=("draw", "chain"))
    theta_posterior = [posterior["alpha"].mean(), posterior["beta"].mean(),
                                    posterior["gamma"].mean(), posterior["delta"].mean()]
    predictions = lv_solved(None, theta_posterior)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
    ax1.plot(observed_lv[:, 0], 'x', label='Pomiary')
    ax1.plot(true_state_lv[:, 0], 'b', label='Modelowe wartości')
    ax1.plot(predictions[:, 0], label='Estymata - próbkowanie przekrojów')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Liczba ofiar')

    ax2.plot(observed_lv[:, 1], 'x', label='Pomiary')
    ax2.plot(true_state_lv[:, 1], 'g', label='Modelowe wartości')
    ax2.plot(predictions[:, 1], label='Estymata - próbkowanie przekrojów')

    plt.xlabel('Czas [tygodnie]')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Liczba drapieżników')
    plt.savefig('obj2_slice_predictions.pdf')

    # plt.show()

    rms = mean_squared_error(true_state_lv, predictions, squared=False)
    print(f'RMSE of lotka-volterra predictions using Slice: {rms}')
    print(f'Sampling time of lotka-volterra using Slice is: {sampling_time}')

    return predictions, sampling_time, trace_slice


def lv_metropolis(observed_lv, true_state_lv, lv_equations, lv_solved, t_lv, theta_lv, init_point_lv):
    start_time = time.time()

    def ode_model_resid(theta_lv):
        return (
                observed_lv - odeint(func=lv_equations, y0=theta_lv[-2:], t=t_lv, args=(theta_lv,))
        ).flatten()

    results_lv = least_squares(ode_model_resid, x0=theta_lv)

    @as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
    def pytensor_forward_model_matrix(theta_lv):
        return odeint(func=lv_equations, y0=init_point_lv, t=t_lv, rtol=0.01, args=(theta_lv,))

    theta = results_lv.x

    with pm.Model() as model:
        alpha = pm.TruncatedNormal("alpha", mu=theta[0], sigma=0.1, lower=0, initval=theta[0])
        beta = pm.TruncatedNormal("beta", mu=theta[1], sigma=0.01, lower=0, initval=theta[1])
        gamma = pm.TruncatedNormal("gamma", mu=theta[2], sigma=0.1, lower=0, initval=theta[2])
        delta = pm.TruncatedNormal("delta", mu=theta[3], sigma=0.01, lower=0, initval=theta[3])

        sigma = pm.HalfNormal("sigma", 10)

        ode_solution = pytensor_forward_model_matrix(
            pm.math.stack([alpha, beta, gamma, delta])
        )

        pm.Normal("Y_obs", mu=ode_solution, sigma=sigma, observed=observed_lv)

    pm.model_to_graphviz(model=model)
    # plt.show()

    with model:
        trace_metropolis = pm.sample(step=[pm.Metropolis()], tune=20000, draws=20000)

    sampling_time = time.time() - start_time

    az.plot_trace(trace_metropolis)
    plt.savefig('obj2_trace_metropolis.pdf')
    # plt.show()
    az.plot_posterior(trace_metropolis)
    plt.savefig('obj2_posterior_metropolis.pdf')
    # plt.show()

    posterior = trace_metropolis.posterior.stack(samples=("draw", "chain"))
    theta_posterior = [posterior["alpha"].mean(), posterior["beta"].mean(),
                                    posterior["gamma"].mean(), posterior["delta"].mean()]
    predictions = lv_solved(None, theta_posterior)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
    ax1.plot(observed_lv[:, 0], 'x', label='Pomiary')
    ax1.plot(true_state_lv[:, 0], 'b', label='Modelowe wartości')
    ax1.plot(predictions[:, 0], label='Estymata - algorytm Metropoolisa')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Liczba ofiar')

    ax2.plot(observed_lv[:, 1], 'x', label='Pomiary')
    ax2.plot(true_state_lv[:, 1], 'g', label='Modelowe wartości')
    ax2.plot(predictions[:, 1], label='Estymata - algorytm Metropoolisa')

    plt.xlabel('Czas [tygodnie]')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Liczba drapieżników')
    plt.savefig('obj2_metropolis_predictions.pdf')

    # plt.show()

    rms = mean_squared_error(true_state_lv, predictions, squared=False)
    print(f'RMSE of lotka-volterra predictions using Metropolis: {rms}')
    print(f'Sampling time of lotka-volterra using Metropolis is: {sampling_time}')
    return predictions, sampling_time, trace_metropolis


def lv_nuts(observed_lv, true_state_lv, lv_equations, lv_solved, t_lv, theta_lv, init_point_lv):
    ode_model = DifferentialEquation(
        func=lv_equations, times=t_lv, n_states=2, n_theta=4, t0=0
    )

    start_time = time.time()

    def ode_model_resid(theta_lv):
        return (
                observed_lv - odeint(func=lv_equations, y0=theta[-2:], t=t_lv, args=(theta_lv,))
        ).flatten()

    results = least_squares(ode_model_resid, x0=theta_lv)

    @as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
    def pytensor_forward_model_matrix(theta_lv):
        return odeint(func=lv_equations, y0=init_point_lv, t=t_lv, rtol=0.01, args=(theta_lv,))

    theta = results.x

    def callback(**kwargs):
        print(kwargs["trace"])
        print(kwargs["draw"])

    with pm.Model() as model:
        alpha = pm.TruncatedNormal("alpha", mu=theta[0], sigma=0.1, lower=0, initval=theta[0])
        beta = pm.TruncatedNormal("beta", mu=theta[1], sigma=0.01, lower=0, initval=theta[1])
        gamma = pm.TruncatedNormal("gamma", mu=theta[2], sigma=0.1, lower=0, initval=theta[2])
        delta = pm.TruncatedNormal("delta", mu=theta[3], sigma=0.01, lower=0, initval=theta[3])

        sigma = pm.HalfNormal("sigma", 10)

        ode_solution = ode_model(y0=init_point_lv, theta=[alpha, beta, gamma, delta])

        # Likelihood
        pm.Normal("Y_obs", mu=ode_solution, sigma=sigma, observed=observed_lv)

    pm.model_to_graphviz(model=model)
    # plt.show()

    with model:
        # trace_slice = pm.sample(nuts_sampler="blackjax", tune=5000, draws=5000, progressbar=True)
        trace_nuts = pm.sample(step=pm.NUTS(), tune=2000, draws=2000, progressbar=True, callback=callback)

    sampling_time = time.time() - start_time

    az.plot_trace(trace_nuts)
    plt.savefig('obj2_trace_nuts.pdf')
    # plt.show()
    az.plot_posterior(trace_nuts)
    plt.savefig('obj2_posterior_nuts.pdf')
    # plt.show()

    posterior = trace_nuts.posterior.stack(samples=("draw", "chain"))
    theta_posterior = [posterior["alpha"].mean(), posterior["beta"].mean(),
                                    posterior["gamma"].mean(), posterior["delta"].mean()]
    predictions = lv_solved(None, theta_posterior)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
    ax1.plot(observed_lv[:, 0], 'x', label='Pomiary')
    ax1.plot(true_state_lv[:, 0], 'b', label='Modelowe wartości')
    ax1.plot(predictions[:, 0], label='Estymata - algorytm NUTS')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Liczba ofiar')

    ax2.plot(observed_lv[:, 1], 'x', label='Pomiary')
    ax2.plot(true_state_lv[:, 1], 'g', label='Modelowe wartości')
    ax2.plot(predictions[:, 1], label='Estymata - algorytm NUTS')

    plt.xlabel('Czas [tygodnie]')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Liczba drapieżników')
    plt.savefig('obj2_nuts_predictions.pdf')

    # plt.show()

    rms = mean_squared_error(true_state_lv, predictions, squared=False)
    print(f'RMSE of lotka-volterra predictions using NUTS: {rms}')
    print(f'Sampling time of lotka-volterra using NUTS is: {sampling_time}')
    return predictions, sampling_time, trace_nuts
