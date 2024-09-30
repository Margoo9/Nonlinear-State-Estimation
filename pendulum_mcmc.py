import arviz as az
import matplotlib.pyplot as plt
from multiprocessing import freeze_support, Pool
import numpy as np
import pymc as pm
from pymc.ode import DifferentialEquation
from pytensor.compile.ops import as_op
import pytensor.tensor as pt
from scipy.integrate import odeint
from sklearn.metrics import mean_squared_error
import time

# np.random.seed(42)


def pend_slice(observed_pend, true_state_pend, pendulum_equations, pendulum_solved, t_pend, theta_pend, init_point_pendulum):
    start_time = time.time()

    @as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
    def pytensor_forward_model_matrix(theta_pend):
        return odeint(func=pendulum_equations, y0=init_point_pendulum, t=t_pend, rtol=0.01, args=(theta_pend,))

    with pm.Model() as model:
        b = pm.Normal("b")
        c = pm.Normal("c")
        sigma = pm.HalfNormal("sigma", 10)

        ode_solution = pytensor_forward_model_matrix(
            pm.math.stack([b, c])
        )

        pm.Normal("Y_obs", mu=ode_solution, sigma=sigma, observed=observed_pend)

    pm.model_to_graphviz(model=model)
    # plt.show()

    with model:
        trace_slice = pm.sample(step=[pm.Slice()], tune=2000, draws=2000)

    sampling_time = time.time() - start_time

    az.plot_trace(trace_slice)
    plt.savefig('obj1_trace_slice.pdf')
    # plt.show()
    az.plot_posterior(trace_slice)
    plt.savefig('obj1_posterior_slice.pdf')
    # plt.show()

    posterior = trace_slice.posterior.stack(samples=("draw", "chain"))
    theta_posterior = [posterior["b"].mean(), posterior["c"].mean()]
    predictions = pendulum_solved(None, theta_posterior)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
    ax1.plot(observed_pend[:, 0], 'x', label='Pomiary')
    ax1.plot(true_state_pend[:, 0], 'b', label='Modelowe wartości')
    ax1.plot(predictions[:, 0], label='Estymata - próbkowanie przekrojów')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Kąt odchylenia [rad]')

    ax2.plot(observed_pend[:, 1], 'x', label='Pomiary')
    ax2.plot(true_state_pend[:, 1], 'g', label='Modelowe wartości')
    ax2.plot(predictions[:, 1], label='Estymata - próbkowanie przekrojów')

    plt.xlabel('Czas [s]')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Prędkość kątowa [rad/s]')
    plt.savefig('obj1_slice_predictions.pdf')

    # plt.show()

    rms = mean_squared_error(true_state_pend, predictions, squared=False)
    print(f'RMSE of pendulum predictions using Slice: {rms}')
    print(f'Sampling time of pendulum using slice is: {sampling_time}')

    # return trace_slice, observed_pend, predictions, rms, true_state_pend
    return predictions, sampling_time, trace_slice


def pend_metropolis(observed_pend, true_state_pend, pendulum_equations, pendulum_solved, t_pend, theta_pend, init_point_pendulum):
    start_time = time.time()

    @as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
    def pytensor_forward_model_matrix(theta_pend):
        return odeint(func=pendulum_equations, y0=init_point_pendulum, t=t_pend, rtol=0.01, args=(theta_pend,))

    with pm.Model() as model:
        b = pm.Normal("b")
        c = pm.Normal("c")
        sigma = pm.HalfNormal("sigma", 10)

        ode_solution = pytensor_forward_model_matrix(
            pm.math.stack([b, c])
        )

        pm.Normal("Y_obs", mu=ode_solution, sigma=sigma, observed=observed_pend)

    pm.model_to_graphviz(model=model)
    # plt.show()

    with model:
        trace_metropolis = pm.sample(step=[pm.Metropolis()], tune=2000, draws=2000)

    sampling_time = time.time() - start_time

    az.plot_trace(trace_metropolis)
    plt.savefig('obj1_trace_metropolis.pdf')
    # plt.show()
    az.plot_posterior(trace_metropolis)
    plt.savefig('obj1_posterior_metropolis.pdf')
    # plt.show()

    posterior = trace_metropolis.posterior.stack(samples=("draw", "chain"))
    theta_posterior = [posterior["b"].mean(), posterior["c"].mean()]
    predictions = pendulum_solved(None, theta_posterior)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
    ax1.plot(observed_pend[:, 0], 'x', label='Pomiary')
    ax1.plot(true_state_pend[:, 0], 'b', label='Modelowe wartości')
    ax1.plot(predictions[:, 0], label='Estymata - algorytm Metropolisa')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Kąt odchylenia [rad]')

    ax2.plot(observed_pend[:, 1], 'x', label='Pomiary')
    ax2.plot(true_state_pend[:, 1], 'g', label='Modelowe wartości')
    ax2.plot(predictions[:, 1], label='Estymata - algorytm Metropolisa')

    plt.xlabel('Czas [s]')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Prędkość kątowa [rad/s]')
    plt.savefig('obj1_metropolis_predictions.pdf')

    # plt.show()

    rms = mean_squared_error(true_state_pend, predictions, squared=False)
    print(f'RMSE of pendulum predictions using Metropolis: {rms}')
    print(f'Sampling time of pendulum using Metropolis is: {sampling_time}')

    # return trace_slice, observed_pend, predictions, rms, true_state_pend
    return predictions, sampling_time, trace_metropolis


def pend_nuts(observed_pend, true_state_pend, pendulum_equations, pendulum_solved, t_pend, theta_pend, init_point_pendulum):
    ode_model = DifferentialEquation(
        func=pendulum_equations, times=t_pend, n_states=2, n_theta=2, t0=0
    )

    start_time = time.time()

    @as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
    def pytensor_forward_model_matrix(theta_pend):
        return odeint(func=pendulum_equations, y0=init_point_pendulum, t=t_pend, rtol=0.01, args=(theta_pend,))

    with pm.Model() as model:

        b = pm.Normal("b")
        c = pm.Normal("c")
        sigma = pm.HalfNormal("sigma", 10)

        ode_solution = ode_model(y0=init_point_pendulum, theta=[b, c])

        pm.Normal("Y_obs", mu=ode_solution, sigma=sigma, observed=observed_pend)

    pm.model_to_graphviz(model=model)
    # plt.show()

    with model:
        step = pm.NUTS()
        trace_nuts = pm.sample(step=step, tune=2000, draws=2000)

    sampling_time = time.time() - start_time

    az.plot_trace(trace_nuts)
    plt.savefig('obj1_trace_nuts.pdf')
    # plt.show()
    az.plot_posterior(trace_nuts)
    plt.savefig('obj1_posterior_nuts.pdf')
    # plt.show()

    posterior = trace_nuts.posterior.stack(samples=("draw", "chain"))
    theta_posterior = [posterior["b"].mean(), posterior["c"].mean()]
    predictions = pendulum_solved(None, theta_posterior)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
    ax1.plot(observed_pend[:, 0], 'x', label='Pomiary')
    ax1.plot(true_state_pend[:, 0], 'b', label='Modelowe wartości')
    ax1.plot(predictions[:, 0], label='Estymata - algorytm NUTS')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Kąt odchylenia [rad]')

    ax2.plot(observed_pend[:, 1], 'x', label='Pomiary')
    ax2.plot(true_state_pend[:, 1], 'g', label='Modelowe wartości')
    ax2.plot(predictions[:, 1], label='Estymata - algorytm NUTS')

    plt.xlabel('Czas [s]')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Prędkość kątowa [rad/s]')
    plt.savefig('obj1_nuts_predictions.pdf')

    # plt.show()

    rms = mean_squared_error(true_state_pend, predictions, squared=False)
    print(f'RMSE of pendulum predictions using NUTS: {rms}')
    print(f'Sampling time of pendulum using NUTS is: {sampling_time}')

    # return trace_slice, observed_pend, predictions, rms, true_state_pend
    return predictions, sampling_time, trace_nuts
