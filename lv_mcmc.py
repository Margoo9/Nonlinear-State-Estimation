from multiprocessing import freeze_support

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import time
from scipy.integrate import odeint
from sklearn.metrics import mean_squared_error
import pandas as pd
import pytensor
import pytensor.tensor as pt

# from numba import njit
from pymc.ode import DifferentialEquation
from pytensor.compile.ops import as_op
from scipy.optimize import least_squares


# np.random.seed(42)


def slice():
    # Definition of parameters
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
    # @njit
    def rhs(X, t, theta):
        """Return the growth rate of fox and rabbit populations."""
        a = theta[0]
        b = theta[1]
        c = theta[2]
        d = theta[3]
        # return np.array([a * X[0] - b * X[0] * X[1], -c * X[1] + d * b * X[0] * X[1]])
        return [a * X[0] - b * X[0] * X[1], -c * X[1] + d * b * X[0] * X[1]]

    # simulator function
    def competition_model(rng, theta, size=None):
        return odeint(rhs, y0=X0, t=t, rtol=0.01, args=(theta,))

    # function for generating noisy data to be used as observed data.
    def add_noise(theta):
        noise = np.random.normal(size=(size, 2))
        simulated = competition_model(None, theta) + noise
        return simulated
    # plotting observed data.
    theta = [a, b, c, d, 15.0, 7.0]
    data = add_noise(theta)                 # moje observed
    # print(data)

    x_y = competition_model(None, theta)    # moje true_state

    start_time = time.time()

    # residuals = data - x_y
    # # print(residuals)
    #
    # # calculate least squares using the Scipy solver
    # results = least_squares(residuals, x0=theta)
    # print(results)

    def ode_model_resid(theta):
        return (
                data - odeint(func=rhs, y0=theta[-2:], t=t, args=(theta,))
        ).flatten()

    # res = ode_model_resid(theta)
    # print(res)
    # print(type(res))

    # calculate least squares using the Scipy solver
    results = least_squares(ode_model_resid, x0=theta)
    # print(results.x)

    # decorator with input and output types a Pytensor double float tensors
    @as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
    def pytensor_forward_model_matrix(theta):
        return odeint(func=rhs, y0=X0, t=t, rtol=0.01, args=(theta,))

    theta = results.x  # least squares solution used to inform the priors

    with pm.Model() as model:
        alpha = pm.TruncatedNormal("alpha", mu=theta[0], sigma=0.1, lower=0, initval=theta[0])
        beta = pm.TruncatedNormal("beta", mu=theta[1], sigma=0.01, lower=0, initval=theta[1])
        gamma = pm.TruncatedNormal("gamma", mu=theta[2], sigma=0.1, lower=0, initval=theta[2])
        delta = pm.TruncatedNormal("delta", mu=theta[3], sigma=0.01, lower=0, initval=theta[3])
        # xt0 = pm.TruncatedNormal("xto", mu=theta[4], sigma=1, lower=0, initval=theta[4])
        # yt0 = pm.TruncatedNormal("yto", mu=theta[5], sigma=1, lower=0, initval=theta[5])
        # sigma = pm.HalfNormal("sigma", 10)

        # Priors
        # alpha = pm.TruncatedNormal("alpha", mu=theta[0], sigma=0.1, lower=0)
        # beta = pm.TruncatedNormal("beta", mu=theta[1], sigma=0.01, lower=0)
        # gamma = pm.TruncatedNormal("gamma", mu=theta[2], sigma=0.1, lower=0)
        # delta = pm.TruncatedNormal("delta", mu=theta[3], sigma=0.01, lower=0)

        # alpha = pm.Normal("alpha")
        # beta = pm.Normal("beta")
        # gamma = pm.Normal("gamma")
        # delta = pm.Normal("delta")

        # xt0 = pm.TruncatedNormal("xto", mu=theta[4], sigma=1, lower=0)
        # yt0 = pm.TruncatedNormal("yto", mu=theta[5], sigma=1, lower=0)
        sigma = pm.HalfNormal("sigma", 10)

        # Ode solution function
        ode_solution = pytensor_forward_model_matrix(
            pm.math.stack([alpha, beta, gamma, delta])
        )

        # Likelihood
        pm.Normal("Y_obs", mu=ode_solution, sigma=sigma, observed=data)

    pm.model_to_graphviz(model=model)
    # plt.show()

    # Inference!
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
    predictions = competition_model(None, theta_posterior)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
    ax1.plot(data[:, 0], 'x', label='Pomiary')
    ax1.plot(x_y[:, 0], 'b', label='Modelowe wartości')
    ax1.plot(predictions[:, 0], label='Estymata - próbkowanie przekrojów')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Liczba ofiar')

    ax2.plot(data[:, 1], 'x', label='Pomiary')
    ax2.plot(x_y[:, 1], 'g', label='Modelowe wartości')
    ax2.plot(predictions[:, 1], label='Estymata - próbkowanie przekrojów')

    plt.xlabel('Czas')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Liczba drapieżników')
    plt.savefig('obj2_slice_predictions.pdf')

    # plt.show()

    rms = mean_squared_error(x_y, predictions, squared=False)
    print(f'RMSE of lotka-volterra predictions using Slice: {rms}')
    print(f'Sampling time of lotka-volterra using SMC is: {sampling_time}')

    # _, ax = plt.subplots(figsize=(14, 6))
    # posterior = trace_slice.posterior.stack(samples=("draw", "chain"))
    # ax.plot(data[:, 0], "o", label="prey", c="C0", mec="k")
    # ax.plot(data[:, 1], "o", label="predator", c="C1", mec="k")
    #
    # ax.plot(x_y[:, 0], label="prey", c="purple")
    # ax.plot(x_y[:, 1], label="predator", c="green")
    #
    # ax.plot(competition_model(None, theta_posterior))
    # # for i in np.random.randint(0, size, 75):
    # #     sim = competition_model(None, posterior["a"][i], posterior["b"][i])
    # #     ax.plot(sim[:, 0], alpha=0.1, c="C0")
    # #     ax.plot(sim[:, 1], alpha=0.1, c="C1")
    # ax.set_xlabel("time")
    # ax.set_ylabel("population")
    # ax.legend()
    # plt.show()


def metropolis():
    # Definition of parameters
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
    # @njit
    def rhs(X, t, theta):
        """Return the growth rate of fox and rabbit populations."""
        a = theta[0]
        b = theta[1]
        c = theta[2]
        d = theta[3]
        # return np.array([a * X[0] - b * X[0] * X[1], -c * X[1] + d * b * X[0] * X[1]])
        return [a * X[0] - b * X[0] * X[1], -c * X[1] + d * b * X[0] * X[1]]

    # simulator function
    def competition_model(rng, theta, size=None):
        return odeint(rhs, y0=X0, t=t, rtol=0.01, args=(theta,))

    # function for generating noisy data to be used as observed data.
    def add_noise(theta):
        noise = np.random.normal(size=(size, 2))
        simulated = competition_model(None, theta) + noise
        return simulated
    # plotting observed data.
    theta = [a, b, c, d, 15.0, 7.0]
    data = add_noise(theta)                 # moje observed
    # print(data)

    x_y = competition_model(None, theta)    # moje true_state

    start_time = time.time()

    # residuals = data - x_y
    # # print(residuals)
    #
    # # calculate least squares using the Scipy solver
    # results = least_squares(residuals, x0=theta)
    # print(results)

    def ode_model_resid(theta):
        return (
                data - odeint(func=rhs, y0=theta[-2:], t=t, args=(theta,))
        ).flatten()

    # res = ode_model_resid(theta)
    # print(res)
    # print(type(res))

    # calculate least squares using the Scipy solver
    results = least_squares(ode_model_resid, x0=theta)
    # print(results.x)

    # decorator with input and output types a Pytensor double float tensors
    @as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
    def pytensor_forward_model_matrix(theta):
        return odeint(func=rhs, y0=X0, t=t, rtol=0.01, args=(theta,))

    theta = results.x  # least squares solution used to inform the priors

    with pm.Model() as model:
        alpha = pm.TruncatedNormal("alpha", mu=theta[0], sigma=0.1, lower=0, initval=theta[0])
        beta = pm.TruncatedNormal("beta", mu=theta[1], sigma=0.01, lower=0, initval=theta[1])
        gamma = pm.TruncatedNormal("gamma", mu=theta[2], sigma=0.1, lower=0, initval=theta[2])
        delta = pm.TruncatedNormal("delta", mu=theta[3], sigma=0.01, lower=0, initval=theta[3])
        # xt0 = pm.TruncatedNormal("xto", mu=theta[4], sigma=1, lower=0, initval=theta[4])
        # yt0 = pm.TruncatedNormal("yto", mu=theta[5], sigma=1, lower=0, initval=theta[5])
        # sigma = pm.HalfNormal("sigma", 10)

        # Priors
        # alpha = pm.TruncatedNormal("alpha", mu=theta[0], sigma=0.1, lower=0)
        # beta = pm.TruncatedNormal("beta", mu=theta[1], sigma=0.01, lower=0)
        # gamma = pm.TruncatedNormal("gamma", mu=theta[2], sigma=0.1, lower=0)
        # delta = pm.TruncatedNormal("delta", mu=theta[3], sigma=0.01, lower=0)

        # alpha = pm.Normal("alpha")
        # beta = pm.Normal("beta")
        # gamma = pm.Normal("gamma")
        # delta = pm.Normal("delta")

        # xt0 = pm.TruncatedNormal("xto", mu=theta[4], sigma=1, lower=0)
        # yt0 = pm.TruncatedNormal("yto", mu=theta[5], sigma=1, lower=0)
        sigma = pm.HalfNormal("sigma", 10)

        # Ode solution function
        ode_solution = pytensor_forward_model_matrix(
            pm.math.stack([alpha, beta, gamma, delta])
        )

        # Likelihood
        pm.Normal("Y_obs", mu=ode_solution, sigma=sigma, observed=data)

    pm.model_to_graphviz(model=model)
    # plt.show()

    # Inference!
    with model:
        trace_slice = pm.sample(step=[pm.Metropolis()], tune=20000, draws=20000)

    sampling_time = time.time() - start_time

    az.plot_trace(trace_slice)
    plt.savefig('obj2_trace_metropolis.pdf')
    # plt.show()
    az.plot_posterior(trace_slice)
    plt.savefig('obj2_posterior_metropolis.pdf')
    # plt.show()

    posterior = trace_slice.posterior.stack(samples=("draw", "chain"))
    theta_posterior = [posterior["alpha"].mean(), posterior["beta"].mean(),
                                    posterior["gamma"].mean(), posterior["delta"].mean()]
    predictions = competition_model(None, theta_posterior)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
    ax1.plot(data[:, 0], 'x', label='Pomiary')
    ax1.plot(x_y[:, 0], 'b', label='Modelowe wartości')
    ax1.plot(predictions[:, 0], label='Estymata - algorytm Metropoolisa')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Liczba ofiar')

    ax2.plot(data[:, 1], 'x', label='Pomiary')
    ax2.plot(x_y[:, 1], 'g', label='Modelowe wartości')
    ax2.plot(predictions[:, 1], label='Estymata - algorytm Metropoolisa')

    plt.xlabel('Czas')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Liczba drapieżników')
    plt.savefig('obj2_metropolis_predictions.pdf')

    # plt.show()

    rms = mean_squared_error(x_y, predictions, squared=False)
    print(f'RMSE of lotka-volterra predictions using Metropolis: {rms}')
    print(f'Sampling time of lotka-volterra using SMC is: {sampling_time}')

    # _, ax = plt.subplots(figsize=(14, 6))
    # posterior = trace_slice.posterior.stack(samples=("draw", "chain"))
    # ax.plot(data[:, 0], "o", label="prey", c="C0", mec="k")
    # ax.plot(data[:, 1], "o", label="predator", c="C1", mec="k")
    #
    # ax.plot(x_y[:, 0], label="prey", c="purple")
    # ax.plot(x_y[:, 1], label="predator", c="green")
    #
    # ax.plot(competition_model(None, theta_posterior))
    # # for i in np.random.randint(0, size, 75):
    # #     sim = competition_model(None, posterior["a"][i], posterior["b"][i])
    # #     ax.plot(sim[:, 0], alpha=0.1, c="C0")
    # #     ax.plot(sim[:, 1], alpha=0.1, c="C1")
    # ax.set_xlabel("time")
    # ax.set_ylabel("population")
    # ax.legend()
    # plt.show()


def nuts():
    # Definition of parameters
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
    # @njit
    def rhs(X, t, theta):
        """Return the growth rate of fox and rabbit populations."""
        a = theta[0]
        b = theta[1]
        c = theta[2]
        d = theta[3]
        # return np.array([a * X[0] - b * X[0] * X[1], -c * X[1] + d * b * X[0] * X[1]])
        return [a * X[0] - b * X[0] * X[1], -c * X[1] + d * b * X[0] * X[1]]

    ode_model = DifferentialEquation(
        func=rhs, times=t, n_states=2, n_theta=4, t0=0
    )

    # simulator function
    def competition_model(rng, theta, size=None):
        return odeint(rhs, y0=X0, t=t, rtol=0.01, args=(theta,))

    # function for generating noisy data to be used as observed data.
    def add_noise(theta):
        noise = np.random.normal(size=(size, 2))
        simulated = competition_model(None, theta) + noise
        return simulated
    # plotting observed data.
    theta = [a, b, c, d, 15.0, 7.0]
    data = add_noise(theta)                 # moje observed
    # print(data)

    x_y = competition_model(None, theta)    # moje true_state

    start_time = time.time()

    # residuals = data - x_y
    # # print(residuals)
    #
    # # calculate least squares using the Scipy solver
    # results = least_squares(residuals, x0=theta)
    # print(results)

    def ode_model_resid(theta):
        return (
                data - odeint(func=rhs, y0=theta[-2:], t=t, args=(theta,))
        ).flatten()

    # res = ode_model_resid(theta)
    # print(res)
    # print(type(res))

    # calculate least squares using the Scipy solver
    results = least_squares(ode_model_resid, x0=theta)
    # print(results.x)

    # decorator with input and output types a Pytensor double float tensors
    @as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
    def pytensor_forward_model_matrix(theta):
        return odeint(func=rhs, y0=X0, t=t, rtol=0.01, args=(theta,))

    theta = results.x  # least squares solution used to inform the priors

    def callback(**kwargs):
        print(kwargs["trace"])
        print(kwargs["draw"])

    with pm.Model() as model:
        alpha = pm.TruncatedNormal("alpha", mu=theta[0], sigma=0.1, lower=0, initval=theta[0])
        beta = pm.TruncatedNormal("beta", mu=theta[1], sigma=0.01, lower=0, initval=theta[1])
        gamma = pm.TruncatedNormal("gamma", mu=theta[2], sigma=0.1, lower=0, initval=theta[2])
        delta = pm.TruncatedNormal("delta", mu=theta[3], sigma=0.01, lower=0, initval=theta[3])
        # xt0 = pm.TruncatedNormal("xto", mu=theta[4], sigma=1, lower=0, initval=theta[4])
        # yt0 = pm.TruncatedNormal("yto", mu=theta[5], sigma=1, lower=0, initval=theta[5])
        # sigma = pm.HalfNormal("sigma", 10)

        # Priors
        # alpha = pm.TruncatedNormal("alpha", mu=theta[0], sigma=0.1, lower=0)
        # beta = pm.TruncatedNormal("beta", mu=theta[1], sigma=0.01, lower=0)
        # gamma = pm.TruncatedNormal("gamma", mu=theta[2], sigma=0.1, lower=0)
        # delta = pm.TruncatedNormal("delta", mu=theta[3], sigma=0.01, lower=0)

        # alpha = pm.Normal("alpha")
        # beta = pm.Normal("beta")
        # gamma = pm.Normal("gamma")
        # delta = pm.Normal("delta")

        # xt0 = pm.TruncatedNormal("xto", mu=theta[4], sigma=1, lower=0)
        # yt0 = pm.TruncatedNormal("yto", mu=theta[5], sigma=1, lower=0)
        sigma = pm.HalfNormal("sigma", 10)

        ode_solution = ode_model(y0=X0, theta=[alpha, beta, gamma, delta])

        # Likelihood
        pm.Normal("Y_obs", mu=ode_solution, sigma=sigma, observed=data)

    pm.model_to_graphviz(model=model)
    # plt.show()

    # Inference!
    with model:
        # trace_slice = pm.sample(nuts_sampler="blackjax", tune=5000, draws=5000, progressbar=True)
        trace_slice = pm.sample(step=pm.NUTS(), tune=2000, draws=2000, progressbar=True, callback=callback)

    sampling_time = time.time() - start_time

    az.plot_trace(trace_slice)
    plt.savefig('obj2_trace_nuts.pdf')
    # plt.show()
    az.plot_posterior(trace_slice)
    plt.savefig('obj2_posterior_nuts.pdf')
    # plt.show()

    posterior = trace_slice.posterior.stack(samples=("draw", "chain"))
    theta_posterior = [posterior["alpha"].mean(), posterior["beta"].mean(),
                                    posterior["gamma"].mean(), posterior["delta"].mean()]
    predictions = competition_model(None, theta_posterior)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
    ax1.plot(data[:, 0], 'x', label='Pomiary')
    ax1.plot(x_y[:, 0], 'b', label='Modelowe wartości')
    ax1.plot(predictions[:, 0], label='Estymata - algorytm NUTS')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Liczba ofiar')

    ax2.plot(data[:, 1], 'x', label='Pomiary')
    ax2.plot(x_y[:, 1], 'g', label='Modelowe wartości')
    ax2.plot(predictions[:, 1], label='Estymata - algorytm NUTS')

    plt.xlabel('Czas')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Liczba drapieżników')
    plt.savefig('obj2_nuts_predictions.pdf')

    # plt.show()

    rms = mean_squared_error(x_y, predictions, squared=False)
    print(f'RMSE of lotka-volterra predictions using NUTS: {rms}')
    print(f'Sampling time of lotka-volterra using SMC is: {sampling_time}')

    # _, ax = plt.subplots(figsize=(14, 6))
    # posterior = trace_slice.posterior.stack(samples=("draw", "chain"))
    # ax.plot(data[:, 0], "o", label="prey", c="C0", mec="k")
    # ax.plot(data[:, 1], "o", label="predator", c="C1", mec="k")
    #
    # ax.plot(x_y[:, 0], label="prey", c="purple")
    # ax.plot(x_y[:, 1], label="predator", c="green")
    #
    # ax.plot(competition_model(None, theta_posterior))
    # # for i in np.random.randint(0, size, 75):
    # #     sim = competition_model(None, posterior["a"][i], posterior["b"][i])
    # #     ax.plot(sim[:, 0], alpha=0.1, c="C0")
    # #     ax.plot(sim[:, 1], alpha=0.1, c="C1")
    # ax.set_xlabel("time")
    # ax.set_ylabel("population")
    # ax.legend()
    # plt.show()


def slicenormal():
    # Definition of parameters
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
    # @njit
    def rhs(X, t, theta):
        """Return the growth rate of fox and rabbit populations."""
        a = theta[0]
        b = theta[1]
        c = theta[2]
        d = theta[3]
        # return np.array([a * X[0] - b * X[0] * X[1], -c * X[1] + d * b * X[0] * X[1]])
        return [a * X[0] - b * X[0] * X[1], -c * X[1] + d * b * X[0] * X[1]]

    # simulator function
    def competition_model(rng, theta, size=None):
        return odeint(rhs, y0=X0, t=t, rtol=0.01, args=(theta,))

    # function for generating noisy data to be used as observed data.
    def add_noise(theta):
        noise = np.random.normal(size=(size, 2))
        simulated = competition_model(None, theta) + noise
        return simulated
    # plotting observed data.
    theta = [a, b, c, d, 15.0, 7.0]
    data = add_noise(theta)                 # moje observed
    # print(data)

    x_y = competition_model(None, theta)    # moje true_state

    start_time = time.time()

    # residuals = data - x_y
    # # print(residuals)
    #
    # # calculate least squares using the Scipy solver
    # results = least_squares(residuals, x0=theta)
    # print(results)

    def ode_model_resid(theta):
        return (
                data - odeint(func=rhs, y0=theta[-2:], t=t, args=(theta,))
        ).flatten()

    # res = ode_model_resid(theta)
    # print(res)
    # print(type(res))

    # calculate least squares using the Scipy solver
    results = least_squares(ode_model_resid, x0=theta)
    # print(results.x)

    # decorator with input and output types a Pytensor double float tensors
    @as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
    def pytensor_forward_model_matrix(theta):
        return odeint(func=rhs, y0=X0, t=t, rtol=0.01, args=(theta,))

    theta = results.x  # least squares solution used to inform the priors

    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        beta = pm.Normal("beta")
        gamma = pm.Normal("gamma")
        delta = pm.Normal("delta")
        # xt0 = pm.TruncatedNormal("xto", mu=theta[4], sigma=1, lower=0, initval=theta[4])
        # yt0 = pm.TruncatedNormal("yto", mu=theta[5], sigma=1, lower=0, initval=theta[5])
        # sigma = pm.HalfNormal("sigma", 10)

        # Priors
        # alpha = pm.TruncatedNormal("alpha", mu=theta[0], sigma=0.1, lower=0)
        # beta = pm.TruncatedNormal("beta", mu=theta[1], sigma=0.01, lower=0)
        # gamma = pm.TruncatedNormal("gamma", mu=theta[2], sigma=0.1, lower=0)
        # delta = pm.TruncatedNormal("delta", mu=theta[3], sigma=0.01, lower=0)

        # alpha = pm.Normal("alpha")
        # beta = pm.Normal("beta")
        # gamma = pm.Normal("gamma")
        # delta = pm.Normal("delta")

        # xt0 = pm.TruncatedNormal("xto", mu=theta[4], sigma=1, lower=0)
        # yt0 = pm.TruncatedNormal("yto", mu=theta[5], sigma=1, lower=0)
        sigma = pm.HalfNormal("sigma", 10)

        # Ode solution function
        ode_solution = pytensor_forward_model_matrix(
            pm.math.stack([alpha, beta, gamma, delta])
        )

        # Likelihood
        pm.Normal("Y_obs", mu=ode_solution, sigma=sigma, observed=data)

    pm.model_to_graphviz(model=model)
    # plt.show()

    # Inference!
    with model:
        trace_slice = pm.sample(step=[pm.Slice()], tune=20000, draws=20000)

    sampling_time = time.time() - start_time

    az.plot_trace(trace_slice)
    plt.savefig('obj2_trace_slice_normal.pdf')
    # plt.show()
    az.plot_posterior(trace_slice)
    plt.savefig('obj2_posterior_slice_normal.pdf')
    # plt.show()

    posterior = trace_slice.posterior.stack(samples=("draw", "chain"))
    theta_posterior = [posterior["alpha"].mean(), posterior["beta"].mean(),
                                    posterior["gamma"].mean(), posterior["delta"].mean()]
    predictions = competition_model(None, theta_posterior)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
    ax1.plot(data[:, 0], 'x', label='Pomiary')
    ax1.plot(x_y[:, 0], 'b', label='Modelowe wartości')
    ax1.plot(predictions[:, 0], label='Estymata - próbkowanie przekrojów')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Liczba ofiar')

    ax2.plot(data[:, 1], 'x', label='Pomiary')
    ax2.plot(x_y[:, 1], 'g', label='Modelowe wartości')
    ax2.plot(predictions[:, 1], label='Estymata - próbkowanie przekrojów')

    plt.xlabel('Czas')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Liczba drapieżników')
    plt.savefig('obj2_slice_predictions_normal.pdf')

    # plt.show()

    rms = mean_squared_error(x_y, predictions, squared=False)
    print(f'RMSE of lotka-volterra predictions using Slice: {rms}')
    print(f'Sampling time of lotka-volterra using SMC is: {sampling_time}')

    # _, ax = plt.subplots(figsize=(14, 6))
    # posterior = trace_slice.posterior.stack(samples=("draw", "chain"))
    # ax.plot(data[:, 0], "o", label="prey", c="C0", mec="k")
    # ax.plot(data[:, 1], "o", label="predator", c="C1", mec="k")
    #
    # ax.plot(x_y[:, 0], label="prey", c="purple")
    # ax.plot(x_y[:, 1], label="predator", c="green")
    #
    # ax.plot(competition_model(None, theta_posterior))
    # # for i in np.random.randint(0, size, 75):
    # #     sim = competition_model(None, posterior["a"][i], posterior["b"][i])
    # #     ax.plot(sim[:, 0], alpha=0.1, c="C0")
    # #     ax.plot(sim[:, 1], alpha=0.1, c="C1")
    # ax.set_xlabel("time")
    # ax.set_ylabel("population")
    # ax.legend()
    # plt.show()


def metropolisnormal():
    # Definition of parameters
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
    # @njit
    def rhs(X, t, theta):
        """Return the growth rate of fox and rabbit populations."""
        a = theta[0]
        b = theta[1]
        c = theta[2]
        d = theta[3]
        # return np.array([a * X[0] - b * X[0] * X[1], -c * X[1] + d * b * X[0] * X[1]])
        return [a * X[0] - b * X[0] * X[1], -c * X[1] + d * b * X[0] * X[1]]

    # simulator function
    def competition_model(rng, theta, size=None):
        return odeint(rhs, y0=X0, t=t, rtol=0.01, args=(theta,))

    # function for generating noisy data to be used as observed data.
    def add_noise(theta):
        noise = np.random.normal(size=(size, 2))
        simulated = competition_model(None, theta) + noise
        return simulated
    # plotting observed data.
    theta = [a, b, c, d, 15.0, 7.0]
    data = add_noise(theta)                 # moje observed
    # print(data)

    x_y = competition_model(None, theta)    # moje true_state

    start_time = time.time()

    # residuals = data - x_y
    # # print(residuals)
    #
    # # calculate least squares using the Scipy solver
    # results = least_squares(residuals, x0=theta)
    # print(results)

    def ode_model_resid(theta):
        return (
                data - odeint(func=rhs, y0=theta[-2:], t=t, args=(theta,))
        ).flatten()

    # res = ode_model_resid(theta)
    # print(res)
    # print(type(res))

    # calculate least squares using the Scipy solver
    results = least_squares(ode_model_resid, x0=theta)
    # print(results.x)

    # decorator with input and output types a Pytensor double float tensors
    @as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
    def pytensor_forward_model_matrix(theta):
        return odeint(func=rhs, y0=X0, t=t, rtol=0.01, args=(theta,))

    theta = results.x  # least squares solution used to inform the priors

    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        beta = pm.Normal("beta")
        gamma = pm.Normal("gamma")
        delta = pm.Normal("delta")
        # xt0 = pm.TruncatedNormal("xto", mu=theta[4], sigma=1, lower=0, initval=theta[4])
        # yt0 = pm.TruncatedNormal("yto", mu=theta[5], sigma=1, lower=0, initval=theta[5])
        # sigma = pm.HalfNormal("sigma", 10)

        # Priors
        # alpha = pm.TruncatedNormal("alpha", mu=theta[0], sigma=0.1, lower=0)
        # beta = pm.TruncatedNormal("beta", mu=theta[1], sigma=0.01, lower=0)
        # gamma = pm.TruncatedNormal("gamma", mu=theta[2], sigma=0.1, lower=0)
        # delta = pm.TruncatedNormal("delta", mu=theta[3], sigma=0.01, lower=0)

        # alpha = pm.Normal("alpha")
        # beta = pm.Normal("beta")
        # gamma = pm.Normal("gamma")
        # delta = pm.Normal("delta")

        # xt0 = pm.TruncatedNormal("xto", mu=theta[4], sigma=1, lower=0)
        # yt0 = pm.TruncatedNormal("yto", mu=theta[5], sigma=1, lower=0)
        sigma = pm.HalfNormal("sigma", 10)

        # Ode solution function
        ode_solution = pytensor_forward_model_matrix(
            pm.math.stack([alpha, beta, gamma, delta])
        )

        # Likelihood
        pm.Normal("Y_obs", mu=ode_solution, sigma=sigma, observed=data)

    pm.model_to_graphviz(model=model)
    # plt.show()

    # Inference!
    with model:
        trace_slice = pm.sample(step=[pm.Metropolis()], tune=20000, draws=20000)

    sampling_time = time.time() - start_time

    az.plot_trace(trace_slice)
    plt.savefig('obj2_trace_metropolis_normal.pdf')
    # plt.show()
    az.plot_posterior(trace_slice)
    plt.savefig('obj2_posterior_metropolis_normal.pdf')
    # plt.show()

    posterior = trace_slice.posterior.stack(samples=("draw", "chain"))
    theta_posterior = [posterior["alpha"].mean(), posterior["beta"].mean(),
                                    posterior["gamma"].mean(), posterior["delta"].mean()]
    predictions = competition_model(None, theta_posterior)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
    ax1.plot(data[:, 0], 'x', label='Pomiary')
    ax1.plot(x_y[:, 0], 'b', label='Modelowe wartości')
    ax1.plot(predictions[:, 0], label='Estymata - algorytm Metropoolisa')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Liczba ofiar')

    ax2.plot(data[:, 1], 'x', label='Pomiary')
    ax2.plot(x_y[:, 1], 'g', label='Modelowe wartości')
    ax2.plot(predictions[:, 1], label='Estymata - algorytm Metropoolisa')

    plt.xlabel('Czas')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Liczba drapieżników')
    plt.savefig('obj2_metropolis_predictions_normal.pdf')

    # plt.show()

    rms = mean_squared_error(x_y, predictions, squared=False)
    print(f'RMSE of lotka-volterra predictions using Metropolis: {rms}')
    print(f'Sampling time of lotka-volterra using SMC is: {sampling_time}')

    # _, ax = plt.subplots(figsize=(14, 6))
    # posterior = trace_slice.posterior.stack(samples=("draw", "chain"))
    # ax.plot(data[:, 0], "o", label="prey", c="C0", mec="k")
    # ax.plot(data[:, 1], "o", label="predator", c="C1", mec="k")
    #
    # ax.plot(x_y[:, 0], label="prey", c="purple")
    # ax.plot(x_y[:, 1], label="predator", c="green")
    #
    # ax.plot(competition_model(None, theta_posterior))
    # # for i in np.random.randint(0, size, 75):
    # #     sim = competition_model(None, posterior["a"][i], posterior["b"][i])
    # #     ax.plot(sim[:, 0], alpha=0.1, c="C0")
    # #     ax.plot(sim[:, 1], alpha=0.1, c="C1")
    # ax.set_xlabel("time")
    # ax.set_ylabel("population")
    # ax.legend()
    # plt.show()


def nutsnormal():
    # Definition of parameters
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
    # @njit
    def rhs(X, t, theta):
        """Return the growth rate of fox and rabbit populations."""
        a = theta[0]
        b = theta[1]
        c = theta[2]
        d = theta[3]
        # return np.array([a * X[0] - b * X[0] * X[1], -c * X[1] + d * b * X[0] * X[1]])
        return [a * X[0] - b * X[0] * X[1], -c * X[1] + d * b * X[0] * X[1]]

    ode_model = DifferentialEquation(
        func=rhs, times=t, n_states=2, n_theta=4, t0=0
    )

    # simulator function
    def competition_model(rng, theta, size=None):
        return odeint(rhs, y0=X0, t=t, rtol=0.01, args=(theta,))

    # function for generating noisy data to be used as observed data.
    def add_noise(theta):
        noise = np.random.normal(size=(size, 2))
        simulated = competition_model(None, theta) + noise
        return simulated
    # plotting observed data.
    theta = [a, b, c, d, 15.0, 7.0]
    data = add_noise(theta)                 # moje observed
    # print(data)

    x_y = competition_model(None, theta)    # moje true_state

    start_time = time.time()

    # residuals = data - x_y
    # # print(residuals)
    #
    # # calculate least squares using the Scipy solver
    # results = least_squares(residuals, x0=theta)
    # print(results)

    def ode_model_resid(theta):
        return (
                data - odeint(func=rhs, y0=theta[-2:], t=t, args=(theta,))
        ).flatten()

    # res = ode_model_resid(theta)
    # print(res)
    # print(type(res))

    # calculate least squares using the Scipy solver
    results = least_squares(ode_model_resid, x0=theta)
    # print(results.x)

    # decorator with input and output types a Pytensor double float tensors
    @as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
    def pytensor_forward_model_matrix(theta):
        return odeint(func=rhs, y0=X0, t=t, rtol=0.01, args=(theta,))

    theta = results.x  # least squares solution used to inform the priors

    with pm.Model() as model:
        alpha = pm.Normal("alpha")
        beta = pm.Normal("beta")
        gamma = pm.Normal("gamma")
        delta = pm.Normal("delta")
        # xt0 = pm.TruncatedNormal("xto", mu=theta[4], sigma=1, lower=0, initval=theta[4])
        # yt0 = pm.TruncatedNormal("yto", mu=theta[5], sigma=1, lower=0, initval=theta[5])
        # sigma = pm.HalfNormal("sigma", 10)

        # Priors
        # alpha = pm.TruncatedNormal("alpha", mu=theta[0], sigma=0.1, lower=0)
        # beta = pm.TruncatedNormal("beta", mu=theta[1], sigma=0.01, lower=0)
        # gamma = pm.TruncatedNormal("gamma", mu=theta[2], sigma=0.1, lower=0)
        # delta = pm.TruncatedNormal("delta", mu=theta[3], sigma=0.01, lower=0)

        # alpha = pm.Normal("alpha")
        # beta = pm.Normal("beta")
        # gamma = pm.Normal("gamma")
        # delta = pm.Normal("delta")

        # xt0 = pm.TruncatedNormal("xto", mu=theta[4], sigma=1, lower=0)
        # yt0 = pm.TruncatedNormal("yto", mu=theta[5], sigma=1, lower=0)
        sigma = pm.HalfNormal("sigma", 10)

        ode_solution = ode_model(y0=X0, theta=[alpha, beta, gamma, delta])

        # Likelihood
        pm.Normal("Y_obs", mu=ode_solution, sigma=sigma, observed=data)

    pm.model_to_graphviz(model=model)
    # plt.show()

    # Inference!
    with model:
        trace_slice = pm.sample(step=[pm.NUTS()], tune=20000, draws=20000)

    sampling_time = time.time() - start_time

    az.plot_trace(trace_slice)
    plt.savefig('obj2_trace_nuts_normal.pdf')
    # plt.show()
    az.plot_posterior(trace_slice)
    plt.savefig('obj2_posterior_nuts_normal.pdf')
    # plt.show()

    posterior = trace_slice.posterior.stack(samples=("draw", "chain"))
    theta_posterior = [posterior["alpha"].mean(), posterior["beta"].mean(),
                                    posterior["gamma"].mean(), posterior["delta"].mean()]
    predictions = competition_model(None, theta_posterior)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
    ax1.plot(data[:, 0], 'x', label='Pomiary')
    ax1.plot(x_y[:, 0], 'b', label='Modelowe wartości')
    ax1.plot(predictions[:, 0], label='Estymata - algorytm NUTS')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Liczba ofiar')

    ax2.plot(data[:, 1], 'x', label='Pomiary')
    ax2.plot(x_y[:, 1], 'g', label='Modelowe wartości')
    ax2.plot(predictions[:, 1], label='Estymata - algorytm NUTS')

    plt.xlabel('Czas')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Liczba drapieżników')
    plt.savefig('obj2_nuts_predictions_normal.pdf')

    # plt.show()

    rms = mean_squared_error(x_y, predictions, squared=False)
    print(f'RMSE of lotka-volterra predictions using NUTS: {rms}')
    print(f'Sampling time of lotka-volterra using SMC is: {sampling_time}')

    # _, ax = plt.subplots(figsize=(14, 6))
    # posterior = trace_slice.posterior.stack(samples=("draw", "chain"))
    # ax.plot(data[:, 0], "o", label="prey", c="C0", mec="k")
    # ax.plot(data[:, 1], "o", label="predator", c="C1", mec="k")
    #
    # ax.plot(x_y[:, 0], label="prey", c="purple")
    # ax.plot(x_y[:, 1], label="predator", c="green")
    #
    # ax.plot(competition_model(None, theta_posterior))
    # # for i in np.random.randint(0, size, 75):
    # #     sim = competition_model(None, posterior["a"][i], posterior["b"][i])
    # #     ax.plot(sim[:, 0], alpha=0.1, c="C0")
    # #     ax.plot(sim[:, 1], alpha=0.1, c="C1")
    # ax.set_xlabel("time")
    # ax.set_ylabel("population")
    # ax.legend()
    # plt.show()


if __name__ == '__main__':
    freeze_support()
    # slice()
    # metropolis()
    nuts()
    # slicenormal()
    # metropolisnormal()
    # nutsnormal()
