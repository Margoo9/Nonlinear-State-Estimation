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


def slice():
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

        pm.Normal("Y_obs", mu=ode_solution, sigma=sigma, observed=data)

    pm.model_to_graphviz(model=model)
    plt.show()

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
    ax1.set_ylabel('Kąt odchylenia')

    ax2.plot(observed_pend[:, 1], 'x', label='Pomiary')
    ax2.plot(true_state_pend[:, 1], 'g', label='Modelowe wartości')
    ax2.plot(predictions[:, 1], label='Estymata - próbkowanie przekrojów')

    plt.xlabel('Czas')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Prędkość kątowa')
    plt.savefig('obj1_slice_predictions.pdf')

    # plt.show()

    rms = mean_squared_error(true_state_pend, predictions, squared=False)
    print(f'RMSE of pendulum predictions using Slice: {rms}')
    print(f'Sampling time of pendulum using slice is: {sampling_time}')

    return trace_slice, observed_pend, predictions, rms, true_state_pend


def metropolis():
    start_time = time.time()

    @as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
    def pytensor_forward_model_matrix(theta_pend):
        return odeint(func=pend_equations, y0=init_point_pendulum, t=t_pend, rtol=0.01, args=(theta_pend,))

    with pm.Model() as model:
        b = pm.Normal("b")
        c = pm.Normal("c")
        sigma = pm.HalfNormal("sigma", 10)

        ode_solution = pytensor_forward_model_matrix(
            pm.math.stack([b, c])
        )

        pm.Normal("Y_obs", mu=ode_solution, sigma=sigma, observed=observed_pend)

    pm.model_to_graphviz(model=model)
    plt.show()

    with model:
        trace_slice = pm.sample(step=[pm.Metropolis()], tune=2000, draws=2000)

    sampling_time = time.time() - start_time

    az.plot_trace(trace_slice)
    plt.savefig('obj1_trace_metropolis.pdf')
    # plt.show()
    az.plot_posterior(trace_slice)
    plt.savefig('obj1_posterior_metropolis.pdf')
    # plt.show()

    posterior = trace_slice.posterior.stack(samples=("draw", "chain"))
    theta_posterior = [posterior["b"].mean(), posterior["c"].mean()]
    predictions = pendulum_solved(None, theta_posterior)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
    ax1.plot(observed_pend[:, 0], 'x', label='Pomiary')
    ax1.plot(true_state_pend[:, 0], 'b', label='Modelowe wartości')
    ax1.plot(predictions[:, 0], label='Estymata - algorytm Metropolisa')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Kąt odchylenia')

    ax2.plot(served_pend[:, 1], 'x', label='Pomiary')
    ax2.plot(true_state_pend[:, 1], 'g', label='Modelowe wartości')
    ax2.plot(predictions[:, 1], label='Estymata - algorytm Metropolisa')

    plt.xlabel('Czas')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Prędkość kątowa')
    plt.savefig('obj1_metropolis_predictions.pdf')

    # plt.show()

    rms = mean_squared_error(true_state_pend, predictions, squared=False)
    print(f'RMSE of pendulum predictions using Metropolis: {rms}')
    print(f'Sampling time of pendulum using Metropolis is: {sampling_time}')

    return trace_slice, observed_pend, predictions, rms, true_state_pend


def nuts():
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

        ode_solution = ode_model(y0=X0, theta=[b, c])

        pm.Normal("Y_obs", mu=ode_solution, sigma=sigma, observed=observed_pend)

    pm.model_to_graphviz(model=model)
    plt.show()

    with model:
        step = pm.NUTS()
        trace_slice = pm.sample(step=step, tune=2000, draws=2000)

    sampling_time = time.time() - start_time

    az.plot_trace(trace_slice)
    plt.savefig('obj1_trace_nuts.pdf')
    # plt.show()
    az.plot_posterior(trace_slice)
    plt.savefig('obj1_posterior_nuts.pdf')
    # plt.show()

    posterior = trace_slice.posterior.stack(samples=("draw", "chain"))
    theta_posterior = [posterior["b"].mean(), posterior["c"].mean()]
    predictions = pendulum_solved(None, theta_posterior)
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
    ax1.plot(observed_pend[:, 0], 'x', label='Pomiary')
    ax1.plot(true_state_pend[:, 0], 'b', label='Modelowe wartości')
    ax1.plot(predictions[:, 0], label='Estymata - algorytm NUTS')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Kąt odchylenia')

    ax2.plot(observed_pend[:, 1], 'x', label='Pomiary')
    ax2.plot(true_state_pend[:, 1], 'g', label='Modelowe wartości')
    ax2.plot(predictions[:, 1], label='Estymata - algorytm NUTS')

    plt.xlabel('Czas')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Prędkość kątowa')
    plt.savefig('obj1_nuts_predictions.pdf')

    # plt.show()

    rms = mean_squared_error(true_state_pend, predictions, squared=False)
    print(f'RMSE of pendulum predictions using NUTS: {rms}')
    print(f'Sampling time of pendulum using NUTS is: {sampling_time}')

    return trace_slice, observed_pend, predictions, rms, true_state_pend


def run_simulation_slice(_):
    return slice()


def run_simulation_metropolis(_):
    return metropolis()


def run_simulation_nuts(_):
    return nuts()


if __name__ == '__main__':
    freeze_support()
    num_runs = 10

    with Pool() as pool:
        results_slice = pool.map(run_simulation_slice, range(num_runs))

    all_traces_slice, all_data_slice, all_predicitons_slice, all_rmse_slice, x_y_slice = zip(*results_slice)

    combined_trace_slice = az.concat(all_traces_slice)

    average_data_slice = np.mean(all_data_slice, axis=0)
    average_predictions_slice = np.mean(all_predicitons_slice, axis=0)
    average_rmse_slice = np.mean(all_rmse_slice)

    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
    ax1.plot(average_data_slice[:, 0], 'x', label='Pomiary')
    ax2.plot(x_y_slice[:, 1], 'g', label='Modelowe wartości')
    ax1.plot(average_predictions_slice[:, 0], label='Estymata - próbkowanie przekrojów')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Kąt odchylenia')

    ax2.plot(average_data_slice[:, 1], 'x', label='Pomiary')
    ax2.plot(x_y_slice[:, 1], 'g', label='Modelowe wartości')
    ax2.plot(average_predictions_slice[:, 1], label='Estymata - próbkowanie przekrojów')

    plt.xlabel('Czas')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Prędkość kątowa')
    plt.savefig('average_slice_predictions.pdf')

    print(f'Average RMSE of pendulum predictions using Slice: {average_rmse_slice}')

    with Pool() as pool:
        results_metropolis = pool.map(run_simulation_metropolis, range(num_runs))

    all_traces_metropolis, all_data_metropolis, all_predicitons_metropolis, all_rmse_metropolis, x_y_metropolis = zip(*results_metropolis)

    combined_trace_metropolis = az.concat(all_traces_metropolis)

    average_data_metropolis = np.mean(all_data_metropolis, axis=0)
    average_predictions_metropolis = np.mean(all_predicitons_metropolis, axis=0)
    average_rmse_metropolis = np.mean(all_rmse_metropolis)

    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
    ax1.plot(average_data_metropolis[:, 0], 'x', label='Pomiary')
    ax2.plot(x_y_metropolis[:, 1], 'g', label='Modelowe wartości')
    ax1.plot(average_predictions_metropolis[:, 0], label='Estymata - algorytm Metropolisa')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Kąt odchylenia')

    ax2.plot(average_data_metropolis[:, 1], 'x', label='Pomiary')
    ax2.plot(x_y_metropolis[:, 1], 'g', label='Modelowe wartości')
    ax2.plot(average_predictions_metropolis[:, 1], label='Estymata - algorytm Metropolisa')

    plt.xlabel('Czas')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Prędkość kątowa')
    plt.savefig('average_metropolis_predictions.pdf')

    print(f'Average RMSE of pendulum predictions using metropolis: {average_rmse_metropolis}')


    with Pool() as pool:
        results_nuts = pool.map(run_simulation_nuts, range(num_runs))

    all_traces_nuts, all_data_nuts, all_predicitons_nuts, all_rmse_nuts, x_y_nuts = zip(*results_nuts)

    combined_trace_nuts = az.concat(all_traces_nuts)

    average_data_nuts = np.mean(all_data_nuts, axis=0)
    average_predictions_nuts = np.mean(all_predicitons_nuts, axis=0)
    average_rmse_nuts = np.mean(all_rmse_nuts)

    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))
    ax1.plot(average_data_nuts[:, 0], 'x', label='Pomiary')
    ax2.plot(x_y_nuts[:, 1], 'g', label='Modelowe wartości')
    ax1.plot(average_predictions_nuts[:, 0], label='Estymata - algorytm NUTS')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('Kąt odchylenia')

    ax2.plot(average_data_nuts[:, 1], 'x', label='Pomiary')
    ax2.plot(x_y_nuts[:, 1], 'g', label='Modelowe wartości')
    ax2.plot(average_predictions_nuts[:, 1], label='Estymata - algorytm NUTS')

    plt.xlabel('Czas')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Prędkość kątowa')
    plt.savefig('average_nuts_predictions.pdf')

    print(f'Average RMSE of pendulum predictions using nuts: {average_rmse_nuts}')
    plt.show()
    
