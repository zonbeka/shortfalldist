import random
import matplotlib.pyplot as plt
import analytical_model as am
import json
import datetime as d_time
from math import floor
from numpy import hstack, vstack, zeros, sqrt

RND = None


class CumStats:
    def __init__(self):
        self.mean = None
        self.sd = None
        self.sum = float(0)
        self.sq_sum = float(0)
        self.counter = 0

    def add(self, value: float):
        self.sum += value
        self.sq_sum += value ** 2
        self.counter += 1
        self.mean = self.sum / self.counter
        if self.counter > 1:
            self.sd = sqrt((self.sq_sum - self.counter * self.mean ** 2) / (self.counter - 1))

    def average(self):
        return self.mean

    def std_dev(self):
        return self.sd


class Order:
    def __init__(self, d, b):
        self.demand = d
        self.batch_time = b
        self.wait_time = 0
        self.process_time = 0
        self.is_done = False

    def increment_wait(self):
        if self.process_time > 0 or self.is_done:
            raise RuntimeError("Order is done.")
        self.wait_time += 1

    def increment_process_time(self):
        if self.is_done:
            raise RuntimeError("Order is done.")
        self.process_time += 1

    def lead_time(self) -> int:
        return self.wait_time + self.process_time

    def __repr__(self):
        return "d=%s t=%s W=%s LT=%s" % (self.demand, self.batch_time, self.wait_time, self.lead_time())

    def done(self):
        self.is_done = True


class Server:
    def __init__(self):
        self._current_order = None

    def assign_order(self, order: Order):
        if self._current_order is not None:
            raise RuntimeError("Server has an assigned order.")
        self._current_order = order

    def free(self):
        if self._current_order is None:
            raise RuntimeError("Server has no assigned order.")
        self._current_order = None

    def is_free(self):
        return self._current_order is None

    def is_busy(self):
        return self._current_order is not None

    def current_order(self) -> Order:
        return self._current_order


def random_from_phase(initial, t_matrix, absorption):
    last_state = 3
    state = 0
    counter = 0
    m = to_single_matrix(initial, t_matrix, absorption)
    while state < last_state:
        state = choose(m[state])
        counter += 1
    return counter - 1


def choose(pmf) -> int:
    assert abs(sum(pmf) - 1) <= 1e-10 and len(pmf) > 1
    r = RND.random()
    i = 0
    cdf = pmf[0]
    while cdf < r:
        i += 1
        cdf += pmf[i]
    return i


def to_single_matrix(initial, t_matrix, absorption):
    size = 4
    m = vstack((
        hstack((zeros(1), initial, zeros(1))),
        hstack((zeros((size - 2, 1)), t_matrix, absorption)),
        zeros((1, size))
    ))
    return m


def get_batch_time(dem, service_initial, service_transition, service_absorption):
    cumulative_time = 0
    for i in range(dem):
        cumulative_time += random_from_phase(service_initial, service_transition, service_absorption)
    return cumulative_time


def simulate(data_set, n_sample, seed=None):
    global RND
    RND = random.Random()
    if seed is not None:
        RND.seed(seed)
    # Read data
    demand_initial = data_set['Demand']['Initial']
    demand_transition = data_set['Demand']['Transition']
    demand_absorption = data_set['Demand']['Absorption']
    service_initial = data_set['Service']['Initial']
    service_transition = data_set['Service']['Transition']
    service_absorption = data_set['Service']['Absorption']

    server1 = Server()
    server2 = Server()
    orders = {}
    queue = []
    stats = []
    cum_time_slot = 0
    n_done = 0
    stats_w = CumStats()
    stats_lt = CumStats()
    stats_ltp = CumStats()
    stats_st = CumStats()
    stats_n_in_system = CumStats()
    stats_demand = CumStats()

    for t in range(1, n_sample):
        demand = random_from_phase(demand_initial, demand_transition, demand_absorption)
        batch_time = get_batch_time(demand, service_initial, service_transition, service_absorption)
        orders[t] = Order(demand, batch_time)
        queue.append(orders[t])
        for _ in [1, 2, 3]:
            if len(queue) > 0 and server1.is_free():
                server1.assign_order(queue[0])
                queue.remove(queue[0])
            if len(queue) > 0 and server2.is_free():
                server2.assign_order(queue[0])
                queue.remove(queue[0])
            for i in queue:
                i.increment_wait()
            n_q = sum([o.demand for o in queue])
            n_s1 = 0
            if server1.is_busy():
                order = server1.current_order()
                order.increment_process_time()
                n_s1 = order.demand
                if order.process_time == order.batch_time or order.lead_time() == 9:
                    order.done()
                    server1.free()
                    stats_w.add(order.wait_time)
                    stats_lt.add(order.lead_time())
                    stats_ltp.add(floor(order.lead_time() / 3))
                    stats_st.add(order.process_time)
                    stats_demand.add(order.demand)
                    n_done += 1
            n_s2 = 0
            if server2.is_busy():
                order = server2.current_order()
                order.increment_process_time()
                n_s2 = order.demand
                if order.process_time == order.batch_time or order.lead_time() == 9:
                    order.done()
                    server2.free()
                    stats_w.add(order.wait_time)
                    stats_lt.add(order.lead_time())
                    stats_ltp.add(floor(order.lead_time() / 3))
                    stats_st.add(order.process_time)
                    stats_demand.add(order.demand)
                    n_done += 1
            if n_q + n_s1 + n_s2 > 0:
                stats_n_in_system.add(n_q + n_s1 + n_s2)
            cum_time_slot += 1
            if n_done > 0:
                stats.append({
                    'time_slot': cum_time_slot,
                    'mean_wait': stats_w.average(),
                    'mean_lead_time': stats_lt.average(),
                    'mean_lead_time_in_periods': stats_ltp.average(),
                    'mean_service_time': stats_st.average(),
                    'mean_n_in_system': stats_n_in_system.average(),
                    'mean_demand': stats_demand.average(),
                    'sd_wait': stats_w.std_dev(),
                    'sd_lead_time': stats_lt.std_dev(),
                    'sd_lead_time_in_periods': stats_ltp.std_dev(),
                    'sd_service_time': stats_st.std_dev(),
                    'sd_n_in_system': stats_n_in_system.std_dev(),
                    'sd_demand': stats_demand.std_dev(),
                    'n': n_done
                })
    done_orders = [o for o in orders.values() if o.is_done]
    summary = {
        'mean_wait': stats_w.average(),
        'mean_lead_time': stats_lt.average(),
        'mean_lead_time_in_periods': stats_ltp.average(),
        'mean_service_time': stats_st.average(),
        'mean_n_in_system': stats_n_in_system.average(),
        'mean_demand': stats_demand.average(),
        'sd_wait': stats_w.std_dev(),
        'sd_lead_time': stats_lt.std_dev(),
        'sd_lead_time_in_periods': stats_ltp.std_dev(),
        'sd_service_time': stats_st.std_dev(),
        'sd_n_in_system': stats_n_in_system.std_dev(),
        'sd_demand': stats_demand.std_dev(),
        'n': n_done
    }
    return stats, done_orders, summary


def compare_once(data, data_id=-1):
    # Run analytical model
    model_results = am.do_analysis(data)
    # Retrieve results
    mean_wt = model_results['expected_wait_time']
    mean_lt = model_results['expected_lead_time_in_time_slots']
    mean_sf = model_results['expected_shortfall']
    mean_d = model_results['expected_demand']
    sdp_wt = model_results['sd_wait_time']
    sdp_lt = model_results['sd_lead_time_in_time_slots']
    sdp_sf = model_results['sd_shortfall']
    sdp_d = model_results['sd_demand']
    # Run simulation
    sim_length = 10000
    replications = 100
    multi_sim = [simulate(data, sim_length) for _ in range(replications)]
    # Retrieve results
    ave_wt = sum([multi_sim[i][2]['mean_wait'] for i in range(replications)]) / replications
    ave_lt = sum([multi_sim[i][2]['mean_lead_time'] for i in range(replications)]) / replications
    ave_sf = sum([multi_sim[i][2]['mean_n_in_system'] for i in range(replications)]) / replications
    ave_d = sum([multi_sim[i][2]['mean_demand'] for i in range(replications)]) / replications
    sds_wt = sum([multi_sim[i][2]['sd_wait'] for i in range(replications)]) / replications
    sds_lt = sum([multi_sim[i][2]['sd_lead_time'] for i in range(replications)]) / replications
    sds_sf = sum([multi_sim[i][2]['sd_n_in_system'] for i in range(replications)]) / replications
    sds_d = sum([multi_sim[i][2]['sd_demand'] for i in range(replications)]) / replications
    # Print results
    print("%d,ana,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f" %
          (data_id, mean_d, sdp_d, mean_lt, sdp_lt, mean_wt, sdp_wt, mean_sf, sdp_sf))
    print("%d,sim,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f"
          % (data_id, ave_d, sds_d, ave_lt, sds_lt, ave_wt, sds_wt, ave_sf, sds_sf))
    return {
        'data_id': data_id,
        'analytical_model': {
            'mean_demand': mean_d,
            'sd_demand': sdp_d,
            'mean_lead_time': mean_lt,
            'sd_lead_time': sdp_lt,
            'mean_wait': mean_wt,
            'sd_wait': sdp_wt,
            'mean_n_in_system': mean_sf,
            'sd_n_in_system': sdp_sf
        },
        'simulation_model': {
            'mean_demand': ave_d,
            'sd_demand': sds_d,
            'mean_lead_time': ave_lt,
            'sd_lead_time': sds_lt,
            'mean_wait': ave_wt,
            'sd_wait': sds_wt,
            'mean_n_in_system': ave_sf,
            'sd_n_in_system': sds_sf,
            'sim_length': sim_length,
            'replications': replications
        }
    }


def compare_multiple(data_sets):
    results = []
    for data_id, data_set in enumerate(data_sets):
        results.append(compare_once(data_set, data_id))
    t = d_time.datetime.utcnow()
    timestamp = "%s%02d%02d%02d%02d%02d" % (t.year, t.month, t.day, t.hour, t.minute, t.second)
    with open(timestamp + '.json', 'w') as fp:
        json.dump(results, fp)


def convergence_plot(data, sim_length=10_000, replications=100):
    # Run analytical model
    model_results = am.do_analysis(data)

    # Collect results
    mean_wt = model_results['expected_wait_time']
    mean_lt = model_results['expected_lead_time_in_time_slots']
    mean_sf = model_results['expected_shortfall']
    mean_d = model_results['expected_demand']
    sdp_wt = model_results['sd_wait_time']
    sdp_lt = model_results['sd_lead_time_in_time_slots']
    sdp_sf = model_results['sd_shortfall']
    sdp_d = model_results['sd_demand']

    # Run simulation
    multi_sim = [simulate(data, sim_length) for _ in range(replications)]

    # Collect results
    ave_wt = sum([multi_sim[i][2]['mean_wait'] for i in range(replications)]) / replications
    ave_lt = sum([multi_sim[i][2]['mean_lead_time'] for i in range(replications)]) / replications
    ave_sf = sum([multi_sim[i][2]['mean_n_in_system'] for i in range(replications)]) / replications
    ave_d = sum([multi_sim[i][2]['mean_demand'] for i in range(replications)]) / replications
    sds_wt = sum([multi_sim[i][2]['sd_wait'] for i in range(replications)]) / replications
    sds_lt = sum([multi_sim[i][2]['sd_lead_time'] for i in range(replications)]) / replications
    sds_sf = sum([multi_sim[i][2]['sd_n_in_system'] for i in range(replications)]) / replications
    sds_d = sum([multi_sim[i][2]['sd_demand'] for i in range(replications)]) / replications

    print("ana,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f" %
          (mean_d, sdp_d, mean_lt, sdp_lt, mean_wt, sdp_wt, mean_sf, sdp_sf))
    print("sim,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f"
          % (ave_d, sds_d, ave_lt, sds_lt, ave_wt, sds_wt, ave_sf, sds_sf))

    # Lead time plot
    fig, ax = plt.subplots()
    for i in range(replications):
        x = []
        y = []
        for s in multi_sim[i][0]:
            x.append(s['time_slot'] / 3)
            y.append(s['mean_lead_time'])
        ax.plot(x, y, alpha=0.15, c="gray", lw=1)
        if i == 0:
            ax.plot(x, y, alpha=0.15, c="gray", lw=1,
                    label="Simulation average")

    ax.plot([0, sim_length], [mean_lt, mean_lt], c="black",
            label="Population mean")
    # ax.axis([0, sim_length, 1, 4])
    ax.legend()
    ax.set_xlabel("Time periods")
    ax.set_ylabel("Mean lead time")
    plt.show()
    fig.savefig('comparing_models_lead_time.pdf', format='pdf')

    # Shortfall plot
    fig, ax = plt.subplots()
    for i in range(replications):
        x = []
        y = []
        for s in multi_sim[i][0]:
            x.append(s['time_slot'] / 3)
            y.append(s['mean_n_in_system'])
        ax.plot(x, y, alpha=0.15, c="gray", lw=1)
        if i == 0:
            ax.plot(x, y, alpha=0.15, c="gray", lw=1,
                    label="Simulation average")

    ax.plot([0, sim_length], [mean_sf, mean_sf], c="black",
            label="Population mean")
    # ax.axis([0, sim_length, 1, 6])
    ax.legend()
    ax.set_xlabel("Time periods")
    ax.set_ylabel("Mean shortfall")
    plt.show()
    fig.savefig('comparing_models_shortfall.pdf', format='pdf')
