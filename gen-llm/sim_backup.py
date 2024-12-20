"""
Simulation Module

This module provides a simple simulation framework using the `simpy` library to model
a production line with sequential workstations. Each component arriving at the production line
undergoes processing at three workstations, each with configurable mean and standard deviation
for processing time. Components arrive at the line based on a normal distribution with a specified
mean and standard deviation.

Classes:
    Simulation: A class that encapsulates the entire simulation environment, managing the
                arrival of components and their processing at each workstation.

Functions:
    run(arrival_mean, arrival_std, ws1_mean, ws1_std, ws2_mean, ws2_std, ws3_mean, ws3_std, simulation_time):
        Executes the simulation with the specified parameters and returns a list of events
        in the format `(time, event_type, component_id, workstation_id)`.

Usage:
    The module can be run as a standalone script to simulate and print events based on default
    parameter values.

Example:
    sim = Simulation()
    event_list = sim.run()
    for event in event_list:
        print(event)

Dependencies:
    - simpy: Simulation library for event-driven discrete-time simulation.
    - numpy: Used to generate normal distributions for arrival and processing times.

"""

import simpy
import numpy as np


class Simulation:
    def __init__(self):
        pass

    def run(
        self,
        arrival_mean=5.0,
        arrival_std=0.1,
        ws1_mean=3.0,
        ws1_std=1.0,
        ws2_mean=2.0,
        ws2_std=2.0,
        ws3_mean=0.5,
        ws3_std=5.0,
        simulation_time=100.0,
    ):
        """
        Runs the simulation with the given parameters.

        Returns:
            events: List of tuples (time, event_type, component_id, workstation_id)
        """
        env = simpy.Environment()
        events = []
        component_id = 0

        # Define resources for each workstation
        ws1 = simpy.Resource(env, capacity=1)
        ws2 = simpy.Resource(env, capacity=1)
        ws3 = simpy.Resource(env, capacity=1)

        def get_positive_normal(mean, std):
            while True:
                sample = np.random.normal(mean, std)
                if sample > 0:
                    return sample

        def component_process(env, cid):
            # Arrival
            events.append((env.now, "arrival", cid, None))

            # Workstation 1
            with ws1.request() as request:
                yield request
                start_time = env.now
                events.append((start_time, "start", cid, 1))
                proc_time = get_positive_normal(ws1_mean, ws1_std)
                yield env.timeout(proc_time)
                end_time = env.now
                events.append((end_time, "end", cid, 1))

            # Workstation 2
            with ws2.request() as request:
                yield request
                start_time = env.now
                events.append((start_time, "start", cid, 2))
                proc_time = get_positive_normal(ws2_mean, ws2_std)
                yield env.timeout(proc_time)
                end_time = env.now
                events.append((end_time, "end", cid, 2))

            # Workstation 3
            with ws3.request() as request:
                yield request
                start_time = env.now
                events.append((start_time, "start", cid, 3))
                proc_time = get_positive_normal(ws3_mean, ws3_std)
                yield env.timeout(proc_time)
                end_time = env.now
                events.append((end_time, "end", cid, 3))

        def arrival_generator(env):
            nonlocal component_id
            while True:
                component_id += 1
                inter_arrival = get_positive_normal(arrival_mean, arrival_std)
                yield env.timeout(inter_arrival)
                env.process(component_process(env, component_id))

        env.process(arrival_generator(env))
        env.run(until=simulation_time)

        # Sort events by time
        events.sort(key=lambda x: x[0])

        return events


# Example usage:
if __name__ == "__main__":
    sim = Simulation()
    event_list = sim.run()
    for event in event_list:
        print(event)
