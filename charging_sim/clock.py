"""
This module defines the clock class. This allows ease of temporal understanding of different timescales
within the simulation.
"""


class Clock:
    """
    This class defines the clock for the simulation. It allows ease of temporal understanding of different timescales.

    :param config: Configuration dictionary.
    :type config: dict

    Derived attributes:
        pf_dt: Powerflow timestep in seconds. \n
        trans_dt: Transformer timestep in seconds. \n
        battery_dt: Battery timestep in seconds. \n
        dt: Clock timestep in seconds. \n
        it: Iteration number (usually starts at 0). \n

    """
    def __init__(self, config):
        self.pf_dt = config['powerflow-timestep']  # Powerflow timestep in seconds.
        self.trans_dt = config['transformer-timestep']  # Transformer timestep in seconds.
        self.battery_dt = config['battery-timestep']  # Battery timestep in seconds.
        self.dt = config['clock-timestep']  # Clock timestep in seconds.
        self.it = config['it']  # Iteration number (usually starts at 0).

    def update(self):
        """Update the clock."""
        self.it += 1

    def reset(self):
        """Reset the clock."""
        self.it = 0

    def __str__(self):
        """Print the clock."""
        return (f'Clock iteration number: {self.it} \n '
                f'Powerflow timestep: {self.pf_dt}s \n '
                f'Transformer timestep: {self.trans_dt}s \n'
                f'Battery timestep: {self.battery_dt}s \n')