"""
Module for the transformer class. This module contains classes for simulating the thermal dynamics of transformers."""

import sys
sys.path.append('../..')


class OilTypeTransformer:
    def __init__(self, config, global_clock=None, temperature_data=None):
        """
        Propagate transformer state from previous timestep to current timestep (currently it is a minute resolution data
        simulated in 10 seconds increment).
        Nonlinear model from Swift et. Al (2001).

        :param config: Configuration file for transformer parameters.
        :param global_clock: Global clock object. Helps to keep track of time for different objects.
        :param temperature_data: Ambient temperature data for a given region.
        """
        self.temperature_data = temperature_data  # This is used for time-varying ambient temperature.
        self.To = config['top-oil-temp']  # Top oil temperature in degrees Celsius. Initial.
        self.Th = config['hot-spot-temp']  # Hotspot temperature in degrees Celsius. Initial.
        self.Ta = config['ambient-temp']  # Ambient temperature in degrees Celsius. Default.
        self.name = config['name']  # Name of the transformer.

        self.int_method = config['integration-method']  # Integration method ['euler' or 'RK4'].
        self.dt = config['dt']   # Timestep in seconds.
        self.power = config['power']    # Power (kVA).

        self.global_clock = global_clock
        self.To_list = []  # Top oil temperature states.
        self.Th_list = []  # Hot-spot temperature states.
        self.Ta_list = []  # Ambient temperature states.
        self.loading_percents = []  # Loading percent states.
        self.steps = []  # Time steps.

        self._R = config['R']  # Ratio of copper loss to iron loss at rated load.
        self._rated_s = config['rated-power']   # Rated power in kVA.
        self._delta_theta_oil_rated = config['delta_theta_oil_rated']
        self._delta_theta_hs_rated = config['delta_theta_hs_rated']

        self._tau_o = config['tau-topoil']  # Top oil time constant in seconds.
        self._tau_h = config['tau-hotspot']  # Hotspot time constant in seconds.

        # Transformer has various cooling modes that determine 'm' and 'n' for transformer.
        # ONAF: Natural convection flow of oil through windings and radiators. Forced convection flow of air
        # over radiators by fans.
        # ONAN: Natural convection flow of oil through the windings and radiators. Natural convection flow of air
        # over tank and radiation.
        self._m = config['m']
        self._n = config['n']

    def thermal_dynamics(self, power):
        """
        Propagate transformer state from previous timestep to current timestep (currently it is a minute resolution data
        simulated in 10 seconds increment).
        Nonlinear model from Swift et. Al (2001).

        :return: None.
        """
        if self.global_clock.it == 0:
            self.To_list += self.To,  # Top oil temperature states.
            self.Th_list += self.Th,  # Hot-spot temperature states.
            self.Ta_list += self.Ta,  # Ambient temperature states.

        if self.temperature_data is not None:
            self.Ta = self.temperature_data[self.global_clock.it]

        # Propagate transformer thermal state from previous timestep to current timestep
        if self.int_method == 'euler':
            # Usually the transformer integration timestep is much smaller than the powerflow timestep.
            for _ in range(int(self.global_clock.pf_dt / self.dt)):
                # Top oil temperature
                self.To = self.To + self.dt * (((self._R * (power / self._rated_s) ** 2 + 1) / (self._R + 1))
                                               * ((self._delta_theta_oil_rated ** (1 / self._n)) / self._tau_o)
                                               - (1 / self._tau_o) * (max(self.To - self.Ta, 0)) ** (1 / self._n))

                # Hot-spot temperature
                self.Th = self.Th + self.dt * (((power / self._rated_s) ** 2) *
                                               ((self._delta_theta_hs_rated ** (1 / self._m)) / (self._tau_h))
                                               - (1 / self._tau_h) * (max(self.Th - self.To, 0)) ** (1 / self._m))

        #   Append to full data list of transformer thermal states.
        self.Ta_list += self.Ta,
        self.To_list += self.To,
        self.Th_list += self.Th,
        self.loading_percents += (power / self._rated_s)*100,

    def plot_states(self):
        """
        Plot the transformer thermal states.

        :return: None.
        """
        import matplotlib.pyplot as plt
        plt.plot(self.To_list, label='Top oil temperature')
        plt.plot(self.Th_list, label='Hot-spot temperature')
        plt.plot(self.Ta_list, label='Ambient temperature')
        plt.ylabel('Temperature (C)')
        plt.legend()
        plt.show()


