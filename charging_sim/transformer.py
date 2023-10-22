"""This file simulates the transformer thermal state by propagating it forward at each time step """

import numpy as np
import gblvar
import sys
sys.path.append('../..')


class Transformer:
    def __init__(self, temperature_data=None, config=None):
        """
        Propagate transformer state from previous timestep to current timestep (currently it is a minute resolution data
        simulated in 10 seconds increment).
        Nonlinear model from Swift et. Al (2001).
        Inputs: temperature_data - ambient temperature data for a given region.
        Returns: -1, just arbitrarily.
        """

        #TODO: include transformer parameters in config file

        self.temperature_data = temperature_data    # This is used for time-varying ambient temperature.
        self.To = config['top-oil-temp']    # Top oil temperature in degrees Celsius.
        self.Th = config['hot-spot-temp']   # Hotspot temperature in degrees Celsius.
        self.Ta = config['ambient-temp']    # Ambient temperature in degrees Celsius.

        self.R = config['resistance']   # Ratio of copper loss to iron loss at rated load.
        self.rated_s = config['rated-power']
        self.delta_theta_oil_rated = config['delta-theta-oil-rated']

        self.tau_o = config['tau-oil']  # Top oil time constant in seconds.
        self.tau_h = config['tau-h']  # Hotspot time constant in seconds.

        self.n = config['n']    #
        self.delta_theta_hs_rated = config['delta-theta-hs-rated']

        self.m = config['m']
        self.int_method = config['Integration-method']
        self.dt = config['Time-step']
        self.power = config['Power']

        self.global_clock = gblvar.it
        self.To_list = []   # Top oil temperature states.
        self.Th_list = []   # Hot-spot temperature states.

    def thermal_dynamics(self):
        """
        Propagate transformer state from previous timestep to current timestep (currently it is a minute resolution data
        simulated in 10 seconds increment).
        Nonlinear model from Swift et. Al (2001).

        :return: None.
        """
        if self.global_clock == 0:
            self.To_list += self.To,   # Top oil temperature states.
            gblvar.trans_Th = np.ones((1, len(gblvar.trans_list))) * float(gblvar.trans_Th0)

        if self.temperature_data is not None:
            self.Ta = self.temperature_data[gblvar.it]

        # Propagate transformer thermal state from previous timestep to current timestep
        if self.int_method == 'euler':
            for _ in range(int(self.global_clock.pf_dt / self.dt)):
                # Top oil temperature
                self.To = self.To + self.dt * (((self.R * (
                        self.power / self.rated_s) ** 2 + 1) / (self.R + 1))
                                                                       * ((self.delta_theta_oil_rated ** (
                                    1 / self.n)) / self.tau_o)
                                    - (1 / self.tau_o) * (max(self.To - self.Ta, 0)) ** (1 / self.n))

                # Hot-spot temperature
                self.Th = self.Th + self.dt * (((self.power / self.rated_s) ** 2) *
                                                       ((self.delta_theta_hs_rated ** (1 / self.m)) / (self.tau_h))
                                                                       - (1 / self.tau_h) * (
                                                                           max(self.Th - self.To, 0)) ** (1 / self.m))
        #   append to full data array of temperatures

    @staticmethod
    def ref():
        # transformer properties
        trans_dt = 10.0  # integration timestep [seconds].
        trans_Ta = 20.0  # ambient temperature[C] {SLIGHTLY HIGHER THAN JUNE AVERAGE IN 2018}

        # Transformer has various cooling modes that determine m and n for transformer.
        # ONAF: Natural convection flow of oil through windings and radiators. Forced convection flow of air over radiators by
        # fans.
        # More details can be found in the transformer modelling guide per IEEE C57.91-1995.

        # ONAN: Natural convection flow of oil through the windings and radiators. Natural convection flow of air over tank
        # and radiation.

        trans_R = 5.0  # ratio of copper loss to iron loss at rated load
        trans_tau_o = 2 * 60 * 60.0  # top oil time constant in seconds
        trans_tau_h = 6 * 60.0  # hotspot time constant in seconds
        trans_n = 0.9  # Depends on transformer cooling mode (forced vs. natural convection). Typical values.
        trans_m = 0.8  # Depends on transformer cooling mode (forced vs. natural convection). Typical values.
        trans_delta_theta_hs_rated = 28.0  # NEED TO DECIDE HOW IMPORTANT THESE WILL BE
        trans_delta_theta_oil_rated = 36.0  # todo: double-verify

        trans_To0 = 30.0  # initial oil temperature [C] (assume we start at a point where oil is slightly hotter than ambient)
        trans_Th0 = 60.0  # initial hot spot temperature [C]    # How is this set? (should not matter long-term)
        trans_int_method = 'euler'  # integration method ['euler' or 'RK4']



def sim_transformer(temperature_data=None):
    """
    Propagate transformer state from previous timestep to current timestep (currently it is a minute resolution data
    simulated in 10 seconds increment).
    Nonlinear model from Swift et. Al (2001).
    Inputs: temperature_data - ambient temperature data for a given region.
    Returns: -1, just arbitrarily.
    """
    # todo: make this a class

    #   if first timestep, initialize state of transformer
    if gblvar.it == 0:
        gblvar.trans_To = np.ones((1, len(gblvar.trans_list))) * float(gblvar.trans_To0)
        gblvar.trans_Th = np.ones((1, len(gblvar.trans_list))) * float(gblvar.trans_Th0)

    if temperature_data is not None:
        self.Ta = self.temperature_data[gblvar.it]

    # Propagate transformer thermal state from previous timestep to current timestep
    if gblvar.trans_int_method == 'euler':
        trans_To_new= gblvar.trans_To[gblvar.it - 1, :]
        trans_Th_new= gblvar.trans_Th[gblvar.it - 1, :]

        # loop through transformers
        for i in range(len(gblvar.trans_list)):
            #   integrate across powerflow timestep
            for _ in range(int(gblvar.pf_dt / gblvar.trans_dt)):
                trans_To_new[i]= trans_To_new[i] + gblvar.trans_dt * (((gblvar.trans_R * (
                        gblvar.trans_power[i] / gblvar.trans_rated_s[i]) ** 2 + 1) / (gblvar.trans_R + 1))
                                                                      * ((gblvar.trans_delta_theta_oil_rated ** (1 / gblvar.trans_n)) / gblvar.trans_tau_o)
                                                                      - (1 / gblvar.trans_tau_o) * (max(trans_To_new[i] - gblvar.trans_Ta, 0)) ** (1 / gblvar.trans_n))  # top oil temperature

                trans_Th_new[i]= trans_Th_new[i] + gblvar.trans_dt * (((
                                                                               gblvar.trans_power[i] / gblvar.trans_rated_s[i]) ** 2) *
                                                                      ((gblvar.trans_delta_theta_hs_rated ** (1 / gblvar.trans_m)) / (
                                                                          gblvar.trans_tau_h))
                                                                      - (1 / gblvar.trans_tau_h) * (max(trans_Th_new[i] - trans_To_new[i], 0)) ** (1 / gblvar.trans_m))  # hot-spot temperature
        #   append to full data array of temperatures
        gblvar.trans_To = np.concatenate((gblvar.trans_To, trans_To_new.reshape(1, -1)), axis=0)   # maybe use list ?
        gblvar.trans_Th = np.concatenate((gblvar.trans_Th, trans_Th_new.reshape(1, -1)), axis=0)

    return -1


