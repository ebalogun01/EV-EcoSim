# import time
import numpy as np
import math
from maps import BatteryMaps
# from data import raw_data_NMC25degC
import matplotlib.pyplot as plt

print('batterysim ok')

"""Will likely import all batteries with their respective power profiles to update the SOH for each of them"""

# TODO: How to model power delivery as a function of time, temperature, SOC and SOH at every time step. Cap as a...
#  function of discharge rate Fixed capacity
#  How to locate batteries spatially
#  Control Signals given to battery may vary slightly from response. How is the discharge current of battery controlled?
#  Is there a relationship between controls cost (temp control etc) with increasing power delivery by battery?
#  How can we incorporate this response delay due to chemistry or reaction responsiveness into simulation?
#  Finishing these will be great first battery sim environment. Maybe ignore details for now.
#  Think about creating a few degradation models from literature.
#  Need to obtain battery discharge curves for real simulation; can we get sample discharge curves or is it necessary?
#  I want to understand the communication protocol between EV and Station when plugged in and how that will affect batt.
#  How is the power from the battery mixed with that from the grid? I assume voltage must match.
#  Need voltage vs SOC curves to use to infer voltage from SOC so degradation can be computed (depends on C-rate too!).
#  Determining a battery’s state of charge from voltage measurement is  vague enough if current is moving
#  through the battery. The vagaries increase exponentially if no current is moving through the battery.
#  Available capacity is dependent on rate of discharge. This will ideally be dynamic in sim but static in control algo.
#  Need to consider Peukert's law: https://en.wikipedia.org/wiki/Peukert%27s_law
#  Topologies for batteries in series vs. parallel and how does affect design as well. Voltage/current boosting.
#  SOC vs. Voltage vs. Discharge Rate
#  How does solution improve with horizon length (how far is looking ahead important (for MPC no point but computation
#  can become expensive. Horizon length 'N' vs. optimal savings analysis.
#  I see how RL can be used here to learn a policy with time. Opt. Algo with improve itself based on feedback from aging
#  sim module. SHOULD I INCLUDE UNCERTAINTY IN VOLTAGE??


class BatterySim:
    """Current aging model is for LiNiMnCoO2 battery cells. Need to populate more aging models for analyses.
    Link to Paper: https://www.sciencedirect.com/science/article/pii/S0378775314001876

    Params:
        beta_cap: capacity fade aging factor for cycle aging
        alpha_cap capacity fade aging factor for calendar aging
        beta_res: resistance growth aging factor for cycle aging
        alpha_res: resistance growth aging factor for calendar aging
    Assumptions:
        Homogenous battery with dynamics modelled as one big cell
        Constant temperature profile in vicinity of battery
        SOC is observable with no error (this is important for other state approximation (voltage)
        """
    def __init__(self, datetime, num_steps, res=15):
        self.num_steps = num_steps
        # self.battery_objects = battery_objects  # storing all battery objects in a list maintained by sim
        self.time = 1  # seconds in a day
        self.ambient_temp = 35 + 273  # absolute temp in K
        self.aging_params = {}  # to be updated later

        self.res = res
        self.cap = 4.85  # Ah
        BM = BatteryMaps()
        self.response_surface = BM.get_response_surface()[0]  # this is used to infer voltage from other states
        # Include battery chemistry later.

    def infer_voltage(self, battery):
        """This method is used to estimate the voltage from estimated SOC, temp, and current. Do this before
        simulating the degradation."""
         # First estimate the SOC
        # for battery in self.battery_objects:
        SOC = battery.SOC.value[0, 0]
        print(SOC)
        battery.true_SOC[0] = SOC
        battery.true_voltage[0] = self.response_surface([0,  SOC])[0]
        Q = battery.Q.value[0,0]
        for i in range(battery.SOC.value.shape[0]-1):
            current = battery.current.value[i,0]
            if current < 0:     # Discharge Dynamics
                # print(current, SOC)
                true_voltage = self.response_surface([abs(current), SOC])[0]
            else:   # Charge Dynamics
                # print(current, SOC)
                true_voltage = self.response_surface([0, SOC])[0] + current * 0.076 # estimated as OCV + bias for now...RC is low so not t
            if round(current, 8) == 0:
                battery.true_voltage[i+1] = battery.OCV.value[i,0]
                battery.true_SOC[i + 1] = SOC
                continue
            battery.true_voltage[i+1] = true_voltage
            Q = Q + (true_voltage + battery.true_voltage[i])/2 * current * self.res/60  # averaging over the voltage drop
            SOC = (self.cap * SOC + current * self.res/60)/self.cap  # Current can be (+) or (-)
            battery.true_SOC[i+1] = SOC
        battery.true_power = np.multiply(battery.true_voltage[1:], battery.current.value)/1000
        plt.figure()
        plt.plot(battery.true_voltage)
        plt.plot(battery.discharge_voltage.value)
        plt.ylabel('Voltage (V)')
        plt.legend(['True Voltage', 'Controller Estimated Voltage'])
        plt.savefig('voltage_plot_{}_Sim.png'.format(battery.id))
        plt.close()

    def get_cyc_aging(self, battery):
        """Detailed aging for simulation environment. Per Johannes Et. Al"""
        self.infer_voltage(battery)
        SOC_vector = battery.true_SOC
        print("SOC vector is: ", SOC_vector)
        print(self.num_steps)
        del_DOD = SOC_vector[0:self.num_steps] - SOC_vector[1:]
        del_DOD[del_DOD < 0] = 0 # remove the charging parts in profile to get DOD
        del_DOD = np.sum(del_DOD) # total discharge depth
        # del_DOD = np.max(SOC_vector) - np.min(SOC_vector) # this is not entirely accurate
        real_voltage = battery.true_voltage
        avg_voltage = np.sqrt(np.average(real_voltage**2))  # quadratic mean voltage for aging
        beta_cap = 7.348 * 10**-3 * (avg_voltage - 3.667)**2 + 7.6 * 10**-4 + 4.081 * 10**-3 * del_DOD
        # print('Beta Cap is, ', beta_cap)
        beta_res = 2.153 * 10**-4 * (avg_voltage - 3.725)**2 - 1.521 * 10**-5 + 2.798 * 10**-4 * del_DOD
        beta_minimum = 1.5 * 10**-5
        if beta_res < beta_minimum:
            beta_res = beta_minimum     # there is a minimum aging factor that needs to be fixed
        charge_thrput = np.abs(battery.current.value * self.res/60)  # in Ah
        capacity_fade = beta_cap * charge_thrput**0.5 * 2.15/self.cap  # time is one day for both
        resistance_growth = beta_res * charge_thrput * 2.15/self.cap
        battery.true_capacity_loss = capacity_fade
        plt.figure()
        plt.plot(capacity_fade)
        plt.title("plot of capacity loss vs time")
        # plt.show()
        plt.close()

        return np.sum(capacity_fade), np.sum(resistance_growth)

    def update_capacity(self, battery):
        """This uses the electrochemical and impedance-based model per Johannes et. Al"""
        cap_fade = self.get_aging_value(battery)[0]
        battery.SOH -= cap_fade/battery.nominal_cap # change this to nom rating
        battery.cap -= cap_fade
        battery.Qmax = battery.max_SOC * battery.cap * battery.SOH
        battery.true_capacity_loss += cap_fade
        battery.true_aging.append(cap_fade)

    def update_resistance(self, battery):
        """This uses the electrochemical and impedance-based model per Johannes et. Al"""
        res_growth = self.get_aging_value(battery)[1]
        battery.properties["resistance"] += res_growth
        battery.resistance_growth += res_growth

    def get_calendar_aging(self, battery):
        # TODO: This currently runs the aging for an entire day after each iteration run. Will need to formulate
        #  learning architecture
        """Estimates the calendar aging of the battery using Schmalsteig Model (Same as above)"""
        alpha_cap = (7.543 * battery.true_voltage - 23.75) * 10**6 * math.exp(-6976/self.ambient_temp)  # aging factors
        alpha_res = (5.270 * battery.true_voltage - 16.32) * 10**5 * math.exp(-5986/self.ambient_temp)  # temp in K
        capacity_fade = alpha_cap * (self.time/96)**0.75 * 2.15/self.cap # for each time-step (this is scaled up for current cell)
        resistance_growth = alpha_res * (self.time/96)**0.75 * 2.15/self.cap # for each time-step (this is scaled up for current cell)
        battery.true_capacity_loss = capacity_fade
        return sum(capacity_fade), sum(resistance_growth)

    def get_total_aging(self, battery):
        return self.get_cyc_aging(battery) + self.get_calendar_aging(battery)

    def get_aging_value(self, battery):
        """returns the actual aging value lost after a cvxpy run..."""
        cap_fade_cycle, res_growth_cycle = self.get_cyc_aging(battery)
        cap_fade_calendar, res_growth_cal = self.get_calendar_aging(battery)
        capacity_fade = cap_fade_cycle + cap_fade_calendar
        res_growth = res_growth_cal + res_growth_cycle
        return [capacity_fade, res_growth]

    def run(self, battery):
        """need to slightly change this for now"""
        # for battery in self.battery_objects:
        self.update_capacity(battery)
        print("Battery Aging and Response Estimated")

    @staticmethod
    def NMC_cal_aging():
        pass

    @staticmethod
    def NMC_cyc_aging():
        pass

    @staticmethod
    def LFP_cal_aging():
        pass

    @staticmethod
    def LFP_cyc_aging():
        pass
