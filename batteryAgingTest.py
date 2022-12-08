import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib

"""This script was used to test the battery aging model, confirming I can reproduce the cited paper"""

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

def get_calendar_aging(temp):
    # TODO: This currently runs the aging for an entire day after each iteration run. Will need to formulate
    #  learning architecture
    """Estimates the calendar aging of the battery using Schmalsteig Model (Same as above)"""
    avg_voltage = 3.699  # mean voltage from paper to replicate results
    # I THINK THIS MUST BE ESTIMATED IN DAYS? AND I THINK IT HAS TO BE CUMULATIVE? NOT PER TIMESTEP?
    alpha_cap = (7.543 * avg_voltage - 23.75) * 10 ** 6 * math.exp(-6976 / temp)  # aging factors
    alpha_res = (5.270 * avg_voltage - 16.32) * 10 ** 5 * math.exp(-5986 / temp)  # temp in K
    return alpha_cap, alpha_res

def main():
    """ Run the simple test here """
    temp_range = (35+273.15, 60+273.15)
    num_points = 8
    temperatures = np.linspace(temp_range[0], temp_range[1], num_points)
    log_ALPHA_CAP_LIST = []
    log_ALPHA_RES_LIST = []
    ALPHA_CAP_LIST = []
    ALPHA_RES_LIST = []
    for T in temperatures:
        ALPHA_cap, ALPHA_res = get_calendar_aging(T)
        log_ALPHA_CAP_LIST.append(np.log(ALPHA_cap))
        log_ALPHA_RES_LIST.append(np.log(ALPHA_res))
        ALPHA_CAP_LIST.append(ALPHA_cap)
        ALPHA_RES_LIST.append(ALPHA_res)

    plt.plot(1/temperatures, log_ALPHA_CAP_LIST)
    plt.plot(1/temperatures, log_ALPHA_RES_LIST)
    plt.ylabel(r" \log aging factor")
    plt.show()
    print(ALPHA_CAP_LIST)
    print(ALPHA_RES_LIST)


if __name__ == "__main__":
    main()

