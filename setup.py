from setuptools import setup

setup(
    name='EV-EcoSim',
    version='1.0.0',
    packages=['analysis', 'test_cases', 'test_cases.battery', 'test_cases.battery.feeder_population',
              'test_cases.base_case', 'test_cases.base_case.feeder_population', 'charging_sim'],
    url='https://ebalogun01.github.io/EV-EcoSim/',
    license='',
    author='Emmanuel Balogun',
    author_email='ebalogun@stanford.edu',
    description='A grid-aware co-simulation platform for design and optimization of EV Charging Infrastructure'
)
