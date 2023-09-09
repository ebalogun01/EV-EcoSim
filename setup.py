from setuptools import setup

setup(
    name='EV50_cosimulation',
    version='1.0.0',
    packages=['analysis', 'test_cases', 'test_cases.battery', 'test_cases.battery.feeder_population',
              'test_cases.base_case', 'charging_sim', 'feeder_population'],
    url='',
    license='',
    author='Emmanuel Balogun',
    author_email='ebalogun@stanford.edu',
    description='A grid-aware co-simulation platform for design and optimization of EV charging stations'
)
