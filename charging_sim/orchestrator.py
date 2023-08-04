"""No use for this file right now. Ignore."""


def load_components(c):
    pass

# the idea was to first initialize objects before assigning them to nodes in
# the charging network.


def initialize_scenario():
    pass


class Orchestrator:
    """USing this to run parallel processes of different scenarios"""
    def __init__(self):
        self.config = None
        self.num_cores = None
