"""This file hosts the class for capacitors. Future work"""
#TODO: develop new class for super-caps for future study: Modelling, constraints, and features


class Capacitor:
    def __init__(self, config):
        self.config = config
        self.name = config["name"]

    def __str__(self):
        return self.name
