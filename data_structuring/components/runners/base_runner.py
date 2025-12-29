"""
Module providing a base runner class for running the different steps of the data structuring process.
"""
from data_structuring.components.database import Database
from data_structuring.config import BaseSettingsISO


class BaseRunner:
    def __init__(self, config: BaseSettingsISO, database: Database):
        self.config = config
        self.database = database
