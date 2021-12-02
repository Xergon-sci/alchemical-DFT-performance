""" This module provides base templates to write validators, preprocessors and predictors.
"""

from abc import ABC, abstractmethod

class Validator(ABC):
    """Validator Template for building validators for models.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def validate(self):
        """validate is an abstract method. Each new preprocessor should overwrite this.
        """
        return NotImplementedError

class Preprocessor(ABC):
    """Preprocessor Template for building preprocessors for models.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def preprocess(self):
        """preprocess is an abstract method. Each new preprocessor should overwrite this.
        """
        return NotImplementedError

class Predictor(ABC):
    """Predictor Template for building predictors for models.
    """

    @abstractmethod
    def __init__(self, model, validator, preprocessor):
        self.model_path = model
        self.validator = validator
        self.preprocessor = preprocessor

    @abstractmethod
    def predict(self):
        """predict is an abstract method. Each new preprocessor should overwrite this.
        """
        return NotImplementedError
