
class PatraIDGenerationError(Exception):
    """Exception raised when Unique ID generation fails."""
    def __init__(self, message):
        super().__init__(message)

class PatraSubmissionError(Exception):
    """Exception raised when model submission to Patra Server fails. """
    def __init__(self, message):
        super().__init__(message)

class PatraModelExistsError(Exception):
    """Exception raised when model already exists in the Patra Knowledge Graph."""
    def __init__(self, message):
        super().__init__(message)