class PatraServerError(Exception):
    """Exception raised for errors when connecting to the Patra Server."""
    def __init__(self, message):
        super().__init__(message)

class PatraIDGenerationError(Exception):
    """Exception raised when Unique ID generation fails."""
    def __init__(self, message):
        super().__init__(message)

class PatraSubmissionError(Exception):
    """Exception raised when model submission to Patra Server fails."""
    def __init__(self, message):
        super().__init__(message)