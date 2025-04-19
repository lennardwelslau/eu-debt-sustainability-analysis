class MissingData(Exception):
    """Raised when required data is missing (unexpectedly NaN)"""

    def __init__(self, what, year, index):
        super().__init__(f"Missing data: {what} for year {year} (index {index})")
