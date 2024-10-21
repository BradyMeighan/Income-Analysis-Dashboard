import logging
# =============================================================================
# Enhanced Logging Configuration
# =============================================================================
class ColorFormatter(logging.Formatter):
    """
    Custom logging formatter to add colors based on log level and include function names.
    """
    # Define ANSI color codes
    COLOR_CODES = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET_CODE = '\033[0m'

    def format(self, record):
        # Apply color to the level name
        color_code = self.COLOR_CODES.get(record.levelname, self.RESET_CODE)
        levelname_color = f"{color_code}{record.levelname}{self.RESET_CODE}"
        record.levelname = levelname_color
        return super().format(record)

# Create a custom logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Create handlers
file_handler = logging.FileHandler('income_analysis.log')
stream_handler = logging.StreamHandler()

# Create formatters
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')
stream_formatter = ColorFormatter('%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')

# Assign formatters to handlers
file_handler.setFormatter(file_formatter)
stream_handler.setFormatter(stream_formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
