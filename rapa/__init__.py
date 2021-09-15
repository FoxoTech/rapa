from ._version import __version__

# Warnings from dependencies are suppressed everywhere, propagating into all created loggers
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from . import rapa, base, utils