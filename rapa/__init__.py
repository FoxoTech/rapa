# Warnings from dependencies are suppressed everywhere, propagating into all created loggers
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from . import Project
from . import version