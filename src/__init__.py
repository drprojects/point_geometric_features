import sys
import os.path as osp
sys.path.append(osp.join(osp.realpath(osp.dirname(osp.dirname(__file__))), "python/bin"))
from pgeof import pgeof

__version__ = '0.0.1'

__all__ = [
    '__version__',
]
