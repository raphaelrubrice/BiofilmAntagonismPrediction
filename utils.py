import os
import json
import numpy as np
import pandas as pd


def is_gpu_available():
    try:
        import numba.cuda

        return numba.cuda.is_available()
    except ImportError:
        return False
