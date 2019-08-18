"""

This file performs training for a provided configuration. I have written it to be somewhat extensible (but it is not a general purpose training/testing framework).

We assume torch is the DNN framework of choice
"""

import os, shutil, glob, json
import tqdm 
import math, random
import logging
import re
from collections import defaultdict
import numpy as np
from bisect import bisect_right

from models import ReidModel




