import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.src.api_export import keras_export
from keras.src.losses import LossFunctionWrapper
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.utils import resample
from tensorflow import keras
from tensorflow.keras import layers
from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
import numpy as np
from typing import Dict, List, Hashable
from Bio.PDB import PDBParser
from scipy.spatial.distance import pdist, squareform
from scipy.special import softmax
import os, tempfile
from io import StringIO
from typing import Literal, Union, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from Bio import AlignIO
from Bio.Align.Applications import MafftCommandline
from collections import Counter
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Create a reverse mapping from a AA_ENCODINGS vector to an amino acid character.
# The score lists are converted to tuples so they can be used as dictionary keys.
# Constants
PHYSYCHE_PROPERTIES = {
    "A": [0.51, 0.169, 0.471, 0.279, 0.141, 0.294, 0, 0.262, 0.512, 0, 0.404],
    "R": [0.667, 0.726, 0.321, 1, 0.905, 0.529, 0.327, 0.169, 0.372, 1, 1],
    "N": [0.745, 0.39, 0.164, 0.658, 0.51, 0.235, 0.14, 0.313, 0.116, 0.065, 0.33],
    "D": [0.745, 0.304, 0.021, 0.793, 0.515, 0.235, 0.14, 0.601, 0.14, 0.956, 0],
    "C": [0.608, 0.314, 0.76, 0.072, 0, 0.559, 0.14, 0.947, 0.907, 0.028, 0.285],
    "Q": [0.667, 0.531, 0.178, 0.649, 0.608, 0.529, 0.14, 0.416, 0.023, 0.068, 0.36],
    "E": [0.667, 0.482, 0.092, 0.883, 0.602, 0.529, 0.14, 0.561, 0.163, 0.96, 0.056],
    "G": [0, 0, 0.275, 0.189, 0.103, 0, 0, 0.24, 0.581, 0, 0.401],
    "H": [0.686, 0.554, 0.326, 0.468, 0.402, 0.529, 0.14, 0.313, 0.581, 0.992, 0.603],
    "I": [1, 0.65, 1, 0, 0.083, 0.824, 0.308, 0.424, 0.93, 0.003, 0.407],
    "L": [0.961, 0.65, 0.734, 0.081, 0.138, 0.824, 0.308, 0.463, 0.907, 0.003, 0.402],
    "K": [0.667, 0.692, 0, 0.568, 1, 0.529, 0.327, 0.313, 0, 0.952, 0.872],
    "M": [0.765, 0.612, 0.603, 0.171, 0.206, 0.765, 0.308, 0.405, 0.814, 0.028, 0.372],
    "F": [0.686, 0.772, 0.665, 0, 0.114, 0.853, 0.682, 0.462, 1, 0.007, 0.339],
    "P": [0.353, 0.372, 0.012, 0.198, 0.411, 0.588, 0.271, 0, 0.302, 0.03, 0.442],
    "S": [0.52, 0.172, 0.155, 0.477, 0.303, 0.206, 0, 0.24, 0.419, 0.032, 0.364],
    "T": [0.49, 0.349, 0.256, 0.523, 0.337, 0.235, 0.14, 0.313, 0.419, 0.032, 0.362],
    "W": [0.686, 1, 0.681, 0.207, 0.219, 1, 1, 0.537, 0.674, 0.04, 0.39],
    "Y": [0.686, 0.796, 0.591, 0.477, 0.454, 0.853, 0.682, 1, 0.419, 0.031, 0.362],
    "V": [0.745, 0.487, 0.859, 0.036, 0.094, 0.647, 0.234, 0.369, 0.674, 0.003, 0.399],
    "B": [0.745, 0.347, 0.093, 0.726, 0.513, 0.235, 0.14, 0.457, 0.128, 0.511, 0.165],  # Average of N and D
    "Z": [0.667, 0.507, 0.135, 0.766, 0.605, 0.529, 0.14, 0.489, 0.093, 0.514, 0.208],  # Average of Q and E
    "X": [0.628, 0.485, 0.408, 0.358, 0.314, 0.518, 0.236, 0.408, 0.522, 0.265, 0.418],
    # Average of all 20 standard amino acids
}

BLOSUM62 = {
    'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, -2, -1, 0],
    'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, -1, 0, -1],
    'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 3, 0, -1],
    'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 4, 1, -1],
    'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2],
    'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0, 3, -1],
    'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1],
    'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, -1, -2, -1],
    'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0, 0, -1],
    'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, -3, -3, -1],
    'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, -4, -3, -1],
    'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0, 1, -1],
    'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, -3, -1, -1],
    'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, -3, -3, -1],
    'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, -2, -1, -2],
    'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0, 0, 0],
    'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, -1, -1, 0],
    'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, -4, -3, -2],
    'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, -3, -2, -1],
    'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, -3, -2, -1],
    'B': [-2, -1, 3, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2, 0, -1, -4, -3, -3, 4, 1, -1],
    'Z': [-1, 0, 0, 1, -3, 3, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0, -1, -3, -2, -2, 1, 4, -1],
    'X': [0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, 0, 0, -2, -1, -1, -1, -1, -1],
}

BLOSUM62_N = {
    "A": [0.533, 0.2, 0.133, 0.133, 0.267, 0.2, 0.2, 0.267, 0.133, 0.2, 0.2, 0.2, 0.2, 0.133, 0.2, 0.333, 0.267, 0.067,
          0.133, 0.267, 0.133, 0.2, 0.267],
    "R": [0.2, 0.6, 0.267, 0.133, 0.067, 0.333, 0.267, 0.133, 0.267, 0.067, 0.133, 0.4, 0.2, 0.067, 0.133, 0.2, 0.2,
          0.067, 0.133, 0.067, 0.2, 0.267, 0.2],
    "N": [0.133, 0.267, 0.667, 0.333, 0.067, 0.267, 0.267, 0.267, 0.333, 0.067, 0.067, 0.267, 0.133, 0.067, 0.133,
          0.333, 0.267, 0.0, 0.133, 0.067, 0.467, 0.267, 0.2],
    "D": [0.133, 0.133, 0.333, 0.667, 0.067, 0.267, 0.4, 0.2, 0.2, 0.067, 0.0, 0.2, 0.067, 0.067, 0.2, 0.267, 0.2, 0.0,
          0.067, 0.067, 0.533, 0.333, 0.2],
    "C": [0.267, 0.067, 0.067, 0.067, 0.867, 0.067, 0.0, 0.067, 0.067, 0.2, 0.2, 0.067, 0.2, 0.133, 0.067, 0.2, 0.2,
          0.133, 0.133, 0.2, 0.067, 0.067, 0.133],
    "Q": [0.2, 0.333, 0.267, 0.267, 0.067, 0.6, 0.4, 0.133, 0.267, 0.067, 0.133, 0.333, 0.267, 0.067, 0.2, 0.267, 0.2,
          0.133, 0.2, 0.133, 0.267, 0.467, 0.2],
    "E": [0.2, 0.267, 0.267, 0.4, 0.0, 0.4, 0.6, 0.133, 0.267, 0.067, 0.067, 0.333, 0.133, 0.067, 0.2, 0.267, 0.2,
          0.067, 0.133, 0.133, 0.333, 0.533, 0.2],
    "G": [0.267, 0.133, 0.267, 0.2, 0.067, 0.133, 0.133, 0.667, 0.133, 0.0, 0.0, 0.133, 0.067, 0.067, 0.133, 0.267,
          0.133, 0.133, 0.067, 0.067, 0.2, 0.133, 0.2],
    "H": [0.133, 0.267, 0.333, 0.2, 0.067, 0.267, 0.267, 0.133, 0.8, 0.067, 0.067, 0.2, 0.133, 0.2, 0.133, 0.2, 0.133,
          0.133, 0.4, 0.067, 0.267, 0.267, 0.2],
    "I": [0.2, 0.067, 0.067, 0.067, 0.2, 0.067, 0.067, 0.0, 0.067, 0.533, 0.4, 0.067, 0.333, 0.267, 0.067, 0.133, 0.2,
          0.067, 0.2, 0.467, 0.067, 0.067, 0.2],
    "L": [0.2, 0.133, 0.067, 0.0, 0.2, 0.133, 0.067, 0.0, 0.067, 0.4, 0.533, 0.133, 0.4, 0.267, 0.067, 0.133, 0.2,
          0.133, 0.2, 0.333, 0.0, 0.067, 0.2],
    "K": [0.2, 0.4, 0.267, 0.2, 0.067, 0.333, 0.333, 0.133, 0.2, 0.067, 0.133, 0.6, 0.2, 0.067, 0.2, 0.267, 0.2, 0.067,
          0.133, 0.133, 0.267, 0.333, 0.2],
    "M": [0.2, 0.2, 0.133, 0.067, 0.2, 0.267, 0.133, 0.067, 0.133, 0.333, 0.4, 0.2, 0.6, 0.267, 0.133, 0.2, 0.2, 0.2,
          0.2, 0.333, 0.067, 0.2, 0.2],
    "F": [0.133, 0.067, 0.067, 0.067, 0.133, 0.067, 0.067, 0.067, 0.2, 0.267, 0.267, 0.067, 0.267, 0.667, 0.0, 0.133,
          0.133, 0.333, 0.467, 0.2, 0.067, 0.067, 0.2],
    "P": [0.2, 0.133, 0.133, 0.2, 0.067, 0.2, 0.2, 0.133, 0.133, 0.067, 0.067, 0.2, 0.133, 0.0, 0.733, 0.2, 0.2, 0.0,
          0.067, 0.133, 0.133, 0.2, 0.133],
    "S": [0.333, 0.2, 0.333, 0.267, 0.2, 0.267, 0.267, 0.267, 0.2, 0.133, 0.133, 0.267, 0.2, 0.133, 0.2, 0.533, 0.333,
          0.067, 0.133, 0.133, 0.267, 0.267, 0.267],
    "T": [0.267, 0.2, 0.267, 0.2, 0.2, 0.2, 0.2, 0.133, 0.133, 0.2, 0.2, 0.2, 0.2, 0.133, 0.2, 0.333, 0.6, 0.133, 0.133,
          0.267, 0.2, 0.2, 0.267],
    "W": [0.067, 0.067, 0.0, 0.0, 0.133, 0.133, 0.067, 0.133, 0.133, 0.067, 0.133, 0.067, 0.2, 0.333, 0.0, 0.067, 0.133,
          1.0, 0.4, 0.067, 0.0, 0.067, 0.133],
    "Y": [0.133, 0.133, 0.133, 0.067, 0.133, 0.2, 0.133, 0.067, 0.4, 0.2, 0.2, 0.133, 0.2, 0.467, 0.067, 0.133, 0.133,
          0.4, 0.733, 0.2, 0.067, 0.133, 0.2],
    "V": [0.267, 0.067, 0.067, 0.067, 0.2, 0.133, 0.133, 0.067, 0.067, 0.467, 0.333, 0.133, 0.333, 0.2, 0.133, 0.133,
          0.267, 0.067, 0.2, 0.533, 0.067, 0.133, 0.2],
    "B": [0.133, 0.2, 0.467, 0.533, 0.067, 0.267, 0.333, 0.2, 0.267, 0.067, 0.0, 0.267, 0.067, 0.067, 0.133, 0.267, 0.2,
          0.0, 0.067, 0.067, 0.533, 0.333, 0.2],
    "Z": [0.2, 0.267, 0.267, 0.333, 0.067, 0.467, 0.533, 0.133, 0.267, 0.067, 0.067, 0.333, 0.2, 0.067, 0.2, 0.267, 0.2,
          0.067, 0.133, 0.133, 0.333, 0.533, 0.2],
    "X": [0.267, 0.2, 0.2, 0.2, 0.133, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.133, 0.267, 0.267, 0.133, 0.2,
          0.2, 0.2, 0.2, 0.2]

}

Physicochemical_Properties_log = {
    "A": [0.2369, 0.0616, 0.5125, 1.0000, 0.5043, 0.1085, 0.1652, 0.0000, 0.0000, 0.0000, 0.2490, 0.3951, 0.4043,
          0.4819],
    "C": [0.0000, 0.5517, 0.4961, 0.2996, 0.4866, 0.3566, 0.3304, 0.0000, 0.0000, 0.0000, 0.2048, 0.0741, 0.2879,
          0.2229],
    "D": [0.7539, 0.5065, 0.1405, 0.5054, 0.0000, 0.4496, 0.3739, 0.0000, 0.0000, 1.0000, 0.8675, 1.0000, 0.0000,
          0.8042],
    "E": [0.8507, 0.0197, 0.7938, 0.6060, 0.3918, 0.5581, 0.5304, 0.0000, 0.0000, 0.6667, 0.8112, 0.9136, 0.0563,
          0.8283],
    "F": [0.1062, 0.2591, 0.8465, 0.4681, 0.5952, 0.6977, 0.7565, 1.0000, 0.0000, 0.0000, 0.0763, 0.0370, 0.3392,
          0.0000],
    "G": [0.3021, 0.8810, 0.7751, 0.8580, 0.8643, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.5062, 0.4005,
          0.6687],
    "H": [0.5290, 0.3071, 0.3929, 0.1769, 0.5154, 0.6202, 0.6087, 0.0000, 0.3333, 0.3333, 0.1124, 0.6790, 0.6033,
          0.5482],
    "I": [0.0328, 0.2710, 0.8771, 0.6817, 0.6610, 0.4341, 0.6609, 0.0000, 0.0000, 0.0000, 0.6707, 0.0370, 0.4068,
          0.0964],
    "K": [1.0000, 0.2671, 0.6737, 0.5005, 0.7965, 0.5504, 0.7565, 0.0000, 0.6667, 0.0000, 0.6867, 0.7901, 0.8723,
          0.9006],
    "L": [0.1021, 0.1490, 0.4143, 0.9178, 0.3795, 0.4341, 0.6609, 0.0000, 0.0000, 0.0000, 0.2811, 0.0000, 0.4018,
          0.1265],
    "M": [0.2142, 0.0000, 0.8883, 0.3037, 0.7255, 0.5736, 0.6609, 0.0000, 0.0000, 0.0000, 0.0000, 0.0988, 0.3717,
          0.2470],
    "N": [0.7209, 0.6524, 0.7712, 0.5297, 0.6801, 0.4419, 0.4174, 0.0000, 0.3333, 0.3333, 0.6747, 0.8272, 0.3304,
          0.9247],
    "P": [0.4827, 1.0000, 0.3986, 0.6893, 0.3014, 0.3101, 0.3652, 0.0000, 0.0000, 0.0000, 0.8594, 0.3827, 0.4418,
          0.6054],
    "Q": [0.7164, 0.3731, 0.2234, 0.4394, 0.2263, 0.5504, 0.5739, 0.0000, 0.3333, 0.3333, 0.7952, 0.6914, 0.3605,
          0.9006],
    "R": [0.9077, 0.4075, 0.7970, 0.6944, 1.0000, 0.7674, 0.8696, 0.0000, 1.0000, 0.0000, 0.9398, 0.6914, 1.0000,
          1.0000],
    "S": [0.3513, 0.8108, 0.0000, 0.7566, 0.0969, 0.2326, 0.2174, 0.0000, 0.3333, 0.3333, 0.8514, 0.5309, 0.3642,
          0.6325],
    "T": [0.4130, 0.5132, 0.8875, 0.8210, 0.7420, 0.3411, 0.3913, 0.0000, 0.3333, 0.3333, 0.5984, 0.4568, 0.3542,
          0.5723],
    "V": [0.0019, 0.3454, 0.5366, 0.9113, 0.3225, 0.3256, 0.4957, 0.0000, 0.0000, 0.0000, 0.3655, 0.1235, 0.3992,
          0.2259],
    "W": [0.2357, 0.4252, 0.6914, 0.0000, 0.4981, 1.0000, 1.0000, 1.0000, 0.3333, 0.0000, 0.0402, 0.0617, 0.3905,
          0.0392],
    "Y": [0.5050, 0.6530, 1.0000, 0.3488, 0.7744, 0.8217, 0.8087, 1.0000, 0.3333, 0.3333, 0.5020, 0.1605, 0.3617,
          0.2500],
    "B": [0.7376, 0.5795, 0.4558, 0.5176, 0.3400, 0.4457, 0.3957, 0.0000, 0.1667, 0.6667, 0.7711, 0.9136, 0.1652,
          0.8645],
    "Z": [0.7836, 0.1964, 0.5086, 0.5227, 0.3090, 0.5543, 0.5522, 0.0000, 0.1667, 0.5000, 0.8032, 0.8025, 0.2090,
          0.8645],
    "X": [0.4231, 0.4227, 0.6058, 0.5754, 0.5281, 0.4651, 0.4957, 0.0000, 0.1667, 0.1667, 0.4217, 0.3827, 0.4043,
          0.5181],
}

# z-score normalized physicochemical properties
Physicochemical_Properties = {
    "A": [-0.7115, -1.3796, -0.3237, 1.7644, -0.0249, -1.7046, -1.5572, -0.3873, -0.7860, -0.7889, -0.9536, -0.1956,
          0.0770, -0.1703],
    "C": [-1.4936, 0.5087, -0.3881, -1.1136, -0.0977, -0.5661, -0.8431, -0.3873, -0.7860, -0.7889, -1.0949, -1.1785,
          -0.4755, -0.9901],
    "D": [0.9952, 0.3345, -1.7827, -0.2679, -2.0926, -0.1392, -0.6552, -0.3873, -0.7860, 2.8401, 1.0251, 1.6568,
          -1.8416, 0.8496],
    "E": [1.3145, -1.5410, 0.7795, 0.1454, -0.4864, 0.3589, 0.0212, -0.3873, -0.7860, 1.6305, 0.8452, 1.3922, -1.5743,
          0.9259],
    "F": [-1.1431, -0.6188, 0.9861, -0.4213, 0.3478, 0.9993, 0.9983, 2.5820, -0.7860, -0.7889, -1.5060, -1.2919,
          -0.2319, -1.6955],
    "G": [-0.4962, 1.7772, 0.7061, 1.1811, 1.4511, -2.2027, -2.2712, -0.3873, -0.7860, -0.7889, 1.4490, 0.1446, 0.0591,
          0.4207],
    "H": [0.2526, -0.4339, -0.7929, -1.6181, 0.0205, 0.6435, 0.3595, -0.3873, 0.5531, 0.4208, -1.3904, 0.6739, 1.0214,
          0.0394],
    "I": [-1.3855, -0.5728, 1.1059, 0.4566, 0.6176, -0.2104, 0.5850, -0.3873, -0.7860, -0.7889, 0.3955, -1.2919, 0.0888,
          -1.3905],
    "K": [1.8074, -0.5878, 0.3083, -0.2879, 1.1733, 0.3233, 0.9983, -0.3873, 1.8922, -0.7889, 0.4469, 1.0141, 2.2985,
          1.1546],
    "L": [-1.1567, -1.0430, -0.7090, 1.4266, -0.5365, -0.2104, 0.5850, -0.3873, -0.7860, -0.7889, -0.8508, -1.4053,
          0.0651, -1.2951],
    "M": [-0.7864, -1.6169, 1.1498, -1.0969, 0.8821, 0.4300, 0.5850, -0.3873, -0.7860, -0.7889, -1.7501, -1.1029,
          -0.0775, -0.9139],
    "N": [0.8860, 0.8966, 0.6906, -0.1679, 0.6957, -0.1748, -0.4673, -0.3873, 0.5531, 0.4208, 0.4083, 1.1276, -0.2735,
          1.2309],
    "P": [0.0997, 2.2356, -0.7704, 0.4877, -0.8570, -0.7796, -0.6928, -0.3873, -0.7860, -0.7889, 0.9994, -0.2334,
          0.2552, 0.2205],
    "Q": [0.8714, -0.1795, -1.4577, -0.5391, -1.1649, 0.3233, 0.2091, -0.3873, 0.5531, 0.4208, 0.7938, 0.7117, -0.1309,
          1.1546],
    "R": [1.5027, -0.0470, 0.7919, 0.5088, 2.0074, 1.3194, 1.4869, -0.3873, 3.2313, -0.7889, 1.2563, 0.7117, 2.9044,
          1.4692],
    "S": [-0.3340, 1.5068, -2.3337, 0.7644, -1.6952, -1.1354, -1.3317, -0.3873, 0.5531, 0.4208, 0.9737, 0.2203, -0.1131,
          0.3063],
    "T": [-0.1301, 0.3601, 1.1468, 1.0288, 0.9495, -0.6373, -0.5801, -0.3873, 0.5531, 0.4208, 0.1642, -0.0066, -0.1606,
          0.1156],
    "V": [-1.4874, -0.2864, -0.2293, 1.4000, -0.7702, -0.7084, -0.1291, -0.3873, -0.7860, -0.7889, -0.5810, -1.0273,
          0.0532, -0.9806],
    "W": [-0.7157, 0.0214, 0.3776, -2.3448, -0.0503, 2.3868, 2.0506, 2.5820, 0.5531, -0.7889, -1.6217, -1.2163, 0.0116,
          -1.5716],
    "Y": [0.1736, 0.8987, 1.5881, -0.9113, 1.0824, 1.5685, 1.2238, 2.5820, 0.5531, 0.4208, -0.1441, -0.9139, -0.1250,
          -0.9043],
    "B": [0.9411, 0.6155, -0.5463, -0.2179, -0.6988, -0.1570, -0.5613, -0.3873, -0.1164, 1.6305, 0.7167, 1.3922,
          -1.0576, 1.0403],
    "Z": [1.0929, -0.8603, -0.3391, -0.1968, -0.8257, 0.3411, 0.1152, -0.3873, -0.1164, 1.0256, 0.8195, 1.0519, -0.8497,
          1.0403],
    "X": [-0.0969, 0.0118, 0.0422, 0.0199, 0.0726, -0.0681, -0.1291, -0.3873, -0.1164, -0.1841, -0.4011, -0.2334,
          0.0770, -0.0560],
}

AA_ENCODINGS = Physicochemical_Properties

# Create a reverse mapping from a AA_ENCODINGS vector to an amino acid character.
# The score lists are converted to tuples so they can be used as dictionary keys.
MASK_TOKEN = -1.0
NORM_TOKEN = 1.0
PAD_TOKEN = -2.0
PAD_VALUE = 0.0
MASK_VALUE = 0.0

AMINO_ACID_MAP = {tuple(v): k for k, v in AA_ENCODINGS.items()}

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INT = {a: i for i, a in enumerate(AA)}
PAD_INDEX_OHE = len(AA)  # 20 total chars for OHE

# Create a mapping from Amino Acid to an integer index
AA_23 = "ARNDCQEGHILKMFPSTWYVBZX"
AA_TO_BLOSUM_IDX = {aa: i for i, aa in enumerate(AA_23)}
PAD_INDEX_62 = len(AA_23)  # The new padding index is 23

# Create a constant TensorFlow tensor to act as a lookup table
AA_ENCODINGS_VECTORS = np.array([AA_ENCODINGS[aa] for aa in AA_23] + [[0.0] * len(AA_ENCODINGS[AA_23[0]])],
                                dtype=np.float32)


def seq_to_onehot(sequence: str, max_seq_len: int) -> np.ndarray:
    """Convert peptide sequence to one-hot encoding"""
    arr = np.full((max_seq_len, 21), PAD_VALUE, dtype=np.float32)  # initialize padding with 0
    for j, aa in enumerate(sequence.upper()[:max_seq_len]):
        arr[j, AA_TO_INT.get(aa, PAD_INDEX_OHE)] = 1.0
        # print number of UNKs in the sequence
    # num_unks = np.sum(arr[:, UNK_IDX_OHE])
    # zero out gaps
    arr[:, PAD_INDEX_OHE] = PAD_VALUE  # Set gaps to PAD_VALUE
    # if num_unks > 0:
    #     print(f"Warning: {num_unks} unknown amino acids in sequence '{sequence}'")
    return arr


def OHE_to_seq(ohe: np.ndarray, gap: bool = False) -> list:
    """
    Convert a one-hot encoded matrix back to a peptide sequence.
    # (B, max_pep_len, 21) -> (B, max_pep_len)
    Args:
        ohe: One-hot encoded matrix of shape (B, N, 21).
    Returns:
        sequence: Peptide sequence as a string. (B,)
    """
    sequence = []
    for i in range(ohe.shape[0]):  # Iterate over batch dimension
        seq = []
        for j in range(ohe.shape[1]):  # Iterate over sequence length
            if gap and np.all(ohe[i, j] == 0):
                seq.append('-')
            else:
                aa_index = np.argmax(ohe[i, j])  # Get index of the max value in one-hot encoding
                if aa_index < len(AA):  # Check if it's a valid amino acid index
                    seq.append(AA[aa_index])
                else:
                    seq.append('-')  # Use '-' for unknown amino acids
        sequence.append(''.join(seq))  # Join the list into a string
    return sequence  # Return list of sequences


def OHE_to_seq_single(ohe: np.ndarray, gap=False) -> str:
    """
    Convert a one-hot encoded matrix back to a peptide sequence.
    Args:
        ohe: One-hot encoded matrix of shape (N, 21).
    Returns:
        sequence: Peptide sequence as a string.
    """
    seq = []
    for j in range(ohe.shape[0]):  # Iterate over sequence length
        if gap and np.all(ohe[j] == 0):
            seq.append('-')
        else:
            aa_index = np.argmax(ohe[j])  # Get index of the max value in one-hot encoding
            seq.append(AA[aa_index])
    return ''.join(seq)  # Join the list into a string


def seq_to_aa_encodings(sequence: str, max_seq_len: int) -> np.ndarray:
    """
    Converts a peptide sequence into a matrix using AA_ENCODINGS substitution scores.
    This function maps each amino acid in the input sequence to its corresponding
    vector of substitution scores from the AA_ENCODINGS matrix. The resulting matrix
    is padded or truncated to a specified maximum length.
    Args:
        sequence (str): The input peptide sequence.
        max_seq_len (int): The target length for the output matrix. Sequences
                           shorter than this will be padded with zeros, and
                           longer ones will be truncated.
    Returns:
        np.ndarray: A NumPy array of shape (max_seq_len, 23) where each row
                    corresponds to an amino acid's AA_ENCODINGS scores.
    """
    # The AA_ENCODINGS matrix has 23 columns corresponding to the score list length.
    num_features = len(next(iter(AA_ENCODINGS.values())))
    # Initialize the output array with the padding value (0.0).
    arr = np.full((max_seq_len, num_features), PAD_VALUE, dtype=np.float32)
    # Use the vector for '-' (unknown) as the default for any character
    # not found in the AA_ENCODINGS dictionary, including gaps ('-').
    default_vector = AA_ENCODINGS['X']
    # Iterate over the sequence up to the maximum length.
    for i, aa in enumerate(sequence.upper()[:max_seq_len]):
        # Retrieve the AA_ENCODINGS vector for the current amino acid.
        # If the amino acid is not a key in the dictionary, use the default vector.
        blosum_vector = AA_ENCODINGS.get(aa, default_vector)
        arr[i, :] = blosum_vector
    return arr


def AA_ENCODINGS_to_seq(blosum_matrix: np.ndarray) -> str:
    """
    Converts a AA_ENCODINGS matrix back into a peptide sequence.
    This function iterates through each row of the input matrix, finds the
    corresponding amino acid from the AA_ENCODINGS mapping, and reconstructs the
    sequence. It stops when it encounters a padding row (all zeros).
    Args:
        blosum_matrix (np.ndarray): A NumPy array of shape (N, 23) containing
                                    AA_ENCODINGS scores.
    Returns:
        str: The reconstructed peptide sequence.
    """
    sequence = []
    # Iterate through each row (vector) in the input matrix.
    for row in blosum_matrix:
        # Check if the row is a padding vector (all elements are PAD_VALUE).
        # If so, we have reached the end of the sequence.
        if not np.any(row):
            break
        # Convert the NumPy array row to a tuple to make it hashable.
        row_tuple = tuple(np.array(row))
        # Look up the amino acid in the reverse map. Default to '-' if not found.
        amino_acid = AMINO_ACID_MAP.get(row_tuple, '-')
        sequence.append(amino_acid)
    return "".join(sequence)


def seq_to_indices_AA_ENCODINGS(seq, max_seq_len):
    """Correctly converts an amino acid sequence to an array of integer indices."""
    indices = np.full(max_seq_len, PAD_INDEX_62, dtype=np.int64)
    for i, aa in enumerate(seq.upper()[:max_seq_len]):
        # Use the correct mapping: character -> index
        indices[i] = AA_TO_BLOSUM_IDX.get(aa, PAD_INDEX_62)
    return indices


def seq_to_ohe_indices(seq, max_seq_len):
    """Converts an amino acid sequence to an array of integer indices FOR OHE TARGET."""
    indices = np.full(max_seq_len, PAD_INDEX_OHE, dtype=np.int64)
    for i, aa in enumerate(seq.upper()[:max_seq_len]):
        indices[i] = AA_TO_INT.get(aa, PAD_INDEX_OHE)
    return indices


def get_embed_key(key: str, emb_dict: Dict[str, str]) -> str:  # why ndarray?
    """
    Get the embedding key for a given allele key.
    If the key is not found in the embedding dictionary, return None.
    # find the matching emb key in the emb_dict.
    Sometimes the emb key is longer than the allele key, so we need to check if the key is a substring of the emb key.
    """
    # Use a generator expression for efficient lookup
    return next((emb_key for emb_key in emb_dict if emb_key.upper().startswith(key.upper())), None)


def get_seq(key: str, seq_dict: Dict[str, str]) -> str:
    """
    Get the sequence for a given allele key.
    If the key is not found in the sequence dictionary, return None.
    """
    return next((seq for seq_key, seq in seq_dict.items() if seq_key.upper().startswith(key.upper())), None)


def clean_key(allele_key: str) -> str:
    """
    Clean allele keys by removing special characters and converting to uppercase.
    This is useful for matching keys in embedding dictionaries.
    """
    if allele_key is None:
        return "None"
    mapping = str.maketrans({'*': '', ':': '', ' ': '', '/': '_'})
    return allele_key.translate(mapping).upper()


def get_mhc_seq_class2(key, embed_map, seq_map):
    # print(f"Processing key: {key}")  # Debugging line
    if key is None: return ''
    key_parts = key.split('_')
    # print(f"Key parts: {key_parts}")  # Debugging line
    if len(key_parts) >= 2:
        key1 = get_embed_key(key_parts[0], embed_map)
        key2 = get_embed_key(key_parts[1], embed_map)
        if embed_map.get(key1, None) is None or embed_map.get(key2, None) is None:
            print(
                f"Warning: Embedding not found for embd_key 1: '{key1}' 2: '{key2}' in input:'{key_parts[0]}', '{key_parts[1]}'")
        # print(f"Key1: {key1}, Key2: {key2}")  # Debugging line
        seq1 = get_seq(key_parts[0], seq_map) if key1 else ''
        seq2 = get_seq(key_parts[1], seq_map) if key2 else ''
        # print(f"Seq1: {seq1}, Seq2: {seq2}")  # Debugging line
        return seq1 + seq2
    else:
        raise ValueError(f"Unexpected MHC class II key format: '{key}'")


def get_embed_key_class2(key, embed_map):
    if key is None: return None
    key_parts = key.split('_')
    if len(key_parts) >= 2:
        key1 = get_embed_key(key_parts[0], embed_map)
        key2 = get_embed_key(key_parts[1], embed_map)
        if embed_map.get(key1, None) is None or embed_map.get(key2, None) is None:
            print(
                f"Warning: Embedding not found for embd_key 1: '{key1}' 2: '{key2}' in input:'{key_parts[0]}', '{key_parts[1]}'")
        return "_".join(filter(None, [key1, key2]))
    else:
        raise ValueError(f"Unexpected MHC class II key format: '{key}'")


### TensorFlow Layers and Functions
@tf.function
def _neg_inf(dtype: tf.dtypes.DType) -> tf.Tensor:
    """Return a large negative constant suited for masking in given dtype.
    required for mixed precision training."""
    if dtype == tf.float16 or dtype == tf.bfloat16:
        return tf.constant(-1e4, dtype=dtype)
    return tf.constant(-1e9, dtype=dtype)


@tf.keras.utils.register_keras_serializable(package='Custom', name='AttentionLayer')
class AttentionLayer(keras.layers.Layer):
    """
    Custom multi-head attention layer supporting self- and cross-attention.

    Args:
        query_dim (int): Input feature dimension for query.
        context_dim (int): Input feature dimension for context (key and value).
        output_dim (int): Output feature dimension.
        type (str): 'self' or 'cross'.
        heads (int): Number of attention heads.
        resnet (bool): Whether to use residual connection.
        return_att_weights (bool): Whether to return attention weights.
        name (str): Layer name.
        epsilon (float): Epsilon for layer normalization.
        gate (bool): Whether to use gating mechanism.
        mask_token (float): Value for masked tokens.
        pad_token (float): Value for padded tokens.
    """

    def __init__(self, query_dim, context_dim, output_dim, type, heads=4,
                 resnet=True, return_att_weights=False, name='attention',
                 epsilon=1e-6, gate=True, mask_token=-1., pad_token=-2., apply_rope=False, **kwargs):
        super().__init__(name=name)
        assert isinstance(query_dim, int) and isinstance(context_dim, int) and isinstance(output_dim, int)
        assert type in ['self', 'cross']
        if resnet:
            assert query_dim == output_dim
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.output_dim = output_dim
        self.type = type
        self.heads = heads
        self.resnet = resnet
        self.return_att_weights = return_att_weights
        self.epsilon = epsilon
        self.gate = gate
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.att_dim = output_dim  # Attention dimension per head
        self.apply_rope = apply_rope

    def build(self, input_shape):
        # Projection weights (will be float32 variables under mixed policy)
        self.q_proj = self.add_weight(shape=(self.heads, self.query_dim, self.att_dim),
                                      initializer='random_normal', trainable=True, name=f'q_proj_{self.name}')
        self.k_proj = self.add_weight(shape=(self.heads, self.context_dim, self.att_dim),
                                      initializer='random_normal', trainable=True, name=f'k_proj_{self.name}')
        self.v_proj = self.add_weight(shape=(self.heads, self.context_dim, self.att_dim),
                                      initializer='random_normal', trainable=True, name=f'v_proj_{self.name}')
        if self.gate:
            self.g = self.add_weight(shape=(self.heads, self.query_dim, self.att_dim),
                                     initializer='random_uniform', trainable=True, name=f'gate_{self.name}')
        self.norm = layers.LayerNormalization(epsilon=self.epsilon, name=f'ln_{self.name}')
        if self.type == 'cross':
            self.norm_context = layers.LayerNormalization(epsilon=self.epsilon, name=f'ln_context_{self.name}')
        self.norm_out = layers.LayerNormalization(epsilon=self.epsilon, name=f'ln_out_{self.name}')
        self.out_w = self.add_weight(shape=(self.heads * self.att_dim, self.output_dim),
                                     initializer='random_normal', trainable=True, name=f'outw_{self.name}')
        self.out_b = self.add_weight(shape=(self.output_dim,), initializer='zeros',
                                     trainable=True, name=f'outb_{self.name}')
        # keep scale in compute dtype
        self.scale = 1.0 / tf.math.sqrt(tf.cast(self.att_dim, tf.keras.mixed_precision.global_policy().compute_dtype))

        if self.apply_rope:
            if (self.att_dim % 2) != 0:
                raise ValueError(f"RotaryEmbedding requires even att_dim, got {self.att_dim}.")
            # q/k have shape (B, H, S, D): sequence_axis=2, feature_axis=-1
            self.rope = RotaryEmbedding(sequence_axis=2, feature_axis=-1, name=f'rope_{self.name}')
        super().build(input_shape)

    @tf.function
    def call(self, x, mask, context=None, context_mask=None):
        """
        Args:
            x: Tensor of shape (B, N, query_dim) for query.
            mask: Tensor of shape (B, N).
            context: Tensor of shape (B, M, context_dim) for key/value in cross-attention.
            context_mask: Tensor of shape (B, M) for context.
        """
        mask = tf.cast(mask, self.compute_dtype)
        if self.type == 'self':
            context = x
            context_mask = mask
            q_input = k_input = v_input = self.norm(x)
            mask_q = mask_k = tf.where(mask == self.pad_token, 0., 1.)
        else:
            assert context is not None and context_mask is not None
            q_input = self.norm(x)
            k_input = v_input = self.norm_context(context)
            mask_q = tf.where(mask == self.pad_token, 0., 1.)
            mask_k = tf.where(tf.cast(context_mask, self.compute_dtype) == self.pad_token, 0., 1.)

        # Project query, key, value
        q = tf.einsum('bnd,hde->bhne', tf.cast(q_input, self.compute_dtype), tf.cast(self.q_proj, self.compute_dtype))
        k = tf.einsum('bmd,hde->bhme', tf.cast(k_input, self.compute_dtype), tf.cast(self.k_proj, self.compute_dtype))
        v = tf.einsum('bmd,hde->bhme', tf.cast(v_input, self.compute_dtype), tf.cast(self.v_proj, self.compute_dtype))

        # Apply RoPE if enabled
        if self.apply_rope:
            q = self.rope(q)
            k = self.rope(k)

        # Compute attention scores
        att = tf.einsum('bhne,bhme->bhnm', q, k) * tf.cast(self.scale, self.compute_dtype)
        mask_q_exp = tf.expand_dims(mask_q, axis=1)
        mask_k_exp = tf.expand_dims(mask_k, axis=1)
        attention_mask = tf.einsum('bqn,bkm->bqnm', mask_q_exp, mask_k_exp)
        attention_mask = tf.broadcast_to(attention_mask, tf.shape(att))
        att += (1.0 - attention_mask) * _neg_inf(att.dtype)
        att = tf.nn.softmax(att, axis=-1) * attention_mask

        # Compute output
        out = tf.einsum('bhnm,bhme->bhne', att, v)
        if self.gate:
            g = tf.einsum('bnd,hde->bhne', tf.cast(q_input, self.compute_dtype), tf.cast(self.g, self.compute_dtype))
            g = tf.nn.sigmoid(g)
            out *= g

        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [tf.shape(x)[0], tf.shape(x)[1], self.heads * self.att_dim])
        out = tf.matmul(out, tf.cast(self.out_w, self.compute_dtype)) + tf.cast(self.out_b, self.compute_dtype)

        if self.resnet:
            out += tf.cast(x, self.compute_dtype)

        out = self.norm_out(out)
        mask_exp = tf.expand_dims(mask_q, axis=-1)
        out *= mask_exp

        return (out, att) if self.return_att_weights else out


@tf.keras.utils.register_keras_serializable(package='custom_layers', name='MaskedEmbedding')
class MaskedEmbedding(keras.layers.Layer):
    def __init__(self, mask_token=-1., pad_token=-2., name='masked_embedding', **kwargs):
        super().__init__(name=name)
        self.mask_token = mask_token
        self.pad_token = pad_token

    @tf.function(reduce_retracing=True)
    def call(self, x, mask):
        """
        Args:
            x: Input tensor of shape (B, N, D)
            mask: Tensor of shape (B, N)
        Returns:
            Tensor with masked positions set to zero.
        """
        mask = tf.cast(mask, tf.float32)
        mask = tf.where((mask == self.pad_token) | (mask == self.mask_token), 0., 1.)
        mask = tf.cast(mask, x.dtype)
        return x * mask[:, :, tf.newaxis]  # Apply mask to zero out positions

    def get_config(self):
        """Serializes the layer's configuration."""
        config = super().get_config()
        config.update({
            'mask_token': self.mask_token,
            'pad_token': self.pad_token,
        })
        return config


@tf.keras.utils.register_keras_serializable(package='custom_layers', name='PositionalEncoding')
class PositionalEncoding(keras.layers.Layer):
    """
    Sinusoidal Positional Encoding layer that applies encodings
    only to non-masked tokens.
    Args:
        embed_dim (int): Dimension of embeddings (must match input last dim).
        max_len (int): Maximum sequence length expected (used to precompute encodings).
    """

    def __init__(self, embed_dim, pos_range=100, mask_token=-1., pad_token=-2., name='positional_encoding', **kwargs):
        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.pos_range = pos_range
        self.mask_token = mask_token
        self.pad_token = pad_token

    def build(self, x):
        # Create (1, pos_range, embed_dim) encoding matrix
        pos = tf.range(self.pos_range, dtype=tf.float32)[:, tf.newaxis]  # (pos_range, 1)
        i = tf.range(self.embed_dim, dtype=tf.float32)[tf.newaxis, :]  # (1, embed_dim)
        # angle_rates = 1 / tf.pow(300.0, (2 * (i // 2)) / tf.cast(self.embed_dim, GLOBAL_DTYPE))
        angle_rates = tf.pow(300.0, -(2.0 * tf.floor(i // 2)) / tf.cast(self.embed_dim, tf.float32))
        angle_rads = pos * angle_rates  # (pos_range, embed_dim)
        # Apply sin to even indices, cos to odd indices
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)  # (max_len, embed_dim)
        pos_encoding = pos_encoding[tf.newaxis, ...]  # (1, max_len, embed_dim)
        # store in compute dtype to reduce casts
        self.pos_encoding = tf.cast(pos_encoding, dtype=self.compute_dtype)

    def call(self, x, mask):
        """
        Args:
            x: Input tensor of shape (B, N, D)
            mask: Tensor of shape (B,N)
        Returns:
            Tensor with positional encodings added for masked and non padded tokens.
        """
        seq_len = tf.shape(x)[1]
        pe = self.pos_encoding[:, :seq_len, :]  # (1, N, D)
        mask = tf.cast(mask[:, :, tf.newaxis], x.dtype)  # (B, N, 1)
        mask = tf.where(mask == self.pad_token, tf.cast(0.0, x.dtype), tf.cast(1.0, x.dtype))
        pe = tf.cast(pe, x.dtype) * mask  # zero out positions where mask is 0
        return x + pe


@tf.function
def select_indices(ind, n, m_range):
    """
    Select top-n indices from `ind` (descending sorted) such that:
    - First index is always selected.
    - Each subsequent index has a distance from all previously selected
      indices between m_range[0] and m_range[1], inclusive.
    Args:
        ind: Tensor of shape (B, N) with descending sorted indices.
        n: Number of indices to select.
        m_range: List or tuple [min_distance, max_distance]
    Returns:
        Tensor of shape (B, n) with selected indices per batch.
    """
    m_min = tf.constant(m_range[0], dtype=tf.int32)
    m_max = tf.constant(m_range[1], dtype=tf.int32)

    def per_batch_select(indices):
        top = indices[0]
        selected = tf.TensorArray(dtype=tf.int32, size=n)
        selected = selected.write(0, top)
        count = tf.constant(1)
        i = tf.constant(1)

        def cond(i, count, selected):
            return tf.logical_and(i < tf.shape(indices)[0], count < n)

        def body(i, count, selected):
            candidate = indices[i]
            selected_vals = selected.stack()[:count]
            distances = tf.abs(selected_vals - candidate)
            if_valid = tf.reduce_all(
                tf.logical_and(distances >= m_min, distances <= m_max)
            )
            selected = tf.cond(if_valid,
                               lambda: selected.write(count, candidate),
                               lambda: selected)
            count = tf.cond(if_valid, lambda: count + 1, lambda: count)
            return i + 1, count, selected

        _, _, selected = tf.while_loop(
            cond, body, [i, count, selected],
            shape_invariants=[i.get_shape(), count.get_shape(), tf.TensorShape(None)]
        )
        return selected.stack()

    return tf.map_fn(per_batch_select, ind, dtype=tf.int32)


@tf.keras.utils.register_keras_serializable(package='custom_layers', name='AnchorPositionExtractor')
class AnchorPositionExtractor(keras.layers.Layer):
    def __init__(self, num_anchors, dist_thr, initial_temperature=1.0, name='anchor_extractor', project=True,
                 mask_token=-1., pad_token=-2., return_att_weights=False, **kwargs):
        super().__init__()
        assert isinstance(dist_thr, list) and len(dist_thr) == 2
        assert num_anchors > 0
        self.num_anchors = num_anchors
        self.dist_thr = dist_thr
        self.name = name
        self.initial_temperature = initial_temperature
        self.project = project
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.return_att_weights = return_att_weights

    def build(self, input_shape):  # att_out (B,N,E)
        b, n, e = input_shape[0], input_shape[1], input_shape[2]
        self.barcode = tf.random.uniform(shape=(1, 1, e))  # add as a token to input
        self.q = self.add_weight(shape=(e, e),
                                 initializer='orthogonal',
                                 trainable=True, name=f'query_{self.name}')
        self.k = self.add_weight(shape=(e, e),
                                 initializer='orthogonal',
                                 trainable=True, name=f'key_{self.name}')
        self.v = self.add_weight(shape=(e, e),
                                 initializer='orthogonal',
                                 trainable=True, name=f'value_{self.name}')
        self.ln = layers.LayerNormalization(name=f'ln_{self.name}')
        if self.project:
            self.g = self.add_weight(shape=(self.num_anchors, e, e),
                                     initializer='random_uniform',
                                     trainable=True, name=f'gate_{self.name}')
            self.w = self.add_weight(shape=(1, self.num_anchors, e, e),
                                     initializer='random_normal',
                                     trainable=True, name=f'w_{self.name}')
        self.log_temperature = self.add_weight(
            shape=(),  # Scalar variable
            initializer=tf.keras.initializers.Constant(tf.math.log(self.initial_temperature)),
            trainable=True,
            name='log_temperature'
        )

    @tf.function(reduce_retracing=True)
    def call(self, input, mask):  # (B,N,E) this is peptide embedding and (B,N) for mask

        mask = tf.cast(mask, tf.float32)  # (B, N)
        mask = tf.where(mask == self.pad_token, 0., 1.)

        barcode = self.barcode
        barcode = tf.broadcast_to(barcode, (tf.shape(input)[0], 1, tf.shape(input)[-1]))  # (B,1,E)
        q = tf.matmul(barcode, self.q)  # (B,1,E)*(E,E)->(B,1,E)
        k = tf.matmul(input, self.k)  # (B,N,E)*(E,E)->(B,N,E)
        v = tf.matmul(input, self.v)  # (B,N,E)*(E,E)->(B,N,E)
        scale = 1 / tf.math.sqrt(tf.cast(tf.shape(input)[-1], tf.float32))
        barcode_att = tf.matmul(q, k, transpose_b=True) * scale  # (B,1,E)*(B,E,N)->(B,1,N)
        # mask: (B,N) => (B,1,N)
        mask_exp = tf.expand_dims(mask, axis=1)
        additive_mask = (1.0 - mask_exp) * _neg_inf(barcode_att.dtype)
        barcode_att += additive_mask
        temperature = tf.exp(self.log_temperature)
        barcode_att = tf.nn.softmax(barcode_att / temperature)
        barcode_att *= mask_exp  # to remove the impact of row wise attention of padded tokens. since all are 1e-9
        barcode_out = tf.matmul(barcode_att, v)  # (B,1,N)*(B,N,E)->(B,1,E)
        # barcode_out represents a vector for all information from peptide
        # barcode_att represents the anchor positions which are the tokens with highest weights
        inds, weights, outs = self.find_anchor(input,
                                               barcode_att)  # (B,num_anchors) (B,num_anchors) (B, num_anchors, E)
        if self.project:
            pos_encoding = tf.broadcast_to(
                tf.expand_dims(inds, axis=-1),
                (tf.shape(outs)[0], tf.shape(outs)[1], tf.shape(outs)[2])
            )
            pos_encoding = tf.cast(pos_encoding, tf.float32)
            dim = tf.cast(tf.shape(outs)[-1], tf.float32)
            ra = tf.range(dim, dtype=tf.float32) / dim
            pos_encoding = tf.sin(pos_encoding / tf.pow(40., ra))
            outs += pos_encoding

            weights_bc = tf.expand_dims(weights, axis=-1)
            weights_bc = tf.broadcast_to(weights_bc, (tf.shape(weights_bc)[0],
                                                      tf.shape(weights_bc)[1],
                                                      tf.shape(outs)[-1]
                                                      ))  # (B,num_anchors, E)
            outs = tf.expand_dims(outs, axis=-2)  # (B, num_anchors, 1, E)
            outs_w = tf.matmul(outs, self.w)  # (B,num_anchors,1,E)*(1,num_anchors,E,E)->(B,num_anchors,1,E)
            outs_g = tf.nn.sigmoid(tf.matmul(outs, self.g))
            outs_w = tf.squeeze(outs_w, axis=-2)  # (B,num_anchors,E)
            outs_g = tf.squeeze(outs_g, axis=-2)
            # multiply by attention weights from barcode_att to choose best anchors and additional feature gating
            outs = outs_w * outs_g * weights_bc  # (B, num_anchors, E)
        outs = self.ln(outs)
        # outs -> anchor info, inds -> anchor indeces, weights -> anchor att weights, barcode_out -> whole peptide features
        # (B,num_anchors,E), (B,num_anchors), (B,num_anchors), (B,E)
        if self.return_att_weights:
            return outs, inds, weights, tf.squeeze(barcode_out, axis=1), barcode_att
        else:
            return outs, inds, weights, tf.squeeze(barcode_out, axis=1)

    def find_anchor(self, input, barcode_att):  # (B,N,E), (B,1,N)
        inds = tf.argsort(barcode_att, axis=-1, direction='DESCENDING', stable=False)  # (B,1,N)
        inds = tf.squeeze(inds, axis=1)  # (B,N)
        selected_inds = select_indices(inds, n=self.num_anchors, m_range=self.dist_thr)  # (B,num_anchors)
        sorted_selected_inds = tf.sort(selected_inds)
        sorted_selected_weights = tf.gather(tf.squeeze(barcode_att, axis=1),
                                            sorted_selected_inds,
                                            axis=1,
                                            batch_dims=1)  # (B,num_anchors)
        sorted_selected_output = tf.gather(input, sorted_selected_inds, axis=1, batch_dims=1)  # (B,num_anchors,E)
        return sorted_selected_inds, sorted_selected_weights, sorted_selected_output


@tf.keras.utils.register_keras_serializable(package='custom_layers', name='concat_mask')
class ConcatMask(keras.layers.Layer):
    def __init__(self, name='concat_mask', **kwargs):
        super().__init__(name=name)

    def call(self, mask1, mask2):
        """
        Args:
            mask1: Tensor of shape (B, N1)
            mask2: Tensor of shape (B, N2)
        Returns:
            Concatenated mask tensor of shape (B, N1 + N2)
        """
        mask1 = tf.cast(mask1, tf.float32)
        mask2 = tf.cast(mask2, tf.float32)
        return tf.concat([mask1, mask2], axis=1)


@tf.keras.utils.register_keras_serializable(package='custom_layers', name='split_layer')
class SplitLayer(keras.layers.Layer):
    def __init__(self, split_size, name='split_layer', **kwargs):
        """
        :param split_size: a float of ints that sum up your input dimension
        :param name: str
        """
        super().__init__(name=name)
        self.split_size = split_size

    def call(self, x):
        """
        Args:
            x: Input tensor of shape (B, N, D)
        Returns:
            List of tensors split along the second dimension.
        """
        return tf.split(x, num_or_size_splits=self.split_size, axis=1)  # Split along the second dimension


@tf.function(reduce_retracing=True)
def masked_categorical_crossentropy(y_true_and_pred, mask, pad_token=-2.0, sample_weight=None, type='cce'):
    """
    Compute masked categorical cross-entropy loss.

    Args:
        y_true_and_pred: Concatenated tensor of true labels and predictions.
        mask: Mask tensor indicating positions to include in the loss.
        pad_token: Value of the padding token in the mask.
        sample_weight: Optional tensor of shape (B, 1) or (B,) to weight samples.

    Returns:
        Mean masked loss (tensor).
    """
    assert type in ['cce', 'mse']
    y_true, y_pred = tf.split(y_true_and_pred, num_or_size_splits=2, axis=-1)
    # Cast to float32 for stable computation
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    # Clip with wider margin for stability
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    # Build a 0/1 mask for non-pad tokens
    mask = tf.cast(tf.not_equal(mask, pad_token), tf.float32)
    if mask.shape.rank is not None and mask.shape.rank > 2 and mask.shape[-1] == 1:
        mask = tf.squeeze(mask, axis=-1)
    if type == 'cce':
        # Use numerically stable log
        log_pred = tf.math.log(y_pred)
        # Replace any NaN/Inf with zeros
        log_pred = tf.where(tf.math.is_finite(log_pred), log_pred, tf.zeros_like(log_pred))
        loss = -tf.reduce_sum(y_true * log_pred, axis=-1)
    elif type == 'mse':
        loss = tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)
    else:
        raise ValueError(f"Unsupported loss type: {type}")
    # Handle any NaN in loss
    loss = tf.where(tf.math.is_finite(loss), loss, tf.zeros_like(loss))
    mask = tf.cast(tf.broadcast_to(mask, tf.shape(loss)), tf.float32)
    masked_loss = loss * mask
    if sample_weight is not None:
        sample_weight = tf.cast(sample_weight, tf.float32)
        if sample_weight.shape.rank == 2 and sample_weight.shape[1] == 1:
            sample_weight = tf.squeeze(sample_weight, axis=-1)
        masked_loss *= sample_weight[:, tf.newaxis]
        mask *= sample_weight[:, tf.newaxis]
    total_loss = tf.reduce_sum(masked_loss)
    total_weight = tf.reduce_sum(mask)
    ce_loss = tf.math.divide_no_nan(total_loss, total_weight)
    return tf.cast(ce_loss, tf.float32)


@tf.function
def split_y_true_y_pred(y_true_y_pred):
    """
    Split concatenated y_true and y_pred tensor into separate tensors.

    Args:
        y_true_y_pred: Tensor of shape (B, N, 2 * C) where C is the number of classes.

    Returns:
        Tuple of (y_true, y_pred) tensors.
    """
    y_true, y_pred = tf.split(y_true_y_pred, num_or_size_splits=2, axis=-1)
    return y_true, y_pred


import tensorflow as tf
from tensorflow import keras


# Helper function (assuming it's defined elsewhere, adding for completeness)
def _neg_inf(dtype):
    """Returns a value close to -infinity."""
    if dtype == tf.float16:
        return tf.constant(-1e4, dtype=dtype)
    elif dtype == tf.float32:
        return tf.constant(-1e9, dtype=dtype)
    elif dtype == tf.float64:
        return tf.constant(-1e18, dtype=dtype)
    else:
        return tf.constant(-1e9, dtype=dtype)


@tf.keras.utils.register_keras_serializable(package='custom_layers', name='rms_norm')
class rms_norm(keras.layers.Layer):
    """
    Root Mean Square Layer Normalization

    Args:
        dim: dimension size of the layer
        epsilon: epsilon value for numerical stability, default 1e-8
        p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        bias: whether to use bias term for RMSNorm, disabled by default
        name: layer name
    """

    def __init__(self, dim, epsilon=1e-8, p=-1., bias=False, name='rms_norm', **kwargs):
        super().__init__(name=name, **kwargs)
        self.dim = dim
        self.epsilon = epsilon
        self.p = p
        self.use_bias = bias

    def build(self, input_shape):
        # Verify the last dimension matches the specified dim
        layer_size = int(input_shape[-1])
        if layer_size != self.dim:
            raise ValueError(
                f"Input dim {layer_size} must match specified dim {self.dim}.")

        # Create scale parameter (gamma)
        self.scale = self.add_weight(
            shape=(self.dim,),
            initializer='ones',
            trainable=True,
            name='scale'
        )

        # Create bias parameter (beta) if enabled
        if self.use_bias:
            self.offset = self.add_weight(
                shape=(self.dim,),
                initializer='zeros',
                trainable=True,
                name='offset'
            )
        else:
            self.offset = 0.

        super().build(input_shape)

    @tf.function(reduce_retracing=True)
    def call(self, x):
        """
        Apply RMS normalization
        Args:
            x: input tensor with shape [batch, ..., dim]
        Returns:
            normalized tensor with same shape as input
        """
        x = tf.cast(x, self.compute_dtype)

        # Compute mean square
        if self.p < 0. or self.p > 1.:
            # Full RMSNorm
            ms = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
        else:
            # Partial RMSNorm
            partial_size = int(self.dim * self.p)
            partial_x = x[..., :partial_size]
            ms = tf.reduce_mean(tf.square(partial_x), axis=-1, keepdims=True)

        # Normalize: scale * x / sqrt(ms + eps) + offset
        rms = tf.sqrt(ms + self.epsilon)
        normalized = tf.cast(self.scale, self.compute_dtype) * x / rms

        if self.use_bias:
            normalized = normalized + tf.cast(self.offset, self.compute_dtype)

        return normalized

    def get_config(self):
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'epsilon': self.epsilon,
            'p': self.p,
            'bias': self.use_bias,
        })
        return config


@tf.keras.utils.register_keras_serializable(package='custom_layers', name='SelfAttentionWith2DMaskRMSNorm')
class SelfAttentionWith2DMaskRMSNorm(keras.layers.Layer):
    """
    Custom self-attention layer that supports 2D masks.
    """

    def __init__(self, query_dim, context_dim, output_dim, heads=4,
                 return_att_weights=False, name='SelfAttentionWith2DMaskRMSNorm',
                 epsilon=1e-6, mask_token=-1., pad_token=-2., self_attn_mhc=True, apply_rope=True,
                 resnet=False, **kwargs):
        super().__init__(name=name)
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.output_dim = output_dim
        self.heads = heads
        self.return_att_weights = return_att_weights
        self.epsilon = epsilon
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.att_dim = output_dim  # Attention dimension per head
        self.self_attn_mhc = self_attn_mhc
        self.apply_rope = apply_rope  # flag for rotary positional embedding
        self.resnet = resnet

    def build(self, input_shape):
        # Validate input dim matches provided dims (since a single x is used for q/k/v)
        in_dim = int(input_shape[-1])
        if in_dim != self.query_dim or in_dim != self.context_dim:
            raise ValueError(
                f"Input dim {in_dim} must match query_dim {self.query_dim} and context_dim {self.context_dim}.")

        # Projection weights
        self.norm1 = rms_norm(dim=self.query_dim, epsilon=self.epsilon, name=f'rms1_{self.name}')
        self.q_proj = self.add_weight(shape=(self.heads, self.query_dim, self.att_dim),
                                      initializer='random_normal', trainable=True, name=f'q_proj_{self.name}')
        self.k_proj = self.add_weight(shape=(self.heads, self.context_dim, self.att_dim),
                                      initializer='random_normal', trainable=True, name=f'k_proj_{self.name}')
        self.v_proj = self.add_weight(shape=(self.heads, self.context_dim, self.att_dim),
                                      initializer='random_normal', trainable=True, name=f'v_proj_{self.name}')
        self.g_proj = self.add_weight(shape=(self.heads, self.query_dim, self.att_dim),
                                      initializer='random_uniform', trainable=True, name=f'g_proj_{self.name}')
        self.norm2 = rms_norm(dim=self.output_dim, epsilon=self.epsilon, name=f'rms2_{self.name}')

        self.out_w = self.add_weight(shape=(self.heads * self.att_dim, self.output_dim),
                                     initializer='random_normal', trainable=True, name=f'outw_{self.name}')
        self.out_b = self.add_weight(shape=(self.output_dim,), initializer='zeros',
                                     trainable=True, name=f'outb_{self.name}')
        self.scale = 1.0 / tf.math.sqrt(tf.cast(self.att_dim, tf.keras.mixed_precision.global_policy().compute_dtype))

        if self.apply_rope:
            if (self.att_dim % 2) != 0:
                raise ValueError(f"RotaryEmbedding requires even att_dim, got {self.att_dim}.")
            # q/k have shape (B, H, S, D): sequence_axis=2, feature_axis=-1
            self.rope = RotaryEmbedding(sequence_axis=2, feature_axis=-1, name=f'rope_{self.name}')

        super().build(input_shape)

    @tf.function(reduce_retracing=True)
    def call(self, x_pmhc, p_mask, m_mask):
        """
        Args:
            x: Tensor of shape (B, N+M, query_dim) for query.
            p_mask: Tensor of shape (B, N) for peptide mask.
            m_mask: Tensor of shape (B, M) for mhc mask.
        Returns:
            Tensor of shape (B, N+M, output_dim)
        """
        residual = x_pmhc
        x_pmhc = self.norm1(x_pmhc)  # Normalize input
        p_mask = tf.cast(p_mask, self.compute_dtype)
        m_mask = tf.cast(m_mask, self.compute_dtype)
        p_mask = tf.where(p_mask == self.pad_token, x=0., y=1.)  # (B, N)
        m_mask = tf.where(m_mask == self.pad_token, x=0., y=1.)  # (B, M)

        q = tf.einsum('bxd,hde->bhxe', tf.cast(x_pmhc, self.compute_dtype),
                      tf.cast(self.q_proj, self.compute_dtype))
        k = tf.einsum('bxd,hde->bhxe', tf.cast(x_pmhc, self.compute_dtype),
                      tf.cast(self.k_proj, self.compute_dtype))
        v = tf.einsum('bxd,hde->bhxe', tf.cast(x_pmhc, self.compute_dtype),
                      tf.cast(self.v_proj, self.compute_dtype))
        g = tf.einsum('bxd,hde->bhxe', tf.cast(x_pmhc, self.compute_dtype),
                      tf.cast(self.g_proj, self.compute_dtype))

        if self.apply_rope:
            q = self.rope(q)
            k = self.rope(k)

        # Compute attention scores with clamping for numerical stability
        att = tf.einsum('bhxe,bhye->bhxy', q, k) * tf.cast(self.scale, self.compute_dtype)

        # Clamp attention logits to prevent overflow in float16
        max_val = tf.constant(65000.0 if self.compute_dtype == tf.float16 else 1e9, dtype=self.compute_dtype)
        att = tf.clip_by_value(att, -max_val, max_val)

        # Create 2D mask
        mask_2d = self.mask_2d(p_mask, m_mask)
        mask_2d = tf.cast(mask_2d, self.compute_dtype)
        mask_2d_neg = (1.0 - mask_2d) * _neg_inf(att.dtype)

        # Compute softmax in float32 for numerical stability
        att_for_softmax = tf.cast(att + tf.expand_dims(mask_2d_neg, axis=1), tf.float32)
        att = tf.cast(tf.nn.softmax(att_for_softmax, axis=-1), self.compute_dtype)
        att *= tf.expand_dims(mask_2d, axis=1)

        out = tf.matmul(att, v)
        out *= tf.nn.sigmoid(g)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [tf.shape(residual)[0], tf.shape(residual)[1], self.heads * self.att_dim])
        out = tf.matmul(out, tf.cast(self.out_w, self.compute_dtype)) + tf.cast(self.out_b, self.compute_dtype)

        # Add residual connection
        if self.resnet:
            out = out + tf.cast(residual, self.compute_dtype)

        out = self.norm2(out)
        if self.return_att_weights:
            return out, att
        else:
            return out

    def mask_2d(self, p_mask, m_mask):
        p_mask = tf.cast(p_mask, self.compute_dtype)
        m_mask = tf.cast(m_mask, self.compute_dtype)
        p_mask = tf.expand_dims(p_mask, axis=-1)
        m_mask = tf.expand_dims(m_mask, axis=-1)  # (B, N, 1), (B, M, 1)
        # zero square masks
        self_peptide_mask = tf.zeros_like(p_mask, dtype=self.compute_dtype)  # (B, N, 1)
        self_peptide_mask_2d = tf.broadcast_to(self_peptide_mask, (
            tf.shape(p_mask)[0], tf.shape(p_mask)[1], tf.shape(p_mask)[1]))  # (B, N, N)
        if self.self_attn_mhc:
            self_mhc_mask = m_mask
        else:
            self_mhc_mask = tf.zeros_like(m_mask, dtype=self.compute_dtype)
        self_mhc_mask_2d = tf.broadcast_to(self_mhc_mask, (
            tf.shape(m_mask)[0], tf.shape(m_mask)[1], tf.shape(m_mask)[1]))  # (B, M, M)
        # one and zero masks
        pep_mhc_mask_secondpart = tf.broadcast_to(p_mask, (tf.shape(p_mask)[0], tf.shape(p_mask)[1],
                                                           tf.shape(m_mask)[-1]))  # (B, N, M)
        pep_mhc_mask_secondpart = pep_mhc_mask_secondpart * tf.transpose(m_mask,
                                                                         perm=[0, 2, 1])  # (B,N,M)*(B,1,M)=(B, N, M)
        mhc_pep_mask_secondpart = tf.broadcast_to(m_mask, (tf.shape(m_mask)[0], tf.shape(m_mask)[1],
                                                           tf.shape(p_mask)[-1]))  # (B, M, N)
        mhc_pep_mask_secondpart = mhc_pep_mask_secondpart * tf.transpose(p_mask,
                                                                         perm=[0, 2, 1])  # (B,M,N)*(B,1,N)=(B, M, N)
        # combined masks (B,N+M,N+M)
        combined_mask_1 = tf.concat([self_peptide_mask_2d, pep_mhc_mask_secondpart], axis=2)  # (B, N, N+M)
        combined_mask_2 = tf.concat([mhc_pep_mask_secondpart, self_mhc_mask_2d], axis=2)  # (B, M, N+M)
        final_mask = tf.concat([combined_mask_1, combined_mask_2], axis=1)  # (B, N+M, N+M)
        final_mask_t = tf.transpose(final_mask, perm=[0, 2, 1])  # (B, N+M, N+M)
        final_mask = tf.multiply(final_mask, final_mask_t)
        return final_mask


@tf.keras.utils.register_keras_serializable(package='Custom', name='SubtractLayer')
class SubtractLayer(keras.layers.Layer):
    """
    Custom layer to subtract a tensor from another tensor.
    Tensor1: (B, P, D) -> (B, P*D) -> (B, M, P*D)
    Tensor2: (B, M, D) -> (B, M, P*D)
    Output: = Tensor2 - Tensor1
    """

    def __init__(self, mask_token=-1., pad_token=-2., **kwargs):
        """Initialize the layer."""
        super(SubtractLayer, self).__init__(**kwargs)
        self.mask_token = mask_token
        self.pad_token = pad_token

    @tf.function(reduce_retracing=True)
    def call(self, peptide, pep_mask, mhc, mhc_mask):
        B = tf.shape(peptide)[0]
        P = tf.shape(peptide)[1]
        D = tf.shape(peptide)[2]
        M = tf.shape(mhc)[1]
        P_D = P * D

        pep_mask = tf.cast(pep_mask, peptide.dtype)
        mhc_mask = tf.cast(mhc_mask, mhc.dtype)

        pep_mask = tf.where(pep_mask == self.pad_token, x=0., y=1.)  # (B,P)
        mhc_mask = tf.where(mhc_mask == self.pad_token, x=0., y=1.)

        # More efficient approach using broadcasting
        # peptide: (B,P,D) -> (B,1,P,D) -> (B,M,P,D) via broadcasting
        # mhc:     (B,M,D) -> (B,M,1,D) -> (B,M,P,D) via broadcasting
        peptide_expanded = peptide[:, tf.newaxis, :, :]  # (B,1,P,D)
        mhc_expanded = mhc[:, :, tf.newaxis, :]  # (B,M,1,D)
        result_4d = mhc_expanded - peptide_expanded  # (B,M,P,D) via broadcasting
        # Flatten to (B,M,P*D)
        result = tf.reshape(result_4d, (B, M, P_D))
        # Optimize masking operations using broadcasting
        # peptide mask: (B,P) -> (B,1,P,1) -> (B,M,P,D) via broadcasting
        pep_mask_4d = pep_mask[:, tf.newaxis, :, tf.newaxis]  # (B,1,P,1)
        pep_mask_expanded = tf.broadcast_to(pep_mask_4d, (B, M, P, D))  # (B,M,P,D)
        pep_mask_PD = tf.reshape(pep_mask_expanded, (B, M, P_D))  # (B,M,P*D)

        # mhc mask: (B,M) -> (B,M,1,1) -> (B,M,P,D) via broadcasting
        mhc_mask_4d = mhc_mask[:, :, tf.newaxis, tf.newaxis]  # (B,M,1,1)
        mhc_mask_expanded = tf.broadcast_to(mhc_mask_4d, (B, M, P, D))  # (B,M,P,D)
        mhc_mask_PD = tf.reshape(mhc_mask_expanded, (B, M, P_D))  # (B,M,P*D)
        combined_mask = tf.logical_and(tf.cast(pep_mask_PD, tf.bool), tf.cast(mhc_mask_PD, tf.bool))
        masked_result = tf.where(combined_mask, result, tf.zeros_like(result))
        return masked_result


@tf.keras.utils.register_keras_serializable(package='Custom', name='GlobalMeanPooling1D')
class GlobalMeanPooling1D(layers.Layer):
    """Global mean pooling layer."""

    def __init__(self, axis=-1, name="global_mean_pooling_", **kwargs):
        super(GlobalMeanPooling1D, self).__init__(name=name)
        self.axis = axis

    def call(self, input_tensor):
        return tf.math.reduce_mean(input_tensor, axis=self.axis, keepdims=False)


@tf.keras.utils.register_keras_serializable(package='Custom', name='GlobalSTDPooling1D')
class GlobalSTDPooling1D(layers.Layer):
    """Global Standard Deviation Pooling layer."""

    def __init__(self, axis=1, name='global_std_pooling', **kwargs):
        super(GlobalSTDPooling1D, self).__init__(name=name)
        self.axis = axis

    def call(self, input_tensor):
        pooled_std = tf.math.reduce_std(input_tensor, axis=self.axis, keepdims=False, name=None)
        return pooled_std + 1e-9


@tf.keras.utils.register_keras_serializable(package='Custom', name='GlobalMaxPooling1D')
class GlobalMaxPooling1D(layers.Layer):
    """Global max pooling layer."""

    def __init__(self, axis=-1, name="global_max_pooling_", **kwargs):
        super(GlobalMaxPooling1D, self).__init__(name=name)
        self.axis = axis

    def call(self, input_tensor):
        return tf.math.reduce_max(input_tensor, axis=self.axis, keepdims=False)


@tf.keras.utils.register_keras_serializable(package='Custom', name='BinaryMCC')
class BinaryMCC(tf.keras.metrics.Metric):
    """
    Matthews Correlation Coefficient for binary classification.

    Simple implementation using the direct formula:
    MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

    Range: -1 (worst) to +1 (perfect), 0 = random
    """

    def __init__(self, name='binary_mcc', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.tn = self.add_weight(name='tn', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    @tf.function(reduce_retracing=True)
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert to binary
        y_pred_binary = tf.cast(y_pred >= self.threshold, tf.float32)
        y_true_binary = tf.cast(y_true, tf.float32)

        # Calculate TP, TN, FP, FN
        tp = tf.reduce_sum(y_true_binary * y_pred_binary)
        tn = tf.reduce_sum((1 - y_true_binary) * (1 - y_pred_binary))
        fp = tf.reduce_sum((1 - y_true_binary) * y_pred_binary)
        fn = tf.reduce_sum(y_true_binary * (1 - y_pred_binary))

        # Update running totals
        self.tp.assign_add(tp)
        self.tn.assign_add(tn)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    @tf.function(reduce_retracing=True)
    def result(self):
        # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
        numerator = self.tp * self.tn - self.fp * self.fn
        denominator = tf.sqrt(
            (self.tp + self.fp) * (self.tp + self.fn) *
            (self.tn + self.fp) * (self.tn + self.fn)
        )

        # Handle division by zero
        return tf.where(
            tf.equal(denominator, 0.0),
            tf.constant(0.0, dtype=self.dtype),
            numerator / denominator
        )

    def reset_state(self):
        self.tp.assign(0.0)
        self.tn.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)


class BinarySpecificity(tf.keras.metrics.Metric):
    """Computes specificity (true negative rate) for binary labels."""

    def __init__(self, threshold=0.5, name="specificity", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_negatives = self.add_weight(name="tn", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), self.dtype)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), self.dtype)
        y_pred = tf.cast(y_pred >= self.threshold, self.dtype)

        tn_values = (1.0 - y_true) * (1.0 - y_pred)
        fp_values = (1.0 - y_true) * y_pred

        if sample_weight is not None:
            sample_weight = tf.cast(tf.reshape(sample_weight, [-1]), self.dtype)
            tn_values *= sample_weight
            fp_values *= sample_weight

        self.true_negatives.assign_add(tf.reduce_sum(tn_values))
        self.false_positives.assign_add(tf.reduce_sum(fp_values))

    def result(self):
        denom = self.true_negatives + self.false_positives
        return tf.where(denom > 0.0, self.true_negatives / denom, tf.zeros_like(denom))

    def reset_state(self):
        self.true_negatives.assign(0.0)
        self.false_positives.assign(0.0)


@tf.keras.utils.register_keras_serializable(package='custom_layers', name='SwiGLU')
class SwiGLU(tf.keras.layers.Layer):
    """
    SwiGLU activation function (Swish-Gated Linear Unit).
    Used in modern transformer architectures like PaLM and LLaMA.

    SwiGLU(x) = Swish(W1·x) ⊙ (W2·x)
    where Swish(x) = x·σ(x) and ⊙ is element-wise multiplication.

    Args:
        dimension: Hidden dimension size for the gating mechanism
        name: Layer name
    """

    def __init__(self, dimension, name='swiglu', **kwargs):
        super().__init__(name=name, **kwargs)
        self.dimension = dimension

    def build(self, input_shape):
        self.linear_1 = tf.keras.layers.Dense(
            self.dimension,
            use_bias=True,
            name=f'{self.name}_linear1'
        )
        self.linear_2 = tf.keras.layers.Dense(
            self.dimension,
            use_bias=True,
            name=f'{self.name}_linear2'
        )
        super().build(input_shape)

    @tf.function(reduce_retracing=True)
    def call(self, x):
        # Cast to compute dtype for mixed precision training
        x = tf.cast(x, self.compute_dtype)

        output = self.linear_1(x)
        output = tf.cast(output, self.compute_dtype)
        swish = output * tf.nn.sigmoid(output)
        swish = tf.cast(swish, self.compute_dtype)
        swiglu = swish * self.linear_2(x)
        swiglu = tf.cast(swiglu, self.compute_dtype)

        return swiglu

    def get_config(self):
        config = super().get_config()
        config.update({'dimension': self.dimension})
        return config


@tf.keras.utils.register_keras_serializable(package='custom_layers', name='SelfAttentionWith2DMaskV2')
class SelfAttentionWith2DMaskV2(keras.layers.Layer):
    """
    Modern self-attention layer with 2D masks
    - RMSNorm instead of LayerNorm
    - Grouped Query Attention (GQA) for efficiency and regularization
    - QK Normalization
    - Residual connections
    """

    def __init__(self, query_dim, context_dim, output_dim, heads=8,
                 num_key_value_heads=None,
                 return_att_weights=False, name='SelfAttentionWith2DMaskV2',
                 epsilon=1e-6, mask_token=-1., pad_token=-2., self_attn_mhc=True,
                 apply_rope=True, attention_dropout=0.0,
                 **kwargs):
        super().__init__(name=name)
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.output_dim = output_dim
        self.heads = heads

        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else heads
        if self.heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"heads ({self.heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})")
        self.num_key_value_groups = self.heads // self.num_key_value_heads

        self.return_att_weights = return_att_weights
        self.epsilon = epsilon
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.att_dim = output_dim // heads
        self.self_attn_mhc = self_attn_mhc
        self.apply_rope = apply_rope
        self.attention_dropout_rate = attention_dropout
        self.ffn_dropout_rate = ffn_dropout
        self.ffn_hidden_dim = output_dim

        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.moe_aux_loss_weight = moe_aux_loss_weight

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        if in_dim != self.query_dim or in_dim != self.context_dim:
            raise ValueError(
                f"Input dim {in_dim} must match query_dim {self.query_dim} and context_dim {self.context_dim}.")

        # === Attention Block ===
        self.attn_norm = rms_norm(dim=self.query_dim, epsilon=self.epsilon, name=f'attn_rms_norm_{self.name}')

        self.q_proj = self.add_weight(
            shape=(self.query_dim, self.heads * self.att_dim),
            initializer='glorot_uniform',
            trainable=True,
            name=f'q_proj_{self.name}'
        )
        self.k_proj = self.add_weight(
            shape=(self.context_dim, self.num_key_value_heads * self.att_dim),
            initializer='glorot_uniform',
            trainable=True,
            name=f'k_proj_{self.name}'
        )
        self.v_proj = self.add_weight(
            shape=(self.context_dim, self.num_key_value_heads * self.att_dim),
            initializer='glorot_uniform',
            trainable=True,
            name=f'v_proj_{self.name}'
        )

        self.q_norm = rms_norm(dim=self.att_dim, epsilon=self.epsilon, name=f'q_norm_{self.name}')
        self.k_norm = rms_norm(dim=self.att_dim, epsilon=self.epsilon, name=f'k_norm_{self.name}')

        self.out_proj = self.add_weight(
            shape=(self.heads * self.att_dim, self.output_dim),
            initializer='glorot_uniform',
            trainable=True,
            name=f'out_proj_{self.name}'
        )
        self.out_bias = self.add_weight(
            shape=(self.output_dim,),
            initializer='zeros',
            trainable=True,
            name=f'out_bias_{self.name}'
        )

        self.attention_dropout = layers.Dropout(
            rate=self.attention_dropout_rate,
            name=f'att_dropout_{self.name}'
        )

        self.scale = 1.0 / tf.math.sqrt(tf.cast(self.att_dim, tf.keras.mixed_precision.global_policy().compute_dtype))

        if self.apply_rope:
            if (self.att_dim % 2) != 0:
                raise ValueError(f"RotaryEmbedding requires even att_dim, got {self.att_dim}.")
            self.rope = RotaryEmbedding(sequence_axis=2, feature_axis=-1, name=f'rope_{self.name}')

        super().build(input_shape)

    @tf.function(reduce_retracing=True)
    def call(self, x_pmhc, p_mask, m_mask, training=None):
        residual = tf.cast(x_pmhc, self.compute_dtype)

        # ========== ATTENTION BLOCK ==========
        x = self.attn_norm(x_pmhc)
        x = tf.cast(x, self.compute_dtype)

        q = tf.matmul(x, tf.cast(self.q_proj, self.compute_dtype))
        k = tf.matmul(x, tf.cast(self.k_proj, self.compute_dtype))
        v = tf.matmul(x, tf.cast(self.v_proj, self.compute_dtype))

        batch_size = tf.shape(x_pmhc)[0]
        seq_len = tf.shape(x_pmhc)[1]

        q = tf.reshape(q, [batch_size, seq_len, self.heads, self.att_dim])
        q = tf.transpose(q, [0, 2, 1, 3])

        k = tf.reshape(k, [batch_size, seq_len, self.num_key_value_heads, self.att_dim])
        v = tf.reshape(v, [batch_size, seq_len, self.num_key_value_heads, self.att_dim])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = tf.cast(q, self.compute_dtype)
        k = tf.cast(k, self.compute_dtype)

        if self.apply_rope:
            q = self.rope(q)
            k = self.rope(k)

        if self.num_key_value_groups > 1:
            k = tf.repeat(k, repeats=self.num_key_value_groups, axis=1)
            v = tf.repeat(v, repeats=self.num_key_value_groups, axis=1)

        att = tf.matmul(q, k, transpose_b=True)
        att = att * tf.cast(self.scale, self.compute_dtype)

        mask_2d = self.mask_2d(p_mask, m_mask)
        mask_2d = tf.cast(mask_2d, self.compute_dtype)
        mask_2d_neg = (1.0 - mask_2d) * _neg_inf(att.dtype)

        att = tf.nn.softmax(att + tf.expand_dims(mask_2d_neg, axis=1), axis=-1)
        att = self.attention_dropout(att, training=training)
        att = att * tf.expand_dims(mask_2d, axis=1)

        out = tf.matmul(att, v)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [batch_size, seq_len, self.heads * self.att_dim])

        out = tf.matmul(out, tf.cast(self.out_proj, self.compute_dtype))
        out = out + tf.cast(self.out_bias, self.compute_dtype)
        out = tf.cast(out, self.compute_dtype)
        x = residual + out
        x = tf.cast(x, self.compute_dtype)

        if self.return_att_weights:
            return x, att
        else:
            return x

    def _moe_forward(self, x, training=None):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        x_flat = tf.reshape(x, [-1, self.output_dim])

        router_logits = tf.matmul(x_flat, tf.cast(self.router, self.compute_dtype))
        top_k_logits, top_k_indices = tf.nn.top_k(router_logits, k=self.num_experts_per_token)
        top_k_weights = tf.nn.softmax(top_k_logits, axis=-1)

        aux_loss = None
        if training:
            aux_loss = self._load_balancing_loss(router_logits)

        # Process each token through selected experts
        num_tokens = tf.shape(x_flat)[0]
        output = tf.zeros_like(x_flat, dtype=self.compute_dtype)

        for k in range(self.num_experts_per_token):
            expert_indices = top_k_indices[:, k]
            expert_weights = top_k_weights[:, k:k + 1]

            # Process each expert
            for expert_id in range(self.num_experts):
                mask = tf.cast(tf.equal(expert_indices, expert_id), self.compute_dtype)
                mask = tf.reshape(mask, [-1, 1])

                expert_output = self.experts[expert_id](x_flat)
                expert_output = tf.cast(expert_output, self.compute_dtype)

                weighted_output = expert_output * expert_weights * mask
                output = output + weighted_output

        output = tf.reshape(output, [batch_size, seq_len, self.output_dim])
        return output, aux_loss

    def _load_balancing_loss(self, router_logits):
        routing_probs = tf.nn.softmax(router_logits, axis=-1)
        expert_usage = tf.reduce_mean(routing_probs, axis=0)

        num_experts_float = tf.cast(self.num_experts, self.compute_dtype)
        target = 1.0 / num_experts_float

        aux_loss = num_experts_float * tf.reduce_sum(tf.square(expert_usage - target))
        return aux_loss

    def mask_2d(self, p_mask, m_mask):
        p_mask = tf.cast(p_mask, self.compute_dtype)
        m_mask = tf.cast(m_mask, self.compute_dtype)

        p_mask = tf.cast(tf.where(p_mask == self.pad_token, x=0., y=1.), self.compute_dtype)
        m_mask = tf.cast(tf.where(m_mask == self.pad_token, x=0., y=1.), self.compute_dtype)

        p_mask = tf.expand_dims(p_mask, axis=-1)
        m_mask = tf.expand_dims(m_mask, axis=-1)

        self_peptide_mask_2d = tf.zeros(
            (tf.shape(p_mask)[0], tf.shape(p_mask)[1], tf.shape(p_mask)[1]),
            dtype=self.compute_dtype
        )

        if self.self_attn_mhc:
            self_mhc_mask = m_mask
        else:
            self_mhc_mask = tf.zeros_like(m_mask, dtype=self.compute_dtype)
        self_mhc_mask_2d = tf.cast(tf.broadcast_to(
            self_mhc_mask,
            (tf.shape(m_mask)[0], tf.shape(m_mask)[1], tf.shape(m_mask)[1])
        ), self.compute_dtype)

        pep_mhc_mask = tf.cast(tf.broadcast_to(
            p_mask,
            (tf.shape(p_mask)[0], tf.shape(p_mask)[1], tf.shape(m_mask)[1])
        ), self.compute_dtype)
        pep_mhc_mask = tf.cast(pep_mhc_mask * tf.transpose(m_mask, perm=[0, 2, 1]), self.compute_dtype)

        mhc_pep_mask = tf.cast(tf.broadcast_to(
            m_mask,
            (tf.shape(m_mask)[0], tf.shape(m_mask)[1], tf.shape(p_mask)[1])
        ), self.compute_dtype)
        mhc_pep_mask = tf.cast(mhc_pep_mask * tf.transpose(p_mask, perm=[0, 2, 1]), self.compute_dtype)

        combined_mask_1 = tf.concat([self_peptide_mask_2d, pep_mhc_mask], axis=2)
        combined_mask_2 = tf.concat([mhc_pep_mask, self_mhc_mask_2d], axis=2)
        final_mask = tf.concat([combined_mask_1, combined_mask_2], axis=1)

        final_mask_t = tf.transpose(final_mask, perm=[0, 2, 1])
        final_mask = tf.multiply(final_mask, final_mask_t)

        return final_mask

    def get_config(self):
        config = super().get_config()
        config.update({
            'query_dim': self.query_dim,
            'context_dim': self.context_dim,
            'output_dim': self.output_dim,
            'heads': self.heads,
            'num_key_value_heads': self.num_key_value_heads,
            'return_att_weights': self.return_att_weights,
            'epsilon': self.epsilon,
            'mask_token': self.mask_token,
            'pad_token': self.pad_token,
            'self_attn_mhc': self.self_attn_mhc,
            'apply_rope': self.apply_rope,
            'attention_dropout': self.attention_dropout_rate,
            'ffn_dropout': self.ffn_dropout_rate,
            'use_moe': self.use_moe,
            'num_experts': self.num_experts,
            'num_experts_per_token': self.num_experts_per_token,
            'moe_aux_loss_weight': self.moe_aux_loss_weight,
        })
        return config


@tf.keras.utils.register_keras_serializable(package='Custom', name='AsymmetricPenaltyBinaryCrossentropy')
class AsymmetricPenaltyBinaryCrossentropy(tf.keras.losses.Loss):
    """
    Asymmetric Penalty Binary Cross-Entropy Loss
    Features:
    - Minimum at smoothed label: p = 1-ε (true=1), p = ε (true=0)
    - Steeper penalty toward opposing class
    - Gentler penalty toward actual class
    Args:
        label_smoothing: Smoothing parameter ε (0.05-0.15 recommended)
        asymmetry_strength: Controls penalty asymmetry (0.3-0.8 recommended)
    """

    def __init__(self, label_smoothing=0.1, asymmetry_strength=0.5,
                 name='asymmetric_penalty_binary_crossentropy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.label_smoothing = label_smoothing
        self.asymmetry_strength = asymmetry_strength

    @tf.function(reduce_retracing=True)
    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)

        # Base: Standard label smoothing (ensures minimum at correct location)
        y_smooth = y_true * (1 - self.label_smoothing) + (1 - y_true) * self.label_smoothing
        base_loss = -tf.reduce_mean(y_smooth * tf.math.log(y_pred) + (1 - y_smooth) * tf.math.log(1 - y_pred))

        # Calculate optimal prediction points
        optimal_true1 = 1 - self.label_smoothing  # 0.9 when ε=0.1
        optimal_true0 = self.label_smoothing  # 0.1 when ε=0.1

        # Distance from optimal points
        dist_from_optimal_true1 = tf.abs(y_pred - optimal_true1)
        dist_from_optimal_true0 = tf.abs(y_pred - optimal_true0)

        # Asymmetric penalties
        # For true=1: Stronger penalty when moving toward 0 (opposing class)
        penalty_true1 = y_true * tf.where(
            y_pred < optimal_true1,  # Moving toward opposing class
            self.asymmetry_strength * dist_from_optimal_true1 ** 2,  # Strong penalty
            self.asymmetry_strength * 0.3 * dist_from_optimal_true1 ** 2  # Weak penalty
        )

        # For true=0: Stronger penalty when moving toward 1 (opposing class)
        penalty_true0 = (1 - y_true) * tf.where(
            y_pred > optimal_true0,  # Moving toward opposing class
            self.asymmetry_strength * dist_from_optimal_true0 ** 2,  # Strong penalty
            self.asymmetry_strength * 0.3 * dist_from_optimal_true0 ** 2  # Weak penalty
        )

        total_loss = base_loss + tf.reduce_mean(penalty_true1 + penalty_true0)
        return total_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'label_smoothing': self.label_smoothing,
            'asymmetry_strength': self.asymmetry_strength
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class BinaryActiveNegativeLoss(tf.keras.losses.Loss):
    """
    Binary Active-Negative Loss (ANL-CE)
    Combines normalized cross entropy (NCE) and normalized negative cross entropy (NNCE)
    to balance active and negative learning signals.

    Args:
        alpha: Weight for the active loss component (default: 1.0)
        beta: Weight for the negative loss component (default: 0.5)
        min_prob: Minimum probability threshold for numerical stability (default: 1e-7)
        reduction: Type of reduction to apply to loss (default: 'sum_over_batch_size')
        name: Optional name for the loss instance
    """

    def __init__(
            self,
            alpha=3.0,
            beta=3.0,
            min_prob=0.02,
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            name='binary_active_negative_loss',
            **kwargs
    ):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.beta = beta
        self.min_prob = min_prob

    def _binary_normalized_cross_entropy(self, y_true, y_pred):
        """Binary Normalized Cross Entropy (NCE)"""
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        bce_numerator = -(
                y_true * tf.math.log(y_pred) +
                (1 - y_true) * tf.math.log(1 - y_pred)
        )
        log_prob_sum = -(tf.math.log(y_pred) + tf.math.log(1 - y_pred))
        log_prob_sum = tf.maximum(log_prob_sum, epsilon)
        nce = bce_numerator / log_prob_sum
        return nce

    def _binary_normalized_negative_cross_entropy(self, y_true, y_pred):
        """Binary Normalized Negative Cross Entropy (NNCE)"""
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, self.min_prob, 1 - self.min_prob)
        A = -tf.math.log(tf.constant(self.min_prob))
        log_p_transform = A + tf.math.log(y_pred)
        log_1mp_transform = A + tf.math.log(1 - y_pred)
        neg_ce_numerator = (
                y_true * log_p_transform +
                (1 - y_true) * log_1mp_transform
        )
        neg_ce_denominator = tf.maximum(
            log_p_transform + log_1mp_transform,
            epsilon
        )

        nnce = 1 - (neg_ce_numerator / neg_ce_denominator)
        return nnce

    def call(self, y_true, y_pred):
        """
        Compute the Binary Active-Negative Loss.
        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
        Returns:
            Loss value (per sample, before reduction)
        """
        active_loss = self._binary_normalized_cross_entropy(y_true, y_pred)
        negative_loss = self._binary_normalized_negative_cross_entropy(y_true, y_pred)
        return self.alpha * active_loss + self.beta * negative_loss

    def get_config(self):
        """Returns the config of the loss for serialization."""
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'beta': self.beta,
            'min_prob': self.min_prob
        })
        return config


@tf.keras.utils.register_keras_serializable(name="SpatialTemporalDropout1D", package="CustomLayers")
class SpatialTemporalDropout1D(keras.layers.Layer):
    """Drops entire feature channels AND entire timesteps."""

    def __init__(self, feature_rate=0.1, timestep_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.feature_rate = feature_rate
        self.timestep_rate = timestep_rate

    @tf.function()
    def call(self, inputs, training=None):
        if not training:
            return inputs
        shape = tf.shape(inputs)  # (B, T, F)
        # Drop entire features: mask shape (B, 1, F)
        feature_mask = tf.random.uniform((shape[0], 1, shape[2])) > self.feature_rate
        # Drop entire timesteps: mask shape (B, T, 1)
        timestep_mask = tf.random.uniform((shape[0], shape[1], 1)) > self.timestep_rate
        # Combine masks
        combined_mask = tf.cast(feature_mask & timestep_mask, inputs.dtype)
        # Scale to maintain expected value
        keep_prob = (1 - self.feature_rate) * (1 - self.timestep_rate)
        return inputs * combined_mask / keep_prob

    def get_config(self):
        config = super().get_config()
        config.update({
            "feature_rate": self.feature_rate,
            "timestep_rate": self.timestep_rate
        })
        return config


@tf.function(reduce_retracing=True)
def likelihood_with_attention_vectorized(nn_model, peptide, mhc, label,
                                         pep_mask, mhc_mask, max_pep_len,
                                         temperature=1.0, variable_len=8,
                                         sample_weight=None, training=True):
    """
    Computes the marginal binding probability by marginalizing over latent cores.

    Probabilistic Model:
        - z_i ∈ C_i: latent variable for true binding core
        - q_ic = P(z_i = c | S_i, MHC) = softmax(g(c, S_i, MHC) / τ)
        - P(bind | z_i = c, MHC) = σ(f(c, MHC))
        - P_ij = Σ_c q_ic · σ(f(c, MHC))  [marginalization]

    Model outputs expected:
        - 'core_logits': g(c, S_i, MHC) - core selection scores
        - 'binding_logits': f(c, MHC) - binding prediction scores

    Args:
        nn_model: Keras model with two output heads
        peptide: (B, P, D) peptide embeddings
        mhc: (B, M, D) MHC embeddings
        label: (B, 1) binary binding labels
        pep_mask: (B, P) peptide mask
        mhc_mask: (B, M) MHC mask
        max_pep_len: int, maximum peptide length
        temperature: float, softmax temperature for core selection
        variable_len: int, minimum core length (cores from variable_len to max_pep_len)
        sample_weight: (B, 1) or (B,) optional sample weights
        training: bool, training mode flag

    Returns:
        loss: scalar, mean negative log-likelihood
        attention_weights: (B, N) core selection probabilities q_ic
        binding_probs_per_core: (B, N) per-core binding probabilities σ(f_c)
        cores_stack: (B, N, P, D) padded core embeddings
    """
    B = tf.shape(peptide)[0]
    P = tf.shape(peptide)[1]
    D = tf.shape(peptide)[2]
    M = tf.shape(mhc)[1]

    # Define core lengths: [variable_len, variable_len+1, ..., max_pep_len]
    if variable_len is None:
        ks = [9]
    else:
        ks = range(variable_len, max_pep_len + 1)

    cores_list = []
    masks_list = []
    valid_flags_list = []

    # Extract all possible cores
    for k in ks:
        num_windows = max_pep_len - k + 1
        if num_windows <= 0:
            continue

        for i in range(num_windows):
            # Extract core: (B, k, D)
            core = peptide[:, i:i + k, :]

            # Check validity: core is valid if last position is not padding
            is_valid = tf.not_equal(pep_mask[:, i + k - 1], PAD_TOKEN)
            valid_flags_list.append(is_valid)

            # Pad core to max_pep_len: (B, max_pep_len, D)
            paddings = [[0, 0], [0, max_pep_len - k], [0, 0]]
            padded_core = tf.pad(core, paddings, "CONSTANT", constant_values=PAD_VALUE)
            cores_list.append(padded_core)

            # Create mask for this core length
            mask_vec = tf.concat([
                tf.fill([k], NORM_TOKEN),
                tf.fill([max_pep_len - k], PAD_TOKEN)
            ], axis=0)
            mask_batch = tf.tile(tf.expand_dims(mask_vec, 0), [B, 1])
            masks_list.append(mask_batch)

    # Handle edge case: no valid cores
    if not cores_list:
        return (tf.constant(0.0),
                tf.zeros((B, 0)),
                tf.zeros((B, 0)),
                tf.zeros((B, 0, max_pep_len, D)))

    # Stack all cores: (B, N, ...)
    cores_stack = tf.stack(cores_list, axis=1)  # (B, N, P, D)
    masks_stack = tf.stack(masks_list, axis=1)  # (B, N, P)
    valid_flags_stack = tf.stack(valid_flags_list, axis=1)  # (B, N)
    valid_flags_float = tf.cast(valid_flags_stack, tf.float32)

    N = tf.shape(cores_stack)[1]

    # Flatten for batch processing through model
    cores_flat = tf.reshape(cores_stack, [B * N, max_pep_len, D])
    masks_flat = tf.reshape(masks_stack, [B * N, max_pep_len])

    # Tile MHC for each core
    mhc_tiled = tf.tile(tf.expand_dims(mhc, 1), [1, N, 1, 1])
    mhc_flat = tf.reshape(mhc_tiled, [B * N, M, D])

    mhc_mask_tiled = tf.tile(tf.expand_dims(mhc_mask, 1), [1, N, 1])
    mhc_mask_flat = tf.reshape(mhc_mask_tiled, [B * N, M])

    # Dummy target (if model requires it)
    dummy_target = tf.zeros((B * N, max_pep_len, 21), dtype=tf.float32)

    # Forward pass
    inputs = [cores_flat, masks_flat, mhc_flat, mhc_mask_flat, dummy_target]
    outputs = nn_model(inputs, training=training)

    # Extract model outputs
    core_logits_flat = outputs['core_logits']  # g(c, S_i, MHC): (B*N, 1)
    binding_logits_flat = outputs['binding_logits']  # f(c, MHC): (B*N, 1)

    # Reshape to (B, N)
    core_logits = tf.reshape(core_logits_flat, [B, N])
    binding_logits = tf.reshape(binding_logits_flat, [B, N])

    # =========================================================
    # Core selection distribution: q_ic = softmax(g_c / τ)
    # =========================================================
    # Cast logits to float32 for stable softmax computation
    core_logits_f32 = tf.cast(core_logits, tf.float32)
    binding_logits_f32 = tf.cast(binding_logits, tf.float32)

    # Check if any sample has ALL invalid cores - this would cause NaN in softmax
    any_valid_per_sample = tf.reduce_any(valid_flags_stack, axis=1)  # (B,)

    # Mask invalid cores with -inf before softmax
    neg_inf_val = tf.constant(-1e9, dtype=tf.float32)  # Use float32 value
    core_logits_masked = tf.where(valid_flags_stack, core_logits_f32, neg_inf_val)

    # Compute attention weights in float32
    attention_weights = tf.nn.softmax(core_logits_masked / temperature, axis=1)  # (B, N)

    # For samples with no valid cores, set uniform attention (will be zeroed anyway)
    uniform_attention = tf.ones_like(attention_weights) / tf.cast(tf.shape(attention_weights)[1], tf.float32)
    attention_weights = tf.where(
        tf.expand_dims(any_valid_per_sample, axis=1),
        attention_weights,
        uniform_attention
    )

    # =========================================================
    # Per-core binding probability: σ(f(c, MHC))
    # =========================================================
    binding_probs_per_core = tf.nn.sigmoid(binding_logits_f32)  # (B, N)
    # Mask invalid cores to 0 for clean output
    binding_probs_per_core = binding_probs_per_core * valid_flags_float

    # =========================================================
    # Marginal binding probability: P_ij = Σ_c q_ic · σ(f_c)
    # =========================================================
    pred_prob = tf.reduce_sum(attention_weights * binding_probs_per_core, axis=1)  # (B,)

    # For samples with no valid cores, use 0.5 as default prediction
    pred_prob = tf.where(any_valid_per_sample, pred_prob, tf.constant(0.5, dtype=tf.float32))

    # =========================================================
    # Negative log-likelihood loss (already in float32)
    # =========================================================
    label_flat = tf.cast(tf.reshape(label, [B]), dtype=tf.float32)

    # Clip for numerical stability
    pred_prob_clipped = tf.clip_by_value(pred_prob, 1e-6, 1.0 - 1e-6)

    # Binary cross-entropy
    loss = tf.keras.backend.binary_crossentropy(label_flat, pred_prob_clipped)

    # Apply sample weights if provided
    if sample_weight is not None:
        sample_weight = tf.cast(sample_weight, tf.float32)
        if len(sample_weight.shape) == 2:
            sample_weight = tf.squeeze(sample_weight, axis=1)
        loss = loss * sample_weight

    # Final safety check
    loss = tf.where(tf.math.is_finite(loss), loss, tf.zeros_like(loss))

    # Cast attention weights back to compute dtype for return
    attention_weights = tf.cast(attention_weights, core_logits.dtype)
    binding_probs_per_core = tf.cast(binding_probs_per_core, core_logits.dtype)

    return tf.reduce_mean(loss), attention_weights, binding_probs_per_core, cores_stack


# Training utilities
def load_embedding_db(npz_path: str):
    """Load embedding database with memory mapping."""
    try:
        # Try loading without explicit allow_pickle (defaults to True)
        return np.load(npz_path, mmap_mode="r")
    except ValueError as e:
        if "allow_pickle" in str(e):
            # If pickle error, try with explicit allow_pickle=True
            print(f"Warning: NPZ file contains pickled data, loading with allow_pickle=True")
            return np.load(npz_path, mmap_mode="r", allow_pickle=True)
        else:
            raise e


def min_max_norm(emb, mhc_class=1):
    """min max of ESM3-open embeddings 25.09.2025"""
    if mhc_class == 2:
        min = -14144.0
        max = 1456.0
    else:
        min = -15360.0
        max = 1440.0
    # normalize embedding
    emb_norm = 2 * (emb - min) / (max - min) - 1
    return emb_norm


def log_norm_zscore(emb, eps=1e-9):
    """z-score normalization after log-transform"""
    emb_shifted = emb - emb.min() + eps
    emb_log = np.log1p(emb_shifted)
    mean, std = emb_log.mean(), emb_log.std()
    emb_norm = (emb_log - mean) / std
    return emb_norm


def _preprocess_df_chunk(args):
    """Top-level helper for multiprocessing chunk processing."""
    chunk, seq_map, embed_map, mhc_class = args
    chunk = chunk.copy()

    # Replicate original logic
    chunk['_cleaned_key'] = chunk.apply(
        lambda r: r.get('mhc_embedding_key', r['allele'].replace(' ', '').replace('*', '').replace(':', '')),
        axis=1
    )
    chunk['_emb_key'] = chunk['_cleaned_key'].apply(lambda k: get_embed_key(clean_key(k), embed_map))

    if mhc_class == 2:
        def _get_mhc_seq_class2(key):
            parts = key.split('_')
            return seq_map.get(get_embed_key(clean_key(parts[0]), seq_map), '') + \
                   seq_map.get(get_embed_key(clean_key(parts[1]), seq_map), '') if len(parts) >= 2 else ''

        chunk['_mhc_seq'] = chunk['_cleaned_key'].apply(_get_mhc_seq_class2)
    else:
        chunk['_mhc_seq'] = chunk['_emb_key'].apply(lambda k: seq_map.get(get_embed_key(clean_key(k), seq_map), ''))

    return chunk


def load_metadata(tfrecord_dir):
    """Load metadata from tfrecords directory"""
    metadata_path = os.path.join(tfrecord_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata


def load_embedding_table(lookup_path, metadata):
    """Load MHC embedding lookup table with correct dtype."""
    with np.load(lookup_path) as data:
        num_embeddings = len(data.files)
        # FIX: Use float32 to avoid potential precision loss from float16.
        table = np.zeros((num_embeddings, metadata['MAX_MHC_LEN'], metadata['ESM_DIM']), dtype=np.float32)
        for i in range(num_embeddings):
            table[i] = data[str(i)]
    return tf.constant(table, dtype=tf.float32)


def normalize_embedding_tf(emb, method="clip_norm1000"):
    """TensorFlow implementation of the normalization logic."""
    if method == "clip_norm1000":
        emb_norm = tf.clip_by_value(emb, -1000.0, 1000.0)
        return 20.0 * (emb_norm - (-1000.0)) / (1000.0 - (-1000.0)) - 10.0
    elif method == "robust_zscore":
        # Per-sample normalization (better for ESM embeddings)
        mean = tf.reduce_mean(emb, axis=-1, keepdims=True)
        std = tf.math.reduce_std(emb, axis=-1, keepdims=True)
        emb_norm = (emb - mean) / (std + 1e-8)
        emb_norm = tf.clip_by_value(emb_norm, -5.0, 5.0)
        return emb_norm
    else:
        return emb  # No normalization


# Utility functions for handling embeddings and sequences
def process_pdb_distance_matrix(pdb_path, threshold, peptide, chainid='A', carbon='CB'):
    # Parse PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    chain = structure[0][chainid]  # Assumes single chain 'A'
    # Extract CB (or CA for Glycine) coordinates
    coords = []
    carbon_not = 'CA' if carbon == 'CB' else 'CB'
    for res in chain:
        if res.get_resname() == 'GLY':  # Glycine
            atom = res['CA']
        else:
            atom = res[carbon] if carbon in res else res[carbon_not]
        coords.append(atom.get_coord())
    coords = np.array(coords)
    # Compute distance matrix
    dist_matrix = squareform(pdist(coords, metric='euclidean'))
    # mask = np.where(dist_matrix < threshold, 0., 1)
    mask = np.eye(dist_matrix.shape[0], dtype=np.float32)  # all masks are==1
    mask = np.where(mask == 1, 0, 1)  # all masks are==0 now
    dist_matrix = 1 / (dist_matrix + 1e-9)
    result = dist_matrix * mask
    # result = result[len(result)-len(peptide):, :len(result)-len(peptide)]
    result = (result - np.min(result)) / (np.max(result) - np.min(result))
    return result


# res = process_pdb_distance_matrix('../pdbs/MHCI_ref_4U6Y.pdb', 9, "FLNKDLEVDGHFVTM", 'C', 1.5)
# # How to use
# thr = 9
# chain = 'A'
# carbons = ['CA', 'CB']
# final_dict = {}
# for i, row in df.iterrows():
#     allele = row['allale']
#     peptide = row['peptide']
#     pdb_path = row['pdb_path']
#     out = []
#     for carbon in carbons:
#         res = process_pdb_distance_matrix(pdb_path, thr, peptide, chain, carbon)
#         out.append(np.expand_dims(res, axis=-1))
#     out = np.concatenate(out, axis=-1) #(P,M,2)
#     final_dict[allele] = out


def cluster_aa_sequences(
        csv_path: str,
        *,
        id_col: str = "allele",
        seq_col: str = "sequence",
        linkage_method: str = "average",
        gap_mode: Literal["count_gaps", "ignore_gaps"] = "ignore_gaps",
        k: int | None = None,
        distance_threshold: float | None = None,
        plot: bool = True,
        mafft_extra: str | None = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Cluster amino-acid sequences contained in `csv_path`.

    Parameters
    ----------
    csv_path : str
        CSV file with at least columns `id_col` and `seq_col`.
    id_col, seq_col : str
        Column names for sequence IDs and sequences.
    linkage_method : str
        Method for scipy.cluster.hierarchy.linkage.
    gap_mode : {"count_gaps", "ignore_gaps"}
        How gaps are treated in the pair-wise distance.
    k : int, optional
        If given, returns exactly k clusters (criterion="maxclust").
    distance_threshold : float, optional
        If given, cut the dendrogram at this height (criterion="distance").
    plot : bool
        If True, show a dendrogram.
    mafft_extra : str, optional
        Extra options appended to the MAFFT command line.

    Returns
    -------
    df_out : pandas.DataFrame
        Original columns plus a new column "Cluster".
    Z : numpy.ndarray
        The linkage matrix from `scipy.cluster.hierarchy.linkage`.
    """
    df = pd.read_csv(csv_path)
    if id_col not in df.columns or seq_col not in df.columns:
        raise ValueError(f"CSV must contain columns '{id_col}' and '{seq_col}'")

    ids = df[id_col].astype(str).tolist()
    seqs = df[seq_col].astype(str).tolist()

    # 2. Write sequences to a temporary FASTA file
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".fasta") as tmp_fa:
        for ident, seq in zip(ids, seqs):
            tmp_fa.write(f">{ident}\n{seq}\n")
        fasta_path = tmp_fa.name

    # 3. Run MAFFT
    extra = mafft_extra if mafft_extra else ""
    mafft_cline = MafftCommandline(input=fasta_path, auto=True)
    if extra:
        mafft_cline.cline += f" {extra}"
    print("Running MAFFT …")
    mafft_stdout, _ = mafft_cline()
    os.unlink(fasta_path)  # clean up temp file
    aln = AlignIO.read(StringIO(mafft_stdout), "fasta")
    aligned_ids = [rec.id for rec in aln]
    aligned_seqs = [str(rec.seq) for rec in aln]
    print(f"Alignment length = {aln.get_alignment_length()} aa")

    # 4. Pair-wise distances on aligned sequences
    def dist_count_gaps(a, b):
        return sum(c1 != c2 for c1, c2 in zip(a, b)) / len(a)

    def dist_ignore_gaps(a, b):
        same = diff = 0
        for c1, c2 in zip(a, b):
            if c1 == "-" or c2 == "-":
                continue
            same += 1
            if c1 != c2:
                diff += 1
        return 0.0 if same == 0 else diff / same

    seq_dist = dist_count_gaps if gap_mode == "count_gaps" else dist_ignore_gaps

    X = np.array(aligned_seqs, dtype=object)[:, None]  # shape (N,1)
    dist_vector = pdist(X, lambda u, v: seq_dist(u[0], v[0]))

    # 5. Hierarchical clustering
    Z = linkage(dist_vector, method=linkage_method)

    # 6. Retrieve flat clusters
    if k is None and distance_threshold is None:
        raise ValueError("Specify either `k` or `distance_threshold`")

    if k is not None:
        clusters = fcluster(Z, t=k, criterion="maxclust")
    else:
        clusters = fcluster(Z, t=distance_threshold, criterion="distance")

    # 7. Optional dendrogram
    if plot:
        plt.figure(figsize=(25, 5))
        dendrogram(
            Z,
            labels=aligned_ids,
            leaf_rotation=90,
            leaf_font_size=9,
            color_threshold=distance_threshold if distance_threshold else None,
        )
        plt.title("Hierarchical clustering (MAFFT aligned)")
        plt.ylabel("Distance")
        plt.tight_layout()
        plt.show()

    df_out = df.copy()
    df_out["cluster"] = clusters
    return df_out, Z


def create_k_fold_leave_one_cluster_out_stratified_cv(
        df: pd.DataFrame,
        k: int = 5,
        target_col: str = "label",
        cluster_col: str = "cluster",
        id_col: str = "allele",
        subset_prop: float = 1.0,
        train_size: float = 0.8,
        random_state: int = 42,
        augmentation: str = "down_sampling",  # {"down_sampling", "GNUSS", None}
        keep_val_only_cluster: bool = True,  # set False if you do NOT want to have a cluster that is only in validation
) -> List[Tuple[pd.DataFrame, pd.DataFrame, Hashable]]:
    """
    Build *k* folds where ***one whole sequence cluster is left out*** of
    both train and validation.

    Workflow for every fold
    -----------------------
    1. Choose one cluster -> `left_out_cluster` (never in train or val).
    2. If `keep_val_only_cluster` is True:
       pick one additional cluster -> `val_only_cluster`
       (present only in validation).
    3. Split remaining rows into train / extra-validation stratified on
       `target_col` with proportion `train_size`.
    4. Apply class-balancing augmentation (down-sampling or GNUSS).

    Returns
    -------
    list[(train_df, val_df, left_out_cluster)]
    """

    rng = np.random.RandomState(random_state)

    # ------------------------------------------------------------
    # (optional) take a subset of the whole frame
    # ------------------------------------------------------------
    if subset_prop < 1.0:
        if not (0.0 < subset_prop <= 1.0):
            raise ValueError("subset_prop must be in (0,1]")
        df = df.sample(frac=subset_prop, random_state=random_state).reset_index(drop=True)
        print(f"Using a {subset_prop:.2%} random subset of the data.")

    # ------------------------------------------------------------
    # 0) basic sanity checks
    # ------------------------------------------------------------
    if cluster_col not in df.columns:
        raise ValueError(f"{cluster_col=} not found in DataFrame")
    if target_col not in df.columns:
        raise ValueError(f"{target_col=} not found in DataFrame")

    unique_clusters = df[cluster_col].unique()
    if k > len(unique_clusters):
        raise ValueError(f"k={k} > #unique clusters ({len(unique_clusters)})")

    left_out_clusters = rng.choice(unique_clusters, size=k, replace=False)
    left_out_ids = df[df[cluster_col].isin(left_out_clusters)][id_col].unique()

    # save the left out ids to a txt file
    with open("left_out_ids.txt", "w") as f:
        for left_out_id in left_out_ids:
            f.write(f"{left_out_id}\n")

    # ------------------------------------------------------------
    # augmentation helpers
    # ------------------------------------------------------------
    def _balance_down_sampling(frame: pd.DataFrame, seed: int):
        min_count = frame[target_col].value_counts().min()
        parts = [
            resample(
                frame[frame[target_col] == lbl],
                replace=False,
                n_samples=min_count,
                random_state=seed,
            )
            for lbl in frame[target_col].unique()
        ]
        return pd.concat(parts, ignore_index=True)

    def _balance_GNUSS(frame: pd.DataFrame, seed: int):
        counts = frame[target_col].value_counts()
        max_count = counts.max()

        numeric_cols = frame.select_dtypes(include="number").columns
        parts = [frame]  # originals

        for label, cnt in counts.items():
            if cnt == max_count:
                continue
            need = max_count - cnt
            sampled = frame[frame[target_col] == label].sample(
                n=need, replace=True, random_state=seed
            )
            # very small noise
            noise = rng.normal(loc=0.0, scale=1e-6, size=(need, len(numeric_cols)))
            sampled.loc[:, numeric_cols] += noise
            parts.append(sampled)

        return pd.concat(parts, ignore_index=True)

    # ------------------------------------------------------------
    # build each fold
    # ------------------------------------------------------------
    folds: List[Tuple[pd.DataFrame, pd.DataFrame, Hashable]] = []

    for fold_idx, left_out_cluster in enumerate(left_out_clusters, 1):
        fold_seed = random_state + fold_idx

        # remove the left-out cluster
        mask_left_out = df[cluster_col] == left_out_cluster
        working_df = df.loc[~mask_left_out].copy()

        # pick ONE cluster that will be *only* in validation
        if keep_val_only_cluster:
            gss = GroupShuffleSplit(n_splits=1, test_size=1, random_state=fold_seed)
            (__, val_only_idx), = gss.split(
                X=np.zeros(len(working_df)),
                groups=working_df[cluster_col],
            )
            val_only_cluster = working_df.iloc[val_only_idx][cluster_col].unique()[0]

            mask_val_only = working_df[cluster_col] == val_only_cluster
            df_val_only = working_df[mask_val_only]
            df_eligible = working_df[~mask_val_only]
        else:
            val_only_cluster = None
            df_val_only = pd.DataFrame(columns=df.columns)
            df_eligible = working_df

        # stratified split of the remaining rows
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=train_size, random_state=fold_seed
        )
        train_idx, extra_val_idx = next(
            sss.split(df_eligible, df_eligible[target_col])
        )
        df_train = df_eligible.iloc[train_idx]
        df_val = pd.concat(
            [df_val_only, df_eligible.iloc[extra_val_idx]],
            ignore_index=True,
        )

        # ---------------------------------------------------- #
        #  balance
        # ---------------------------------------------------- #
        if augmentation == "GNUSS":
            df_train_bal = _balance_GNUSS(df_train, fold_seed)
            df_val_bal = _balance_GNUSS(df_val, fold_seed)
        elif augmentation == "down_sampling":
            df_train_bal = _balance_down_sampling(df_train, fold_seed)
            df_val_bal = _balance_down_sampling(df_val, fold_seed)
        elif augmentation is None or augmentation.lower() == "none":
            df_train_bal = df_train.copy()
            df_val_bal = df_val.copy()
        else:
            raise ValueError(f"Unknown augmentation: {augmentation}")

        df_train_bal = (
            df_train_bal.sample(frac=1.0, random_state=fold_seed).reset_index(drop=True)
        )
        df_val_bal = (
            df_val_bal.sample(frac=1.0, random_state=fold_seed).reset_index(drop=True)
        )

        folds.append((df_train_bal, df_val_bal, left_out_cluster))

        print(
            f"[fold {fold_idx}/{k}] left-out cluster={left_out_cluster} | "
            f"val-only cluster={val_only_cluster} | "
            f"train={len(df_train_bal)}, val={len(df_val_bal)}"
        )

    return folds


def apply_dynamic_masking(features, emd_mask_d2=True):
    """
    Applies random masking for training augmentation inside the tf.data pipeline.
    This version is corrected to match the original DataGenerator logic.
    """
    # Peptide Masking
    valid_pep_positions = tf.where(tf.equal(features["pep_mask"], NORM_TOKEN))
    num_valid_pep = tf.shape(valid_pep_positions)[0]
    # At least 2 positions, or 15% of the valid sequence length
    num_to_mask_pep = tf.maximum(2, tf.cast(tf.cast(num_valid_pep, tf.float32) * 0.15, tf.int32))
    shuffled_pep_indices = tf.random.shuffle(valid_pep_positions)[:num_to_mask_pep]
    if tf.shape(shuffled_pep_indices)[0] > 0:
        # Update the mask to MASK_TOKEN (-1.0)
        features["pep_mask"] = tf.tensor_scatter_nd_update(features["pep_mask"], shuffled_pep_indices,
                                                           tf.repeat(MASK_TOKEN, num_to_mask_pep))
        # Zero out the feature values for the masked positions
        feat_dtype = features["pep_emb"].dtype
        mask_updates_pep = tf.fill([num_to_mask_pep, tf.shape(features["pep_emb"])[-1]],
                                   tf.cast(MASK_VALUE, feat_dtype))
        features["pep_emb"] = tf.tensor_scatter_nd_update(features["pep_emb"], shuffled_pep_indices,
                                                          mask_updates_pep)
    # MHC Masking
    valid_mhc_positions = tf.where(tf.equal(features["mhc_mask"], NORM_TOKEN))
    num_valid_mhc = tf.shape(valid_mhc_positions)[0]
    # At least 5 positions, or 15% of the valid sequence length
    num_to_mask_mhc = tf.maximum(10, tf.cast(tf.cast(num_valid_mhc, tf.float32) * 0.15, tf.int32))
    shuffled_mhc_indices = tf.random.shuffle(valid_mhc_positions)[:num_to_mask_mhc]
    if tf.shape(shuffled_mhc_indices)[0] > 0:
        # Update the mask to MASK_TOKEN (-1.0)
        features["mhc_mask"] = tf.tensor_scatter_nd_update(features["mhc_mask"], shuffled_mhc_indices,
                                                           tf.repeat(MASK_TOKEN, num_to_mask_mhc))
        # Zero out the feature values for the masked positions
        mhc_dtype = features["mhc_emb"].dtype
        mask_updates_mhc = tf.fill([num_to_mask_mhc, tf.shape(features["mhc_emb"])[-1]], tf.cast(MASK_VALUE, mhc_dtype))
        features["mhc_emb"] = tf.tensor_scatter_nd_update(features["mhc_emb"], shuffled_mhc_indices, mask_updates_mhc)

    # Dimension-level masking for MHC embeddings
    if emd_mask_d2:
        # Find positions that are STILL valid (not padded and not positionally masked)
        remaining_valid_mhc = tf.where(tf.equal(features["mhc_mask"], NORM_TOKEN))
        if tf.shape(remaining_valid_mhc)[0] > 0:
            # Get the embeddings at these remaining valid positions
            valid_embeddings = tf.gather_nd(features["mhc_emb"], remaining_valid_mhc)
            # Create a random mask for the feature dimensions
            dim_mask = tf.random.uniform(shape=tf.shape(valid_embeddings), dtype=features["mhc_emb"].dtype) < tf.cast(
                0.15, features["mhc_emb"].dtype)
            # Apply the mask (multiply by 0 where True, 1 where False)
            masked_embeddings = valid_embeddings * tf.cast(~dim_mask, features["mhc_emb"].dtype)
            # Scatter the modified embeddings back into the original tensor
            features["mhc_emb"] = tf.tensor_scatter_nd_update(features["mhc_emb"], remaining_valid_mhc,
                                                              masked_embeddings)
    return features


# Reduced alphabet mapping based on MMseqs2
mmseqs2_reduced_alphabet = {
    "L": ["L", "M"],
    "I": ["I", "V"],
    "K": ["K", "R"],
    "E": ["E", "Q"],
    "A": ["A", "S", "T"],
    "N": ["N", "D"],
    "F": ["F", "Y"],
    "G": ["G"],
    "H": ["H"],
    "C": ["C"],
    "P": ["P"],
    "W": ["W"],
}

mmseqs2_reduced_alphabet_rev = {
    "A": "A",
    "C": "C",
    "D": "N",
    "E": "E",
    "F": "F",
    "G": "G",
    "H": "H",
    "I": "I",
    "K": "K",
    "L": "L",
    "M": "L",
    "N": "N",
    "P": "P",
    "Q": "E",
    "R": "K",
    "S": "A",
    "T": "A",
    "V": "I",
    "W": "W",
    "Y": "F",
}


def reduced_anchor_pair(peptide_seq: str) -> str:
    """ Gets the second and last amino acid, maps them to the MMseqs2 reduced alphabet, and returns them as a pair string. """
    # remove gaps
    peptide_seq = peptide_seq.replace("-", "").replace("*", "").replace(" ", "").replace("X", "").upper()
    # print(peptide_seq)
    if len(peptide_seq) < 2:
        return "Short"
    p2 = peptide_seq[1].upper()  # second amino acid
    p_omega = peptide_seq[-1].upper()  # last amino acid

    # Map to reduced alphabet, handle missing keys gracefully
    p2_reduced = mmseqs2_reduced_alphabet_rev.get(p2, 'X')
    # print(f"p2: {p2}, reduced: {p2_reduced}")
    p_omega_reduced = mmseqs2_reduced_alphabet_rev.get(p_omega, 'X')
    # print(f"p_omega: {p_omega}, reduced: {p_omega_reduced}")

    return f"{p2_reduced}-{p_omega_reduced}"  # e.g. "A-S" should return 144 combinations


# --- Helper function and data for physiochemical properties ---
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
POLAR_UNCHARGED = set("STNQCY")
CHARGED = set("KRHDE")
POLAR_SET = POLAR_UNCHARGED | CHARGED


def peptide_properties_biopython(seq: str, pH: float = 7.4) -> Dict[str, float]:
    """
    Compute peptide properties using Biopython's ProtParam:
      - Kyte–Doolittle average hydrophobicity (GRAVY)
      - Net charge at given pH (includes termini)
      - Fraction of polar residues (charged + polar uncharged)
      - Isoelectric point (pI)
      - Aromaticity
      - Instability index
      - Molecular weight
      - Extinction coefficients (reduced and with cystines)
    Unknowns/gaps (e.g., X, - , *) are removed before analysis.
    """
    if not seq:
        return {
            "original_length": 0,
            "analyzed_length": 0,
            "hydrophobicity": 0.0,
            "charge": 0.0,
            "fraction_polar": 0.0,
            "pI": 0.0,
            "aromaticity": 0.0,
            "instability_index": 0.0,
            "molecular_weight": 0.0,
            "extinction_coeff_reduced": 0,
            "extinction_coeff_cystine": 0,
        }

    seq = seq.upper()
    cleaned = "".join(aa for aa in seq if aa in VALID_AA)
    L = len(cleaned)

    if L == 0:
        return {
            "original_length": len(seq),
            "analyzed_length": 0,
            "hydrophobicity": 0.0,
            "charge": 0.0,
            "fraction_polar": 0.0,
            "pI": 0.0,
            "aromaticity": 0.0,
            "instability_index": 0.0,
            "molecular_weight": 0.0,
            "extinction_coeff_reduced": 0,
            "extinction_coeff_cystine": 0,
        }

    pa = ProteinAnalysis(cleaned)
    counts = Counter(cleaned)

    gravy = pa.gravy()  # Kyte–Doolittle average hydropathy
    charge = pa.charge_at_pH(pH)  # includes side chains + N/C termini
    frac_polar = sum(n for aa, n in counts.items() if aa in POLAR_SET) / L

    pI = pa.isoelectric_point()
    aromaticity = pa.aromaticity()
    instability = pa.instability_index()
    mw = pa.molecular_weight()
    # Returns (reduced, with cystines)
    ext_reduced, ext_cystine = pa.molar_extinction_coefficient()

    return {
        "original_length": len(seq),
        "analyzed_length": L,
        "hydrophobicity": gravy,
        "charge": charge,
        "fraction_polar": frac_polar,
        "pI": pI,
        "aromaticity": aromaticity,
        "instability_index": instability,
        "molecular_weight": mw,
        "extinction_coeff_reduced": ext_reduced,
        "extinction_coeff_cystine": ext_cystine,
    }


def cn_terminal_amino_acids(peptide: str, n_term_len: int = 3, c_term_len: int = 3) -> dict:
    """
    Analyzes a peptide sequence to determine the dominant chemical property
    (polar, hydrophobic, or charged) of its N-terminal, C-terminal, and core regions.

    Args:
        peptide (str): The amino acid sequence of the peptide.
        n_term_len (int): The number of amino acids to consider for the N-terminus.
        c_term_len (int): The number of amino acids to consider for the C-terminus.

    Returns:
        dict: A dictionary with keys 'N-term', 'Core', and 'C-term', where the values
              are strings indicating the region and its dominant property
              (e.g., 'N-term_polar').
    """
    # --- Define Amino Acid Properties ---
    # Using sets for efficient membership checking
    polar_aa = {'S', 'T', 'C', 'N', 'Q', 'Y'}
    hydrophobic_aa = {'A', 'V', 'I', 'L', 'M', 'F', 'W', 'P', 'G'}
    # Both acidic (-) and basic (+) are considered 'charged'
    charged_aa = {'D', 'E', 'R', 'K', 'H'}

    def get_property(amino_acid: str) -> str:
        """Helper function to classify a single amino acid."""
        if amino_acid in polar_aa:
            return 'polar'
        if amino_acid in hydrophobic_aa:
            return 'hydrophobic'
        if amino_acid in charged_aa:
            return 'charged'
        return 'unknown'  # Should not happen with standard peptides

    def get_dominant_property(sequence: str) -> str:
        """
        Helper function to find the most common property in a sequence.
        In case of a tie, the priority is hydrophobic > polar > charged.
        """
        if not sequence:
            return 'none'  # Handle empty core sequences

        properties = [get_property(aa) for aa in sequence]

        # Count occurrences of each property
        counts = Counter(properties)

        # Determine the dominant property with a tie-breaking rule
        # The key for sorting is a tuple of the count and a priority value
        # Higher count is better, lower priority value is better for ties
        priority = {'hydrophobic': 0, 'polar': 1, 'charged': 2, 'none': 3, 'unknown': 4}

        # Find the property with the maximum count, using priority to break ties
        dominant_prop = max(counts, key=lambda prop: (counts[prop], -priority[prop]))

        return dominant_prop

    # --- Define Peptide Regions ---
    # Ensure the function handles peptides shorter than the defined terminal lengths
    actual_n_len = min(n_term_len, len(peptide))
    actual_c_len = min(c_term_len, len(peptide))

    n_terminal_seq = peptide[:actual_n_len]
    c_terminal_seq = peptide[-actual_c_len:]

    # The core is what's left in the middle
    # Handle cases where terminals overlap or there is no core
    if len(peptide) > actual_n_len + actual_c_len:
        core_seq = peptide[actual_n_len:-actual_c_len]
    else:
        core_seq = ""  # No distinct core

    # --- Classify Each Region ---
    n_term_prop = get_dominant_property(n_terminal_seq)
    c_term_prop = get_dominant_property(c_terminal_seq)
    core_prop = get_dominant_property(core_seq)

    # --- Format Output ---
    # This format matches the keys in your `segment_color_map`
    result = {
        'N-term': f'N-term_{n_term_prop}',
        'Core': f'Core_{core_prop}',
        'C-term': f'C-term_{c_term_prop}'
    }

    return result


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the latent vector."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        # KL Divergence Regularization Loss
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        return z, kl_loss


def binary_focal_loss(y_true,
                      y_pred,
                      gamma=2.0,
                      alpha=0.25,
                      from_logits=False,
                      apply_class_balancing=False,
                      label_smoothing=0.0,
                      rounding_thr=0.05):
    """
    TensorFlow implementation of binary focal loss.
    """
    # 1. Apply sigmoid if predictions are logits
    if from_logits:
        y_pred = tf.nn.sigmoid(y_pred)
    # 2. Apply label smoothing if requested
    if label_smoothing > 0:
        y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
    # 3. Compute cross-entropy (standard binary CE part)
    bce = -(y_true * tf.math.log(y_pred + 1e-7) +
            (1 - y_true) * tf.math.log(1 - y_pred + 1e-7))
    if rounding_thr > 0:
        bce = tf.where(bce <= rounding_thr, bce / 2., bce)
    # 4. Compute modulating factor (focusing on hard examples)
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    modulating_factor = tf.pow(1.0 - p_t, gamma)
    # 5. Apply alpha balancing if needed
    if apply_class_balancing:
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
    else:
        alpha_factor = 1.0
    # 6. Final focal loss
    focal_loss = alpha_factor * modulating_factor * bce
    return focal_loss


def determine_conv_params(input_dim, output_dim, max_kernel_size=5, max_strides=2):
    """
    Determine kernel size and strides for a single Conv1D layer.

    Args:
        input_dim (int): Input sequence length.
        output_dim (int): Desired output sequence length.
        max_kernel_size (int): Maximum allowed kernel size.
        max_strides (int): Maximum allowed strides.

    Returns:
        tuple: (kernel_size, strides) if found, else None.
    """
    candidates = []
    for strides in range(1, max_strides + 1):
        for kernel_size in range(1, max_kernel_size + 1):
            if (input_dim - kernel_size) // strides + 1 == output_dim:
                candidates.append((kernel_size, strides))
    if candidates:
        candidates.sort(key=lambda x: (x[1], x[0]))  # Prefer smaller strides, then kernel size
        return candidates[0]
    return None


def determine_ks_dict(initial_input_dim, output_dims, max_kernel_size=50, max_strides=20):
    """
    Determine kernel sizes and strides for four sequential Conv1D layers.

    Args:
        initial_input_dim (int): Initial input sequence length.
        output_dims (list of int): List of four output sequence lengths after each layer.
        max_kernel_size (int): Maximum allowed kernel size.
        max_strides (int): Maximum allowed strides.

    Returns:
        dict: Dictionary with keys "k1", "s1", ..., "k4", "s4", or None if no valid parameters.
    """
    if len(output_dims) != 4:
        raise ValueError("output_dims must contain exactly four integers.")

    ks_dict = {}
    current_dim = initial_input_dim

    for i, output_dim in enumerate(output_dims, start=1):
        result = determine_conv_params(current_dim, output_dim, max_kernel_size, max_strides)
        if result is not None:
            kernel_size, strides = result
            ks_dict[f"k{i}"] = kernel_size
            ks_dict[f"s{i}"] = strides
            current_dim = output_dim  # Update input for next layer
        else:
            print(f"No valid parameters found for layer {i}: {current_dim} → {output_dim}")
            return None

    return ks_dict
