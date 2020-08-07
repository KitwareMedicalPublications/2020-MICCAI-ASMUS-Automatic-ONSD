'''
Reads Clarius video data into a numpy video matrix
'''

import numpy as npy
import skvideo
skvideo.setFFmpegPath("C:\\src\\ffmpeg-4.2.2-win64-dev\\bin")
import skvideo.io
from skimage.transform import SimilarityTransform

EYE_WIDTH_MM = 24.2
EYE_HEIGHT_MM = 23.7
CLARIUS_EYE_WIDTH = 400
def set_clarius_dimension(img):
    '''
    Hard-coded estimate for the CLARIUS spacing.  TODO: get this somehow from Clarius.
    '''
    # assume uniform spacing
    spacing = EYE_WIDTH_MM / CLARIUS_EYE_WIDTH
    img.SetSpacing([spacing, spacing])

def load(filepath):
    '''
    Loads a ultrasound video from a Clarius device and returns it as a cropped npy array.
    
    Parameters:
        filepath (string): filepath to .mp4 file
        
    Returns:
        (cropped video (ndarray), translation array (tuple)): video array [frames, rows, columns] cropped to exclude letterbox, array to translate coordinates
    '''
    
    # take single RGB channel for grey
    vid = skvideo.io.vread(filepath)[::,::,::,1]
 

