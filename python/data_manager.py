import os.path as pth
import xml.etree.ElementTree as ET
import numpy as npy

class DataManager:
    ''' Provides a layer of abstraction for filenames in a study'''
    def __init__(self, study_id):
        '''
        
        Parameters:
            study_id (string): The subdirectory of the study in the data directory (do not add prefix)'''

        self.study_id = study_id
        self.study_path = pth.join('..', 'data', study_id)
        self.typ_suffix = {'video': '.mp4', 'annotation': '-annotation.xml'}
    

    def get_by_absid(self, typ, absid):
        '''
        Get absolute paths to files by semantic description.
        
        Parameters:
            typ (string): The "type" of the file.  E.g. "video" for original video, "annotation" for annotation XML
            absid (string): The full name of the data.  I.e., the filename (no suffix) of the video file.
        
        Returns:
            string: Absolute filepath to requested data file.
        '''
        subdir = absid.split('-')[0]
        suff = self.typ_suffix[typ]
        return pth.join(self.study_path, subdir, absid + suff)


class EyeNerveAnnotation:
    def __init__(self, filepath):
        self.filepath = filepath
        self.root = ET.parse(filepath).getroot()
        self.eye_boxes = self.root.findall("./track[@label='Eye Box']/box[@occluded='0']")
        self.nerve_boxes = self.root.findall("./track[@label='Nerve Box']/box[@occluded='0']")
        
        self.eye_frames = npy.asarray([ x.get('frame') for x in self.eye_boxes ], dtype=npy.dtype(float))
        self.nerve_frames = npy.asarray([ x.get('frame') for x in self.nerve_boxes ], dtype=npy.dtype(float))
        
        self.eye_toplefts = npy.asarray([ (x.get('xtl'), x.get('ytl')) for x in self.eye_boxes ], dtype=npy.dtype(float))
        self.nerve_toplefts = npy.asarray([ (x.get('xtl'), x.get('ytl')) for x in self.nerve_boxes ], dtype=npy.dtype(float))
        
        self.eye_bottomrights = npy.asarray([ (x.get('xbr'), x.get('ybr')) for x in self.eye_boxes ], dtype=npy.dtype(float))
        self.nerve_bottomrights = npy.asarray([ (x.get('xbr'), x.get('ybr')) for x in self.nerve_boxes ], dtype=npy.dtype(float))
        