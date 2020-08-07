from scipy.ndimage import binary_hit_or_miss
import numpy as np
import scipy.interpolate
import skimage.measure
import skvideo
skvideo.setFFmpegPath("C:\\src\\ffmpeg-4.2.2-win64-dev\\bin")
import skvideo.io
import clarius
from collections import defaultdict
from abc import ABC, abstractmethod
from scipy import integrate
import math
import itk
import pickle
import os.path as path
import matplotlib.pyplot as plt
from datetime import datetime
import os
'''
Module: ocularus (Ocular Ultrasound)

Notes
-----
Be aware there are 4 coordinate systems.  1. ITK's physical coordinate system.  2. ITK's index coordinate system.  3.  Numpy's index coordinate system.  4.  pyplot's
coordinate system.  ITK's index coordinate system (#2) is the raw video coordinate system, even if we crop the image.

'''




# Abstractions
# Identify optic nerve
    # Identify optic nerve width
# Identify eye
    # Identify eye center, eye socket, etc
# will need
# all nerves from video
# all eyes from video
# stuff?

# first cludgy part, handle clarius dimension and cropping

ImageType = itk.Image[itk.F,2]

class CropPreprocess:
    def __init__(self, ref, percentage_x, percentage_y):
        '''
        Crops any black rows/columns (i.e. letterboxing) and a fixed percentage of the image
        
        Parameters
        ----------
        ref (itk.Image): reference image to define crop
        percentage_x (real): percentage to remove from either side (2.5% * 2 = 5% removal post-letterbox removal)
        percentage_y (real):
        '''
        npimg = itk.array_from_image(ref)
        nz_cols = np.nonzero(np.amax(npimg,axis=0))[0] # only 1 dimension returned
        nz_sc = nz_cols[0]
        nz_ec = nz_cols[-1]
        padc = round((nz_ec-nz_sc)*percentage_x)
        
        nz_rows = np.nonzero(np.amax(npimg,axis=1))[0]
        nz_sr = nz_rows[0]
        nz_er = nz_rows[-1]
        padr = round((nz_er-nz_sr)*percentage_y)
    
        
        self.crop_region = itk.ImageRegion[2]()
        self.crop_region.SetIndex(0, int(nz_sc+padc))
        self.crop_region.SetIndex(1, int(nz_sr+padr))

        self.crop_region.SetSize(0, int(nz_ec-nz_sc -2*padc + 1))
        self.crop_region.SetSize(1, int(nz_er-nz_sr -2*padr + 1))
        
        # it's confusing to me whether to use this or RegionOfInterestFilter
        # we'll see what the index consequences are when bouncing between this and numpy arrays
        # annotations will be specified in indices of the uncropped image
        # so how do we deal with that?
        
        # OK, ExtractImage filter and the index being weird is too difficult to track.  I'll normalize later.
        self.filter = itk.ExtractImageFilter[ImageType, ImageType].New(ExtractionRegion=self.crop_region)
        #self.filter = itk.RegionOfInterestImageFilter[ImageType, ImageType].New(RegionOfInterest=self.crop_region)
        
    def crop(self, image):
        '''
        Maintains image origin and spacing but will create a new image with a different LargestPossibleRegion covering the
        cropped region.
        '''
        self.filter.SetInput(image)
        self.filter.Update()
        return self.filter.GetOutput()
    
def image_from_array(array, ref):
    '''
    TODO maybe deprecate this once xarray support has stabilized.
    Returns an ITK image from a numpy array with the correct metadata.
    
    TODO consider this with indexing (i.e. the extractimagefilter effect on ImageRegion.Index)
    This doesn't work that great.  The index of the region isn't maintained.
    
    Parameters
    ----------
    array (np.ndarray): array to convert
    ref (itk.Image): reference image to copy spacing and coordinates from
    '''
    ans = itk.image_from_array(array)
    ans.SetOrigin(ref.GetOrigin())
    ans.SetSpacing(ref.GetSpacing())
    ans.SetDirection(ref.GetDirection())
    ans.SetLargestPossibleRegion(ref.GetLargestPossibleRegion())
    return ans
    
def transform_to_physical(indices, image):
    '''
    Transform [y,x] indices to physical locations in an image.  Note, this is not the same as ITK's index scheme.
    '''
    start_index = np.asarray(image.GetLargestPossibleRegion().GetIndex())
    return np.apply_along_axis(lambda x: np.array(image.TransformIndexToPhysicalPoint(wrap_itk_index(x))), 1, np.fliplr(indices) + start_index[np.newaxis,:])

def transform_to_indices(pts, image):
    # TODO find usage of this and figure out if I want to get rid of it
    '''
    Transform ITK's physical locations to [y,x] indices.  Note, this is not the same as ITK's index scheme.
    '''
    start_index = np.asarray(image.GetLargestPossibleRegion().GetIndex())
    return np.fliplr(np.apply_along_axis(lambda x: np.array(image.TransformPhysicalPointToIndex(wrap_itk_point(x))), 1, pts) - start_index[np.newaxis,:])

def normalize_angle(t):
    '''
    Transform t to [0, 2pi]
    '''
    ans = t
    if np.abs(ans) > 2*np.pi:
        ans = np.fmod(ans, 2*np.pi)
    if t < 0:
        ans = 2*np.pi + ans
    return ans

def wrap_itk_index(x):
        idx = itk.Index[2]()
        idx.SetElement(0, int(x[0]))
        idx.SetElement(1, int(x[1]))
        return idx

def wrap_itk_point(x):
    # TODO, why itk.F?
    pt = itk.Point[itk.F,2]()
    pt.SetElement(0, x[0])
    pt.SetElement(1, x[1])
    return pt

def arclength(theta1, theta2, a, b):
    '''
    Returns arc length of ellipse perimeter from theta1 to theta2.  To get the length in the other direction reverse the assignments of theta1 and theta2.
    
    Parameters
    ---------
    theta1 (float):
    theta2 (float):
    a: major axis (1/2 ellipse width)
    b: minor axis (1/2 ellipse height)
    
    Returns
    -------
    arc length (float):  i.e. Integral[theta2, theta1]
    '''
    #assert not reverse, "reverse=True not implemented"
    assert 0 <= theta1 and theta1 <= 2*np.pi and 0 <= theta2 and theta2 <= 2*np.pi, "0 <= theta1, theta2, <= 2*np.pi"
    
    def foo(theta, a, b):
        return a*np.sqrt(1 - (1 - (b/a)**2)*np.sin(theta)**2)

    if theta1 == theta2:
        return 0
    elif theta1 < theta2:        
        return integrate.quad(foo, theta1, theta2, args=(a, b))[0]
    else: # theta1 > theta2
        return integrate.quad(foo, theta1, 2*np.pi, args=(a,b))[0] + integrate.quad(foo, 0, theta2, args=(a,b))[0]
    
        

def linspace_ellipse(model, theta1, theta2, arc_step, reverse=False):
    '''
    Returns regularly sampled thetas from theta1 to theta2 with an arc_step arc length spacing between them.
    
    Parameters
    ----------
    model (EllipseModel): really only need a, b from the model
    theta1 (float):
    theta2 (float):
    arc_step (float): distance between points (> 0)
    
    Returns
    -------
    (np.array) : [theta1, x2, x3, ..., xn] where arclength(xn, theta2) < arc_step and 0 <= xi <= 2PI
    '''
    
    assert arc_step > 0
    
    a, b = model.params[2:4]
    thetac = theta1
    ans = []
    while np.abs(thetac - theta2) > arc_step:
        ans.append(thetac)
        thetan = normalize_angle(scipy.optimize.minimize_scalar(lambda x: (arclength(thetac, normalize_angle(x), a, b) - arc_step)**2, method='Brent').x)
        thetac = thetan
        #print(thetac)
        
    return np.array(ans)

def nearest_angle(x, model):
    '''
    Given a point returns the parametric angle of the nearest point on the ellipse.
    '''
    return normalize_angle(scipy.optimize.minimize_scalar(lambda t: np.linalg.norm(model.predict_xy(np.array([t])) - x), method='Brent').x)

def is_inside(pt, image):
    idx = image.TransformPhysicalPointToIndex(pt)
    return image.GetLargestPossibleRegion().IsInside(idx)
    
def resample_by_grid(grid, image):
    '''
    grid is a MxNx2
    '''
    interp = itk.LinearInterpolateImageFunction[ImageType, itk.D].New()
    interp.SetInputImage(image)
    ans = np.zeros([grid.shape[0], grid.shape[1]], dtype='float32')
    good = np.repeat(True, grid.shape[1])
    for c in range(ans.shape[1]):
        for r in range(ans.shape[0]):
            if is_inside(grid[r,c], image):
                pt = itk.Point[itk.D, 2](grid[r, c])
                ans[r, c] = interp.Evaluate(pt)
            else:
                good[c] = False
                break
    return ans[:,good], grid[:,good]
# inliers
# 
# inliers appear to be presorted by x
# cut-off a few points on either end in case an outlier snuck in


def fig_to_array(fig):
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def create_eye_figure(img, eye):
    model = eye.ellipse_model
    inliers = eye.eyesocket_points
    filtered_pts = eye.nerve_search_points 
    seed_pt = eye.nerve_seed_point
    
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box') # make sure this is displayed correctly
    plt.imshow(img, cmap='gray')
    img_xlim = plt.xlim()
    img_ylim = plt.ylim()
    if model is not None:
        ts = np.linspace(0, 2*np.pi, num=1000)
        pts2 = transform_to_indices(model.predict_xy(ts), img)
#         ax.fill(pts2[::,1], pts2[::,0], facecolor='red', alpha=0.2, zorder=2)

        # print inliers
        pts3 = transform_to_indices(inliers, img)
        ax.scatter(pts3[:,1], pts3[:,0], color='pink', alpha=0.8)
        
        if filtered_pts is not None:
            tmp = transform_to_indices(np.array([filtered_pts[:,:,0].flatten(), filtered_pts[:,:,1].flatten()]).T, img)
            ax.scatter(tmp[:,1], tmp[:,0], color='green', alpha=0.01)

        #tmp4 = transform_to_indices(inliers, img)
        if seed_pt is not None:
            seed_idx = transform_to_indices(seed_pt[np.newaxis,:], img).flatten()
            ax.scatter(seed_idx[1], seed_idx[0], color='blue') 
    
    plt.xlim(img_xlim)
    plt.ylim(img_ylim)
    return fig

def overlay(img, objects, colors, alphas=None):
    '''
    Creates an overlay of objecs on top of img.  Sequential blending of pixels according to color and alpha value.
    
    Parameters
    ----------
    img (numpy 2D array): normalized pixel values 0 to 1
    objects (list of numpy 2D arrays): overlay labels the same size of img.  Non-zero values will be overlayed.
    colors (list of numpy 3-element arrays): RGB colors, each element 0 to 1
    alphas (list of alpha values or None=1):
    '''
    
#     imgrgb = np.stack((img, img, img), axis=2)
#     imgones = np.ones(imgrgb.shape)
#     imgzeros = np.zeros(imgrgb.shape)
    alphas = alphas if alphas is not None else np.ones(len(colors))

    imgflat = img.flatten()
    ansflat = np.stack((imgflat, imgflat, imgflat), axis=1)
    
    for i in range(len(objects)):
        obj = objects[i].flatten() > 0
        c = colors[i] 
        
        overlay = np.zeros((obj.shape[0], 3))
        overlay[obj,:] = c
        
        ansflat[obj,:] = (1-alphas[i])*ansflat[obj,:] + alphas[i]*overlay[obj,:]
    
    return np.stack((ansflat[:,0].reshape(img.shape), ansflat[:,1].reshape(img.shape), ansflat[:,2].reshape(img.shape)), axis=2) # probably could do this with one reshape
        
        
    
    # combining colors = 0 0 1 and 1 0 0 to .5 0 .5
    # what operation is that?
    # just add then divide by norm?
    # flatten and split out channels?
    # is it additive or subtractive?
    # image is pure white [1 1 1] and overlay is red [1 0 0]
    # now what?  now i've blended and want [0.5 0 0.5], now what?
    # make 1 default and add to zeros, unmasked pixels are set to white at the end?
    # what about black?  no?  does the mask have to add to 1?
    
#     overr
#     overg
#     overb
    
    
    # ok, we are shifting the original image from white to color, so labeling is just a shift on the rgb
    # so, the label color sum = 1
    
    # maybe calculate intensity first, then weight by colors, then multiple by masks, then set pixels in image
    

def create_nerve_figure(img, nerve):
    myimg =  overlay(itk.array_from_image(img2), [itk.array_from_image(nerve.nerve_mask), itk.array_from_image(nerve.skeleton_image)], [np.array([0, 1, 1]), np.array([1, 0, 1])], [0.5, 1])
    return plt.imshow(myimg)
#     model = eye.ellipse_model
#     inliers = eye.eyesocket_points
#     filtered_pts = eye.nerve_search_points 
#     seed_pt = eye.nerve_seed_point
    
#     fig, ax = plt.subplots()
#     ax.set_aspect('equal', 'box') # make sure this is displayed correctly
#     plt.imshow(img)
#     img_xlim = plt.xlim()
#     img_ylim = plt.ylim()
#     if model is not None:
#         ts = np.linspace(0, 2*np.pi, num=1000)
#         pts2 = transform_to_indices(model.predict_xy(ts), img)
#         ax.fill(pts2[::,1], pts2[::,0], facecolor='red', alpha=0.2, zorder=2)

#         # print inliers
#         pts3 = transform_to_indices(inliers, img)
#         ax.scatter(pts3[:,1], pts3[:,0], color='pink', alpha=0.8)
        
#         if filtered_pts is not None:
#             tmp = transform_to_indices(np.array([filtered_pts[:,:,0].flatten(), filtered_pts[:,:,1].flatten()]).T, img)
#             ax.scatter(tmp[:,1], tmp[:,0], color='green', alpha=0.01)

#         #tmp4 = transform_to_indices(inliers, img)
#         if seed_pt is not None:
#             seed_idx = transform_to_indices(seed_pt[np.newaxis,:], img).flatten()
#             ax.scatter(seed_idx[1], seed_idx[0], color='blue') 
    
#     plt.xlim(img_xlim)
#     plt.ylim(img_ylim)
#     return fig

class VideoReader(ABC):
    @abstractmethod
    def get_next(self):
        pass
    def at_end(self):
        pass
    
def load_results(fp):
    '''
    Loads the entirety of the results in directory, fp.
    '''
    i = 0
    prefix = fp + '/' + str(i)
    img_path = prefix + '.mha'
    results = []
    while os.path.exists(img_path):
        img = itk.imread(img_path)
        eye = EyeSegmentationRANSAC.Eye.load(prefix)
        nerve = NerveSegmentationSkeleton.Nerve.load(prefix)
        with open(prefix + '-duration.p', 'rb') as f:
            duration = pickle.load(f)
        
        results.append((img, eye, nerve, duration))
        
        i += 1
        prefix = fp + '/' + str(i)
        img_path = prefix + '.mha'
    return results
    
def write_result(fp, i, img, eye, nerve, duration):
    os.makedirs(fp, exist_ok=True)

    if img is not None:
        itk.imwrite(img, fp + '/' + str(i) + '.mha')
    if eye is not None:
        eye.save(fp + '/' + str(i))
    if nerve is not None:
        nerve.save(fp + '/' + str(i))
    with open(fp + '/' + str(i) + '-duration.p', 'wb') as f:
        pickle.dump(duration, f)    


def estimate_width(results, lower_perc=0.02, upper_perc=0.1):
    nerve_plot = np.array([[np.median(r[2].nerve_width[:,1]), np.median(np.abs(r[2].edge_values.flatten()))] for r in results if r[2] is not None and r[2].nerve_width is not None])
    lower = round(nerve_plot.shape[0] * lower_perc)
    upper = round(nerve_plot.shape[0] * upper_perc)
    sort_idx = np.flip(nerve_plot[:,1].argsort())
    good_pts = nerve_plot[sort_idx[lower:upper], :]
#     plt.scatter(good_pts[:,0], good_pts[:,1])
    estimate = np.mean(good_pts[:,0])
    return estimate
        
class ClariusOfflineReader(VideoReader):
    # TODO, get rid of this hardcoded stuff
    EYE_WIDTH_MM = 24.2
    EYE_HEIGHT_MM = 23.7
    CLARIUS_EYE_WIDTH = 400
    
    
    def __init__(self, filepath, probe_width=37.57):
        self.video = (skvideo.io.vread(filepath)[::,::,::,1] / 255).astype('float32')
        
        npimg = np.amax(self.video, axis=0)
        nz_cols = np.nonzero(np.amax(npimg,axis=0))[0] # only 1 dimension returned
        nz_sc = nz_cols[0]
        nz_ec = nz_cols[-1]
        s = probe_width/(nz_ec-nz_sc)
        self.spacing = [s, s]
        self.current = 0
    
    def get_next(self):
        image = self.get(self.current)
        self.current += 1
        return image
    
    def get(self, i):
        image = itk.image_from_array(self.video[i])
        self.set_clarius_dimension(image)
        return image
    
    def at_end(self):
        return self.current >= self.video.shape[0]
    
    def set_clarius_dimension(self, img):
        '''
        Hard-coded estimate for the CLARIUS spacing.  TODO: get this somehow from Clarius.
        '''
        img.SetSpacing(self.spacing)

class ONSDFrame:
    def __init__(self, image, preprocess, eye_segmentation, nerve_segmentation):
        self.image = image
        self.eye_segmentation = eye_segmentation
        self.nerve_segmentation = nerve_segmentation
        # etc etc etc
    def process(self):
        self.eye = None
        self.nerve = None
        
        self.input_image = self.preprocess(self.image)
        self.eye = self.eye_segmentation.process(self.input_image)
        if self.eye.nerve_point is not None:
            self.nerve = self.nerve_segmentation.process(self.input_image, self.eye)

class EyeSegmentationRANSAC:
    class Eye:
        OBJECT_SUFFIX = '-eye.p'
        def __init__(self, ellipse_model, eyesocket_points, nerve_search_points, nerve_seed_point):
            self.ellipse_model = ellipse_model
            self.eyesocket_points = eyesocket_points
            self.nerve_search_points = nerve_search_points
            self.nerve_seed_point = nerve_seed_point
        def save(self, prefix):
            # pickle object
            with open(prefix + self.OBJECT_SUFFIX, 'wb') as f:
                pickle.dump(self, f)
        def found(self):
            return self.nerve_seed_point is not None 
        @classmethod
        def load(cls, prefix):
            if not path.exists(prefix + cls.OBJECT_SUFFIX):
                return None
            
            with open(prefix + cls.OBJECT_SUFFIX, 'rb') as f:
                ans = pickle.load(f)
                
            return ans
            # load pickle
            # load nerve_image
            
    def __init__(self, blur_sigma=[1,1], downscale=[6,6], canny_threshold=[0.03, 0.06], edge_angle_threshold=[np.pi + .176, 2*np.pi - .176], ransac_min_samples=5, ransac_residual_threshold=1, ransac_max_trials=200, ellipse_width_threshold=[8.47, 21.78], ellipse_height_threshold=[8.295, 21.33], peak_sigma=10):
        self.blur_sigma = blur_sigma
        self.downscale = downscale
        self.canny_threshold = canny_threshold
        self.edge_angle_threshold=edge_angle_threshold
        self.ransac_min_samples=ransac_min_samples
        self.ransac_residual_threshold = ransac_residual_threshold
        self.ransac_max_trials=200
        self.ellipse_width_threshold=ellipse_width_threshold
        self.ellipse_height_threshold=ellipse_height_threshold
        self.peak_sigma = peak_sigma
    
    def load_eye(self, prefix):
        return self.Eye.load(prefix)
    
    def _good_eye(self, model, data):
    # TODO: also, what is height and what is width (a and b) might be arbitrary in the EllipseModel
        xc, yc, a, b, theta = model.params
        ans1 = self.ellipse_width_threshold[0] < a and a < self.ellipse_width_threshold[1] and self.ellipse_height_threshold[0] < b and b < self.ellipse_height_threshold[1]
        return ans1
    
    def _fit_ellipse(self, input_image):
#         crp = vid[x, crp_pad:-crp_pad, crp_pad:-crp_pad]
#         img = itk.image_from_array(crp.copy()) # using copy because i think ITK has a bug with the sliced version
#         set_clarius_dimension(img)
#         downscale = [6, 6]
#         sigma = [1, 1]
#         print("Here1: ", input_image.GetSpacing()) # TODO: REMOVE
        blur_filter = itk.SmoothingRecursiveGaussianImageFilter[ImageType, ImageType].New(SigmaArray=self.blur_sigma, Input=input_image)
#                                                                                           Input=itk.image_duplicator(input_image))
        blur_filter.Update()
#         print("Here2: ", input_image.GetSpacing())
        shrink_filter = itk.ShrinkImageFilter[ImageType, ImageType].New(ShrinkFactors=self.downscale, Input=blur_filter.GetOutput())
        shrink_filter.Update()
#         print("Here3: ", input_image.GetSpacing())
        canny_filter = itk.CannyEdgeDetectionImageFilter[ImageType, ImageType].New(Input=shrink_filter.GetOutput(), LowerThreshold=self.canny_threshold[0], UpperThreshold=self.canny_threshold[1])
        canny_filter.Update()
#         print("Here4: ", input_image.GetSpacing())
        grad_filter = itk.GradientImageFilter[ImageType, itk.F, itk.F].New(Input=shrink_filter.GetOutput())
        grad_filter.Update()
#         print("Here5: ", input_image.GetSpacing())
        #TODO: ADD CHECKS THAT THESE AREN'T EMPTY ARRAYS (i.e. no points matching criteria, then return None model)
        indices = np.argwhere(itk.array_from_image(canny_filter.GetOutput()) > 0)
        if indices.shape[0] == 0:
            return None, None

        #print(indices.shape)
        #pts = transform_to_physical(indices)
        grad = -itk.array_from_image(grad_filter.GetOutput()) # negate so eye gradients point inward
        edge_grad = grad[indices[:,0], indices[:,1],:]
        grad_angles = np.arctan2(edge_grad[:,1], edge_grad[:,0])
        tmp_idx = grad_angles < 0
        grad_angles[tmp_idx] = 2*np.pi + grad_angles[tmp_idx]
        filter_idx = (self.edge_angle_threshold[0] < grad_angles) & (grad_angles < self.edge_angle_threshold[1])
        filter_indices = indices[filter_idx, :]
        if filter_indices.shape[0] == 0:
            return None, None
        filter_pts = transform_to_physical(filter_indices, canny_filter.GetOutput())

        model, inliers = skimage.measure.ransac(filter_pts, skimage.measure.EllipseModel, self.ransac_min_samples, self.ransac_residual_threshold, is_model_valid=self._good_eye, max_trials=self.ransac_max_trials)
        return model, filter_pts[inliers,:]

    def _find_seed(self, img, filtered_pts, peak):
        seed_pt = filtered_pts[math.floor(filtered_pts.shape[0]/2), peak].flatten()
        return seed_pt

    def _find_peak(self, npimg):
        npimg1d = np.sum(npimg, axis=0)
        npimg1d = npimg1d / np.max(npimg1d)
        npimg1d = scipy.ndimage.filters.gaussian_filter(npimg1d, self.peak_sigma)
        peaks, properties = scipy.signal.find_peaks(1-npimg1d, prominence=.1)
        if (len(peaks) >= 1):
            return peaks[np.argmax(properties['prominences'])]
        else:
            return None

    def _find_nerve_search(self, image, model, inliers):
        # TODO pull out these parameters
        cut = 3
        search_start = 3 # 1mm
        search_thick = 5 # 3mm
        search_spacing = np.array([.1, .1]) # also in mm

    # get an ellipse larger (a little further from the boundary of model)
        e1 = skimage.measure.EllipseModel()
        e1.params = model.params + np.array([0, 0, search_start, search_start, 0])
        e2 = skimage.measure.EllipseModel()
        e2.params = e1.params + np.array([0, 0, search_thick, search_thick, 0])

    # get rightmost and leftmost points on model's perimeter
        inliers = inliers[np.argsort(inliers[:,0]),:]
        pt1 = inliers[cut,:]
        pt2 = inliers[-(cut+1),:]

    # find equally spaced arcs along the outer ellipse, because we don't know how the optimizer rotated and stretched the best-fitting ellipse
    # we don't know what half of the ellipse (defined by theta1 and theta2) is the bottom part.  So, we'll calculate both arcs and pick the
    # one with the lowest point
        theta1 = nearest_angle(pt1, model)
        theta2 = nearest_angle(pt2, model) # these angles still work on the larger ellipses cuz of uniform scaling (i think)
        ts1 = linspace_ellipse(e2, theta1, theta2, search_spacing[0]) # .1mm steps along the outer ellipse
        miny1 = np.min(e2.predict_xy(ts1)[:,1])
        ts2 = np.flip(linspace_ellipse(e2, theta2, theta1, search_spacing[0]))
        miny2 = np.min(e2.predict_xy(ts2)[:,1])
        ts = ts1 if miny1 > miny2 else ts2

        ss = np.arange(0, search_thick, search_spacing[1]) / search_thick # no longer in mm, in proportion of search_thick

        search_img = np.zeros([len(ss), len(ts)])
        search_physical_pts = np.zeros([search_img.shape[0], search_img.shape[1], 2])

        ds = []
        x1s = []
        for i in range(len(ts)):
            t = ts[i]
            x1 = e1.predict_xy(t)
            x2 = e2.predict_xy(t)
            d = x2 - x1

            x1s.append(x1)
            ds.append(d) # comment this out

            pts = np.array([x1[0] + d[0] * ss, x1[1] + d[1] * ss]).T
            search_physical_pts[:,i,:] = pts

        return resample_by_grid(search_physical_pts, image)
    
    def process(self, input_image):
        peak = None
        filtered_pts = None
        seed_idx = None
        seed_pt = None
#         t3 = datetime.now()
        model, inliers = self._fit_ellipse(input_image)
        if model is not None and inliers.shape[0] > 6: #TODO - change this to a function of cut in find_nerve_search
            npimg, filtered_pts = self._find_nerve_search(input_image, model, inliers)
            peak = self._find_peak(npimg)
            if peak is not None:
                seed_pt = self._find_seed(input_image, filtered_pts, peak)
#         t4 = datetime.now()
#        frame_times.append(t4-t3)
#         fig = create_eye_figure(input_image, model, inliers, filtered_pts, seed_pt)
        return EyeSegmentationRANSAC.Eye(model, inliers, filtered_pts, seed_pt)
        #return fig
#         return eye



class ONSDVideo:
    # todo, make this throw an event when a nerve is reliably found
    def __init__(self, reader):
        pass
    def get_next(self):
        pass
    def at_end(self):
        pass

def resample_by_grid_point(grid, image):
    '''
    grid is a MxNx2
    '''
    img_size = np.asarray(image.GetLargestPossibleRegion().GetSize())
    interp = itk.NearestNeighborInterpolateImageFunction[ImageType, itk.D].New()
    interp.SetInputImage(image)
    ans = np.zeros([grid.shape[0], grid.shape[1]], dtype='float32')
    for c in range(ans.shape[1]):
        for r in range(ans.shape[0]):
            # constructor ought to work, bet this is due to lack of bounds checking
            pt = itk.Point[itk.D, 2](grid[r,c])
            if image.GetLargestPossibleRegion().IsInside(image.TransformPhysicalPointToIndex(pt)):
            #if (0 <= grid[r, c, 0] and grid[r, c, 0] < img_size[0]) and (0 <= grid[r, c, 1] and grid[r, c, 1] < img_size[1]):
                ans[r, c] = interp.Evaluate(pt)
    return ans

class Curve:
    def __init__(self, nodes, downsample=4, extrapolatesample=2):
        '''
        downsample is how to downsample nodes, extrapolatesample is the number of nodes at the end and beginning of the Curve to average derivatives for extrapolation
        '''
        self.nodes = nodes
        self.downsample=downsample
        self.extrapolatesample=extrapolatesample
        
        # guh, there's probably a better way to do the subsampling, this will add random asymmetry in the sampling distance
        self.spline_points = np.array([x.point for x in self.nodes[::downsample]])
        if not np.array_equal(self.spline_points[-1,:], np.array(self.nodes[-1].point)): # make sure we keep 0 and last points no matter downsample
            self.spline_points = np.vstack([self.spline_points, self.nodes[-1].point])
        
        # diffs is difference vectors between points
        # norms is the magnitudes of those vectors
        # want to average the diffs at the beginning and end of that curve, then just use that 1st derivative vector as our extrapolation vector
        diffs = np.vstack([[0,0], np.diff(self.spline_points, axis=0)])
        norms = np.linalg.norm(diffs, axis=1)
        norm_diffs = diffs / norms[:,np.newaxis]
        self.deriv1_1 = np.mean(-norm_diffs[1:(extrapolatesample+1),:], axis=0)
        self.deriv1_2 = np.mean(norm_diffs[(-extrapolatesample)::,:], axis=0)
        
        self.spline_t = np.cumsum(norms)
        self.length = self.spline_t[-1]
        self.spline = scipy.interpolate.CubicSpline(self.spline_t, self.spline_points)
        
    def reverse(self):
        '''
        Return a new reversed curve.  May affect spline near ends of curve, flips handedness of the derivative, flips order of vertices.
        '''
        return Curve(self.nodes[::-1], self.downsample, self.extrapolatesample)
    
    def evaluate(self, t):
        #if t.ndim == 1:
        #    t = t[:,np.newaxis]
        ans = np.zeros([len(t), 2])
        
        idx = (0 <= t) & (t <= self.length)
        ans[idx,:] = self.spline(t[idx])
   
        idx = (t < 0)
        ans[idx,:] = (self.deriv1_1[:,np.newaxis]*(-t[idx]) + self.spline_points[0,np.newaxis].T).T
        
        idx = (t > self.length)
        ans[idx,:] = (self.deriv1_2[:,np.newaxis]*(t[idx]-self.length) + self.spline_points[-1,np.newaxis].T).T
        return ans

    def normal(self, t):
        ans = np.zeros([len(t), 2])

        idx = (0 <= t) & (t <= self.length)
        tmp = self.spline.derivative(1)(t[idx])
        ans[idx,0] = tmp[:,1]
        ans[idx,1] = -tmp[:,0]
        idx = (t < 0)
        ans[idx,:] = np.array([-self.deriv1_1[1], self.deriv1_1[0]])
        
        idx = (t > self.length)
        ans[idx,:] = np.array([self.deriv1_2[1], -self.deriv1_2[0]])
        
        return ans
    
    @property
    def vertices(self):
        return (self.nodes[0], self.nodes[-1])

class Node:
    '''
    
    '''
    def __init__(self, index, point, connectivity=None, neighbors=None):
        self.index = index
        self.point = point
        #self.connectivity = connectivity
        self.neighbors = neighbors if neighbors is not None else set()
        self.is_vertex = False
        
    def __eq__(self, other):
        return self.index[0] == other.index[0] and self.index[1] == other.index[1]
    
    @property
    def connectivity(self):
        return len(self.neighbors)
    
    def get_other_neighbors(self, source):
        return [x for x in self.neighbors if x != source]
    
    def __hash__(self):
        return id(self)
    
    def __str__(self):
        return '({},{}), connectivity: {}, neighbors: {}'.format(self.index[0], self.index[1], self.connectivity, len(self.neighbors))    

    
class CurveGraphFactory:
    R1_NEIGHBORHOOD_CIRC = np.array([[-1, 0, 1, 1, 1, 0, -1, -1, -1], [-1, -1, -1, 0, 1, 1, 1, 0, -1]]).T # clockwise starting upper left and overlap with begin == end
    R1_NEIGHBORHOOD = np.array([[-1, 0, 1, 1, 1, 0, -1, -1], [-1, -1, -1, 0, 1, 1, 1, 0]]).T # clockwise starting upper left
    VISITED = np.array([[-1, -1, 0, 1], [0, -1, -1, -1]]).T
    CORNER1 = (np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0]]), np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]]))
    CORNER2 = (np.array([[0, 0, 0], [0, 1, 1], [0, 1, 0]]), np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]]))
    CORNER3 = (np.array([[0, 0, 0], [1, 1, 0], [0, 1, 0]]), np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]]))
    CORNER4 = (np.array([[0, 1, 0], [1, 1, 0], [0, 0, 0]]), np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1]]))
    
    def __init__(self, itkimage):
        self.nodes = dict()
        self.itkimage = itkimage
        self.image = itk.array_from_image(itkimage)
        
        # cheap way of handling border, TODO: REMOVE
        self.image[0,:] = 0
        self.image[:,0] = 0
        self.image[self.image.shape[0]-1,:] = 0
        self.image[:,self.image.shape[1]-1] = 0
        
        self.to_connect8(self.image) # preprocess the binary thinned image
        
        self.pass1(self.image) # create nodes with connected neighbors
        
        self.merge_adjacent_junctions() # combine adjacent junctions to avoid weird edge cases
        
        # not sure if the binary thinning allows this, but we can't do a one-point curve
        salt = {n for n in self.nodes.values() if n.connectivity == 0}
        for n in salt:
            self.nodes.pop((n.index[0], n.index[1]))
        
        edges = {n for n in self.nodes.values() if n.connectivity == 2}
        vertices = {n for n in self.nodes.values() if n.connectivity != 2}
        self.pass2(edges, vertices) # break loops, trace from all end points and junctions, create curves

    def set_connectivity(self, image, n):
        #y = CurveGraphFactory.R1_NEIGHBORHOOD + n.index
        #n.connectivity = np.sum(image[y[:,1], y[:,0]])
        
        z = CurveGraphFactory.VISITED + n.index
        q = image[z[:,1], z[:,0]] > 0
        r = z[q,:]
        for i in range(r.shape[0]):
            k = self.nodes[(r[i,0], r[i,1])]
            n.neighbors.add(k)
            k.neighbors.add(n)
        
#     def set_connectivity4(self, image, n):
#         '''
#         This is an untested connectivity measure for 4-connected
#         '''
#         y = CurveGraphFactory.R1_NEIGHBORHOOD_CIRC + n.index
#         n.connectivity = np.sum(np.abs(np.diff(image[y[:,1], y[:,0]])))/2

#         z = CurveGraphFactory.VISITED + n.index
#         q = image[z[:,1], z[:,0]] > 0
#         r = z[q,:]
#         for i in range(r.shape[0]):
#             k = self.nodes[(r[i,0], r[i,1])]
#             n.neighbors.add(k)
#             k.neighbors.add(n)
            
    
    def to_connect8(self, image):
        self.image = image
        self.image = self.image - binary_hit_or_miss(self.image, CurveGraphFactory.CORNER1[0], CurveGraphFactory.CORNER1[1])
        self.image = self.image - binary_hit_or_miss(self.image, CurveGraphFactory.CORNER2[0], CurveGraphFactory.CORNER2[1])
        self.image = self.image - binary_hit_or_miss(self.image, CurveGraphFactory.CORNER3[0], CurveGraphFactory.CORNER3[1])
        self.image = self.image - binary_hit_or_miss(self.image, CurveGraphFactory.CORNER4[0], CurveGraphFactory.CORNER4[1])

    def pass1(self, image):
        '''
        Computes nodes and their connectivity (first guess)
        '''
        for y in range(1, image.shape[0]-1):
            for x in range(1, image.shape[1]-1):
                if image[y, x] > 0:
                    index = np.array([x, y]) # i think this is the only place where the index is read
#                     itkindex = itk.Index[2]()
#                     itkindex.SetElement(0, int(index[0]))
#                     itkindex.SetElement(1, int(index[1]))
                    n = Node(index, transform_to_physical(index[np.newaxis,::-1], self.itkimage).flatten())
                    self.nodes[(x, y)] = n
                    self.set_connectivity(image, n)

    def pass2(self, edges, vertices):
        self.curvegraph = CurveGraph()
        while len(vertices) > 0:
            v = vertices.pop()
            for x2 in v.neighbors:
                if x2 in edges: # important check, for example, junction with a loop, or the other end of the curve has already been visited
                    nodes = [v]
                    x1 = v
                    while x2.connectivity == 2 and x2 != v:
                        edges.remove(x2)
                        nodes.append(x2)
                        tmp = x2.get_other_neighbors(x1)[0] # know there's only 1 cuz of connectivity constraint
                        x1 = x2
                        x2 = tmp
                    if x2 != v:
                        nodes.append(x2)
                    else: # x2 == v, we have a cycle and we'll break it
                        x1.neighbors.remove(x2)
                    curve = Curve(nodes)
                    self.curvegraph.add(curve)
        if len(edges) > 0: # we have a free loop, break it
            e1 = edges.pop()
            e2 = e1.neighbors.pop() # get arbitraty neighbor and remove e1's connection
            e2.neighbors.remove(e1) # remove e2's link to e1
            vertices.add(e1)
            vertices.add(e2)
            self.pass2(edges, vertices) # in case we have more than one loop
            
        
        
    def merge_adjacent_junctions(self):
        '''
        Merge any adjacent junctions into one junction.
        
        Imagine a T-junction.  Left, right, and middle nodes all think they are 3-junctions due to 8-connectedness.  Merge these into one junction.
        '''
        merged = True
        while merged:
            merged = False
            junctions = {n for n in self.nodes.values() if n.connectivity > 2}
            for j in junctions:
                for n in j.neighbors:
                    if n.connectivity > 2:
                        self.combine_with(n, j)
                        merged = True
                        break
    
                    
        
    def combine_with(self, target, source):
        '''
        Connects all of source's neighbors with target and then removes source from the self.nodes
        '''
        for n in source.neighbors:
            n.neighbors.remove(source)
            if n != target:
                n.neighbors.add(target)
                target.neighbors.add(n)
        self.nodes.pop((source.index[0], source.index[1]))
                        
            
    # consider loops
    # consider junctions
    # blergh
#     def first_pass(self):
#         for i
#         return
    
#     def second_pass(self):
#         return
        
class CurveGraph:
    # TODO: get vertex by index or point
    # algorithm:
    # mark all vertices as curves or junctions
    # group all vertices in a neighborhood
    # mark lines that only have one connection or are connected to a junction as an endpoint
    def __init__(self, vertices=None, adjacency_list=None):
        self.vertices = dict() if vertices is None else vertices
        self.adjacency_list = defaultdict(dict)
        self.curves = []
    def add(self, curve):
        for i in range(2):
            v = curve.vertices[i]
            idx = (v.index[0], v.index[1])
            if idx not in self.vertices:
                self.vertices[idx] = v
            self.adjacency_list[idx][curve.vertices[i-1]] = curve # the i-1 let's me swap between first and last element
        self.curves.append(curve)
        
# need closest point search
def nearest_curve(curvegraph, model):
    '''
    Return the nearest curve in the correct orientation (closest vertex first) relative to eye model
    
    Parameters
    ----------
    curvegraph (ocularus.CurveGraph)
    model (skimage.measure.EllipseModel)
    '''
    min_dist = np.Inf
    min_curve = None
    for c in curvegraph.curves:
        v1 = np.asarray(c.vertices[0].point)
        v2 = np.asarray(c.vertices[1].point)
    
        d1 = np.linalg.norm(model.predict_xy(nearest_angle(v1, model)) - v1)
        d2 = np.linalg.norm(model.predict_xy(nearest_angle(v2, model)) - v2)
        if d1 < min_dist:
            min_dist = d1
            min_curve = c
        if d2 < min_dist:
            min_dist = d2
            min_curve = c.reverse()
    return (min_curve, min_dist)


class NerveSegmentationSkeleton:
    class Nerve:
        MASK_SUFFIX = '-nerve_mask.mha'
        IMAGE_SUFFIX = '-nerve_image.mha'
        SKELETON_IMAGE_SUFFIX = '-nerve_skeleton.mha'
        OBJECT_SUFFIX = '-nerve.p'
        
        def __init__(self, skeleton, skeleton_image, nerve_mask, nerve_image, nerve_width, edge_values):
            self.skeleton = skeleton
            self.skeleton_image = skeleton_image
            self.nerve_mask = nerve_mask
            self.nerve_image = nerve_image
            self.nerve_width = nerve_width
            self.edge_values = edge_values
        def save(self, prefix):
            # save images, set to None as they won't pickle
            tmp1 = None
            tmp2 = None 
            tmp3 = None
            if self.skeleton_image is not None:
                itk.imwrite(self.skeleton_image, prefix + self.SKELETON_IMAGE_SUFFIX)
            if self.nerve_mask is not None:
                itk.imwrite(self.nerve_mask, prefix + self.MASK_SUFFIX)
            if self.nerve_image is not None:
                itk.imwrite(self.nerve_image, prefix + self.IMAGE_SUFFIX)
            tmp1 = self.nerve_mask
            tmp2 = self.nerve_image
            tmp3 = self.skeleton_image
            self.nerve_mask = None
            self.nerve_image = None
            self.skeleton_image = None
            
            # pickle object
            with open(prefix + self.OBJECT_SUFFIX, 'wb') as f:
                pickle.dump(self, f)
            
            # reset images
            self.nerve_mask = tmp1
            self.nerve_image = tmp2
            self.skeleton_image = tmp3
        
        @classmethod
        def load(cls, prefix):
            if not path.exists(prefix + cls.OBJECT_SUFFIX):
                return None
            
            with open(prefix + cls.OBJECT_SUFFIX, 'rb') as f:
                ans = pickle.load(f)
            
            f0 = prefix + cls.SKELETON_IMAGE_SUFFIX
            if path.exists(f0):
                ans.skeleton_image = itk.imread(f0)
            
            f1 = prefix + cls.MASK_SUFFIX
            if path.exists(f1):
                ans.nerve_mask = itk.imread(f1)
            
            f2 = prefix + cls.IMAGE_SUFFIX
            if path.exists(f2):
                ans.nerve_image = itk.imread(f2)
                
            return ans
            # load pickle
            # load nerve_image
        
    def __init__(self, level=0, threshold=0.1, radius=1, sigma=1.5, erosion=10, nerve_offset=1, nerve_image_dimension=[6,12], nerve_image_sampling=[50,100], width_sigma=3):
        self.level = level
        self.threshold = threshold
        self.radius = radius
        self.sigma = sigma
        self.erosion = erosion
        self.nerve_offset = nerve_offset
        self.nerve_image_dimension = np.asarray(nerve_image_dimension)
        self.nerve_image_sampling = np.asarray(nerve_image_sampling)
        self.width_sigma = width_sigma # supposedly in nerve_image_sampling units
    
    def _watershed(self, img, ref_point, level, threshold, radius, sigma):
        smoothing = itk.MedianImageFilter[ImageType, ImageType].New(Radius=radius, Input=img)

        gradient = itk.GradientMagnitudeRecursiveGaussianImageFilter[ImageType, ImageType].New(Sigma=sigma, Input=smoothing.GetOutput())

        watershed = itk.WatershedImageFilter[ImageType].New( \
            Level=level, \
            Threshold=threshold, \
            Input=gradient.GetOutput())
        watershed.Update()

        cast1 = itk.CastImageFilter[itk.Image[itk.ULL,2],itk.Image[itk.UC,2]].New(Input=watershed.GetOutput())
        cast1.Update()
        tmp = cast1.GetOutput()
        #itk.cast_image_filter(watershed.GetOutput(), ttype=(itk.Image[itk.ULL,2], itk.Image[itk.UC,2]))
        LabelMapType = itk.LabelMap[itk.StatisticsLabelObject[itk.UL,2]]
        #print(type(watershed.GetOutput()))
        labelmap = itk.LabelImageToLabelMapFilter[itk.Image[itk.UC,2],LabelMapType].New(Input=tmp)
        
        labelselector = itk.LabelSelectionLabelMapFilter[LabelMapType].New(Input=labelmap, Label=tmp.GetPixel(tmp.TransformPhysicalPointToIndex(ref_point)))
        
        labelimage = itk.LabelMapToLabelImageFilter[LabelMapType, itk.Image[itk.UC,2]].New(Input=labelselector.GetOutput())
        labelimage.Update()
        return labelimage.GetOutput(), gradient.GetOutput(), labelselector

    def _map_nerve(self, input_image, eye):
        label, gradient, labelselector = self._watershed(input_image, eye.nerve_seed_point, self.level, self.threshold, self.radius, self.sigma)
#         nerve_labels[j] = itk.array_from_image(label)
#TODO: speedup by limiting convolution to bounding box around label
        erosion = itk.BinaryErodeImageFilter[itk.Image[itk.UC,2], itk.Image[itk.UC,2], itk.FlatStructuringElement[2]].New( \
            Input=label, \
            BoundaryToForeground=False, \
            Kernel=itk.FlatStructuringElement[2].Ball(self.erosion))
        erosion.Update()
        
        binary = itk.BinaryThinningImageFilter[itk.Image[itk.UC,2], itk.Image[itk.UC,2]].New(Input=erosion.GetOutput())
        binary.Update()
        
        cfg = CurveGraphFactory(binary.GetOutput())
        c, dist = nearest_curve(cfg.curvegraph, eye.ellipse_model)
        if c is None:
            nerve_image = None
        else:
            ts = np.linspace(-dist+self.nerve_offset,  self.nerve_image_dimension[0]-dist+self.nerve_offset, self.nerve_image_sampling[0]) # 6mm
            ys = c.evaluate(ts)
            zs = c.normal(ts)

            # resample everything, do the medial projection, wow numpy broadcasting!
            qs = np.linspace(-self.nerve_image_dimension[1]/2, self.nerve_image_dimension[1]/2, self.nerve_image_sampling[1])
            transform_pts = qs[:,np.newaxis,np.newaxis] * zs + ys[np.newaxis,:,:]

            nerve_image = itk.image_from_array(resample_by_grid_point(transform_pts, input_image))
            nerve_image.SetSpacing(self.nerve_image_dimension / self.nerve_image_sampling) # this is a bit false as this is really a distorted mesh
        return nerve_image, erosion.GetOutput(), cfg.curvegraph, binary.GetOutput()
    
    def _nerve_width(self, nerve_image):
        # OK, there are two ways of doing this:
        # could look for the first peak from the middle
        # or look for the max gradient (max peak)
        # max gradient is simpler (to code), this is what this is
        # TODO: don't really want this, I want only the magnitude, or the heck, the value, in y
#         gradmag = itk.GradientMagnitudeRecursiveGaussianImageFilter[ImageType, ImageType].New(InputImage=nerve_image, Sigma=self.width_sigma)
#         gradmag.Update()
#         blur = itk.RecursiveGaussianImageFilter[ImageType, ImageType].New(Input=nerve_image, sigma=self.width_sigma)
        blur = itk.MedianImageFilter[ImageType, ImageType].New(Input=nerve_image, Radius=self.width_sigma)
        
        grad = itk.GradientImageFilter[ImageType, itk.F, itk.F].New(Input=blur.GetOutput())
        grad.Update()
        plt.imshow(blur.GetOutput())
        npgrad = itk.array_from_image(grad.GetOutput())[:,:,1]
        mid = round(npgrad.shape[0]/2)
        top = mid - np.argmin(npgrad[0:mid,:], axis=0)
        bottom = np.argmax(npgrad[mid::, :], axis=0)
        
        xindices = np.arange(0, npgrad.shape[1])
        ydiffs = top + bottom
        spacing = np.asarray(nerve_image.GetSpacing())
        xpts = xindices * spacing[0]
        ypts = ydiffs * spacing[1]
        
#         values = np.array(npgrad[np.concatenate((-top+mid, bottom+mid)), np.concatenate((xindices, xindices))])
        values = np.array([npgrad[-top+mid, xindices], npgrad[bottom+mid, xindices]])
    
        return np.array([xpts, ypts]).T, values
    
    def process(self, input_image, eye):
        nerve_image, nerve_mask, skeleton, skeleton_image = self._map_nerve(input_image, eye)
        if nerve_image is None:
            nerve_width = None
            edge_values = None
        else:
            nerve_width, edge_values = self._nerve_width(nerve_image)
        
        return NerveSegmentationSkeleton.Nerve(skeleton, skeleton_image, nerve_mask, nerve_image, nerve_width, edge_values)

    def load_nerve(self, prefix):
        return NerveSegmentationSkeleton.Nerve.load(prefix)


