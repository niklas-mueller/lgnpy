
import cv2
import numpy as np
from scipy.ndimage.filters import convolve, correlate
from scipy.signal import convolve2d
# from scipy.ndimage import correlate
from scipy.interpolate import interp1d
from copy import deepcopy

def weibullNewtonHist(g, x, h):

    x_g = x ** g
    sum_x_g = np.sum(x_g*h)
    x_i = x_g / sum_x_g

    ln_x_i = np.log(x_i)

    _lambda = x_g * (np.log(x) * sum_x_g - np.sum(h * x_g * np.log(x)) ) / (sum_x_g ** 2)
    f = 1 + np.sum(ln_x_i * h) - np.sum(x_i * ln_x_i * h)
    f_prime = np.sum(_lambda * h * (sum_x_g / x_g - ln_x_i - 1))

    return f / f_prime

def weibullMleHist(ax, h):
    eps = 0.01 # precision of Newton-Raphson method's solution
    shape = 0.1 #initial value of gamma parameter

    h = h / np.sum(h)
    shape_next = shape - weibullNewtonHist(shape, ax, h)
    n_iteration = 0

    while np.abs(shape_next - shape) > eps:
        if np.isnan(shape_next) or np.isinf(shape_next) or shape_next > 20 or n_iteration > 30:
            break

        if shape_next <= 0:
            shape_next = 0.000001
            break

        shape = shape_next
        shape_next = shape - weibullNewtonHist(shape, ax, h)

        n_iteration += 1

    shape = shape_next
    scale = np.power(np.sum((np.power(ax, shape) * h)), 1/shape)



    return scale, shape

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])

    from https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python

    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def local_cov(e, sigma):
    break_off_sigma = 3.0
    filter_size = np.round(break_off_sigma * sigma)
    h = matlab_style_gauss2D((filter_size, filter_size), sigma)

    # term1 = convolve(e**2, h, mode='nearest')
    # term2 = convolve(e, h, mode='nearest')**2
    # term1 = correlate(e**2, h, mode='constant', origin=-1)
    # term2 = correlate(e, h, mode='constant', origin=-1)**2
    term1 = cv2.filter2D(e**2, -1, h)
    term2 = cv2.filter2D(e, -1, h)**2
    local_std = np.sqrt(np.max(term1 - term2, 0))

    # local_mean = convolve2d(e, h, mode='nearest') + np.finfo(float).tiny
    # local_mean = correlate(e, h, mode='constant', origin=-1) + np.finfo(float).tiny
    local_mean = cv2.filter2D(e, -1, h) + np.finfo(float).tiny

    return local_std / local_mean

def create_hist(data, h_bins):

    i_bin = np.array(range(h_bins))
    h = np.histogram(data, bins=i_bin)
    delta = (np.max(data) - np.min(data)) / h_bins
    ax = ((np.min(data) + i_bin * delta) + (np.min(data) + i_bin-1 * delta)) / 2

    ind = np.argwhere(h)
    h = h[ind]
    ax = ax[ind]

    return ax, h

def rgb2e(im):
    # convert RGB to e, el, ell

    R = im[:,:,0]
    G = im[:,:,1]
    B = im[:,:,2]

    # JMG: slightly different values than in PAMI 2001 paper;
    # simply assuming correctly white balanced camera
    E   = (0.3*R + 0.58*G + 0.11*B ) / 255.0
    El  = (0.25*R  + 0.25*G - 0.5*B ) / 255.0
    Ell = (0.5*R - 0.5*G) / 255.0

    # # As in original in PAMI 2001 paper
    # E   = (0.06*R + 0.63*G + 0.27*B ) / 255.0
    # El  = (0.3*R  + 0.04*G - 0.35*B ) / 255.0
    # Ell = (0.34*R - 0.6*G+0.17*B) / 255.0

    return E, El, Ell

def conv2padded(varargin):
    # CONV2PADDED  Two-dimensional convolution with padding.
    #    Y = CONV2PADDED(X,H) applies 2D filter H to X with constant extension
    #    padding. 
    # 
    #    Y = CONV2PADDED(H1,H2,X) first applies 1D filter H1 along the rows and 
    #    then applies 1D filter H2 along the columns.
    # 
    #    If X is a 3D array, filtering is done separately on each channel.

    #  Pascal Getreuer 2009

    if len(varargin) == 2:
        x, h = varargin
        if len(h.shape) > 1:
            vertical = h.shape[0]
            horizontal = h.shape[1]
        else:
            vertical = 1
            horizontal = h.shape[0]
            h = np.resize(h, (1, h.shape[0]))

        top = int(np.ceil(vertical / 2) - 1)
        bottom = int(np.floor(vertical / 2))
        left = int(np.ceil(horizontal/2) - 1)
        right = int(np.floor(horizontal / 2))


    elif len(varargin) == 3:
        h1, h2, x = varargin
        top = int(np.ceil(len(h1)/2) - 1)
        bottom = int(np.floor(len(h1)/2))
        left = int(np.ceil(len(h2)/2)-1)
        right = int(np.floor(len(h2)/2))
    else:
        raise AttributeError()
    

    # pad the input image
    x_padded = np.pad(x, pad_width=[(top, bottom), (left, right)], mode='edge')

    def conv2(v1, v2, m, mode='same'):
        """
        Two-dimensional convolution of matrix m by vectors v1 and v2

        First convolves each column of 'm' with the vector 'v1'
        and then it convolves each row of the result with the vector 'v2'.

        from https://stackoverflow.com/questions/24231285/is-there-a-python-equivalent-to-matlabs-conv2h1-h2-a-same

        """
        tmp = np.apply_along_axis(np.convolve, 0, m, v1, mode)
        return np.apply_along_axis(np.convolve, 1, tmp, v2, mode)

    if x.ndim == 2:
        x.resize((x.shape[0], x.shape[1], 1))
        x_padded.resize((x_padded.shape[0], x_padded.shape[1], 1))


    
    for p in range(x.shape[2]):
        if len(varargin) == 2:
            x[:,:,p] = convolve2d(x_padded[:,:,p], h, mode='valid')
        else:
            x[:,:,p] = conv2(h1, h2, x_padded[:,:,p], mode='valid')

    return x

def filter_lgn(im, sigma):
    break_off_sigma = 3
    filter_size = break_off_sigma * sigma
    x = np.array([i for i in range(-filter_size, filter_size+1)])

    gauss = 1 / (np.sqrt(2*np.pi) * sigma) * np.exp((x**2) / (-2 * sigma * sigma))
    Gx = (x**2 / np.power(sigma, 4) - 1/sigma**2) * gauss
    Gx = Gx - sum(Gx) / len(x)
    Gx = Gx / sum(0.5 * x * x * Gx)

    Gy = (x**2 / np.power(sigma, 4) - 1/sigma**2) * gauss
    Gy = Gy - sum(Gy) / len(x)
    Gy = Gy / sum(0.5 * x * x * Gy)

    if im.shape[2] == 1:
        im = im / np.max(im)
        Ex = conv2padded((deepcopy(im), Gx))
        Ey = conv2padded((deepcopy(im), np.matrix(Gy).H))
        e = np.sqrt(Ex**2 + Ey**2)
        el = []
        ell = []

    else:
        e, el, ell = rgb2e(im)

        # im = e
        Ex = conv2padded((deepcopy(e), Gx))
        Ey = conv2padded((deepcopy(e), np.matrix(Gy).H))
        e = np.sqrt(Ex**2 + Ey**2).squeeze()

        # im = el
        Elx = conv2padded((deepcopy(el), Gx))
        Ely = conv2padded((deepcopy(el), np.matrix(Gy).H))
        el = np.sqrt(Elx**2 + Ely**2).squeeze()

        # im = ell
        Ellx = conv2padded((deepcopy(ell), Gx))
        Elly = conv2padded((deepcopy(ell), np.matrix(Gy).H))
        ell = np.sqrt(Ellx**2 + Elly**2).squeeze()


    
    return e, el, ell

def lgn_statistics(im, threshold_lgn, viewing_dist=1, dot_pitch=0.00035, fov_beta=1.5, fov_gamma = 5):

    ce =None
    sc = None
    beta = None
    gamma = None

    # threshold_lgn = np.load('ThresholdLGN') # TODO create a npy file with these values

    if type(im) is str:
        im = cv2.imread(im)

    # 
    # Set image parameters 
    # 


    if im.shape[-1] == 2:
        IMTYPE = 1 #Gray
    elif im.shape[-1] == 3:
        IMTYPE = 2 #Color

    imsize = im.shape[:2]
    minmag1 = np.zeros(imsize)
    minmag2 = np.zeros(imsize)
    minmag3 = np.zeros(imsize)


    #######################################################
    # Set parameters for field of view
    #######################################################

    CT0 = 1/75                         # constant from Geisler&Perry
    alpha = (0.106)*1                  # constant from Geisler&Perry
    epsilon2 = 2.3                     # constant from Geisler&Perry
    fovx = round(imsize[1]/2)          # x-pixel loc. of fovea center
    fovy = round(imsize[0]/2)          # y-pixel loc. of fovea center
    # ex and ey are the x- and y- offsets of each pixel compared to
    # the point of focus (fovx,fovy) in pixels.
    # ex, ey = np.meshgrid(-fovx+1:imsize[2]-fovx, -fovy+1:imsize[1]-fovy)
    ex, ey = np.meshgrid(np.arange(start=-fovx+1, stop=imsize[1]-fovx+1), 
                            np.arange(start=-fovy+1, stop=imsize[0]-fovy+1))
    # eradius is the radial distance between each point and the point
    # of gaze.  This is in meters.
    eradius = dot_pitch * np.sqrt(ex**2+ey**2)
    #del ex, ey
    # calculate ec, the eccentricity from the foveal center, for each
    # point in the image.  ec is in degrees.
    ec = 180*np.arctan(eradius / viewing_dist)/np.pi
    # select the pixels that fall within the input visual field of view
    imfovbeta = np.argwhere(ec<fov_beta).squeeze()
    imfovgamma = np.argwhere(ec<fov_gamma).squeeze()

    ######
    # Computing edges
    ######

    par1 = np.zeros(imsize)
    par2 = np.zeros(imsize)
    par3 = np.zeros(imsize)
    mag1 = np.zeros(imsize)
    mag2 = np.zeros(imsize)
    mag3 = np.zeros(imsize)

    for sigma_iterations in np.array([[48, 24, 12, 6, 3], [64, 32, 16, 8, 4]]):

        offset = 0
        filter_nr = 0
        scale_nr = 0

        for sigma in sigma_iterations:
            print(f"Sigma: {sigma}")
            filter_nr += 1

            sigmas = np.array([1,2,4,8,16,32])
            v1 = np.squeeze(threshold_lgn[:,0])
            t1_interp = interp1d(sigmas, v1, kind='linear', bounds_error=False, fill_value=np.nan)
            t1 = t1_interp(sigma)
            v2 = np.squeeze(threshold_lgn[:,1])
            t2_interp = interp1d(sigmas, v2, kind='linear', bounds_error=False, fill_value=np.nan)
            t2 = t2_interp(sigma)
            v3 = np.squeeze(threshold_lgn[:,2])
            t3_interp = interp1d(sigmas, v3, kind='linear', bounds_error=False, fill_value=np.nan)
            t3 = t3_interp(sigma)

            o1, o2, o3 = filter_lgn(im, sigma)
            s1 = local_cov(o1, sigma)
            e1 = (o1 * np.max(o1, axis=0) / (o1 + np.max(o1, axis=0) * s1))
            minm1 = e1 - t1
            index1 = np.argwhere(minm1 > 0.0000001)
            par1[index1] = minm1[index1]

            if IMTYPE == 2:
                s2 = local_cov(o2, sigma)
                e2 = (o2 * np.max(o2, axis=0) / (o2 + np.max(o2, axis=0) * s2))
                minm2 = e2 - t2
                index2 = np.argwhere(minm2 > 0.0000001)
                par2[index2] = minm2[index2]

                s3 = local_cov(o3, sigma)
                e3 = (o3 * np.max(o3, axis=0) / (o3 + np.max(o3, axis=0) * s3))
                minm3 = e3 - t3
                index3 = np.argwhere(minm3 > 0.0000001)
                par3[index3] = minm3[index3]


    ##############
    # Compute Feature Energy and Spatial Coherence
    ##############

    print("Compute CE and SC")

    magnitude = np.abs(par1[imfovbeta])
    ce[0] = np.mean(magnitude)
    if IMTYPE == 2:
        magnitude = np.abs(par2[imfovbeta])
        ce[1]= np.mean(magnitude)
        magnitude = np.abs(par3[imfovbeta])
        ce[2]= np.mean(magnitude)


    magnitude = np.abs(mag1[imfovgamma])
    sc[0] = np.mean(magnitude)
    if IMTYPE == 2:
        magnitude = np.abs(mag2[imfovgamma])
        sc[1]= np.mean(magnitude)
        magnitude = np.abs(mag3[imfovgamma])
        sc[2]= np.mean(magnitude)


    #################
    # Compute Weibull parameters beta and gamma
    #################

    print("Compute Weibull parameters beta and gamma")

    n_bins = 1000
    magnitude = np.abs(par1[imfovbeta])
    ax, h = create_hist(magnitude, n_bins)
    beta[0], _ = weibullMleHist(ax, h)

    if IMTYPE == 2:
        magnitude = np.abs(par2[imfovbeta])
        ax, h = create_hist(magnitude, n_bins)
        beta[1], _ = weibullMleHist(ax, h)

        magnitude = np.abs(par3[imfovbeta])
        ax, h = create_hist(magnitude, n_bins)
        beta[2], _ = weibullMleHist(ax, h)


    magnitude = np.abs(mag1[imfovgamma])
    ax, h = create_hist(magnitude, n_bins)
    _, gamma[0] = weibullMleHist(ax, h)

    if IMTYPE == 2:
        magnitude = np.abs(mag2[imfovgamma])
        ax, h = create_hist(magnitude, n_bins)
        _, gamma[1] = weibullMleHist(ax, h)

        magnitude = np.abs(mag3[imfovgamma])
        ax, h = create_hist(magnitude, n_bins)
        _, gamma[2] = weibullMleHist(ax, h)

    return (ce, sc, beta, gamma)