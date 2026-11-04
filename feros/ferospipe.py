import sys
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(_REPO_ROOT / ".matplotlib-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from pylab import *

base = str(_REPO_ROOT) + '/'
sys.path.append(base+"utils/Correlation")
sys.path.append(base+"utils/GLOBALutils")
sys.path.append(base+"utils/OptExtract")
sys.path.append(base+"utils/CCF")
sys.path.append(base+"utils/SSEphem")

baryc_dir= base+'utils/SSEphem/'
sys.path.append(baryc_dir)
ephemeris='DEc403'

# ceres modules
import ferosutils
import correlation
import GLOBALutils
import Marsh

# other useful modules
import argparse
import ephem

#import jplephem
#from PyAstronomy import pyasl
from math import radians as rad
from astropy.io import fits as pyfits
import pickle
import scipy
import scipy.interpolate
from scipy import interpolate

import statsmodels.api as sm
lowess = sm.nonparametric.lowess

# Recive input parameters
parser = argparse.ArgumentParser()
parser.add_argument('directorio')
parser.add_argument('-avoid_plot', action="store_true", default=False)
parser.add_argument('-dirout',default='default')
parser.add_argument('-calibdir',default='default')
parser.add_argument('-do_class', action="store_true", default=False)
parser.add_argument('-just_extract', action="store_true", default=False)
parser.add_argument('-npools', default=1)
parser.add_argument('-o2do',default='all')
parser.add_argument('-reffile',default='default')
parser.add_argument('-lamp',default='LAMP3')
parser.add_argument('-binning',default=1)
parser.add_argument('-use_astropy',default=True)
parser.add_argument('-rmsmax',default=100)

args   = parser.parse_args()
dirin            = args.directorio
avoid_plot       = args.avoid_plot
dirout           = args.dirout
calibdir         = args.calibdir
DoClass          = args.do_class
JustExtract      = args.just_extract
npools           = int(args.npools)
object2do        = args.o2do
reffile          = args.reffile
lamp             = str(args.lamp)
binning          = int(args.binning)
use_astropy      = args.use_astropy
rmsmax      = int(args.rmsmax)

####### GLOBAL VARIABLES #####
## perhaps put into options ##
force_pre_process  = False
force_flat_extract = False
force_thar_extract = False
force_thar_wavcal  = False #set to True to force ThAr Wavelength calibrations
force_tharxc       = False
force_sci_extract  = False
force_stellar_pars = False
force_trace        = False
dumpargon          = False
force_spectral_file_build = True
dark_correction    = False

bad_colummn        = True
Inverse_m          = True
use_cheby          = True

MRMS               = 90   # max rms in m/s, global wav solution
if binning == 2:
    minlines = 900
else:
    minlines = 1200

trace_degree       = 4
Marsh_alg          = 0
ext_aperture       = 6//binning
NSigma_Marsh       = 10
NCosmic_Marsh      = 10
S_Marsh            = 0.4
N_Marsh            = 4      # grado polinomio
min_extract_col    = 50//binning
max_extract_col    = 4000//binning

porder_single = 5
ncoef_x            = 5#6
ncoef_m            = 7#7
npar_wsol = (min(ncoef_x,ncoef_m) + 1) * (2*max(ncoef_x,ncoef_m) - min(ncoef_x,ncoef_m) + 2) // 2
nconts = np.array([2,2,3,3,2,3,3,3,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
lim_iz = np.array([100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,150,200,250,300,350,400,450,500,500,800])//binning


if dirin[-1] != '/':
    dirin = dirin + '/'

if dirout == 'default':
    dirout = dirin[:-1]+'_red/'

if calibdir == 'default':
    calibdir = dirout

if calibdir[-1] != '/':
    calibdir = calibdir + '/'

external_calibdir = os.path.abspath(calibdir) != os.path.abspath(dirout)

if external_calibdir and not os.access(calibdir,os.F_OK):
    raise SystemExit('external calibration directory does not exist: '+calibdir)

if not os.access(dirout,os.F_OK):
    os.system('mkdir '+dirout)
if os.access(dirout+'proc',os.F_OK) and force_spectral_file_build:
    os.system('rm -r '+dirout+'proc')
os.system('mkdir '+dirout+'proc')

print('Dirin and dirout directories:', dirin, dirout)
print('Calibration products directory:', calibdir)
f_res = open(dirout+'proc/'+'results.txt','w')

if reffile == 'default':
    reffile = dirin+'reffile.txt'

models_path = base+"data/COELHO_MODELS/R_40000b/"
order_dir   = base+"feros/wavcals/"

o0  = 8
OO0 = 26
n_useful = 25    # up to which order do we care?
thar_end = '.harps.3iwdat'

# file containing the log
log = dirout+'night.log'

print("\n\n\tFEROS MPG2.2m  PIPELINE\n")
print("\tRAW data is in ",dirin)
print("\tProducts of reduction will be in",dirout)
print('\n')

biases, flats, ThArNe_ref, ThAr_Ne_ref, simThAr_sci, simSky_sci, ThAr_ref_dates, \
        ThAr_Ne_ref_dates, darks, dark_times = ferosutils.FileClassify(dirin,log, lamp=lamp)
#ThAr_Ne_ref,ThAr_Ne_ref_dates = ThAr_Ne_ref[:8],ThAr_Ne_ref_dates[:8]
if ( (os.access(calibdir+'Flat.fits',os.F_OK) == False)        or \
     (os.access(calibdir+'trace.pkl',os.F_OK) == False)        or \
     (os.access(calibdir+'MasterBias.fits',os.F_OK) == False)  or \
         (force_pre_process) ):
    print("\tNo previous pre-processing files or found")
    pre_process = 1
else:
    print("\tPre-processing files found, going straight to extraction")
    pre_process = 0

if (pre_process == 1):
    if external_calibdir:
        raise SystemExit('external calibration directory is missing pre-processing products: '+calibdir)
    print("\t\tGenerating Master calibration frames...")
    # median combine Biases

    MasterBias, RO_bias, GA_bias = ferosutils.MedianCombine(biases) # (4096, 2048)
    hdu = pyfits.PrimaryHDU( MasterBias )
    if (os.access(calibdir+'MasterBias.fits',os.F_OK)):
        os.remove(calibdir+'MasterBias.fits')
    hdu.writeto(calibdir+'MasterBias.fits')
    print("\t\t-> Masterbias: done!")

    # median combine list of flats
    Flat, RO_flat, GA_flat = ferosutils.MedianCombine(flats, zero_bo=True, zero=calibdir+'MasterBias.fits')
    # save this file for later reference
    hdu = pyfits.PrimaryHDU( Flat )
    if (os.access(calibdir+'Flat.fits',os.F_OK)):
        os.remove(calibdir+'Flat.fits')
    hdu.writeto(calibdir+'Flat.fits')
    print("\t\t-> Masterflat: done!")

    Flat = Flat.T # (2048, 4096)
    print("\tTracing echelle orders...")

    c_all, nord_all = GLOBALutils.get_them(Flat, int(np.ceil(5/binning)), trace_degree, maxords=-1,mode=2,startfrom=40//binning,endat=1900//binning, nsigmas=5)
    # these are polynomial coefficients

    print(nord_all)
    if nord_all//2 == nord_all/2:
        c_all = c_all[2:]
        nord_all -= 2
    else:
        c_all = c_all[2:-1]
        nord_all -= 3
    print(nord_all)

    I1 = np.arange(0,nord_all,2) # 0, 2, 4, ... 70
    I2 = np.arange(1,nord_all+1,2) # 1, 3, 5,...
    c_ob, c_co = c_all[I1], c_all[I2] # coefficients?
    nord_ob, nord_co = len(I1), len(I2) # order and comparison fiber
    print("\t\t"+str(nord_ob)+" object orders found...")
    print("\t\t"+str(nord_co)+" comparison orders found...")


    # to check whether it worked...

    # x = np.arange(Flat.shape[1])
    # f = plt.figure()
    # ax = f.add_subplot(111)
    # ax.imshow(Flat)
    # for i in range(len(c_ob)):
    #   ax.plot(x,np.poly1d(c_ob[i])(x), 'r')
    # for i in range(len(c_co)):
    #   ax.plot(x,np.poly1d(c_co[i])(x), 'r--')
    # ax.set_xlim( np.shape(Flat)[1]/2,  np.shape(Flat)[1]/2+100/binning)
    # ax.set_ylim( 150/binning,50/binning)
    # first order around pixel 99
    trace_dict = {'c_ob':c_ob, 'c_co':c_co, 'c_all':c_all, \
                  'nord_ob':nord_ob, 'nord_co':nord_co, 'nord_all':nord_all,\
                  'GA_flat':GA_flat,'RO_flat':RO_flat}

    pickle.dump( trace_dict, open( calibdir+"trace.pkl", 'wb' ) )

else:
    trace_dict = pickle.load( open( calibdir+"trace.pkl", 'rb' ) )
    c_co = trace_dict['c_co']
    c_ob = trace_dict['c_ob']
    c_all = trace_dict['c_all']
    nord_ob = trace_dict['nord_ob']
    nord_co = trace_dict['nord_co']
    nord_all = trace_dict['nord_all']
    # recover GA*, RO*
    GA_flat = trace_dict['GA_flat']
    RO_flat = trace_dict['RO_flat']
    # recover flats & master bias
    h = pyfits.open(calibdir+'Flat.fits')
    Flat = h[0].data
    Flat = Flat.T
    h = pyfits.open(calibdir+'MasterBias.fits')
    MasterBias = h[0].data

dark_times = np.around(dark_times).astype('int')
uni_times = np.unique(dark_times)
dark_names = []
for tim in uni_times:
    I = np.where(dark_times == tim)[0]
    Dark, RO_dark, GA_dark = ferosutils.MedianCombine(darks[I], zero_bo=True, zero=calibdir+'MasterBias.fits')
    dname = dirout+'MasterDark_'+str(tim)+'.fits'
    dark_names.append(dname)
    hdu = pyfits.PrimaryHDU( Dark )
    if (os.access(dname,os.F_OK)):
        os.remove(dname)
    hdu.writeto(dname)
    print('Dark of', tim,' seconds done!')



print('\n\tExtraction of Flat calibration frames:')

P_ob_fits = calibdir + 'P_ob.fits'
P_co_fits = calibdir + 'P_co.fits'
P_fits    = calibdir + 'P.fits'
S_flat_ob_fits        = calibdir +'S_flat_ob.fits'
S_flat_ob             = np.zeros((nord_ob, 3, Flat.shape[1]) )
S_flat_co_fits        = calibdir +'S_flat_co.fits'
S_flat_co             = np.zeros((nord_co, 3, Flat.shape[1]) )

#bacfile = dirout + 'BAC_Flat.fits'
force_flat_extract = False
if ( os.access(P_ob_fits,os.F_OK) == False )             or ( os.access(P_co_fits,os.F_OK) == False )             or \
   ( os.access(S_flat_ob_fits,os.F_OK) == False )        or ( os.access(S_flat_co_fits,os.F_OK) == False )        or \
   (force_flat_extract):
    if external_calibdir:
        raise SystemExit('external calibration directory is missing flat-extraction products: '+calibdir)
    "\t\t\tComputing Background..."
    """
    Centers = np.zeros((len(c_all),Flat.shape[1]))
    for i in range(c_all.shape[0]):
        Centers[i,:]=scipy.polyval(c_all[i,:],np.arange(len(Centers[i,:])))
    bac = GLOBALutils.get_scat(Flat,Centers,span=10,typ='min')
    """
    #plot(Flat[:,2000])
    #plot(bac[:,2000])
    #show()
    #print gfd
    #Flat -= bac

    P_ob = GLOBALutils.obtain_P(Flat,c_ob,ext_aperture,RO_flat,\
                                GA_flat,NSigma_Marsh, S_Marsh, \
                                N_Marsh, Marsh_alg, min_extract_col,\
                                max_extract_col, npools)
    P_co = GLOBALutils.obtain_P(Flat,c_co,ext_aperture,RO_flat,\
                                GA_flat,NSigma_Marsh, S_Marsh, \
                                N_Marsh, Marsh_alg, min_extract_col,\
                                max_extract_col, npools)
    P = P_ob + P_co

    if ( os.access(P_fits,os.F_OK) ):
        os.remove(P_fits)
    if (os.access(P_ob_fits,os.F_OK)):
        os.remove( P_ob_fits )
    if (os.access(P_co_fits,os.F_OK)):
        os.remove( P_co_fits )

    hdu = pyfits.PrimaryHDU( P )
    hdu.writeto( P_fits )
    hdu = pyfits.PrimaryHDU( P_ob )
    hdu.writeto( P_ob_fits )
    hdu = pyfits.PrimaryHDU( P_co )
    hdu.writeto( P_co_fits )

    print("\t\t\tNo extracted flat object spectra found or extraction forced, extracting and saving...")

    S_flat_ob  = GLOBALutils.optimal_extraction(Flat,P_ob,c_ob,ext_aperture,\
                                            RO_flat,GA_flat,S_Marsh,NCosmic_Marsh,\
                                            min_extract_col,max_extract_col,npools)

    # write P_on and S_flat_ob as fits files
    if (os.access(S_flat_ob_fits,os.F_OK)):
        os.remove( S_flat_ob_fits )

    hdu = pyfits.PrimaryHDU( S_flat_ob )
    hdu.writeto( S_flat_ob_fits )

    print("\t\t\tNo extracted flat comparison spectra found or extraction forced, extracting and saving...")
    S_flat_co  = GLOBALutils.optimal_extraction(Flat,P_co,c_co,ext_aperture,RO_flat,GA_flat,\
                                            S_Marsh,NCosmic_Marsh,min_extract_col,\
                                            max_extract_col,npools)

    # write P_on and S_flat_co as fits files

    if (os.access(S_flat_co_fits,os.F_OK)):
        os.remove( S_flat_co_fits )

    hdu = pyfits.PrimaryHDU( S_flat_co )
    hdu.writeto( S_flat_co_fits )

else:
    print("\t\tExtracted flat object spectra found, loading...")
    print("\t\tExtracted flat comparison spectra found, loading...")
    P_ob             = pyfits.getdata( P_ob_fits )
    P_co             = pyfits.getdata( P_co_fits ) #added this line
    S_flat_ob        = pyfits.getdata( S_flat_ob_fits )
    S_flat_co        = pyfits.getdata( S_flat_co_fits )

# Normalize flat field spectra.
S_flat_ob_n, norms_ob = GLOBALutils.FlatNormalize_single( S_flat_ob, mid=int(.5*S_flat_ob.shape[2]))
S_flat_co_n, norms_co = GLOBALutils.FlatNormalize_single( S_flat_co, mid=int(.5*S_flat_co.shape[2]))

if nord_ob < n_useful:
    n_useful = thar_S_ob.shape[0]
print('\n\tExtraction of ThAr calibration frames:')
# Extract all ThAr+Ne files
for fsim in ThAr_Ne_ref:
    print("\t\tWorking on ThAr+Ne file ", fsim, "...")
    thar_fits_ob = calibdir + fsim.split('/')[-1][:-4]+'spec.ob.fits.S'
    thar_fits_co = calibdir + fsim.split('/')[-1][:-4]+'spec.co.fits.S'
    thar_fits_ob_simple = calibdir + fsim.split('/')[-1][:-4]+'spec.simple.ob.fits.S'
    thar_fits_co_simple = calibdir + fsim.split('/')[-1][:-4]+'spec.simple.co.fits.S'

    missing_thar_extract = ( os.access(thar_fits_ob,os.F_OK) == False ) or \
                           ( os.access(thar_fits_co,os.F_OK) == False ) or \
                           ( os.access(thar_fits_ob_simple,os.F_OK) == False ) or \
                           ( os.access(thar_fits_co_simple,os.F_OK) == False ) or \
                           (force_thar_extract)

    if missing_thar_extract:
        if external_calibdir:
            raise SystemExit('external calibration directory is missing extracted ThAr products for '+fsim+': '+calibdir)

        print("\t\tNo previous extraction or extraction forced for ThAr file", fsim, "extracting...")

        dthar = ferosutils.OverscanTrim(pyfits.getdata(fsim))
        dthar = dthar - MasterBias
        dthar = dthar.T

        RO_thar, GA_thar = ferosutils.get_RG(pyfits.getheader(fsim))

        if bad_colummn:
            dthar = ferosutils.b_col(dthar)

        thar_Ss_ob = GLOBALutils.simple_extraction(dthar,c_ob,ext_aperture,min_extract_col,max_extract_col,npools)
        thar_S_ob  = GLOBALutils.optimal_extraction(dthar,P_ob,c_ob,ext_aperture,RO_thar, GA_thar,S_Marsh,100.,min_extract_col,max_extract_col,npools)
        thar_Ss_co = GLOBALutils.simple_extraction(dthar,c_co,ext_aperture,min_extract_col,max_extract_col,npools)
        thar_S_co  = GLOBALutils.optimal_extraction(dthar,P_co,c_co,ext_aperture,RO_thar, GA_thar,S_Marsh,100.,min_extract_col,max_extract_col,npools)

        if (os.access(thar_fits_ob,os.F_OK)):
            os.remove( thar_fits_ob )
        if (os.access(thar_fits_co,os.F_OK)):
            os.remove( thar_fits_co )
        if (os.access(thar_fits_ob_simple,os.F_OK)):
            os.remove( thar_fits_ob_simple )
        if (os.access(thar_fits_co_simple,os.F_OK)):
            os.remove( thar_fits_co_simple )

        hdu = pyfits.PrimaryHDU( thar_S_ob )
        hdu.writeto( thar_fits_ob )
        hdu = pyfits.PrimaryHDU( thar_S_co )
        hdu.writeto( thar_fits_co )
        hdu = pyfits.PrimaryHDU( thar_Ss_ob )
        hdu.writeto( thar_fits_ob_simple )
        hdu = pyfits.PrimaryHDU( thar_Ss_co )
        hdu.writeto( thar_fits_co_simple )
    else:
        print("\t\tThAr file", fsim, "all ready extracted, loading...")

sorted_ThAr_Ne_dates = np.argsort( ThAr_Ne_ref_dates )

c_p2w,c_p2w_c = [],[]
print("\n\tWavelength solution of ThAr calibration spectra:")
for i in range(len(sorted_ThAr_Ne_dates)):
    index      = sorted_ThAr_Ne_dates[i]
    fname_base = os.path.splitext(os.path.basename(ThAr_Ne_ref[index]))[0]
    wavsol_pkl = os.path.join(calibdir, fname_base + '.wavsolpars.pkl') #Trying to define wavsol_pkl to avoid crashing
    print(f'wavsol_pkl={wavsol_pkl}') #should end in .pkl to properly work
    if ( os.access(wavsol_pkl,os.F_OK) == False ) or (force_thar_wavcal): #NameError: Name wavsol_pkl is not defined
        if external_calibdir:
            raise SystemExit('external calibration directory is missing wavelength solution for '+ThAr_Ne_ref[index]+': '+calibdir)
        print("\t\tComputing wavelength solution of ThAr file", ThAr_Ne_ref[index])

        hthar        = pyfits.open( ThAr_Ne_ref[index] )
        mjd, mjd0    = ferosutils.mjd_fromheader( hthar )

        thar_fits_ob = calibdir + ThAr_Ne_ref[index].split('/')[-1][:-4]+'spec.ob.fits.S'
        thar_fits_co = calibdir + ThAr_Ne_ref[index].split('/')[-1][:-4]+'spec.co.fits.S'
        thar_S_ob    = pyfits.getdata( thar_fits_ob )
        thar_S_co    = pyfits.getdata( thar_fits_co )

        lines_thar_ob  = thar_S_ob[:,1,:]
        iv_thar_ob     = thar_S_ob[:,2,:]
        lines_thar_co  = thar_S_co[:,1,:]
        iv_thar_co     = thar_S_co[:,2,:]

        All_Pixel_Centers = np.array([])
        All_Wavelengths   = np.array([])
        All_Orders        = np.array([])
        All_Centroids     = np.array([])
        All_Sigmas        = np.array([])
        All_Intensities   = np.array([])
        All_residuals   = np.array([])

        if thar_S_ob.shape[0] < n_useful:
            n_useful = thar_S_ob.shape[0]
        #print(f'Useful orders: {n_useful}, ThAr_S_ob.shape[0]={thar_S_ob.shape[0]}')
        wavss   =[]
        orss    = []
        order   = o0
        mid_wav = []
        mid_col_wv = []

        while order < min(o0 + n_useful, thar_S_ob.shape[0]): #preventing the IndexError
            order_s = str(order+1)
            #print(f'order={order}, o0 + n_useful={o0  + n_useful}')
            if (order < 9):
                order_s = '0'+str(order+1)

            thar_order_orig = lines_thar_ob[order,:] #this threw an IndexError for night 2025-05-12
            IV              = iv_thar_ob[order,:] # seems to be the star spectrum??
            wei             = np.sqrt( IV )
            bkg             = ferosutils.Lines_mBack(thar_order_orig, IV,  thres_rel=3)
            thar_order      = thar_order_orig - bkg

            coeffs_pix2wav, coeffs_pix2sigma, pixel_centers, wavelengths,\
            rms_ms, residuals, centroids, sigmas, intensities =\
                                         GLOBALutils.Initial_Wav_Calibration(order_dir+'order_'+\
                                         order_s+thar_end, thar_order, order, wei, rmsmax=100*binning, \
                                         minlines=30,FixEnds=False,Dump_Argon=dumpargon,\
                                         Dump_AllLines=True, Cheby=use_cheby,porder=ncoef_x,del_width=3.0/binning, binning=binning)
            #coef2 = np.polyfit(pixel_centers, wavelengths,5)
            #plot(pixel_centers,wavelengths-np.polyval(coef2,pixel_centers),'ro')
            #axhline(0.)
            #show()
            mid_col_wv.append(GLOBALutils.Cheby_eval(coeffs_pix2wav, int(.5*len(thar_order)), len(thar_order)))
            c_p2w.append(coeffs_pix2wav)
            mid_wav.append(GLOBALutils.Cheby_eval(coeffs_pix2wav,int(.25*len(thar_order)), len(thar_order)))
            wavss.append(GLOBALutils.Cheby_eval(coeffs_pix2wav,int(len(thar_order)/2.), len(thar_order)))

            orss.append(float(order))

            if (order == 16):
                if (use_cheby):
                    Global_ZP = GLOBALutils.Cheby_eval( coeffs_pix2wav, int(.5*len(thar_order)), len(thar_order) )
                else:
                    Global_ZP = scipy.polyval( coeffs_pix2wav, 0.0 )

            All_Pixel_Centers = np.append( All_Pixel_Centers, pixel_centers )
            All_Wavelengths   = np.append( All_Wavelengths, wavelengths )
            All_Orders        = np.append( All_Orders, np.zeros( len(pixel_centers) ) + order )
            All_Centroids     = np.append( All_Centroids, centroids)
            All_Sigmas        = np.append( All_Sigmas, sigmas)
            All_Intensities   = np.append( All_Intensities, intensities )
            All_residuals     = np.append( All_residuals, residuals )
            order+=1
        """
        refx = 1./(np.arange(len(mid_col_wv)).astype('float')+1)
        mid_col_wv = np.array(mid_col_wv)#s[5:]
        coef1 = np.polyfit(refx,mid_col_wv,6)
        coef2 = np.polyfit(refx,mid_col_wv,7)
        coef3 = np.polyfit(refx,mid_col_wv,8)
        coef4 = np.polyfit(refx,mid_col_wv,9)

        plot(refx,mid_col_wv-np.polyval(coef1,refx),'o')
        axhline(0.)
        #show()
        plot(refx,mid_col_wv-np.polyval(coef2,refx),'o')
        axhline(0.)
        #show()
        plot(refx,mid_col_wv-np.polyval(coef3,refx),'o')
        axhline(0.)
        plot(refx,mid_col_wv-np.polyval(coef4,refx),'o')
        axhline(0.)
        show()
        """
        p0    = np.zeros( npar_wsol )
        p0[0] =  (16+OO0) * Global_ZP
        p1, G_pix, G_ord, G_wav, II, rms_ms, G_res = \
            GLOBALutils.Fit_Global_Wav_Solution(All_Pixel_Centers, All_Wavelengths, All_Orders,\
                                                np.ones(All_Intensities.shape), p0, Cheby=use_cheby,\
                                                maxrms=MRMS, Inv=Inverse_m,minlines=minlines,order0=OO0, \
                                                ntotal=n_useful,npix=len(thar_order),nx=ncoef_x,nm=ncoef_m)

        """
        ejxx = np.arange(4096)
        for i in np.unique(G_ord):
                I = np.where(G_ord == i)[0]
                print i,len(I)
                #m = G_ord+OO0
                #m2 = m[I][0]+np.zeros(len(ejxx))
                #chebs = ferosutils.Calculate_chebs(ejxx, m2, OO0, n_useful,4096,1,Inverse=Inverse_m)
                #ret = (1.0/m2) * FEROSutils.Joint_Polynomial_Cheby(p1, chebs)
                plot(G_wav[I],G_res[I],'.')
                #plot(ejxx,ret,'k')
        #plot(G_pix,G_wav,'ro')
        show()
        """

        # Now calibrate COMPARISON orders. Use p1 above as p0
        All_Pixel_Centers_co = np.array([])
        All_Wavelengths_co   = np.array([])
        All_Orders_co        = np.array([])
        All_Centroids_co     = np.array([])
        All_Sigmas_co        = np.array([])
        All_Intensities_co   = np.array([])
        All_residuals_co     = np.array([])
        order = o0
        while order <  min(o0 + n_useful, lines_thar_co.shape[0]): #insert the min case to prevent IndexErrors
            order_s = str(order+1)
            if (order < 9):
                order_s = '0'+str(order+1)

            thar_order_orig = lines_thar_co[order,:]
            IV              = iv_thar_co[order,:]
            wei             = np.sqrt( IV )
            bkg             = ferosutils.Lines_mBack(thar_order_orig, IV,  thres_rel=3)
            thar_order      = thar_order_orig - bkg

            coeffs_pix2wav, coeffs_pix2sigma, pixel_centers, wavelengths,\
            rms_ms, residuals, centroids, sigmas, intensities =\
                                         GLOBALutils.Initial_Wav_Calibration(order_dir+'order_'+\
                                         order_s+thar_end, thar_order, order, wei, rmsmax=100*binning, \
                                         minlines=30,FixEnds=False,Dump_Argon=dumpargon,\
                                         Dump_AllLines=True, Cheby=use_cheby,porder=ncoef_x,del_width=3.5/binning,binning=binning)

            c_p2w_c.append(coeffs_pix2wav)
            All_Pixel_Centers_co = np.append( All_Pixel_Centers_co, pixel_centers )
            All_Wavelengths_co   = np.append( All_Wavelengths_co, wavelengths )
            All_Orders_co        = np.append( All_Orders_co, np.zeros( len(pixel_centers) ) + order )
            All_Centroids_co     = np.append( All_Centroids_co, centroids)
            All_Sigmas_co        = np.append( All_Sigmas_co, sigmas)
            All_Intensities_co   = np.append( All_Intensities_co, intensities )
            All_residuals_co     = np.append( All_residuals_co, residuals )
            order+=1

        p1_co, G_pix_co, G_ord_co, G_wav_co, II_co, rms_ms_co, G_res_co = \
            GLOBALutils.Fit_Global_Wav_Solution(All_Pixel_Centers_co, All_Wavelengths_co, All_Orders_co,\
                                                np.ones(All_Intensities_co.shape), p1, Cheby=use_cheby,\
                                                maxrms=MRMS, Inv=Inverse_m,minlines=minlines,order0=OO0, \
                                                ntotal=n_useful,npix=len(thar_order),nx=ncoef_x,nm=ncoef_m)

        #for io in range(int(G_ord_co.min()),int(G_ord_co.max()+1),1):
        #       I = np.where(G_ord_co == io)[0]
        #       tmpw,tmpr = G_wav_co[I],G_res_co[I]
        #       IJ = np.argsort(tmpw)
        #       tmpw,tmpr = tmpw[IJ],tmpr[IJ]
        #       plot(tmpw,tmpr,'.')
        #       plot(tmpw,scipy.signal.medfilt(tmpr,21),'.')
        #axhline(0.)
        #show()

        spec_ob,spec_co = np.zeros((2,n_useful,len(thar_order))),np.zeros((2,n_useful,len(thar_order)))
        equis = np.arange( len(thar_order) )
        order = o0
        ind   = 0
        while order <  min(o0 + n_useful, thar_S_ob.shape[0]): #Same here to prevent IndexError. Originally order < o0 + n_useful
            oss = order - o0
            m   = order + OO0
            chebs = GLOBALutils.Calculate_chebs(equis, m, order0=OO0,\
                    ntotal=n_useful, npix=len(equis), Inverse=Inverse_m,nx=ncoef_x,nm=ncoef_m)
            WavSol_ob = (1.0/float(m)) * GLOBALutils.Joint_Polynomial_Cheby(p1,chebs,ncoef_x,ncoef_m)
            WavSol_co = (1.0/float(m)) * GLOBALutils.Joint_Polynomial_Cheby(p1_co,chebs,ncoef_x,ncoef_m)
            spec_ob[0,oss,:] = WavSol_ob
            spec_ob[1,oss,:] = lines_thar_ob[order]
            spec_co[0,oss,:] = WavSol_co
            spec_co[1,oss,:] = lines_thar_co[order]
            order+=1

        hdu_ob, hdu_co = pyfits.PrimaryHDU(spec_ob),pyfits.PrimaryHDU(spec_co)
        if os.access(calibdir + ThAr_Ne_ref[index].split('/')[-1][:-5]+'_sp_ob.fits',os.F_OK):
            os.system('rm ' + calibdir + ThAr_Ne_ref[index].split('/')[-1][:-5]+'_sp_ob.fits')
        hdu_ob.writeto(calibdir + ThAr_Ne_ref[index].split('/')[-1][:-5]+'_sp_ob.fits')
        if os.access(calibdir + ThAr_Ne_ref[index].split('/')[-1][:-5]+'_sp_co.fits',os.F_OK):
            os.system('rm ' + calibdir + ThAr_Ne_ref[index].split('/')[-1][:-5]+'_sp_co.fits')
        hdu_co.writeto(calibdir + ThAr_Ne_ref[index].split('/')[-1][:-5]+'_sp_co.fits')

        # end COMPARISON orders.

        pdict = {'c_p2w':c_p2w,'p1':p1,'mjd':mjd, 'G_pix':G_pix, 'G_ord':G_ord,\
                 'G_wav':G_wav, 'II':II, 'rms_ms':rms_ms,'G_res':G_res,\
                 'All_Centroids':All_Centroids, 'All_Wavelengths':All_Wavelengths,\
                 'All_Orders':All_Orders, 'All_Pixel_Centers':All_Pixel_Centers,\
                 'All_Sigmas':All_Sigmas, 'c_p2w_c':c_p2w_c,'p1_co':p1_co,\
                 'G_pix_co':G_pix_co, 'G_ord_co':G_ord_co, 'G_wav_co':G_wav_co,\
                 'II_co':II_co, 'rms_ms_co':rms_ms_co, 'G_res_co':G_res_co,\
                 'All_Centroids_co':All_Centroids_co,'All_Wavelengths_co':All_Wavelengths_co,\
                 'All_Orders_co':All_Orders_co, 'All_Pixel_Centers_co':All_Pixel_Centers_co}

        pickle.dump( pdict, open( wavsol_pkl, 'wb' ) )

    else:
        print("\t\tUsing previously computed wavelength solution in file",wavsol_pkl)
        pdict = pickle.load(open(wavsol_pkl,'rb'))

#print gfd
ThAr_all       = np.hstack(( np.array(ThArNe_ref), np.array(ThAr_Ne_ref) ))
ThAr_all_dates = np.hstack(( np.array(ThAr_ref_dates), np.array(ThAr_Ne_ref_dates) ))
p_shifts = []
p_mjds   = []
refidx = 0
fname = calibdir +'thar_shifts.pdf'
dct_shfts = {}

force_shift = False
if os.access(calibdir+'shifts.pkl',os.F_OK):
    dct_shfts = pickle.load(open(calibdir+'shifts.pkl','rb'))
    for fl in ThAr_Ne_ref:
        if not fl in dct_shfts['names']:
            force_shift = True
            dct_shfts = {}
            break
else:
    force_shift = True
#"""
print(len(ThAr_all))
if force_shift and len(sorted_ThAr_Ne_dates)>6:
    if external_calibdir:
        raise SystemExit('external calibration directory is missing compatible shifts.pkl: '+calibdir)
    f, axarr = plt.subplots(len(sorted_ThAr_Ne_dates), sharex=True,figsize=(5, 30))
    Thar_shifts_out = calibdir + 'ThAr_Ne_shifts.dat'
    difs = 0
    mindif = 9999999999
    j = 0
    dct_shft = {}
    vec_dif = []
    vec_nam = []
    while j < len(sorted_ThAr_Ne_dates):
        fref   = ThAr_Ne_ref[sorted_ThAr_Ne_dates[j]]
        vec_nam.append(fref)
        refdct = pickle.load( open(calibdir + fref.split('/')[-1][:-4]+'wavsolpars.pkl','rb' ) )
        p_shifts = []
        p_shifts_ob = []
        p_mjds   = []
        i = 0
        while i < len(sorted_ThAr_Ne_dates):
            fsim  = ThAr_Ne_ref[sorted_ThAr_Ne_dates[i]]
            pdict = pickle.load( open(calibdir + fsim.split('/')[-1][:-4]+'wavsolpars.pkl','rb' ) )
            p_shift, pix_centers, orders, wavelengths, I, rms_ms, residuals  = \
                                    GLOBALutils.Global_Wav_Solution_vel_shift(pdict['All_Pixel_Centers_co'],\
                                    pdict['All_Wavelengths_co'], pdict['All_Orders_co'],\
                                    np.ones(len(pdict['All_Wavelengths_co'])), refdct['p1_co'],\
                                    minlines=1200, maxrms=MRMS,order0=OO0, ntotal=n_useful,\
                                    Cheby=use_cheby, Inv=Inverse_m, npix=Flat.shape[1],nx=ncoef_x,nm=ncoef_m)
            p_shifts.append(p_shift)
            p_mjds.append(pdict['mjd'])
            p_shift_ob, pix_centers_ob, orders_ob, wavelengths_ob, I_ob, rms_ms_ob, residuals_ob  = \
                        GLOBALutils.Global_Wav_Solution_vel_shift(pdict['All_Pixel_Centers'],\
                        pdict['All_Wavelengths'], pdict['All_Orders'],\
                        np.ones(len(pdict['All_Wavelengths'])), refdct['p1'],\
                        minlines=1200, maxrms=MRMS,order0=OO0, ntotal=n_useful,\
                        Cheby=use_cheby, Inv=Inverse_m, npix=Flat.shape[1],nx=ncoef_x,nm=ncoef_m)
            p_shifts_ob.append(p_shift_ob)
            i+=1
        p_shifts = np.array(p_shifts)
        p_shifts_ob = np.array(p_shifts_ob)
        dif = np.absolute(np.mean(p_shifts-p_shifts_ob))
        vec_dif.append(dif)
        axarr[j].plot(p_shifts-p_shifts_ob)
        axarr[j].axhline(0)
        axarr[j].set_title(fref.split('/')[-1]+'offset: '+str(np.around(dif,5)))
        if dif < mindif:
            mindif = dif
            difs = j
        print(j, dif)
        j+=1
    dct_shfts['vals']=np.array(vec_dif)
    dct_shfts['names']=np.array(vec_nam)

    refidx = difs
    p_shifts = list(p_shifts)
    savefig(fname,format='pdf')

    pickle.dump(dct_shfts, open(calibdir+'shifts.pkl','wb'))

elif len(sorted_ThAr_Ne_dates)>6:
    dct_shfts = pickle.load(open(calibdir+'shifts.pkl','rb'))
    I = np.argmin(dct_shfts['vals'])
    goodname = dct_shfts['names'][I]
    j = 0
    while j < len(sorted_ThAr_Ne_dates):
        if ThAr_Ne_ref[sorted_ThAr_Ne_dates[j]] == goodname:
            refidx = j
        j+=1
else:
    refidx = 0
print('This:',ThAr_Ne_ref[sorted_ThAr_Ne_dates[refidx]])
clf()
#"""
### start of science frame reductions ###
new_list         = []
new_list_obnames = []
new_sky         = []
new_sky_obnames = []
print('\n\tThe following targets will be processed:')
for i in range(len(simThAr_sci)):
    fsim = simThAr_sci[i]
    hd = pyfits.getheader(fsim)
    h  = pyfits.open(fsim)
    obname = h[0].header['OBJECT']
    if object2do.lower() == 'all':
        new_list.append(fsim)
        new_list_obnames.append( obname )
        print("\t\t"+obname)
    else:
        if (obname.lower() == object2do.lower()):
            new_list.append(fsim)
            new_list_obnames.append( obname )
            print("\t\t"+obname)

for i in range(len(simSky_sci)):
    fsim = simSky_sci[i]
    hd = pyfits.getheader(fsim)
    h  = pyfits.open(fsim)
    obname = h[0].header['OBJECT']
    if object2do.lower() == 'all':
        new_sky.append(fsim)
        new_sky_obnames.append( obname )
        print("\t\t"+obname)
    else:
        if (obname.lower() == object2do.lower()):
            new_sky.append(fsim)
            new_sky_obnames.append( obname )
            print("\t\t"+obname)

if n_useful>nord_ob:
    n_useful=nord_ob

comp_list = new_list + new_sky
moon_alts,moon_ills = {},{}

# Does any image have a special requirement for dealing with the moonlight?
if os.access(dirin + 'moon_corr.txt', os.F_OK):
    fmoon = open(dirin + 'moon_corr.txt','r')
    moon_lns = fmoon.readlines()
    spec_moon = []
    use_moon = []
    for line in moon_lns:
        spec_moon.append(line.split()[0])
        if line.split()[1] == '0':
            use_moon.append(False)
        else:
            use_moon.append(True)
else:
    spec_moon = []
    use_moon = []

spec_moon = np.array(spec_moon)
use_moon  = np.array(use_moon)

for fsim in comp_list:

    h        = pyfits.open(fsim)
    print('\n')
    print("\t--> Working on image: ", fsim)

    mjd,mjd0      = ferosutils.mjd_fromheader(h)
    ronoise, gain = ferosutils.get_RG(pyfits.getheader(fsim))

    # Object name
    obname    = h[0].header['OBJECT']
    observer  = h[0].header['OBSERVER']
    comp_type = h[0].header['HIERARCH ESO DPR TYPE'].split(',')[-1]

    print("\t\tObject name:",obname)
    print("\t\tComparison fiber is:", comp_type)

    # Open file, trim, overscan subtract and MasterBias subtract
    data = h[0].data
    data = ferosutils.OverscanTrim( data )
    if bad_colummn:
        data = ferosutils.b_col(data)
    data -= MasterBias
    #plot(data[:,2000])
    if dark_correction:
        dark = ferosutils.get_dark(np.around(h[0].header['EXPTIME']),dark_names,uni_times)
        #plot(dark[:,2000])
        #show()
        data -= dark
    data = data.T

    bacfile = dirout + 'BAC_' + fsim.split('/')[-1][:-4]+'fits'
    if (os.access(bacfile,os.F_OK))==False:
        Centers = np.zeros((len(c_all),data.shape[1]))
        for i in range(c_all.shape[0]):
            Centers[i,:]=np.polyval(c_all[i,:],np.arange(len(Centers[i,:])))
        bac = GLOBALutils.get_scat(data,Centers,span=10)
        hdbac = pyfits.PrimaryHDU( bac )
        hdbac.writeto(bacfile)
    else:
        bac = pyfits.getdata(bacfile)
    data -= bac

    ra          = h[0].header['RA']
    dec         = h[0].header['DEC']

    ra2,dec2 = GLOBALutils.getcoords(obname,mjd,filen=reffile)
    if ra2 !=0 and dec2 != 0:
        ra = ra2
        dec = dec2
    else:
        print('\t\tUsing the coordinates found in the image header.')

    altitude    =  2335.
    latitude    = -29.2543
    longitude   = -70.7346
    epoch       =  2000.000

    #bcvel_baryc, bjd = pyasl.helcorr(longitude, latitude, altitude, ra, dec, mjd+2400000.5, debug=True)
    #print corr, hjd
    #print gfd

    if not use_astropy:
        iers                    = GLOBALutils.JPLiers( baryc_dir, mjd-999.0, mjd+999.0 )
        obsradius, R0           = GLOBALutils.JPLR0( latitude, altitude)
        obpos                   = GLOBALutils.obspos( longitude, obsradius, R0 )

        jplephem.set_ephemeris_dir( baryc_dir , ephemeris )
        jplephem.set_observer_coordinates( float(obpos[0]), float(obpos[1]), float(obpos[2]) )

        res         = jplephem.doppler_fraction(float(ra/15.0), float(dec), long(mjd), float(mjd%1), 1, 0.0)
        lbary_ltopo = 1.0 + res['frac'][0]
        bcvel_baryc = ( lbary_ltopo - 1.0 ) * 2.99792458E5

        res  = jplephem.pulse_delay(ra/15.0, dec, int(mjd), mjd%1, 1, 0.0)
        mbjd = mjd + res['delay'][0] / (3600.0 * 24.0)

        # Moon Phase Calculations

        gobs      = ephem.Observer()
        gobs.name = 'Eso2.2'
        gobs.lat  = rad(latitude)  # lat/long in decimal degrees
        gobs.long = rad(longitude)
        gobs.date = h[0].header['DATE-OBS'][:10] + ' ' + h[0].header['DATE-OBS'][11:]
        mephem    = ephem.Moon()
        mephem.compute(gobs)
        Mcoo = jplephem.object_track("Moon", int(mjd), float(mjd%1), 1, 0.0)
        Mp   = jplephem.barycentric_object_track("Moon", int(mjd), float(mjd%1), 1, 0.0)
        Sp   = jplephem.barycentric_object_track("Sun", int(mjd), float(mjd%1), 1, 0.0)
        res  = jplephem.object_doppler("Moon", int(mjd), mjd%1, 1, 0.0)
        lunation,moon_state,moonsep,moonvel = GLOBALutils.get_lunar_props(ephem,gobs,Mcoo,Mp,Sp,res,ra,dec)
        refvel = bcvel_baryc + moonvel
        print('\t\tRadial Velocity of sacttered moonlight:',refvel)
        moonalt = float(mephem.alt)
        smoonalt = str(mephem.alt)


    else:
        from black_holes import heliocentric_correction, calculate_barycenter_correction_kms
        mbjd = heliocentric_correction(mjd_utc = mjd, ra = float(ra), dec = float(dec), site = 'lasilla') - mjd0
        bcvel_baryc = calculate_barycenter_correction_kms(float(ra), float(dec), mbjd + mjd0,site='lasilla')
        bjd_out= mbjd + mjd0 #added this line to circumvent NameError when post-processing
        refvel = -1
        moon_state = 'nan'
        lunation = -1
        moonsep = -1
        moonalt = -1
        smoonalt = '-1'


    print("\t\tBarycentric velocity:", bcvel_baryc)
    lbary_ltopo = bcvel_baryc/2.99792458E5 + 1


    ThAr_Ne_ref_m       = ThAr_Ne_ref
    ThAr_Ne_ref_dates_m = ThAr_Ne_ref_dates

    sorted_indices     = np.argsort( np.abs( np.array(ThAr_Ne_ref_dates_m) - mjd ) )
    sci_fits_ob        = dirout + fsim.split('/')[-1][:-4]+'spec.ob.fits.S'
    sci_fits_co        = dirout + fsim.split('/')[-1][:-4]+'spec.co.fits.S'
    sci_fits_ob_simple = dirout + fsim.split('/')[-1][:-4]+'spec.simple.ob.fits.S'
    sci_fits_co_simple = dirout + fsim.split('/')[-1][:-4]+'spec.simple.co.fits.S'

    if ( os.access(sci_fits_ob,os.F_OK) == False )        or ( os.access(sci_fits_co,os.F_OK) == False )  or \
    ( os.access(sci_fits_ob_simple,os.F_OK) == False ) or ( os.access(sci_fits_co_simple,os.F_OK) == False ) or \
    ( force_sci_extract ):
        print("\t\tNo previous extraction or extraction forced for science file", fsim, "extracting...")
        sci_Ss_ob = GLOBALutils.simple_extraction(data,c_ob,ext_aperture,min_extract_col,max_extract_col,npools)
        sci_Ss_co = GLOBALutils.simple_extraction(data,c_co,ext_aperture,min_extract_col,max_extract_col,npools)
        apsnr = np.sqrt(np.median(sci_Ss_ob[18//binning,1700//binning:2100//binning]))
        if apsnr < 50:
            NCosmic_Marsh = 10.
        else:
            NCosmic_Marsh = 100.

        sci_S_ob  = GLOBALutils.optimal_extraction(data,P_ob,c_ob,ext_aperture,ronoise,gain,S_Marsh,NCosmic_Marsh,min_extract_col,max_extract_col,npools)
        sci_S_co  = GLOBALutils.optimal_extraction(data,P_co,c_co,ext_aperture,ronoise,gain,S_Marsh,50.,min_extract_col,max_extract_col,npools)

        if (os.access(sci_fits_ob,os.F_OK)):
            os.remove( sci_fits_ob )
        if (os.access(sci_fits_co,os.F_OK)):
            os.remove( sci_fits_co )
        if (os.access(sci_fits_ob_simple,os.F_OK)):
            os.remove( sci_fits_ob_simple )
        if (os.access(sci_fits_co_simple,os.F_OK)):
            os.remove( sci_fits_co_simple )

        hdu = pyfits.PrimaryHDU( sci_S_ob )
        hdu = GLOBALutils.update_header(hdu,'OBJECT', obname)
        hdu = GLOBALutils.update_header(hdu,'HIERARCH MJD', mjd)
        hdu = GLOBALutils.update_header(hdu,'HIERARCH MBJD', mbjd)
        hdu = GLOBALutils.update_header(hdu,'HIERARCH SHUTTER START DATE', h[0].header['DATE-OBS'][:10] )
        hdu = GLOBALutils.update_header(hdu,'HIERARCH SHUTTER START UT',  h[0].header['DATE-OBS'][11:])
        hdu = GLOBALutils.update_header(hdu,'HIERARCH TEXP (S)',h[0].header['EXPTIME'])
        hdu = GLOBALutils.update_header(hdu,'HIERARCH BARYCENTRIC CORRECTION (KM/S)', bcvel_baryc)
        hdu = GLOBALutils.update_header(hdu,'HIERARCH (LAMBDA_BARY / LAMBDA_TOPO)', lbary_ltopo)
        hdu = GLOBALutils.update_header(hdu,'HIERARCH TARGET NAME', obname)
        hdu = GLOBALutils.update_header(hdu,'HIERARCH RA',h[0].header['RA'])
        hdu = GLOBALutils.update_header(hdu,'HIERARCH DEC',h[0].header['DEC'])
        hdu = GLOBALutils.update_header(hdu,'HIERARCH RA BARY',ra)
        hdu = GLOBALutils.update_header(hdu,'HIERARCH DEC BARY',dec)
        hdu = GLOBALutils.update_header(hdu,'HIERARCH EQUINOX',h[0].header['EQUINOX'])
        hdu = GLOBALutils.update_header(hdu,'HIERARCH OBS LATITUDE',h[0].header['HIERARCH ESO TEL GEOLAT'])
        hdu = GLOBALutils.update_header(hdu,'HIERARCH OBS LONGITUDE',h[0].header['HIERARCH ESO TEL GEOLON'])
        hdu = GLOBALutils.update_header(hdu,'HIERARCH OBS ALTITUDE',h[0].header['HIERARCH ESO TEL GEOELEV'])
        hdu = GLOBALutils.update_header(hdu,'HIERARCH TARG AIRMASS START',h[0].header['HIERARCH ESO TEL AIRM START'])
        hdu = GLOBALutils.update_header(hdu,'HIERARCH MOON_VEL',refvel,'[km/s]')
        hdu = GLOBALutils.update_header(hdu,'HIERARCH MOONST',moon_state)
        hdu = GLOBALutils.update_header(hdu,'HIERARCH LUNATION',lunation)
        hdu = GLOBALutils.update_header(hdu,'HIERARCH MOONSEP',moonsep)
        hdu = GLOBALutils.update_header(hdu,'HIERARCH MOONALT',moonalt)
        hdu = GLOBALutils.update_header(hdu,'HIERARCH SMOONALT',smoonalt)

        hdu.writeto( sci_fits_ob )
        hdu = pyfits.PrimaryHDU( sci_S_co )
        hdu.writeto( sci_fits_co )
        hdu = pyfits.PrimaryHDU( sci_Ss_ob )
        hdu.writeto( sci_fits_ob_simple )
        hdu = pyfits.PrimaryHDU( sci_Ss_co )
        hdu.writeto( sci_fits_co_simple )
    else:
        print("\t\t"+fsim+"has already been extracted, reading in product fits files...")
        sci_S_ob  = pyfits.getdata( sci_fits_ob )
        sci_S_co  = pyfits.getdata( sci_fits_co )
        sci_Ss_ob = pyfits.getdata( sci_fits_ob_simple )
        sci_Ss_co = pyfits.getdata( sci_fits_co_simple )

    fout = 'proc/' + obname + '_' + h[0].header['DATE-OBS'][:4] + h[0].header['DATE-OBS'][5:7] + h[0].header['DATE-OBS'][8:10] + '_' +'UT' + h[0].header['DATE-OBS'][11:] + '_sp.fits'

    #Build spectra
    if ( os.access(dirout+fout ,os.F_OK) == False ) or (force_spectral_file_build):
        spec = np.zeros((11, n_useful, data.shape[1]))
        hdu  = pyfits.PrimaryHDU( spec )
        hdu = GLOBALutils.update_header(hdu,'HIERARCH MJD', mjd)
        hdu = GLOBALutils.update_header(hdu,'HIERARCH MBJD', mbjd)
        hdu = GLOBALutils.update_header(hdu,'HIERARCH SHUTTER START DATE', h[0].header['DATE-OBS'][:10] )
        hdu = GLOBALutils.update_header(hdu,'HIERARCH SHUTTER START UT',  h[0].header['DATE-OBS'][11:])
        hdu = GLOBALutils.update_header(hdu,'HIERARCH TEXP (s)',h[0].header['EXPTIME'])
        hdu = GLOBALutils.update_header(hdu,'HIERARCH BARYCENTRIC CORRECTION (KM/S)', bcvel_baryc)
        hdu = GLOBALutils.update_header(hdu,'HIERARCH (LAMBDA_BARY / LAMBDA_TOPO)', lbary_ltopo)
        hdu = GLOBALutils.update_header(hdu,'HIERARCH TARGET NAME', obname)
        hdu = GLOBALutils.update_header(hdu,'HIERARCH RA',h[0].header['RA'])
        hdu = GLOBALutils.update_header(hdu,'HIERARCH DEC',h[0].header['DEC'])
        hdu = GLOBALutils.update_header(hdu,'HIERARCH RA BARY',ra)
        hdu = GLOBALutils.update_header(hdu,'HIERARCH DEC BARY',dec)
        hdu = GLOBALutils.update_header(hdu,'HIERARCH EQUINOX',h[0].header['EQUINOX'])
        hdu = GLOBALutils.update_header(hdu,'HIERARCH OBS LATITUDE',h[0].header['HIERARCH ESO TEL GEOLAT'])
        hdu = GLOBALutils.update_header(hdu,'HIERARCH OBS LONGITUDE',h[0].header['HIERARCH ESO TEL GEOLON'])
        hdu = GLOBALutils.update_header(hdu,'HIERARCH OBS ALTITUDE',h[0].header['HIERARCH ESO TEL GEOELEV'])
        hdu = GLOBALutils.update_header(hdu,'HIERARCH TARG AIRMASS START',h[0].header['HIERARCH ESO TEL AIRM START'])
        hdu = GLOBALutils.update_header(hdu,'HIERARCH MOON_VEL',refvel,'[KM/S]')
        hdu = GLOBALutils.update_header(hdu,'HIERARCH MOONST',moon_state)
        hdu = GLOBALutils.update_header(hdu,'HIERARCH LUNATION',lunation)
        hdu = GLOBALutils.update_header(hdu,'HIERARCH MOONSEP',moonsep)
        hdu = GLOBALutils.update_header(hdu,'HIERARCH MOONALT',moonalt)
        hdu = GLOBALutils.update_header(hdu,'HIERARCH SMOONALT',smoonalt)

        # get ThAr closest in time
        indice = sorted_ThAr_Ne_dates[refidx]
        hdu = GLOBALutils.update_header(hdu,'HIERARCH THAR REF',ThAr_Ne_ref_m[indice].split('/')[-1][:-5]+'_sp_ob.fits')
        hdu = GLOBALutils.update_header(hdu,'HIERARCH THAR REF CO',ThAr_Ne_ref_m[indice].split('/')[-1][:-5]+'_sp_co.fits')
        thar_fits_ob = calibdir + ThAr_Ne_ref_m[indice].split('/')[-1][:-4]+'spec.ob.fits.S'
        thar_fits_co = calibdir + ThAr_Ne_ref_m[indice].split('/')[-1][:-4]+'spec.co.fits.S'
        pkl_wsol = calibdir + ThAr_Ne_ref_m[indice].split('/')[-1][:-4]+'wavsolpars.pkl'
        print("\t\tUnpickling wavelength solution from", pkl_wsol, " ...")
        wsol_dict = pickle.load(open(pkl_wsol,'rb'))

        if comp_type == 'WAVE':
            # Extract ThAr lines from comparison orders
            lines_thar_co  = sci_S_co[:,1,:]
            #lines_thar_co  = sci_Ss_co
            iv_thar_co     = sci_S_co[:,2,:]

            All_Pixel_Centers_co = np.array([])
            All_Wavelengths_co   = np.array([])
            All_Orders_co        = np.array([])
            All_Centroids_co     = np.array([])
            All_Sigmas_co        = np.array([])
            All_Intensities_co   = np.array([])
            All_residuals_co     = np.array([])

            if lines_thar_co.shape[0] < n_useful:
                n_useful = lines_thar_co.shape[0]
            order = o0
            c_p2w_c = []
            shifts = []

            while order < min(o0 + n_useful, lines_thar_co.shape[0]): #Same here to prevent IndexError. Originally order < o0 + n_useful

                order_s = str(order+1)
                if (order < 9):
                    order_s = '0'+str(order+1)
                thar_order_orig = lines_thar_co[order,:] #/ S_flat_co_n[order,1,:]
                IV              = iv_thar_co[order,:]
                wei             = np.sqrt( IV )
                bkg             = ferosutils.Lines_mBack(thar_order_orig, IV,  thres_rel=3)
                thar_order      = thar_order_orig - bkg

                coeffs_pix2wav, coeffs_pix2sigma, pixel_centers, wavelengths,\
                    rms_ms, residuals, centroids, sigmas, intensities =\
                                         GLOBALutils.Initial_Wav_Calibration(order_dir+'order_'+\
                                         order_s+thar_end, thar_order, order, wei, rmsmax=rmsmax*binning, \
                                         minlines=30,FixEnds=False,Dump_Argon=dumpargon,\
                                         Dump_AllLines=True, Cheby=use_cheby,porder=5,del_width=3.5/binning,binning=binning)
                All_Pixel_Centers_co = np.append( All_Pixel_Centers_co, pixel_centers )
                All_Wavelengths_co   = np.append( All_Wavelengths_co, wavelengths )
                All_Orders_co        = np.append( All_Orders_co, np.zeros( len(pixel_centers) ) + order )
                All_Sigmas_co        = np.append( All_Sigmas_co, sigmas)
                All_Intensities_co   = np.append( All_Intensities_co, intensities )
                All_residuals_co     = np.append( All_residuals_co, residuals )
                order+=1

            p1_co, G_pix_co, G_ord_co, G_wav_co, II_co, rms_ms_co, G_res_co = \
                            GLOBALutils.Fit_Global_Wav_Solution(All_Pixel_Centers_co, All_Wavelengths_co, All_Orders_co,\
                                                np.ones(All_Intensities_co.shape), wsol_dict['p1_co'], Cheby=use_cheby,\
                                                maxrms=MRMS, Inv=Inverse_m,minlines=minlines,order0=OO0, \
                                                ntotal=n_useful,npix=len(thar_order),nx=ncoef_x,nm=ncoef_m)

            p_shift, pix_centers, orders, wavelengths, I, rms_ms, residuals  = \
                            GLOBALutils.Global_Wav_Solution_vel_shift(All_Pixel_Centers_co,\
                            All_Wavelengths_co, All_Orders_co, np.ones(len(All_Wavelengths_co)), wsol_dict['p1_co'],\
                            minlines=1000, maxrms=MRMS,order0=OO0, ntotal=n_useful,\
                            Cheby=use_cheby, Inv=Inverse_m, npix=len(thar_order),nx=ncoef_x,nm=ncoef_m)

            precision    = rms_ms/np.sqrt(len(I))
            good_quality = True
            if (precision > 5):
                good_quality = False
                #p_shift = 0. kareem edit
            p_shifts.append(p_shift)
            p_mjds.append(mjd)

            spec_co = np.zeros((2,n_useful,len(thar_order)))
            equis = np.arange( len(thar_order) )
            order = o0

            while order < min(o0 + n_useful, lines_thar_co.shape[0]): #Here as well to prevent IndexError crashes
                oss = order - o0
                m   = order + OO0
                chebs = GLOBALutils.Calculate_chebs(equis, m, order0=OO0,\
                            ntotal=n_useful, npix=len(equis), Inverse=Inverse_m,nx=ncoef_x,nm=ncoef_m)
                WavSol_co = (1.0/float(m)) * GLOBALutils.Joint_Polynomial_Cheby(p1_co,chebs,ncoef_x,ncoef_m)
                spec_co[0,oss,:] = WavSol_co
                spec_co[1,oss,:] = lines_thar_co[order]
                order+=1
            hdu_co = pyfits.PrimaryHDU(spec_co)
            if os.access(dirout + fsim.split('/')[-1][:-5]+'_sp_co.fits',os.F_OK):
                os.system('rm ' + dirout + fsim.split('/')[-1][:-5]+'_sp_co.fits')
            hdu_co.writeto(dirout + fsim.split('/')[-1][:-5]+'_sp_co.fits')
            hdu = GLOBALutils.update_header(hdu,'HIERARCH THAR CO',fsim.split('/')[-1][:-5]+'_sp_co.fits')


            hdu = GLOBALutils.update_header(hdu,'HIERARCH GOOD QUALITY WAVSOL', good_quality)
            hdu = GLOBALutils.update_header(hdu,'HIERARCH WAVSOL ERROR', precision, '[m/s]')

        # Apply new wavelength solution including barycentric correction
        equis = np.arange( data.shape[1] )
        order = o0
        temp_spec = np.zeros((n_useful,data.shape[1]))
        while order < min(n_useful+o0, sci_S_ob.shape[0]): #Again, IndexError was raised here
            oss = order - o0
            m   = order + OO0
            chebs = GLOBALutils.Calculate_chebs(equis, m, order0=OO0,\
                ntotal=n_useful, npix=len(equis), Inverse=Inverse_m,nx=ncoef_x,nm=ncoef_m)

            if comp_type == 'WAVE':
                WavSol = lbary_ltopo * (1.0 + 1.0e-6*p_shift) * (1.0/float(m)) * \
                GLOBALutils.Joint_Polynomial_Cheby(wsol_dict['p1'],chebs,ncoef_x,ncoef_m)
            else:
                WavSol = lbary_ltopo * (1.0/float(m)) * \
                GLOBALutils.Joint_Polynomial_Cheby(wsol_dict['p1'],chebs,ncoef_x,ncoef_m)

            spec[0,oss,:] = WavSol
            spec[1,oss,:] = sci_S_ob[order,1, :]
            spec[1,oss,:lim_iz[oss]] = 0.
            spec[2,oss,:] = sci_S_ob[order,2, :]
            fn  = S_flat_ob_n[order,1,:]
            fn2 = S_flat_co_n[order,1,:]

            if comp_type == 'WAVE':
                L  = np.where( fn == 0 )[0]
                spec[3,oss,:] = spec[1,oss,:] / S_flat_ob_n[order,1,:]
                spec[4,oss,:] = sci_S_ob[order,2,:] * ( S_flat_ob_n[order,1,:] ** 2 )
                spec[3,oss,L] = 0.
                spec[4,oss,L] = 0.
            else:
                L1  = np.where( fn == 0 )[0]
                L2  = np.where( fn2 == 0 )[0]
                L = np.unique(np.hstack([L1,L2]))
                spec[3,oss,:] = spec[1,oss,:] / S_flat_ob_n[order,1,:] - sci_S_co[order,1] /  S_flat_co_n[order,1,:]
                spec[4,oss,:] = sci_S_ob[order,2,:] * ( S_flat_ob_n[order,1,:] ** 2 )   #OJO cambiar esto
                spec[3,oss,L] = 0.
                spec[4,oss,L] = 0.
            order+=1

        ron  = ronoise
        gain = gain
        order = o0

        while order < min(n_useful+o0, S_flat_ob_n.shape[0]): #same here
            oss = order-o0
            L  = np.where( spec[1,oss] != 0 )
            ccoef = GLOBALutils.get_cont_single(spec[0,oss],spec[3,oss],spec[4,oss],ll=1.5,lu=5,nc=nconts[oss])
            spec[5,oss,:][L] = spec[3,oss,L] / np.polyval(ccoef,spec[0,oss][L])
            #plot(spec[0,oss,:],spec[3,oss,:])
            #plot(spec[0,oss,:][L],np.polyval(ccoef,spec[0,oss][L]))
            nJ = np.where(np.isnan(spec[5,oss])==True)[0]
            nJ2 = np.where(np.isinf(spec[5,oss])==True)[0]
            spec[5,oss,nJ] = 1.0
            spec[5,oss,nJ2] = 1.0
            ratio            = spec[3,oss,:][L] / spec[5,oss,:][L]
            spec[6,oss,:][L] = spec[4,oss,:][L] * (ratio ** 2 )
            spec[7,oss,:][L] = ratio
            spec[8,oss,:][L] = ratio * S_flat_ob_n[order,1,:][L] / np.sqrt( ratio * S_flat_ob_n[order,1,:][L] / gain + (ronoise/gain)**2 )
            #print spec[8,oss,1890:2010]
            # masking stron emission lines
            # rI = np.where(spec[5,oss] > 1. + 8./spec[8,oss])
            # spec[5,oss,rI] = 1.
            # rI = np.where(spec[5,oss] < - (1. + 8./spec[8,oss]))
            # spec[5,oss,rI] = 1.
            spl           = scipy.interpolate.splrep(np.arange(WavSol.shape[0]), WavSol,k=3)
            dlambda_dx    = scipy.interpolate.splev(np.arange(WavSol.shape[0]), spl, der=1)
            NN            = np.average(dlambda_dx)
            dlambda_dx    /= NN

            spec[9,oss,:][L]  = spec[5,oss,:][L] * (dlambda_dx[L] ** 1)
            spec[10,oss,:][L] = spec[6,oss,:][L] / (dlambda_dx[L] ** 2)
            order+=1
        #show()
        if os.access(dirout + fout, os.F_OK):
            os.remove(dirout + fout)

        # Merge the orders
        wl_grid, flux_grid, weight_grid, merged_spec = GLOBALutils.join_orders(spec)

        # Create a new PrimaryHDU with the original header
        hdu_primary = pyfits.PrimaryHDU(header=hdu.header)

        # Create ImageHDU objects for each of the arrays
        hdu_wavelength = pyfits.ImageHDU(spec[0,:,:], name='Wavelength')
        hdu_extracted_flux = pyfits.ImageHDU(spec[1,:,:], name='Extracted Flux')
        hdu_error_extracted_flux = pyfits.ImageHDU(spec[2,:,:], name='Measurement of the error in the extracted flux [1./sqrt(Var)]')
        hdu_flux_blaze_corrected = pyfits.ImageHDU(spec[3,:,:], name='Blaze corrected Flux')
        hdu_error_blaze_corrected = pyfits.ImageHDU(spec[4,:,:], name='Measurement of the error in the blaze corrected flux')
        hdu_flux_normalized = pyfits.ImageHDU(spec[5,:,:], name='Continuum normalized Flux')
        hdu_error_normalized = pyfits.ImageHDU(spec[6,:,:], name='Measurement of the Error in the continuum normalized Flux')
        hdu_continuum = pyfits.ImageHDU(spec[7,:,:], name='Estimated Continuum')
        hdu_snr = pyfits.ImageHDU(spec[8,:,:], name='Signal-to-noise ratio')
        hdu_flux_derivative = pyfits.ImageHDU(spec[9,:,:], name='Continumm normalized flux multiplied by the derivative of the wavelength with respect to the pixels')
        hdu_error_flux_derivative = pyfits.ImageHDU(spec[10,:,:], name='Corresponding error of the 9th entrance')

        # Create new ImageHDUs for the merged data
        hdu_wl_grid = pyfits.ImageHDU(wl_grid, name='Wavelength Grid')
        hdu_flux_grid = pyfits.ImageHDU(flux_grid, name='Flux Grid')
        hdu_weight_grid = pyfits.ImageHDU(weight_grid, name='Weight Grid')
        hdu_merged_spec = pyfits.ImageHDU(merged_spec, name='Merged Spectrum')

        # Create a HDUList and write to a FITS file
        hdul = pyfits.HDUList([hdu_primary, hdu_wavelength, hdu_extracted_flux, hdu_error_extracted_flux, hdu_flux_blaze_corrected, hdu_error_blaze_corrected, hdu_flux_normalized, hdu_error_normalized, hdu_continuum, hdu_snr, hdu_flux_derivative, hdu_error_flux_derivative, hdu_wl_grid, hdu_flux_grid, hdu_weight_grid, hdu_merged_spec])
        hdul.writeto(dirout + fout)

p_mjds,p_shifts = np.array(p_mjds),np.array(p_shifts)
I = np.argsort(p_mjds)
p_mjds,p_shifts = p_mjds[I],p_shifts[I]

#plot(p_mjds,p_shifts,'ro')
#show()

if len(p_mjds) > 3:
    tck_shift = scipy.interpolate.splrep(p_mjds,p_shifts,k=1)
elif len(p_mjds) > 1:
    tck_shift = scipy.interpolate.splrep(p_mjds,p_shifts,k=1)

for fsim in new_sky:
    h        = pyfits.open(fsim)
    obname   = h[0].header['OBJECT']
    mjd,mjd0 = ferosutils.mjd_fromheader(h)
    if len(p_mjds) > 1:
        p_shift  = scipy.interpolate.splev(mjd,tck_shift)
    else:
        p_shift = 0.
    fout = 'proc/' + obname + '_' + h[0].header['DATE-OBS'][:4] + h[0].header['DATE-OBS'][5:7] +\
            h[0].header['DATE-OBS'][8:10] + '_' +'UT' + h[0].header['DATE-OBS'][11:] + '_sp.fits'
    hdu = pyfits.open(dirout + fout,mode='update')
    # hdu[0].data[0,:,:] *= (1.0 + 1.0e-6*p_shift)
    hdu['Wavelength'].data *= (1.0 + 1.0e-6*p_shift)
    hdu.flush()
    hdu.close()

print("\n\t Starting with the post-processing:")
#JustExtract = True
if (not JustExtract):
    for fsim in comp_list:
        know_moon = False
        #if fsim.split('/')[-1] in spec_moon:
        #    I = np.where(fsim.split('/')[-1] == spec_moon)[0]
        #    know_moon = True
        #    here_moon = use_moon[I]
        h        = pyfits.open(fsim)
        obname   = h[0].header['OBJECT']
        mjd,mjd0 = ferosutils.mjd_fromheader(h)
        #gobs.date = h[0].header['DATE-OBS'][:10] + ' ' + h[0].header['DATE-OBS'][11:]
        #mephem    = ephem.Moon()
        #mephem.compute(gobs)

        fout = 'proc/' + obname + '_' + h[0].header['DATE-OBS'][:4] + h[0].header['DATE-OBS'][5:7] +\
        h[0].header['DATE-OBS'][8:10] + '_' +'UT' + h[0].header['DATE-OBS'][11:] + '_sp.fits'

        print("\n\t--> Working on spectrum: ", fout)

        hdu    = pyfits.open(dirout + fout,mode='update')
        # spec   = hdu[0].data
        spec_list = []
        for ext in hdu[1:12]:
            spec_list.append(ext.data)
        spec = np.array(spec_list)
        # print('spec.shape =', spec.shape)
        refvel = hdu[0].header['MOON_VEL']
        mbjd   = hdu[0].header['MBJD']
        bjd_out = mbjd + mjd0
        lbary_ltopo = hdu[0].header['(LAMBDA_BARY / LAMBDA_TOPO)']
        lunation    = hdu[0].header['LUNATION']
        moonsep     = hdu[0].header['MOONSEP']
        moon_state  = hdu[0].header['MOONST']

        SNR_5130 = np.median(spec[8,10,1900//binning:2101//binning] )
        if SNR_5130 < 1.:
            SNR_5130 = 1.

        if DoClass:
            # spectral analysis
            print("\t\tSpectral Analysis...")
            query_success = False
            # First, query SIMBAD with the object name
            query_success,sp_type_query = GLOBALutils.simbad_query_obname(obname)
            # Now, query SIMBAD by coordinates if above not successful
            if (not query_success):
                query_success,sp_type_query = GLOBALutils.simbad_query_coords('12:00:00','00:00:00')
            print("\t\t\tSpectral type returned by SIMBAD query:",sp_type_query)

            hdu[0] = GLOBALutils.update_header(hdu[0],'HIERARCH SIMBAD SPTYP', sp_type_query)
            pars_file = dirout + fsim.split('/')[-1][:-4]+'_stellar_pars.txt'
            if os.access(pars_file,os.F_OK) == False or force_stellar_pars:
                print("\t\t\tEstimating atmospheric parameters:")
                Rx = np.around(1./np.sqrt(1./40000.**2 - 1./50000**2))
                spec2 = spec.copy()
                for i in range(spec.shape[1]):
                    IJ = np.where(spec[5,i]!=0.)[0]
                    spec2[5,i,IJ] = GLOBALutils.convolve(spec[0,i,IJ],spec[5,i,IJ],Rx)
                T_eff, logg, Z, vsini, vel0, ccf = correlation.CCF(spec2,model_path=models_path,npools=npools)
                line = "%6d %4.1f %4.1f %8.1f %8.1f\n" % (T_eff,logg, Z, vsini, vel0)
                f = open(pars_file,'w')
                f.write(line)
                f.close()
            else:
                print("\t\t\tAtmospheric parameters loaded from file:")
                T_eff, logg, Z, vsini, vel0 = np.loadtxt(pars_file,unpack=True)

            print("\t\t\t\tT_eff=",T_eff,"log(g)=",logg,"Z=",Z,"vsin(i)=",vsini,"vel0",vel0)

        else:
            T_eff, logg, Z, vsini, vel0 = -999,-999,-999,-999,-999
        T_eff_epoch = T_eff
        logg_epoch  = logg
        Z_epoch     = Z
        vsini_epoch = vsini
        vel0_epoch  = vel0
        hdu[0] = GLOBALutils.update_header(hdu[0],'HIERARCH TEFF', float(T_eff))
        hdu[0] = GLOBALutils.update_header(hdu[0],'HIERARCH LOGG', float(logg))
        hdu[0] = GLOBALutils.update_header(hdu[0],'HIERARCH Z', Z)
        hdu[0] = GLOBALutils.update_header(hdu[0],'HIERARCH VSINI', vsini)
        hdu[0] = GLOBALutils.update_header(hdu[0],'HIERARCH VEL0', vel0)

        print("\t\tRadial Velocity analysis:")
        # assign mask
        sp_type, mask = GLOBALutils.get_mask_reffile(obname,reffile=reffile,base=base+'data/xc_masks/')
        print("\t\t\tWill use",sp_type,"mask for CCF.")
        velw  = 300
        velsh = 3.
        # Read in mask
        ml, mh, weight = np.loadtxt(mask,unpack=True)
        ml_v = GLOBALutils.ToVacuum( ml )
        mh_v = GLOBALutils.ToVacuum( mh )
        # make mask larger accounting for factor ~2.5 lower res in FEROS w/r to HARPS
        av_m = 0.5*( ml_v + mh_v )
        ml_v -= 1.5*(av_m - ml_v)
        mh_v += 1.5*(mh_v - av_m)
        mask_hw_kms = (GLOBALutils.Constants.c/1e3) * 0.5*(mh_v - ml_v) / av_m
        disp = GLOBALutils.get_disp(obname, reffile=reffile)
        # print('disp =',disp, 'vsini =', vsini)
        if disp == 0:
            known_sigma = False
            if vsini != -999 and vsini != 0.:
                disp = vsini
            else:
                disp = 3.
        else:
            known_sigma = True
        # print('disp =',disp)
        mask_hw_wide = av_m * disp / (GLOBALutils.Constants.c/1.0e3)
        ml_v = av_m - mask_hw_wide
        mh_v = av_m + mask_hw_wide

        print('\t\t\tComputing the CCF...')
        boot_ind = np.arange(spec.shape[1])
        #boot_ind = np.arange(5,18,1)
        try:
            if True:
                cond = True
                while (cond):
                    # first rough correlation to find the minimum
                    vels, xc_full, sn, nlines_ccf, W_ccf = \
                            GLOBALutils.XCor(spec, ml_v, mh_v, weight, 0, lbary_ltopo, vel_width=velw,vel_step=velsh,\
                                                spec_order=9,iv_order=10,sn_order=8,max_vel_rough=velw)

                    xc_av = GLOBALutils.Average_CCF(xc_full, sn, sn_min=3., Simple=True, W=W_ccf,boot_ind=boot_ind)
                    outt = np.vstack((vels,xc_av))
                    yy     = scipy.signal.medfilt(xc_av,11)
                    pred = lowess(yy, vels,frac=0.4,it=10,return_sorted=False)
                    tck1 = scipy.interpolate.splrep(vels,pred,k=1)
                    xc_av_orig = xc_av.copy()
                    xc_av /= pred
                    vel0_xc = vels[ np.argmin( xc_av ) ]
                    rvels, rxc_av, rpred, rxc_av_orig, rvel0_xc = vels.copy(), xc_av.copy(), pred.copy(), xc_av_orig.copy(), vel0_xc
                    xc_av_rough = xc_av
                    vels_rough  = vels

                    vel_width = np.maximum( 20.0, 6*disp )
                    # print('0) vel_width before calling the 2nd XCor =', vel_width)
                    vels, xc_full, sn, nlines_ccf, W_ccf =\
                            GLOBALutils.XCor(spec, ml_v, mh_v, weight, vel0_xc, lbary_ltopo, vel_width=vel_width,vel_step=0.2,\
                                                spec_order=9,iv_order=10,sn_order=8,max_vel_rough=velw)

                    xc_av = GLOBALutils.Average_CCF(xc_full, sn, sn_min=3., Simple=True, W=W_ccf,boot_ind=boot_ind)
                    pred = scipy.interpolate.splev(vels,tck1)
                    xc_av /= pred

                    if sp_type == 'M5':
                        moon_sig = 2.5
                    elif sp_type == 'K5':
                        moon_sig = 3.3
                    else:
                        moon_sig = 4.5
                        moon_sig = 0.6*np.sqrt(1.**2+disp**2)

                    p1,XCmodel,p1gau,XCmodelgau,Ls2 = GLOBALutils.XC_Final_Fit( vels, xc_av , sigma_res = 4, horder=8, moonv = refvel, moons = moon_sig, moon = False)
                    moonmatters = False
                    if (know_moon and here_moon):
                        moonmatters = True
                        ismoon = True
                        confused = False
                        p1_m,XCmodel_m,p1gau_m,XCmodelgau_m,Ls2_m = GLOBALutils.XC_Final_Fit( vels, xc_av , sigma_res = 4, horder=8, moonv = refvel, moons = moon_sig, moon = True)
                        moon_flag = 1
                    else:
                        confused = False
                        ismoon = False
                        p1_m,XCmodel_m,p1gau_m,XCmodelgau_m,Ls2_m = p1,XCmodel,p1gau,XCmodelgau,Ls2
                        moon_flag = 0

                    bspan = GLOBALutils.calc_bss(vels,xc_av)
                    SP = bspan[0]
                    SP2 = GLOBALutils.calc_bss2(vels,xc_av,p1gau)
                    # print('3) disp before p1gau =', disp)
                    # print('p1gau =', p1gau)
                    if (not known_sigma):
                        disp = np.floor(p1gau[2])
                        if (disp < 3.0):
                            disp = 3.0
                        mask_hw_wide = av_m * disp / (GLOBALutils.Constants.c/1.0e3)
                        ml_v = av_m - mask_hw_wide
                        mh_v = av_m + mask_hw_wide
                        known_sigma = True
                    else:
                        cond = False
                    # print('4) disp value before finishing the while loop =', disp, 'vel_width =', vel_width)
                BSerr = -999.00

                xc_dict = {'vels':vels,'xc_av':xc_av,'XCmodelgau':XCmodelgau,'Ls2':Ls2,'refvel':refvel,\
                        'rvels':rvels,'rxc_av':rxc_av,'rpred':rpred,'rxc_av_orig':rxc_av_orig,\
                        'rvel0_xc':rvel0_xc,'xc_full':xc_full, 'p1':p1, 'sn':sn, 'p1gau':p1gau,\
                        'p1_m':p1_m,'XCmodel_m':XCmodel_m,'p1gau_m':p1gau_m,'Ls2_m':Ls2_m,\
                        'XCmodelgau_m':XCmodelgau_m,'W_ccf':W_ccf}

                moon_dict = {'moonmatters':moonmatters,'moon_state':moon_state,'moonsep':moonsep,\
                            'lunation':lunation,'mephem':-1,'texp':h[0].header['EXPTIME']}

                pkl_xc = dirout + fsim.split('/')[-1][:-4]+obname+'_XC_'+sp_type+'.pkl'
                pickle.dump( xc_dict, open( pkl_xc, 'wb' ) )

                ccf_pdf = dirout + 'proc/' + fsim.split('/')[-1][:-4] + obname + '_XCs_' + sp_type + '.pdf'


                if not avoid_plot:
                    GLOBALutils.plot_CCF(xc_dict,moon_dict,path=ccf_pdf)

                airmass  = h[0].header['ESO TEL AIRM START']

            seeing   = h[0].header['ESO TEL AMBI FWHM START']
            #moonsep  = h[0].header['ESO TEL MOON DIST']
            TEXP = h[0].header['EXPTIME']

            if sp_type=='G2':
                if T_eff < 6000:
                    A = 0.11081
                    B = 0.0016
                    D = 0.32815
                    C = 0.00453
                else:
                    A = 0.11081
                    B = 0.0016
                    D = 0.32815
                    C = 0.00453
            elif  sp_type == 'K5':
                A = 0.08900
                B = 0.00311
                D = 0.27404
                C = 0.00433
            else:
                A = 0.08900
                B = 0.00311
                D = 0.27404
                C = 0.00433

            RVerr =  B + ( 1.6 + 0.2 * p1gau[2] ) * A / np.round(SNR_5130)
            BSerr = D / float(np.round(SNR_5130)) + C

            RVerr =  B + (1.6+0.2*p1gau[2])*A/np.round(SNR_5130)
            depth_fact = 1. + p1gau[0]/(p1gau[2]*np.sqrt(2*np.pi))
            if depth_fact < 0.6:
                depth_fact = 0.6

            if depth_fact >= 1.:
                RVerr2 = -999.000
            else:
                depth_fact = (1 - 0.6) / (1 - depth_fact)
                RVerr2 = RVerr * depth_fact

            if (RVerr2 <= 0.007):
                RVerr2 = 0.007

            RV     = np.around(p1gau_m[1],4)
            BS     = np.around(SP,4)
            BS2     = np.around(SP2,4)
            RVerr2 = np.around(RVerr2,4)
            BSerr  = np.around(BSerr,4)

            print('\t\t\tRV = '+str(RV)+' +- '+str(RVerr2))
            print('\t\t\tBS = '+str(BS)+' +- '+str(BSerr))
            print('\t\t\tBS2 = '+str(BS2)+' +- '+str(BSerr))
            T_eff_err = 100
            logg_err = 0.5
            Z_err = 0.5
            vsini_err = 2
            XC_min = np.abs(np.around(np.min(XCmodel),2))

            SNR_5130 = np.around(SNR_5130)
            SNR_5130_R = np.around(SNR_5130*np.sqrt(2.3))
            # write to output
            disp_epoch = np.around(p1gau_m[2],1)
            hdu[0] = GLOBALutils.update_header(hdu[0],'RV', RV)
            hdu[0] = GLOBALutils.update_header(hdu[0],'RV_E', RVerr2)
            hdu[0] = GLOBALutils.update_header(hdu[0],'BS', BS)
            hdu[0] = GLOBALutils.update_header(hdu[0],'BS_E', BSerr)
            hdu[0] = GLOBALutils.update_header(hdu[0],'DISP', disp_epoch)
            hdu[0] = GLOBALutils.update_header(hdu[0],'SNR', SNR_5130)
            hdu[0] = GLOBALutils.update_header(hdu[0],'SNR_R', SNR_5130_R)
            hdu[0] = GLOBALutils.update_header(hdu[0],'INST', 'FEROS')
            hdu[0] = GLOBALutils.update_header(hdu[0],'RESOL', '50000')
            hdu[0] = GLOBALutils.update_header(hdu[0],'PIPELINE', 'CERES')
            hdu[0] = GLOBALutils.update_header(hdu[0],'XC_MIN', XC_min)
            hdu[0] = GLOBALutils.update_header(hdu[0],'BJD_OUT', bjd_out)

            line_out = "%-15s %18.8f %9.4f %7.4f %9.3f %5.3f   feros   ceres   50000 %6d %5.2f %5.2f %5.1f %4.2f %5.2f %6.1f %4d %s\n"%\
                        (obname, bjd_out, RV, RVerr2, BS2, BSerr, T_eff_epoch, logg_epoch, Z_epoch, vsini_epoch, XC_min, disp_epoch,\
                        TEXP, SNR_5130_R, ccf_pdf)
            f_res.write(line_out)
            GLOBALutils.write_log(hdu[0].header, log_dir=dirout)
            hdu.close()
        except MemoryError:
            print("MemoryError encountered. Skipping post-processing for spectrum: ", fsim)
            GLOBALutils.write_log(hdu[0].header, notes='Post-processing aborted due to MemoryError', log_dir=dirout)
            hdu.close()
GLOBALutils.save_log_to_csv(log_dir=dirout)
f_res.close()
