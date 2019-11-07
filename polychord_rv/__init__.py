import os
import sys
import numpy as np
import importlib
import datetime
import time
import shutil

# MPI imports
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# PolyChord imports
import pypolychord as polychord
import pypolychord.settings as polysettings

from .config import read_config

HOME = os.getenv('HOME')


def runpoly(configfile, nlive=None, nplanets=None, modelargs={}, **kwargs):

    # Read dictionaries from configuration file
    rundict, datadict, priordict, fixeddict, priors = read_config(configfile, nplanets)
    parnames = list(priordict.keys())

    # Import model module
    models_path = os.path.join(os.getenv('HOME'), 'run/targets/{target}/models/{runid}'.format(**rundict))
    modulename = 'model_{target}_{runid}'.format(**rundict)
    sys.path.insert(0, models_path)
    mod = importlib.import_module(modulename) # modulename, models_path)

    # Instantiate model class (pass additional arguments)
    mymodel = mod.Model(fixeddict, datadict, parnames, **modelargs)

    # Function to convert from hypercube to physical parameter space
    def prior(hypercube):
        """ Priors for each parameter. """
        theta = []
        for i, x in enumerate(hypercube):
            theta.append(priordict[parnames[i]].ppf(x))
        return theta    

    # LogLikelihood
    def loglike(x):
        return (mymodel.lnlike(x), [])

    # Prepare run
    nderived = 0
    ndim = len(parnames)

    # Starting time to identify this specific run
    # Define it only for rank 0 and broadcoast to the rest
    if rank == 0:
        isodate = datetime.datetime.today().isoformat()
    else:
        isodate = None
    isodate = comm.bcast(isodate, root=0)

    # Define PolyChord settings
    settings = polysettings.PolyChordSettings(ndim, nderived, )
    settings.do_clustering = True
    if nlive is None:
        settings.nlive = 25*ndim
    else:
        settings.nlive = nlive

    # Define fileroot name (identifies this specific run)
    fileroot = rundict['target']+'_'+rundict['runid']
    if rundict['comment'] != '':
        fileroot += '_'+rundict['comment']

    # Label the run with nr of planets, live points, nr of cores, sampler and date
    fileroot += '_k{}'.format(mymodel.nplanets)
    fileroot += '_nlive{}'.format(settings.nlive)
    fileroot += '_ncores{}'.format(size)
    fileroot += '_polychord'
    fileroot += '_'+isodate

    settings.file_root = fileroot
    settings.read_resume = False
    settings.num_repeats = ndim * 5
    settings.feedback = 1
    settings.precision_criterion = 0.01
    # Base directory
    ref_dir = os.path.join('ExP', rundict['target'], rundict['runid'], fileroot, 'polychains')
    if 'spectro' in HOME:
        # If it's runing in cluster -> save in scratch folder
        base_dir = os.path.join('/scratch/nunger', ref_dir)
    else:
        # Running locally
        base_dir = os.path.join(HOME, ref_dir)
    settings.base_dir = base_dir

    # Initialise clocks
    ti = time.process_time()

    # ----- Run PolyChord ------
    output = polychord.run_polychord(loglike, ndim, nderived, settings, prior)
    # --------------------------

    # Stop clock
    tf = time.process_time()
    run_time = datetime.timedelta(seconds=tf-ti)

    # Cleanup of parameter names
    paramnames = [(x, x) for x in parnames]
    output.make_paramnames_files(paramnames)
    parnames.insert(0, 'loglike')
    parnames.insert(0, 'weight')
    old_cols = output.samples.columns.values.tolist()
    output.samples.rename(columns=dict(zip(old_cols, parnames)), inplace=True)

    # Assign additional parameters to output
    output.runtime = run_time
    output.target = rundict['target']
    output.runid = rundict['runid']
    output.comment = rundict.get('comment', '')
    output.nplanets = mymodel.nplanets
    output.nlive = settings.nlive
    output.nrepeats = settings.num_repeats
    output.isodate = isodate
    output.ncores = size
    output.priors = priors
    output.starparams = rundict['star_params']
    output.datadict = datadict
    output.parnames = parnames
    output.fixeddict = fixeddict

    if rank == 0:
        # Print run time
        print('\nTotal run time was: {}'.format(run_time))

        # Save output as pickle file
        dump2pickle_poly(output)

        # Copy post processing script to this run's folder
        shutil.copy(os.path.join(HOME,'run/post_processing.py'), os.path.join(output.base_dir, '..'))

        # Copy model file to this run's folder
        model = os.path.join(models_path, modulename+'.py')
        shutil.copy(model, os.path.join(output.base_dir, '..'))

    return output


def dump2pickle_poly(output, savedir=None):
    """ Takes the output from PolyChord and saves it as a pickle file. """

    try:
        import pickle
    except ImportError:
        print('Install pickle to save the output. Try running:\n pip install pickle')
        return

    if savedir is None:
        # Save directory in parent of base dir
        pickledir = os.path.join(output.base_dir, '..')
    else:
        # Unless specified otherwhise
        pickledir = savedir

    # Create directory if it doesn't exist.
    os.makedirs(pickledir, exist_ok=True)

    full_path = os.path.join(pickledir, output.file_root+'.dat')
    with open(full_path, 'wb') as f:
        pickle.dump(output, f)

    return
