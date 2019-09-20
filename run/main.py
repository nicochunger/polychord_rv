import os
import sys
import numpy as np
import importlib
import datetime
import pickle
import time
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import pypolychord as polychord
import pypolychord.settings as polysettings

from .config import read_config

HOME = os.getenv('HOME')


def runpoly(configfile, nlive=None, nplanets=None, modelargs={}, **kwargs):

    # Read dictionaries from configuration file
    rundict, datadict, priordict, fixeddict = read_config(configfile, nplanets)
    parnames = list(priordict.keys())

    # Import model module
    models_path = os.path.join(os.getenv('HOME'), 'run/targets/{target}/models'.format(**rundict))
    modulename = 'model_{target}_{runid}'.format(**rundict)
    sys.path.insert(0, models_path)
    mod = importlib.import_module(modulename) # modulename, models_path)

    # Instantiate model class (pass additional arguments)
    mymodel = mod.Model(fixeddict, datadict, priordict, **modelargs)

    # Function to convert from hypercube to physical parameter space
    def prior(hypercube):
        """ Priors for each parameter. """
        theta = []
        for i, x in enumerate(hypercube):
            theta.append(priordict[parnames[i]].ppf(x))
        return theta

    def loglike(x):
        return (mymodel.lnlike(x), [])

    # Prepare run
    nderived = 0
    ndim = len(parnames)

    # Starting time to identify chain
    isodate = datetime.datetime.today().isoformat()

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

    # Add number of planets, live points, sampler and date
    fileroot += '_k{}'.format(mymodel.nplanets)
    fileroot += '_{}nlive'.format(settings.nlive)
    fileroot += '_polychord'
    fileroot += '_'+isodate

    settings.file_root = fileroot
    settings.read_resume = False
    settings.num_repeats = ndim * 5
    settings.feedback = 1
    settings.precision_criterion = 0.01
    # Base directory
    if rank == 0:
        base_dir = os.path.join(HOME, 'ExP', rundict['target'], fileroot, 'polychains')
        if rank == 0:
            if not os.path.isdir(base_dir):
                os.makedirs(base_dir)
        settings.base_dir = os.path.join(base_dir)

    # Initialise clocks
    ti = time.clock()
    tw = time.time()

    # Run PolyChord
    output = polychord.run_polychord(loglike, ndim, nderived, settings, prior)

    output.runtime = time.clock() - ti
    output.walltime = time.time() - tw

    output.target = rundict['target']
    output.runid = rundict['runid']
    output.comment = rundict.get('comment', '')
    output.nplanets = mymodel.nplanets
    output.nlive = settings.nlive
    output.isodate = isodate

    if output.comment != '':
        output.comment = '_'+output.comment

    print(
        f'\nTotal run time was: {datetime.timedelta(seconds=int(output.runtime))}')
    print(
        f'Total wall time was: {datetime.timedelta(seconds=int(output.walltime))}')
    print(f'\nlog10(Z) = {output.logZ*0.43429} \n')  # Log10 of the evidence

    if rank == 0:
        dump2pickle_poly(output)

    return output


def dump2pickle_poly(output, savedir=None):

    pickledict = {'target': output.target,
                  'runid': output.runid,
                  'comm': output.comment,
                  'nplanets': output.nplanets,
                  'nlive': output.nlive,
                  'sampler': 'polychord',
                  'date': output.isodate}

    if savedir is None:
        pickledir = os.path.join(os.getenv('HOME'), 'ExP',
                                 output.target, output.file_root)
    else:
        pickledir = savedir

    # Check if path exists; create if not
    if not os.path.isdir(pickledir):
        os.makedirs(pickledir)

    f = open(os.path.join(pickledir,
                          '{target}_{runid}{comm}_k{nplanets}_{nlive}live_'
                          '{sampler}_{date}.dat'.format(**pickledict)), 'wb')

    pickle.dump(output, f)
    f.close()
    return
