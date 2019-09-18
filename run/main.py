import os
import sys
import numpy as np
import importlib
import datetime
import pickle
import time

try:
    import pypolychord as polychord
    import pypolychord.settings as polysettings
except ModuleNotFoundError:
    pass

from .config import read_config

HOME = os.getenv('HOME')

def runpoly(configfile, nlive=None, modelargs={}, **kwargs):
    
    # Read dictionaries from configuration file
    rundict, initdict, datadict, priordict, fixeddict = read_config(
        configfile)
    parnames = list(priordict.keys())

    # Import model module
    modulename = 'model_{target}_{runid}'.format(**rundict)
    mod = importlib.import_module(modulename)

    # Instantaniate model class (pass additional arguments)
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
    ndim = len(initdict)

    # Fix starting time to identify chain.
    isodate = datetime.datetime.today().isoformat()
    
    # Define PolyChord settings
    settings = polysettings.PolyChordSettings(ndim, nderived, )
    settings.do_clustering = True
    if nlive is None:
        settings.nlive = 25*ndim
    else:
        settings.nlive = nlive
        
    fileroot = rundict['target']+'_'+rundict['runid']
    if rundict['comment'] != '':
        fileroot += '_'+rundict['comment']
        
    # add date
    fileroot += '_'+isodate
    
    settings.file_root = fileroot
    settings.read_resume = False
    settings.num_repeats = ndim * 5
    settings.feedback = 1
    settings.precision_criterion = 0.001
    # base directory
    base_dir = os.path.join(HOME, 'ExP', rundict['target'], 'polychains')
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

    if output.comment != '':
        output.comment = '_'+output.comment
    
    print(f'\nTotal run time was: {datetime.timedelta(seconds=int(output.runtime))}')
    print(f'Total wall time was: {datetime.timedelta(seconds=int(output.walltime))}')
    print(f'\nlog10(Z) = {output.logZ*0.43429} \n') # Log10 of the evidence

    dump2pickle_poly(output)    
    
    return output

def dump2pickle_poly(output, savedir=None):

    pickledict = {'target': output.target,
                  'runid': output.runid,
                  'comm': output.comment,
                  'nlive': output.nlive, 
                  'sampler': 'polychord',
                  'date': datetime.datetime.today().isoformat()}

    if savedir is None:
        pickledir = os.path.join(os.getenv('HOME'), 'ExP',
                                output.target, 'samplers')
    else:
        pickledir = savedir

    # Check if path exists; create if not
    if not os.path.isdir(pickledir):
        os.makedirs(pickledir)

    f = open(os.path.join(pickledir,
                          '{target}_{runid}{comm}_{nlive}live_'
                          '{sampler}_{date}.dat'.format(**pickledict)), 'wb')
    
    pickle.dump(output, f)
    f.close()
    return
    
