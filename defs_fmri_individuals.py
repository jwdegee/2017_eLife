#!/usr/bin/env python
# encoding: utf-8
"""
================================================
Created by Jan Willem de Gee on 2014-06-01.     
Copyright (c) 2009 jwdegee. All rights reserved.
================================================
"""

from IPython import embed as shell

# -------------------------------
# import general functionality: -
# -------------------------------

import os, sys, subprocess, datetime, time
import tempfile, logging, pickle
import numpy as np
import scipy as sp
from scipy.stats import *
from scipy.stats import norm
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import random
from random import *
import bottleneck as bn
from itertools import *

import glob
import copy
import nibabel as nib

# --------------------------------
# import specific functionality: -
# --------------------------------

import mne
import nitime
#from skimage import *
import sklearn
from nifti import *
# from pypsignifit import *
from nitime import fmri
from scipy import misc
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from scipy.stats.stats import pearsonr
import statsmodels.api as sm

# -------------------------------
# import custom functionality:  -
# -------------------------------

from Tools.Sessions import *
from Tools.Run import *
from Tools.Operators import *
from Tools.other_scripts import functions_jw as myfuncs
from Tools.other_scripts import functions_jw_GLM as myglm

class defs_fmri_individuals(Session):
    
    """
    Template Class for fMRI sessions analysis.
    """
    def __init__(self, ID, date, project, subject, parallelize = True, loggingLevel = logging.DEBUG):
        super(VisualSession, self).__init__(ID, date, project, subject, parallelize = parallelize, loggingLevel = loggingLevel)
        
        self.tr = 2.0
        self.fig_dir = '/home/shared/Niels_UvA/Visual_UvA/figures/'
        
    # HOUSEHOLD:
    # ----------
    def rename(self, conditions = [], postFix_old = [], postFix_new = []):
        
        for cond in conditions:
            for r in [self.runList[i] for i in self.conditionDict[cond]]:
                os.system('mv ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix_old) + ' ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix_new) )
    
    def remove(self, conditions = [], postFix = []):
        
        for cond in conditions:
            for r in [self.runList[i] for i in self.conditionDict[cond]]:
                os.system('rm ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix) )
                os.system('rm -r ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension='.feat') )
                os.system('rm ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension='.fsf') )
                os.system('rm -r ' + self.runFile(stage = 'processed/mri', run = r, postFix = postFix, extension='.mat') )
    
    def copy_files(self):
        
        copy_in = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['T2_anat'][0]])
        copy_out = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['TSE_anat'][0]])
        copy_out = copy_out.split('.')[0][:-1] + '6.nii.gz'
        os.system('cp ' + copy_in + ' ' + copy_out)
    
    def all_timings(self):
        """stimulus_timings uses behavior operators to distil:
        - the times at which stimulus presentation began and ended per task type
        - the times at which the task buttons were pressed. 
        """
        
        # shell()
        
        # create FSL ready text files:
        if not os.path.isdir(os.path.join(self.project.base_dir, self.subject.initials, 'files')):
            os.mkdir(os.path.join(self.project.base_dir, self.subject.initials, 'files'))
        
        session = [self.runList[i] for i in self.conditionDict['task']][0].session
        nr_runs = len([self.runList[i] for i in self.conditionDict['task']])
        
        nr_trs = []
        for j, r in enumerate([self.runList[i] for i in self.conditionDict['task']]):
            
            # fMRI:
            nr_trs.append(NiftiImage(self.runFile(stage = 'processed/mri', run = r)).timepoints)
            
            # load key press and trigger data:
            edf_file = glob.glob(self.runFolder(stage = 'processed/eye', run = r,) + '/*.edf')[0]
            eo = EDFOperator.EDFOperator(edf_file)
            eo.read_key_events()
            eo.read_trials()
            # compute timings of triggers:
            events = eo.events
            parameters = eo.parameters
            nr_trials = eo.nr_trials
            nr_trial_phases = eo.nr_trial_phases
            nr_phases = nr_trial_phases / nr_trials
            start_time = np.array([event['EL_timestamp'] for event in events if (event['up_down'] == 'Down') if (event['key'] == 116)])[0] # experiment script waited for this first trigger (in ms!).
            
            # start_time = eo.trial_phases[0]['trial_phase_EL_timestamp'] # / 1000.0
            # # start_time = eo.trial_starts[0][0] / 1000.0
            # end_time = eo.trial_ends[-1][0] / 1000.0
            
            triggers = np.zeros(nr_trs[j])
            triggers[:] = np.NaN
            triggers[0:np.array([event['EL_timestamp'] for event in events if (event['up_down'] == 'Down') if (event['key'] == 116)]).shape[0]] = (np.array([event['EL_timestamp'] for event in events if (event['up_down'] == 'Down') if (event['key'] == 116)]) - start_time) / 1000.0
            
            # drop in shell if we have missed a trigger:
            t = (np.array([event['EL_timestamp'] for event in events if (event['up_down'] == 'Down') if (event['key'] == 116)]) - start_time) / 1000.0
            if (np.array(np.diff(t),dtype=int)>2).sum() > 0:
                'error!'
                shell()
            
            buttons = (np.array([event['EL_timestamp'] for event in events if (event['up_down'] == 'Down') if (event['key'] == 98) or (event['key'] == 101)]) - start_time) / 1000.0
            phases = (np.array([eo.trial_phases[i]['trial_phase_EL_timestamp'] for i in range(nr_trial_phases)]).reshape(nr_trials,nr_phases) - start_time) / 1000.0
            
            rt = phases[:,2] - phases[:,1]
            
            # check for button within 2-s after phase forward:
            for i in range(nr_trials):
                for b in buttons:
                    # print b - phases[i,2]
                    if sum(((b - phases[i,2]) > 0) * ((b - phases[i,2]) < 2)) > 0:
                        rt[i] = b - phases[i,1]
            
            print 'min rt = {}'.format(min(rt))
            print 'max rt = {}'.format(max(rt))

            # trial types:
            omissions = (np.array([parameters[i]['signal_present'] for i in range(nr_trials)], dtype=int) == -1) + (rt<0.25)
            signal = (np.array([parameters[i]['signal_present'] for i in range(nr_trials)], dtype=int) == 1)
            correct = (np.array([parameters[i]['correct'] for i in range(nr_trials)], dtype=int) == 1)
            yes = ((signal*correct) + (-signal*-correct))*-omissions
            no = (-yes)*-omissions
            hit = (yes*correct)*-omissions
            fa = (yes*-correct)*-omissions
            miss = (no*-correct)*-omissions
            cr = (no*correct)*-omissions
                
            # print
            # print r.ID
            # print sum(omissions)

            # save as FSL type txt file:
            for i, c in enumerate([hit, fa, miss, cr]):
                txt_file = np.ones((nr_trials,3))
                # option 1:
                txt_file[:,0] = phases[:,1]
                txt_file[:,1] = rt

                # # option 2:
                # txt_file[:,0] = buttons

                # # option 3:
                # txt_file[:,2] = 1 / self.rt

                if i == 0:
                    condition = 'hit'
                if i == 1:
                    condition = 'fa'
                if i == 2:
                    condition = 'miss'
                if i == 3:
                    condition = 'cr'
                np.savetxt(self.runFile(stage = 'processed/behavior', run = r, extension = '.txt', postFix = ['all_trials']), txt_file, fmt = '%3.2f', delimiter = '\t')
                np.savetxt(self.runFile(stage = 'processed/behavior', run = r, extension = '.txt', postFix = [condition]), txt_file[c,:], fmt = '%3.2f', delimiter = '\t')
            
            # save as pd dataFrame:
            d = {
            'trial_nr' : pd.Series(np.arange(nr_trials)),
            'omissions' : pd.Series(np.array(omissions)),
            'hit' : pd.Series(np.array(hit)),
            'fa' : pd.Series(np.array(fa)),
            'miss' : pd.Series(np.array(miss)),
            'cr' : pd.Series(np.array(cr)),
            'rt' : pd.Series(np.array(rt)),
            'cue_onset' : pd.Series(np.array(phases[:,1])),
            }
            data = pd.DataFrame(d)
            data.to_csv(self.runFile(stage = 'processed/behavior', run = r, extension = '.csv', postFix = ['condition_times']))
            
            # blinks:
            eo.read_eyelink_events()
            blinks = eo.blinks_from_message_file
            blink_starts = (np.array([b['start_timestamp'] for b in blinks]) - start_time) / 1000.0
            blink_starts = blink_starts[blink_starts>0]
            blink_ends = (np.array([b['end_timestamp'] for b in blinks]) - start_time) / 1000.0
            blink_ends = blink_ends[blink_ends>blink_starts[0]]
            blink_durs = blink_ends - blink_starts
            blink_array = np.vstack((blink_starts, blink_durs, np.ones(blink_starts.shape[0]))).T
            np.savetxt(self.runFile(stage = 'processed/behavior', run = r, extension = '.txt', postFix = ['blinks']), blink_array, fmt = '%3.2f', delimiter = '\t')
            
            # saccades:
            eo.read_eyelink_events()
            sacs = eo.saccades_from_message_file
            sac_starts = (np.array([b['start_timestamp'] for b in sacs]) - start_time) / 1000.0
            sac_starts = sac_starts[sac_starts>0]
            sac_ends = (np.array([b['end_timestamp'] for b in sacs]) - start_time) / 1000.0
            sac_ends = sac_ends[sac_ends>sac_starts[0]]
            sac_durs = sac_ends - sac_starts
            sac_array = np.vstack((sac_starts, sac_durs, np.ones(sac_starts.shape[0]))).T
            np.savetxt(self.runFile(stage = 'processed/behavior', run = r, extension = '.txt', postFix = ['sacs']), sac_array, fmt = '%3.2f', delimiter = '\t')
            
            # triggers:
            np.save(self.runFile(stage = 'processed/behavior', run = r, extension = '.npy', postFix = ['triggers']), triggers)
        
        # concatenate over runs:
        # ----------------------

        # load behavioural data:
        condition_times = []
        all_trials = []
        hit = []
        fa = []
        miss = []
        cr = []
        blinks = []
        sacs = []
        triggers = []
        for j, r in enumerate([self.runList[i] for i in self.conditionDict['task']]):
            condition_times.append(pd.read_csv(self.runFile(stage = 'processed/behavior', run = r, extension = '.csv', postFix = ['condition_times'])))
            all_trials.append(np.loadtxt(self.runFile(stage = 'processed/behavior', run = r, postFix = ['all_trials'], extension = '.txt')))
            blinks.append(np.loadtxt(self.runFile(stage = 'processed/behavior', run = r, postFix = ['blinks'], extension = '.txt')))
            sacs.append(np.loadtxt(self.runFile(stage = 'processed/behavior', run = r, postFix = ['sacs'], extension = '.txt')))
            triggers.append(np.load(self.runFile(stage = 'processed/behavior', run = r, extension = '.npy', postFix = ['triggers'])))
        
        # concat behavioural data:
        time_to_add = 0
        for i, r in enumerate([self.runList[i] for i in self.conditionDict['task']]):
            all_trials[i][:,0] = all_trials[i][:,0] + time_to_add
            blinks[i][:,0] = blinks[i][:,0] + time_to_add
            sacs[i][:,0] = sacs[i][:,0] + time_to_add
            condition_times[i].cue_onset = condition_times[i].cue_onset + time_to_add
            triggers[i] = triggers[i] + time_to_add
            
            time_to_add += nr_trs[i] * self.tr
            
            print nr_trs[i] * self.tr
            
        all_trials = np.vstack(all_trials)
        condition_times = pd.concat(condition_times)
        
        condition_times.to_csv(os.path.join(self.stageFolder('processed/behavior/task'), 'condition_times.csv'))
        np.savetxt(os.path.join(self.stageFolder('processed/behavior/task'), 'all_trials.txt'), all_trials, fmt = '%3.2f', delimiter = '\t')
        np.savetxt(os.path.join(self.stageFolder('processed/behavior/task'), 'hit.txt'), all_trials[np.array(condition_times.hit),:], fmt = '%3.2f', delimiter = '\t')
        np.savetxt(os.path.join(self.stageFolder('processed/behavior/task'), 'fa.txt'), all_trials[np.array(condition_times.fa),:], fmt = '%3.2f', delimiter = '\t')
        np.savetxt(os.path.join(self.stageFolder('processed/behavior/task'), 'miss.txt'), all_trials[np.array(condition_times.miss),:], fmt = '%3.2f', delimiter = '\t')
        np.savetxt(os.path.join(self.stageFolder('processed/behavior/task'), 'cr.txt'), all_trials[np.array(condition_times.cr),:], fmt = '%3.2f', delimiter = '\t')
        np.savetxt(os.path.join(self.stageFolder('processed/behavior/task'), 'blinks.txt'), np.vstack(blinks), fmt = '%3.2f', delimiter = '\t')
        np.savetxt(os.path.join(self.stageFolder('processed/behavior/task'), 'sacs.txt'), np.vstack(sacs), fmt = '%3.2f', delimiter = '\t')
        
        np.savetxt(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'blinks_{}.txt'.format(session)), np.vstack(blinks), fmt = '%3.2f', delimiter = '\t')
        np.savetxt(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'sacs_{}.txt'.format(session)), np.vstack(sacs), fmt = '%3.2f', delimiter = '\t')
        np.savetxt(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'triggers_{}.txt'.format(session)), np.concatenate(triggers), fmt = '%3.2f', delimiter = '\t')
        
        # load pupil data:
        condition_times = pd.read_csv(os.path.join(self.stageFolder('processed/behavior/task'), 'condition_times.csv'))
        pupil_data = pd.read_csv(os.path.join(self.project.base_dir, self.subject.initials, 'pupil_data.csv'))
        nr_runs_total = (pupil_data.trial_nr == 0).sum()
        pupil_data = pupil_data[(pupil_data.session == session)]
        nr_runs_pupil = np.unique(pupil_data.run).shape[0]
        nr_runs_pupil_BOLD = len([self.runList[i] for i in self.conditionDict['task']])
        if nr_runs_pupil != nr_runs_pupil_BOLD:
            shell()
        
        # test:
        try:
            mean_rt_diff = np.mean(abs(np.array(pupil_data.rt) - np.array(condition_times.rt)))
            print 'mean rt diff = {}'.format(mean_rt_diff)
            if mean_rt_diff > 0.1:
                shell()
        except:
            pass
            shell()
        
        # add cue times to pupil_data:
        pupil_data['cue_onset'] = np.array(condition_times.cue_onset)
        pupil_data.to_csv(os.path.join(self.project.base_dir, self.subject.initials, 'pupil_data_{}.csv'.format(session)))
        
        # LOCALIZER:
        # ----------
        
        nr_trs = []
        for j, r in enumerate([self.runList[i] for i in self.conditionDict['loc']]):
            
            # fMRI:
            nr_trs.append(NiftiImage(self.runFile(stage = 'processed/mri', run = r)).timepoints)
            
            # load key press and trigger data:
            edf_file = glob.glob(self.runFolder(stage = 'processed/eye', run = r,) + '/*.edf')[0]
            eo = EDFOperator.EDFOperator(edf_file)
            eo.read_key_events()
            
            # compute timings of triggers:
            events = eo.events
            parameters = eo.parameters
            nr_trials = eo.nr_trials
            nr_trial_phases = eo.nr_trial_phases
            nr_phases = nr_trial_phases / nr_trials
            start_time = np.array([event['EL_timestamp'] for event in events if (event['up_down'] == 'Down') if (event['key'] == 116)])[0] # experiment script waited for this first trigger (in ms!).
            triggers = (np.array([event['EL_timestamp'] for event in events if (event['up_down'] == 'Down') if (event['key'] == 116)]) - start_time) / 1000.0
            buttons = (np.array([event['EL_timestamp'] for event in events if (event['up_down'] == 'Down') if (event['key'] == 98) or (event['key'] == 101)]) - start_time) / 1000.0
            phases = (np.array([eo.trial_phases[i]['trial_phase_EL_timestamp'] for i in range(nr_trial_phases)]).reshape(nr_trials,nr_phases) - start_time) / 1000.0
            
            txt_file = np.ones((nr_trials,3))
            txt_file[:,0] = phases[:,2]
            txt_file[:,1] = 11.5
            np.savetxt(self.runFile(stage = 'processed/behavior', run = r, extension = '.txt', postFix = ['center']), txt_file, fmt = '%3.2f', delimiter = '\t')
            
            np.savetxt(self.runFile(stage = 'processed/behavior', run = r, extension = '.txt', postFix = ['center_cw']), txt_file[::2], fmt = '%3.2f', delimiter = '\t')
            np.savetxt(self.runFile(stage = 'processed/behavior', run = r, extension = '.txt', postFix = ['center_ccw']), txt_file[1::2], fmt = '%3.2f', delimiter = '\t')
            
            txt_file = np.ones((buttons.shape[0],3))
            txt_file[:,0] = buttons
            np.savetxt(self.runFile(stage = 'processed/behavior', run = r, extension = '.txt', postFix = ['button']), txt_file, fmt = '%3.2f', delimiter = '\t')    

    def create_brain_masks(self):
        
        inputObject = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['T2_anat'][0]])
        outputObject = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['T2_anat'][0]], postFix=['mask'])
        fmO = FSLMathsOperator(inputObject=inputObject)
        fmO.configure(outputFileName=outputObject, **{'-bin': ""})
        fmO.execute()
        
        subprocess.Popen('cp ' + outputObject + ' ' + self.runFile(stage = 'processed/mri/masks/anat', base='brain_mask'), shell=True, stdout=PIPE).communicate()[0]
        
        inputObject = self.runFile(stage = 'processed/mri/masks/anat', base='brain_mask')
        outputObject = self.runFile(stage = 'processed/mri/masks/anat', base='brain_mask')
        fmO = FSLMathsOperator(inputObject=inputObject)
        fmO.configure(outputFileName=outputObject, **{'-subsamp2offc': ''})
        fmO.execute()
    
    def register_TSE_epi(self):
        
        # # Transfrom brainstem mask to high res T2:
        # # ------------------------------------------
        #
        # # transform mask:
        # inputObject = '/home/shared/UvA/Niels_UvA/mni_masks/brainstem/brain_stem_25.nii.gz'
        # referenceFileName = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['T2_anat'][0]])
        # transformMatrixFileName = self.runFile(stage = 'processed/mri/reg', base = 'register_{}{}_MNI_inv'.format(self.project.projectName, self.subject.initials), extension='.mat' )
        # outputFileName = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['T2_anat'][0]], postFix=['brainstem_mask'])
        # fO = FlirtOperator(inputObject=inputObject, referenceFileName=referenceFileName)
        # fO.configureApply(transformMatrixFileName=transformMatrixFileName, outputFileName=outputFileName, sinc=False)
        # fO.execute()
        #
        # # manipulate mask:
        # brain_mask = nib.load(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['T2_anat'][0]], postFix=['mask']))
        # brain_mask_data = np.array(brain_mask.get_data(), dtype=bool)
        # brainstem_mask = nib.load(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['T2_anat'][0]], postFix=['brainstem_mask']))
        # brainstem_mask_data = brainstem_mask.get_data() # 0 - 100
        # brainstem_mask_data[brainstem_mask_data==0] = 5 # make at least 5 everywhere
        # brainstem_mask_data[~brain_mask_data] = 0 # make 0 everywhere outside brain
        # brainstem_mask.data = brainstem_mask_data # brain is now 5, brainstem is now 100
        # nib.save(brainstem_mask, self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['T2_anat'][0]], postFix=['brainstem_mask2']))
        
        # T2 to TSE whole brain:
        # ----------------------
        
        inputObject = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['T2_anat'][0]],)
        referenceFileName = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['TSE_anat_whole'][0]],)
        # inweight = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['T2_anat'][0]], postFix=['brainstem_mask2'])
        fO = FlirtOperator(inputObject, referenceFileName)
        fO.configureRun()
        fO.execute()
        
        # invert:
        invFl = InvertFlirtOperator(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['T2_anat'][0]], postFix=['trans'], extension='.mat'))
        invFl.configure()
        invFl.execute()
        
        # # transform mask:
        # inputObject = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['T2_anat'][0]], postFix=['brainstem_mask2'])
        # referenceFileName = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['TSE_anat_whole'][0]])
        # transformMatrixFileName = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['T2_anat'][0]], postFix=['trans'], extension='.mat')
        # outputFileName = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['TSE_anat_whole'][0]], postFix=['brainstem_mask2'])
        # fO = FlirtOperator(inputObject=inputObject, referenceFileName=referenceFileName)
        # fO.configureApply(transformMatrixFileName=transformMatrixFileName, outputFileName=outputFileName, sinc=False)
        # fO.execute()
        
        # TSE whole brain to TSE:
        # -----------------------
        
        # step 1
        inputObject = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['TSE_anat_whole'][0]])
        referenceFileName = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['TSE_anat'][0]])
        transformMatrixFileName = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['TSE_anat_whole'][0]], postFix=['trans1'], extension='.mat')
        # inweight = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['TSE_anat_whole'][0]], postFix=['brainstem_mask2'])
        fO = FlirtOperator(inputObject=inputObject, referenceFileName=referenceFileName)
        # fO.configureRun(transformMatrixFileName=transformMatrixFileName, extra_args='-schedule ${}/etc/flirtsch/xyztrans.sch -inweight {}'.format('FSLDIR', inweight))
        fO.configureRun(transformMatrixFileName=transformMatrixFileName, extra_args='-schedule ${}/etc/flirtsch/xyztrans.sch'.format('FSLDIR'))
        fO.execute()
        
        # step 2
        inputObject = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['TSE_anat_whole'][0]],)
        referenceFileName = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['TSE_anat'][0]],)
        transformMatrixFileName = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['TSE_anat_whole'][0]], postFix=['trans2'], extension='.mat')
        init = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['TSE_anat_whole'][0]], postFix=['trans1'], extension='.mat')
        # inweight = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['TSE_anat_whole'][0]], postFix=['brainstem_mask2'])
        fO = FlirtOperator(inputObject=inputObject, referenceFileName=referenceFileName)
        # fO.configureRun(transformMatrixFileName=transformMatrixFileName, extra_args='-init {} -dof {} -inweight {} -nosearch'.format(init, 6, inweight))
        fO.configureRun(transformMatrixFileName=transformMatrixFileName, extra_args='-init {} -dof {} -nosearch'.format(init, 6))
        fO.execute()
        
        # invert:
        invFl = InvertFlirtOperator(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['TSE_anat_whole'][0]], postFix=['trans2'], extension='.mat'))
        invFl.configure()
        invFl.execute()
        
        # TSE to T2:
        # ----------
        
        # concatenate transformation matrices:
        inputObject = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['TSE_anat_whole'][0]], postFix=['trans2', 'inv'], extension='.mat')
        secondInputFile = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['T2_anat'][0]], postFix=['trans', 'inv'], extension='.mat')
        outputFileName = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['TSE_anat'][0]], postFix=['TSE_to_T2'], extension='.mat')
        cf = ConcatFlirtOperator(inputObject=inputObject)
        cf.configure(secondInputFile, outputFileName)
        cf.execute()
        
        # TSE to T2 (for visual inspection):
        inputObject = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['TSE_anat'][0]])
        referenceFileName = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['T2_anat'][0]])
        transformMatrixFileName = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['TSE_anat'][0]], postFix=['TSE_to_T2'], extension='.mat')
        fO = FlirtOperator(inputObject, referenceFileName)
        fO.configureApply(transformMatrixFileName=transformMatrixFileName)
        fO.execute()
        
        # Save transformation matrices in reg folder:
        # -------------------------------------------
        
        # copy:
        copy_in = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['TSE_anat'][0]], postFix=['TSE_to_T2'], extension='.mat')
        copy_out = self.runFile(stage = 'processed/mri/reg', base = 'TSE_to_T2', extension='.mat')
        subprocess.Popen('cp ' + copy_in + ' ' + copy_out, shell=True, stdout=PIPE).communicate()[0]
        
        # invert:
        invfO = InvertFlirtOperator(self.runFile(stage = 'processed/mri/reg', base = 'TSE_to_T2', extension='.mat'))
        invfO.configure()
        invfO.execute()
        
    def grab_LC_masks(self):
        
        for r in [self.runList[i] for i in self.conditionDict['TSE_anat']]:
            
            mask_in = os.path.join(self.project.base_dir, 'LCs', self.runFile('processed/mri', run=r, postFix=['LC', 'JW']).split('.')[-3].split('/')[-1] + '.nii.gz')
            mask_out = self.runFile('processed/mri', run=r, postFix=['LC', 'JW']) 
            os.system('cp ' + mask_in + ' ' + mask_out)
            
            mask_in = os.path.join(self.project.base_dir, 'LCs', self.runFile('processed/mri', run=r, postFix=['ventricle']).split('.')[-3].split('/')[-1] + '.nii.gz')
            mask_out = self.runFile('processed/mri', run=r, postFix=['ventricle']) 
            os.system('cp ' + mask_in + ' ' + mask_out)                
    
    
    def transform_standard_brainmasks(self, min_nr_voxels=12):
        
        for m in [
                'LC_standard_2',
                'LC_standard_1',
                'mean_fullMB',
                'mean_SN',
                'mean_VTA',
                'basal_forebrain_4',
                'basal_forebrain_123',
                'inferior_colliculus',
                'superior_colliculus',
                'AAN_VTA',
                'AAN_PAG',
                'AAN_PBC',
                'AAN_PO',
                'AAN_PPN',
                'AAN_LC',
                'AAN_MR',
                'AAN_MRF',
                'AAN_DR',
                'inf_col_jw',
                'sup_col_jw',
                ]:
            
            # transform:
            inputObject = '/home/shared/Niels_UvA/brainstem_masks/{}.nii.gz'.format(m)
            referenceFileName = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['task'][0]])
            transformMatrixFileName = self.runFile(stage = 'processed/mri/reg', base = 'register_{}{}_MNI_inv'.format(self.project.projectName, self.subject.initials), extension='.mat' )
            outputFileName = self.runFile(stage = 'processed/mri/masks/anat', base=m) 
            fO = FlirtOperator(inputObject, referenceFileName)
            fO.configureApply(transformMatrixFileName=transformMatrixFileName, outputFileName=outputFileName, sinc=False)
            fO.execute()
            
            # treshold:
            mask = NiftiImage(outputFileName)
            mask_data = mask.data
            threshold = mask_data.ravel()[np.argsort(mask_data.ravel())[-min_nr_voxels]]
            mask_data[mask_data < threshold] = 0
            # mask_data[mask_data >= threshold] = 1
            mask.data = mask_data
            mask.save(outputFileName)
    
    def transform_choice_areas(self,):
        
        for m in [
                'lr_aIPS', 
                'lr_PCeS', 
                'lr_M1',
                'sl_IPL',
                'sl_SPL1',
                'sl_SPL2',
                'sl_pIns',
                ]:
            
            for hm in ['lh', 'rh']:
            
                # transform:
                inputObject = '/home/shared/UvA/Niels_UvA/surface_masks/{}.{}.nii.gz'.format(hm,m)
                referenceFileName = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['task'][0]])
                transformMatrixFileName = self.runFile(stage = 'processed/mri/reg', base = 'register_{}{}_MNI_inv'.format(self.project.projectName, self.subject.initials), extension='.mat' )
                outputFileName = self.runFile(stage = 'processed/mri/masks/anat', base='{}.{}'.format(hm, m)) 
                fO = FlirtOperator(inputObject, referenceFileName)
                fO.configureApply(transformMatrixFileName=transformMatrixFileName, outputFileName=outputFileName, sinc=False)
                fO.execute()
                
                # shell()
                
                # # treshold:
                # mask = NiftiImage(outputFileName)
                # mask_data = mask.data
                # threshold = mask_data.ravel()[np.argsort(mask_data.ravel())[-min_nr_voxels]]
                # mask_data[mask_data < threshold] = 0
                # # mask_data[mask_data >= threshold] = 1
                # mask.data = mask_data
                # mask.save(outputFileName)
                
    def concatenate_data_runs(self, conditions=['task'], postFix=['B0', 'mcf', 'sgtf', '0.01', 'psc']):
        
        # load:
        niftis = []
        nr_trs = []
        for cond in conditions:
            for r in [self.runList[i] for i in self.conditionDict[cond]]:
                niftis.append(self.runFile(stage = 'processed/mri', run = r, postFix = postFix))
                nr_trs.append(NiftiImage(self.runFile(stage = 'processed/mri', run = r, postFix = postFix)).timepoints)
        session = r.session

        # concat fMRI data:
        # -----------------
        mo = FSLMergeOperator(inputObject = niftis)
        mo.configure(outputFileName = os.path.join(self.project.base_dir, self.subject.initials, 'files', 'detection_data_psc_{}_{}'.format(self.subject.initials, r.session)), TR=self.tr)
        mo.execute()

        # concat retroicor:
        # -----------------
        for ri in xrange(34):
            retroicor_regs = []
            for cond in conditions:
                for r in [self.runList[i] for i in self.conditionDict[cond]]:
                    retroicor_regs.append([reg for reg in np.sort(glob.glob(os.path.join(self.runFolder(stage = 'processed/mri', run = r), 'retroicor', 'retroicor'+'ev*.nii*')))][ri])
            mo = FSLMergeOperator(inputObject = retroicor_regs)
            if ri <= 9:
                mo.configure(outputFileName = os.path.join(self.project.base_dir, self.subject.initials, 'files', 'retroicor_data_{}_{}_0{}'.format(self.subject.initials, session, ri)), TR=self.tr)
            else:
                mo.configure(outputFileName = os.path.join(self.project.base_dir, self.subject.initials, 'files', 'retroicor_data_{}_{}_{}'.format(self.subject.initials, session, ri)), TR=self.tr)
            mo.execute()
        regressors = [reg for reg in np.sort(glob.glob(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'retroicor_data_{}_{}*.nii.gz'.format(self.subject.initials, session))))]
        text_file = open(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'retroicor_data_{}_{}_evs_list.txt'.format(self.subject.initials, session)), 'w')
        for reg in regressors:
            text_file.write('{}\n'.format(reg))
        text_file.close()

        # concat ones:
        total_trs = np.sum(nr_trs)
        loc = 0
        for j, r in enumerate([self.runList[i] for i in self.conditionDict['task']]):
            ones = np.zeros(total_trs)
            ones[loc:loc+nr_trs[j]] = np.ones(nr_trs[j])
            loc += nr_trs[j]
            np.savetxt(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'ones_{}_{}.txt'.format(r.session, j)), ones, fmt = '%3.2f', delimiter = '\t')

        # copy blinks:
        subprocess.Popen('cp ' + os.path.join(self.stageFolder('processed/behavior/task'), 'blinks.txt') + ' ' + os.path.join(self.project.base_dir, self.subject.initials, 'files', 'blinks_{}.txt'.format(r.session)), shell=True, stdout=PIPE).communicate()[0]
    
    def GLM_nuisance(self):
        
        session = [self.runList[i] for i in self.conditionDict['task']][0].session
        nr_runs = len([self.runList[i] for i in self.conditionDict['task']])
        nii_file = nib.load(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'detection_data_psc_{}_{}.nii.gz'.format(self.subject.initials, session)))
        nii_file_data = nii_file.get_data()
        blinks = np.loadtxt(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'blinks_{}.txt'.format(session)))
        sacs = np.loadtxt(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'sacs_{}.txt'.format(session)))
        with open(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'retroicor_data_{}_{}_evs_list.txt'.format(self.subject.initials, session)), 'r') as f:
            retroicor_files = [x.strip('\n') for x in f.readlines()]
        retroicor_evs = []
        for r in retroicor_files:
            retroicor_evs.append(nib.load(r))
        brain_mask = np.array(nib.load(self.runFile(stage = 'processed/mri/masks/anat', base='brain_mask')).get_data(), dtype=bool)
        brain_mask[:,:,:] = True
        
        residuals = np.zeros(nii_file_data.shape)
        nr_slices = nii_file.shape[2]
        
        for s in range(nr_slices):
            
            slice_indices = (np.arange(nr_slices) == s)
            mask = brain_mask[:,:,slice_indices][:,:,0]
            
            events = [blinks, sacs]
            retroicor_slice_wise_evs = []
            for r in retroicor_evs:
                retroicor_slice_wise_evs.append(r.get_data()[0,0,s,:])
            retroicor_slice_wise_evs = np.vstack(retroicor_slice_wise_evs)
                
            run_design = Design(nii_file.shape[3], self.tr, subSamplingRatio = 1)
            run_design.configure(events, hrfType = 'doubleGamma', hrfParameters = {'a1' : 6, 'a2' : 12, 'b1' : 0.9, 'b2' : 0.9, 'c' : 0.35})
            
            joined_design_matrix = np.mat(np.vstack([run_design.designMatrix, retroicor_slice_wise_evs]).T)
            data = nii_file_data[mask,s,:].T
            
            # f = pl.figure(figsize = (10, 10))
            # s = f.add_subplot(111)
            # pl.imshow(joined_design_matrix)
            # s.set_title('nuisance design matrix')
            # pl.savefig(self.runFile(stage = 'processed/mri', run = r, postFix = postFix, base = 'nuisance_design', extension = '.pdf' ))
            
            betas = ((joined_design_matrix.T * joined_design_matrix).I * joined_design_matrix.T) * np.mat(data.T).T
            explained_signal = np.array((np.mat(joined_design_matrix) * np.mat(betas)))
            res = data - explained_signal
            
            # plt.plot(data[:,900], color='b', alpha=0.75)
            # plt.plot(res[:,900], color='r', alpha=0.75)
            # plt.plot(explained_signal[:,900]*-1, color='k', alpha=0.25)
            
            residuals[mask,s,:] = res.T
        
        res_nii_file = nib.Nifti1Image(residuals, affine=nii_file.get_affine(), header=nii_file.get_header())
        res_nii_file.set_data_dtype(np.float32)
        nib.save(res_nii_file, os.path.join(self.project.base_dir, self.subject.initials, 'files', 'detection_NEW_data_clean_{}_{}.nii.gz'.format(self.subject.initials, session)))
        
    def clean_to_MNI(self,):
        
        session = [self.runList[i] for i in self.conditionDict['task']][0].session
        
        # transform:
        inputObject = os.path.join(self.project.base_dir, self.subject.initials, 'files', 'detection_NEW_data_clean_{}_{}.nii.gz'.format(self.subject.initials, session))
        referenceFileName = self.runFile(stage = 'processed/mri/reg/feat', base = 'standard')
        transformMatrixFileName = self.runFile(stage = 'processed/mri/reg/feat', base = 'example_func2standard', extension='.mat')
        outputFileName = os.path.join(self.project.base_dir, self.subject.initials, 'files', 'detection_data_clean_MNI_{}_{}.nii.gz'.format(self.subject.initials, session))
        fO = FlirtOperator(inputObject, referenceFileName)
        fO.configureApply(transformMatrixFileName=transformMatrixFileName, outputFileName=outputFileName, sinc=False)
        fO.execute()

        # # transform brain mask:
        # inputObject = os.path.join(self.project.base_dir, self.subject.initials, 'files', 'masks', 'brain_mask_{}.nii.gz'.format(session))
        # referenceFileName = self.runFile(stage = 'processed/mri/reg/feat', base = 'standard')
        # transformMatrixFileName = self.runFile(stage = 'processed/mri/reg/feat', base = 'example_func2standard', extension='.mat')
        # outputFileName = os.path.join(self.project.base_dir, self.subject.initials, 'files', 'masks', 'brain_mask_MNI_{}.nii.gz'.format(session))
        # fO = FlirtOperator(inputObject, referenceFileName)
        # fO.configureApply(transformMatrixFileName=transformMatrixFileName, outputFileName=outputFileName, sinc=False)
        # fO.execute()
        # cmdline = 'fslmaths {} -thr 0.5 -bin {}'.format(outputFileName, outputFileName)
        # subprocess.call( cmdline, shell=True, bufsize=0,)

    def combine_rois_across_hemispheres(self, rois):
        
        for roi in rois:
            if 'V2' in roi or 'V3' in roi:
                roi_data = NiftiImage(self.runFile(stage = 'processed/mri/masks/anat', base='lh.' + roi[:2]+'d'+roi[2:]))
                roi_data.data = NiftiImage(self.runFile(stage = 'processed/mri/masks/anat', base='lh.' + roi[:2]+'d'+roi[2:])).data + NiftiImage(self.runFile(stage = 'processed/mri/masks/anat', base='lh.' + roi[:2]+'v'+roi[2:])).data + NiftiImage(self.runFile(stage = 'processed/mri/masks/anat', base='rh.' + roi[:2]+'d'+roi[2:])).data + NiftiImage(self.runFile(stage = 'processed/mri/masks/anat', base='rh.' + roi[:2]+'v'+roi[2:])).data
                roi_data.save(self.runFile(stage = 'processed/mri/masks/anat', base=roi), roi_data)
            else:
                roi_data = NiftiImage(self.runFile(stage = 'processed/mri/masks/anat', base='lh.' + roi))
                roi_data.data = NiftiImage(self.runFile(stage = 'processed/mri/masks/anat', base='lh.' + roi)).data + NiftiImage(self.runFile(stage = 'processed/mri/masks/anat', base='rh.' + roi)).data
                roi_data.save(self.runFile(stage = 'processed/mri/masks/anat', base=roi), roi_data)
    
    def transfrom_destrieux_to_MNI(self, rois, transform=False):
        
        # transform labels to standard space:
        if transform == True:
            reg_file = os.path.join(os.environ["FREESURFER_HOME"], "average/mni152.register.dat")
            template_file = '/home/shared/Niels_UvA/brainstem_masks/MNI_standard_2x2x2.nii.gz'
            for roi in rois:
                hemi = roi.split('.')[0]
                outputFileName = '/home/shared/Niels_UvA/brainstem_masks/destrieux/' + '{}.nii.gz'.format(roi)
                lvo = LabelToVolOperator('$SUBJECTS_DIR/fsaverage/label/aparc.a2009s/{}'.format(roi))
                lvo.configure(templateFileName = template_file, hemispheres = [hemi], register = reg_file, fsSubject = 'fsaverage', outputFileName = outputFileName, threshold = 0.05, surfType = 'label')
                lvo.execute()
        
        # compute center of mass:
        x = np.zeros(len(rois))
        y = np.zeros(len(rois))
        z = np.zeros(len(rois))
        for i, roi in enumerate(rois):
            center_of_mass = sp.ndimage.measurements.center_of_mass(NiftiImage('/home/shared/Niels_UvA/brainstem_masks/destrieux/' + '{}.nii.gz'.format(roi)).data)
            x[i] = center_of_mass[2]
            y[i] = center_of_mass[1]
            z[i] = center_of_mass[0]
        
        # make csv_file:
        d = {
        'roi' : pd.Series(rois),
        'x' : pd.Series(x),
        'y' : pd.Series(y),
        'z' : pd.Series(z),
        }
        data = pd.DataFrame(d)
        data.to_csv('/home/shared/Niels_UvA/brainstem_masks/destrieux/rois.csv')    

    
    def create_session_rois(self, rois):
        
        try:
            os.system('mkdir {}'.format(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'masks')))
        except:
            pass
        
        session = [self.runList[i] for i in self.conditionDict['task']][0].session
        for roi in rois:
            os.system('cp {} {}'.format(self.runFile(stage = 'processed/mri/masks/anat', base=roi), os.path.join(self.project.base_dir, self.subject.initials, 'files', 'masks', roi + '_{}.nii.gz'.format(session))))
        
        # control LC masks:
        # -----------------
        
        threshold = 0.1
        mask_a = np.array(NiftiImage(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'masks', 'LC_JW_{}.nii.gz'.format(session))).data, dtype=float)
        mask_a[mask_a<threshold] = 0
        mask_a[mask_a>=threshold] = 1
        mask_a = np.array(mask_a, dtype=bool)
        
        mask_b = np.array(NiftiImage(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'masks', 'LC_standard_1_{}.nii.gz'.format(session))).data, dtype=float)
        mask_b[mask_b<threshold] = 0
        mask_b[mask_b>=threshold] = 1
        mask_b = np.array(mask_b, dtype=bool)
        
        mask_c = np.array(NiftiImage(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'masks', 'LC_standard_2_{}.nii.gz'.format(session))).data, dtype=float)
        mask_c[mask_c<threshold] = 0
        mask_c[mask_c>=threshold] = 1
        mask_c = np.array(mask_c, dtype=bool)
        
        mask_d = np.array(NiftiImage(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'masks', '4th_ventricle_{}.nii.gz'.format(session))).data, dtype=float)
        mask_d[mask_d<threshold] = 0
        mask_d[mask_d>=threshold] = 1
        mask_d = np.array(mask_d, dtype=bool)
        
        nifti = NiftiImage(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'masks', 'LC_standard_2_{}.nii.gz'.format(session)))
        
        # create LC_standard_1 minus LC:
        new_mask_1 = mask_b - (mask_a*mask_b)
        newImage = NiftiImage(np.array(new_mask_1, dtype=int))
        newImage.header = nifti.header
        newImage.filename = os.path.join(self.project.base_dir, self.subject.initials, 'files', 'masks', 'LC_standard_1-LC_JW_{}.nii.gz'.format(session))
        newImage.save()
        
        # create inflated LC (one ring around LC_standard_2), subtract 4th ventricle:
        dilate = 1
        new_mask_2 = sp.ndimage.binary_dilation(mask_a+mask_c, iterations=dilate)
        new_mask_2 = new_mask_2 * -(mask_a+mask_c+mask_d)
        newImage = NiftiImage(np.array(new_mask_2, dtype=int))
        newImage.header = nifti.header
        newImage.filename = os.path.join(self.project.base_dir, self.subject.initials, 'files', 'masks', 'LC_standard_2_dilated_{}.nii.gz'.format(session))
        newImage.save()
        
        # create inflated LC (one ring around LC_standard_2 previous), subtract 4th ventricle:
        dilate = 2
        new_mask_3 = sp.ndimage.binary_dilation(mask_a+mask_c, iterations=dilate)
        new_mask_3 = new_mask_3 * -(new_mask_2+mask_a+mask_c+mask_d)
        newImage = NiftiImage(np.array(new_mask_3, dtype=int))
        newImage.header = nifti.header
        newImage.filename = os.path.join(self.project.base_dir, self.subject.initials, 'files', 'masks', 'LC_standard_2_dilated_2_{}.nii.gz'.format(session))
        newImage.save()

    def GLM_localizer(self, conditions=['loc']):
        """
        Take all transition events and use them as event regressors
        Run FSL on this
        """
        for condition in conditions:
            for i, run in enumerate(self.conditionDict[condition]):
                
                # remove previous feat directories
                try:
                    # self.logger.debug('rm -rf ' + self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf', 'sgtf'], extension = '.feat'))
                    os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['B0', 'mcf'], extension = '.feat'))
                    os.system('rm -rf ' + self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['B0', 'mcf'], extension = '.fsf'))
                except OSError:
                    pass
                
                # this is where we start up fsl feat analysis after creating the feat .fsf file and the like
                if condition == 'loc':
                    thisFeatFile = '/home/shared/Niels_UvA/Visual_UvA/analysis/feat_localizers/design_loc.fsf'
                    
                    REDict = {
                    '---TR---':str(NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['B0', 'mcf'])).rtime),
                    '---NR_TRS---':str(NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['B0', 'mcf'])).timepoints),
                    '---NR_VOXELS---':str(np.prod(np.array(NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['B0', 'mcf'])).getExtent()))),
                    '---FUNC_FILE---':self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['B0', 'mcf']), 
                    '---CONFOUND_LIST---':os.path.join(self.runFolder(stage = 'processed/mri', run = self.runList[run]), 'retroicor', 'retroicor_evs_list.txt'),
                    '---EV1---':self.runFile(stage = 'processed/behavior', run = self.runList[run], postFix = ['center_cw'], extension = '.txt'), 
                    '---EV2---':self.runFile(stage = 'processed/behavior', run = self.runList[run], postFix = ['center_ccw'], extension = '.txt'), 
                    '---EV3---':self.runFile(stage = 'processed/behavior', run = self.runList[run], postFix = ['button'], extension = '.txt'), 
                    }
                    
                # if condition == 'loc_t':
                #     thisFeatFile = '/home/shared/Niels_UvA/Retmap_UvA/analysis/feat_localizers/design_t.fsf'
                #
                #     REDict = {
                #     '---TR---':str(NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf'])).rtime),
                #     '---NR_TRS---':str(NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf'])).timepoints),
                #     '---NR_VOXELS---':str(np.prod(np.array(NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf'])).getExtent()))),
                #     '---FUNC_FILE---':self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['mcf']),
                #     '---CONFOUND_LIST---':os.path.join(self.runFolder(stage = 'processed/mri', run = self.runList[run]), 'retroicor', 'retroicor_evs_list.txt'),
                #     '---EV1---':self.runFile(stage = 'processed/behavior', run = self.runList[run], postFix = ['left'], extension = '.txt'),
                #     '---EV2---':self.runFile(stage = 'processed/behavior', run = self.runList[run], postFix = ['right'], extension = '.txt'),
                #     '---EV3---':self.runFile(stage = 'processed/behavior', run = self.runList[run], postFix = ['button'], extension = '.txt'),
                #     }
                    
                featFileName = self.runFile(stage = 'processed/mri', run = self.runList[run], extension = '.fsf')
                featOp = FEATOperator(inputObject = thisFeatFile)
                
                # In serial:
                # featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = True )
                # In parallel:
                featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = False )
                
                # run feat
                featOp.execute()    
    
    def load_concatenated_data(self, data_type, bold=True):
        
        session = [self.runList[i] for i in self.conditionDict['task']][0].session
        nr_runs = len([self.runList[i] for i in self.conditionDict['task']])
        
        if bold:
            # BOLD data:
            if data_type == 'clean' or data_type == 'clean_ventricle' or data_type == 'clean_midbrain' or data_type == 'clean_LC' or data_type == 'clean_z':
                BOLD_data_1 = nib.load(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'detection_NEW_data_clean_{}_1.nii.gz'.format(self.subject.initials)))
                BOLD_data_2 = nib.load(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'detection_NEW_data_clean_{}_2.nii.gz'.format(self.subject.initials)))
            if (data_type == 'clean_MNI') or (data_type == 'clean_MNI_smooth'):
                BOLD_data_1 = nib.load(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'detection_data_{}_{}_1.nii.gz'.format(data_type, self.subject.initials)))
                BOLD_data_2 = nib.load(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'detection_data_{}_{}_2.nii.gz'.format(data_type, self.subject.initials)))
            if data_type == 'fsl_clean':
                BOLD_data_1 = nib.load(os.path.join(self.project.base_dir, 'across', 'glm', 'detection_data_clean_{}_1.feat'.format(self.subject.initials, session), 'stats', 'res4d.nii'))
                BOLD_data_2 = nib.load(os.path.join(self.project.base_dir, 'across', 'glm', 'detection_data_clean_{}_2.feat'.format(self.subject.initials, session), 'stats', 'res4d.nii'))
            if data_type == 'psc':
                BOLD_data_1 = nib.load(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'detection_data_psc_{}_1.nii.gz'.format(self.subject.initials)))
                BOLD_data_2 = nib.load(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'detection_data_psc_{}_2.nii.gz'.format(self.subject.initials)))
        
        # trs:
        self.trs_1 = np.loadtxt(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'triggers_1.txt'))
        self.trs_2 = np.loadtxt(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'triggers_2.txt'))
        if self.subject.initials == 'DL':
            self.trs_3 = np.loadtxt(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'triggers_3.txt'))
        
        # pupil:
        self.pupil_data_1 = pd.read_csv(os.path.join(self.project.base_dir, self.subject.initials, 'pupil_data_1.csv'))
        self.pupil_data_2 = pd.read_csv(os.path.join(self.project.base_dir, self.subject.initials, 'pupil_data_2.csv'))
        self.pupil_data_2['cue_onset'] = self.pupil_data_2['cue_onset'] + (len(self.trs_1) * self.tr)
        
        # concatenate across sessions:
        if self.subject.initials == 'DL':
            if bold:
                if data_type == 'clean' or data_type == 'clean_ventricle' or data_type == 'clean_midbrain' or data_type == 'clean_LC' or data_type == 'clean_z':
                    BOLD_data_3 = nib.load(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'detection_NEW_data_clean_{}_3.nii.gz'.format(self.subject.initials)))
                if data_type == 'fsl_clean':
                    BOLD_data_3 = nib.load(os.path.join(self.project.base_dir, 'across', 'glm', 'detection_data_clean_{}_3.feat'.format(self.subject.initials, session), 'stats', 'res4d.nii'))
                if (data_type == 'clean_MNI') or (data_type == 'clean_MNI_smooth'):
                    BOLD_data_3 = nib.load(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'detection_data_{}_{}_3.nii.gz'.format(data_type, self.subject.initials)))
                if data_type == 'psc':
                    BOLD_data_3 = nib.load(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'detection_data_psc_{}_3.nii.gz'.format(self.subject.initials)))
            self.pupil_data_3 = pd.read_csv(os.path.join(self.project.base_dir, self.subject.initials, 'pupil_data_3.csv'))
            self.pupil_data_3['cue_onset'] = self.pupil_data_3['cue_onset'] + ((len(self.trs_1)+len(self.trs_2)) * self.tr)
            self.pupil_data = pd.concat((self.pupil_data_1, self.pupil_data_2, self.pupil_data_3))
        else:
            self.pupil_data = pd.concat((self.pupil_data_1, self.pupil_data_2))
        
        if bold:
            self.BOLD_data_1 = BOLD_data_1.get_data()
            self.BOLD_data_2 = BOLD_data_2.get_data()
            if self.subject.initials == 'DL':
                self.BOLD_data_3 = BOLD_data_3.get_data()
                
    def find_nearest(self,array,value):
        idx = (np.abs(array-value)).argmin()
        return idx
    
    def find_last(self,array,value):
        idx = (np.abs(array[::-1]-value)).argmin()
        idx = len(array) - idx
        return idx
    
    
    def ROI_event_related_average(self, rois, data_type='mcf_phys', postFix=['mcf','phys'], mask_direction=None, use_hdf5=False):
        
        """
        run deconvolution analysis on the input (mcf_psc_hpf) data that is stored in the reward hdf5 file. 
        Event data will be extracted from the .txt fsl event files used for the initial glm.
        roi argument specifies the region from which to take the data.
        """
        
        session = [self.runList[i] for i in self.conditionDict['task']][0].session
        nr_runs = len([self.runList[i] for i in self.conditionDict['task']])
        
        if session == 2:
            
            self.load_concatenated_data(data_type=data_type)
            
            for roi in rois:
                
                scalars = np.zeros((self.pupil_data.shape[0], 4))
                scalars_b = np.zeros((self.pupil_data.shape[0], 4))
                
                # trs:
                trs_1 = np.loadtxt(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'triggers_1.txt'))
                trs_2 = np.loadtxt(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'triggers_2.txt')) + (trs_1.shape[0]*2.0)
                trs = np.concatenate((trs_1, trs_2))
                if self.subject.initials == 'DL':
                    trs_3 = np.loadtxt(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'triggers_3.txt')) + (trs_1.shape[0]*2.0) + (trs_2.shape[0]*2.0)
                    trs = np.concatenate((trs_1, trs_2, trs_3))
                
                # get rois:
                mask_1 = np.array(nib.load(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'masks', roi + '_1.nii.gz')).get_data(), dtype=float)
                mask_2 = np.array(nib.load(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'masks', roi + '_2.nii.gz')).get_data(), dtype=float)
                roi_data_1 = self.BOLD_data_1[mask_1>0,:]
                roi_data_2 = self.BOLD_data_2[mask_2>0,:]
                mask_1 = mask_1[mask_1>0]
                mask_2 = mask_2[mask_2>0]
                roi_data_1 = (roi_data_1 * np.atleast_2d(mask_1).T).sum(axis=0) / sum(mask_1)
                roi_data_2 = (roi_data_2 * np.atleast_2d(mask_2).T).sum(axis=0) / sum(mask_2)
                if self.subject.initials == 'DL':
                    mask_3 = np.array(nib.load(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'masks', roi + '_3.nii.gz')).get_data(), dtype=float)
                    roi_data_3 = self.BOLD_data_3[mask_3>0,:]
                    mask_3 = mask_3[mask_3>0]
                    roi_data_3 = (roi_data_3 * np.atleast_2d(mask_3).T).sum(axis=0) / sum(mask_3)
                    
                if self.subject.initials == 'DL':
                    roi_data_m = np.concatenate((roi_data_1, roi_data_2, roi_data_3))
                else:
                    roi_data_m = np.concatenate((roi_data_1, roi_data_2,))
                print
                print roi_data_m.shape
                
                for project_out in [False, '4th_ventricle']:
                # for project_out in [False]:
                    if project_out:
                        mask_1 = np.array(nib.load(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'masks', '{}_1.nii.gz'.format(project_out))).get_data(), dtype=bool)
                        roi_data_1 = bn.nanmean(self.BOLD_data_1[mask_1,:], axis=0)
                        mask_2 = np.array(nib.load(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'masks', '{}_2.nii.gz'.format(project_out))).get_data(), dtype=bool)
                        roi_data_2 = bn.nanmean(self.BOLD_data_2[mask_2,:], axis=0)
                        if self.subject.initials == 'DL':
                        
                            mask_3 = np.array(nib.load(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'masks', '{}_3.nii.gz'.format(project_out))).get_data(), dtype=bool)
                            roi_data_3 = bn.nanmean(self.BOLD_data_3[mask_3,:], axis=0)
                            to_project_out = np.concatenate((roi_data_1, roi_data_2, roi_data_3))
                        else:
                            to_project_out = np.concatenate((roi_data_1, roi_data_2,))
                        roi_data_m_m = roi_data_m.copy()
                        roi_data_m_m[~np.isnan(roi_data_m)] = myfuncs.lin_regress_resid(roi_data_m[~np.isnan(roi_data_m)], [to_project_out[~np.isnan(roi_data_m)]]) + roi_data_m[~np.isnan(roi_data_m)].mean()
                    else:
                        roi_data_m_m = roi_data_m.copy()
                    
                    def find_nearest(array,value):
                        idx = (np.abs(array-value)).argmin()
                        return idx
                    
                    for time_locked in ['stim_locked', 'resp_locked',]:
                        
                        len_kernel = 9 # trs
                        
                        trs[np.isnan(trs)] = -100
                        
                        full_array = np.zeros((len(self.pupil_data.cue_onset), len_kernel))
                        full_array[:,:] = np.NaN
                        
                        ids = []
                        if time_locked == 'stim_locked':
                            # for i, e in enumerate(np.round(np.array(self.pupil_data.stim_onset) / self.tr,0)):     #   GIVES THE SAME THING!
                            for i, e in enumerate(np.array(self.pupil_data.cue_onset)):
                                index = find_nearest(trs,e)
                                ids.append(index)
                                full_array[i,:len(roi_data_m_m[index-2:index+7])] = roi_data_m_m[index-2:index+7]
                        if time_locked == 'resp_locked':
                            # for i, e in enumerate(np.round((np.array(self.pupil_data.cue_onset)+np.array(self.pupil_data.rt)) / self.tr,0)):        GIVES THE SAME THING!
                            for i, e in enumerate(np.array(self.pupil_data.cue_onset)+np.array(self.pupil_data.rt)):
                                index = find_nearest(trs,e)
                                full_array[i,:len(roi_data_m_m[index-3:index+6])] = roi_data_m_m[index-3:index+6] # - np.mean(roi_data_m_m[index-3:index+6][0:2])
                        
                        # save (with omissions included!):
                        np.save(os.path.join(self.project.base_dir, 'across', 'event_related_average', 'full_array_{}_{}_{}_{}_{}'.format(data_type, project_out, roi, time_locked, self.subject.initials)), full_array)
                        
    def V123_univariate(self, data_type='clean', nr_voxels=50,):
        
        import copy
        session = [self.runList[i] for i in self.conditionDict['task']][0].session
        self.load_concatenated_data(data_type=data_type)
        
        for roi in ['V1', 'V2', 'V3']:
            for center in ['center', 'surround']:
                mask = np.array(nib.load(self.runFile(stage = 'processed/mri/masks/anat', base='{}_{}'.format(roi, center))).get_data(), dtype=bool)
                
                for i, run in enumerate(self.conditionDict['loc']):
                    nifti = nib.load(self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['B0', 'mcf', 'sgtf', '0.01', 'psc'])).get_data()
                    nifti_m = nifti[mask,:]
            
                use_feat = False
                if use_feat:
                    for condition in conditions:
                        for i, run in enumerate(self.conditionDict[condition]):
                            contrast_results = nib.load(os.path.join(self.runFile(stage = 'processed/mri', run = self.runList[run], postFix = ['B0', 'mcf'], extension = '.feat'), 'stats', 'zstat1.nii.gz')).get_data()
                            contrast_results_m = contrast_results[mask]
                else:
                    start = 2
                    end = 8
                    times_cw = np.array([12, 60, 108, 156, 204, 252]) / self.tr
                    times_ccw = np.array([36, 84, 132, 180, 228, 276]) / self.tr
                    times_blank_cw = np.array([0, 48, 96, 144, 192, 240]) / self.tr
                    times_blank_ccw = np.array([24, 72, 120, 168, 216, 264]) / self.tr
        
                    time_array_cw = np.zeros((len(times_cw), 12, nifti_m.shape[0]))
                    time_array_cw [:,:] = np.NaN
                    time_array_ccw = np.zeros((len(times_ccw), 12, nifti_m.shape[0]))
                    time_array_ccw [:,:] = np.NaN
                    time_array_blank_cw = np.zeros((len(times_blank_cw), 12, nifti_m.shape[0]))
                    time_array_blank_cw [:,:] = np.NaN
                    time_array_blank_ccw = np.zeros((len(times_blank_ccw), 12, nifti_m.shape[0]))
                    time_array_blank_ccw [:,:] = np.NaN
                    for v in range(nifti_m.shape[0]):
                        for i in range(len(times_cw)):
                
                            t = times_cw[i]
                            trial = nifti_m[v,t:t+12]
                            trial = trial - nifti_m[v,t-1:t+2].mean(axis=0)
                            time_array_cw[i,:len(trial),v] = trial
                
                            t = times_ccw[i]
                            trial = nifti_m[v,t:t+12]
                            trial = trial - nifti_m[v,t-1:t+2].mean(axis=0)
                            time_array_ccw[i,:len(trial),v] = trial
                
                            t = times_blank_cw[i]
                            trial = nifti_m[v,t:t+12]
                            trial = trial - nifti_m[v,t:t+2].mean(axis=0)
                            time_array_blank_cw[i,:len(trial),v] = trial
                
                            t = times_blank_ccw[i]
                            trial = nifti_m[v,t:t+12]
                            trial = trial - nifti_m[v,t:t+2].mean(axis=0)
                            time_array_blank_ccw[i,:len(trial),v] = trial
                
                    scalars_cw = time_array_cw[:,start:end,:].mean(axis=1)
                    scalars_ccw = time_array_ccw[:,start:end,:].mean(axis=1)
                    scalars_blank_cw = time_array_blank_cw[:,start:end,:].mean(axis=1)
                    scalars_blank_ccw = time_array_blank_ccw[:,start:end,:].mean(axis=1)
                    
                    # contrast_present = (((scalars_cw+scalars_ccw)/2.0) - ((scalars_blank_cw+scalars_blank_ccw)/2.0)).mean(axis=0)
                    # contrast_cw = (scalars_cw - scalars_ccw).mean(axis=0)
                    
                    contrast_present = sp.stats.ttest_1samp( np.vstack((scalars_cw, scalars_ccw)), 0)[0] # - np.vstack((scalars_blank_cw, scalars_blank_ccw)))[0]
                    contrast_cw = sp.stats.ttest_rel(scalars_cw, scalars_ccw)[0]
                    
                for nr_voxels in np.array(np.linspace(10,150,15), dtype=int):
                
                    # indices:
                    if center == 'center':
                        ind = np.argsort(contrast_present)[-nr_voxels:][::-1]
                    if center == 'surround':
                        ind = np.argsort(contrast_present)[:nr_voxels][::-1]
                    
                    # get BOLD and pupil:
                    pupil_data = pd.read_csv(os.path.join(self.project.base_dir, self.subject.initials, 'pupil_data_{}.csv'.format(session)))
                    exec('BOLD_data = self.BOLD_data_' + str(session))
                    nifti_m = BOLD_data[mask,:]
            
                    # mask BOLD data:
                    nifti_m_m = nifti_m[ind,:]

                    # trs:
                    exec('trs = self.trs_' + str(session))
                    
                    for time_locked in ['stim_locked',]:

                        len_kernel = 9 # trs

                        trs[np.isnan(trs)] = -100

                        voxel_arrays = np.zeros((pupil_data[-np.array(pupil_data.omissions, dtype=bool)].shape[0], len_kernel, len(ind)))
                        voxel_arrays[:,:,:] = np.NaN
                        for v in range(len(ind)):
                            for i, e in enumerate(np.array(pupil_data.cue_onset[-np.array(pupil_data.omissions, dtype=bool)])):
                                index = self.find_nearest(trs,e)
                                voxel_arrays[i,:len(nifti_m_m[v,index-2:index+7]),v] = nifti_m_m[v,index-2:index+7]

                        step_lim = [-4,14]
                        step = pd.Series(np.linspace(step_lim[0], step_lim[1], len_kernel), name='time from cue (s)')
                        time_of_interest_b = [-2,2]
                        time_of_interest_b_ind = np.array(step >= time_of_interest_b[0]) * np.array(step <= time_of_interest_b[1])
                        time_of_interest = [2,12]
                        time_of_interest_ind = np.array(step >= time_of_interest[0]) * np.array(step <= time_of_interest[1])

                        # get scalars:
                        baselines = bn.nanmean(voxel_arrays[:,time_of_interest_b_ind,:], axis=1)
                        phasics = bn.nanmean(voxel_arrays[:,time_of_interest_ind,:], axis=1) - baselines

                        # save:
                        np.save(os.path.join(self.project.base_dir, 'across', 'V123', 'univariate', 'scalars_b_{}_{}_{}_{}_{}'.format(self.subject.initials, session, roi, nr_voxels, center)), baselines)
                        np.save(os.path.join(self.project.base_dir, 'across', 'V123', 'univariate', 'scalars_d_{}_{}_{}_{}_{}'.format(self.subject.initials, session, roi, nr_voxels, center)), phasics)
                        
    def V123_multivariate(self, data_type='clean',):
        
        session = [self.runList[i] for i in self.conditionDict['task']][0].session
        nr_runs = len([self.runList[i] for i in self.conditionDict['task']])
        self.load_concatenated_data(data_type=data_type)
        
        for roi in ['V1_center', 'V2_center', 'V3_center']:
        # for roi in ['V1_surround', 'V2_surround', 'V3_surround']:
        # for roi in ['V1_center',]:
            
            # mask of interest:
            mask = np.array(nib.load(self.runFile(stage = 'processed/mri/masks/anat', base=roi)).get_data(), dtype=bool)
            
            # get BOLD and pupil:
            exec('BOLD_data = self.BOLD_data_' + str(session))
            BOLD_data = BOLD_data[mask,:]
            pupil_data = pd.read_csv(os.path.join(self.project.base_dir, self.subject.initials, 'pupil_data_{}.csv'.format(session)))
    
            # trs:
            exec('trs = self.trs_' + str(session))
    
            for time_locked in ['stim_locked',]:
                len_kernel = 9 # trs
                trs[np.isnan(trs)] = -100
                voxel_arrays = np.zeros((pupil_data[-np.array(pupil_data.omissions, dtype=bool)].shape[0], len_kernel, BOLD_data.shape[0]))
                voxel_arrays[:,:,:] = np.NaN
                for v in range(BOLD_data.shape[0]):
                    for i, e in enumerate(np.array(pupil_data.cue_onset[-np.array(pupil_data.omissions, dtype=bool)])):
                        index = self.find_nearest(trs,e)
                        voxel_arrays[i,:len(BOLD_data[v,index-2:index+7]),v] = BOLD_data[v,index-2:index+7]
                step_lim = [-4,14]
                step = pd.Series(np.linspace(step_lim[0], step_lim[1], len_kernel), name='time from cue (s)')
                time_of_interest_b = [-2,2]
                time_of_interest_b_ind = np.array(step >= time_of_interest_b[0]) * np.array(step <= time_of_interest_b[1])
                time_of_interest = [2,12]
                time_of_interest_ind = np.array(step >= time_of_interest[0]) * np.array(step <= time_of_interest[1])
                # get scalars:
                baselines = bn.nanmean(voxel_arrays[:,time_of_interest_b_ind,:], axis=1)
                phasics = bn.nanmean(voxel_arrays[:,time_of_interest_ind,:], axis=1) - baselines
            
            present = np.array(pupil_data['present'])[-np.array(pupil_data['omissions'])]
            hit = np.array(pupil_data['hit'])[-np.array(pupil_data['omissions'])]
            cr = np.array(pupil_data['cr'])[-np.array(pupil_data['omissions'])]
            cw = np.array(pupil_data['signal_orientation'])[-np.array(pupil_data['omissions'])] == 45
            ccw = np.array(pupil_data['signal_orientation'])[-np.array(pupil_data['omissions'])] == 135
            # contrast_present = phasics[hit,:].mean(axis=0) - phasics[cr,:].mean(axis=0)
            # contrast_cw = phasics[hit&cw,:].mean(axis=0) - phasics[hit&ccw,:].mean(axis=0)
            
            # contrast_present = phasics[present,:].mean(axis=0) - phasics[~present,:].mean(axis=0)
            # contrast_cw = phasics[present&cw,:].mean(axis=0) - phasics[present&ccw,:].mean(axis=0)
            
            contrast_present = sp.stats.ttest_1samp(phasics[present,:],0)[0] - sp.stats.ttest_1samp(phasics[~present,:],0)[0]
            contrast_cw = sp.stats.ttest_1samp(phasics[present&cw,:],0)[0] - sp.stats.ttest_1samp(phasics[present&ccw,:],0)[0]
            
            for nr_voxels in np.array(np.linspace(10,150,15), dtype=int):
                
                # select n most responsive voxels, sort them by orientation preference:
                stim_ind = np.concatenate((np.argsort(contrast_present)[-nr_voxels/2:][::-1], np.argsort(contrast_present)[:nr_voxels/2:][::-1]))
                ori_ind = np.concatenate((np.argsort(contrast_cw)[-nr_voxels/2:][::-1], np.argsort(contrast_cw)[:nr_voxels/2:][::-1]))
                
                for ind, ind_type, contrast_results in zip([stim_ind, ori_ind], ['present', 'ori'], [contrast_present, contrast_cw]):
                    BOLD_data_m = BOLD_data[ind,:]
                    
                    for time_locked in ['stim_locked',]:
            
                        len_kernel = 9 # trs
                        trs[np.isnan(trs)] = -100
                        voxel_arrays = np.zeros((pupil_data[-np.array(pupil_data.omissions, dtype=bool)].shape[0], len_kernel, len(ind)))
                        voxel_arrays[:,:,:] = np.NaN
                        for v in range(len(ind)):
                            for i, e in enumerate(np.array(pupil_data.cue_onset[-np.array(pupil_data.omissions, dtype=bool)])):
                                index = self.find_nearest(trs,e)
                                voxel_arrays[i,:len(BOLD_data_m[v,index-2:index+7]),v] = BOLD_data_m[v,index-2:index+7]
                        step_lim = [-4,14]
                        step = pd.Series(np.linspace(step_lim[0], step_lim[1], len_kernel), name='time from cue (s)')
                        time_of_interest_b = [-2,2]
                        time_of_interest_b_ind = np.array(step >= time_of_interest_b[0]) * np.array(step <= time_of_interest_b[1])
                        time_of_interest = [2,12]
                        time_of_interest_ind = np.array(step >= time_of_interest[0]) * np.array(step <= time_of_interest[1])
            
                        # get scalars:
                        baselines = bn.nanmean(voxel_arrays[:,time_of_interest_b_ind,:], axis=1)
                        phasics = bn.nanmean(voxel_arrays[:,time_of_interest_ind,:], axis=1) - baselines
            
                        # save:
                        np.save(os.path.join(self.project.base_dir, 'across', 'V123', 'multivariate', 'template_d_{}_{}_{}_{}_{}'.format(roi, ind_type, nr_voxels, self.subject.initials, session)), contrast_results[ind])
                        np.save(os.path.join(self.project.base_dir, 'across', 'V123', 'multivariate', 'scalars_b_{}_{}_{}_{}_{}'.format(roi, ind_type, nr_voxels, self.subject.initials, session)), baselines)
                        np.save(os.path.join(self.project.base_dir, 'across', 'V123', 'multivariate', 'scalars_d_{}_{}_{}_{}_{}'.format(roi, ind_type, nr_voxels, self.subject.initials, session)), phasics)
                        
            # all voxels:
            stim_ind = np.argsort(contrast_present)
            ori_ind = np.argsort(contrast_cw)
            
            for ind, ind_type, contrast_results in zip([stim_ind, ori_ind], ['present', 'ori'], [contrast_present, contrast_cw]):
                BOLD_data_m = BOLD_data[ind,:]
                
                for time_locked in ['stim_locked',]:
                    
                    len_kernel = 9 # trs
                    trs[np.isnan(trs)] = -100
                    voxel_arrays = np.zeros((pupil_data[-np.array(pupil_data.omissions, dtype=bool)].shape[0], len_kernel, len(ind)))
                    voxel_arrays[:,:,:] = np.NaN
                    for v in range(len(ind)):
                        for i, e in enumerate(np.array(pupil_data.cue_onset[-np.array(pupil_data.omissions, dtype=bool)])):
                            index = self.find_nearest(trs,e)
                            voxel_arrays[i,:len(BOLD_data_m[v,index-2:index+7]),v] = BOLD_data_m[v,index-2:index+7]
                    step_lim = [-4,14]
                    step = pd.Series(np.linspace(step_lim[0], step_lim[1], len_kernel), name='time from cue (s)')
                    time_of_interest_b = [-2,2]
                    time_of_interest_b_ind = np.array(step >= time_of_interest_b[0]) * np.array(step <= time_of_interest_b[1])
                    time_of_interest = [2,12]
                    time_of_interest_ind = np.array(step >= time_of_interest[0]) * np.array(step <= time_of_interest[1])
                    
                    # get scalars:
                    baselines = bn.nanmean(voxel_arrays[:,time_of_interest_b_ind,:], axis=1)
                    phasics = bn.nanmean(voxel_arrays[:,time_of_interest_ind,:], axis=1) - baselines
                    
                    # save:
                    np.save(os.path.join(self.project.base_dir, 'across', 'V123', 'multivariate', 'template_d_{}_{}_{}_{}_{}'.format(roi, ind_type, 'all', self.subject.initials, session)), contrast_results[ind])
                    np.save(os.path.join(self.project.base_dir, 'across', 'V123', 'multivariate', 'scalars_b_{}_{}_{}_{}_{}'.format(roi, ind_type, 'all', self.subject.initials, session)), baselines)
                    np.save(os.path.join(self.project.base_dir, 'across', 'V123', 'multivariate', 'scalars_d_{}_{}_{}_{}_{}'.format(roi, ind_type, 'all', self.subject.initials, session)), phasics)
                    
                # # save:
                # np.save(os.path.join(self.project.base_dir, 'across', 'V123', 'voxels_{}_{}_center_stim_selective_{}_{}.npy'.format(roi, nr_voxels, self.subject.initials, session)), ind)
                # np.save(os.path.join(self.project.base_dir, 'across', 'V123', 'matrix_cw_{}_{}_{}_{}'.format(roi, nr_voxels, self.subject.initials, session)), time_array_cw)
                # np.save(os.path.join(self.project.base_dir, 'across', 'V123', 'matrix_ccw_{}_{}_{}_{}'.format(roi, nr_voxels, self.subject.initials, session)), time_array_ccw)
                # np.save(os.path.join(self.project.base_dir, 'across', 'V123', 'matrix_diff_{}_{}_{}_{}'.format(roi, nr_voxels, self.subject.initials, session)), time_array_diff)
                #
                # if (nr_voxels == 100) & (roi == 'V1_center'):
                #
                #     # line = int(np.where(np.diff(contrast_array[contrast_sort]) == np.diff(contrast_array[contrast_sort]).max())[0])
                #     x = np.arange(0,24,2)
                #     line = np.argmax(contrast_results_m.mean(axis=0)[ind]<0)
                #
                #     import matplotlib.gridspec as gridspec
                #
                #     fig = plt.figure(figsize=(9, 4))
                #     gs = gridspec.GridSpec(1, 6, width_ratios=[4,1,4,1,4,1])
                #
                #     ax = plt.subplot(gs[0])
                #     im = ax.pcolormesh(x, np.arange(nr_voxels+1), time_array_cw.T, cmap='bwr', vmin=-3, vmax=3)
                #     plt.axhline(line, color='k')
                #     plt.xlim(x[0],x[-1])
                #     plt.title('CW')
                #     plt.xlabel('Time (s)')
                #     plt.ylabel('Voxel # (sorted)')
                #     fig.colorbar(im, ax=ax, fraction=0.25, pad=0.1)
                #
                #     ax = plt.subplot(gs[1])
                #     ax.pcolormesh(np.atleast_2d(np.array([0,1])), np.arange(nr_voxels+1), np.atleast_2d(time_array_cw.T[:,start:end].mean(axis=1)).T, cmap='bwr', vmin=-3, vmax=3)
                #     plt.axhline(line, color='k')
                #     plt.title('CW')
                #     ax.axes.get_xaxis().set_ticks([])
                #     ax.axes.get_yaxis().set_ticks([])
                #
                #     ax = plt.subplot(gs[2])
                #     im = ax.pcolormesh(x, np.arange(nr_voxels+1), time_array_ccw.T, cmap='bwr', vmin=-3, vmax=3)
                #     plt.axhline(line, color='k')
                #     plt.xlim(x[0],x[-1])
                #     plt.title('CCW')
                #     plt.xlabel('Time (s)')
                #     fig.colorbar(im, ax=ax, fraction=0.25, pad=0.1)
                #
                #     ax = plt.subplot(gs[3])
                #     ax.pcolormesh(np.atleast_2d(np.array([0,1])), np.arange(nr_voxels+1), np.atleast_2d(time_array_ccw.T[:,start:end].mean(axis=1)).T, cmap='bwr', vmin=-3, vmax=3)
                #     plt.axhline(line, color='k')
                #     plt.title('CCW')
                #     ax.axes.get_xaxis().set_ticks([])
                #     ax.axes.get_yaxis().set_ticks([])
                #
                #     ax = plt.subplot(gs[4])
                #     im = ax.pcolormesh(x, np.arange(nr_voxels+1), time_array_diff.T, cmap='bwr', vmin=-3, vmax=3)
                #     plt.axhline(line, color='k')
                #     plt.xlim(x[0],x[-1])
                #     plt.title('CW - CCW')
                #     plt.xlabel('Time (s)')
                #     fig.colorbar(im, ax=ax, fraction=0.25, pad=0.1)
                #
                #     ax = plt.subplot(gs[5])
                #     ax.pcolormesh(np.atleast_2d(np.array([0,1])), np.arange(nr_voxels+1), np.atleast_2d(time_array_diff.T[:,start:end].mean(axis=1)).T, cmap='bwr', vmin=-3, vmax=3)
                #     plt.axhline(line, color='k')
                #     plt.title('CW - CCW')
                #     ax.axes.get_xaxis().set_ticks([])
                #     ax.axes.get_yaxis().set_ticks([])
                #
                #     plt.tight_layout()
                #     fig.savefig(os.path.join(self.stageFolder(stage = 'processed/mri/figs'), 'loc_matrix_cw.pdf'))
                #     fig.savefig(os.path.join(self.fig_dir, 'multivariate', '{}_loc_matrices_{}.pdf'.format(self.subject.initials, session)))
    
    def WHOLEBRAIN_event_related_average(self, data_type='clean_MNI'):
        
        session = [self.runList[i] for i in self.conditionDict['task']][0].session
        nr_runs = len([self.runList[i] for i in self.conditionDict['task']])
        
        if session == 2:
            
            self.load_concatenated_data(data_type=data_type)
            self.pupil_data = self.pupil_data[-np.array(self.pupil_data.omissions, dtype=bool)]
            
            brain_mask = nib.load('/home/shared/UvA/Niels_UvA/mni_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz')
            mask = np.array(brain_mask.get_data(), dtype=bool)
            
            # BOLD:
            BOLD_data = np.hstack((self.BOLD_data_1[mask,:], self.BOLD_data_2[mask,:]))
            
            # trs:
            trs_1 = np.loadtxt(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'triggers_1.txt'))
            trs_2 = np.loadtxt(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'triggers_2.txt')) + (trs_1.shape[0]*2.0)
            trs = np.concatenate((trs_1, trs_2))
                
            if self.subject.initials == 'DL':
                BOLD_data = np.hstack((self.BOLD_data_1[mask,:], self.BOLD_data_2[mask,:], self.BOLD_data_3[mask,:]))
                trs_3 = np.loadtxt(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'triggers_3.txt')) + (trs_1.shape[0]*2.0) + (trs_2.shape[0]*2.0)
                trs = np.concatenate((trs_1, trs_2, trs_3))
                
            for time_locked in ['stim_locked']:
            # for time_locked in ['resp_locked']:
                
                len_kernel = 9 # trs
                
                trs[np.isnan(trs)] = -100
                
                nr_trials = self.pupil_data.shape[0]
                
                def find_nearest(array,value):
                    idx = (np.abs(array-value)).argmin()
                    return idx
                
                # get baselines based on stim_locked data:
                # ----------------------------------------
                
                voxel_arrays = np.zeros((BOLD_data.shape[0], nr_trials, len_kernel), dtype=np.float32)
                voxel_arrays[:,:,:] = np.NaN
                for v in range(BOLD_data.shape[0]):
                    for i, e in enumerate(np.array(self.pupil_data.cue_onset)):
                        index = find_nearest(trs,e)
                        voxel_arrays[v,i,:len(BOLD_data[index-2:index+7])] = BOLD_data[v,index-2:index+7]
                
                
                time_of_interest_b = [-2,2]
                step_lim = [-4,14]
                step = pd.Series(np.linspace(step_lim[0], step_lim[1], len_kernel), name='time from cue (s)')
                time_of_interest_b_ind = np.array(step >= time_of_interest_b[0]) * np.array(step <= time_of_interest_b[1])
                
                # get baselines:
                baselines = bn.nanmean(voxel_arrays[:,:,time_of_interest_b_ind], axis=-1)
                
                # get phasics:
                # ------------
                
                if time_locked == 'stim_locked':
                    pass
                if time_locked == 'resp_locked':
                    voxel_arrays = np.zeros((BOLD_data.shape[0], nr_trials, len_kernel), dtype=np.float32)
                    voxel_arrays[:,:,:] = np.NaN
                    for v in range(BOLD_data.shape[0]):
                        for i, e in enumerate((np.array(self.pupil_data.cue_onset)+np.array(self.pupil_data.rt))):
                            index = find_nearest(trs,e)
                            voxel_arrays[v,i,:len(BOLD_data[index-2:index+7])] = BOLD_data[v,index-2:index+7]
                    
                # for type_response in ['mean', 'std']:
                # params:
                # if type_response == 'mean':
                if time_locked == 'stim_locked':
                    step_lim = [-4,14]
                    time_of_interest = [2,12]
                if time_locked == 'resp_locked':
                    step_lim = [-6,12]
                    time_of_interest = [2,12]
                # elif type_response == 'std':
                #     if time_locked == 'stim_locked':
                #         step_lim = [-4,14]
                #         time_of_interest = [6,12]
                #     if time_locked == 'resp_locked':
                #         step_lim = [-6,12]
                #         time_of_interest = [6,12]
                
                # step size:
                if time_locked == 'stim_locked':
                    step = pd.Series(np.linspace(step_lim[0], step_lim[1], len_kernel), name='time from cue (s)')
                if time_locked == 'resp_locked':
                    step = pd.Series(np.linspace(step_lim[0], step_lim[1], len_kernel), name='time from report (s)')
        
                # time window of interest:
                time_of_interest_ind = np.array(step >= time_of_interest[0]) * np.array(step <= time_of_interest[1])
            
                # get scalars:
                phasics = bn.nanmean(voxel_arrays[:,:,time_of_interest_ind], axis=-1) - baselines
                
                print
                print 'phasics'
                print phasics.shape
                print
                
                # regress out RT:
                rt = np.array(self.pupil_data['rt'])
                for i in range(phasics.shape[0]):
                    for s in np.array(np.unique(self.pupil_data['session']), dtype=int):
                        session_ind = np.array(self.pupil_data.session == s)
                        phasics[i,session_ind] = myfuncs.lin_regress_resid(phasics[i,session_ind], [rt[session_ind]]) + phasics[i,session_ind].mean()
                
                print
                print 'phasics'
                print phasics.shape
                print
                
                # save:
                results = np.zeros((mask.shape[0],mask.shape[1],mask.shape[2], nr_trials))
                results[mask,:] = baselines
                res_nii_file = nib.Nifti1Image(results, affine=brain_mask.get_affine(), header=brain_mask.get_header())
                res_nii_file.set_data_dtype(np.float32)
                nib.save(res_nii_file, os.path.join(self.project.base_dir, 'across', 'event_related_average_wb', 'scalars_{}_{}_{}_b.nii.gz'.format(data_type, time_locked, self.subject.initials)))
                
                results = np.zeros((mask.shape[0],mask.shape[1],mask.shape[2], nr_trials))
                results[mask,:] = phasics
                res_nii_file = nib.Nifti1Image(results, affine=brain_mask.get_affine(), header=brain_mask.get_header())
                res_nii_file.set_data_dtype(np.float32)
                nib.save(res_nii_file, os.path.join(self.project.base_dir, 'across', 'event_related_average_wb', 'scalars_{}_{}_{}_d.nii.gz'.format(data_type, time_locked, self.subject.initials)))

    def WHOLEBRAIN_correlation(self, data_type='clean_MNI'):
        
        session = [self.runList[i] for i in self.conditionDict['task']][0].session
        nr_runs = len([self.runList[i] for i in self.conditionDict['task']])
        time_locked = 'stim_locked'
        if session == 2:
            
            self.load_concatenated_data(data_type=data_type, bold=False)
            omissions = np.array(self.pupil_data.omissions, dtype=bool)
            self.pupil_data = self.pupil_data[-omissions]
            
            brain_mask = nib.load('/home/shared/UvA/Niels_UvA/mni_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz')
            mask = np.array(brain_mask.get_data(), dtype=bool)
            
            scalars_d = np.array(nib.load(os.path.join(self.project.base_dir, 'across', 'event_related_average_wb', 'scalars_{}_{}_{}_d.nii.gz'.format(data_type, time_locked, self.subject.initials))).get_data())[mask,:]
            
            # single trial correlation pupil and BOLD:
            # ----------------------------------------
            iti = np.array(self.pupil_data['iti'])
            rt = np.array(self.pupil_data['rt'])
            present = np.array(self.pupil_data['present'])
            
            if time_locked == 'stim_locked':
                step_range = [-4,12]
                time_of_interest = [2,12]
            if time_locked == 'resp_locked':
                step_range = [-6,12]
                time_of_interest = [2,12]
            time_of_interest_b = [-2,2]
            ventricle = np.load(os.path.join(self.project.base_dir, 'across', 'event_related_average', 'full_array_{}_{}_{}_{}.npy'.format('clean_False', '4th_ventricle', 'stim_locked', self.subject.initials)))[-omissions,:]
            kernel_length = ventricle.shape[1]
            step = pd.Series(np.linspace(step_range[0], step_range[1], kernel_length), name='time from cue (s)')
            time_of_interest_b_ind = np.array(step >= time_of_interest_b[0]) * np.array(step < time_of_interest_b[1])
            time_of_interest_ind = np.array(step >= time_of_interest[0]) * np.array(step < time_of_interest[1])
            ventricle_baselines = bn.nanmean(ventricle[:,time_of_interest_b_ind], axis=1)
            ventricle_phasics = bn.nanmean(ventricle[:,time_of_interest_ind], axis=1) - ventricle_baselines
            
            # dilation:
            corrs = np.zeros(scalars_d.shape[0])
            for v in range(scalars_d.shape[0]):
                BOLD = scalars_d[v,:]
                pupil = np.array(self.pupil_data['pupil_d'])
                for s in np.array(np.unique(self.pupil_data['session']), dtype=int):
                    session_ind = np.array(self.pupil_data.session == s)
                    pupil[session_ind] = myfuncs.lin_regress_resid(pupil[session_ind], [rt[session_ind], present[session_ind]])
                    if data_type == 'clean_MNI':
                        BOLD[session_ind] = myfuncs.lin_regress_resid(BOLD[session_ind], [rt[session_ind], present[session_ind], ventricle_phasics[session_ind]] )
                    else:
                        BOLD[session_ind] = myfuncs.lin_regress_resid(BOLD[session_ind], [rt[session_ind], present[session_ind]])
                corrs[v] = sp.stats.pearsonr(pupil, BOLD)[0]
            
            # save:
            results = np.zeros((mask.shape[0],mask.shape[1],mask.shape[2]))
            results[mask] = corrs
            res_nii_file = nib.Nifti1Image(results, affine=brain_mask.get_affine(), header=brain_mask.get_header())
            res_nii_file.set_data_dtype(np.float32)
            nib.save(res_nii_file, os.path.join(self.project.base_dir, 'across', 'correlation', 'whole_brain_{}_{}_{}_{}_{}.nii.gz'.format('mean', data_type, time_locked, self.subject.initials, 'pupil_d')))
    
    def WHOLEBRAIN_correlation_choice(self, data_type='clean_MNI'):
        
        session = [self.runList[i] for i in self.conditionDict['task']][0].session
        nr_runs = len([self.runList[i] for i in self.conditionDict['task']])
        time_locked = 'stim_locked'
        
        self.load_concatenated_data(data_type=data_type, bold=False)
        omissions = np.array(self.pupil_data.omissions, dtype=bool)
        self.pupil_data = self.pupil_data[-omissions]
        
        brain_mask = nib.load('/home/shared/UvA/Niels_UvA/mni_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz')
        mask = np.array(brain_mask.get_data(), dtype=bool)
        
        scalars_d = np.array(nib.load(os.path.join(self.project.base_dir, 'across', 'event_related_average_wb', 'scalars_{}_{}_{}_d.nii.gz'.format(data_type, time_locked, self.subject.initials))).get_data())[mask,:]
        
        # load dataframe:
        data_frame = pd.read_csv(os.path.join(self.project.base_dir, 'across', 'event_related_average_wb', 'MNI_dataframe_1.csv'))
        data_frame = data_frame[data_frame.subject == self.subject.initials]
        
        # per session:
        session_ind = np.array(data_frame['session'], dtype=int) == session
        scalars_d = scalars_d[:,session_ind]
        rt = np.array(self.pupil_data['rt'])[session_ind]
        yes = np.array(self.pupil_data['yes'], dtype=bool)[session_ind]
        no = ~yes
        present = np.array(self.pupil_data['present'], dtype=bool)[session_ind]
        absent = ~present
        
        # single trial correlation BOLD - stimulus / choice:
        # --------------------------------------------------
        
        logistic = sklearn.linear_model.LogisticRegression(C=1e5, fit_intercept=True)
        corrs_stim = np.zeros(scalars_d.shape[0])
        corrs_choice = np.zeros(scalars_d.shape[0])
        for v in range(scalars_d.shape[0]):
            
            # BOLD:
            BOLD = scalars_d[v,:]
            
            # stim:
            logistic.fit(np.atleast_2d(BOLD[yes]).T, np.array(present[yes], dtype=int))
            r1 = np.exp(logistic.coef_) / (1 + np.exp(logistic.coef_))
            logistic.fit(np.atleast_2d(BOLD[no]).T, np.array(present[no], dtype=int))
            r2 = np.exp(logistic.coef_) / (1 + np.exp(logistic.coef_))
            corrs_stim[v] = (r1+r2)/2.0
        
            # choice:
            logistic.fit(np.atleast_2d(BOLD[present]).T, np.array(yes[present], dtype=int))
            r1 = np.exp(logistic.coef_) / (1 + np.exp(logistic.coef_))
            logistic.fit(np.atleast_2d(BOLD[absent]).T, np.array(yes[absent], dtype=int))
            r2 = np.exp(logistic.coef_) / (1 + np.exp(logistic.coef_))
            corrs_choice[v] = (r1+r2)/2.0
            
        # save:
        results = np.zeros((mask.shape[0],mask.shape[1],mask.shape[2]))
        results[mask] = corrs_stim
        res_nii_file = nib.Nifti1Image(results, affine=brain_mask.get_affine(), header=brain_mask.get_header())
        res_nii_file.set_data_dtype(np.float32)
        nib.save(res_nii_file, os.path.join(self.project.base_dir, 'across', 'correlation', 'whole_brain_{}_{}_{}_{}_BOLD_present_s{}.nii.gz'.format('mean', data_type, time_locked, self.subject.initials, session)))
        
        results = np.zeros((mask.shape[0],mask.shape[1],mask.shape[2]))
        results[mask] = corrs_choice
        res_nii_file = nib.Nifti1Image(results, affine=brain_mask.get_affine(), header=brain_mask.get_header())
        res_nii_file.set_data_dtype(np.float32)
        nib.save(res_nii_file, os.path.join(self.project.base_dir, 'across', 'correlation', 'whole_brain_{}_{}_{}_{}_BOLD_choice_s{}.nii.gz'.format('mean', data_type, time_locked, self.subject.initials, session)))
        
    def WHOLEBRAIN_searchlight_decoding(self, data_type='clean_MNI'):
        
        from sklearn import svm
        from sklearn.cross_validation import StratifiedKFold
        from sklearn.cross_validation import KFold
        import nilearn.decoding
        from nilearn.image import index_img
        
        session = [self.runList[i] for i in self.conditionDict['task']][0].session
        nr_runs = len([self.runList[i] for i in self.conditionDict['task']])
        
        time_locked = 'stim_locked'
        
        self.load_concatenated_data(data_type=data_type, bold=False)
        self.pupil_data = self.pupil_data[-np.array(self.pupil_data.omissions, dtype=bool)]
    
        # load dataframe:
        data_frame = pd.read_csv(os.path.join(self.project.base_dir, 'across', 'event_related_average_wb', 'MNI_dataframe_1.csv'))
        data_frame = data_frame[data_frame.subject == self.subject.initials]
        
        # per session:
        session_ind = np.array(data_frame['session'], dtype=int) == session
        rt = np.array(self.pupil_data['rt'])[session_ind]
        yes = np.array(self.pupil_data['yes'], dtype=bool)[session_ind]
        no = ~yes
        present = np.array(self.pupil_data['present'], dtype=bool)[session_ind]
        absent = ~present
        
        # mask:
        mask_img = nib.load('/home/shared/Niels_UvA/brainstem_masks/epi_box.nii.gz')
        
        # fMRI image:
        fmri_img = nib.load(os.path.join(self.project.base_dir, 'across', 'event_related_average_wb', 'scalars_{}_{}_{}_RT_d.nii.gz'.format(data_type, time_locked, self.subject.initials)))
        fmri_img = index_img(fmri_img, session_ind)
        
        # njobs
        n_jobs = 10
        n_folds = 3
        
        from nilearn.image import new_img_like, load_img
        process_mask = mask_img.get_data().astype(np.int)
        picked_slice = 40
        process_mask[..., (picked_slice + 1):] = 0
        process_mask[..., :picked_slice] = 0
        process_mask[:, 30:, :] = 0
        process_mask_img = new_img_like(mask_img, process_mask)
        
        
        # shell()
        
        # FOR CHOICE WITH STIMULUS FACTORED OUT:#
        #########################################
        
        # for signal present:
        fmri_img_present = index_img(fmri_img, present)
        # cv = KFold(yes[present].size, n_folds=n_folds)
        cv = StratifiedKFold(yes[present], n_folds=n_folds)
        searchlight = nilearn.decoding.SearchLight(mask_img, process_mask_img=mask_img, radius=10, n_jobs=n_jobs, verbose=1, cv=cv)
        # searchlight = nilearn.decoding.SearchLight(mask_img, process_mask_img=process_mask_img, radius=10, n_jobs=n_jobs, verbose=1, cv=cv, estimator=svm.LinearSVC(class_weight='balanced',), scoring='accuracy')
        searchlight.fit(fmri_img_present, np.array(yes[present], dtype=int))
        res_present = searchlight.scores_
        a = res_present[np.array(process_mask, dtype=bool)]
        
        fmri_img_absent = index_img(fmri_img, absent)
        # cv = KFold(yes[~present].size, n_folds=n_folds)
        cv = StratifiedKFold(yes[absent], n_folds=n_folds)
        searchlight = nilearn.decoding.SearchLight(mask_img, process_mask_img=mask_img, radius=10, n_jobs=n_jobs, verbose=1, cv=cv)
        # searchlight = nilearn.decoding.SearchLight(mask_img, process_mask_img=process_mask_img, radius=10, n_jobs=n_jobs, verbose=1, cv=cv, estimator=svm.LinearSVC(class_weight='balanced',), scoring='accuracy')
        searchlight.fit(fmri_img_absent, np.array(yes[absent], dtype=int))
        res_absent = searchlight.scores_
        b = res_absent[np.array(process_mask, dtype=bool)]
        
        cv = StratifiedKFold(yes, n_folds=n_folds)
        searchlight = nilearn.decoding.SearchLight(mask_img, process_mask_img=mask_img, radius=10, n_jobs=n_jobs, verbose=1, cv=cv)
        # searchlight = nilearn.decoding.SearchLight(mask_img, process_mask_img=process_mask_img, radius=10, n_jobs=n_jobs, verbose=1, cv=cv, estimator=svm.LinearSVC(class_weight='balanced',), scoring='precision')
        searchlight.fit(fmri_img, np.array(yes, dtype=int))
        res_all = searchlight.scores_
        c = res_all[np.array(process_mask, dtype=bool)]
        
        
        
        
        
        # # for signal present:
        # fmri_img_present = index_img(fmri_img, present)
        # # cv = KFold(yes[present].size, n_folds=n_folds)
        # cv = StratifiedKFold(yes[present], n_folds=n_folds)
        # searchlight = nilearn.decoding.SearchLight(mask_img, process_mask_img=mask_img, radius=10, n_jobs=n_jobs, verbose=1, cv=cv, estimator=svm.SVC(kernel='linear', class_weight='balanced',), scoring='precision')
        # searchlight.fit(fmri_img_present, np.array(yes[present], dtype=int))
        # res_present = searchlight.scores_
        #
        # # for signal absent:
        # fmri_img_absent = index_img(fmri_img, absent)
        # # cv = KFold(yes[~present].size, n_folds=n_folds)
        # cv = StratifiedKFold(yes[~present], n_folds=n_folds)
        # searchlight = nilearn.decoding.SearchLight(mask_img, process_mask_img=mask_img, radius=10, n_jobs=n_jobs, verbose=1, cv=cv, estimator=svm.SVC(kernel='linear', class_weight='balanced',), scoring='precision')
        # searchlight.fit(fmri_img_absent, np.array(yes[absent], dtype=int))
        # res_absent = searchlight.scores_
        #
        # # combine:
        # results = (res_present+res_absent) / 2.0
        # results = nib.Nifti1Image(results, affine=nib.load('/home/shared/Niels_UvA/brainstem_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz').affine)
        # nib.save(results, os.path.join(self.project.base_dir, 'across', 'event_related_average_wb', 'searchlight_choice_{}_{}_{}_s{}.nii.gz'.format(data_type, time_locked, self.subject.initials, session)))
        
        # combine:
        results = res_present.copy()
        results = nib.Nifti1Image(results, affine=nib.load('/home/shared/Niels_UvA/brainstem_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz').affine)
        nib.save(results, os.path.join(self.project.base_dir, 'across', 'event_related_average_wb', 'searchlight_choice_{}_{}_{}_s{}_present.nii.gz'.format(data_type, time_locked, self.subject.initials, session)))
        results = res_absent.copy()
        results = nib.Nifti1Image(results, affine=nib.load('/home/shared/Niels_UvA/brainstem_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz').affine)
        nib.save(results, os.path.join(self.project.base_dir, 'across', 'event_related_average_wb', 'searchlight_choice_{}_{}_{}_s{}_absent.nii.gz'.format(data_type, time_locked, self.subject.initials, session)))
        results = (res_present+res_absent) / 2.0
        results = nib.Nifti1Image(results, affine=nib.load('/home/shared/Niels_UvA/brainstem_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz').affine)
        nib.save(results, os.path.join(self.project.base_dir, 'across', 'event_related_average_wb', 'searchlight_stimulus_{}_{}_{}_s{}.nii.gz'.format(data_type, time_locked, self.subject.initials, session)))
        results = res_all.copy()
        results = nib.Nifti1Image(results, affine=nib.load('/home/shared/Niels_UvA/brainstem_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz').affine)
        nib.save(results, os.path.join(self.project.base_dir, 'across', 'event_related_average_wb', 'searchlight_choice_{}_{}_{}_s{}_all.nii.gz'.format(data_type, time_locked, self.subject.initials, session)))
        
        # # FOR STIMULUS WITH CHOICE FACTORED OUT:#
        # #########################################
        #
        # # for signal present:
        # fmri_img_yes = index_img(fmri_img, yes)
        # # cv = KFold(present[yes].size, n_folds=n_folds)
        # cv = StratifiedKFold(present[yes], n_folds=n_folds)
        # searchlight = nilearn.decoding.SearchLight(mask_img, process_mask_img=mask_img, radius=2, n_jobs=n_jobs, verbose=1, cv=cv)
        # searchlight.fit(fmri_img_yes, np.array(present[yes], dtype=int))
        # res_yes = searchlight.scores_
        #
        # # for signal absent:
        # fmri_img_no = index_img(fmri_img, ~yes)
        # # cv = KFold(present[~yes].size, n_folds=n_folds)
        # cv = StratifiedKFold(present[~yes], n_folds=n_folds)
        # searchlight = nilearn.decoding.SearchLight(mask_img, process_mask_img=mask_img, radius=2, n_jobs=n_jobs, verbose=1, cv=cv)
        # searchlight.fit(fmri_img_no, np.array(present[~yes], dtype=int))
        # res_no = searchlight.scores_
        #
        # # combine:
        # results = (res_yes+res_no) / 2.0
        # results = nib.Nifti1Image(results, affine=nib.load('/home/shared/Niels_UvA/brainstem_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz').affine)
        # nib.save(results, os.path.join(self.project.base_dir, 'across', 'event_related_average_wb', 'searchlight_stimulus_{}_{}_{}_s{}.nii.gz'.format(data_type, time_locked, self.subject.initials, session)))
        
    def WHOLEBRAIN_searchlight_decoding2(self, data_type='clean_MNI'):
        
        from sklearn import svm
        from sklearn.cross_validation import StratifiedKFold
        from sklearn.cross_validation import KFold
        import nilearn.decoding
        from nilearn.image import index_img
        
        session = [self.runList[i] for i in self.conditionDict['task']][0].session
        nr_runs = len([self.runList[i] for i in self.conditionDict['task']])
        
        time_locked = 'stim_locked'
        
        self.load_concatenated_data(data_type=data_type, bold=False)
        self.pupil_data = self.pupil_data[-np.array(self.pupil_data.omissions, dtype=bool)]
    
        # load dataframe:
        data_frame = pd.read_csv(os.path.join(self.project.base_dir, 'across', 'event_related_average_wb', 'MNI_dataframe_1.csv'))
        data_frame = data_frame[data_frame.subject == self.subject.initials]
        
        # per session:
        session_ind = np.array(data_frame['session'], dtype=int) == session
        rt = np.array(self.pupil_data['rt'])[session_ind]
        yes = np.array(self.pupil_data['yes'], dtype=int)[session_ind]
        present = np.array(self.pupil_data['present'], dtype=int)[session_ind]
        
        # mask:
        mask_img = nib.load('/home/shared/UvA/Niels_UvA/mni_masks/2014_fMRI_yesno_epi_box_half.nii.gz')
        
        # fMRI image:
        fmri_img = nib.load(os.path.join(self.project.base_dir, 'across', 'event_related_average_wb', 'scalars_{}_{}_{}_RT_d.nii.gz'.format(data_type, time_locked, self.subject.initials)))
        fmri_img = index_img(fmri_img, session_ind)
        
        # njobs
        n_jobs = 1
        n_folds = 3
        
        # FOR CHOICE WITH STIMULUS FACTORED OUT:#
        #########################################
        
        # for signal present:
        cv = StratifiedKFold(yes, n_folds=n_folds)
        # searchlight = nilearn.decoding.SearchLight(mask_img, process_mask_img=mask_img, radius=10, n_jobs=n_jobs, verbose=1, cv=cv)
        searchlight = nilearn.decoding.SearchLight(mask_img, process_mask_img=mask_img, radius=10, n_jobs=n_jobs, verbose=1, cv=cv, estimator=svm.LinearSVC(class_weight='balanced',),) #scoring='accuracy')
        searchlight.fit(fmri_img, np.array(yes, dtype=int))
        results = searchlight.scores_
        
        # combine:
        results = nib.Nifti1Image(results, affine=nib.load('/home/shared/UvA/Niels_UvA/mni_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz').affine)
        nib.save(results, os.path.join(self.project.base_dir, 'across', 'event_related_average_wb', 'searchlight2_choice_{}_{}_{}_s{}_choice.nii.gz'.format(data_type, time_locked, self.subject.initials, session)))
    


    
    def create_post_preprocessing_dir(self,):
        
        session = [self.runList[i] for i in self.conditionDict['task']][0].session
        subj_id = self.subject.initials
        
        if subj_id == 'AV':
            sub = 'sub-01'
        elif subj_id == 'BL':
            sub = 'sub-02'
        elif subj_id == 'DE':
            sub = 'sub-03'
        elif subj_id == 'DL':
            sub = 'sub-04'
        elif subj_id == 'EP':
            sub = 'sub-05'
        elif subj_id == 'JG':
            sub = 'sub-06'
        elif subj_id == 'JS':
            sub = 'sub-07'
        elif subj_id == 'LH':
            sub = 'sub-08'
        elif subj_id == 'LP':
            sub = 'sub-09'
        elif subj_id == 'MG':
            sub = 'sub-10'
        elif subj_id == 'NS':
            sub = 'sub-11'
        elif subj_id == 'OC':
            sub = 'sub-12'
        elif subj_id == 'TK':
            sub = 'sub-13'
        elif subj_id == 'TN':
            sub = 'sub-14'
        
        os.system('mkdir {}'.format(os.path.join(self.project.base_dir, 'post_preprocess',)))
        os.system('mkdir {}'.format(os.path.join(self.project.base_dir, 'post_preprocess', sub)))
        os.system('cp /home/shared/UvA/Niels_UvA/Visual_UvA/data/{}/files/detection_data_clean_{}_{}.nii.gz /home/shared/UvA/Niels_UvA/Visual_UvA/data/post_preprocess/{}/{}-ses-0{}-task.nii.gz'.format(subj_id, subj_id, session, sub, sub, session))
        os.system('cp /home/shared/UvA/Niels_UvA/Visual_UvA/data/{}/files/detection_data_clean_MNI_{}_{}.nii.gz /home/shared/UvA/Niels_UvA/Visual_UvA/data/post_preprocess/{}/{}-ses-0{}-task_mni.nii.gz'.format(subj_id, subj_id, session, sub, sub, session))
        if session == 2:
            os.system('cp -r /home/shared/UvA/Niels_UvA/Visual_UvA/data/{}/files/masks/ /home/shared/UvA/Niels_UvA/Visual_UvA/data/post_preprocess/{}/masks/'.format(subj_id, sub,))
        
    def copy_freesurfer_labels(self):
        
        labels = [
        'G_parietal_sup', 
        'G_pariet_inf-Supramar',
        'G_pariet_inf-Angular',
        'G_insular_short',
        'S_intrapariet_and_P_trans',
        'S_front_middle',
        'S_front_sup',
        'S_front_inf',
        'G_and_S_cingul-Ant',
        'G_and_S_cingul-Mid-Ant',
        'G_and_S_cingul-Mid-Post',
        'G_pariet_inf-Supramar',
        'G_front_middle',
        'S_circular_insula_ant',
        ]
        for cortex in ['lh', 'rh']:
            for label in labels:
                copy_in = '$SUBJECTS_DIR/{}/label/aparc.a2009s/{}.{}.label'.format(self.subject.standardFSID, cortex, label)
                copy_out = '$SUBJECTS_DIR/{}/label/2014_custom/{}.{}.label'.format(self.subject.standardFSID, cortex, label)
                os.system('cp ' + copy_in + ' ' + copy_out)
    

        
    def grab_LC_masks2(self, base_name):
        
        for r in [self.runList[i] for i in self.conditionDict['TSE_anat']]:
        
            mask_in = os.path.join(self.stageFolder('processed/mri/masks/anat'), base_name + '.nii.gz')
            mask_out = os.path.join(self.project.base_dir, 'LCs', self.runFile('processed/mri', run=r).split('.')[-3].split('/')[-1] + '_' + base_name + '.nii.gz') 
            os.system('cp ' + mask_in + ' ' + mask_out)
    
    def number_voxels(self, roi, threshold=0.1):
        
        session = [self.runList[i] for i in self.conditionDict['task']][0].session
        
        if roi == 'superior_colliculus':
            threshold = 0.9
        
        mask = np.array(NiftiImage(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'masks', '{}_{}.nii.gz'.format(roi, session))).data, dtype=bool)
        
        print 'subj: {}, roi: {}, session: {}, voxels: {}'.format(self.subject.initials, roi, session, sum(mask))
        
        return sum(mask)
            
    
    def register_SPM_anatomy(self):
        
        np.savetxt('/home/shared/Niels_UvA/brainstem_masks/raw/SPM_anatomy_transform/eye.mtx', np.eye(4), fmt = '%1.1f')
        
        fO = FlirtOperator(inputObject = '/home/shared/Niels_UvA/brainstem_masks/raw/SPM_anatomy_transform/MNI_standard_2x2x2.nii.gz',  referenceFileName = '/home/shared/Niels_UvA/brainstem_masks/raw/SPM_anatomy_transform/MNI_standard_SPM_1x1x1.nii.gz')
        fO.configureRun( transformMatrixFileName = '/home/shared/Niels_UvA/brainstem_masks/raw/SPM_anatomy_transform/trans_standard_to_standard_SPM.mat', outputFileName = '/home/shared/Niels_UvA/brainstem_masks/raw/SPM_anatomy_transform/MNI_standard_2x2x2_TRANS.nii.gz' ) 
        # fO.configureApply( transformMatrixFileName = None, outputFileName = self.runFile(stage = 'processed/mri', run = r, postFix = ['mcf','res'] ) )
        fO.execute()
        
        # invert:
        invFl = InvertFlirtOperator('/home/shared/Niels_UvA/brainstem_masks/raw/SPM_anatomy_transform/trans_standard_to_standard_SPM.mat')
        invFl.configure()
        invFl.execute()
        
        # transform basal forebrain:
        fO = FlirtOperator(inputObject = '/home/shared/Niels_UvA/brainstem_masks/raw/SPM_anatomy_transform/Bforebrain_123.nii.gz',  referenceFileName = '/home/shared/Niels_UvA/brainstem_masks/raw/SPM_anatomy_transform/MNI_standard_2x2x2.nii.gz')
        fO.configureApply( transformMatrixFileName = '/home/shared/Niels_UvA/brainstem_masks/raw/SPM_anatomy_transform/trans_standard_to_standard_SPM_inv.mat', outputFileName = '/home/shared/Niels_UvA/brainstem_masks/raw/SPM_anatomy_transform/Bforebrain_123_TRANS.nii.gz', sinc=False) 
        fO.execute()
        
    def transform_LC_mask_TSE2Func(self, min_nr_voxels=12):
        
        nr_voxels =  (NiftiImage(self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['TSE_anat'][0]], postFix=['LC', 'JW'])).data > 0).ravel().sum()
        
        return nr_voxels
        
        # # transform LC:
        # ##############
        # inputObject = self.runFile(stage='processed/mri', run=self.runList[self.conditionDict['TSE_anat'][0]], postFix=['LC', 'JW'])
        # mask = NiftiImage(inputObject)
        # mask_data = mask.data
        # mask_data[mask_data == 99] = 1
        # mask.data = mask_data
        # mask.save(inputObject)
        #
        # # trilinear:
        # inputObject = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['TSE_anat'][0]], postFix=['LC', 'JW'])
        # referenceFileName = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['task'][0]])
        # transformMatrixFileName = self.runFile(stage = 'processed/mri/reg', base='TSE_to_T2', extension='.mat')
        # outputFileName = self.runFile(stage = 'processed/mri/masks/anat', base='LC', postFix=['JW'])
        # fO = FlirtOperator(inputObject, referenceFileName)
        # fO.configureApply(transformMatrixFileName=transformMatrixFileName, outputFileName=outputFileName, sinc=False)
        # fO.execute()
        #
        # # treshold:
        # mask = NiftiImage(outputFileName)
        # mask_data = mask.data
        # threshold = mask_data.ravel()[np.argsort(mask_data.ravel())[-min_nr_voxels]]
        # mask_data[mask_data < threshold] = 0
        # # mask_data[mask_data >= threshold] = 1
        # mask.data = mask_data
        # mask.save(outputFileName)
        #
        # # trilinear:
        # inputObject = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['TSE_anat'][0]], postFix=['LC', 'JW'])
        # referenceFileName = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['task'][0]])
        # transformMatrixFileName = self.runFile(stage = 'processed/mri/reg', base='TSE_to_T2', extension='.mat')
        # outputFileName = self.runFile(stage = 'processed/mri/masks/anat', base='LC', postFix=['JW_nn'])
        # fO = FlirtOperator(inputObject, referenceFileName)
        # fO.configureApply(transformMatrixFileName=transformMatrixFileName, outputFileName=outputFileName, sinc=False)
        # fO.execute()
        #
        # # treshold:
        # mask = NiftiImage(outputFileName)
        # mask_data = mask.data
        # threshold = mask_data.ravel()[np.argsort(mask_data.ravel())[-2]]
        # mask_data[mask_data < threshold] = 0
        # # mask_data[mask_data >= threshold] = 1
        # mask.data = mask_data
        # mask.save(outputFileName)
        #
        # # nearest neighbour:
        # inputObject = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['TSE_anat'][0]], postFix=['LC', 'JW'])
        # referenceFileName = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['task'][0]])
        # transformMatrixFileName = self.runFile(stage = 'processed/mri/reg', base='TSE_to_T2', extension='.mat')
        # outputFileName = self.runFile(stage = 'processed/mri/masks/anat', base='LC', postFix=['JW_nn'])
        # fO = FlirtOperator(inputObject, referenceFileName)
        # fO.configureApply(transformMatrixFileName=transformMatrixFileName, outputFileName=outputFileName, sinc=False, extra_args = ' -interp nearestneighbour')
        # fO.execute()
        #
        #
        #
        # # transform 4th ventricle:
        # ##########################
        #
        # inputObject = self.runFile(stage='processed/mri', run=self.runList[self.conditionDict['TSE_anat'][0]], postFix=['ventricle'])
        # mask = NiftiImage(inputObject)
        # mask_data = mask.data
        # mask_data[mask_data == 99] = 1
        # mask.data = mask_data
        # mask.save(inputObject)
        #
        # inputObject = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['TSE_anat'][0]], postFix=['ventricle'])
        # referenceFileName = self.runFile(stage = 'processed/mri', run = self.runList[self.conditionDict['task'][0]])
        # transformMatrixFileName = self.runFile(stage = 'processed/mri/reg', base='TSE_to_T2', extension='.mat')
        # outputFileName = self.runFile(stage = 'processed/mri/masks/anat', base='4th_ventricle')
        # fO = FlirtOperator(inputObject, referenceFileName)
        # fO.configureApply(transformMatrixFileName=transformMatrixFileName, outputFileName=outputFileName, sinc=False)
        # fO.execute()
        #
        # # treshold:
        # mask = NiftiImage(outputFileName)
        # mask_data = mask.data
        # threshold = 0.5
        # mask_data[mask_data < threshold] = 0
        # # mask_data[mask_data >= threshold] = 1
        # mask.data = mask_data
        # mask.save(outputFileName)
    

            
    def correct_FS_subcortical_masks(self):
        LC = NiftiImage(os.path.join(self.stageFolder(stage = 'processed/mri/masks/anat'), 'LC_standard_1.nii.gz'))
        
        # correct Brain Stem:
        brain_stem = NiftiImage(os.path.join(self.stageFolder(stage = 'processed/mri/masks/anat'), 'Brain_Stem.nii.gz'))
        data = np.array(np.array(brain_stem.data, dtype=bool) * -np.array(LC.data, dtype=bool), dtype=int)
        newImage = NiftiImage(data)
        newImage.header = brain_stem.header
        newImage.filename = os.path.join(self.stageFolder(stage = 'processed/mri/masks/anat'), 'Brain_Stem_corrected.nii.gz')
        newImage.save()
        
        # correct 4th ventricle:
        ventricle = NiftiImage(os.path.join(self.stageFolder(stage = 'processed/mri/masks/anat'), 'Fourth_Ventricle.nii.gz'))
        data = np.array(np.array(ventricle.data, dtype=bool) * -np.array(LC.data, dtype=bool) * -np.array(brain_stem.data, dtype=bool), dtype=int)
        newImage = NiftiImage(data)
        newImage.header = ventricle.header
        newImage.filename = os.path.join(self.stageFolder(stage = 'processed/mri/masks/anat'), 'Fourth_Ventricle_corrected.nii.gz')
        newImage.save()
        
    def check_rois(self, rois):
        
        session = [self.runList[i] for i in self.conditionDict['task']][0].session
        
        threshold = 0.1
        
        # brain mask:
        brain_mask = np.array(NiftiImage(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'masks', 'brain_mask_B0_{}.nii.gz'.format(session))).data, dtype=float)
        brain_mask[brain_mask<0.99] = 0
        brain_mask[brain_mask>=0.99] = 1
        brain_mask = np.array(brain_mask, dtype=bool)
        
        nr_voxels = np.zeros(len(rois))
        for i, roi in enumerate(rois):
            mask = np.array(NiftiImage(os.path.join(self.project.base_dir, self.subject.initials, 'files', 'masks', roi + '_{}.nii.gz'.format(session))).data, dtype=float)
            mask[mask<threshold] = 0
            mask[mask>=threshold] = 1
            mask = np.array(mask, dtype=bool)
            
            nr_voxels[i] = (brain_mask * mask).ravel().sum()
            
            
            # print (mask).ravel().sum() - (brain_mask * mask).ravel().sum()
            # if (mask).ravel().sum() - (brain_mask * mask).ravel().sum() > 15.0:
            #     wrong_rois[i] = True
            
        return nr_voxels 