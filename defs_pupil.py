#!/usr/bin/env python
# encoding: utf-8
"""
================================================
Created by Jan Willem de Gee on 2014-06-01.     
Copyright (c) 2009 jwdegee. All rights reserved.
================================================
"""

import os
import sys
import datetime
import pickle
import math
import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy.linalg as LA
import bottleneck as bn
import glob
from joblib import Parallel, delayed
import itertools
from itertools import chain
import logging
import logging.handlers
import logging.config

import copy

from IPython import embed as shell

sys.path.append(os.environ['ANALYSIS_HOME'])
from Tools.log import *
from Tools.Operators import ArrayOperator, EDFOperator, HDFEyeOperator, EyeSignalOperator
from Tools.Operators.EyeSignalOperator import detect_saccade_from_data
from Tools.Operators.CommandLineOperator import ExecCommandLine
from Tools.other_scripts.plotting_tools import *
from Tools.other_scripts.circularTools import *
from Tools.other_scripts import functions_jw as myfuncs
from Tools.other_scripts import functions_jw_GLM as GLM

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set(style='ticks', font='Arial', font_scale=1, rc={
    'axes.linewidth': 0.25, 
    'axes.labelsize': 7, 
    'axes.titlesize': 7, 
    'xtick.labelsize': 6, 
    'ytick.labelsize': 6, 
    'legend.fontsize': 6, 
    'xtick.major.width': 0.25, 
    'ytick.major.width': 0.25,
    'text.color': 'Black',
    'axes.labelcolor':'Black',
    'xtick.color':'Black',
    'ytick.color':'Black',} )
sns.plotting_context()

class pupilPreprocessSession(object):
    """pupilPreprocessing"""
    def __init__(self, subject, experiment_name, experiment_nr, version, project_directory, loggingLevel = logging.DEBUG, sample_rate_new = 50):
        self.subject = subject
        self.experiment_name = experiment_name
        self.experiment = experiment_nr
        self.version = version
        try:
            os.mkdir(os.path.join(project_directory, experiment_name))
            os.mkdir(os.path.join(project_directory, experiment_name, self.subject.initials))
        except OSError:
            pass
        self.project_directory = project_directory
        self.base_directory = os.path.join(self.project_directory, self.experiment_name, self.subject.initials)
        self.create_folder_hierarchy()
        self.hdf5_filename = os.path.join(self.base_directory, 'processed', self.subject.initials + '.hdf5')
        self.ho = HDFEyeOperator.HDFEyeOperator(self.hdf5_filename)
        self.velocity_profile_duration = self.signal_profile_duration = 100
        self.loggingLevel = loggingLevel
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(self.loggingLevel)
        addLoggingHandler(logging.handlers.TimedRotatingFileHandler(os.path.join(self.base_directory, 'log', 'sessionLogFile.log'), when='H', delay=2, backupCount=10), loggingLevel=self.loggingLevel)
        loggingLevelSetup()
        for handler in logging_handlers:
            self.logger.addHandler(handler)
        
        self.logger.info('starting analysis in ' + self.base_directory)
        self.sample_rate_new = int(sample_rate_new)
        self.downsample_rate = int(1000 / sample_rate_new)
        
    def create_folder_hierarchy(self):
        """createFolderHierarchy does... guess what."""
        this_dir = self.project_directory
        for d in [self.experiment_name, self.subject.initials]:
            try:
                this_dir = os.path.join(this_dir, d)
                os.mkdir(this_dir)
            except OSError:
                pass

        for p in ['raw',
         'processed',
         'figs',
         'log']:
            try:
                os.mkdir(os.path.join(self.base_directory, p))
            except OSError:
                pass
    
    def delete_hdf5(self):
        os.system('rm {}'.format(os.path.join(self.base_directory, 'processed', self.subject.initials + '.hdf5')))
    
    def import_raw_data(self, edf_files, aliases):
        """import_raw_data loops across edf_files and their respective aliases and copies and renames them into the raw directory."""
        for (edf_file, alias,) in zip(edf_files, aliases):
            self.logger.info('importing file ' + edf_file + ' as ' + alias)
            ExecCommandLine('cp "' + edf_file + '" "' + os.path.join(self.base_directory, 'raw', alias + '.edf"'))
    
    def import_all_data(self, aliases):
        """import_all_data loops across the aliases of the sessions and converts the respective edf files, adds them to the self.ho's hdf5 file. """
        for alias in aliases:
            self.ho.add_edf_file(os.path.join(self.base_directory, 'raw', alias + '.edf'))
            self.ho.edf_message_data_to_hdf(alias=alias)
            self.ho.edf_gaze_data_to_hdf(alias=alias)
    
    def compute_omission_indices(self):
        """
        Here we're going to determine which trials should be counted as omissions due to
        (i) fixation errors (in decision interval):
                ->gaze of 150px or more away from fixation
                ->10 percent of time gaze of 75px or more away from fixation.
        (ii) blinks (in window 0.5s before decision interval till 0.5s after decision interval)
        (iii) too long (>3000ms, >2500ms) or too short (<250ms) RT
        (iv) first two trials
        (v) indicated by subject (only in exp 1)
        """
        
        self.omission_indices_answer = (self.rt > 4000.0)
        # self.omission_indices_answer = (self.rt > 4000.0) * (np.array(self.parameters.answer == 0))
        
        self.omission_indices_sac = np.zeros(self.nr_trials, dtype=bool)
        self.omission_indices_blinks = np.zeros(self.nr_trials, dtype=bool)
        if self.artifact_rejection == 'strict':
            # based on sacs:
            middle_x = 0
            middle_y = 0
            cut_off = 75
            x_matrix = []
            y_matrix = []
            for t in range(self.nr_trials):
                try:
                    indices = (self.time > self.cue_times[t]) * (self.time < self.choice_times[t])
                except:
                    shell()
                x = self.gaze_x[indices]
                x = x - bn.nanmean(x)
                y = self.gaze_y[indices]
                y = y - bn.nanmean(y)
                if (x < -175).sum() > 0 or (x > 175).sum() > 0:
                    self.omission_indices_sac[t] = True
                if (y < -175).sum() > 0 or (y > 175).sum() > 0:
                    self.omission_indices_sac[t] = True
                if ((x > middle_x + cut_off).sum() + (x < middle_x - cut_off).sum()) / float(self.rt[t]) * 100 > 10:
                    self.omission_indices_sac[t] = True
                if ((y > middle_y + cut_off).sum() + (y < middle_y - cut_off).sum()) / float(self.rt[t]) * 100 > 10:
                    self.omission_indices_sac[t] = True
            # based on blinks:
            for t in range(self.nr_trials):
                if sum((self.blink_start_times > self.cue_times[t]) * (self.blink_end_times < self.choice_times[t])) > 0:
                    self.omission_indices_blinks[t] = True
        
        self.omission_indices_rt = np.zeros(self.nr_trials, dtype=bool)
        for t in range(self.nr_trials):
            if self.rt[t] < 250:
                self.omission_indices_rt[t] = True
        
        self.omission_indices_first = np.zeros(self.nr_trials, dtype=bool)
        self.omission_indices_first[0] = True
        if self.experiment == 1:
            self.omission_indices_subject = np.array(self.parameters['confidence'] == -1)
            self.omission_indices_subject = np.array(self.parameters['correct'] == -1)
        else:
            self.omission_indices_subject = np.zeros(self.nr_trials, dtype=bool)
        self.omission_indices = self.omission_indices_answer + self.omission_indices_sac + self.omission_indices_blinks + self.omission_indices_rt + self.omission_indices_first + self.omission_indices_subject
    
    def trial_params(self):
        blinks_nr = np.zeros(self.nr_trials)
        number_blinks = np.zeros(self.nr_trials)
        for t in range(self.nr_trials):
            # blinks_nr[t] = sum((self.blink_start_times > self.cue_times[t] - 5500) * (self.blink_start_times < self.cue_times[t] - 500))
            blinks_nr[t] = sum((self.blink_start_times > self.cue_times[t]) * (self.blink_start_times < self.choice_times[t]+1000))

        sacs_nr = np.zeros(self.nr_trials)
        sacs_dur = np.zeros(self.nr_trials)
        sacs_vel = np.zeros(self.nr_trials)
        for t in range(self.nr_trials):
            saccades_in_trial_indices = (self.saccade_start_times > self.cue_times[t] - 500) * (self.saccade_start_times < self.choice_times[t] + 1500)
            sacs_nr[t] = sum(saccades_in_trial_indices)
            sacs_dur[t] = sum(self.saccade_durs[saccades_in_trial_indices])
            if sacs_nr[t] != 0:
                sacs_vel[t] = max(self.saccade_peak_velocities[saccades_in_trial_indices])
        if self.version == 3:
            present = np.array(self.parameters['signal_present'] == 1)
        else:
            present = np.array(self.parameters['target_present_in_stimulus'] == 1)
        correct = np.array(self.parameters['correct'] == 1)
        yes = present * correct + -present * -correct
        hit = present * yes
        fa = -present * yes
        miss = present * -yes
        cr = -present * -yes
        
        run_nr = int(self.alias.split('_')[-2])
        session_nr = int(self.alias.split('_')[-1])
        
        self.parameters['omissions'] = self.omission_indices
        self.parameters['omissions_answer'] = self.omission_indices_answer
        self.parameters['omissions_sac'] = self.omission_indices_sac
        self.parameters['omissions_blinks'] = self.omission_indices_blinks
        self.parameters['omissions_rt'] = self.omission_indices_rt
        self.parameters['rt'] = self.rt
        self.parameters['iti'] = self.ITI
        self.parameters['yes'] = yes
        self.parameters['present'] = present
        self.parameters['hit'] = hit
        self.parameters['fa'] = fa
        self.parameters['miss'] = miss
        self.parameters['cr'] = cr
        self.parameters['blinks_nr'] = blinks_nr
        self.parameters['sacs_nr'] = sacs_nr
        self.parameters['sacs_dur'] = sacs_dur
        self.parameters['sacs_vel'] = sacs_vel
        self.parameters['run'] = run_nr
        self.parameters['session'] = session_nr
        self.ho.data_frame_to_hdf(self.alias, 'parameters2', self.parameters)
        
        print '{} total trials'.format(self.nr_trials)
        print '{} hits'.format(sum(hit))
        print '{} false alarms'.format(sum(fa))
        print '{} misses'.format(sum(miss))
        print '{} correct rejects'.format(sum(cr))
        print '{} omissions'.format(sum(self.omission_indices))
        (d_prime, criterion,) = myfuncs.SDT_measures(present, hit, fa)
        print "d' = {}\nc = {}".format(round(d_prime, 4), round(criterion, 4))
        print '{} mean RT (including omissions)'.format(round(np.mean(self.rt), 4))
        print ''
    
    def pupil_zscore(self):
        start = self.cue_times[(-self.omission_indices)] - 500
        end = self.choice_times[(-self.omission_indices)] + 1500
        include_indices = np.zeros(len(self.time), dtype=bool)
        for i in range(len(start)):
            include_indices[(self.time > start[i]) * (self.time < end[i])] = True

        mean = self.pupil_lp[include_indices].mean()
        std = self.pupil_lp[include_indices].std()
        self.pupil_lp_z = (self.pupil_lp - mean) / std
        mean = self.pupil_bp[include_indices].mean()
        std = self.pupil_bp[include_indices].std()
        self.pupil_bp_z = (self.pupil_bp - mean) / std
    
    def create_timelocked_arrays(self, pupil_data):
        
        len_response_a = 7500
        offset_a = 1000
        len_response_b = 7500 
        offset_b = 4000
        len_response_c = 5000 
        offset_c = 1000
        
        # shell()
        
        cue_locked_array = np.empty((self.nr_trials, (len_response_a / self.downsample_rate)+1)) * np.nan
        for i in range(self.nr_trials):
            indices = (self.time > self.cue_times[i] - offset_a) * (self.time < self.cue_times[i] + (len_response_a-offset_a))
            cue_locked_array[i,:pupil_data[indices].shape[0]] = pupil_data[indices]
        
        choice_locked_array = np.empty((self.nr_trials, (len_response_b / self.downsample_rate)+1)) * np.nan
        for i in range(self.nr_trials):
            indices = (self.time > self.choice_times[i] - offset_b) * (self.time < self.choice_times[i] + (len_response_b-offset_b))
            choice_locked_array[i,:pupil_data[indices].shape[0]] = pupil_data[indices]
        
        if self.experiment == 1:
            feedback_locked_array = np.empty((self.nr_trials, (len_response_c / self.downsample_rate)+1)) * np.nan
            for i in range(self.nr_trials):
                indices = (self.time > self.feedback_times[i] - offset_c) * (self.time < self.feedback_times[i] + (len_response_c-offset_c))
                feedback_locked_array[i,:pupil_data[indices].shape[0]] = pupil_data[indices]
        
        # to hdf5:
        self.ho.data_frame_to_hdf(self.alias, 'time_locked_cue', pd.DataFrame(cue_locked_array))
        self.ho.data_frame_to_hdf(self.alias, 'time_locked_choice', pd.DataFrame(choice_locked_array))
        if self.experiment == 1:
            self.ho.data_frame_to_hdf(self.alias, 'time_locked_feedback', pd.DataFrame(feedback_locked_array))
    
    def problem_trials(self):
        
        # problem trials:
        answers = np.array(self.parameters.answer)
        if self.version == 3:
            problem_trials = np.where(answers == -1)[0]
        else:
            problem_trials = np.where(answers == 0)[0]
        
        print
        print answers
        print
        
        # fix:
        for i in problem_trials:
            
            events = self.events[(self.events.EL_timestamp > self.trial_starts[i]) * (self.events.EL_timestamp < self.trial_ends[i])]
            response_times = np.array(events[((events.key == 275) + (events.key == 276)) * (events.up_down == 'Down')].EL_timestamp)
            
            if len(response_times) != 0:
                
                try:
                    ind = np.where(response_times>=self.choice_times[i])[0][0]
                    response_time = response_times[ind]
                    response_key = np.array(events[((events.key == 275) + (events.key == 276)) * (events.up_down == 'Down')].key)[ind]
                    
                    # choice time:
                    self.choice_times[i] = response_time
            
                    # answer & correct:
                    if self.version == 1:
                        if response_key == 275:
                            self.parameters.answer[i] = -1
                            if self.parameters.target_present_in_stimulus[i] == 0:
                                self.parameters.correct[i] = 1
                            else:
                                self.parameters.correct[i] = 0
                        if response_key == 276:
                            self.parameters.answer[i] = 1
                            if self.parameters.target_present_in_stimulus[i] == 0:
                                self.parameters.correct[i] = 0
                            else:
                                self.parameters.correct[i] = 1
                    if self.version == 2:
                        if response_key == 275:
                            self.parameters.answer[i] = 1
                            if self.parameters.target_present_in_stimulus[i] == 0:
                                self.parameters.correct[i] = 0
                            else:
                                self.parameters.correct[i] = 1
                        if response_key == 276:
                            self.parameters.answer[i] = -1
                            if self.parameters.target_present_in_stimulus[i] == 0:
                                self.parameters.correct[i] = 1
                            else:
                                self.parameters.correct[i] = 0
                except:
                    pass
            
            else:
                pass
            
    
    def create_pupil_BOLD_regressor(self):
        
        # load number of trs in nifti (unequal to above because we aborted runs manually, and not all trs are recorded by experiment script):
        if self.subject.initials == 'AV':
            trs = [259, 264, 254, 257, 251, 249, 256, 247, 249, 250, 246, 247,]
        if self.subject.initials == 'BL':
            trs = [273, 267, 255, 259, 254, 266, 252, 273, 261, 259, 256, 250,] 
        if self.subject.initials == 'DL':
            trs = [214, 299, 183, 259, 260, 266, 259, 253, 261, 251, 258, 249, 258, 256, 258, 259,]
        if self.subject.initials == 'EP':
            trs = [252, 255, 258, 264, 255, 257, 255, 258, 257, 254, 247,]
        if self.subject.initials == 'JG':
            trs = [252, 269, 274, 253, 257, 260, 256, 250, 254, 260, 260, 251,]
        if self.subject.initials == 'LH':
            trs = [264, 300, 300, 294, 251, 260, 263, 253, 256, 260,]
        if self.subject.initials == 'LP':
            trs = [249, 252, 249, 259, 260, 261, 262, 256, 256, 253, 255, 257,]
        if self.subject.initials == 'MG':
            trs = [265, 265, 252, 255, 304, 256, 251, 250, 256, 258, 253, 248,]
        if self.subject.initials == 'NM':
            trs = [248, 254, 253, 255, 267, 274, 289, 258, 253]
        if self.subject.initials == 'NS':
            trs = [254, 254, 259, 253, 255, 251, 254, 261, 253, 254, 263, 253]
        if self.subject.initials == 'OC':
            trs = [266, 255, 259, 253, 251, 262, 263, 250, 252, 247, 250, 246]
        if self.subject.initials == 'TK':
            trs = [246, 252, 258, 263, 258, 255, 259, 248, 248, 253, 247, 247]
        if self.subject.initials == 'TN':
            trs = [256, 267, 246, 248, 249, 252, 245, 247, 247, 247, 246, 246]
        
        nr_trs_nifti = trs[int(self.alias.split('_')[-1]) - 1]
        
        # load trigger timings:
        trigger_timings = np.array(self.events['EL_timestamp'][(self.events.up_down == 'Down') & (self.events.key == 116)])
        tr = min(np.diff(trigger_timings))
        print 'tr = {} ms'.format(tr)
        print 'len trigger exp = {}'.format(len(trigger_timings))
        print 'len trigger nifti = {}'.format(nr_trs_nifti)
        print
        
        if nr_trs_nifti - len(trigger_timings) == -1: # this is the case for the last run of subjects EP... I don't understand this
            trigger_timings = trigger_timings[:-1]
        
        # create pupil timeseries in fMRI tr resolution:
        pupil = np.zeros((3, nr_trs_nifti))
        pupil[:,:] = np.NaN
        try:
            pupil[0,:len(trigger_timings)] = np.array([np.mean(self.pupil_lp_z[(self.time>t)*(self.time<(t+tr))]) for t in trigger_timings])
            pupil[1,:len(trigger_timings)] = np.array([np.mean(self.pupil_bp_z[(self.time>t)*(self.time<(t+tr))]) for t in trigger_timings])
            pupil[2,:len(trigger_timings)] = np.array([np.mean(np.array(self.pupil_data[(self.eye + '_pupil_lp_diff')])[(self.time>t)*(self.time<(t+tr))]) for t in trigger_timings])
        except ValueError:
            shell()
        
        # save:
        self.ho.data_frame_to_hdf(self.alias, 'pupil_BOLD_regressors', pd.DataFrame(pupil))
        
    def process_runs(self, alias, artifact_rejection='strict', create_pupil_BOLD_regressor=False):
        print 'subject {}; {}'.format(self.subject.initials, alias)
        print '##############################'
        
        self.artifact_rejection = artifact_rejection
        
        # load data:
        self.alias = alias
        self.events = self.ho.read_session_data(alias, 'events')
        self.parameters = self.ho.read_session_data(alias, 'parameters')
        self.nr_trials = len(self.parameters['trial_nr'])
        self.trial_times = self.ho.read_session_data(alias, 'trials')
        self.session_start = self.trial_times['trial_start_EL_timestamp'][0]
        self.trial_starts = np.array(self.trial_times['trial_start_EL_timestamp'])
        self.trial_ends = np.array(self.trial_times['trial_end_EL_timestamp'])
        self.phase_times = self.ho.read_session_data(alias, 'trial_phases')
        self.baseline_times = np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 1)])
        self.cue_times = np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 2)])
        self.choice_times = np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 3)])
        self.problem_trials()
        self.rt = self.choice_times - self.cue_times
        self.ITI = np.zeros(len(self.rt))
        self.ITI[:] = np.NaN
        self.ITI[1:] = self.baseline_times[1:] - self.choice_times[:-1] 
        if self.experiment == 1:
            self.confidence_times = np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 5)])
            self.feedback_times = np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 6)])
        self.blink_data = self.ho.read_session_data(alias, 'blinks_from_message_file')
        self.saccade_data = self.ho.read_session_data(alias, 'saccades_from_message_file')
        self.blink_start_times = np.array(self.blink_data['start_timestamp'])
        self.blink_end_times = np.array(self.blink_data['end_timestamp'])
        self.saccade_start_times = np.array(self.saccade_data['start_timestamp'])
        self.saccade_end_times = np.array(self.saccade_data['end_timestamp'])
        self.saccade_durs = np.array(self.saccade_data['duration'])
        self.saccade_peak_velocities = np.array(self.saccade_data['peak_velocity'])
        self.eye = self.ho.eye_during_period((np.array(self.trial_times['trial_start_EL_timestamp'])[0], np.array(self.trial_times['trial_end_EL_timestamp'])[-1]), self.alias)
        self.pupil_data = self.ho.data_from_time_period((np.array(self.trial_times['trial_start_EL_timestamp'])[0], np.array(self.trial_times['trial_end_EL_timestamp'])[-1]), self.alias)
        
        self.time = np.array(self.pupil_data['time'])
        self.time = self.time[::self.downsample_rate]
        
        self.pupil = np.array(self.pupil_data[(self.eye + '_pupil_bp_clean_psc')])
        self.pupil = sp.signal.decimate(self.pupil, self.downsample_rate, 1)
        
        # self.pupil_raw = np.array(self.pupil_data[(self.eye + '_pupil')])
        # self.pupil_lp = np.array(self.pupil_data[(self.eye + '_pupil_lp')])
        # self.pupil_bp = np.array(self.pupil_data[(self.eye + '_pupil_bp')])
        # self.pupil_lp_psc = np.array(self.pupil_data[(self.eye + '_pupil_lp_psc')])
        # self.pupil_bp_psc = np.array(self.pupil_data[(self.eye + '_pupil_bp_psc')])
        
        # shell()
        
        self.gaze_x = np.array(self.pupil_data[(self.eye + '_gaze_x')])
        self.gaze_y = np.array(self.pupil_data[(self.eye + '_gaze_y')])
        
        self.compute_omission_indices()
        self.trial_params()
        # self.pupil_zscore()
        # self.create_timelocked_arrays(pupil_data=self.pupil,)
        
        if create_pupil_BOLD_regressor:
            self.create_pupil_BOLD_regressor()
        
    
    def process_across_runs(self, aliases, create_pupil_BOLD_regressor=False):
        
        # load data:
        parameters = []
        pupil_BOLD_regressors = []
        bp_lp = []
        bp_bp = []
        bp_feed_lp = []
        bp_feed_bp = []
        tpr_lp = []
        tpr_bp = []
        tpr_feed_lp = []
        tpr_feed_bp = []
        for alias in aliases:
            parameters.append(self.ho.read_session_data(alias, 'parameters2'))
            if create_pupil_BOLD_regressor:
                pupil_BOLD_regressors.append(np.array(self.ho.read_session_data(alias, 'pupil_BOLD_regressors')))
            trial_times = self.ho.read_session_data(alias, 'trials')
            eye = self.ho.eye_during_period((np.array(trial_times['trial_start_EL_timestamp'])[0], np.array(trial_times['trial_end_EL_timestamp'])[-1]), alias)
            pupil_data = self.ho.data_from_time_period((np.array(trial_times['trial_start_EL_timestamp'])[0], np.array(trial_times['trial_end_EL_timestamp'])[-1]), alias)
            session_start = trial_times['trial_start_EL_timestamp'][0]
            pupil_bp = np.array(pupil_data[(eye + '_pupil_bp_clean_psc')])
            pupil_lp = np.array(pupil_data[(eye + '_pupil_lp_clean_psc')])
            time = np.array(pupil_data['time']) - session_start
            phase_times = self.ho.read_session_data(alias, 'trial_phases')
            cue_times = np.array(phase_times['trial_phase_EL_timestamp'][(phase_times['trial_phase_index'] == 2)]) - session_start
            choice_times = np.array(phase_times['trial_phase_EL_timestamp'][(phase_times['trial_phase_index'] == 3)]) - session_start
            if self.experiment == 1:
                confidence_times = np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 5)]) - session_start
                feedback_times = np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 6)]) - session_start
            
            # baseline pupil measures:
            bp_lp.append( np.array([np.mean(pupil_lp[(time>i-500)*(time<i)]) for i in cue_times]) )
            bp_bp.append( np.array([np.mean(pupil_bp[(time>i-500)*(time<i)]) for i in cue_times]) )
            if self.experiment == 1:
                bp_feed_lp.append( np.array([np.mean(pupil_lp[(time>i-500)*(time<i)]) for i in feedback_times]) )
                bp_feed_bp.append( np.array([np.mean(pupil_bp[(time>i-500)*(time<i)]) for i in feedback_times]) )
        
            # phasic pupil responses 
            tpr_lp.append( np.array([np.mean(pupil_lp[(time>i-1000)*(time<i+1500)]) for i in choice_times]) - bp_lp[-1] )
            tpr_bp.append( np.array([np.mean(pupil_bp[(time>i-1000)*(time<i+1500)]) for i in choice_times]) - bp_bp[-1] )
            if self.experiment == 1:
                tpr_feed_lp.append( np.array([np.mean(pupil_lp[(time>i+500)*(time<i+1500)]) for i in feedback_times]) - bp_feed_lp[-1]  )
                tpr_feed_bp.append( np.array([np.mean(pupil_bp[(time>i+500)*(time<i+1500)]) for i in feedback_times]) - bp_feed_bp[-1]  )
            
        # join over runs:
        parameters_joined = pd.concat(parameters)
        if create_pupil_BOLD_regressor:
            pupil_BOLD_regressors_joined = np.hstack(pupil_BOLD_regressors)
            np.save(os.path.join(self.project_directory, 'data', self.subject.initials, 'pupil_BOLD_regressors'), pupil_BOLD_regressors_joined)
        bp_lp = np.concatenate(bp_lp)
        bp_bp = np.concatenate(bp_bp)
        tpr_lp = np.concatenate(tpr_lp)
        tpr_bp = np.concatenate(tpr_bp)
        if self.experiment == 1:
            bp_feed_lp = np.concatenate(bp_feed_lp)
            bp_feed_bp = np.concatenate(bp_feed_bp)
            tpr_feed_lp = np.concatenate(tpr_feed_lp)
            tpr_feed_bp = np.concatenate(tpr_feed_bp)
        
        # trial params
        target_joined = parameters_joined['present'][(-parameters_joined['omissions'])]
        hit_joined = parameters_joined['hit'][(-parameters_joined['omissions'])]
        fa_joined = parameters_joined['fa'][(-parameters_joined['omissions'])]
        (d, c,) = myfuncs.SDT_measures(target_joined, hit_joined, fa_joined)
        target = [ param['present'][(-param['omissions'])] for param in parameters ]
        hit = [ param['hit'][(-param['omissions'])] for param in parameters ]
        fa = [ param['fa'][(-param['omissions'])] for param in parameters ]
        d_run = []
        c_run = []
        for i in range(len(aliases)):
            d_run.append(myfuncs.SDT_measures(target[i], hit[i], fa[i])[0])
            c_run.append(myfuncs.SDT_measures(target[i], hit[i], fa[i])[1])
        
        # add to dataframe and save to hdf5:
        parameters_joined['pupil_b'] = bp_bp
        parameters_joined['pupil_d'] = tpr_bp
        parameters_joined['pupil_b_lp'] = bp_lp
        parameters_joined['pupil_d_lp'] = tpr_lp
        parameters_joined['d_prime'] = d
        parameters_joined['criterion'] = c
        for i in range(len(aliases)):
            parameters_joined['d_prime_' + str(i)] = d_run[i]
            parameters_joined['criterion_' + str(i)] = c_run[i]
        if self.experiment == 1:
            parameters_joined['pupil_b_feed'] = bp_feed_bp
            parameters_joined['pupil_d_feed'] = tpr_feed_bp
            parameters_joined['pupil_b_feed_lp'] = bp_feed_lp
            parameters_joined['pupil_d_feed_lp'] = tpr_feed_lp
        parameters_joined['subject'] = self.subject.initials
        self.ho.data_frame_to_hdf('', 'parameters_joined', parameters_joined)
    
class pupilAnalyses(object):
    """pupilAnalyses"""
    def __init__(self, subject, experiment_name, experiment_nr, project_directory, aliases, sample_rate_new=50):
        self.subject = subject
        self.experiment_name = experiment_name
        self.experiment = experiment_nr
        self.project_directory = project_directory
        self.base_directory = os.path.join(self.project_directory, self.experiment_name, self.subject.initials)
        self.hdf5_filename = os.path.join(self.base_directory, 'processed', self.subject.initials + '.hdf5')
        self.ho = HDFEyeOperator.HDFEyeOperator(self.hdf5_filename)
        self.sample_rate_new = int(sample_rate_new)
        self.downsample_rate = int(1000 / sample_rate_new)
        
        self.parameters_joined = self.ho.read_session_data('', 'parameters_joined')
        self.omissions = np.array(self.parameters_joined['omissions'])
        
        # regress out RT per session:
        for s in np.unique(self.parameters_joined.session):
            ind = (self.parameters_joined.session == s)
            rt = np.array(self.parameters_joined['rt'][ind]) / 100.0
            pupil_d = np.array(self.parameters_joined['pupil_d'][ind])
            pupil_d = myfuncs.lin_regress_resid(pupil_d, [rt]) + pupil_d.mean()
            self.parameters_joined['pupil_d'][ind] = pupil_d
        
        self.parameters_joined = self.parameters_joined
        
        self.rt = self.parameters_joined['rt']
        self.iti = self.parameters_joined['iti']
        
        self.hit = np.array(self.parameters_joined['hit']) 
        self.fa = np.array(self.parameters_joined['fa']) 
        self.miss = np.array(self.parameters_joined['miss']) 
        self.cr = np.array(self.parameters_joined['cr'])
        
        self.yes = np.array(self.parameters_joined['yes']) 
        self.correct = np.array(self.parameters_joined['correct'])
        self.present = np.array(self.parameters_joined['present'])
        
        self.pupil_b = np.array(self.parameters_joined['pupil_b'])
        self.pupil_d = np.array(self.parameters_joined['pupil_d'])
        
        self.criterion = np.array(self.parameters_joined['criterion'])[0]
        
        if self.experiment == 1:
            self.pupil_d_feed = np.array(self.parameters_joined['ppr_peak_feed'])
            self.pupil_b_feed = np.array(self.parameters_joined['bpd_feed'])
        
        # pupil split:
        d = self.parameters_joined
        p_h = []
        p_l = []
        for s in np.array(np.unique(d['session']), dtype=int):
            pupil_b = np.array(d['pupil_b'])[np.array(d.session) == s]
            pupil_d = np.array(d['pupil_d'])[np.array(d.session) == s]
            pupil = pupil_d
            p_l.append( pupil <= np.percentile(pupil, 40) )
            p_h.append( pupil >= np.percentile(pupil, 60) )
        self.pupil_l_ind = np.concatenate(p_l)
        self.pupil_h_ind = np.concatenate(p_h)
        self.pupil_rest_ind = -(self.pupil_h_ind + self.pupil_l_ind)
        
    def trial_wise_pupil(self):
        
        nr_runs = (pd.Series(np.array(self.parameters_joined.trial_nr))==0).sum()
        start_run = np.where(pd.Series(np.array(self.parameters_joined.trial_nr))==0)[0]
        subject = np.repeat(self.subject.initials,len(np.array(self.parameters_joined.trial_nr)))
        
        if self.subject.initials in ['DE', 'DL', 'JG', 'LH', 'LP', 'NS', 'TK']:
            self.right = self.yes
        else:
            self.right = -self.yes
        
        # data:
        d = {
        'subject' : pd.Series(subject),
        'trial_nr' : pd.Series(np.array(self.parameters_joined.trial_nr, dtype=int)),
        'run' : pd.Series(np.array(self.parameters_joined.run, dtype=int)),
        'session' : pd.Series(np.array(self.parameters_joined.session, dtype=int)),
        'signal_orientation' : pd.Series(np.array(self.parameters_joined.signal_orientation, dtype=int)),
        'omissions' : pd.Series(np.array(self.omissions)),
        'pupil_b' : pd.Series(np.array(self.pupil_b)),
        'pupil_d' : pd.Series(np.array(self.pupil_d)),
        'yes' : pd.Series(np.array(self.yes)),
        'right' : pd.Series(np.array(self.right)),
        'correct' : pd.Series(np.array(self.correct)),
        'present' : pd.Series(np.array(self.present)),
        'hit' : pd.Series(np.array(self.hit)),
        'fa' : pd.Series(np.array(self.fa)),
        'miss' : pd.Series(np.array(self.miss)),
        'cr' : pd.Series(np.array(self.cr)),
        'rt' : pd.Series(np.array(self.rt)/1000.0),
        'iti' : pd.Series(np.array(self.iti)/1000.0),
        'nr_blinks' : pd.Series(np.array(self.parameters_joined.blinks_nr, dtype=int)),
        }
        data = pd.DataFrame(d)
        data.to_csv(os.path.join(self.project_directory, 'data', self.subject.initials, 'pupil_data.csv'))
        
    def timelocked_plots(self):
        
        # baseline:
        for i in range(len(self.pupil_b)):
            self.cue_locked_array_joined[i,:] = self.cue_locked_array_joined[i,:] - self.pupil_b[i] 
            self.choice_locked_array_joined[i,:] = self.choice_locked_array_joined[i,:] - self.pupil_b[i]
        
        cue_timings = [-500, 4000]
        choice_timings = [-2500, 2000]
        
        cue_data = self.cue_locked_array_joined[:,(cue_timings[0]+1000)/self.downsample_rate:(cue_timings[1]+1000)/self.downsample_rate]
        choice_data = self.choice_locked_array_joined[:,(choice_timings[0]+4000)/self.downsample_rate:(choice_timings[1]+4000)/self.downsample_rate]
        
        
        # shell()
        
        if self.experiment == 1:
            
            # baseline:
            feed_timings = [-499, 2000]
            feed_data = self.feedback_locked_array_joined[:,feed_timings[0]+1000:feed_timings[1]+1000]
            for i in range(len(self.pupil_b_feed)):
                feed_data[i,:] = feed_data[i,:] - self.pupil_b_feed[i]
            
            # indices:
            if self.subject.initials == 'dh':
                conf1 = np.array(self.parameters_joined['confidence'] == 1)
            else:
                conf1 = np.array(self.parameters_joined['confidence'] == 0)
            conf2 = np.array(self.parameters_joined['confidence'] == 1)
            conf3 = np.array(self.parameters_joined['confidence'] == 2)
            conf4 = np.array(self.parameters_joined['confidence'] == 3)
            correct_conf1 = conf1 * (self.hit + self.cr)
            correct_conf2 = conf2 * (self.hit + self.cr)
            correct_conf3 = conf3 * (self.hit + self.cr)
            correct_conf4 = conf4 * (self.hit + self.cr)
            error_conf1 = conf1 * (self.fa + self.miss)
            error_conf2 = conf2 * (self.fa + self.miss)
            error_conf3 = conf3 * (self.fa + self.miss)
            error_conf4 = conf4 * (self.fa + self.miss)
            yes_conf1 = conf1 * (self.hit + self.fa)
            yes_conf2 = conf2 * (self.hit + self.fa)
            yes_conf3 = conf3 * (self.hit + self.fa)
            yes_conf4 = conf4 * (self.hit + self.fa)
            no_conf1 = conf1 * (self.cr + self.miss)
            no_conf2 = conf2 * (self.cr + self.miss)
            no_conf3 = conf3 * (self.cr + self.miss)
            no_conf4 = conf4 * (self.cr + self.miss)
        
        for aaaaaaaaa in range(4):
            
            if self.experiment == 1:
                if aaaaaaaaa == 0:
            
                    condition = [self.hit, self.fa, self.miss, self.cr]
                    colors = ['r', 'r', 'b', 'b']
                    alphas = [1,0.5,0.5,1]
                    labels = ['H', 'FA', 'M', 'CR']
                    filename = ''
            
                if aaaaaaaaa == 1:
            
                    condition = [conf1, conf2, conf3, conf4]
                    colors = ['b', 'b', 'r', 'r']
                    alphas = [1,0.5,0.5,1]
                    labels = ['--', '-', '+', '++']
                    filename = 'confidence_'
            
                if aaaaaaaaa == 2:
            
                    condition = [correct_conf1, correct_conf2, correct_conf3, correct_conf4, error_conf1, error_conf2, error_conf3, error_conf4]
                    colors = ['g', 'g', 'g', 'g', 'r', 'r', 'r', 'r']
                    alphas = [0.25,0.5,0.75,1,0.25,0.5,0.75,1]
                    labels = ['correct --', 'correct -', 'correct +', 'correct ++', 'error --', 'error -', 'error +', 'error ++']
                    filename = 'confidence_correct_'
            
                if aaaaaaaaa == 3:
            
                    condition = [yes_conf1, yes_conf2, yes_conf3, yes_conf4, no_conf1, no_conf2, no_conf3, no_conf4]
                    colors = ['r', 'r', 'r', 'r', 'b', 'b', 'b', 'b']
                    alphas = [0.25,0.5,0.75,1,0.25,0.5,0.75,1]
                    labels = ['yes --', 'yes -', 'yes +', 'yes ++', 'no --', 'no -', 'no +', 'no ++']
                    filename = 'confidence_yes_'
            
            if self.experiment == 2:
                condition = [self.hit, self.fa, self.miss, self.cr]
                colors = ['r', 'r', 'b', 'b']
                alphas = [1,0.5,0.5,1]
                labels = ['H', 'FA', 'M', 'CR']
                filename = ''
            
            print filename
            
            # ----------------------
            # do some plotting:    -
            # ----------------------
    
            # create downsampled means and sems:
            cue_means = []
            cue_sems = []
            choice_means = []
            choice_sems = []
            for i in range(len(condition)):
                cue_means.append(bn.nanmean(cue_data[condition[i]], axis=0))
                cue_sems.append(bn.nanstd(cue_data[condition[i]], axis=0) / sp.sqrt(condition[i].sum()))
                choice_means.append(bn.nanmean(choice_data[condition[i]], axis=0))
                choice_sems.append(bn.nanstd(choice_data[condition[i]], axis=0) / sp.sqrt(condition[i].sum()))
    
            # stuff for ylim:
            max_y_cue = max(np.concatenate(np.vstack(cue_means) + np.vstack(cue_sems)))
            min_y_cue = min(np.concatenate(np.vstack(cue_means) - np.vstack(cue_sems)))
            diff_cue = max_y_cue - min_y_cue
            max_y_choice = max(np.concatenate(np.vstack(choice_means) + np.vstack(choice_sems)))
            min_y_choice = min(np.concatenate(np.vstack(choice_means) - np.vstack(choice_sems)))
            diff_choice = max_y_choice - min_y_choice
            max_y = max((max_y_cue, max_y_choice))
            min_y = max((min_y_cue, min_y_choice))
            diff = abs(max_y - min_y)
    
            # cue locked plot:
            fig = plt.figure(figsize=(4, 3))
            a = plt.subplot(111)
            x = np.linspace(cue_timings[0], cue_timings[1], len(cue_means[0]))
            for i in range(len(condition)):
                a.plot(x, cue_means[i], linewidth=2, color=colors[i], alpha=alphas[i], label=labels[i] + ' ' + str(condition[i].sum()) + ' trials')
                a.fill_between(x, cue_means[i] + cue_sems[i], cue_means[i] - cue_sems[i], color=colors[i], alpha=0.1)
            # a.set_xlim((-500, 4000))
            leg = plt.legend(loc=2, fancybox=True)
            leg.get_frame().set_alpha(0.9)
            if leg:
                for t in leg.get_texts():
                    t.set_fontsize(7)

                for l in leg.get_lines():
                    l.set_linewidth(2)
            a.axes.tick_params(axis='both', which='major', labelsize=8)
            a.set_xticks((0, 1000, 2000, 3000, 4000))
            a.set_xticklabels((0, 1, 2, 3, 4))
            if diff < 0.5:
                a.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
            elif diff < 1:
                a.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.25))
            else:
                a.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
            a.set_ylim(ymin=min_y - diff / 20.0, ymax=max_y + diff / 20.0)
            a.set_title('{}; c = {}'.format(self.subject.initials, round(self.criterion, 3)), size=12)
            a.set_ylabel('Pupil diameter (s.d.)', size=10)
            a.set_xlabel('Time from cue (s)', size=10)
            a.vlines(np.mean(self.rt[(condition[0] + condition[1])]), plt.axis()[2], plt.axis()[3], color='r', linestyle='--', alpha=0.5)
            a.vlines(np.mean(self.rt[(condition[2] + condition[3])]), plt.axis()[2], plt.axis()[3], color='b', linestyle='--', alpha=0.5)
            a.vlines(0, plt.axis()[2], plt.axis()[3], linewidth=1)
            plt.tight_layout()
            fig.savefig(os.path.join(self.base_directory, 'figs', 'pupil_response_cue_locked_' + filename + self.subject.initials + '.pdf'))
    
            # choice locked plot:
            fig = plt.figure(figsize=(4, 3))
            a = plt.subplot(111)
            x = np.linspace(choice_timings[0], choice_timings[1], len(choice_means[0]))
            for i in range(len(condition)):
                a.plot(x, choice_means[i], linewidth=2, color=colors[i], alpha=alphas[i], label=labels[i] + ' ' + str(condition[i].sum()) + ' trials')
                a.fill_between(x, choice_means[i] + choice_sems[i], choice_means[i] - choice_sems[i], color=colors[i], alpha=0.1)
            a.set_xlim((-2500, 2000))
            leg = plt.legend(loc=2, fancybox=True)
            leg.get_frame().set_alpha(0.9)
            if leg:
                for t in leg.get_texts():
                    t.set_fontsize(7)
                for l in leg.get_lines():
                    l.set_linewidth(2)
            a.axes.tick_params(axis='both', which='major', labelsize=8)
            a.set_xticks((-2000, -1000, 0, 1000, 2000))
            a.set_xticklabels((-2, -1, 0, 1, 2))
            if diff < 0.5:
                a.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
            elif diff < 1:
                a.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.25))
            else:
                a.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
            a.set_ylim(ymin=min_y - diff / 20.0, ymax=max_y + diff / 20.0)
            a.set_title('{}; c = {}'.format(self.subject.initials, round(self.criterion, 3)), size=12)
            a.set_ylabel('Pupil diameter (s.d.)', size=10)
            a.set_xlabel('Time from choice (s)', size=10)
            a.vlines(0 - np.mean(self.rt[(condition[0] + condition[1])]), plt.axis()[2], plt.axis()[3], color='r', linestyle='--', alpha=0.5)
            a.vlines(0 - np.mean(self.rt[(condition[2] + condition[3])]), plt.axis()[2], plt.axis()[3], color='b', linestyle='--', alpha=0.5)
            a.vlines(0, plt.axis()[2], plt.axis()[3], linewidth=1)
            plt.tight_layout()
            fig.savefig(os.path.join(self.base_directory, 'figs', 'pupil_response_choice_locked_' + filename + self.subject.initials + '.pdf'))
    
            if self.experiment == 1:
        
                # create downsampled means and sems:
                feed_means = []
                feed_sems = []
                for i in range(len(condition)):
                    feed_means.append(sp.signal.decimate(bn.nanmean(feed_data[condition[i]], axis=0), self.downsample_rate, 1))
                    feed_sems.append(sp.signal.decimate(bn.nanstd(feed_data[condition[i]], axis=0), self.downsample_rate, 1) / sp.sqrt(condition[i].sum()))
        
                # stuff for ylim:
                max_y_feed = max(np.concatenate( np.vstack(feed_means)+np.vstack(feed_sems) ))
                min_y_feed = min(np.concatenate( np.vstack(feed_means)-np.vstack(feed_sems) ))
                diff_feed = max_y_feed - min_y_feed
        
                # feedback locked plot:
                fig = plt.figure(figsize=(4, 3))
                a = plt.subplot(111)
                x = np.linspace(feed_timings[0], feed_timings[1], len(feed_means[0]))
                for i in range(len(condition)):
                    a.plot(x, feed_means[i], linewidth=2, color=colors[i], alpha=alphas[i], label=labels[i] + ' ' + str(condition[i].sum()) + ' trials')
                    a.fill_between(x, feed_means[i] + feed_sems[i], feed_means[i] - feed_sems[i], color=colors[i], alpha=0.1)
                a.set_xlim((-500, 2000))
                leg = plt.legend(loc=2, fancybox=True)
                leg.get_frame().set_alpha(0.9)
                if leg:
                    for t in leg.get_texts():
                        t.set_fontsize(7)
                    for l in leg.get_lines():
                        l.set_linewidth(2)
                a.axes.tick_params(axis='both', which='major', labelsize=8)
                a.set_xticks([-500,-0,500,1000,1500,2000])
                a.set_xticklabels([-.5,0,.5,1,1.5,2.0])
                a.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.25))
                a.set_ylim(ymin=min_y_feed-(diff_feed/20.0), ymax=max_y_feed+(diff_feed/20.0))
                a.set_title('{}; c = {}'.format(self.subject.initials, round(self.criterion, 3)), size=12)
                a.set_ylabel('Pupil diameter (s.d.)', size=10)
                a.set_xlabel('Time from feedback (s)', size=10)
                a.vlines(0, plt.axis()[2], plt.axis()[3], linewidth=1)
                plt.tight_layout()
                fig.savefig(os.path.join(self.base_directory, 'figs', 'pupil_response_feedback_locked_' + filename + self.subject.initials + '.pdf'))
                
                if aaaaaaaaa == 2:
                    shell()
                    
                    means = np.vstack(feed_means)
                    np.savetxt(os.path.join(self.base_directory, 'figs', 'means.csv'), means, delimiter=",")
                    sems = np.vstack(feed_sems)
                    np.savetxt(os.path.join(self.base_directory, 'figs', 'sems.csv'), sems, delimiter=",")
                
                
    def behavior_confidence(self):
        
        conf1 = np.array(self.parameters_joined['confidence'] == 0)
        conf2 = np.array(self.parameters_joined['confidence'] == 1)
        conf3 = np.array(self.parameters_joined['confidence'] == 2)
        conf4 = np.array(self.parameters_joined['confidence'] == 3)
        
        # conditions = [conf1+conf2, conf3+conf4]
        conditions = [conf1, conf2, conf3, conf4]
        
        d_prime = []
        criterion = []
        for cond in conditions:
            d, c = myfuncs.SDT_measures((self.hit+self.miss)[cond], self.hit[cond], self.fa[cond],)
            d_prime.append(d)
            criterion.append(c)
        
        my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}

        N = 4
        ind = np.linspace(0,2,N)  # the x locations for the groups
        bar_width = 0.6   # the width of the bars
        spacing = [0, 0, 0, 0]
        
        # FIGURE 1
        MEANS = np.array(d_prime)
        MEANS2 = np.array(criterion)
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot(111)
        ax.plot(ind, MEANS, color = 'g', label="d'")
        ax2 = ax.twinx()
        ax2.plot(ind, MEANS2, color = 'k', label='criterion')
        ax.set_xticklabels( ('--','-','+','++') )
        ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
        ax.tick_params(axis='x', which='major', labelsize=10)
        ax.tick_params(axis='y', which='major', labelsize=10)
        leg = plt.legend(loc=2, fancybox=True)
        leg.get_frame().set_alpha(0.9)
        if leg:
            for t in leg.get_texts():
                t.set_fontsize(7)
            for l in leg.get_lines():
                l.set_linewidth(2)
        ax.set_ylabel("d'")
        ax2.set_ylabel('criterion')
        ax.set_xlabel('confidence')
        ax.set_ylim(ymin=0, ymax=2)
        ax2.set_ylim(ymin=-0.7, ymax=0.7)
        plt.gca().spines["bottom"].set_linewidth(.5)
        plt.gca().spines["left"].set_linewidth(.5)
        plt.title('subject {}, c = {}'.format((self.subject.initials), round(np.array(self.parameters_joined['criterion'])[0],3)))
        plt.tight_layout()
        fig.savefig(os.path.join(self.base_directory, 'figs', "behavior_confidence_" + self.subject.initials + '.pdf'))
    
    def pupil_bars(self):
        
        conf1 = np.array(self.parameters_joined['confidence'] == 0)
        conf2 = np.array(self.parameters_joined['confidence'] == 1)
        conf3 = np.array(self.parameters_joined['confidence'] == 2)
        conf4 = np.array(self.parameters_joined['confidence'] == 3)
        
        conditions = [conf1, conf2, conf3, conf4]
        
        my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}

        N = 4
        ind = np.linspace(0,2,N)  # the x locations for the groups
        bar_width = 0.6   # the width of the bars
        spacing = [0, 0, 0, 0]
        
        # FIGURE 1
        MEANS_yes = np.array([np.mean(self.pupil_d[(self.hit+self.fa)*cond]) for cond in conditions])
        SEMS_yes = np.array([sp.stats.sem(self.pupil_d[(self.hit+self.fa)*cond]) for cond in conditions])
        MEANS_no = np.array([np.mean(self.pupil_d[(self.miss+self.cr)*cond]) for cond in conditions])
        SEMS_no = np.array([sp.stats.sem(self.pupil_d[(self.miss+self.cr)*cond]) for cond in conditions])
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot(111)
        ax.errorbar(ind, MEANS_yes, yerr=SEMS_yes, color = 'r', capsize = 0)
        ax.errorbar(ind, MEANS_no, yerr=SEMS_no, color = 'b', capsize = 0)
        simpleaxis(ax)
        spine_shift(ax)
        ax.set_xticklabels( ('--','-','+','++') )
        ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
        ax.tick_params(axis='x', which='major', labelsize=10)
        ax.tick_params(axis='y', which='major', labelsize=10)
        ax.set_ylim(ymin=0.2, ymax=1.6)
        plt.gca().spines["bottom"].set_linewidth(.5)
        plt.gca().spines["left"].set_linewidth(.5)
        plt.xlabel('confidence')
        plt.ylabel('pupil response (a.u.)')
        plt.title('subject {}, c = {}'.format((self.subject.initials), round(np.array(self.parameters_joined['criterion'])[0],3)))
        plt.tight_layout()
        fig.savefig(os.path.join(self.base_directory, 'figs', 'pupil_confidence_choice_2_' + self.subject.initials + '.pdf'))
        
        # FIGURE 1
        MEANS = np.array([np.mean(self.pupil_d[(self.hit+self.fa)*cond])-np.mean(self.pupil_d[(self.miss+self.cr)*cond]) for cond in conditions])
        SEMS = np.array([(sp.stats.sem(self.pupil_d[(self.hit+self.fa)*cond])+sp.stats.sem(self.pupil_d[(self.miss+self.cr)*cond]))/2 for cond in conditions])
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot(111)
        ax.errorbar(ind, MEANS, yerr=SEMS, color = 'k', capsize = 0)
        simpleaxis(ax)
        spine_shift(ax)
        ax.set_xticklabels( ('--','-','+','++') )
        ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
        ax.tick_params(axis='x', which='major', labelsize=10)
        ax.tick_params(axis='y', which='major', labelsize=10)
        ax.set_ylim(ymin=-0.4, ymax=1.0)
        plt.gca().spines["bottom"].set_linewidth(.5)
        plt.gca().spines["left"].set_linewidth(.5)
        plt.xlabel('confidence')
        plt.ylabel('pupil response (a.u.)')
        plt.title('subject {}, c = {}'.format((self.subject.initials), round(np.array(self.parameters_joined['criterion'])[0],3)))
        plt.tight_layout()
        fig.savefig(os.path.join(self.base_directory, 'figs', 'pupil_confidence_choice_' + self.subject.initials + '.pdf'))
        
        # FIGURE 2
        MEANS = np.array([np.mean(self.pupil_d[(self.hit+self.cr)*cond])-np.mean(self.pupil_d[(self.fa+self.miss)*cond]) for cond in conditions])
        SEMS = np.array([(sp.stats.sem(self.pupil_d[(self.hit+self.cr)*cond])+sp.stats.sem(self.pupil_d[(self.fa+self.miss)*cond]))/2 for cond in conditions])
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot(111)
        for i in range(N):
            ax.bar(ind[i]+spacing[i], MEANS[i], width = bar_width, yerr=SEMS[i], color = 'b', alpha = [.25,0.5,0.75,1][i], edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
        simpleaxis(ax)
        spine_shift(ax)
        ax.set_xticklabels( ('--','-','+','++') )
        ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
        ax.tick_params(axis='x', which='major', labelsize=10)
        ax.tick_params(axis='y', which='major', labelsize=10)
        plt.gca().spines["bottom"].set_linewidth(.5)
        plt.gca().spines["left"].set_linewidth(.5)
        plt.ylabel('pupil correctness effect (a.u.)')
        plt.title('subject {}, c = {}'.format((self.subject.initials), round(np.array(self.parameters_joined['criterion'])[0],3)))
        plt.tight_layout()
        fig.savefig(os.path.join(self.base_directory, 'figs', 'pupil_confidence_correct_' + self.subject.initials + '.pdf'))
        
        # FIGURE 3
        MEANS = np.array([np.mean(self.pupil_d[(self.hit+self.cr+self.fa+self.miss)*cond]) for cond in conditions])
        SEMS = np.array([sp.stats.sem(self.pupil_d[(self.hit+self.cr+self.fa+self.miss)*cond]) for cond in conditions])
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot(111)
        for i in range(N):
            ax.bar(ind[i]+spacing[i], MEANS[i], width = bar_width, yerr=SEMS[i], color = 'b', alpha = [.25,0.5,0.75,1][i], edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
        simpleaxis(ax)
        spine_shift(ax)
        ax.set_xticklabels( ('--','-','+','++') )
        ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
        ax.tick_params(axis='x', which='major', labelsize=10)
        ax.tick_params(axis='y', which='major', labelsize=10)
        plt.gca().spines["bottom"].set_linewidth(.5)
        plt.gca().spines["left"].set_linewidth(.5)
        plt.ylabel('pupil response (a.u.)')
        plt.title('subject {}, c = {}'.format((self.subject.initials), round(np.array(self.parameters_joined['criterion'])[0],3)))
        plt.tight_layout()
        fig.savefig(os.path.join(self.base_directory, 'figs', 'pupil_confidence_overall_' + self.subject.initials + '.pdf'))
        
        if self.experiment == 1:
            # FIGURE 4
            MEANS_yes = np.array([np.mean(self.pupil_d_feed[(self.hit+self.cr)*cond]) for cond in conditions])
            SEMS_yes = np.array([sp.stats.sem(self.pupil_d_feed[(self.hit+self.cr)*cond]) for cond in conditions])
            MEANS_no = np.array([np.mean(self.pupil_d_feed[(self.fa+self.miss)*cond]) for cond in conditions])
            SEMS_no = np.array([sp.stats.sem(self.pupil_d_feed[(self.fa+self.miss)*cond]) for cond in conditions])
            fig = plt.figure(figsize=(4,3))
            ax = fig.add_subplot(111)
            ax.errorbar(ind, MEANS_no-MEANS_yes, yerr=(SEMS_yes+SEMS_no)/2.0, color = 'k', capsize = 0)
            simpleaxis(ax)
            spine_shift(ax)
            ax.set_xticklabels( ('--','-','+','++') )
            ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
            ax.tick_params(axis='x', which='major', labelsize=10)
            ax.tick_params(axis='y', which='major', labelsize=10)
            # ax.set_ylim(ymin=0.2, ymax=1.6)
            plt.gca().spines["bottom"].set_linewidth(.5)
            plt.gca().spines["left"].set_linewidth(.5)
            plt.xlabel('confidence')
            plt.ylabel('prediction error (a.u.)')
            plt.title('subject {}, c = {}'.format((self.subject.initials), round(np.array(self.parameters_joined['criterion'])[0],3)))
            plt.tight_layout()
            fig.savefig(os.path.join(self.base_directory, 'figs', 'pupil_prediction_error_' + self.subject.initials + '.pdf'))
            
    
    def sequential_effects(self):
        
        high_feedback_n = np.concatenate((np.array([False]), np.array(self.parameters_joined['ppr_peak_feed_lp'] > np.median(self.parameters_joined['ppr_peak_feed_lp']))))[:-1]
        
        d, c = myfuncs.SDT_measures(np.array(self.parameters_joined['present'])[high_feedback_n], np.array(self.parameters_joined['hit'])[high_feedback_n], np.array(self.parameters_joined['fa'])[high_feedback_n])
        d1, c1 = myfuncs.SDT_measures(np.array(self.parameters_joined['present'])[-high_feedback_n], np.array(self.parameters_joined['hit'])[-high_feedback_n], np.array(self.parameters_joined['fa'])[-high_feedback_n])
        
        print
        print
        print self.subject.initials
        print '---------------------'
        print 'd prime: {} vs {}'.format(d, d1)
        print 'criterion: {} vs {}'.format(c, c1)
        print
        print
        
    def rescorla_wagner(self):
        
        # rescorla wagner model:
        def simulate_run(reward_history, conf_history, learning_rate=.2):
            global_reward = np.zeros(len(reward_history))
            global_error = np.zeros(len(reward_history))
            local_error = np.zeros(len(reward_history))
            for i in xrange(0, len(reward_history)):
                local_error[i] = reward_history[i] - conf_history[i]
            for i in xrange(0, len(reward_history)): 
                try:
                    global_error[i] = global_reward[i] - reward_history[i]
                    global_reward[i+1] = global_reward[i] + (learning_rate * global_error[i])
                except IndexError:
                    pass
            return global_reward, global_error, local_error
        
        # variables:
        correct = np.array(parameters_joined['correct'] == 1, dtype=int)[-np.array(parameters_joined['omissions'])]
        conf = np.array(parameters_joined['confidence'])[-np.array(parameters_joined['omissions'])]
        conf[conf == 0] = .20
        conf[conf == 1] = .40
        conf[conf == 2] = .60
        conf[conf == 3] = .80
        ppd = np.array(parameters_joined['ppr_peak_feed_lp'])[-np.array(parameters_joined['omissions'])]
        global_reward, global_error, local_error = simulate_run(reward_history=correct, conf_history=conf, learning_rate=1.0)
        
        # boxplot:
        import matplotlib.collections as collections
        errors = np.unique(local_error)
        data = [ppd[local_error == error] for error in errors] 
        fig = plt.figure(figsize=(3,5))
        ax = plt.subplot(111)
        ax.boxplot(data)
        ax.set_xticklabels(errors)
        c1 = collections.BrokenBarHCollection(xranges=[(0.0,4.5)], yrange=ax.get_ylim(), facecolor='red', alpha=0.25)
        c2 = collections.BrokenBarHCollection(xranges=[(4.5,9.0)], yrange=ax.get_ylim(), facecolor='green', alpha=0.25)
        ax.add_collection(c1)
        ax.add_collection(c2)
        ax.set_xlabel('prediction error')
        ax.set_ylabel('pupil response (a.u.)')
        fig.savefig(os.path.join(self.base_directory, 'figs', self.subject.initials + '_suprise.pdf'))
        
    def GLM(self, aliases):
        
        downsample_rate = float(50) # 50
        new_sample_rate = 1000 / downsample_rate
        
        correct_choice_times = 0.24
        # correct_choice_times = 0.0
        
        # load data:
        parameters = []
        pupil = []
        pupil_diff = []
        time = []
        cue_times = []
        choice_times = []
        trial_end_times = []
        # feedback_times = []
        blink_times = []
        time_to_add = 0
        for alias in aliases:
            parameters.append(self.ho.read_session_data(alias, 'parameters2'))
            
            self.alias = alias
            self.trial_times = self.ho.read_session_data(alias, 'trials')
            
            # load pupil:
            self.eye = self.ho.eye_during_period((np.array(self.trial_times['trial_start_EL_timestamp'])[0], np.array(self.trial_times['trial_end_EL_timestamp'])[-1]), self.alias)
            self.pupil_data = self.ho.data_from_time_period((np.array(self.trial_times['trial_start_EL_timestamp'])[0], np.array(self.trial_times['trial_end_EL_timestamp'])[-1]), self.alias)
            # self.pupil_bp = np.array(self.pupil_data[(self.eye + '_pupil_bp')])
            # pupil.append(self.pupil_bp / np.std(self.pupil_bp))
            
            pupil.append(np.array(self.pupil_data[(self.eye + '_pupil_bp_clean_psc')]))
            # pupil.append(np.array(self.pupil_data[(self.eye + '_pupil_bp_zscore')]))
            
            # load times:
            self.session_start = self.trial_times['trial_start_EL_timestamp'][0]
            self.time = np.array(self.pupil_data['time']) - self.session_start
            time.append( self.time + time_to_add)
            self.phase_times = self.ho.read_session_data(alias, 'trial_phases')
            cue_times.append( np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 2)]) - self.session_start + time_to_add )
            choice_times.append( np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 3)]) - self.session_start + time_to_add )
            trial_end_times.append( np.array(self.trial_times['trial_end_EL_timestamp']) - self.session_start + time_to_add )
            # feedback_times.append( np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 5)]) - self.session_start + time_to_add )
            # load blinks:
            self.blink_data = self.ho.read_session_data(alias, 'blinks_from_message_file')
            blink_times.append( np.array(self.blink_data['end_timestamp']) - self.session_start + time_to_add )
            
            time_to_add += self.time[-1]
        
        # join over runs:
        parameters_joined = pd.concat(parameters)
        omissions = np.array(parameters_joined.omissions, dtype=bool)
        
        pupil = np.concatenate(pupil)
        pupil = sp.signal.decimate(pupil, int(downsample_rate))
        time = np.concatenate(time)
        blink_times = np.concatenate(blink_times)
        cue_times = np.array(np.concatenate(cue_times)[-omissions], dtype=int)
        choice_times = np.array(np.concatenate(choice_times)[-omissions], dtype=int)
        trial_end_times = np.array(np.concatenate(trial_end_times)[-omissions], dtype=int)
        rt = np.array(choice_times - cue_times, dtype=float)
        
        correct = np.array(parameters_joined.correct, dtype=bool)[-omissions]
        error = -np.array(parameters_joined.correct, dtype=bool)[-omissions]
        hit = np.array(parameters_joined.hit, dtype=bool)[-omissions]
        fa = np.array(parameters_joined.fa, dtype=bool)[-omissions]
        miss = np.array(parameters_joined.miss, dtype=bool)[-omissions]
        cr = np.array(parameters_joined.cr, dtype=bool)[-omissions]
        
        # for i in range(len(trial_end_times)):
        #     for b in blink_times:
        #         if trial_end_times[i] - cue_times[i] < 5000:
        #             trial_end_times[i] = cue_times[i] + 5000
        #         if trial_end_times[i] - choice_times[i] > 8000:
        #             trial_end_times[i] = choice_times[i] + 8000
        #         if (b > choice_times[i]) & (b < trial_end_times[i]):
        #             trial_end_times[i] = b
        
        # # don't epoch:
        # pupil_epoched = pupil.copy()
        # baseline_ind = np.zeros(len(pupil_epoched), dtype=bool)
        # for i in ((cue_times)/downsample_rate):
        #     baseline_ind[i-(500/downsample_rate):i] = True
        # pupil_epoched = pupil_epoched - pupil_epoched[baseline_ind].mean()
        # cue_times_epoched = cue_times.copy() / 1000.0
        # choice_times_epoched = choice_times.copy() / 1000.0
        
        # epoch:
        edge1 = 1000
        len_epoch = 6000
        pupil_epoched = []
        for i in ((cue_times-edge1)/downsample_rate):
            epoch = pupil[i:i+(len_epoch/downsample_rate)]
            epoch = epoch - epoch[0:edge1/downsample_rate].mean()
            # epoch[0:edge1/downsample_rate] = sp.signal.detrend(epoch[0:edge1/downsample_rate])
            # epoch = sp.signal.detrend(epoch)
            pupil_epoched.append(epoch)
        pupil_epoched = np.concatenate(pupil_epoched)
        cue_times_epoched = np.arange(0,len_epoch*cue_times.shape[0],len_epoch) + edge1
        choice_times_epoched = cue_times_epoched + (choice_times-cue_times)
        cue_times_epoched = cue_times_epoched / 1000.0
        choice_times_epoched = choice_times_epoched / 1000.0
        
        # correct choice_times:
        choice_times_epoched = choice_times_epoched - correct_choice_times
        
        try:
            os.system('mkdir {}'.format(os.path.join(self.base_directory, 'figs', 'GLM')))
        except:
            pass
        
        event_1 = np.ones((len(choice_times),3))
        event_1[:,0] = cue_times_epoched
        event_1[:,1] = 0
        event_2 = np.ones((len(choice_times),3))
        event_2[:,0] = choice_times_epoched
        event_2[:,1] = 0
        event_3 = np.ones((len(choice_times),3))
        event_3[:,0] = cue_times_epoched
        event_3[:,1] = choice_times_epoched-cue_times_epoched
        
        ind_params = False
        if ind_params:
            if self.subject.initials == 'AV':
                params = {'ID': 'tk', 'R': 0.5, 'tmax': 0.93, 'w': 10.1}
            if self.subject.initials == 'BL':
                params = {'ID': 'bl', 'R': 0.12493082998181665, 'tmax': 1.3, 'w': 12.0}
            if self.subject.initials == 'DE':
                params = {'ID': 'de', 'R': 0.30253841965962169, 'tmax': 0.54210526315789476, 'w': 4.8421052631578947}
            if self.subject.initials == 'DL':
                params = {'ID': 'dl', 'R': 0.50441537379877599, 'tmax': 0.75263157894736843, 'w': 4.0}
            if self.subject.initials == 'EP':
                params = {'ID': 'ep', 'R': 0.12725375961498106, 'tmax': 0.66842105263157892, 'w': 6.5263157894736841}
            if self.subject.initials == 'JG':
                params = {'ID': 'jw', 'R': 0.39353728468683857, 'tmax': 1.0894736842105264, 'w': 4.4210526315789469}
            if self.subject.initials == 'JS':
                params = {'ID': 'js', 'R': 0.4266433024004806, 'tmax': 0.83684210526315783, 'w': 4.0}
            if self.subject.initials == 'LH':
                params = {'ID': 'lh', 'R': 0.56086514440224444, 'tmax': 1.3, 'w': 7.3684210526315788}
            if self.subject.initials == 'LP':
                params = {'ID': 'lp', 'R': 0.21058816320730264, 'tmax': 1.1736842105263157, 'w': 11.578947368421051}
            if self.subject.initials == 'MG':
                params = {'ID': 'mg', 'R': 0.23887929555225784, 'tmax': 0.62631578947368416, 'w': 5.2631578947368425}
            if self.subject.initials == 'NS':
                params = {'ID': 'ns', 'R': 0.18308429749626953, 'tmax': 0.62631578947368416, 'w': 4.0}
            if self.subject.initials == 'OC':
                params = {'ID': 'oc', 'R': 0.16988708334682065, 'tmax': 1.131578947368421, 'w': 12.0}
            if self.subject.initials == 'TK':
                params = {'ID': 'tk', 'R': 0.48245753595522584, 'tmax': 1.3, 'w': 4.8421052631578947}
            if self.subject.initials == 'TN':
                params = {'ID': 'tk', 'R': 0.5, 'tmax': 0.93, 'w': 10.1}
            w = params['w']
            tmax = params['tmax']
        else:
            w = 10.1
            tmax = 0.93
        
        # shell()
        
        basis_set = False
        events = [event_1[:,:], event_2[:,:], event_3[:,:],]
        regressor_types = ['stick', 'stick', 'box',]
        # events = [event_3[:,:],]
        # regressor_types = ['upramp',]
        linear_model = GLM.GeneralLinearModel(input_object=pupil_epoched, event_object=events, sample_dur=1.0/new_sample_rate, new_sample_dur=1.0/new_sample_rate)
        linear_model.configure(IRF='pupil', IRF_params={'dur':4, 's':1.0/(10**26), 'n':w, 'tmax':tmax}, regressor_types=regressor_types, demean=False, basis_set=basis_set, normalize_sustained=True)
        linear_model.execute()
        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(linear_model.predicted, linear_model.working_data_array)
        best_R = r_value
        
        if basis_set:
            betas_0 = np.linalg.norm(linear_model.betas.reshape(len(regressor_types),3), axis=1) * np.sign(linear_model.betas.reshape(len(regressor_types),3))[:,0]
        else:
            betas_0 = linear_model.betas
        np.save(os.path.join(self.project_directory, 'data', 'across', 'pupil_GLM', 'GLM_betas_0_{}.npy'.format(self.subject.initials)), betas_0)
        np.save(os.path.join(self.project_directory, 'data', 'across', 'pupil_GLM', 'GLM_betas_0_R_{}.npy'.format(self.subject.initials)), best_R)
        
        # plot cue locked:
        x = np.linspace(-1,5,len_epoch / downsample_rate)
        pupil_mean = np.mean(np.vstack([linear_model.working_data_array[floor(i):floor(i)+(len_epoch / downsample_rate)] for i in ((cue_times_epoched[1:-1]*1000/downsample_rate)-20)]), axis=0)
        predicted_mean = np.mean(np.vstack([linear_model.predicted[floor(i):floor(i)+(len_epoch / downsample_rate)] for i in ((cue_times_epoched[1:-1]*1000/downsample_rate)-20)]), axis=0)
        fig = plt.figure(figsize=(2,2))
        plt.plot(x,pupil_mean, color='gray', label='measured, normal')
        plt.plot(x,predicted_mean, color='k', label='predicted, normal')
        plt.legend()
        plt.xlim(-1,5)
        plt.ylim(-1,5)
        plt.title('R={}, w={}, tmax={}'.format(r_value, w, tmax))
        plt.xlabel('Time (s)')
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.base_directory, 'figs', 'GLM_2a.pdf'))

        # plot response locked:
        x = np.linspace(-3.5,2.5,len_epoch / downsample_rate)
        pupil_mean = np.mean(np.vstack([linear_model.working_data_array[floor(i):floor(i)+(len_epoch / downsample_rate)] for i in ((choice_times_epoched[1:-1]*1000/downsample_rate)-70)]), axis=0)
        predicted_mean = np.mean(np.vstack([linear_model.predicted[floor(i):floor(i)+(len_epoch / downsample_rate)] for i in ((choice_times_epoched[1:-1]*1000/downsample_rate)-70)]), axis=0)
        fig = plt.figure(figsize=(2,2))
        plt.plot(x,pupil_mean, color='gray', label='measured, normal')
        plt.plot(x,predicted_mean, color='k', label='predicted, normal')
        plt.legend()
        plt.xlim(-3.5,2.5)
        plt.ylim(-1,5)
        plt.title('R={}, w={}, tmax={}'.format(r_value, w, tmax))
        plt.xlabel('Time (s)')
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.base_directory, 'figs', 'GLM_2b.pdf'))

        #
        fig = plt.figure(figsize=(12,2))
        x = np.linspace(0,linear_model.working_data_array.shape[0]/new_sample_rate,linear_model.working_data_array.shape[0])
        plt.plot(x, linear_model.working_data_array, color='r', lw=0.5)
        plt.plot(x, linear_model.predicted, color='b', lw=1)
        plt.legend(('measured', 'predicted'))
        line = 0
        for i in cue_times_epoched:
            plt.axvline(i, lw=0.5, alpha=0.5)
        plt.xlim(x[-1]-150,x[-1])
        plt.ylim([-10,10])
        plt.title('R={}, w={}, tmax={}'.format(r_value, w, tmax))
        plt.xlabel('Time (s)')
        plt.tight_layout()
        fig.savefig(os.path.join(self.base_directory, 'figs', 'measured_vs_predicted_2.pdf'))
        
        
        # separate for high and low TPR
        basis_set = False
        events = [event_1[self.pupil_h_ind,:], event_1[self.pupil_l_ind,:], event_2[self.pupil_h_ind,:], event_2[self.pupil_l_ind,:], event_3[self.pupil_h_ind,:], event_3[self.pupil_l_ind,:]]
        regressor_types = ['stick', 'stick', 'stick', 'stick', 'box', 'box']
        # events = [event_3[:,:],]
        # regressor_types = ['upramp',]
        linear_model = GLM.GeneralLinearModel(input_object=pupil_epoched, event_object=events, sample_dur=1.0/new_sample_rate, new_sample_dur=1.0/new_sample_rate)
        linear_model.configure(IRF='pupil', IRF_params={'dur':4, 's':1.0/(10**26), 'n':w, 'tmax':tmax}, regressor_types=regressor_types, demean=False, basis_set=basis_set, normalize_sustained=True)
        linear_model.execute()
            
        if basis_set:
            betas_0 = np.linalg.norm(linear_model.betas.reshape(len(regressor_types),3), axis=1) * np.sign(linear_model.betas.reshape(len(regressor_types),3))[:,0]
        else:
            betas_0 = linear_model.betas
        np.save(os.path.join(self.project_directory, 'data', 'across', 'pupil_GLM', 'GLM_betas_0_{}_pupil_split.npy'.format(self.subject.initials)), betas_0)
        
        return {'ID':self.subject.initials, 'R':best_R, 'w':w, 'tmax':tmax,}
        
        
        
    
    def GLM_extra_stick(self, aliases):
        
        downsample_rate = float(50) # 50
        new_sample_rate = 1000 / downsample_rate
        
        # load data:
        parameters = []
        pupil = []
        pupil_diff = []
        time = []
        cue_times = []
        choice_times = []
        trial_end_times = []
        # feedback_times = []
        blink_times = []
        time_to_add = 0
        for alias in aliases:
            parameters.append(self.ho.read_session_data(alias, 'parameters2'))
            
            self.alias = alias
            self.trial_times = self.ho.read_session_data(alias, 'trials')
            
            # load pupil:
            self.eye = self.ho.eye_during_period((np.array(self.trial_times['trial_start_EL_timestamp'])[0], np.array(self.trial_times['trial_end_EL_timestamp'])[-1]), self.alias)
            self.pupil_data = self.ho.data_from_time_period((np.array(self.trial_times['trial_start_EL_timestamp'])[0], np.array(self.trial_times['trial_end_EL_timestamp'])[-1]), self.alias)
            # self.pupil_bp = np.array(self.pupil_data[(self.eye + '_pupil_bp')])
            # pupil.append(self.pupil_bp / np.std(self.pupil_bp))
            
            pupil.append(np.array(self.pupil_data[(self.eye + '_pupil_lp_psc')]))
            # pupil.append(np.array(self.pupil_data[(self.eye + '_pupil_bp_zscore')]))
            
            # load times:
            self.session_start = self.trial_times['trial_start_EL_timestamp'][0]
            self.time = np.array(self.pupil_data['time']) - self.session_start
            time.append( self.time + time_to_add)
            self.phase_times = self.ho.read_session_data(alias, 'trial_phases')
            cue_times.append( np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 2)]) - self.session_start + time_to_add )
            choice_times.append( np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 3)]) - self.session_start + time_to_add )
            trial_end_times.append( np.array(self.trial_times['trial_end_EL_timestamp']) - self.session_start + time_to_add )
            # feedback_times.append( np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 5)]) - self.session_start + time_to_add )
            # load blinks:
            self.blink_data = self.ho.read_session_data(alias, 'blinks_from_message_file')
            blink_times.append( np.array(self.blink_data['end_timestamp']) - self.session_start + time_to_add )
            
            time_to_add += self.time[-1]
        
        # join over runs:
        parameters_joined = pd.concat(parameters)
        omissions = np.array(parameters_joined.omissions, dtype=bool)
        
        pupil = np.concatenate(pupil)
        pupil = sp.signal.decimate(pupil, int(downsample_rate))
        time = np.concatenate(time)
        blink_times = np.concatenate(blink_times)
        cue_times = np.array(np.concatenate(cue_times)[-omissions], dtype=int)
        choice_times = np.array(np.concatenate(choice_times)[-omissions], dtype=int)
        trial_end_times = np.array(np.concatenate(trial_end_times)[-omissions], dtype=int)
        rt = np.array(choice_times - cue_times, dtype=float)
        
        correct = np.array(parameters_joined.correct, dtype=bool)[-omissions]
        error = -np.array(parameters_joined.correct, dtype=bool)[-omissions]
        hit = np.array(parameters_joined.hit, dtype=bool)[-omissions]
        fa = np.array(parameters_joined.fa, dtype=bool)[-omissions]
        miss = np.array(parameters_joined.miss, dtype=bool)[-omissions]
        cr = np.array(parameters_joined.cr, dtype=bool)[-omissions]
        
        for i in range(len(trial_end_times)):
            for b in blink_times:
                if trial_end_times[i] - cue_times[i] < 5000:
                    trial_end_times[i] = cue_times[i] + 5000
                if trial_end_times[i] - choice_times[i] > 8000:
                    trial_end_times[i] = choice_times[i] + 8000
                if (b > choice_times[i]) & (b < trial_end_times[i]):
                    trial_end_times[i] = b
        
        # epoch:
        edge1 = 500
        len_epoch = 5500
        pupil_epoched = []
        for i in ((cue_times-edge1)/downsample_rate):
            epoch = sp.signal.detrend(pupil[i:i+(len_epoch/downsample_rate)])
            pupil_epoched.append(epoch)
        pupil_epoched = np.concatenate(pupil_epoched)
        cue_times_epoched = np.arange(0,len_epoch*cue_times.shape[0],len_epoch) + edge1
        choice_times_epoched = cue_times_epoched + (choice_times-cue_times)
        cue_times_epoched = cue_times_epoched / 1000.0
        choice_times_epoched = choice_times_epoched / 1000.0
        
        try:
            os.system('mkdir {}'.format(os.path.join(self.base_directory, 'figs', 'GLM_l')))
        except:
            pass
        
        event_1 = np.ones((len(choice_times),3))
        event_1[:,0] = cue_times_epoched
        event_1[:,1] = 0
        event_2 = np.ones((len(choice_times),3))
        event_2[:,0] = choice_times_epoched
        event_2[:,1] = 0
        event_3 = np.ones((len(choice_times),3))
        event_3[:,0] = cue_times_epoched
        event_3[:,1] = choice_times_epoched-cue_times_epoched
        
        event_4 = np.ones((len(choice_times),3))
        event_4[:,0] = choice_times_epoched
        event_4[:,1] = 0
        
        best_R = 0
        best_latency = 0
        
        w = 10.1
        tmax = 0.93
        
        # look for best latency:
        for latency in np.linspace(1,2,20):
            basis_set = False
            
            
            event_4_l = copy.copy(event_4)
            event_4_l[:,0] = event_4[:,0] + latency
            
            events = [event_1[:,:], event_2[:,:], event_3[:,:], event_4_l[:,:]]
            regressor_types = ['stick', 'stick', 'box', 'stick']
            linear_model = GLM.GeneralLinearModel(input_object=pupil_epoched, event_object=events, sample_dur=1.0/new_sample_rate, new_sample_dur=1.0/new_sample_rate)
            linear_model.configure(IRF='pupil', IRF_params={'dur':4, 's':1.0/(10**26), 'n':w, 'tmax':tmax}, regressor_types=regressor_types, demean=True, basis_set=basis_set, normalize_sustained=False)
            linear_model.execute()
            slope, intercept, r_value, p_value, std_err = sp.stats.linregress(linear_model.predicted, linear_model.working_data_array)
            if r_value > best_R:
                best_R = r_value
                best_latency = latency
        
        basis_set = False
        event_4_l = copy.copy(event_4)
        event_4_l[:,0] = event_4[:,0] + best_latency
        events = [event_1[:,:], event_2[:,:], event_3[:,:], event_4_l]
        regressor_types = ['stick', 'stick', 'box', 'stick']
        linear_model = GLM.GeneralLinearModel(input_object=pupil_epoched, event_object=events, sample_dur=1.0/new_sample_rate, new_sample_dur=1.0/new_sample_rate)
        linear_model.configure(IRF='pupil', IRF_params={'dur':4, 's':1.0/(10**26), 'n':w, 'tmax':tmax}, regressor_types=regressor_types, demean=True, basis_set=basis_set, normalize_sustained=False)
        # linear_model.configure(IRF='pupil', IRF_params={'dur':4, 's':1.0/(10**26), 'n':10.1, 'tmax':0.93}, regressor_types=regressor_types, demean=True, basis_set=basis_set, normalize_sustained=False)
        linear_model.execute()
        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(linear_model.predicted, linear_model.working_data_array)
        best_R = r_value
        
        if basis_set:
            betas_0 = np.linalg.norm(linear_model.betas.reshape(len(regressor_types),3), axis=1) * np.sign(linear_model.betas.reshape(len(regressor_types),3))[:,0]
        else:
            betas_0 = linear_model.betas
        
        # plot cue locked:
        x = np.linspace(-0.5,4.5,100)
        pupil_mean = np.mean(np.vstack([linear_model.working_data_array[floor(i):floor(i)+100] for i in ((cue_times_epoched[1:-1]*1000/downsample_rate)-10)]), axis=0)
        predicted_mean = np.mean(np.vstack([linear_model.predicted[floor(i):floor(i)+100] for i in ((cue_times_epoched[1:-1]*1000/downsample_rate)-10)]), axis=0)
        fig = plt.figure()
        plt.plot(x,pupil_mean, color='gray', label='measured, normal')
        plt.plot(x,predicted_mean, color='k', label='predicted, normal')
        plt.legend()
        plt.xlim(-0.5,4.5)
        plt.title('R={}, latency={}'.format(r_value, best_latency))
        plt.xlabel('Time (s)')
        fig.savefig(os.path.join(self.base_directory, 'figs', 'GLM_l', 'GLM_2a.pdf'))

        # plot response locked:
        x = np.linspace(-3,2,100)
        pupil_mean = np.mean(np.vstack([linear_model.working_data_array[floor(i):floor(i)+100] for i in ((choice_times_epoched[1:-1]*1000/downsample_rate)-60)]), axis=0)
        predicted_mean = np.mean(np.vstack([linear_model.predicted[floor(i):floor(i)+100] for i in ((choice_times_epoched[1:-1]*1000/downsample_rate)-60)]), axis=0)
        fig = plt.figure()
        plt.plot(x,pupil_mean, color='gray', label='measured, normal')
        plt.plot(x,predicted_mean, color='k', label='predicted, normal')
        plt.legend()
        plt.xlim(-3,2)
        plt.title('R={}, latency={}'.format(r_value, best_latency))
        plt.xlabel('Time (s)')
        fig.savefig(os.path.join(self.base_directory, 'figs', 'GLM_l', 'GLM_2b.pdf'))

        # # save:
        # np.save(os.path.join(self.project_directory, 'across_data', 'GLM_betas_0_{}.npy'.format(self.subject.initials)), betas_0)
        fig = plt.figure(figsize=(12,2))
        x = np.linspace(0,linear_model.working_data_array.shape[0]/new_sample_rate,linear_model.working_data_array.shape[0])
        plt.plot(x, linear_model.working_data_array, color='r', lw=0.5)
        plt.plot(x, linear_model.predicted, color='b', lw=1)
        plt.legend(('measured', 'predicted'))
        line = 0
        for i in cue_times_epoched:
            plt.axvline(i, lw=0.5, alpha=0.5)
        plt.xlim(x[-1]-150,x[-1])
        plt.ylim([-10,10])
        plt.title('R={}, latency={}'.format(r_value, best_latency))
        plt.xlabel('Time (s)')
        plt.tight_layout()
        fig.savefig(os.path.join(self.base_directory, 'figs', 'GLM_l', 'measured_vs_predicted_2.pdf'))
        
        print
        print 'latency = {}, subj = {}'.format(best_latency, self.subject.initials)
        print

        return {'ID':self.subject.initials, 'R':best_R, 'latency':best_latency,}
    
    def GLM_fit(self, aliases):
        
        downsample_rate = float(50) # 50
        new_sample_rate = 1000 / downsample_rate
        
        # load data:
        parameters = []
        pupil = []
        pupil_diff = []
        time = []
        cue_times = []
        choice_times = []
        trial_end_times = []
        # feedback_times = []
        blink_times = []
        time_to_add = 0
        for alias in aliases:
            parameters.append(self.ho.read_session_data(alias, 'parameters2'))
            
            self.alias = alias
            self.trial_times = self.ho.read_session_data(alias, 'trials')
            
            # load pupil:
            self.eye = self.ho.eye_during_period((np.array(self.trial_times['trial_start_EL_timestamp'])[0], np.array(self.trial_times['trial_end_EL_timestamp'])[-1]), self.alias)
            self.pupil_data = self.ho.data_from_time_period((np.array(self.trial_times['trial_start_EL_timestamp'])[0], np.array(self.trial_times['trial_end_EL_timestamp'])[-1]), self.alias)
            # self.pupil_bp = np.array(self.pupil_data[(self.eye + '_pupil_bp')])
            # pupil.append(self.pupil_bp / np.std(self.pupil_bp))
            
            pupil.append(np.array(self.pupil_data[(self.eye + '_pupil_lp_psc')]))
            # pupil.append(np.array(self.pupil_data[(self.eye + '_pupil_bp_zscore')]))
            
            # load times:
            self.session_start = self.trial_times['trial_start_EL_timestamp'][0]
            self.time = np.array(self.pupil_data['time']) - self.session_start
            time.append( self.time + time_to_add)
            self.phase_times = self.ho.read_session_data(alias, 'trial_phases')
            cue_times.append( np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 2)]) - self.session_start + time_to_add )
            choice_times.append( np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 3)]) - self.session_start + time_to_add )
            trial_end_times.append( np.array(self.trial_times['trial_end_EL_timestamp']) - self.session_start + time_to_add )
            # feedback_times.append( np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 5)]) - self.session_start + time_to_add )
            # load blinks:
            self.blink_data = self.ho.read_session_data(alias, 'blinks_from_message_file')
            blink_times.append( np.array(self.blink_data['end_timestamp']) - self.session_start + time_to_add )
            
            time_to_add += self.time[-1]
        
        # join over runs:
        parameters_joined = pd.concat(parameters)
        omissions = np.array(parameters_joined.omissions, dtype=bool)
        
        pupil = np.concatenate(pupil)
        pupil = sp.signal.decimate(pupil, int(downsample_rate))
        time = np.concatenate(time)
        blink_times = np.concatenate(blink_times)
        cue_times = np.array(np.concatenate(cue_times)[-omissions], dtype=int)
        choice_times = np.array(np.concatenate(choice_times)[-omissions], dtype=int)
        trial_end_times = np.array(np.concatenate(trial_end_times)[-omissions], dtype=int)
        rt = np.array(choice_times - cue_times, dtype=float)
        
        correct = np.array(parameters_joined.correct, dtype=bool)[-omissions]
        error = -np.array(parameters_joined.correct, dtype=bool)[-omissions]
        hit = np.array(parameters_joined.hit, dtype=bool)[-omissions]
        fa = np.array(parameters_joined.fa, dtype=bool)[-omissions]
        miss = np.array(parameters_joined.miss, dtype=bool)[-omissions]
        cr = np.array(parameters_joined.cr, dtype=bool)[-omissions]
        
        for i in range(len(trial_end_times)):
            for b in blink_times:
                if trial_end_times[i] - cue_times[i] < 5000:
                    trial_end_times[i] = cue_times[i] + 5000
                if trial_end_times[i] - choice_times[i] > 8000:
                    trial_end_times[i] = choice_times[i] + 8000
                if (b > choice_times[i]) & (b < trial_end_times[i]):
                    trial_end_times[i] = b
        
        # epoch:
        edge1 = 500
        len_epoch = 5500
        pupil_epoched = []
        for i in ((cue_times-edge1)/downsample_rate):
            epoch = sp.signal.detrend(pupil[i:i+(len_epoch/downsample_rate)])
            pupil_epoched.append(epoch)
        pupil_epoched = np.concatenate(pupil_epoched)
        cue_times_epoched = np.arange(0,len_epoch*cue_times.shape[0],len_epoch) + edge1
        choice_times_epoched = cue_times_epoched + (choice_times-cue_times)
        cue_times_epoched = cue_times_epoched / 1000.0
        choice_times_epoched = choice_times_epoched / 1000.0
        
        # edge1 = 500
        # len_epoch = 5500
        # for i in range(len(cue_times)):
        #     pupil[int((cue_times[i]-edge1)/downsample_rate):int(trial_end_times[i]/downsample_rate)] = sp.signal.detrend(pupil[int((cue_times[i]-edge1)/downsample_rate):int(trial_end_times[i]/downsample_rate)])
        # pupil_epoched = []
        # for i in ((cue_times-edge1)/downsample_rate):
        #     epoch = pupil[i:i+(len_epoch/downsample_rate)]
        #     pupil_epoched.append(epoch)
        # pupil_epoched = np.concatenate(pupil_epoched)
        # cue_times_epoched = np.arange(0,len_epoch*cue_times.shape[0],len_epoch) + edge1
        # choice_times_epoched = cue_times_epoched + (choice_times-cue_times)
        # cue_times_epoched = cue_times_epoched / 1000.0
        # choice_times_epoched = choice_times_epoched / 1000.0
        
        #     len_epoch.append(epoch.shape[0])
        #     # print epoch.shape[0] * downsample_rate / 1000.0
        # pupil_epoched = np.concatenate(pupil_epoched)
        # len_epoch = np.array(len_epoch)
        # cue_times_epoched = (np.cumsum(np.array(len_epoch)) - ((trial_end_times-choice_times)/downsample_rate) - (rt/downsample_rate)) / 1000.0 * downsample_rate
        # choice_times_epoched = (np.cumsum(np.array(len_epoch)) - ((trial_end_times-choice_times)/downsample_rate)) / 1000.0 * downsample_rate
        
        
        try:
            os.system('mkdir {}'.format(os.path.join(self.base_directory, 'figs', 'GLM')))
        except:
            pass
        
        event_1 = np.ones((len(choice_times),3))
        event_1[:,0] = cue_times_epoched
        event_1[:,1] = 0
        event_2 = np.ones((len(choice_times),3))
        event_2[:,0] = choice_times_epoched
        event_2[:,1] = 0
        event_3 = np.ones((len(choice_times),3))
        event_3[:,0] = cue_times_epoched
        event_3[:,1] = choice_times_epoched-cue_times_epoched
        
        best_R = 0
        best_w = 0
        best_tmax = 0
        
        # look for best fitting IRF:
        for t_max in np.linspace(0.5,1.3,20):
            for w in np.linspace(4,12,20):
                basis_set = False
                events = [event_1[:,:], event_2[:,:], event_3[:,:],]
                regressor_types = ['stick', 'stick', 'box',]
                linear_model = GLM.GeneralLinearModel(input_object=pupil_epoched, event_object=events, sample_dur=1.0/new_sample_rate, new_sample_dur=1.0/new_sample_rate)
                linear_model.configure(IRF='pupil', IRF_params={'dur':4, 's':1.0/(10**26), 'n':w, 'tmax':t_max}, regressor_types=regressor_types, demean=True, basis_set=basis_set, normalize_sustained=False)
                linear_model.execute()
                slope, intercept, r_value, p_value, std_err = sp.stats.linregress(linear_model.predicted, linear_model.working_data_array)
                if r_value > best_R:
                    best_R = r_value
                    best_tmax = t_max
                    best_w = w
        
        basis_set = False
        events = [event_1[:,:], event_2[:,:], event_3[:,:],]
        regressor_types = ['stick', 'stick', 'box',]
        linear_model = GLM.GeneralLinearModel(input_object=pupil_epoched, event_object=events, sample_dur=1.0/new_sample_rate, new_sample_dur=1.0/new_sample_rate)
        linear_model.configure(IRF='pupil', IRF_params={'dur':4, 's':1.0/(10**26), 'n':best_w, 'tmax':best_tmax}, regressor_types=regressor_types, demean=True, basis_set=basis_set, normalize_sustained=False)
        # linear_model.configure(IRF='pupil', IRF_params={'dur':4, 's':1.0/(10**26), 'n':10.1, 'tmax':0.93}, regressor_types=regressor_types, demean=True, basis_set=basis_set, normalize_sustained=False)
        linear_model.execute()
        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(linear_model.predicted, linear_model.working_data_array)
        best_R = r_value
        
        if basis_set:
            betas_0 = np.linalg.norm(linear_model.betas.reshape(len(regressor_types),3), axis=1) * np.sign(linear_model.betas.reshape(len(regressor_types),3))[:,0]
        else:
            betas_0 = linear_model.betas
        
        # plot cue locked:
        x = np.linspace(-0.5,4.5,100)
        pupil_mean = np.mean(np.vstack([linear_model.working_data_array[floor(i):floor(i)+100] for i in ((cue_times_epoched[1:-1]*1000/downsample_rate)-10)]), axis=0)
        predicted_mean = np.mean(np.vstack([linear_model.predicted[floor(i):floor(i)+100] for i in ((cue_times_epoched[1:-1]*1000/downsample_rate)-10)]), axis=0)
        fig = plt.figure()
        plt.plot(x,pupil_mean, color='gray', label='measured, normal')
        plt.plot(x,predicted_mean, color='k', label='predicted, normal')
        plt.legend()
        plt.xlim(-0.5,4.5)
        plt.title('R={}, w={}, tmax={}'.format(r_value, best_w, best_tmax))
        plt.xlabel('Time (s)')
        fig.savefig(os.path.join(self.base_directory, 'figs', 'GLM', 'GLM_2a.pdf'))

        # plot response locked:
        x = np.linspace(-3,2,100)
        pupil_mean = np.mean(np.vstack([linear_model.working_data_array[floor(i):floor(i)+100] for i in ((choice_times_epoched[1:-1]*1000/downsample_rate)-60)]), axis=0)
        predicted_mean = np.mean(np.vstack([linear_model.predicted[floor(i):floor(i)+100] for i in ((choice_times_epoched[1:-1]*1000/downsample_rate)-60)]), axis=0)
        fig = plt.figure()
        plt.plot(x,pupil_mean, color='gray', label='measured, normal')
        plt.plot(x,predicted_mean, color='k', label='predicted, normal')
        plt.legend()
        plt.xlim(-3,2)
        plt.title('R={}, w={}, tmax={}'.format(r_value, best_w, best_tmax))
        plt.xlabel('Time (s)')
        fig.savefig(os.path.join(self.base_directory, 'figs', 'GLM', 'GLM_2b.pdf'))

        # # save:
        # np.save(os.path.join(self.project_directory, 'across_data', 'GLM_betas_0_{}.npy'.format(self.subject.initials)), betas_0)
        fig = plt.figure(figsize=(12,2))
        x = np.linspace(0,linear_model.working_data_array.shape[0]/new_sample_rate,linear_model.working_data_array.shape[0])
        plt.plot(x, linear_model.working_data_array, color='r', lw=0.5)
        plt.plot(x, linear_model.predicted, color='b', lw=1)
        plt.legend(('measured', 'predicted'))
        line = 0
        for i in cue_times_epoched:
            plt.axvline(i, lw=0.5, alpha=0.5)
        plt.xlim(x[-1]-150,x[-1])
        plt.ylim([-10,10])
        plt.title('R={}, w={}, tmax={}'.format(r_value, best_w, best_tmax))
        plt.xlabel('Time (s)')
        plt.tight_layout()
        fig.savefig(os.path.join(self.base_directory, 'figs', 'GLM', 'measured_vs_predicted_2.pdf'))
        
        print
        print 'w = {}, tmax = {}, subj = {}'.format(best_w, best_tmax, self.subject.initials)
        print
        
        return {'ID':self.subject.initials, 'R':best_R, 'w':best_w, 'tmax':best_tmax,} 
        
    
    
    def split_by_GLM(self, aliases, perms=100, run=False):
        
        downsample_rate = float(50) # 50
        new_sample_rate = 1000 / downsample_rate
        
        # load data:
        parameters = []
        pupil = []
        pupil_diff = []
        time = []
        cue_times = []
        choice_times = []
        trial_end_times = []
        # feedback_times = []
        blink_times = []
        time_to_add = 0
        for alias in aliases:
            parameters.append(self.ho.read_session_data(alias, 'parameters2'))
            
            self.alias = alias
            self.trial_times = self.ho.read_session_data(alias, 'trials')
            
            # load pupil:
            self.eye = self.ho.eye_during_period((np.array(self.trial_times['trial_start_EL_timestamp'])[0], np.array(self.trial_times['trial_end_EL_timestamp'])[-1]), self.alias)
            self.pupil_data = self.ho.data_from_time_period((np.array(self.trial_times['trial_start_EL_timestamp'])[0], np.array(self.trial_times['trial_end_EL_timestamp'])[-1]), self.alias)
            # self.pupil_bp = np.array(self.pupil_data[(self.eye + '_pupil_bp')])
            # pupil.append(self.pupil_bp / np.std(self.pupil_bp))
            
            pupil.append(np.array(self.pupil_data[(self.eye + '_pupil_lp_psc')]))
            # pupil.append(np.array(self.pupil_data[(self.eye + '_pupil_bp_zscore')]))
            
            # load times:
            self.session_start = self.trial_times['trial_start_EL_timestamp'][0]
            self.time = np.array(self.pupil_data['time']) - self.session_start
            time.append( self.time + time_to_add)
            self.phase_times = self.ho.read_session_data(alias, 'trial_phases')
            cue_times.append( np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 2)]) - self.session_start + time_to_add )
            choice_times.append( np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 3)]) - self.session_start + time_to_add )
            trial_end_times.append( np.array(self.trial_times['trial_end_EL_timestamp']) - self.session_start + time_to_add )
            # feedback_times.append( np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 5)]) - self.session_start + time_to_add )
            # load blinks:
            self.blink_data = self.ho.read_session_data(alias, 'blinks_from_message_file')
            blink_times.append( np.array(self.blink_data['end_timestamp']) - self.session_start + time_to_add )
            
            time_to_add += self.time[-1]
        
        # join over runs:
        parameters_joined = pd.concat(parameters)
        omissions = np.array(parameters_joined.omissions, dtype=bool)
        
        pupil = np.concatenate(pupil)
        pupil = sp.signal.decimate(pupil, int(downsample_rate))
        time = np.concatenate(time)
        blink_times = np.concatenate(blink_times)
        cue_times = np.array(np.concatenate(cue_times)[-omissions], dtype=int)
        choice_times = np.array(np.concatenate(choice_times)[-omissions], dtype=int)
        trial_end_times = np.array(np.concatenate(trial_end_times)[-omissions], dtype=int)
        rt = np.array(choice_times - cue_times, dtype=float)
        
        correct = np.array(parameters_joined.correct, dtype=bool)[-omissions]
        error = -np.array(parameters_joined.correct, dtype=bool)[-omissions]
        hit = np.array(parameters_joined.hit, dtype=bool)[-omissions]
        fa = np.array(parameters_joined.fa, dtype=bool)[-omissions]
        miss = np.array(parameters_joined.miss, dtype=bool)[-omissions]
        cr = np.array(parameters_joined.cr, dtype=bool)[-omissions]
        
        for i in range(len(trial_end_times)):
            for b in blink_times:
                if trial_end_times[i] - cue_times[i] < 5000:
                    trial_end_times[i] = cue_times[i] + 5000
                if trial_end_times[i] - choice_times[i] > 8000:
                    trial_end_times[i] = choice_times[i] + 8000
                if (b > choice_times[i]) & (b < trial_end_times[i]):
                    trial_end_times[i] = b
        
        # epoch:
        edge1 = 500
        len_epoch = 5500
        pupil_epoched = []
        for i in ((cue_times-edge1)/downsample_rate):
            epoch = sp.signal.detrend(pupil[i:i+(len_epoch/downsample_rate)])
            pupil_epoched.append(epoch)
        pupil_epoched = np.concatenate(pupil_epoched)
        cue_times_epoched = np.arange(0,len_epoch*cue_times.shape[0],len_epoch) + edge1
        choice_times_epoched = cue_times_epoched + (choice_times-cue_times)
        cue_times_epoched = cue_times_epoched / 1000.0
        choice_times_epoched = choice_times_epoched / 1000.0
        
        # edge1 = 500
        # len_epoch = 5500
        # for i in range(len(cue_times)):
        #     pupil[int((cue_times[i]-edge1)/downsample_rate):int(trial_end_times[i]/downsample_rate)] = sp.signal.detrend(pupil[int((cue_times[i]-edge1)/downsample_rate):int(trial_end_times[i]/downsample_rate)])
        # pupil_epoched = []
        # for i in ((cue_times-edge1)/downsample_rate):
        #     epoch = pupil[i:i+(len_epoch/downsample_rate)]
        #     pupil_epoched.append(epoch)
        # pupil_epoched = np.concatenate(pupil_epoched)
        # cue_times_epoched = np.arange(0,len_epoch*cue_times.shape[0],len_epoch) + edge1
        # choice_times_epoched = cue_times_epoched + (choice_times-cue_times)
        # cue_times_epoched = cue_times_epoched / 1000.0
        # choice_times_epoched = choice_times_epoched / 1000.0
        
        nr_trials = choice_times.shape[0]
        
        # shell()
        
        if run:
            
            regressor_types = ['upramp', 'upramp',]
            
            inds_0 = []
            inds_1 = []
            diff_box = np.zeros(perms)
            # betas = np.zeros((perms, 6))
            betas = np.zeros((perms, 2))
            for perm in range(perms):
            
                nr_trials_split = floor(nr_trials / 100.0 * 40)
                ind = np.zeros(nr_trials)
                ind[:nr_trials_split] = 1
                ind[-nr_trials_split:] = 2
                ind = ind[np.argsort(np.random.rand(nr_trials))]
                ind_0 = (ind==1)
                ind_1 = (ind==2)
                
                event_1 = np.ones((sum(ind_0),3))
                event_1[:,0] = cue_times_epoched[ind_0]
                event_1[:,1] = choice_times_epoched[ind_0]-cue_times_epoched[ind_0]
    
                event_2 = np.ones((sum(ind_1),3))
                event_2[:,0] = cue_times_epoched[ind_1]
                event_2[:,1] = choice_times_epoched[ind_1]-cue_times_epoched[ind_1]
                
                # event_1 = np.ones((sum(ind_0),3))
                # event_1[:,0] = cue_times_epoched[ind_0]
                # event_1[:,1] = 0
                # event_2 = np.ones((sum(ind_0),3))
                # event_2[:,0] = choice_times_epoched[ind_0]
                # event_2[:,1] = 0
                # event_3 = np.ones((sum(ind_0),3))
                # event_3[:,0] = cue_times_epoched[ind_0]
                # event_3[:,1] = choice_times_epoched[ind_0]-cue_times_epoched[ind_0]
                #
                # event_4 = np.ones((sum(ind_1),3))
                # event_4[:,0] = cue_times_epoched[ind_1]
                # event_4[:,1] = 0
                # event_5 = np.ones((sum(ind_1),3))
                # event_5[:,0] = choice_times_epoched[ind_1]
                # event_5[:,1] = 0
                # event_6 = np.ones((sum(ind_1),3))
                # event_6[:,0] = cue_times_epoched[ind_1]
                # event_6[:,1] = choice_times_epoched[ind_1]-cue_times_epoched[ind_1]
                #
                # regressor_types = ['stick', 'stick', 'box', 'stick', 'stick', 'box',]
                
                # original
                basis_set = False
                # events = [event_1[:,:], event_2[:,:], event_3[:,:], event_4[:,:], event_5[:,:], event_6[:,:]]
                events = [event_1[:,:], event_2[:,:],]
                linear_model = GLM.GeneralLinearModel(input_object=pupil_epoched, event_object=events, sample_dur=1.0/new_sample_rate, new_sample_dur=1.0/new_sample_rate)
                linear_model.configure(IRF='pupil', IRF_params={'dur':3, 's':1.0/(10**26), 'n':10.1, 'tmax':0.93}, regressor_types=regressor_types, demean=True, basis_set=basis_set, normalize_sustained=False)
            
                # assumptions:
                # dm = np.array(linear_model.design_matrix)
                # print 'assumptions:'
                # print '------------'
                # for i in range(dm.shape[0]):
                #     for j in range(i+1, dm.shape[0]):
                #         print 'reg {} x reg {}: R = {}'.format(i, j, round(np.mean(dm[i,:]*dm[j,:]),3))
            
                # execute:
                linear_model.execute()
                # if basis_set:
                #     betas_0 = np.linalg.norm(linear_model.betas.reshape(len(regressor_types),2), axis=1)
                # else:
                betas_0 = linear_model.betas
                r_value, p_value = sp.stats.pearsonr(linear_model.predicted, linear_model.working_data_array)
                # print 'R = {}'.format(r_value)
            
                # append:
                inds_0.append( ind_0 )
                inds_1.append( ind_1 )
                # diff_box[perm] = betas_0[5] - betas_0[2]
                diff_box[perm] = betas_0[1] - betas_0[0]
                betas[perm,:] = betas_0
        
            ind_max_0 = inds_0[np.argsort(diff_box)[-1]]
            ind_max_1 = inds_1[np.argsort(diff_box)[-1]]
            betas_max = betas[np.argsort(diff_box)[-1],:]
            print 'max diff boxcar = {}'.format(diff_box[np.argsort(diff_box)[-1]])
            
            
            # save:
            np.save(os.path.join(self.project_directory, 'data', 'across', 'ind_max_0_{}.npy'.format(self.subject.initials)), ind_max_0)
            np.save(os.path.join(self.project_directory, 'data', 'across', 'ind_max_1_{}.npy'.format(self.subject.initials)), ind_max_1)
            np.save(os.path.join(self.project_directory, 'data', 'across', 'betas_max_{}.npy'.format(self.subject.initials)), betas_max)
            
        else:
        
            ind_max_0 = np.load(os.path.join(self.project_directory, 'data', 'across', 'ind_max_0_{}.npy'.format(self.subject.initials)))
            ind_max_1 = np.load(os.path.join(self.project_directory, 'data', 'across', 'ind_max_1_{}.npy'.format(self.subject.initials)))
            betas_max = np.load(os.path.join(self.project_directory, 'data', 'across', 'betas_max_{}.npy'.format(self.subject.initials)))
        
            # fit maximised model:
            # event_1 = np.ones((sum(ind_max_0),3))
            # event_1[:,0] = cue_times_epoched[ind_max_0]
            # event_1[:,1] = 0
            # event_2 = np.ones((sum(ind_max_0),3))
            # event_2[:,0] = choice_times_epoched[ind_max_0]
            # event_2[:,1] = 0
            # event_3 = np.ones((sum(ind_max_0),3))
            # event_3[:,0] = cue_times_epoched[ind_max_0]
            # event_3[:,1] = choice_times_epoched[ind_max_0]-cue_times_epoched[ind_max_0]
            #
            # event_4 = np.ones((sum(ind_max_1),3))
            # event_4[:,0] = cue_times_epoched[ind_max_1]
            # event_4[:,1] = 0
            # event_5 = np.ones((sum(ind_max_1),3))
            # event_5[:,0] = choice_times_epoched[ind_max_1]
            # event_5[:,1] = 0
            # event_6 = np.ones((sum(ind_max_1),3))
            # event_6[:,0] = cue_times_epoched[ind_max_1]
            # event_6[:,1] = choice_times_epoched[ind_max_1]-cue_times_epoched[ind_max_1]
            #
            # regressor_types = ['stick', 'stick', 'box', 'stick', 'stick', 'box',]
            
            event_1 = np.ones((sum(ind_max_0),3))
            event_1[:,0] = cue_times_epoched[ind_max_0]
            event_1[:,1] = choice_times_epoched[ind_max_0]-cue_times_epoched[ind_max_0]
    
            event_2 = np.ones((sum(ind_max_1),3))
            event_2[:,0] = cue_times_epoched[ind_max_1]
            event_2[:,1] = choice_times_epoched[ind_max_1]-cue_times_epoched[ind_max_1]
            
            regressor_types = ['upramp', 'upramp',]
            
            # original
            basis_set = False
            # events = [event_1[:,:], event_2[:,:], event_3[:,:], event_4[:,:], event_5[:,:], event_6[:,:]]
            events = [event_1[:,:], event_2[:,:],]
            linear_model = GLM.GeneralLinearModel(input_object=pupil_epoched, event_object=events, sample_dur=1.0/new_sample_rate, new_sample_dur=1.0/new_sample_rate)
            linear_model.configure(IRF='pupil', IRF_params={'dur':3, 's':1.0/(10**26), 'n':10.1, 'tmax':0.93}, regressor_types=regressor_types, demean=True, basis_set=basis_set, normalize_sustained=False)
        
            # assumptions:
            dm = np.array(linear_model.design_matrix)
            print 'assumptions:'
            print '------------'
            for i in range(dm.shape[0]):
                for j in range(i+1, dm.shape[0]):
                    print 'reg {} x reg {}: R = {}'.format(i, j, round(np.mean(dm[i,:]*dm[j,:]),3))
        
            # execute:
            linear_model.execute()
            if basis_set:
                betas_0 = np.linalg.norm(linear_model.betas.reshape(len(regressor_types),2), axis=1)
            else:
                betas_0 = linear_model.betas
            slope, intercept, r_value, p_value, std_err = sp.stats.linregress(linear_model.predicted, linear_model.working_data_array)
            print 'R = {}'.format(r_value)
        
            x = np.linspace(-0.5,4.5,100)
            pupil_mean_low = np.mean(np.vstack([linear_model.working_data_array[int(floor(i)):int(floor(i))+100] for i in ((cue_times_epoched[ind_max_0][1:-1]*1000/downsample_rate)-10)]), axis=0)
            pupil_mean_high = np.mean(np.vstack([linear_model.working_data_array[int(floor(i)):int(floor(i))+100] for i in ((cue_times_epoched[ind_max_1][1:-1]*1000/downsample_rate)-10)]), axis=0)
            predicted_mean_low = np.mean(np.vstack([linear_model.predicted[int(floor(i)):int(floor(i))+100] for i in ((cue_times_epoched[ind_max_0][1:-1]*1000/downsample_rate)-10)]), axis=0)
            predicted_mean_high = np.mean(np.vstack([linear_model.predicted[int(floor(i)):int(floor(i))+100] for i in ((cue_times_epoched[ind_max_1][1:-1]*1000/downsample_rate)-10)]), axis=0)
            fig = plt.figure()
            plt.plot(x,pupil_mean_low, color='b', ls=':', alpha=0.5, label='measured low')
            plt.plot(x,pupil_mean_high, color='r', ls=':', alpha=0.5, label='measured high')
            plt.plot(x,predicted_mean_low, color='b', label='predicted low')
            plt.plot(x,predicted_mean_high, color='r', label='predicted high')
            plt.legend()
            plt.xlim(-0.5,4.5)
            plt.title('R = {}'.format(r_value))
            plt.xlabel('time (s)')
            fig.savefig(os.path.join(self.base_directory, 'figs', 'GLM_split_0.pdf'))
        
            # plot response locked:
            x = np.linspace(-3,2,100)
            pupil_mean_low = np.mean(np.vstack([linear_model.working_data_array[int(floor(i)):int(floor(i))+100] for i in ((choice_times_epoched[ind_max_0][1:-1]*1000/downsample_rate)-60)]), axis=0)
            pupil_mean_high = np.mean(np.vstack([linear_model.working_data_array[int(floor(i)):int(floor(i))+100] for i in ((choice_times_epoched[ind_max_1][1:-1]*1000/downsample_rate)-60)]), axis=0)
            predicted_mean_low = np.mean(np.vstack([linear_model.predicted[int(floor(i)):int(floor(i))+100] for i in ((choice_times_epoched[ind_max_0][1:-1]*1000/downsample_rate)-60)]), axis=0)
            predicted_mean_high = np.mean(np.vstack([linear_model.predicted[int(floor(i)):int(floor(i))+100] for i in ((choice_times_epoched[ind_max_1][1:-1]*1000/downsample_rate)-60)]), axis=0)
            fig = plt.figure()
            plt.plot(x,pupil_mean_low, color='b', ls=':', alpha=0.5, label='measured low')
            plt.plot(x,pupil_mean_high, color='r', ls=':', alpha=0.5, label='measured high')
            plt.plot(x,predicted_mean_low, color='b', label='predicted low')
            plt.plot(x,predicted_mean_high, color='r', label='predicted high')
            plt.legend()
            plt.xlim(-3,2)
            plt.title('R = {}'.format(r_value))
            plt.xlabel('time (s)')
            fig.savefig(os.path.join(self.base_directory, 'figs', 'GLM_split_1.pdf'))

        

        
        
class pupilAnalysesAcross(object):
    def __init__(self, subjects, experiment_name, project_directory, sample_rate_new=50):
        
        self.subjects = subjects
        self.nr_subjects = len(self.subjects)
        self.experiment_name = experiment_name
        self.project_directory = project_directory
        self.sample_rate_new = int(sample_rate_new)
        self.downsample_rate = int(1000 / sample_rate_new)
        
        parameters = []
        for s in self.subjects:
            self.base_directory = os.path.join(self.project_directory, self.experiment_name, s)
            self.hdf5_filename = os.path.join(self.base_directory, 'processed', s + '.hdf5')
            self.ho = HDFEyeOperator.HDFEyeOperator(self.hdf5_filename)
            
            try:
                parameters.append(self.ho.read_session_data('', 'parameters_joined'))
            except:
                shell()
        
        self.parameters_joined = pd.concat(parameters)
        self.omissions = np.array(self.parameters_joined['omissions']) + (np.array(self.parameters_joined['correct']) == -1)
        
        for subj in np.unique(self.parameters_joined.subject):
            print round(np.mean(np.array((self.parameters_joined[self.parameters_joined.subject == subj]['blinks_nr'] > 0))), 3)
        
        # # ceiling effect control:
        # omissions = []
        # for subj_idx in self.subjects:
        #     d = self.parameters_joined[self.parameters_joined.subject == subj_idx]
        #     for r in np.array(np.unique(d['run_nr']), dtype=int):
        #         pupil_b = np.array(d['pupil_b'])[np.array(d.run_nr == r)]
        #         pupil_d = np.array(d['pupil_d'])[np.array(d.run_nr == r)]
        #         omissions.append(pupil_d > max(pupil_b))
        # omissions = np.concatenate(omissions)
        # self.omissions = self.omissions + omissions
        #
        
        self.parameters_joined = self.parameters_joined[-self.omissions]

        # regress out RT per session:
        for subj in np.unique(self.parameters_joined.subject):
            for s in np.unique(self.parameters_joined.session[self.parameters_joined.subject == subj]):
                ind = (self.parameters_joined.subject == subj) * (self.parameters_joined.session == s)
                rt = np.array(self.parameters_joined['rt'][ind]) / 1000.0
                pupil_d = np.array(self.parameters_joined['pupil_d'][ind])
                pupil_d = myfuncs.lin_regress_resid(pupil_d, [rt]) + pupil_d.mean()
                self.parameters_joined['pupil_d'][ind] = pupil_d
        
        # # regress out time on task per session:
        # for subj in np.unique(self.parameters_joined.subject):
        #     for r in np.unique(self.parameters_joined.run[self.parameters_joined.subject == subj]):
        #         ind = (self.parameters_joined.subject == subj) * (self.parameters_joined.run == r)
        #
        #         # pupil_b = np.array(self.parameters_joined['pupil_b_lp'][ind])
        #         # trial_nr = np.arange(len(pupil_b))
        #         # pupil_b = myfuncs.exp_regress_resid(pupil_b, trial_nr)
        #
        #         pupil_b = np.array(self.parameters_joined['pupil_b'][ind])
        #         pupil_b = (pupil_b - pupil_b.mean()) / pupil_b.std()
        #
        #         self.parameters_joined['pupil_b_lp'][ind] = pupil_b
        
        self.rt = np.array(self.parameters_joined['rt'])
        self.hit = np.array(self.parameters_joined['hit'], dtype=bool)
        self.fa = np.array(self.parameters_joined['fa'], dtype=bool)
        self.miss = np.array(self.parameters_joined['miss'], dtype=bool)
        self.cr = np.array(self.parameters_joined['cr'], dtype=bool)
        self.yes = np.array(self.parameters_joined['yes'], dtype=bool)
        self.no = -np.array(self.parameters_joined['yes'], dtype=bool)
        self.run = np.array(self.parameters_joined['run'], dtype=int)
        self.session = np.array(self.parameters_joined['session'], dtype=int)
        try:
            self.present = np.array(self.parameters_joined['signal_present'], dtype=bool)
        except:
            self.present = np.array(self.parameters_joined['target_present_in_stimulus'], dtype=bool)
        self.absent = -self.present
        self.correct = np.array(self.parameters_joined['correct'], dtype=bool)
        self.error = -np.array(self.parameters_joined['correct'], dtype=bool)
        self.pupil_b = np.array(self.parameters_joined['pupil_b'])
        self.pupil_d = np.array(self.parameters_joined['pupil_d'])
        self.subj_idx = np.concatenate(np.array([np.repeat(i, sum(self.parameters_joined['subject'] == self.subjects[i])) for i in range(len(self.subjects))]))
        self.criterion = np.array([np.array(self.parameters_joined[self.parameters_joined['subject']==subj]['criterion'])[0] for subj in self.subjects])
        
        # pupil split:
        self.pupil_l_ind = []
        self.pupil_h_ind = []
        for subj_idx in self.subjects:
            d = self.parameters_joined[self.parameters_joined.subject == subj_idx]
            p_h = []
            p_l = []
            for s in np.array(np.unique(d['session']), dtype=int):
                pupil = np.array(d['pupil_d'])[np.array(d.session) == s]
                p_l.append( pupil <= np.percentile(pupil, 40) )
                p_h.append( pupil >= np.percentile(pupil, 60) )
            self.pupil_l_ind.append(np.concatenate(p_l))
            self.pupil_h_ind.append(np.concatenate(p_h))
        self.pupil_l_ind = np.concatenate(self.pupil_l_ind)
        self.pupil_h_ind = np.concatenate(self.pupil_h_ind)
        self.pupil_rest_ind = -(self.pupil_h_ind + self.pupil_l_ind)
        
        # self.pupil_l_ind = []
        # self.pupil_h_ind = []
        # for subj_idx in self.subjects:
        #     d = self.parameters_joined[self.parameters_joined.subject == subj_idx]
        #     p_h = []
        #     p_l = []
        #     for r in np.array(np.unique(d['run']), dtype=int):
        #         pupil = np.array(d['pupil_b_lp'])[np.array(d.run) ==r]
        #         p_l.append( (pupil <= np.percentile(pupil, 25)) + (pupil >= np.percentile(pupil, 75)) )
        #         p_h.append( (pupil > np.percentile(pupil, 25)) & (pupil < np.percentile(pupil, 75)) )
        #     self.pupil_l_ind.append(np.concatenate(p_l))
        #     self.pupil_h_ind.append(np.concatenate(p_h))
        # self.pupil_l_ind = np.concatenate(self.pupil_l_ind)
        # self.pupil_h_ind = np.concatenate(self.pupil_h_ind)
        # self.pupil_rest_ind = -(self.pupil_h_ind + self.pupil_l_ind)
        
        # initialize behavior operator:
        d = {
        'subj_idx' : pd.Series(self.subj_idx),
        'choice_a' : pd.Series(np.array(self.yes, dtype=int)),
        'stimulus' : pd.Series(np.array(self.present, dtype=int)),
        'rt' : pd.Series(np.array(self.rt)) / 1000.0,
        'pupil_b' : pd.Series(np.array(self.pupil_b)),
        'pupil_d' : pd.Series(np.array(self.pupil_d)),
        'pupil_high' : pd.Series(self.pupil_h_ind),
        'run' : pd.Series(np.array(self.run, dtype=int)),
        'session' : pd.Series(np.array(self.session, dtype=int)),
        }
        self.df = pd.DataFrame(d)
        self.behavior = myfuncs.behavior(self.df)
        
    def behavior_choice(self):
        
        print np.median([np.median(self.rt[(self.subj_idx ==s) & (self.yes)]) for s in np.unique(self.subj_idx)])
        print np.median([np.median(self.rt[(self.subj_idx ==s) & (self.no)]) for s in np.unique(self.subj_idx)])
        
        
        shell()
        
        # values:
        df = self.behavior.choice_fractions()
        
        # SDT fractions:
        MEANS_correct = (np.mean(df.hit), np.mean(df.cr))
        SEMS_correct = (sp.stats.sem(df.hit), sp.stats.sem(df.cr))
        MEANS_error = (np.mean(df.miss), np.mean(df.fa))
        SEMS_error = (sp.stats.sem(df.miss), sp.stats.sem(df.fa))
        N = 2
        ind = np.linspace(0,N/2,N)
        bar_width = 0.50
        fig = plt.figure(figsize=(1.25,1.75))
        ax = fig.add_subplot(111)
        for i in range(N):
            ax.bar(ind[i], height=MEANS_correct[i], width = bar_width, yerr = SEMS_correct[i], color = ['r', 'b'][i], alpha = 1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
        for i in range(N):
            ax.bar(ind[i], height=MEANS_error[i], bottom = MEANS_correct[i], width = bar_width, color = ['b', 'r'][i], alpha = 0.5, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
        ax.set_ylabel('fraction of trials',)
        plt.xticks(ind, ('signal+noise', 'noise'),)
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_SDT_fractions.pdf'))
        
        # d-prime and criterion:
        dft = df.ix[:,np.array(["d'", 'crit'])]
        dft = dft.stack().reset_index()
        dft.columns = ['subject', 'measure', 'value']
        fig = plt.figure(figsize=(1.5,2))
        ax = fig.add_subplot(111)
        sns.barplot(x='measure',  y='value', units='subject', data=dft, ci=None, ax=ax, alpha=0.5)
        sns.stripplot(x='measure', y='value', data=dft, jitter=True, size=2, edgecolor='black', linewidth=0.25, alpha=1, ax=ax)
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_SDT_measures.pdf'))
        
        
        # shell()
        
        # measures high vs low:
        behavior = myfuncs.behavior(self.df[~self.pupil_rest_ind])
        df_h = behavior.choice_fractions(split_by='pupil_high', split_target=1)
        df_l = behavior.choice_fractions(split_by='pupil_high', split_target=0)
        titles = ['rt', 'acc', "d'", 'crit', 'crit_abs', 'c_a', 'rtcv']
        ylim_max = [2.5, 1, 2, 0.6, 1, 0.8, 10]
        ylim_min = [0.0, 0, 0, 0.0, 0, 0.2, -4]
        for i, t in enumerate(titles):
            dft = pd.concat((df_h.ix[:,'{}_1'.format(t)], df_l.ix[:,'{}_0'.format(t)]), axis=1)
            dft = dft.stack().reset_index()
            dft.columns = ['subject', 'measure', 'value']
            fig = plt.figure(figsize=(1.5,2))
            ax = fig.add_subplot(111)
            sns.barplot(x='measure',  y='value', units='subject', order=['{}_0'.format(t), '{}_1'.format(t)], data=dft, ci=None, alpha=0.5, palette=['b', 'r'], ax=ax)
            sns.stripplot(x='measure', y='value', order=['{}_0'.format(t), '{}_1'.format(t)], data=dft, jitter=False, size=2, edgecolor='black', linewidth=0.25, alpha=1, palette=['b', 'r'], ax=ax)
            values = np.vstack((dft[dft['measure'] == '{}_0'.format(t)].value, dft[dft['measure'] == '{}_1'.format(t)].value))
            ax.plot(np.array([0, 1]), values, color='black', lw=0.5, alpha=0.5)
            ax.set_title('p = {}'.format(round(myfuncs.permutationTest(values[0,:], values[1,:], paired=True)[1],3)))
            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_SDT_measures_{}.pdf'.format(t)))
        
        columns = [c.split('_1')[0] for c in df_h.columns]
        df_h.columns = columns
        columns = [c.split('_0')[0] for c in df_l.columns]
        df_l.columns = columns
        df = pd.concat((df_l, df_h))
        df['TPR'] = np.array(np.concatenate((np.zeros(len(self.subjects)), np.ones(len(self.subjects)))), dtype=int)
        df = df.ix[:, ["d'", 'crit', 'c_a', 'TPR']]
        df.columns = ["d'", 'crit', 'yes_frac', 'TPR']
        df.to_csv(os.path.join(self.project_directory, 'figures', 'fig2A_source_data.csv'))
        
    def behavior_normalized(self, prepare=False):
        
        self.df = self.df[~self.pupil_rest_ind]
        split_ind = np.array(eval('self.df.' + 'pupil_high') == 1, dtype=bool)
        
        if prepare:
            perms = 1000
            inds = []
            for r in xrange(perms):
                print '{} out of {}'.format(r, perms)
            
                # compute random indices equalizing signal present trials:
                signal_presence_ind = []
                imbalances_present = []
                imbalances_absent = []
                for s in xrange(self.nr_subjects):
                    del_ind = np.ones(sum(self.df.subj_idx==s), dtype=bool)
                    subj_ind = np.array(self.df.subj_idx==s)
                    stim_ind = np.array(self.df.stimulus == 1)
                    present_h = sum(stim_ind * subj_ind * split_ind)
                    present_l = sum(stim_ind * subj_ind * -split_ind)
                    absent_h = sum(-stim_ind * subj_ind * split_ind)
                    absent_l = sum(-stim_ind * subj_ind * -split_ind)
                    lowest = min(present_h, present_l, absent_h, absent_l)
                    if present_h > lowest:
                        del_trials = np.random.choice( np.where((split_ind*stim_ind)[subj_ind])[0], present_h-lowest, replace=False)
                        del_ind[del_trials] = False
                    if present_l > lowest:
                        del_trials = np.random.choice( np.where((-split_ind*stim_ind)[subj_ind])[0], present_l-lowest, replace=False)
                        del_ind[del_trials] = False
                    if absent_h > lowest:
                        del_trials = np.random.choice( np.where((split_ind*-stim_ind)[subj_ind])[0], absent_h-lowest, replace=False)
                        del_ind[del_trials] = False
                    if absent_l > lowest:
                        del_trials = np.random.choice( np.where((-split_ind*-stim_ind)[subj_ind])[0], absent_l-lowest, replace=False)
                        del_ind[del_trials] = False
                    signal_presence_ind.append(del_ind)
                inds.append( np.array(np.concatenate(signal_presence_ind), dtype=bool) )
            np.save(os.path.join(self.project_directory, 'figures', 'trial_balance_indices.npy'), inds)
        
        else:
            
            indds = np.load(os.path.join(self.project_directory, 'figures', 'trial_balance_indices.npy'))
            perms = 1000
            yes_high = np.zeros((perms, self.nr_subjects))
            yes_low = np.zeros((perms, self.nr_subjects))
            correct_high = np.zeros((perms, self.nr_subjects))
            correct_low = np.zeros((perms, self.nr_subjects))
            rt_high = np.zeros((perms, self.nr_subjects))
            rt_low = np.zeros((perms, self.nr_subjects))
            c_high = np.zeros((perms, self.nr_subjects))
            c_low = np.zeros((perms, self.nr_subjects))
            d_high = np.zeros((perms, self.nr_subjects))
            d_low = np.zeros((perms, self.nr_subjects))
            for r in xrange(perms):
                print '{} out of {}'.format(r, perms)
            
                # update data:
                data = self.df[indds[r,:]]
                split_ind2 = np.array(eval('data.' + 'pupil_high') == 1, dtype=bool)
            
                # # check number of trials:
                # for s in xrange(self.nr_subjects):
                #     print
                #     print sum(np.array(data[(data.subj_idx==s) & split_ind2].stimulus, dtype=bool))
                #     print sum(np.array(data[(data.subj_idx==s) & ~split_ind2].stimulus, dtype=bool))
                #     print sum(~np.array(data[(data.subj_idx==s) & split_ind2].stimulus, dtype=bool))
                #     print sum(~np.array(data[(data.subj_idx==s) & ~split_ind2].stimulus, dtype=bool))
                #     print sum(np.array(data[(data.subj_idx==s) & split_ind2].stimulus, dtype=bool)) + sum(np.array(data[(data.subj_idx==s)&-split_ind2].stimulus, dtype=bool)) + sum(-np.array(data[(data.subj_idx==s)&split_ind2].stimulus, dtype=bool)) + sum(-np.array(data[(data.subj_idx==s)&-split_ind2].stimulus, dtype=bool))
            
                # # compute SDT measures:
                # behavior = myfuncs.behavior(data)
                # df_h = behavior.choice_fractions(split_by='pupil_high', split_target=1)
                # rt_high[r,:] = df_h['rt_1']
                # correct_high[r,:] = df_h['acc_1']
                # d_high[r,:] = df_h["d'_1"]
                # c_high[r,:] = df_h['crit_1']
                # yes_high[r,:] = df_h['c_a_1']
                # df_l = behavior.choice_fractions(split_by='pupil_high', split_target=0)
                # rt_low[r,:] = df_l['rt_0']
                # correct_low[r,:] = df_l['acc_0']
                # d_low[r,:] = df_l["d'_0"]
                # c_low[r,:] = df_l['crit_0']
                # yes_low[r,:] = df_l['c_a_0']
            
                # compute fractions:
                yes_high[r,:] = np.array([sum((data.choice_a==1) & (data.subj_idx==i) & split_ind2) / float(sum((data.subj_idx==i) & split_ind2)) for i in range(self.nr_subjects)])
                yes_low[r,:] = np.array([sum((data.choice_a==1) & (data.subj_idx==i) & -split_ind2) / float(sum((data.subj_idx==i) & -split_ind2)) for i in range(self.nr_subjects)])
                
            # # correct for inf's
            # d_high[d_high == np.inf] = np.NaN
            # d_low[d_low == np.inf] = np.NaN
            # c_high[c_high == np.inf] = np.NaN
            # c_low[c_low == np.inf] = np.NaN
            #
            # # collapse:
            # rt_high_c = rt_high.mean(axis=0)
            # rt_low_c = rt_low.mean(axis=0)
            # correct_high_c = correct_high.mean(axis=0)
            # correct_low_c = correct_low.mean(axis=0)
            # d_high_c = bn.nanmean(d_high, axis=0)
            # d_low_c = bn.nanmean(d_low, axis=0)
            # c_high_c = bn.nanmean(c_high, axis=0)
            # c_low_c = bn.nanmean(c_low, axis=0)
            yes_high_c = yes_high.mean(axis=0)
            yes_low_c = yes_low.mean(axis=0)
        
            # measures high vs low:
            dfh = pd.DataFrame(yes_high_c, columns=['c_a_1'])
            dfl = pd.DataFrame(yes_low_c, columns=['c_a_0'])
            titles = ['c_a',]
            ylim_max = [0.8,]
            ylim_min = [0.2,]
            for i, t in enumerate(titles):
                dft = pd.concat((dfh.ix[:,'{}_1'.format(t)], dfl.ix[:,'{}_0'.format(t)]), axis=1)
                dft = dft.stack().reset_index()
                dft.columns = ['subject', 'measure', 'value']
                fig = plt.figure(figsize=(1.5,2))
                ax = fig.add_subplot(111)
                sns.barplot(x='measure',  y='value', units='subject', data=dft, ci=None, alpha=0.5, palette=['r', 'b'], ax=ax)
                sns.stripplot(x='measure', y='value', data=dft, jitter=False, size=2, edgecolor='black', linewidth=0.25, alpha=1, palette=['r', 'b'], ax=ax)
                values = np.vstack((dft[dft['measure'] == '{}_1'.format(t)].value, dft[dft['measure'] == '{}_0'.format(t)].value))
                ax.plot(np.array([0, 1]), values, color='black', lw=0.5, alpha=0.5)
                ax.set_title('p = {}'.format(round(myfuncs.permutationTest(values[0,:], values[1,:], paired=True)[1],3)))
                sns.despine(offset=10, trim=True)
                plt.tight_layout()
                fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_SDT_measures_normalized_{}.pdf'.format(t)))
            
            # correlate to DDM:
            params = pd.read_csv(os.path.join(self.project_directory, 'DDM_params.csv'))
            params = params[np.array(params['subject'] >= 22)]
            delta_dc = np.array(params['dc'][params['pupil']==1]) - np.array(params['dc'][params['pupil']==0])
            delta_sp = np.array(params['z'][params['pupil']==1]) - np.array(params['z'][params['pupil']==0])
            delta_c = np.array(dfh['c_a_1']) - np.array(dfl['c_a_0'])

            fig = plt.figure(figsize=(2,2))
            ax = fig.add_subplot(111)
            myfuncs.correlation_plot(delta_dc, delta_c, line=True, ax=ax)
            plt.ylabel('Change in fraction yes-choices')
            plt.xlabel('Change in drift criterion')
            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(self.project_directory, 'figures', 'criterion_dc.pdf'))

            fig = plt.figure(figsize=(2,2))
            ax = fig.add_subplot(111)
            myfuncs.correlation_plot(delta_sp, delta_c, line=True, ax=ax)
            plt.ylabel('Change in fraction yes-choices')
            plt.xlabel('Change in starting point')
            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(self.project_directory, 'figures', 'criterion_starting_point.pdf'))
            
            # make dataframe to save:
            behavior = myfuncs.behavior(self.df)
            df_h = behavior.choice_fractions(split_by='pupil_high', split_target=1)
            df_l = behavior.choice_fractions(split_by='pupil_high', split_target=0)
            columns = [c.split('_1')[0] for c in df_h.columns]
            df_h.columns = columns
            columns = [c.split('_0')[0] for c in df_l.columns]
            df_l.columns = columns
            df = pd.concat((df_l, df_h))
            df['TPR'] = np.array(np.concatenate((np.zeros(len(self.subjects)), np.ones(len(self.subjects)))), dtype=int)
            df = df.ix[:, ["d'", 'crit', 'c_a', 'TPR']]
            df.columns = ["d'", 'crit', 'yes_frac', 'TPR']
            df.ix[:,'yes_frac'] = np.concatenate((dfl['c_a_0'], dfh['c_a_1']))
            df.to_csv(os.path.join(self.project_directory, 'figures', 'fig2A_source_data.csv'))
    
    def autocorrelation(self):
        
        shell()
        
        data_split = self.df.copy()
        lags = 20
        auto_cor_f = np.zeros((len(self.subjects), lags+1))
        for i in range(len(self.subjects)):
            auto_cor_f_subj = []
            for r in np.unique(data_split[self.subj_idx == i].run):
                pupil = np.array(data_split['pupil_b'])[(self.subj_idx == i) & (self.run ==r)]
                cor_lags = np.zeros(lags+1)
                cor_lags[0] = 1
                for l, lag in enumerate(np.arange(lags)+1):
                    cor_lags[l+1] = sp.stats.pearsonr(pupil[:-lag], pupil[lag:])[0]
                auto_cor_f_subj.append(cor_lags)
            auto_cor_f[i,:] = np.vstack(auto_cor_f_subj).mean(axis=0)
        fig = plt.figure(figsize=(3,2))
        ax = fig.add_subplot(111)
        sns.tsplot(auto_cor_f, time=np.arange(lags+1), value='autocorrelation', err_style='ci_band', lw=1, ls='-', ax=ax)
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.project_directory, 'figures', 'autocorrelation.pdf'))
    
    
    def behavior_variability(self):
        
        d = {
        'subj_idx' : pd.Series(self.subj_idx),
        'correct' : pd.Series(np.array(self.correct, dtype=int)),
        'choice' : pd.Series(np.array(self.yes, dtype=int)),
        'stimulus' : pd.Series(np.array(self.present, dtype=int)),
        'rt' : pd.Series(np.array(self.rt)) / 1000.0,
        'pupil_b' : pd.Series(np.array(self.pupil_b)),
        'pupil_d' : pd.Series(np.array(self.pupil_d)),
        'pupil_t' : pd.Series(np.array(self.pupil_t)),
        'pupil_high' : pd.Series(self.pupil_h_ind),
        }
        self.df = pd.DataFrame(d)
        # self.df = self.df[(self.df.correct != -1) & -self.GLM_split_rest]
        self.df = self.df[(self.df.correct != -1)]
        self.behavior = myfuncs.behavior(self.df)
        choice_var_0, choice_var_1, rt_var_0, rt_var_1 = self.behavior.behavior_variability()
        
        d = {
        'subj_idx' : pd.Series(self.subj_idx),
        'correct' : pd.Series(np.array(self.correct, dtype=int)),
        'choice' : pd.Series(np.array(self.yes, dtype=int)),
        'stimulus' : pd.Series(np.array(self.present, dtype=int)),
        'rt' : pd.Series(np.array(self.rt)) / 1000.0,
        'pupil_b' : pd.Series(np.array(self.pupil_b)),
        'pupil_d' : pd.Series(np.array(self.pupil_d)),
        'pupil_t' : pd.Series(np.array(self.pupil_t)),
        'pupil_high' : pd.Series(self.pupil_h_ind),
        }
        self.df = pd.DataFrame(d)
        # self.df = self.df[(self.df.correct != -1) & -self.GLM_split_rest]
        self.df = self.df[(self.df.correct != -1) & -self.pupil_rest_ind]
        self.behavior = myfuncs.behavior(self.df)
        choice_var_0h, choice_var_1h, rt_var_0h, rt_var_1h = self.behavior.behavior_variability(split_by='pupil_high', split_target=1)
        choice_var_0l, choice_var_1l, rt_var_0l, rt_var_1l = self.behavior.behavior_variability(split_by='pupil_high', split_target=0)
        
        choice_var_p_0 = (choice_var_0h + choice_var_0l) / 2.0
        choice_var_p_1 = (choice_var_1h + choice_var_1l) / 2.0
        rt_var_p_0 = (rt_var_0h + rt_var_0l) / 2.0
        rt_var_p_1 = (rt_var_1h + rt_var_1l) / 2.0
        
        values = [choice_var_0, choice_var_0h, choice_var_0l, choice_var_1, choice_var_1h, choice_var_1l]
        MEANS = np.array([v.mean() for v in values]) 
        SEMS = np.array([sp.stats.sem(v) for v in values]) 
        ind = np.arange(0,len(MEANS))
        bar_width = 0.90
        fig = plt.figure(figsize=(3,2))
        ax = fig.add_subplot(111)
        my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}
        for i in range(len(MEANS)):
            ax.bar(ind[i], MEANS[i], width = bar_width, yerr=SEMS[i], color = ['k','r','b','k','r','b'][i], alpha = [0.5,0.5,0.5,1,1,1][i], align = 'center', edgecolor = 'k', linewidth = 0, capsize = 0, ecolor = 'k', error_kw={'elinewidth':0.5,})
        ax.tick_params(axis='y', which='major', labelsize=6)
        ax.set_xticks(ind)
        ax.set_ylabel('Choice variability (s.d.)', size=7)
        sns.despine(offset=10, trim=True)
        ax.set_xticklabels(['absent', 'absent', 'absent', 'present', 'present', 'present'], rotation=45, size=7)
        plt.tight_layout()
        fig.savefig(os.path.join(self.project_directory, 'figures', 'variability_choice.pdf'))
        
        values = [rt_var_0, rt_var_0h, rt_var_0l, rt_var_1, rt_var_1h, rt_var_1l]
        MEANS = np.array([v.mean() for v in values]) 
        SEMS = np.array([sp.stats.sem(v) for v in values]) 
        ind = np.arange(0,len(MEANS))
        bar_width = 0.90
        fig = plt.figure(figsize=(3,2))
        ax = fig.add_subplot(111)
        my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}
        for i in range(len(MEANS)):
            ax.bar(ind[i], MEANS[i], width = bar_width, yerr=SEMS[i], color = ['k','r','b','k','r','b'][i], alpha = [0.5,0.5,0.5,1,1,1][i], align = 'center', edgecolor = 'k', linewidth = 0, capsize = 0, ecolor = 'k', error_kw={'elinewidth':0.5,})
        ax.tick_params(axis='y', which='major', labelsize=6)
        ax.set_xticks(ind)
        ax.set_ylabel('RT variability (s.d.)', size=7)
        sns.despine(offset=10, trim=True)
        ax.set_xticklabels(['absent', 'absent', 'absent', 'present', 'present', 'present'], rotation=45, size=7)
        plt.tight_layout()
        fig.savefig(os.path.join(self.project_directory, 'figures', 'variability_rt.pdf'))
        
        # shell()
        
        import statsmodels.formula.api as sm
        model = 'logit'
        
        xx = []
        mm = []
        aic_1 = []
        bic_1 = []
        aic_2 = []
        bic_2 = []
        
        var_1 = []
        var_2 = []
        
        accuracy_1 = []
        accuracy_2 = []
        
        for s in range(len(self.subjects)):
            
            X = np.array(self.present, dtype=int)[self.subj_idx == s]
            Y = np.array(self.yes, dtype=int)[self.subj_idx == s]
            M = self.pupil_d[self.subj_idx == s]
        
            # X = (X-X.mean()) / X.std()
            # M = (M-M.mean()) / M.std()
            # XM = X*M
            # if model == 'ols':
            #     Y = (Y-Y.mean()) / Y.std()
    
            d = {'X' : pd.Series(X),
                'Y' : pd.Series(Y),
                'M' : pd.Series(M),
                # 'XM' : pd.Series(XM),
                }
            data = pd.DataFrame(d)
    
            # direct:
            f = 'Y ~ X'
            if model == 'ols':
                m = sm.ols(formula=f, data=data)
            if model == 'logit':
                m = sm.logit(formula=f, data=data)
            fit = m.fit()
            
            predict = fit.predict(data)
            predict[predict > 0.5] = 1
            predict[predict < 0.5] = 0
            accuracy_1.append(np.mean(predict == Y))
            
            # with modulation:
            f2 = 'Y ~ X + M'
            if model == 'ols':
                m2 = sm.ols(formula=f2, data=data)
            if model == 'logit':
                m2 = sm.logit(formula=f2, data=data)
            fit2 = m2.fit()
            
            predict = fit2.predict(data)
            predict[predict > 0.5] = 1
            predict[predict < 0.5] = 0
            accuracy_2.append(np.mean(predict == Y))
            
            x = fit.params['X']
            x_1 = fit2.params['X']
            m_1 = fit2.params['M']
            
            aic_1.append(fit.aic)
            bic_1.append(fit.bic)
            aic_2.append(fit2.aic)
            bic_2.append(fit2.bic)
            
            # xx.append(np.exp(x_1) / (1 + np.exp(x_1)))
            # mm.append(np.exp(m_1) / (1 + np.exp(m_1)))
            
            xx.append(x_1)
            mm.append(m_1)
            
            var_1.append(fit.resid_response.std())
            var_2.append(fit2.resid_response.std())
            
        xx = np.array(xx)
        mm = np.array(mm)
        
        var_1 = np.array(var_1)
        var_2 = np.array(var_2)
        
        aic_1 = np.array(aic_1)
        aic_2 = np.array(aic_2)
        bic_1 = np.array(bic_1)
        bic_2 = np.array(bic_2)
        
        print myfuncs.permutationTest(xx, np.zeros(len(mm)), paired=True)
        print myfuncs.permutationTest(mm, np.zeros(len(mm)), paired=True)
        
        delta_dc = np.load(os.path.join(self.project_directory, 'data', 'across', 'params_fMRI_d1.npy'))[2,:] - np.load(os.path.join(self.project_directory, 'data', 'across', 'params_fMRI_d1.npy'))[3,:]
        delta_sp = np.load(os.path.join(self.project_directory, 'data', 'across', 'params_fMRI_d1.npy'))[0,:] - np.load(os.path.join(self.project_directory, 'data', 'across', 'params_fMRI_d1.npy'))[1,:]
        
        criterion = np.concatenate((self.criterion, np.array([ 0.05257109,  0.66486615,  0.46866103, -0.0849936 ,  0.43264189,
                                0.36767314,  0.32851001,  0.40868768, -0.0022005 , -0.153563  ,
                                0.10212666,  0.37943029,  0.08221574, -0.22190617, -0.25067594,
                                0.28810509,  0.58158442])))
        
        delta_c = np.concatenate((np.array([-0.34463984, -0.0832034 , -0.18009341, -0.1015964 , -0.06295312,
                               -0.14365612, -0.32182599, -0.25757145, -0.0901839 , -0.28132948,
                                0.04149885, -0.12125546, -0.37666829, -0.359104  ]), np.array([ 0.42649073,  0.0581071 , -0.48621555, -0.18705913, -0.04379978,
                               -0.48553687, -0.11126099, -0.81116426, -0.14268033, -0.06148672,
                                0.10480672, -0.40472821, -0.04983578, -0.10098832,  0.1545283 ,
                               -0.06870388, -0.26855592])))
        
        delta_drift_criterion = np.concatenate((delta_dc, np.array([-0.45839162,  0.04583137,  0.48624604,  0.15292648,  0.20132418,
                                0.41112521,  0.14136415,  0.71120639,  0.24409918,  0.08471458,
                                -0.0193405 ,  0.34359647,  0.14710836,  0.1026163 , -0.09620626,
                                0.18772292,  0.21276053])))
        
        mm_pnas = np.array([-0.06283157, -0.01155205,  0.06588889,  0.02029722,  0.03028909,
                                0.06778479,  0.02658199,  0.113548  ,  0.03140745, -0.00282024,
                                -0.00892159,  0.06291905,  0.00913931,  0.01979994, -0.02677293,
                                0.00910799,  0.06337524])
        mmm = np.concatenate((mm, mm_pnas))
        
        
        fig = plt.figure(figsize=(2,2))
        ax = fig.add_subplot(111)
        myfuncs.correlation_plot(criterion, mmm, ax=ax, line=True)
        plt.xlabel('Criterion')
        plt.ylabel('TPR beta (a.u.)')
        plt.tight_layout()
        fig.savefig(os.path.join(self.project_directory, 'figures', 'TPR-choice_beta.pdf'))
        
        
        MEANS = (mm.mean(), mm_pnas.mean())
        SEMS = (sp.stats.sem(mm), sp.stats.sem(mm_pnas))
        N = 2
        ind = np.linspace(0,N/2,N)
        bar_width = 0.50
        fig = plt.figure(figsize=(1.25,1.75))
        ax = fig.add_subplot(111)
        for i in range(N):
            ax.bar(ind[i], height=MEANS[i], width = bar_width, yerr = SEMS[i], color = 'k', alpha = 1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
        # ax.set_title('N={}'.format(self.nr_subjects), size=8)
        ax.set_xticks( (ind) )
        ax.set_xticklabels( ("d'", 'c') )
        plt.gca().spines["bottom"].set_linewidth(.5)
        plt.gca().spines["left"].set_linewidth(.5)
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.project_directory, 'figures', 'TPR-choice_beta_1.pdf'))
        
        
        
        
        
        # from sklearn.linear_model import LogisticRegression
        # from sklearn.svm import SVC
        # C = 1.0
        # classifiers = {    'L1 logistic': LogisticRegression(C=C, penalty='l1'),
        #                 'Linear SVC': SVC(C=C, probability=True, random_state=0),
        #                 # 'Linear SVC': SVC(kernel='linear', C=C, probability=True, random_state=0),
        #                 }
        # n_classifiers = len(classifiers)
        # for s in range(len(self.subjects)):
        #
        #     print
        #
        #     x = np.array(self.present, dtype=int)[self.subj_idx == s]
        #     y = np.array(self.yes, dtype=int)[self.subj_idx == s]
        #     m = self.pupil_d[self.subj_idx == s]
        #
        #     X = np.vstack((x,x)).T
        #     XX = np.vstack((x,m)).T
        #
        #     for index, (name, classifier) in enumerate(classifiers.items()):
        #
        #         y_pred = np.zeros(y.size)
        #         for t in range(y.size):
        #             trials = np.ones(y.size, dtype=bool)
        #             trials[t] = False
        #             classifier.fit(X[trials,:], y[trials])
        #             y_pred[t] = classifier.predict(X[t,:])
        #         classif_rate = np.mean(y_pred.ravel() == y.ravel()) * 100
        #         print("classif_rate for %s : %f " % (name, classif_rate))
        #
        #         y_pred = np.zeros(y.size)
        #         for t in range(y.size):
        #             trials = np.ones(y.size, dtype=bool)
        #             trials[t] = False
        #             classifier.fit(XX[trials,:], y[trials])
        #             y_pred[t] = classifier.predict(XX[t,:])
        #         classif_rate = np.mean(y_pred.ravel() == y.ravel()) * 100
        #         print("classif_rate for %s : %f " % (name, classif_rate))
        
    def behavior_rt_kde(self):
        
        # RT histograms:
        # --------------

        x_grid = [0, 4, 100]
        c0_pdf, c1_pdf, c_correct_pdf, c_error_pdf, c0_correct_pdf, c0_error_pdf, c1_correct_pdf, c1_error_pdf = self.behavior.rt_kernel_densities(x_grid=x_grid, bandwidth=0.2)

        yes = np.vstack(c1_pdf)
        no = np.vstack(c0_pdf)
        correct = np.vstack(c_correct_pdf)
        error = np.vstack(c_error_pdf)
        cr = np.vstack(c0_correct_pdf)
        miss = np.vstack(c0_error_pdf)
        hit = np.vstack(c1_correct_pdf)
        fa = np.vstack(c1_error_pdf)

        step = pd.Series(np.linspace(x_grid[0], x_grid[1], x_grid[2]), name='RT (s)')

        # # Make the plt.plot
        # fig = plt.figure(figsize=(2, 3))
        # ax = plt.subplot(211)
        # conditions = pd.Series(['hits'], name='trial type')
        # sns.tsplot(hit, time=step, condition=conditions, value='kde', color='red', ci=66, lw=1, ls='-', ax=ax)
        # conditions = pd.Series(['miss'], name='trial type')
        # sns.tsplot(miss, time=step, condition=conditions, value='kde', color='blue', alpha=0.5, ci=66, lw=1, ls='-', ax=ax)
        # ax.axvline(np.mean([np.median(self.df.rt[(self.df.choice==1) & (self.df.stimulus==1) & (self.df.subj_idx==i)]) for i in range(self.nr_subjects)]), color='r', linestyle='--', lw=1)
        # ax.axvline(np.mean([np.median(self.df.rt[(self.df.choice==0) & (self.df.stimulus==1) & (self.df.subj_idx==i)]) for i in range(self.nr_subjects)]), color='b', linestyle='--', lw=1)
        # ax = plt.subplot(212)
        # conditions = pd.Series(['cr'], name='trial type')
        # sns.tsplot(cr, time=step, condition=conditions, value='kde', color='blue', ci=66, lw=1, ls='-', ax=ax)
        # conditions = pd.Series(['fa'], name='trial type')
        # sns.tsplot(fa, time=step, condition=conditions, value='kde', color='red', alpha=0.5, ci=66, lw=1, ls='-', ax=ax)
        # ax.axvline(np.mean([np.median(self.df.rt[(self.df.choice==1) & (self.df.stimulus==0) & (self.df.subj_idx==i)]) for i in range(self.nr_subjects)]), color='r', linestyle='--', lw=1)
        # ax.axvline(np.mean([np.median(self.df.rt[(self.df.choice==0) & (self.df.stimulus==0) & (self.df.subj_idx==i)]) for i in range(self.nr_subjects)]), color='b', linestyle='--', lw=1)
        # sns.despine(offset=10, trim=True)
        # plt.tight_layout()
        # fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_rt_hists.pdf'))
        
        # Make the plt.plot
        fig = plt.figure(figsize=(2, 1.5))
        ax = plt.subplot(111)
        conditions = pd.Series(['yes'], name='trial type')
        sns.tsplot(yes, time=step, condition=conditions, value='kde', color='red', ci=66, lw=1, ls='-', ax=ax)
        conditions = pd.Series(['no'], name='trial type')
        sns.tsplot(no, time=-step, condition=conditions, value='kde', color='blue', alpha=0.5, ci=66, lw=1, ls='-', ax=ax)
        # ax.axvline(np.mean([np.median(self.df.rt[(self.df.choice==1) & (self.df.subj_idx==i)]) for i in range(self.nr_subjects)]), color='r', linestyle='--', lw=1)
        # ax.axvline(-np.mean([np.median(self.df.rt[(self.df.choice==0) & (self.df.subj_idx==i)]) for i in range(self.nr_subjects)]), color='b', linestyle='--', lw=1)
        ax.set_xlim(-4,4)
        ax.legend()
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_rt_hists.pdf'))
        
        
        
        # shell()
        
        fig = plt.figure(figsize=(2,1.5))
        ax = plt.subplot(1,1,1)
        
        for j in range(2):
            
            
            x_grid = [0, 4, 100]
            c0_pdf, c1_pdf, c_correct_pdf, c_error_pdf, c0_correct_pdf, c0_error_pdf, c1_correct_pdf, c1_error_pdf = self.behavior.rt_kernel_densities(x_grid=x_grid, bandwidth=0.2, split_by='pupil_high', split_target=j)
            
            yes = np.vstack(c1_pdf)
            no = np.vstack(c0_pdf)
            correct = np.vstack(c_correct_pdf)
            error = np.vstack(c_error_pdf)
            cr = np.vstack(c0_correct_pdf)
            miss = np.vstack(c0_error_pdf)
            hit = np.vstack(c1_correct_pdf)
            fa = np.vstack(c1_error_pdf)
            
            conditions = pd.Series(['yes'], name='trial type')
            sns.tsplot(yes, time=step, condition=conditions, value='kde', color=['blue', 'red'][j], ci=66, lw=1, ls='-', ax=ax)
            conditions = pd.Series(['no'], name='trial type')
            sns.tsplot(no, time=-step, condition=conditions, value='kde', color=['blue', 'red'][j], ci=66, lw=1, ls='-', ax=ax)
            x = [-np.mean([np.mean(self.df.rt[(self.df.choice==0) & (self.df.subj_idx==i) & (self.df.pupil_high==j)]) for i in range(self.nr_subjects)]), np.mean([np.mean(self.df.rt[(self.df.choice==1) & (self.df.subj_idx==i) & (self.df.pupil_high==j)]) for i in range(self.nr_subjects)])]
            x_err = [np.std([np.mean(self.df.rt[(self.df.choice==0) & (self.df.subj_idx==i) & (self.df.pupil_high==j)]) for i in range(self.nr_subjects)]), np.std([np.mean(self.df.rt[(self.df.choice==1) & (self.df.subj_idx==i) & (self.df.pupil_high==j)]) for i in range(self.nr_subjects)])]
            ax.errorbar(x=x, y=[[0.1,0.1],[0,0]][j], xerr=x_err, fmt='o', markersize=2, color=['r','b'][j], elinewidth=0.5, )
            ax.set_xlim(-4,4)
            ax.legend()
            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_rt_hists_pupil.pdf'))
            
            
        
            
        x_grid = [0, 4, 100]
        c0_pdf_low, c1_pdf_low, c_correct_pdf, c_error_pdf, c0_correct_pdf, c0_error_pdf, c1_correct_pdf, c1_error_pdf = self.behavior.rt_kernel_densities(x_grid=x_grid, bandwidth=0.15, split_by='pupil_high', split_target=0)
        c0_pdf_high, c1_pdf_high, c_correct_pdf, c_error_pdf, c0_correct_pdf, c0_error_pdf, c1_correct_pdf, c1_error_pdf = self.behavior.rt_kernel_densities(x_grid=x_grid, bandwidth=0.15, split_by='pupil_high', split_target=1)
    
        yes_high = np.vstack(c1_pdf_high)
        no_high = np.vstack(c0_pdf_high)
        yes_low = np.vstack(c1_pdf_low)
        no_low = np.vstack(c0_pdf_low)
        
        fig = plt.figure(figsize=(4,1.5))
        ax = plt.subplot(1,2,1)
        conditions = pd.Series(['high'], name='trial type')
        sns.tsplot(yes_high, time=step, condition=conditions, value='kde', color='r', ci=66, lw=1, ls='-', ax=ax)
        conditions = pd.Series(['low'], name='trial type')
        sns.tsplot(yes_low, time=step, condition=conditions, value='kde', color='b', ci=66, lw=1, ls='-', ax=ax)
        # x = [-np.mean([np.mean(self.df.rt[(self.df.choice==0) & (self.df.subj_idx==i) & (self.df.pupil_high==j)]) for i in range(self.nr_subjects)]), np.mean([np.mean(self.df.rt[(self.df.choice==1) & (self.df.subj_idx==i) & (self.df.pupil_high==j)]) for i in range(self.nr_subjects)])]
        # x_err = [np.std([np.mean(self.df.rt[(self.df.choice==0) & (self.df.subj_idx==i) & (self.df.pupil_high==j)]) for i in range(self.nr_subjects)]), np.std([np.mean(self.df.rt[(self.df.choice==1) & (self.df.subj_idx==i) & (self.df.pupil_high==j)]) for i in range(self.nr_subjects)])]
        # ax.errorbar(x=x, y=[[0.1,0.1],[0,0]][j], xerr=x_err, fmt='o', markersize=2, color=['r','b'][j], elinewidth=0.5, )
        ax.set_xlim(0,4)
        ax.set_ylim(0,0.025)
        ax.set_title("Yes")
        ax.legend()
        
        ax = plt.subplot(1,2,2)
        conditions = pd.Series(['high'], name='trial type')
        sns.tsplot(no_high, time=step, condition=conditions, value='kde', color='r', ci=66, lw=1, ls='-', ax=ax)
        conditions = pd.Series(['low'], name='trial type')
        sns.tsplot(no_low, time=step, condition=conditions, value='kde', color='b', ci=66, lw=1, ls='-', ax=ax)
        # x = [-np.mean([np.mean(self.df.rt[(self.df.choice==0) & (self.df.subj_idx==i) & (self.df.pupil_high==j)]) for i in range(self.nr_subjects)]), np.mean([np.mean(self.df.rt[(self.df.choice==1) & (self.df.subj_idx==i) & (self.df.pupil_high==j)]) for i in range(self.nr_subjects)])]
        # x_err = [np.std([np.mean(self.df.rt[(self.df.choice==0) & (self.df.subj_idx==i) & (self.df.pupil_high==j)]) for i in range(self.nr_subjects)]), np.std([np.mean(self.df.rt[(self.df.choice==1) & (self.df.subj_idx==i) & (self.df.pupil_high==j)]) for i in range(self.nr_subjects)])]
        # ax.errorbar(x=x, y=[[0.1,0.1],[0,0]][j], xerr=x_err, fmt='o', markersize=2, color=['r','b'][j], elinewidth=0.5, )
        ax.set_xlim(0,4)
        ax.set_ylim(0,0.025)
        ax.set_title("No")
        ax.legend()
        
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_rt_hists_pupil_{}.pdf'.format(j)))
        
        
            
        # shell()
            
            # # RT histograms:
            # # --------------
            #
            #
            # yes = np.vstack(c0_pdf)
            # no = np.vstack(c1_pdf)
            # correct = np.vstack(c_correct_pdf)
            # error = np.vstack(c_error_pdf)
            # cr = np.vstack(c0_correct_pdf)
            # miss = np.vstack(c0_error_pdf)
            # hit = np.vstack(c1_correct_pdf)
            # fa = np.vstack(c1_error_pdf)
            #
            # step = pd.Series(np.linspace(x_grid[0], x_grid[1], x_grid[2]), name='rt (s)')
            #
            # # Make the plt.plot
            # fig = plt.figure(figsize=(2,3))
            # ax = plt.subplot(211)
            # conditions = pd.Series(['hits'], name='trial type')
            # sns.tsplot(hit, time=step, condition=conditions, value='kde', color='red', ci=66, lw=1, ls='-', ax=ax)
            # conditions = pd.Series(['miss'], name='trial type')
            # sns.tsplot(miss, time=step, condition=conditions, value='kde', color='blue', alpha=0.5, ci=66, lw=1, ls='-', ax=ax)
            # ax.axvline(np.mean([np.median(self.df.rt[(self.df.choice==1) & (self.df.stimulus==1) & (self.df.subj_idx==i) & (self.df.pupil_high==j)]) for i in range(self.nr_subjects)]), color='r', linestyle='--', lw=1)
            # ax.axvline(np.mean([np.median(self.df.rt[(self.df.choice==0) & (self.df.stimulus==1) & (self.df.subj_idx==i) & (self.df.pupil_high==j)]) for i in range(self.nr_subjects)]), color='b', linestyle='--', lw=1)
            # ax = plt.subplot(212)
            # conditions = pd.Series(['cr'], name='trial type')
            # sns.tsplot(cr, time=step, condition=conditions, value='kde', color='blue', ci=66, lw=1, ls='-', ax=ax)
            # conditions = pd.Series(['fa'], name='trial type')
            # sns.tsplot(fa, time=step, condition=conditions, value='kde', color='red', alpha=0.5, ci=66, lw=1, ls='-', ax=ax)
            # ax.axvline(np.mean([np.median(self.df.rt[(self.df.choice==1) & (self.df.stimulus==0) & (self.df.subj_idx==i) & (self.df.pupil_high==j)]) for i in range(self.nr_subjects)]), color='r', linestyle='--', lw=1)
            # ax.axvline(np.mean([np.median(self.df.rt[(self.df.choice==0) & (self.df.stimulus==0) & (self.df.subj_idx==i) & (self.df.pupil_high==j)]) for i in range(self.nr_subjects)]), color='b', linestyle='--', lw=1)
            # sns.despine(offset=10, trim=True)
            # plt.tight_layout()
            # fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_rt_hists_{}.pdf'.format(j)))
            #
            #
            # # Make the plt.plot
            # fig = plt.figure(figsize=(2, 1.5))
            # ax = plt.subplot(111)
            # conditions = pd.Series(['correct'], name='trial type')
            # sns.tsplot(correct, time=step, condition=conditions, value='kde', color='green', ci=66, lw=1, ls='-', ax=ax)
            # conditions = pd.Series(['error'], name='trial type')
            # sns.tsplot(error, time=step, condition=conditions, value='kde', color='red', alpha=0.5, ci=66, lw=1, ls='-', ax=ax)
            # ax.axvline(np.mean([np.median(self.df.rt[(self.df.correct==1) & (self.df.subj_idx==i) & (self.df.pupil_high==j)]) for i in range(self.nr_subjects)]), color='g', linestyle='--', lw=1)
            # ax.axvline(np.mean([np.median(self.df.rt[(self.df.correct==0) & (self.df.subj_idx==i) & (self.df.pupil_high==j)]) for i in range(self.nr_subjects)]), color='r', linestyle='--', lw=1)
            # sns.despine(offset=10, trim=True)
            # plt.tight_layout()
            # fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_rt_hists_correct_{}.pdf'.format(j)))

        
            
    def behavior_two_conditions(self):
        
        sns.set(style="ticks")
        
        for j in range(2):
            
            shell()
            
            if j == 0:
                ind = self.pupil_l_ind
            if j == 1:
                ind = self.pupil_h_ind
            
            # SDT fractions:
            # --------------
            hit_fraction = np.zeros(len(self.subjects))
            fa_fraction = np.zeros(len(self.subjects))
            miss_fraction = np.zeros(len(self.subjects))
            cr_fraction = np.zeros(len(self.subjects))
            for i in range(len(self.subjects)):
                hit_fraction[i] = sum(self.hit[(self.subj_idx==i)*ind]) / float(sum(self.present[(self.subj_idx==i)*ind]))
                fa_fraction[i] = sum(self.fa[(self.subj_idx==i)*ind]) / float(sum(self.absent[(self.subj_idx==i)*ind]))
                miss_fraction[i] = sum(self.miss[(self.subj_idx==i)*ind]) / float(sum(self.present[(self.subj_idx==i)*ind]))
                cr_fraction[i] = sum(self.cr[(self.subj_idx==i)*ind]) / float(sum(self.absent[(self.subj_idx==i)*ind]))
            MEANS_correct = (np.mean(hit_fraction), np.mean(cr_fraction))
            SEMS_correct = (sp.stats.sem(hit_fraction), sp.stats.sem(cr_fraction))
            MEANS_error = (np.mean(miss_fraction), np.mean(fa_fraction))
            SEMS_error = (sp.stats.sem(miss_fraction), sp.stats.sem(fa_fraction))
            
            N = 2
            locs = np.linspace(0,N/2,N)  # the x locations for the groups
            bar_width = 0.50       # the width of the bars
            fig = plt.figure(figsize=(2,3))
            ax = fig.add_subplot(111)
            for i in range(N):
                ax.bar(locs[i], height=MEANS_correct[i], width = bar_width, yerr = SEMS_correct[i], color = ['r', 'b'][i], alpha = 1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
            for i in range(N):
                ax.bar(locs[i], height=MEANS_error[i], bottom = MEANS_correct[i], width = bar_width, color = ['b', 'r'][i], alpha = 0.5, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
            ax.axhline(np.mean(MEANS_correct), color='g', ls='--')
            print np.mean(MEANS_correct)
            ax.set_ylabel('fraction of trials', size = 10)
            ax.set_title('SDT fractions', size = 12)
            ax.set_xticks( (locs) )
            ax.set_xticklabels( ('signal+\nnoise', 'noise') )
            ax.set_ylim((0,1))
            plt.gca().spines["bottom"].set_linewidth(.5)
            plt.gca().spines["left"].set_linewidth(.5)
            plt.tight_layout()
            sns.despine(offset=10, trim=True)
            fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_SDT_fractions_{}.pdf'.format(j)))
            
            # RT histograms:
            # --------------
            nbins = 20
            
            RESPONSE_TIME = np.array(self.rt)/1000.0
            # max_y = max( max(plt.hist(RESPONSE_TIME[self.hit*ind], bins=nbins)[0]), max(plt.hist(RESPONSE_TIME[self.cr*ind], bins=nbins)[0]) )
            max_y = 80
            y1,binEdges1 = np.histogram(RESPONSE_TIME[self.hit*ind],bins=nbins)
            bincenters1 = 0.5*(binEdges1[1:]+binEdges1[:-1])
            y2,binEdges2 = np.histogram(RESPONSE_TIME[self.fa*ind],bins=nbins)
            bincenters2 = 0.5*(binEdges2[1:]+binEdges2[:-1])
            y3,binEdges3 = np.histogram(RESPONSE_TIME[self.miss*ind],bins=nbins)
            bincenters3 = 0.5*(binEdges3[1:]+binEdges3[:-1])
            y4,binEdges4 = np.histogram(RESPONSE_TIME[self.cr*ind],bins=nbins)
            bincenters4 = 0.5*(binEdges4[1:]+binEdges4[:-1])
        
            fig = plt.figure(figsize=(4, 5))
            # present:
            a = plt.subplot(211)
            a.plot(bincenters1,y1,'-', color='r', label='hit')
            a.plot(bincenters3,y3,'-', color='b', alpha=0.5, label='miss')
            a.legend()
            a.set_ylim(ymax=max_y)
            a.set_xlim(xmin=0.25, xmax=3.5)
            simpleaxis(a)
            spine_shift(a)
            a.axes.tick_params(axis='both', which='major', labelsize=8)
            a.set_title('RT histograms', size = 12)
            a.set_ylabel('# trials')
            a.axvline(np.median(RESPONSE_TIME[self.hit*ind]), color='r', linestyle='--')
            a.axvline(np.median(RESPONSE_TIME[self.miss*ind]), color='b', linestyle='--')
            plt.gca().spines["bottom"].set_linewidth(.5)
            plt.gca().spines["left"].set_linewidth(.5)
            # absent:
            b = plt.subplot(212)
            b.plot(bincenters2,y2,'-', color='r', alpha=0.5, label='fa')
            b.plot(bincenters4,y4,'-', color='b', label='cr')
            b.legend()
            b.set_ylim(ymax=max_y)
            b.set_xlim(xmin=0.25, xmax=3.5)
            simpleaxis(b)
            spine_shift(b)
            b.axes.tick_params(axis='both', which='major', labelsize=8)
            b.set_xlabel('RT (s)')
            b.set_ylabel('# trials')
            b.axvline(np.median(RESPONSE_TIME[self.fa*ind]), color='r', linestyle='--')
            b.axvline(np.median(RESPONSE_TIME[self.cr*ind]), color='b', linestyle='--')
            plt.gca().spines["bottom"].set_linewidth(.5)
            plt.gca().spines["left"].set_linewidth(.5)
            plt.tight_layout()
            fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_rt_hists_{}.pdf'.format(j)))
            
        # SDT fractions:
        # --------------
        hit_fraction = np.zeros(len(self.subjects))
        fa_fraction = np.zeros(len(self.subjects))
        miss_fraction = np.zeros(len(self.subjects))
        cr_fraction = np.zeros(len(self.subjects))
        for i in range(len(self.subjects)):
            hit_fraction[i] = sum(self.yes[(self.subj_idx==i)*self.pupil_l_ind]) / float(sum(self.pupil_l_ind[self.subj_idx==i]))
            fa_fraction[i] = sum(self.no[(self.subj_idx==i)*self.pupil_h_ind]) / float(sum(self.pupil_h_ind[self.subj_idx==i]))
            miss_fraction[i] = sum(self.no[(self.subj_idx==i)*self.pupil_l_ind]) / float(sum(self.pupil_l_ind[self.subj_idx==i]))
            cr_fraction[i] = sum(self.yes[(self.subj_idx==i)*self.pupil_h_ind]) / float(sum(self.pupil_h_ind[self.subj_idx==i]))
        MEANS_correct = (np.mean(hit_fraction), np.mean(cr_fraction))
        SEMS_correct = (sp.stats.sem(hit_fraction), sp.stats.sem(cr_fraction))
        MEANS_error = (np.mean(miss_fraction), np.mean(fa_fraction))
        SEMS_error = (sp.stats.sem(miss_fraction), sp.stats.sem(fa_fraction))
        
        N = 2
        locs = np.linspace(0,N/2,N)  # the x locations for the groups
        bar_width = 0.50       # the width of the bars
        fig = plt.figure(figsize=(2,3))
        ax = fig.add_subplot(111)
        for i in range(N):
            ax.bar(locs[i], height=MEANS_correct[i], width = bar_width, yerr = SEMS_correct[i], color = ['r', 'r'][i], alpha = 1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
        for i in range(N):
            ax.bar(locs[i], height=MEANS_error[i], bottom = MEANS_correct[i], width = bar_width, color = ['b', 'b'][i], alpha = 1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
        ax.axhline(np.mean(MEANS_correct), color='g', ls='--')
        print np.mean(MEANS_correct)
        ax.set_ylabel('fraction of yes trials', size = 10)
        ax.set_xticks( (locs) )
        ax.set_xticklabels( ('low', 'high') )
        ax.set_ylim((0,0.6))
        plt.gca().spines["bottom"].set_linewidth(.5)
        plt.gca().spines["left"].set_linewidth(.5)
        plt.tight_layout()
        sns.despine(offset=10, trim=True)
        fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_SDT_fractions_2_{}.pdf'.format(j)))
    
    def rt_distributions(self, bins=25):
        
        quantiles = [0.5, 10, 30, 50, 70, 90, 99.5]
        
        data = self.parameters_joined
        pupil = 'ppr_mean'
        
        data.rt = data.rt / 1000.0
        
        
        plot_width = self.nr_subjects * 4
        
        # plot 1 -- rt combined
        plot_nr = 1
        fig = plt.figure(figsize=(plot_width,4))
        for i in xrange(self.nr_subjects):
            
            ax1 = fig.add_subplot(1,self.nr_subjects,plot_nr)
            data_subj = data[data.subject==self.subjects[i]]
            rt = np.array(data_subj.rt)
            myfuncs.hist_q(rt, bins=bins, quantiles=quantiles, ax=ax1, xlim=(0,4))
            ax1.set_xlabel('rt')
            plot_nr += 1
        plt.tight_layout()
        fig.savefig(os.path.join(self.project_directory, 'figures', 'rt.pdf'))

        # plot 2 -- rt split by pupil
        plot_nr = 1
        fig = plt.figure(figsize=(plot_width,8))
        for i in xrange(self.nr_subjects):

            data_subj = data[data.subject==self.subjects[i]]
            max_ylim = max(max(np.histogram(np.array(data_subj.rt[data_subj[pupil] < np.median(data_subj[pupil])]), bins=bins)[0]), max(np.histogram(np.array(data_subj.rt[data_subj[pupil] > np.median(data_subj[pupil])]), bins=bins)[0]))

            ax1 = plt.subplot(2,self.nr_subjects,plot_nr)
            rt = np.array(data_subj.rt[data_subj[pupil] < np.median(data_subj[pupil])])
            myfuncs.hist_q(rt, bins=bins, quantiles=quantiles, ax=ax1, quantiles_color='k', alpha=0.75, xlim=(0,4), ylim=(0,max_ylim))
            ax2 = plt.subplot(2,self.nr_subjects,plot_nr+self.nr_subjects)
            rt = np.array(data_subj.rt[data_subj[pupil] > np.median(data_subj[pupil])])
            myfuncs.hist_q(rt, bins=bins, quantiles=quantiles, ax=ax2, quantiles_color='k', xlim=(0,4), ylim=(0,max_ylim))
            ax1.set_xlabel('rt')
            ax2.set_xlabel('rt')
            plot_nr += 1
        plt.tight_layout()
        fig.savefig(os.path.join(self.project_directory, 'figures', 'rt_split_' + pupil + '.pdf'))

        # plot 3 -- rt split by SDT trial types
        plot_nr = 1
        fig = plt.figure(figsize=(plot_width,16))
        for i in xrange(self.nr_subjects):

            data_subj = data[data.subject==self.subjects[i]]
            max_ylim = max(max(np.histogram(np.array(data_subj.rt[data_subj.hit]), bins=bins)[0]), max(np.histogram(np.array(data_subj.rt[data_subj.cr]), bins=bins)[0]))

            ax1 = plt.subplot(4, self.nr_subjects,plot_nr)
            rt = np.array(data_subj.rt[data_subj.hit])
            myfuncs.hist_q(rt, bins=bins, quantiles=quantiles, ax=ax1, quantiles_color='r', xlim=(0,4), ylim=(0,max_ylim))
            ax2 = plt.subplot(4, self.nr_subjects,plot_nr+self.nr_subjects)
            rt = np.array(data_subj.rt[data_subj.fa])
            myfuncs.hist_q(rt, bins=bins, quantiles=quantiles, ax=ax2, quantiles_color='r', alpha=0.5, xlim=(0,4), ylim=(0,max_ylim))
            ax3 = plt.subplot(4, self.nr_subjects,plot_nr+(2*self.nr_subjects))
            rt = np.array(data_subj.rt[data_subj.miss])
            myfuncs.hist_q(rt, bins=bins, quantiles=quantiles, ax=ax3, quantiles_color='b', alpha=0.5, xlim=(0,4), ylim=(0,max_ylim))
            ax4 = plt.subplot(4, self.nr_subjects,plot_nr+(3*self.nr_subjects))
            rt = np.array(data_subj.rt[data_subj.cr])
            myfuncs.hist_q(rt, bins=bins, quantiles=quantiles, ax=ax4, quantiles_color='b', xlim=(0,4), ylim=(0,max_ylim))
            ax1.set_xlabel('rt')
            ax2.set_xlabel('rt')
            ax3.set_xlabel('rt')
            ax4.set_xlabel('rt')
            plot_nr += 1
        plt.tight_layout()
    
        fig.savefig(os.path.join(self.project_directory, 'figures', 'rt_split_answer.pdf'))
    
    def pupil_bars(self):
        
        ppr_hit = [np.mean(self.pupil_d[self.subj_idx == i][self.hit[[self.subj_idx == i]]]) for i in range(self.nr_subjects)]
        ppr_fa = [np.mean(self.pupil_d[self.subj_idx == i][self.fa[[self.subj_idx == i]]]) for i in range(self.nr_subjects)]
        ppr_miss = [np.mean(self.pupil_d[self.subj_idx == i][self.miss[[self.subj_idx == i]]]) for i in range(self.nr_subjects)]
        ppr_cr = [np.mean(self.pupil_d[self.subj_idx == i][self.cr[[self.subj_idx == i]]]) for i in range(self.nr_subjects)]
        
        ppr_yes = [np.mean(self.pupil_d[self.subj_idx == i][(self.hit+self.fa)[[self.subj_idx == i]]]) for i in range(self.nr_subjects)]
        ppr_no = [np.mean(self.pupil_d[self.subj_idx == i][(self.miss+self.cr)[[self.subj_idx == i]]]) for i in range(self.nr_subjects)]
        ppr_correct = [np.mean(self.pupil_d[self.subj_idx == i][(self.hit+self.cr)[[self.subj_idx == i]]]) for i in range(self.nr_subjects)]
        ppr_error = [np.mean(self.pupil_d[self.subj_idx == i][(self.miss+self.fa)[[self.subj_idx == i]]]) for i in range(self.nr_subjects)]
        
        bpd_hit = [np.mean(self.pupil_b[self.subj_idx == i][self.hit[[self.subj_idx == i]]]) for i in range(self.nr_subjects)]
        bpd_fa = [np.mean(self.pupil_b[self.subj_idx == i][self.fa[[self.subj_idx == i]]]) for i in range(self.nr_subjects)]
        bpd_miss = [np.mean(self.pupil_b[self.subj_idx == i][self.miss[[self.subj_idx == i]]]) for i in range(self.nr_subjects)]
        bpd_cr = [np.mean(self.pupil_b[self.subj_idx == i][self.cr[[self.subj_idx == i]]]) for i in range(self.nr_subjects)]
        
        bpd_yes = [np.mean(self.pupil_b[self.subj_idx == i][(self.hit+self.fa)[[self.subj_idx == i]]]) for i in range(self.nr_subjects)]
        bpd_no = [np.mean(self.pupil_b[self.subj_idx == i][(self.miss+self.cr)[[self.subj_idx == i]]]) for i in range(self.nr_subjects)]
        bpd_correct = [np.mean(self.pupil_b[self.subj_idx == i][(self.hit+self.cr)[[self.subj_idx == i]]]) for i in range(self.nr_subjects)]
        bpd_error = [np.mean(self.pupil_b[self.subj_idx == i][(self.miss+self.fa)[[self.subj_idx == i]]]) for i in range(self.nr_subjects)]
        
        my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}
        
        
        # shell()
        
        N = 4
        ind = np.linspace(0,2,N)  # the x locations for the groups
        bar_width = 0.6   # the width of the bars
        spacing = [0, 0, 0, 0]
        
        sns.set_style("ticks")
        
        # FIGURE 1
        # p_values = np.array([myfuncs.permutationTest(ppr_hit, ppr_miss)[1], myfuncs.permutationTest(ppr_fa, ppr_cr)[1]])
        p_values = np.array([sp.stats.ttest_rel(ppr_hit, ppr_miss)[1], sp.stats.ttest_rel(ppr_fa, ppr_cr)[1]])
        print
        print p_values
        print
        ppr = [ppr_hit, ppr_miss, ppr_fa, ppr_cr]
        MEANS = np.array([np.mean(values) for values in ppr])
        SEMS = np.array([sp.stats.sem(values) for values in ppr])
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot(111)
        for i in range(N):
            ax.bar(ind[i]+spacing[i], MEANS[i], width = bar_width, yerr=SEMS[i], color=['r','b','r','b'][i], alpha=[1,0.5,0.5,1][i], edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
        simpleaxis(ax)
        spine_shift(ax)
        ax.set_xticklabels( ('H','M','FA','CR') )
        ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
        ax.tick_params(axis='x', which='major', labelsize=10)
        ax.tick_params(axis='y', which='major', labelsize=10)
        plt.gca().spines["bottom"].set_linewidth(.5)
        plt.gca().spines["left"].set_linewidth(.5)
        plt.title('phasic pupil responses')
        plt.ylabel('pupil response (a.u.)')
        plt.text(x=np.mean((ind[0],ind[1])), y=0, s='p = {}'.format(round(p_values[0],3)), horizontalalignment='center')
        plt.text(x=np.mean((ind[2],ind[3])), y=0, s='p = {}'.format(round(p_values[1],3)), horizontalalignment='center')
        plt.tight_layout()
        fig.savefig(os.path.join(self.project_directory, 'figures', 'STD1_ppr.pdf'))
        
        # FIGURE 2
        # p_values = np.array([myfuncs.permutationTest(ppr_yes, ppr_no)[1], myfuncs.permutationTest(ppr_correct, ppr_error)[1]])
        p_values = np.array([sp.stats.ttest_rel(ppr_yes, ppr_no)[1], sp.stats.ttest_rel(ppr_correct, ppr_error)[1]])
        print
        print p_values
        print
        ppr = [ppr_yes, ppr_no, ppr_correct, ppr_error]
        MEANS = np.array([np.mean(values) for values in ppr])
        SEMS = np.array([sp.stats.sem(values) for values in ppr])
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot(111)
        for i in range(N):
            ax.bar(ind[i]+spacing[i], MEANS[i], width = bar_width, yerr=SEMS[i], color=['r','b','k','k'][i], alpha=[1,1,1,0.5][i], edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
        simpleaxis(ax)
        spine_shift(ax)
        ax.set_xticklabels( ('YES','NO','CORRECT','ERROR') )
        ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
        ax.tick_params(axis='x', which='major', labelsize=10)
        ax.tick_params(axis='y', which='major', labelsize=10)
        plt.gca().spines["bottom"].set_linewidth(.5)
        plt.gca().spines["left"].set_linewidth(.5)
        plt.title('phasic pupil responses')
        plt.ylabel('pupil response (a.u.)')
        plt.text(x=np.mean((ind[0],ind[1])), y=0, s='p = {}'.format(round(p_values[0],3)), horizontalalignment='center')
        plt.text(x=np.mean((ind[2],ind[3])), y=0, s='p = {}'.format(round(p_values[1],3)), horizontalalignment='center')
        plt.tight_layout()
        fig.savefig(os.path.join(self.project_directory, 'figures', 'STD2_ppr.pdf'))
        
        # FIGURE 3
        # p_values = np.array([myfuncs.permutationTest(bpd_hit, bpd_miss)[1], myfuncs.permutationTest(bpd_fa, bpd_cr)[1]])
        p_values = np.array([sp.stats.ttest_rel(bpd_hit, bpd_miss)[1], sp.stats.ttest_rel(bpd_fa, bpd_cr)[1]])
        bpd = [bpd_hit, bpd_miss, bpd_fa, bpd_cr]
        MEANS = np.array([np.mean(values) for values in bpd])
        SEMS = np.array([sp.stats.sem(values) for values in bpd])
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot(111)
        for i in range(N):
            ax.bar(ind[i]+spacing[i], MEANS[i], width = bar_width, yerr=SEMS[i], color=['r','b','r','b'][i], alpha=[1,0.5,0.5,1][i], edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
        simpleaxis(ax)
        spine_shift(ax)
        ax.set_xticklabels( ('H','M','FA','CR') )
        ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
        ax.tick_params(axis='x', which='major', labelsize=10)
        ax.tick_params(axis='y', which='major', labelsize=10)
        plt.gca().spines["bottom"].set_linewidth(.5)
        plt.gca().spines["left"].set_linewidth(.5)
        plt.title('baseline pupil responses')
        plt.ylabel('pupil response (z)')
        plt.text(x=np.mean((ind[0],ind[1])), y=0, s='p = {}'.format(round(p_values[0],3)), horizontalalignment='center')
        plt.text(x=np.mean((ind[2],ind[3])), y=0, s='p = {}'.format(round(p_values[1],3)), horizontalalignment='center')
        plt.tight_layout()
        fig.savefig(os.path.join(self.project_directory, 'figures', 'STD1_bpd.pdf'))
        
        # FIGURE 4
        # p_values = np.array([myfuncs.permutationTest(bpd_yes, bpd_no)[1], myfuncs.permutationTest(bpd_correct, bpd_error)[1]])
        p_values = np.array([sp.stats.ttest_rel(bpd_yes, bpd_no)[1], sp.stats.ttest_rel(bpd_correct, bpd_error)[1]])
        bpd = [bpd_yes, bpd_no, bpd_correct, bpd_error]
        MEANS = np.array([np.mean(values) for values in bpd])
        SEMS = np.array([sp.stats.sem(values) for values in bpd])
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot(111)
        for i in range(N):
            ax.bar(ind[i]+spacing[i], MEANS[i], width = bar_width, yerr=SEMS[i], color=['r','b','k','k'][i], alpha=[1,1,1,0.5][i], edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
        simpleaxis(ax)
        spine_shift(ax)
        ax.set_xticklabels( ('YES','NO','CORRECT','ERROR') )
        ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
        ax.tick_params(axis='x', which='major', labelsize=10)
        ax.tick_params(axis='y', which='major', labelsize=10)
        plt.gca().spines["bottom"].set_linewidth(.5)
        plt.gca().spines["left"].set_linewidth(.5)
        plt.title('baseline pupil responses')
        plt.ylabel('pupil response (a.u.)')
        plt.text(x=np.mean((ind[0],ind[1])), y=0, s='p = {}'.format(round(p_values[0],3)), horizontalalignment='center')
        plt.text(x=np.mean((ind[2],ind[3])), y=0, s='p = {}'.format(round(p_values[1],3)), horizontalalignment='center')
        plt.tight_layout()
        fig.savefig(os.path.join(self.project_directory, 'figures', 'STD2_bpd.pdf'))
    
    
    def pupil_criterion(self):
        
        # shell()
        
        fig = plt.figure(figsize=(2,2))
        ax = fig.add_subplot(111)
        myfuncs.quantile_plot(conditions=np.ones(len(self.rt)), rt=self.rt, corrects=self.correct, subj_idx=self.subj_idx, ax=ax)
        
        
        ppr_yes = np.array([np.mean(self.pupil_d[(self.subj_idx == i) & self.yes]) for i in range(self.nr_subjects)])
        ppr_no = np.array([np.mean(self.pupil_d[(self.subj_idx == i) & self.no]) for i in range(self.nr_subjects)])
        
        fig = plt.figure(figsize=(2,2))
        ax = fig.add_subplot(111)
        myfuncs.correlation_plot(self.criterion, ppr_yes-ppr_no, ax=ax, line=True)
        plt.xlabel('criterion')
        plt.ylabel('choice effect (yes-no)')
        plt.tight_layout()
        fig.savefig(os.path.join(self.project_directory, 'figures', 'criterion.pdf'))
        
    def pupil_prediction_error(self):
        
        self.confidence = np.array(self.parameters_joined['confidence'])
        
        conf1 = np.array(self.parameters_joined['confidence'] == 0)
        conf1[self.subj_idx ==np.where(np.array(self.subjects) == 'dh')[0][0]] = np.array(self.parameters_joined['confidence'][self.subj_idx ==np.where(np.array(self.subjects) == 'dh')[0][0]] == 1)
        conf2 = np.array(self.parameters_joined['confidence'] == 1)
        conf3 = np.array(self.parameters_joined['confidence'] == 2)
        conf4 = np.array(self.parameters_joined['confidence'] == 3)
        
        
        self.pupil_d_feed = np.array(self.parameters_joined['ppr_mean_feed_lp'])
        # self.pupil_d_feed = np.array(self.parameters_joined['ppr_proj_feed_lp'])
        
        ppr_correct_0 = np.array([np.mean(self.pupil_d_feed[self.subj_idx == i][((self.hit+self.cr)*conf1)[[self.subj_idx == i]]]) for i in range(self.nr_subjects)])
        ppr_error_0 = np.array([np.mean(self.pupil_d_feed[self.subj_idx == i][((self.miss+self.fa)*conf1)[[self.subj_idx == i]]]) for i in range(self.nr_subjects)])
        ppr_correct_1 = np.array([np.mean(self.pupil_d_feed[self.subj_idx == i][((self.hit+self.cr)*conf2)[[self.subj_idx == i]]]) for i in range(self.nr_subjects)])
        ppr_error_1 = np.array([np.mean(self.pupil_d_feed[self.subj_idx == i][((self.miss+self.fa)*conf2)[[self.subj_idx == i]]]) for i in range(self.nr_subjects)])
        ppr_correct_2 = np.array([np.mean(self.pupil_d_feed[self.subj_idx == i][((self.hit+self.cr)*conf3)[[self.subj_idx == i]]]) for i in range(self.nr_subjects)])
        ppr_error_2 = np.array([np.mean(self.pupil_d_feed[self.subj_idx == i][((self.miss+self.fa)*conf3)[[self.subj_idx == i]]]) for i in range(self.nr_subjects)])
        ppr_correct_3 = np.array([np.mean(self.pupil_d_feed[self.subj_idx == i][((self.hit+self.cr)*conf4)[[self.subj_idx == i]]]) for i in range(self.nr_subjects)])
        ppr_error_3 = np.array([np.mean(self.pupil_d_feed[self.subj_idx == i][((self.miss+self.fa)*conf4)[[self.subj_idx == i]]]) for i in range(self.nr_subjects)])
        
        performance_0 = np.array([sum(self.correct[self.subj_idx == i][conf1[self.subj_idx == i]]) / float(len(self.correct[self.subj_idx == i][conf1[self.subj_idx == i]])) for i in range(self.nr_subjects)]) * 100.0
        performance_1 = np.array([sum(self.correct[self.subj_idx == i][conf2[self.subj_idx == i]]) / float(len(self.correct[self.subj_idx == i][conf2[self.subj_idx == i]])) for i in range(self.nr_subjects)]) * 100.0 
        performance_2 = np.array([sum(self.correct[self.subj_idx == i][conf3[self.subj_idx == i]]) / float(len(self.correct[self.subj_idx == i][conf3[self.subj_idx == i]])) for i in range(self.nr_subjects)]) * 100.0
        performance_3 = np.array([sum(self.correct[self.subj_idx == i][conf4[self.subj_idx == i]]) / float(len(self.correct[self.subj_idx == i][conf4[self.subj_idx == i]])) for i in range(self.nr_subjects)]) * 100.0
        
        my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}

        N = 4
        ind = np.linspace(0,2,N)  # the x locations for the groups
        bar_width = 0.6   # the width of the bars
        spacing = [0, 0, 0, 0]
        
        # FIGURE 1
        ppr = [ppr_error_0-ppr_correct_0, ppr_error_1-ppr_correct_1, ppr_error_2-ppr_correct_2, ppr_error_3-ppr_correct_3]
        MEANS = np.array([np.mean(values) for values in ppr])
        SEMS = np.array([sp.stats.sem(values) for values in ppr])
        
        performance = [performance_0, performance_1, performance_2, performance_3]
        MEANS2 = np.array([np.mean(values) for values in performance])
        SEMS2 = np.array([sp.stats.sem(values) for values in performance])
        
                
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot(111)
        ax.plot(ind, MEANS, ls='--', color='k', alpha=0.75)
        ax.errorbar(ind, MEANS, yerr=SEMS, color = 'k', fmt="o", ms=10, capsize = 1, label='pupil')
        ax.set_xticklabels( ('--','-','+','++') )
        ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
        ax.set_ylabel('pupil response amplitude\n(error - correct) (Z)')
        ax2 = plt.twinx(ax)
        ax2.plot(ind, MEANS2, ls='--', color='g', alpha=0.75)
        ax2.errorbar(ind, MEANS2, yerr=SEMS2, color = 'g', fmt="o", ms=10, capsize = 1, label='accuracy')
        ax2.set_ylabel('accuracy (% correct)')
        plt.xlim(ind[0]-0.5, ind[-1]+0.5)
        ax.legend(loc=2)
        ax2.legend()
        plt.title('N=6')
        plt.xlabel('confidence')
        plt.tight_layout()
        fig.savefig(os.path.join(self.project_directory, 'figures', 'prediction_error_ppr.pdf'))
        
        # FIGURE 2
        ppr = np.concatenate([ppr_error_0-ppr_correct_0, ppr_error_1-ppr_correct_1, ppr_error_2-ppr_correct_2, ppr_error_3-ppr_correct_3])
        conf = np.concatenate((np.ones(self.nr_subjects), np.ones(self.nr_subjects)*2, np.ones(self.nr_subjects)*3, np.ones(self.nr_subjects)*4))
        # fig = myfuncs.correlation_plot2(X=conf, Y=ppr, labelX='confidence', labelY='pupil', xlim=(-0.2, 0.8), ylim=(0,4))
        fig = myfuncs.correlation_plot(conf, ppr)
        plt.xlim(0,5)
        plt.ylim(ymax=0.75)
        plt.title('phasic pupil responses')
        plt.ylabel('pupil response (z)')
        plt.xticks( (1,2,3,4), ('--','-','+','++') )
        plt.tight_layout()
        fig.savefig(os.path.join(self.project_directory, 'figures', 'prediction_error2_ppr.pdf'))
        
        d = {
        'subj_idx' : pd.Series(self.subj_idx),
        'correct' : pd.Series(np.array(self.parameters_joined['correct'], dtype=int)),
        'confidence' : pd.Series(np.array(self.parameters_joined['confidence'], dtype=int)),
        'pupil' : pd.Series(np.array(self.pupil_d_feed)),
        }
        data_response = pd.DataFrame(d)
        data_response.to_csv(os.path.join(self.project_directory, 'feedback_data_jw.csv'))
        
        
        
        
        
        
        
        
        
        
    #     # FIGURE 1
    #     MEANS_yes = np.array([np.mean(self.pupil_d[(self.hit+self.fa)*cond]) for cond in conditions])
    #     SEMS_yes = np.array([sp.stats.sem(self.pupil_d[(self.hit+self.fa)*cond]) for cond in conditions])
    #     MEANS_no = np.array([np.mean(self.pupil_d[(self.miss+self.cr)*cond]) for cond in conditions])
    #     SEMS_no = np.array([sp.stats.sem(self.pupil_d[(self.miss+self.cr)*cond]) for cond in conditions])
    #     fig = plt.figure(figsize=(4,3))
    #     ax = fig.add_subplot(111)
    #     ax.errorbar(ind, MEANS_yes, yerr=SEMS_yes, color = 'r', capsize = 0)
    #     ax.errorbar(ind, MEANS_no, yerr=SEMS_no, color = 'b', capsize = 0)
    #     simpleaxis(ax)
    #     spine_shift(ax)
    #     ax.set_xticklabels( ('--','-','+','++') )
    #     ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
    #     ax.tick_params(axis='x', which='major', labelsize=10)
    #     ax.tick_params(axis='y', which='major', labelsize=10)
    #     ax.set_ylim(ymin=0.2, ymax=1.6)
    #     plt.gca().spines["bottom"].set_linewidth(.5)
    #     plt.gca().spines["left"].set_linewidth(.5)
    #     plt.xlabel('confidence')
    #     plt.ylabel('pupil response (a.u.)')
    #     plt.title('subject {}, c = {}'.format((self.subject.initials), round(np.array(self.parameters_joined['criterion'])[0],3)))
    #     plt.tight_layout()
    #     fig.savefig(os.path.join(self.base_directory, 'figs', 'pupil_confidence_choice_2_' + self.subject.initials + '.pdf'))
    #
    #     # FIGURE 1
    #     MEANS = np.array([np.mean(self.pupil_d[(self.hit+self.fa)*cond])-np.mean(self.pupil_d[(self.miss+self.cr)*cond]) for cond in conditions])
    #     SEMS = np.array([(sp.stats.sem(self.pupil_d[(self.hit+self.fa)*cond])+sp.stats.sem(self.pupil_d[(self.miss+self.cr)*cond]))/2 for cond in conditions])
    #     fig = plt.figure(figsize=(4,3))
    #     ax = fig.add_subplot(111)
    #     ax.errorbar(ind, MEANS, yerr=SEMS, color = 'k', capsize = 0)
    #     simpleaxis(ax)
    #     spine_shift(ax)
    #     ax.set_xticklabels( ('--','-','+','++') )
    #     ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
    #     ax.tick_params(axis='x', which='major', labelsize=10)
    #     ax.tick_params(axis='y', which='major', labelsize=10)
    #     ax.set_ylim(ymin=-0.4, ymax=1.0)
    #     plt.gca().spines["bottom"].set_linewidth(.5)
    #     plt.gca().spines["left"].set_linewidth(.5)
    #     plt.xlabel('confidence')
    #     plt.ylabel('pupil response (a.u.)')
    #     plt.title('subject {}, c = {}'.format((self.subject.initials), round(np.array(self.parameters_joined['criterion'])[0],3)))
    #     plt.tight_layout()
    #     fig.savefig(os.path.join(self.base_directory, 'figs', 'pupil_confidence_choice_' + self.subject.initials + '.pdf'))
    #
    #     # FIGURE 2
    #     MEANS = np.array([np.mean(self.pupil_d[(self.hit+self.cr)*cond])-np.mean(self.pupil_d[(self.fa+self.miss)*cond]) for cond in conditions])
    #     SEMS = np.array([(sp.stats.sem(self.pupil_d[(self.hit+self.cr)*cond])+sp.stats.sem(self.pupil_d[(self.fa+self.miss)*cond]))/2 for cond in conditions])
    #     fig = plt.figure(figsize=(4,3))
    #     ax = fig.add_subplot(111)
    #     for i in range(N):
    #         ax.bar(ind[i]+spacing[i], MEANS[i], width = bar_width, yerr=SEMS[i], color = 'b', alpha = [.25,0.5,0.75,1][i], edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
    #     simpleaxis(ax)
    #     spine_shift(ax)
    #     ax.set_xticklabels( ('--','-','+','++') )
    #     ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
    #     ax.tick_params(axis='x', which='major', labelsize=10)
    #     ax.tick_params(axis='y', which='major', labelsize=10)
    #     plt.gca().spines["bottom"].set_linewidth(.5)
    #     plt.gca().spines["left"].set_linewidth(.5)
    #     plt.ylabel('pupil correctness effect (a.u.)')
    #     plt.title('subject {}, c = {}'.format((self.subject.initials), round(np.array(self.parameters_joined['criterion'])[0],3)))
    #     plt.tight_layout()
    #     fig.savefig(os.path.join(self.base_directory, 'figs', 'pupil_confidence_correct_' + self.subject.initials + '.pdf'))
    #
    #
    #     if self.experiment == 1:
    #         # FIGURE 4
    #         MEANS_yes = np.array([np.mean(self.pupil_d_feed[(self.hit+self.cr)*cond]) for cond in conditions])
    #         SEMS_yes = np.array([sp.stats.sem(self.pupil_d_feed[(self.hit+self.cr)*cond]) for cond in conditions])
    #         MEANS_no = np.array([np.mean(self.pupil_d_feed[(self.fa+self.miss)*cond]) for cond in conditions])
    #         SEMS_no = np.array([sp.stats.sem(self.pupil_d_feed[(self.fa+self.miss)*cond]) for cond in conditions])
    #         fig = plt.figure(figsize=(4,3))
    #         ax = fig.add_subplot(111)
    #         ax.errorbar(ind, MEANS_no-MEANS_yes, yerr=(SEMS_yes+SEMS_no)/2.0, color = 'k', capsize = 0)
    #         simpleaxis(ax)
    #         spine_shift(ax)
    #         ax.set_xticklabels( ('--','-','+','++') )
    #         ax.set_xticks( (ind[0], ind[1], ind[2], ind[3]) )
    #         ax.tick_params(axis='x', which='major', labelsize=10)
    #         ax.tick_params(axis='y', which='major', labelsize=10)
    #         # ax.set_ylim(ymin=0.2, ymax=1.6)
    #         plt.gca().spines["bottom"].set_linewidth(.5)
    #         plt.gca().spines["left"].set_linewidth(.5)
    #         plt.xlabel('confidence')
    #         plt.ylabel('prediction error (a.u.)')
    #         plt.title('subject {}, c = {}'.format((self.subject.initials), round(np.array(self.parameters_joined['criterion'])[0],3)))
    #         plt.tight_layout()
    #         fig.savefig(os.path.join(self.base_directory, 'figs', 'pupil_prediction_error_' + self.subject.initials + '.pdf'))
    #
    #
    #
    # self.pupil_d
    # self.pupil_b
    #
    # conf1 = np.array(self.parameters_joined['confidence'] == 0)
    # conf2 = np.array(self.parameters_joined['confidence'] == 1)
    # conf3 = np.array(self.parameters_joined['confidence'] == 2)
    # conf4 = np.array(self.parameters_joined['confidence'] == 3)
    #
    # conditions = [conf1, conf2, conf3, conf4]
    
    def pupil_signal_presence(self):
        
        # shell()
        
        tp_h = np.zeros(len(self.subjects))
        tp_l = np.zeros(len(self.subjects))
        ta_h = np.zeros(len(self.subjects))
        ta_l = np.zeros(len(self.subjects))
        d_h = np.zeros(len(self.subjects))
        d_l = np.zeros(len(self.subjects))
        criterion_h = np.zeros(len(self.subjects))
        criterion_l = np.zeros(len(self.subjects))
        h = np.zeros(len(self.subjects))
        l = np.zeros(len(self.subjects))
        for i in range(len(self.subjects)):
            params = self.parameters_joined[self.subj_idx == i]
            params = params[-params['omissions']]
            high_pupil_ind = np.array(params['ppr_proj_lp']>np.median(params['ppr_proj_lp']))
            
            tp_h[i] = sum((params['yes']*params['present'])[high_pupil_ind]) / float(sum((params['present'])[high_pupil_ind]))
            tp_l[i] = sum((params['yes']*params['present'])[-high_pupil_ind]) / float(sum((params['present'])[-high_pupil_ind]))
            ta_h[i] = sum((params['yes']*-params['present'])[high_pupil_ind]) / float(sum((-params['present'])[high_pupil_ind]))
            ta_l[i] = sum((params['yes']*-params['present'])[-high_pupil_ind]) / float(sum((-params['present'])[-high_pupil_ind]))
            
            h[i] = (tp_h[i]+ta_h[i]) / 2.0
            l[i] = (tp_l[i]+ta_l[i]) / 2.0
            
            d_h[i] = myfuncs.SDT_measures(np.array(params['present'])[high_pupil_ind], np.array(params['hit'])[high_pupil_ind], np.array(params['fa'])[high_pupil_ind])[0]
            d_l[i] = myfuncs.SDT_measures(np.array(params['present'])[-high_pupil_ind], np.array(params['hit'])[-high_pupil_ind], np.array(params['fa'])[-high_pupil_ind])[0]
            
            criterion_h[i] = myfuncs.SDT_measures(np.array(params['present'])[high_pupil_ind], np.array(params['hit'])[high_pupil_ind], np.array(params['fa'])[high_pupil_ind])[1]
            criterion_l[i] = myfuncs.SDT_measures(np.array(params['present'])[-high_pupil_ind], np.array(params['hit'])[-high_pupil_ind], np.array(params['fa'])[-high_pupil_ind])[1]
        
        fig = myfuncs.correlation_plot(self.criterion, tp_h-tp_l)
        plt.title('signal present')
        plt.xlabel('criterion (c)')
        plt.ylabel('fraction yes high pupil - low pupil')
        fig.savefig(os.path.join(self.project_directory, 'figures', 'pupil_split_signal_present.pdf'))
        
        fig = myfuncs.correlation_plot(self.criterion, ta_h-ta_l)
        plt.title('signal absent')
        plt.xlabel('criterion (c)')
        plt.ylabel('fraction yes high pupil - low pupil')
        fig.savefig(os.path.join(self.project_directory, 'figures', 'pupil_split_signal_absent.pdf'))
        
        fig = myfuncs.correlation_plot(self.criterion, criterion_h-criterion_l)
        plt.xlabel('criterion (c)')
        plt.ylabel('criterion high pupil - low pupil')
        fig.savefig(os.path.join(self.project_directory, 'figures', 'pupil_split_criterion.pdf'))
        
        fig = myfuncs.correlation_plot(self.criterion, d_h-d_l)
        plt.xlabel('criterion (c)')
        plt.ylabel('d prime high pupil - low pupil')
        fig.savefig(os.path.join(self.project_directory, 'figures', 'pupil_split_d_prime.pdf'))
        

    def drift_diffusion(self):
        
        d = {
        'subj_idx' : pd.Series(self.subj_idx) + 22,
        'stimulus' : pd.Series(np.array(self.parameters_joined['present'], dtype=int)),
        'response' : pd.Series(np.array(self.parameters_joined['yes'], dtype=int)),
        'rt' : pd.Series(np.array(self.parameters_joined['rt']))/1000.0,
        'pupil_b_lp' : pd.Series(np.array(self.parameters_joined['pupil_b_lp'])),
        'pupil_b' : pd.Series(np.array(self.parameters_joined['pupil_b'])),
        'pupil_d' : pd.Series(np.array(self.parameters_joined['pupil_d'])),
        'run' : pd.Series(np.array(self.run)),
        'session' : pd.Series(np.array(self.session)),
        # 'split' : pd.Series(np.array(self.pupil_h_ind, dtype=int)),
        }
        data_response = pd.DataFrame(d)
        data_response.to_csv(os.path.join(self.project_directory, 'data_response.csv'))
        
    def average_pupil_responses(self):
        
        nr_runs = [12,12,12,16,10,12,11,10,12,12,12,12,11,12]
        
        fig = plt.figure(figsize=(6, 6))
        for i, s in enumerate(self.subjects):
            
            aliases = ['detection_{}'.format(r+1) for r in range(nr_runs[i])]
        
            self.base_directory = os.path.join(self.project_directory, self.experiment_name, s)
            self.hdf5_filename = os.path.join(self.base_directory, 'processed', s + '.hdf5')
            self.ho = HDFEyeOperator.HDFEyeOperator(self.hdf5_filename)
            
            downsample_rate = 50 # 50
            new_sample_rate = 1000 / downsample_rate

            # load data:
            parameters = []
            pupil = []
            time = []
            cue_times = []
            choice_times = []
            blink_times = []
            time_to_add = 0
            for alias in aliases:
            
                parameters.append(self.ho.read_session_data(alias, 'parameters2'))
                self.alias = alias
                self.trial_times = self.ho.read_session_data(alias, 'trials')
                
                # load pupil:
                self.eye = self.ho.eye_during_period((np.array(self.trial_times['trial_start_EL_timestamp'])[0], np.array(self.trial_times['trial_end_EL_timestamp'])[-1]), self.alias)
                self.pupil_data = self.ho.data_from_time_period((np.array(self.trial_times['trial_start_EL_timestamp'])[0], np.array(self.trial_times['trial_end_EL_timestamp'])[-1]), self.alias)
                # pupil.append(np.array(self.pupil_data[(self.eye + '_pupil_lp_psc')]))
                
                # p =
                pupil.append(np.array(self.pupil_data[(self.eye + '_pupil_bp_clean_psc')]))
                
                # load times:
                self.session_start = self.trial_times['trial_start_EL_timestamp'][0]
                self.time = np.array(self.pupil_data['time']) - self.session_start
                time.append( self.time + time_to_add)
                self.phase_times = self.ho.read_session_data(alias, 'trial_phases')
                cue_times.append( np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 2)]) - self.session_start + time_to_add )
                choice_times.append( np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 3)]) - self.session_start + time_to_add )
                # load blinks:
                self.blink_data = self.ho.read_session_data(alias, 'blinks_from_message_file')
                blink_times.append( np.array(self.blink_data['end_timestamp']) - self.session_start + time_to_add )
                
                time_to_add += self.time[-1]
            
            pupil_h_ind = self.pupil_h_ind[self.subj_idx == i]
            pupil_l_ind = self.pupil_l_ind[self.subj_idx == i]
    
            parameters_joined = pd.concat(parameters)
            pupil = np.concatenate(pupil)
            time = np.concatenate(time)
            cue_times = np.concatenate(cue_times) / 1000.0
            choice_times = np.concatenate(choice_times) / 1000.0
            blink_times = np.concatenate(blink_times) / 1000.0
            omissions = np.array(parameters_joined.omissions, dtype=bool)
            correct = np.array(parameters_joined.correct, dtype=bool)*-omissions
            error = -np.array(parameters_joined.correct, dtype=bool)*-omissions
            hit = np.array(parameters_joined.hit, dtype=bool)*-omissions
            fa = np.array(parameters_joined.fa, dtype=bool)*-omissions
            miss = np.array(parameters_joined.miss, dtype=bool)*-omissions
            cr = np.array(parameters_joined.cr, dtype=bool)*-omissions
    
            # event related averages:
            interval = 10

            # output:
            kernel_choice_high = np.vstack([sp.signal.decimate(pupil[floor(t):floor(t)+10000], downsample_rate) - (self.pupil_b[self.subj_idx == i])[pupil_h_ind][j] for j, t in enumerate((choice_times[-omissions][pupil_h_ind]-5)*1000)])
            kernel_choice_low = np.vstack([sp.signal.decimate(pupil[floor(t):floor(t)+10000], downsample_rate) - (self.pupil_b[self.subj_idx == i])[pupil_l_ind][j] for j, t in enumerate((choice_times[-omissions][pupil_l_ind]-5)*1000)])
            
            kernels_pupil_l = kernel_choice_low
            kernels_pupil_h = kernel_choice_high
            step_lim = [-5,5]
            xlim = [-2.5,2.5]
    
            # length kernel:
            kernel_length = kernels_pupil_l.shape[1]
    
            # step size:
            step = pd.Series(np.linspace(step_lim[0], step_lim[1], kernel_length), name='Time from choice (s)')
            
            xlim_indices = np.array(step >= xlim[0]) * np.array(step <= xlim[1])
        
            ax = fig.add_subplot(4,4,i+1)
            ax.set_title('S{}'.format(i+1))
            ax.axvline(0, lw=0.25, alpha=0.5, color = 'k')
            ax.axhline(0, lw=0.25, alpha=0.5, color = 'k')
            colors = ['black']
            # conditions = pd.Series(['all trials'], name='trial type')
            # sns.tsplot(kernels_pupil_a[:,xlim_indices], time=step[xlim_indices], condition=conditions, value='fMRI response (% signal change)', color='grey', ci=66, lw=1, ls='-', ax=ax)
            conditions = pd.Series(['high pupil'], name='trial type')
            sns.tsplot(kernels_pupil_h[:,xlim_indices], time=step[xlim_indices], condition=None, value=None, color='red', ci=66, lw=1, ls='-', ax=ax)
            conditions = pd.Series(['low pupil'], name='trial type')
            sns.tsplot(kernels_pupil_l[:,xlim_indices], time=step[xlim_indices], condition=None, value=None, color='blue', ci=66, lw=1, ls='-', ax=ax)
            # ax.set_ylim(ylim)
            ax.set_xlim(xlim)
            ax.legend(loc='upper left', fontsize=6)
            # for i in range(kernels_h_hit.shape[1]):
            #     if p[1][i] < 0.05:
            #         ax.plot(step[i], ylim[0]+0.1, 'o', color='g', marker='o', markeredgecolor='w', ms=3)
            # if time_locked == 'stim_locked':
            #     sns.distplot(np.array(self.rt[pupil_h_ind])/1000.0, bins=20, hist=False, kde_kws={"shade": True}, color='r', ax=ax)
            #     sns.distplot(np.array(self.rt[pupil_l_ind])/1000.0, bins=20, hist=False, kde_kws={"shade": True}, color='b', ax=ax)
            # if time_locked == 'resp_locked':
            #     sns.distplot(0-(np.array(self.rt[pupil_h_ind])/1000.0), bins=20, hist=False, kde_kws={"shade": True}, color='r', ax=ax)
            #     sns.distplot(0-(np.array(self.rt[pupil_l_ind])/1000.0), bins=20, hist=False, kde_kws={"shade": True}, color='b', ax=ax)
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(0.25)
                ax.tick_params(width=0.25)
            y_limits = ax.get_ylim()
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.project_directory, 'figures', 'average_responses.pdf'))
        
    
    def grand_average_pupil_response(self,):
        
        # shell()
        
        sample_rate = 1000
        downsample_rate = 50
        new_sample_rate = sample_rate / downsample_rate
        
        kernel_cue_all = []
        kernel_cue_high = []
        kernel_cue_med = []
        kernel_cue_low = []
        kernel_cue_present = []
        kernel_cue_absent = []
        kernel_cue_yes = []
        kernel_cue_no = []
        kernel_cue_hit = []
        kernel_cue_fa = []
        kernel_cue_miss = []
        kernel_cue_cr = []
        
        kernel_choice_all = []
        kernel_choice_high = []
        kernel_choice_med = []
        kernel_choice_low = []
        kernel_choice_present = []
        kernel_choice_absent = []
        kernel_choice_yes = []
        kernel_choice_no = []
        kernel_choice_hit = []
        kernel_choice_fa = []
        kernel_choice_miss = []
        kernel_choice_cr = []
        
        parameters_joined2 = []
        for i, s in enumerate(self.subjects):
            runs = np.unique(self.run[self.subj_idx == i])
            sessions = [self.session[self.subj_idx == i][self.run[self.subj_idx == i] == r][0] - 1 for r in runs] 
            aliases = ['detection_{}_{}'.format(run, session) for run, session in zip(runs, sessions)]
            self.base_directory = os.path.join(self.project_directory, self.experiment_name, s)
            self.hdf5_filename = os.path.join(self.base_directory, 'processed', s + '.hdf5')
            self.ho = HDFEyeOperator.HDFEyeOperator(self.hdf5_filename)
            parameters_joined2.append(self.ho.read_session_data('', 'parameters_joined'))
            
            # load data:
            parameters = []
            pupil = []
            time = []
            cue_times = []
            choice_times = []
            blink_times = []
            time_to_add = 0
            for alias in aliases:
                parameters.append(self.ho.read_session_data(alias, 'parameters2'))
                self.alias = alias
                self.trial_times = self.ho.read_session_data(alias, 'trials')
                self.eye = self.ho.eye_during_period((np.array(self.trial_times['trial_start_EL_timestamp'])[0], np.array(self.trial_times['trial_end_EL_timestamp'])[-1]), self.alias)
                self.pupil_data = self.ho.data_from_time_period((np.array(self.trial_times['trial_start_EL_timestamp'])[0], np.array(self.trial_times['trial_end_EL_timestamp'])[-1]), self.alias)
                self.session_start = self.trial_times['trial_start_EL_timestamp'][0]
                p = np.array(self.pupil_data[(self.eye + '_pupil_bp_clean_psc')])
                p = p[-np.isnan(p)]
                pupil.append(p)
                ti = np.array(self.pupil_data['time']) - self.session_start
                time.append(ti + time_to_add)
                self.phase_times = self.ho.read_session_data(alias, 'trial_phases')
                cue_times.append( np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 2)]) - self.session_start + time_to_add )
                choice_times.append( np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 3)]) - self.session_start + time_to_add )
                self.blink_data = self.ho.read_session_data(alias, 'blinks_from_message_file')
                blink_times.append( np.array(self.blink_data['end_timestamp']) - self.session_start + time_to_add )
                time_to_add += ti[-1]
            
            parameters_joined = pd.concat(parameters)
            pupil = np.concatenate(pupil)
            time = np.concatenate(time)
            cue_times = np.concatenate(cue_times) / 1000.0
            choice_times = np.concatenate(choice_times) / 1000.0
            blink_times = np.concatenate(blink_times) / 1000.0
            omissions = np.array(parameters_joined.omissions, dtype=bool)
            pupil_m_ind = self.pupil_rest_ind[self.subj_idx == i]
            pupil_h_ind = self.pupil_h_ind[self.subj_idx == i]
            pupil_l_ind = self.pupil_l_ind[self.subj_idx == i]
            present_ind = self.present[self.subj_idx == i]
            absent_ind = self.absent[self.subj_idx == i]
            yes_ind = self.yes[self.subj_idx == i]
            no_ind = self.no[self.subj_idx == i]
            hit_ind = self.hit[self.subj_idx == i]
            fa_ind = self.fa[self.subj_idx == i]
            miss_ind = self.miss[self.subj_idx == i]
            cr_ind = self.cr[self.subj_idx == i]
            
            # event related averages:
            interval = 15
            cue_locked_array = np.zeros((sum(-omissions), interval*new_sample_rate))
            cue_locked_array[:,:] = np.NaN
            choice_locked_array = np.zeros((sum(-omissions), interval*new_sample_rate))
            choice_locked_array[:,:] = np.NaN
            for j, t in enumerate((cue_times[-omissions]-2)*sample_rate):
                cue_locked_array[j,:len(sp.signal.decimate(pupil[np.argmax(time>t):np.argmax(time>t)+(interval*sample_rate)], downsample_rate))] = sp.signal.decimate(pupil[np.argmax(time>t):np.argmax(time>t)+(interval*sample_rate)], downsample_rate) - self.pupil_b[self.subj_idx == i][j]
            for j, t in enumerate((choice_times[-omissions]-5)*sample_rate):
                choice_locked_array[j,:len(sp.signal.decimate(pupil[np.argmax(time>t):np.argmax(time>t)+(interval*sample_rate)], downsample_rate))] = sp.signal.decimate(pupil[np.argmax(time>t):np.argmax(time>t)+(interval*sample_rate)], downsample_rate) - self.pupil_b[self.subj_idx == i][j]
            
            kernel_cue_all.append( bn.nanmean(cue_locked_array, axis=0) )
            kernel_cue_high.append( bn.nanmean(cue_locked_array[pupil_h_ind,:], axis=0) )
            kernel_cue_med.append( bn.nanmean(cue_locked_array[pupil_m_ind,:], axis=0) )
            kernel_cue_low.append( bn.nanmean(cue_locked_array[pupil_l_ind,:], axis=0) )
            kernel_cue_hit.append( bn.nanmean(cue_locked_array[hit_ind,:], axis=0) )
            kernel_cue_fa.append( bn.nanmean(cue_locked_array[fa_ind,:], axis=0) )
            kernel_cue_miss.append( bn.nanmean(cue_locked_array[miss_ind,:], axis=0) )
            kernel_cue_cr.append( bn.nanmean(cue_locked_array[cr_ind,:], axis=0) )
            kernel_cue_present.append( (kernel_cue_hit[i]+kernel_cue_miss[i]) / 2.0 )
            kernel_cue_absent.append( (kernel_cue_fa[i]+kernel_cue_cr[i]) / 2.0 )
            kernel_cue_yes.append( (kernel_cue_hit[i]+kernel_cue_fa[i]) / 2.0 )
            kernel_cue_no.append( (kernel_cue_miss[i]+kernel_cue_cr[i]) / 2.0 )
            kernel_choice_all.append( bn.nanmean(choice_locked_array, axis=0) )
            kernel_choice_high.append( bn.nanmean(choice_locked_array[pupil_h_ind,:], axis=0) )
            kernel_choice_med.append( bn.nanmean(choice_locked_array[pupil_m_ind,:], axis=0) )
            kernel_choice_low.append( bn.nanmean(choice_locked_array[pupil_l_ind,:], axis=0) )
            kernel_choice_hit.append( bn.nanmean(choice_locked_array[hit_ind,:], axis=0) )
            kernel_choice_fa.append( bn.nanmean(choice_locked_array[fa_ind,:], axis=0) )
            kernel_choice_miss.append( bn.nanmean(choice_locked_array[miss_ind,:], axis=0) )
            kernel_choice_cr.append( bn.nanmean(choice_locked_array[cr_ind,:], axis=0) )
            kernel_choice_present.append( (kernel_choice_hit[i]+kernel_choice_miss[i]) / 2.0 )
            kernel_choice_absent.append( (kernel_choice_fa[i]+kernel_choice_cr[i]) / 2.0 )
            kernel_choice_yes.append( (kernel_choice_hit[i]+kernel_choice_fa[i]) / 2.0 )
            kernel_choice_no.append( (kernel_choice_miss[i]+kernel_choice_cr[i]) / 2.0 )
            
        conditions_data_cue = [[np.vstack(kernel_cue_high), np.vstack(kernel_cue_med), np.vstack(kernel_cue_low)], [np.vstack(kernel_cue_hit), np.vstack(kernel_cue_fa), np.vstack(kernel_cue_miss), np.vstack(kernel_cue_cr)], [np.vstack(kernel_cue_yes), np.vstack(kernel_cue_no)], [np.vstack(kernel_cue_present), np.vstack(kernel_cue_absent)], [np.vstack(kernel_cue_all)]]
        conditions_data_choice = [[np.vstack(kernel_choice_high), np.vstack(kernel_choice_med), np.vstack(kernel_choice_low)], [np.vstack(kernel_choice_hit), np.vstack(kernel_choice_fa), np.vstack(kernel_choice_miss), np.vstack(kernel_choice_cr)], [np.vstack(kernel_choice_yes), np.vstack(kernel_choice_no)], [np.vstack(kernel_choice_present), np.vstack(kernel_choice_absent)], [np.vstack(kernel_choice_all)]]
        conditions_labels = [['high', 'med', 'low'], ['H', 'FA', 'M', 'CR'], ['yes', 'no'], ['present', 'absent'], ['all']]
        conditions_colors = [['red', 'black', 'blue'], ['orange', 'orange', 'green', 'green'], ['orange', 'green'], ['orange', 'green'], ['black']]
        conditions_alphas = [[1, 1, 1], [1, 0.5, 0.5, 1], [1, 1], [1, 1], [1]]
        conditions_titles = ['pupil', 'SDT', 'yes', 'present', 'all']
        
        mean_rt = np.array([np.median(self.rt[self.subj_idx == i]) for i, s in enumerate(self.subjects)]).mean() / 1000.0
        
        for c in range(len(conditions_titles)):
            fig_spacing = (4,14)
            fig = plt.figure(figsize=(8.27, 5.845))
            ax1 = plt.subplot2grid(fig_spacing, (0, 0), colspan=2)
            ax2 = plt.subplot2grid(fig_spacing, (1, 0), colspan=2)
            axes = [ax1, ax2,]
            ax_nr = 0
            ylim = (-4,10)
            for time_locked in ['stim_locked', 'resp_locked']:
            
                if time_locked == 'stim_locked':
                    data = conditions_data_cue[c]
                    step_lim = [-2,13]
                    # xlim = [-1,12]
                    xlim = [-1,6]
                if time_locked == 'resp_locked':
                    data = conditions_data_choice[c]
                    step_lim = [-5,10]
                    # xlim = [-4,9]
                    xlim = [-4,3]
                
                # length kernel:
                kernel_length = conditions_data_cue[0][0].shape[1]
            
                # step size:
                if time_locked == 'stim_locked':                    
                    step = pd.Series(np.linspace(step_lim[0], step_lim[1], kernel_length), name='Time from cue (s)')
                if time_locked == 'resp_locked':
                    step = pd.Series(np.linspace(step_lim[0], step_lim[1], kernel_length), name='Time from report (s)')
                xlim_indices = np.array(step >= xlim[0]) * np.array(step <= xlim[1])
                
                # timeseries:
                ax = axes[ax_nr]
                ax.axvline(0, lw=0.5, alpha=0.5, color = 'k')
                if time_locked == 'stim_locked':
                    ax.axvline(mean_rt, ls='--', lw=0.5, alpha=0.5, color = 'k')
                if time_locked == 'resp_locked':
                    ax.axvline(mean_rt*-1.0, ls='--', lw=0.5, alpha=0.5, color = 'k')
                ax.axhline(0, lw=0.5, alpha=0.5, color = 'k')
                for cc in range(len(conditions_labels[c])):
                    sns.tsplot(data[cc][:,xlim_indices], time=step[xlim_indices], condition=conditions_labels[c][cc], value='Pupil response\n(% signal change)', color=conditions_colors[c][cc], alpha=conditions_alphas[c][cc], ci=66, lw=1, ls='-', ax=ax)
                ax.set_ylim(ylim)
                ax.set_xlim(xlim)
                # stats:
                ax.legend(loc='upper left', fontsize=6)
                if c == 1:
                    myfuncs.cluster_sig_bar_1samp(array=data[0][:,xlim_indices]-data[1][:,xlim_indices], x=np.array(step[xlim_indices]), yloc=1, color='orange', ax=ax, threshold=0.05, nrand=5000)
                    myfuncs.cluster_sig_bar_1samp(array=data[2][:,xlim_indices]-data[3][:,xlim_indices], x=np.array(step[xlim_indices]), yloc=2, color='green', ax=ax, threshold=0.05, nrand=5000)
                if c == 2 or c == 3:
                    myfuncs.cluster_sig_bar_1samp(array=data[0][:,xlim_indices]-data[1][:,xlim_indices], x=np.array(step[xlim_indices]), yloc=1, color='orange', ax=ax, threshold=0.05, nrand=5000)
                ax_nr+=1
            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(self.project_directory, 'figures', 'grand_mean_{}.pdf'.format(conditions_titles[c])))
    
    def SDT_correlation(self, bin_by='pupil_d', bins=10):
        
        for bin_by in ['pupil_d']:
            for y in ['c', 'd',]:
            # for y in ['rt']:
                # model_comp = 'bayes'
                model_comp = 'seq'
                fig, rs = self.behavior.SDT_correlation(bin_by=bin_by, n_bins=bins, y1=y, model_comp=model_comp)
                # fig = self.behavior.SDT_correlation(bin_by=bin_by, n_bins=bins, y1='correct', y2='correct')
                fig.savefig(os.path.join(self.project_directory, 'figures', 'SDT_correlation_{}_{}_{}.pdf'.format(bin_by, y, model_comp)))
    
    def SDT_across_time(self,):
        
        n_bins = 5
        sample_rate = 1000
        downsample_rate = 50
        new_sample_rate = sample_rate / downsample_rate
        for regress_rt in [True, False]:
        # for regress_rt in [False]:
            kernel_cue_dprime = []
            kernel_cue_criterion = []
            kernel_cue_pupil = []
            kernel_choice_dprime = []
            kernel_choice_criterion = []
            kernel_choice_pupil = []
            parameters_joined2 = []
            for i, s in enumerate(self.subjects):
                print
                print 'subject {}'.format(s)
                print
                runs = np.unique(self.run[self.subj_idx == i])
                sessions = [self.session[self.subj_idx == i][self.run[self.subj_idx == i] == r][0] - 1 for r in runs] 
                aliases = ['detection_{}_{}'.format(run, session) for run, session in zip(runs, sessions)]
                self.base_directory = os.path.join(self.project_directory, self.experiment_name, s)
                self.hdf5_filename = os.path.join(self.base_directory, 'processed', s + '.hdf5')
                self.ho = HDFEyeOperator.HDFEyeOperator(self.hdf5_filename)
                parameters_joined2.append(self.ho.read_session_data('', 'parameters_joined'))
        
                # load data:
                parameters = []
                pupil = []
                time = []
                cue_times = []
                choice_times = []
                blink_times = []
                time_to_add = 0
                for alias in aliases:
                    parameters.append(self.ho.read_session_data(alias, 'parameters2'))
                    self.alias = alias
                    self.trial_times = self.ho.read_session_data(alias, 'trials')
                    self.eye = self.ho.eye_during_period((np.array(self.trial_times['trial_start_EL_timestamp'])[0], np.array(self.trial_times['trial_end_EL_timestamp'])[-1]), self.alias)
                    self.pupil_data = self.ho.data_from_time_period((np.array(self.trial_times['trial_start_EL_timestamp'])[0], np.array(self.trial_times['trial_end_EL_timestamp'])[-1]), self.alias)
                    self.session_start = self.trial_times['trial_start_EL_timestamp'][0]
                    p = np.array(self.pupil_data[(self.eye + '_pupil_bp_clean_psc')])
                    p = p[-np.isnan(p)]
                    pupil.append(p)
                    ti = np.array(self.pupil_data['time']) - self.session_start
                    time.append(ti + time_to_add)
                    self.phase_times = self.ho.read_session_data(alias, 'trial_phases')
                    cue_times.append( np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 2)]) - self.session_start + time_to_add )
                    choice_times.append( np.array(self.phase_times['trial_phase_EL_timestamp'][(self.phase_times['trial_phase_index'] == 3)]) - self.session_start + time_to_add )
                    self.blink_data = self.ho.read_session_data(alias, 'blinks_from_message_file')
                    blink_times.append( np.array(self.blink_data['end_timestamp']) - self.session_start + time_to_add )
                    time_to_add += ti[-1]
                
                parameters_joined = pd.concat(parameters)
                omissions = np.array(parameters_joined.omissions, dtype=bool) + (np.array(parameters_joined.correct) == -1)
                parameters_joined = parameters_joined[~omissions]
                
                pupil = np.concatenate(pupil)
                time = np.concatenate(time)
                cue_times = (np.concatenate(cue_times) / 1000.0)[~omissions]
                choice_times = (np.concatenate(choice_times) / 1000.0)[~omissions]
                present_ind = np.array(parameters_joined['present'] == 1)
                absent_ind = ~present_ind
                yes_ind = np.array(parameters_joined['yes'] == 1)
                no_ind = ~yes_ind
                hit_ind = yes_ind * present_ind
                fa_ind = yes_ind * ~present_ind
                miss_ind = ~yes_ind * present_ind
                cr_ind = ~yes_ind * ~present_ind
                rt = np.array(parameters_joined['rt']) / 1000.0
                
                # event related averages:
                interval = 10
                cue_locked_array = np.zeros((parameters_joined.shape[0], interval*new_sample_rate))
                cue_locked_array[:,:] = np.NaN
                choice_locked_array = np.zeros((parameters_joined.shape[0], interval*new_sample_rate))
                choice_locked_array[:,:] = np.NaN
                for j, t in enumerate((cue_times-2)*sample_rate):
                    cue_locked_array[j,:len(sp.signal.decimate(pupil[floor(t):floor(t)+(interval*sample_rate)], downsample_rate))] = sp.signal.decimate(pupil[floor(t):floor(t)+(interval*sample_rate)], downsample_rate)
                for j, t in enumerate((choice_times-5)*sample_rate):
                    choice_locked_array[j,:len(sp.signal.decimate(pupil[floor(t):floor(t)+(interval*sample_rate)], downsample_rate))] = sp.signal.decimate(pupil[floor(t):floor(t)+(interval*sample_rate)], downsample_rate)
                
                len_signal = np.ceil(interval*new_sample_rate)
                cue_criterion = np.zeros(len_signal)
                cue_d_prime = np.zeros(len_signal)
                cue_pupil = np.zeros(len_signal)
                cue_xx = np.linspace(-2,8,len_signal)
                for j, k in enumerate(np.arange(0,len_signal)):
                    # scalars = bn.nanmean(cue_locked_array[:,k:k+width], axis=1) - self.pupil_b[self.subj_idx == i]
                    scalars = cue_locked_array[:,k] - cue_locked_array[:,1.5*new_sample_rate:2*new_sample_rate].mean(axis=1)
                    clean_trials = ~np.isnan(scalars)
                    scalars = scalars[clean_trials]
                    if regress_rt:
                        sessions = np.array(parameters_joined.session)[clean_trials]
                        for s in np.array(np.unique(sessions), dtype=int):
                            scalars[sessions==s] = myfuncs.lin_regress_resid(scalars[sessions==s], [rt[clean_trials][sessions==s]]) + scalars[sessions==s].mean()

                    inds_s = np.zeros((parameters_joined.shape[0], n_bins), dtype=bool)[clean_trials]
                    trial_nr = 0
                    for s in np.unique(parameters_joined.session):
                        bin_measure = scalars[np.array(parameters_joined.session == s)[clean_trials]]
                        nr_trials_in_run = len(bin_measure)
                        inds = np.array_split(np.argsort(bin_measure), n_bins)
                        for b in range(n_bins):
                            ind = np.zeros(len(bin_measure), dtype=bool)
                            ind[inds[b]] = True
                            inds_s[trial_nr:trial_nr+nr_trials_in_run, b] = ind
                        trial_nr += nr_trials_in_run

                    pupil_b = np.zeros(n_bins)
                    criterion_b = np.zeros(n_bins)
                    d_prime_b = np.zeros(n_bins)
                    for y, ind in enumerate(inds_s.T):
                        pupil_b[y] = scalars[ind].mean()
                        d_prime_b[y], criterion_b[y] = myfuncs.SDT_measures(target=present_ind[clean_trials][ind], hit=hit_ind[clean_trials][ind], fa=fa_ind[clean_trials][ind])
                        # d_prime_b[y] = rt[clean_trials][ind].mean()
                    cue_d_prime[j] = sp.stats.spearmanr(pupil_b, d_prime_b)[0]
                    cue_pupil[j] = pupil_b.mean()
                    cue_criterion[j] = sp.stats.spearmanr(pupil_b, criterion_b)[0]
                    
                choice_criterion = np.zeros(len_signal)
                choice_d_prime = np.zeros(len_signal)
                choice_pupil = np.zeros(len_signal)
                choice_xx = np.linspace(-5,5,len_signal)
                for j, k in enumerate(np.arange(0,len_signal)):
                    # scalars = bn.nanmean(choice_locked_array[:,k:k+width], axis=1) - self.pupil_b[self.subj_idx == i]
                    scalars = choice_locked_array[:,k] - choice_locked_array[:,(5-np.mean(rt)-0.5)*new_sample_rate:(5-np.mean(rt))*new_sample_rate].mean(axis=1)
                    # scalars = choice_locked_array[:,k] - cue_locked_array[:,1.5*new_sample_rate:2*new_sample_rate].mean(axis=1)
                    clean_trials = ~np.isnan(scalars)
                    scalars = scalars[clean_trials]
                    if regress_rt:
                        sessions = np.array(parameters_joined.session)[clean_trials]
                        for s in np.array(np.unique(sessions), dtype=int):
                            scalars[sessions==s] = myfuncs.lin_regress_resid(scalars[sessions==s], [rt[clean_trials][sessions==s]]) + scalars[sessions==s].mean()
                    
                    inds_s = np.zeros((parameters_joined.shape[0], n_bins), dtype=bool)[clean_trials]
                    trial_nr = 0
                    for s in np.unique(parameters_joined.session):
                        bin_measure = scalars[np.array(parameters_joined.session == s)[clean_trials]]
                        nr_trials_in_run = len(bin_measure)
                        inds = np.array_split(np.argsort(bin_measure), n_bins)
                        for b in range(n_bins):
                            ind = np.zeros(len(bin_measure), dtype=bool)
                            ind[inds[b]] = True
                            inds_s[trial_nr:trial_nr+nr_trials_in_run, b] = ind
                        trial_nr += nr_trials_in_run
                    
                    pupil_b = np.zeros(n_bins)
                    criterion_b = np.zeros(n_bins)
                    d_prime_b = np.zeros(n_bins)
                    for y, ind in enumerate(inds_s.T):
                        pupil_b[y] = scalars[ind].mean()
                        d_prime_b[y], criterion_b[y] = myfuncs.SDT_measures(target=present_ind[clean_trials][ind], hit=hit_ind[clean_trials][ind], fa=fa_ind[clean_trials][ind])
                        # d_prime_b[y] = rt[clean_trials][ind].mean()
                    choice_d_prime[j] = sp.stats.spearmanr(pupil_b, d_prime_b)[0]
                    choice_pupil[j] = pupil_b.mean()
                    choice_criterion[j] = sp.stats.spearmanr(pupil_b, criterion_b)[0]
                    
                kernel_cue_dprime.append(cue_d_prime)
                kernel_cue_criterion.append(cue_criterion)
                kernel_cue_pupil.append(cue_pupil)
                kernel_choice_dprime.append(choice_d_prime)
                kernel_choice_criterion.append(choice_criterion)
                kernel_choice_pupil.append(choice_pupil)
            
            conditions_data_cue = [[np.vstack(kernel_cue_dprime), np.vstack(kernel_cue_criterion), np.vstack(kernel_cue_pupil)]]
            conditions_data_choice = [[np.vstack(kernel_choice_dprime), np.vstack(kernel_choice_criterion), np.vstack(kernel_choice_pupil)]]
            
            for fix in range(conditions_data_choice[0][0].shape[0]):
                conditions_data_cue[0][0][fix,:] = pd.rolling_mean(conditions_data_cue[0][0][fix,:], window=5, center=True)
                conditions_data_cue[0][1][fix,:] = pd.rolling_mean(conditions_data_cue[0][1][fix,:], window=5, center=True)
                conditions_data_choice[0][0][fix,:] = pd.rolling_mean(conditions_data_choice[0][0][fix,:], window=5, center=True)
                conditions_data_choice[0][1][fix,:] = pd.rolling_mean(conditions_data_choice[0][1][fix,:], window=5, center=True)
            
            conditions_labels = [["d'", 'criterion', 'pupil'],]
            conditions_colors = [['mediumslateblue', 'k', 'pink'], ]
            conditions_alphas = [[1, 1, 0.5],]
            conditions_titles = ['SDT']
            ylocs = [1, 2, 3]
            mean_rt = np.array([np.median(self.rt[self.subj_idx == i]) for i, s in enumerate(self.subjects)]).mean() / 1000.0
            for c in range(len(conditions_titles)):
                fig = plt.figure(figsize=(6,2))
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                axes = [ax1, ax2,]
                ax_nr = 0
                ylim = (-0.8,0.4)
                for time_locked in ['stim_locked', 'resp_locked']:
                    # if time_locked == 'stim_locked':
                    #     data = conditions_data_cue[c]
                    #     step_lim = [cue_xx[0], cue_xx[-1]]
                    #     # xlim = [-1,12]
                    #     xlim = [-0.5,(mean_rt/2.0)]
                    # if time_locked == 'resp_locked':
                    #     data = conditions_data_choice[c]
                    #     step_lim = [choice_xx[0], choice_xx[-1]]
                    #     # xlim = [-4,9]
                    #     xlim = [-(mean_rt/2.0),3]
                    if time_locked == 'stim_locked':
                        data = conditions_data_cue[c]
                        step_lim = [cue_xx[0], cue_xx[-1]]
                        # xlim = [-1,12]
                        xlim = [-0.5,mean_rt+3]
                    if time_locked == 'resp_locked':
                        data = conditions_data_choice[c]
                        step_lim = [choice_xx[0], choice_xx[-1]]
                        # xlim = [-4,9]
                        xlim = [-mean_rt-0.5,3]
                    
                    # length kernel:
                    kernel_length = conditions_data_cue[0][0].shape[1]
        
                    # step size:
                    if time_locked == 'stim_locked':                    
                        step = pd.Series(np.linspace(step_lim[0], step_lim[1], kernel_length), name='Time from cue (s)')
                    if time_locked == 'resp_locked':
                        step = pd.Series(np.linspace(step_lim[0], step_lim[1], kernel_length), name='Time from report (s)')
                    xlim_indices = np.array(step >= xlim[0]) * np.array(step <= xlim[1])
                    
                    # timeseries:
                    ax = axes[ax_nr]
                    ax.axvline(0, lw=0.5, alpha=0.5, color = 'k')
                    if time_locked == 'stim_locked':
                        ax.axvline(mean_rt, ls='--', lw=0.5, alpha=0.5, color = 'k')
                    if time_locked == 'resp_locked':
                        ax.axvline(mean_rt*-1.0, ls='--', lw=0.5, alpha=0.5, color = 'k')
                    ax.axhline(0, lw=0.5, alpha=0.5, color = 'k')
                    for cc in range(2):
                        sns.tsplot(data[cc][:,xlim_indices], time=step[xlim_indices], condition=conditions_labels[c][cc], value='Correlation coefficient', color=conditions_colors[c][cc], alpha=conditions_alphas[c][cc], ci=66, lw=1, ls='-', ax=ax)
                    ax.set_ylim(ylim)
                    ax.set_xlim(xlim)
                    ax.legend(loc='upper left', fontsize=6)
                    # stats:
                    for cc in range(2):
                        myfuncs.cluster_sig_bar_1samp(array=data[cc][:,xlim_indices], x=np.array(step[xlim_indices]), yloc=ylocs[cc], color=conditions_colors[c][cc], ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)
                    ax = ax.twinx()
                    sns.tsplot(data[2][:,xlim_indices], time=step[xlim_indices], condition=conditions_labels[c][2], value='Pupil size\n(% signal change)', color=conditions_colors[c][2], alpha=conditions_alphas[c][2], ci=66, lw=1, ls='-', ax=ax)
                    ax.set_ylim(ylim[0]*15, ylim[1]*15)
                    
                    ax_nr+=1
                sns.despine(offset=10, trim=True, right=False)
                plt.tight_layout()
                if regress_rt:
                    fig.savefig(os.path.join(self.project_directory, 'figures', 'sliding_1_{}.pdf'.format(conditions_titles[c])))
                else:
                    fig.savefig(os.path.join(self.project_directory, 'figures', 'sliding_0_{}.pdf'.format(conditions_titles[c])))
                    
    def binned_baseline_pupil_behavior(self):
        
        shell()
        
        from lmfit import minimize, Parameters, Parameter, report_fit
        from lmfit.models import QuadraticModel
        
        width = 100
        acc_ylim = (0.5,1.0)
        rt_ylim = (1.5,3.0)
        max_rt = 5
        
        correct_curves = []
        rt_curves = []
        
        fig = plt.figure(figsize=(10,8))
        plt_nr = 1
        rt_v_h = []
        rt_v_l = []
        chi = []
        for i in range(len(self.subjects)):
            rt = np.array(self.parameters_joined['rt'])[self.subj_idx == i] / 1000.0
            correct = np.array(self.parameters_joined['correct'])[self.subj_idx == i]
            scalars = np.array(self.parameters_joined['pupil_b'])[self.subj_idx == i]
            scalars_sorted = np.argsort(scalars)
            baseline_curve = np.zeros(scalars.shape[0]-width)
            correct_curve = np.zeros(scalars.shape[0]-width)
            rt_curve = np.zeros(scalars.shape[0]-width)
            for j in range(0,scalars.shape[0]-width):
                baseline_curve[j] = np.mean(scalars[scalars_sorted[j:j+width]])
                correct_curve[j] = np.mean(correct[scalars_sorted[j:j+width]])
                rt_curve[j] = np.mean(rt[scalars_sorted[j:j+width]][rt[scalars_sorted[j:j+width]] < max_rt])
            
            x = baseline_curve.copy()
            mod = QuadraticModel()
            params = Parameters()
            params.add('c', value=0, min=-np.inf, max=np.inf)
            params.add('b', value=0, min=-np.inf, max=np.inf)
            params.add('a', value=-0.5, min=-np.inf, max=-1e-25)
            # params.add('a', value=-0.5, min=-np.inf, max=np.inf)
            d_prime_result = mod.fit(correct_curve, params, x=x,)
            d_prime_fit = d_prime_result.eval()
            correct_curves.append(d_prime_result.eval(x=np.linspace(-10,10,100)))
            
            x = baseline_curve.copy()
            mod = QuadraticModel()
            params = Parameters()
            params.add('c', value=0, min=-np.inf, max=np.inf)
            params.add('b', value=0, min=-np.inf, max=np.inf)
            params.add('a', value=0.5, min=1e-25, max=np.inf)
            # params.add('a', value=-0.5, min=-np.inf, max=np.inf)
            rt_result = mod.fit(rt_curve, params, x=x,)
            rt_fit = rt_result.eval()
            rt_curves.append(rt_result.eval(x=np.linspace(-10,10,100)))
            
            ax = fig.add_subplot(5,5,plt_nr)
            ax.plot(baseline_curve, correct_curve, color='grey', linewidth=1)
            ax.plot(baseline_curve, d_prime_fit, color='black', linewidth=1)
            ax.set_title('Subj. {}'.format(i+1))
            ax.set_xlabel('Baseline pupil size (% signal change)')
            ax.set_ylabel("Accuracy")
            ax.set_ylim(acc_ylim)
            
            ax = ax.twinx()
            ax.plot(baseline_curve, rt_curve, color='pink', linewidth=1)
            ax.plot(baseline_curve, rt_fit, color='red', linewidth=1)
            ax.set_ylabel("RT (s)")
            ax.set_ylim(rt_ylim)
            
            plt_nr+=1
        
        ax = fig.add_subplot(5,5,plt_nr)
        ax.plot(np.linspace(-10,10,100), np.vstack(correct_curves).mean(axis=0), color='black', linewidth=1.5)
        ax.fill_between(np.linspace(-10,10,100), np.vstack(correct_curves).mean(axis=0)-sp.stats.sem(np.vstack(correct_curves), axis=0), np.vstack(correct_curves).mean(axis=0)+sp.stats.sem(np.vstack(correct_curves), axis=0), color='black', alpha=0.25)
        ax.set_title('Group average (N={})'.format(len(self.subjects)))
        ax.set_xlabel('Baseline pupil size (% signal change)')
        ax.set_ylabel("Accuracy")
        
        ax = ax.twinx()
        ax.plot(np.linspace(-10,10,100), np.vstack(rt_curves).mean(axis=0), color='r', linewidth=1.5)
        ax.fill_between(np.linspace(-10,10,100), np.vstack(rt_curves).mean(axis=0)-sp.stats.sem(np.vstack(rt_curves), axis=0), np.vstack(rt_curves).mean(axis=0)+sp.stats.sem(np.vstack(rt_curves), axis=0), color='r', alpha=0.25)
        ax.set_ylim(rt_ylim)
        
        sns.despine(offset=10, trim=True, right=False)
        fig.tight_layout()
        fig.savefig(os.path.join(self.project_directory, 'figures', 'baseline2.pdf'))
    
    
    def binned_baseline_pupil_behavior_2(self):
        
        from sklearn.utils import resample
        
        shell()
        n_bins = 7
        nrand = 10000
        
        # for measure in ["d'", 'criterion', 'RT', 'RT_var']:
        
        fig = plt.figure(figsize=(10,8))
        plt_nr = 1
        
        rt_v_h = []
        rt_v_l = []
        chi = []
        for i in range(len(self.subjects)):
            scalars = np.array(self.parameters_joined['pupil_b'])[self.subj_idx == i]            
            inds = np.array_split(np.argsort(scalars), n_bins)
        
        
            inds_s = np.zeros((len(scalars), n_bins), dtype=bool)
            trial_nr = 0
            for s in np.array(np.unique(self.session[self.subj_idx == i]), dtype=int):
                bin_measure = scalars[self.session[self.subj_idx == i] == s]
                nr_trials_in_run = len(bin_measure)
                inds = np.array_split(np.argsort(bin_measure), n_bins)
                for b in range(n_bins):
                    ind = np.zeros(len(bin_measure), dtype=bool)
                    ind[inds[b]] = True
                    inds_s[trial_nr:trial_nr+nr_trials_in_run, b] = ind
                trial_nr += nr_trials_in_run
            
            pupil_b = np.zeros(n_bins)
            accuracy_b = np.zeros(n_bins)
            criterion_b = np.zeros(n_bins)
            d_prime_b = np.zeros(n_bins)
            RT_b = np.zeros(n_bins)
            RT_v_b = np.zeros(n_bins)
            
            accuracy_b_err = np.zeros(n_bins)
            criterion_b_err = np.zeros(n_bins)
            d_prime_b_err = np.zeros(n_bins)
            RT_b_err = np.zeros(n_bins)
            RT_v_b_err = np.zeros(n_bins)
            for y, ind in enumerate(inds_s.T):
                
                pupil_b[y] = scalars[ind].mean()
                d_prime_b[y], criterion_b[y] = myfuncs.SDT_measures(target=self.present[self.subj_idx == i][ind], hit=self.hit[self.subj_idx == i][ind], fa=self.fa[self.subj_idx == i][ind])
                RT_b[y] = (self.rt[self.subj_idx == i][ind] / 1000.0).mean()
                RT_v_b[y] = (self.rt[self.subj_idx == i][ind] / 1000.0).std()
                accuracy_b[y] = (self.correct[(self.subj_idx == i)][ind]).mean()
                
                acc_err = []
                criterion_err = []
                d_prime_err = []
                RT_err = []
                RT_v_err = []
                
                indices = np.arange(sum(ind))
                for nrand in range(nrand):
                    
                    indices_resampled = resample(indices)
                    
                    d, c = myfuncs.SDT_measures(target=self.present[self.subj_idx == i][ind][indices_resampled], hit=self.hit[self.subj_idx == i][ind][indices_resampled], fa=self.fa[self.subj_idx == i][ind][indices_resampled])
                    d_prime_err.append(d)
                    criterion_err.append(c)
                    acc_err.append((self.correct[(self.subj_idx == i)][ind][indices_resampled]).mean())
                    RT_err.append((self.rt[self.subj_idx == i][ind][indices_resampled] / 1000.0).mean())
                    RT_v_err.append((self.rt[self.subj_idx == i][ind][indices_resampled] / 1000.0).std())
                criterion_b_err[y] = np.array(d_prime_err).std()
                d_prime_b_err[y] = np.array(criterion_err).std()
                accuracy_b_err[y] = np.array(acc_err).std()
                RT_b_err[y] = np.array(RT_err).std()
                RT_v_b_err[y] = np.array(RT_v_err).std()
            
            
            # z = np.polyfit(pupil_b, d_prime_b, deg=2, w=1.0/d_prime_b_err)
            # f = np.poly1d(z)
            # x_new = np.linspace(pupil_b[0], pupil_b[-1], n_bins)
            # d_prime_fit = f(x_new)
            # z_v = np.polyfit(pupil_b, RT_v_b, deg=2, w=1.0/RT_v_b_err)
            # f_v = np.poly1d(z_v)
            # x_new = np.linspace(pupil_b[0], pupil_b[-1], n_bins)
            # RT_v_fit = f_v(x_new)
            # peak = f.deriv()[0] / -f.deriv()[1]
            
            from lmfit import minimize, Parameters, Parameter, report_fit
            from lmfit.models import QuadraticModel
            
            # def poly(params, x):
            #     a = params['a']
            #     b = params['b']
            #     c = params['c']
            #     return c + (b*x) + ((a*x)**2)
            #
            # def poly_ls(params, x, data):
            #     a = params['a']
            #     b = params['b']
            #     c = params['c']
            #     model = c + (b*x) + ((a*x)**2)
            #     return model - data
            
            x = np.linspace(pupil_b[0], pupil_b[-1], n_bins)
            
            mod = QuadraticModel()
            params = Parameters()
            params.add('c', value=0, min=-np.inf, max=np.inf)
            params.add('b', value=0, min=-np.inf, max=np.inf)
            params.add('a', value=-0.5, min=-np.inf, max=-1e-25)
            d_prime_result = mod.fit(d_prime_b, params, x=x, weights=1.0/d_prime_b_err)
            d_prime_fit = d_prime_result.eval()
            
            mod = QuadraticModel()
            params = Parameters()
            params.add('c', value=0, min=-np.inf, max=np.inf)
            params.add('b', value=0, min=-np.inf, max=np.inf)
            params.add('a', value=0, min=-np.inf, max=np.inf)
            RT_v_result  = mod.fit(RT_v_b, params, x=x, weights=1.0/RT_v_b_err)
            RT_v_fit = RT_v_result.eval()
            
            chi.append(d_prime_result.chisqr)
            
            peak = d_prime_result.values['b'] / -(2*d_prime_result.values['a'])
            
            good_bins = np.argsort(abs(peak-pupil_b))[0:3]
            bad_bins = np.argsort(abs(peak-pupil_b))[3:]
            rt_var_bad = RT_v_b[bad_bins].mean()
            rt_var_good = RT_v_b[good_bins].mean()
            rt_var_bad_sem = sp.stats.sem(RT_v_b[bad_bins])
            rt_var_good_sem = sp.stats.sem(RT_v_b[good_bins])
            rt_v_h.append(rt_var_good)
            rt_v_l.append(rt_var_bad)
            
            trials_good = np.array(np.sum(inds_s[:,good_bins], axis=1), dtype=bool)
            trials_bad = np.array(np.sum(inds_s[:,bad_bins], axis=1), dtype=bool)
            np.save(os.path.join(self.project_directory, 'data', 'across', 'baseline_splits', 'baselines_bad_{}.npy'.format(self.subjects[i])), trials_good)
            np.save(os.path.join(self.project_directory, 'data', 'across', 'baseline_splits', 'baselines_good_{}.npy'.format(self.subjects[i])), trials_bad)
            
            ax = fig.add_subplot(5,5,plt_nr)
            ax.axvspan(min(pupil_b[good_bins]), max(pupil_b[good_bins]), facecolor='g', alpha=0.1)            
            ax.errorbar(pupil_b, d_prime_b, yerr=d_prime_b_err, fmt='o', color='k',)
            ax.plot(pupil_b, d_prime_fit, color='k', linestyle='--', linewidth=1.5)
            ax.set_title('Subj. {}\n Chi2={}'.format(i+1, round(d_prime_result.chisqr),3),)
            ax.set_xlabel('baseline pupil size (% signal change)')
            ax.set_ylabel("d'")
            # ax = ax.twinx()
            # ax.errorbar(pupil_b, RT_v_b, yerr=RT_v_b_err, alpha=0.5, fmt='o', color='r')
            # ax.plot(pupil_b, RT_v_fit, alpha=0.5, color='r', linestyle='--', linewidth=1.5)
            # ax.set_ylabel('RT (s.d.)')
            
            plt_nr+=1
        
        
        sns.despine(offset=10, trim=True)
        fig.tight_layout()
        fig.savefig(os.path.join(self.project_directory, 'figures', 'baseline.pdf'))
        
        
        good = []
        bad = []
        for i in range(len(self.subjects)):
            good.append( np.load(os.path.join(self.project_directory, 'data', 'across', 'baseline_splits', 'baselines_bad_{}.npy'.format(self.subjects[i]))) )
            bad.append( np.load(os.path.join(self.project_directory, 'data', 'across', 'baseline_splits', 'baselines_good_{}.npy'.format(self.subjects[i]))) )
        good = np.array(np.concatenate(good), dtype=bool)
        bad = np.array(np.concatenate(bad), dtype=bool)    
        
        
        
        rt_good = np.array([self.rt[(self.subj_idx==s) & good].mean() for s in range(len(self.subjects))]) / 1000.0
        rt_bad = np.array([self.rt[(self.subj_idx==s) & bad].mean() for s in range(len(self.subjects))]) / 1000.0
        
        rt_v_good = np.array([self.rt[(self.subj_idx==s) & good].std() for s in range(len(self.subjects))]) / 1000.0
        rt_v_bad = np.array([self.rt[(self.subj_idx==s) & bad].std() for s in range(len(self.subjects))]) / 1000.0
        
        
        measures_1 = [rt_good, rt_v_good]
        measures_0 = [rt_bad, rt_v_bad]
        
        titles = ['rt', 'rt_var']
        ylim_max = [3.0, 0.5]
        ylim_min = [0.0, 0,]
        
        for m in range(len(measures_0)):
        
            MEANS = (measures_1[m].mean(), measures_0[m].mean())
            SEMS = (sp.stats.sem(measures_1[m]), sp.stats.sem(measures_0[m]))
            N = 2
            ind = np.linspace(0,N/2,N)
            bar_width = 0.50
            fig = plt.figure(figsize=(1.25,1.75))
            ax = fig.add_subplot(111)
            for i in range(N):
                ax.bar(ind[i], height=MEANS[i], width = bar_width, yerr = SEMS[i], color = ['g','r'][i], alpha = 1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
            ax.set_ylim( ylim_min[m],ylim_max[m] )
            ax.set_title('N={}'.format(self.nr_subjects), size=7)
            ax.set_ylabel(titles[m], size=7)
            ax.set_xticks( (ind) )
            ax.set_xticklabels( ('good', 'bad'), rotation=45 )
            plt.text(x=np.mean((ind[0], ind[1])), y=ax.axis()[3] - ((ax.axis()[3]-ax.axis()[2]) / 10.0), s='p = {}'.format(round(myfuncs.permutationTest(measures_0[m], measures_1[m], paired=True)[1],3)), size=6, horizontalalignment='center')
            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(self.project_directory, 'figures', 'baseline_bars_{}.pdf'.format(titles[m])))
        
        
        
        
        
        
        chi = np.array(chi)
        bad_fits = chi >= 25
        # bad_fits[:] = False
        
        rt_var_bad = np.array(rt_v_l).mean()
        rt_var_good = np.array(rt_v_h).mean()
        rt_var_bad_sem = sp.stats.sem(np.array(rt_v_l))
        rt_var_good_sem = sp.stats.sem(np.array(rt_v_h))
        
        p = myfuncs.permutationTest(np.array(rt_v_h)[-bad_fits], np.array(rt_v_l)[-bad_fits], paired=True)[1]
        p_all = myfuncs.permutationTest(np.array(rt_v_h), np.array(rt_v_l), paired=True)[1]
        
        fig = plt.figure(figsize=(2,2))
        ax = fig.add_subplot(111)
        ax.bar([0,1], [rt_var_bad, rt_var_good], yerr=[rt_var_bad_sem, rt_var_good_sem], align='center', color=['r','g'], edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0,)
        ax.set_title('N={}/{}, p={}\nN={}/{}, p={}'.format(len(bad_fits), len(bad_fits), round(p_all,3), sum(~bad_fits), len(bad_fits), round(p,3)))
        ax.set_xticks([0,1])
        ax.set_xticklabels(['bad', 'good'], rotation=45)
        ax.set_ylabel('RT varibility (s.d.)')
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.project_directory, 'figures', 'baseline_bars.pdf'))
        
    def correlation_PPRa_BPD(self):
        
        sns.set_style('darkgrid')
        
        fig = plt.figure(figsize=(15,8))
        for i in range(len(self.subjects)):
        
            varX = self.pupil_b[self.subj_idx == i]
            varY = self.pupil_d[self.subj_idx == i]
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(varX,varY)
            (m,b) = sp.polyfit(varX, varY, 1)
            regression_line = sp.polyval([m,b], varX)
            
            ax = fig.add_subplot(3,5,i+1)
            ax.plot(varX,regression_line, color = 'k', linewidth = 1.5)
            ax.scatter(varX, varY, color='#808080', alpha = 0.75, rasterized=True)
            ax.set_title('subj.' + str(i+1) + ' (r = ' + str(round(r_value, 3)) + ')', size = 12)
            ax.set_ylabel('phasic response (% signal change)', size = 10)
            ax.set_xlabel('baseline (% signal change)', size = 10)
            plt.tick_params(axis='both', which='major', labelsize=10)
            # if round(p_value,5) < 0.005:
            #     ax.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/8), plt.axis()[2]+((abs(plt.axis()[2])+abs(plt.axis()[3]))/8),'r = ' + str(round(r_value, 3)) + '\np < 0.005', size = 12)
            # else:    
            #     ax.text(plt.axis()[0]+((abs(plt.axis()[0])+abs(plt.axis()[1]))/8), plt.axis()[2]+((abs(plt.axis()[2])+abs(plt.axis()[3]))/8),'r = ' + str(round(r_value, 3)) + '\np = ' + str(round(p_value, 5)), size = 12)
            plt.tight_layout()
            # plt.gca().spines["bottom"].set_linewidth(.5)
            # plt.gca().spines["left"].set_linewidth(.5)
        
        fig.savefig(os.path.join(self.project_directory, 'figures', 'correlation_bpd_ppr.pdf'))    

    def GLM_betas(self):
        
        import rpy2.robjects as robjects
        import rpy2.rlike.container as rlc
        
        for b in [0]:

            # get data in place:            
            ppr = np.array([self.pupil_d[self.subj_idx == s].mean() for s in range(len(self.subjects))])
            betas = np.vstack([np.load(os.path.join(self.project_directory, 'data', 'across', 'pupil_GLM', 'GLM_betas_{}_{}.npy'.format(b,s))) for s in self.subjects])
            Rs = np.vstack([np.load(os.path.join(self.project_directory, 'data', 'across', 'pupil_GLM', 'GLM_betas_{}_R_{}.npy'.format(b,s))) for s in self.subjects])
            df = pd.DataFrame(betas, columns=['cue', 'choice', 'box'])
            df['subject'] = np.arange(len(self.subjects))
            rois = 'cue', 'choice', 'box'
            rois_name = 'cue', 'choice', 'box'
            colors = ['grey']
            df = df.ix[:,np.concatenate((np.array(rois), np.array(['subject'])))]
            k = df.groupby(['subject']).mean()
            k_s = k.stack().reset_index()
            k_s.columns = ['subject', 'area', 'bold']
            
            # plot:
            locs = np.arange(0,len(rois))
            bar_width = 0.2
            fig = plt.figure(figsize=(1.5,2))
            ax = fig.add_subplot(111)
            sns.barplot(x='area',  y='bold', units='subject', data=k_s, palette=colors, ci=None, linewidth=0, alpha=0.5, ax=ax)
            sns.stripplot(x="area", y="bold", data=k_s, jitter=False, size=2, palette=colors, edgecolor='black', linewidth=0.25, ax=ax, split=True, alpha=1)
            values = np.vstack((k_s[(k_s['area'] == rois[0])].bold, k_s[(k_s['area'] == rois[1])].bold, k_s[(k_s['area'] == rois[2])].bold))
            x = locs.copy()
            ax.plot(x, values, color='black', lw=0.5, alpha=0.5)
            # add p-values:
            for r in range(len(rois)):
                p1 = myfuncs.permutationTest(k_s[(k_s['area']==rois[r])].bold, np.zeros(len(self.subjects)), paired=True)[1]
                if p1 < 0.05:
                    plt.text(s='{}'.format(round(p1, 3)), x=locs[r], y=plt.gca().get_ylim()[1]-((plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]) / 10.0), size=5, horizontalalignment='center',)
            # ax.legend_.remove()
            plt.xticks(locs, rois_name, rotation=45)
            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(self.project_directory, 'figures', 'betas_across_{}.pdf'.format(b)))
            
            print myfuncs.permutationTest(betas[:,0],betas[:,1], paired=True)[1]
            print myfuncs.permutationTest(betas[:,1],betas[:,2], paired=True)[1]
        
            print 'mean R2 = {}'.format(np.mean(Rs**2))
            print 'range R2 = [{} - {}]'.format(np.min(Rs**2), np.max(Rs**2))
            
            titles = ['cue', 'choice', 'box']
            bs = [betas[:,0], betas[:,1], betas[:,2]]
            
            for beta, title in zip(bs, titles):
                fig = plt.figure(figsize=(2,2))
                ax = fig.add_subplot(111)
                myfuncs.correlation_plot(beta, ppr, line=True, ax=ax)
                plt.ylabel('Pupil response\n(% signal change)')
                plt.xlabel(title + ' beta')
                sns.despine(offset=10, trim=True)
                plt.tight_layout()
                fig.savefig(os.path.join(self.project_directory, 'figures', 'betas_correlation_{}.pdf'.format(title)))
            
            # get data in place:
            betas = np.vstack([np.load(os.path.join(self.project_directory, 'data', 'across', 'pupil_GLM', 'GLM_betas_{}_{}_pupil_split.npy'.format(b,s))) for s in self.subjects])
            df = pd.DataFrame(np.vstack(( np.concatenate((betas[:,1], betas[:,0])), np.concatenate((betas[:,3], betas[:,2])), np.concatenate((betas[:,5], betas[:,4])), )).T, columns=['cue', 'choice', 'box'])
            df['subject'] = np.concatenate((np.arange(len(self.subjects)), np.arange(len(self.subjects))))
            df['pupil'] = np.concatenate((np.ones(len(self.subjects)), np.zeros(len(self.subjects))))
            trial_type = 'pupil'
            rois = 'cue', 'choice', 'box'
            rois_name = 'cue', 'choice', 'box'
            colors = ['r', 'b']
            df = df.ix[:,np.concatenate((np.array(rois), np.array(['subject', trial_type])))]
            k = df.groupby(['subject', trial_type]).mean()
            k_s = k.stack().reset_index()
            k_s.columns = ['subject', trial_type, 'area', 'bold']
            
            # plot:
            locs = np.arange(0,len(rois))
            bar_width = 0.2
            fig = plt.figure(figsize=( (1+(len(rois)*0.3)),2))
            ax = fig.add_subplot(111)
            sns.barplot(x='area',  y='bold', units='subject', hue=trial_type, hue_order=[0,1], data=k_s, palette=colors, ci=None, linewidth=0, alpha=0.5, ax=ax)
            sns.stripplot(x="area", y="bold", hue=trial_type, hue_order=[0,1], data=k_s, jitter=False, size=2, palette=colors, edgecolor='black', linewidth=0.25, ax=ax, split=True, alpha=1)
            for r in range(len(rois)):
                values = np.vstack((k_s[(k_s['area'] == rois[r]) & (k_s[trial_type] == 0)].bold, k_s[(k_s['area'] == rois[r]) & (k_s[trial_type] == 1)].bold))
                x = np.array([locs[r]-bar_width, locs[r]+bar_width])
                ax.plot(x, values, color='black', lw=0.5, alpha=0.5)
            # add p-values:
            for r in range(len(rois)):
                p1 = myfuncs.permutationTest(k_s[(k_s[trial_type]==0) & (k_s['area']==rois[r])].bold, k_s[(k_s[trial_type]==1) & (k_s['area']==rois[r])].bold, paired=True)[1]
                # if p1 < 0.05:
                plt.text(s='{}'.format(round(p1, 3)), x=locs[r], y=plt.gca().get_ylim()[1]-((plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]) / 10.0), size=5, horizontalalignment='center',)
            ax.legend_.remove()
            plt.xticks(locs, rois_name, rotation=45)
            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(self.project_directory, 'figures', 'betas_across_{}_pupil_split.pdf'.format(b)))
            
            # shell()
            
            # ANOVA:
            data = np.concatenate((betas[:,0], betas[:,1], betas[:,2], betas[:,3], betas[:,4], betas[:,5]))
            subject = np.concatenate((np.arange(len(self.subjects)), np.arange(len(self.subjects)), np.arange(len(self.subjects)), np.arange(len(self.subjects)), np.arange(len(self.subjects)), np.arange(len(self.subjects))))
            component = np.concatenate((np.zeros(len(self.subjects)), np.zeros(len(self.subjects)), np.ones(len(self.subjects)), np.ones(len(self.subjects)), 2*np.ones(len(self.subjects)), 2*np.ones(len(self.subjects))))
            pupil = np.concatenate((np.zeros(len(self.subjects)), np.ones(len(self.subjects)), np.zeros(len(self.subjects)), np.ones(len(self.subjects)), np.zeros(len(self.subjects)), np.ones(len(self.subjects))))
            
            d = rlc.OrdDict([('component', robjects.FactorVector(list(component.ravel()))), ('pupil', robjects.FactorVector(list(pupil.ravel()))), ('subject', robjects.FactorVector(list(subject.ravel()))), ('data', robjects.FloatVector(list(data.ravel())))])
            robjects.r.assign('dataf', robjects.DataFrame(d))
            robjects.r('attach(dataf)')
            statres = robjects.r('res = summary(aov(data ~ component*pupil + Error(subject/(component*pupil)), dataf))')
            # statres = robjects.r('res = summary(aov(data ~ component*pupil + Error(subject), dataf))')
            
            text_file = open(os.path.join(self.project_directory, 'figures', 'ANOVA_{}.txt'.format(int(1))), 'w')
            for string in statres:
                text_file.write(str(string))
            text_file.close()
            
            
            print statres
            
            shell()
            
    def mean_slow_drift(self,):
        
        # shell()
        
        baseline_2 = []
        baseline = []
        gaze_x = []
        gaze_y = []
        
        parameters_joined2 = []
        for i, s in enumerate(self.subjects):
            runs = np.unique(self.run[self.subj_idx == i])
            sessions = [self.session[self.subj_idx == i][self.run[self.subj_idx == i] == r][0] - 1 for r in runs] 
            aliases = ['detection_{}_{}'.format(run, session) for run, session in zip(runs, sessions)]
        
            self.base_directory = os.path.join(self.project_directory, self.experiment_name, s)
            self.hdf5_filename = os.path.join(self.base_directory, 'processed', s + '.hdf5')
            self.ho = HDFEyeOperator.HDFEyeOperator(self.hdf5_filename)
            parameters_joined2.append(self.ho.read_session_data('', 'parameters_joined'))
            
            downsample_rate = 50 # 50
            new_sample_rate = 1000 / downsample_rate

            # load data:
            parameters = []
            pupil = []
            pupil2 = []
            g_x = []
            g_y = []
            min_length = 1000000000000
            for alias in aliases:
                
                parameters.append(self.ho.read_session_data(alias, 'parameters2'))
                self.alias = alias
                self.trial_times = self.ho.read_session_data(alias, 'trials')
                
                # load pupil:
                self.eye = self.ho.eye_during_period((np.array(self.trial_times['trial_start_EL_timestamp'])[0], np.array(self.trial_times['trial_end_EL_timestamp'])[-1]), self.alias)
                self.pupil_data = self.ho.data_from_time_period((np.array(self.trial_times['trial_start_EL_timestamp'])[0], np.array(self.trial_times['trial_end_EL_timestamp'])[-1]), self.alias)
                
                # load times:
                self.session_start = self.trial_times['trial_start_EL_timestamp'][0]
                
                # load pupil:
                p = np.array(self.pupil_data[(self.eye + '_pupil_lp_clean_psc')])
                p = p[-np.isnan(p)]
                
                # load x:
                x = np.array(self.pupil_data[(self.eye + '_gaze_x')])
                x = x[-np.isnan(x)]
                
                # load y:
                y = np.array(self.pupil_data[(self.eye + '_gaze_y')])
                y = y[-np.isnan(y)]
                
                # regress out x & y from pupil:
                p2 = myfuncs.lin_regress_resid(p, [x,y])
                
                # save:
                pp = np.zeros(1000000)
                pp[:] = np.NaN
                pp[0:len(p)] = p
                pupil.append(pp)

                pp = np.zeros(1000000)
                pp[:] = np.NaN
                pp[0:len(p2)] = p2
                pupil2.append(pp)

                pp = np.zeros(1000000)
                pp[:] = np.NaN
                pp[0:len(x)] = x
                g_x.append(pp)

                pp = np.zeros(1000000)
                pp[:] = np.NaN
                pp[0:len(y)] = y
                g_y.append(pp)
                
            baseline_2.append(np.vstack(pupil2).mean(axis=0))
            baseline.append(np.vstack(pupil).mean(axis=0))
            gaze_x.append(np.vstack(g_x).mean(axis=0))
            gaze_y.append(np.vstack(g_y).mean(axis=0))
        
        
        for data, title, min_amp, max_amp, start_value in zip([baseline, baseline_2, gaze_x, gaze_y], ['pupil', 'pupil2', 'gaze_x', 'gaze_y'], [0, 0, -1000, -1000], [1000, 1000, 0, 0], [1, 1,-1,-1]):
        
            baselines = np.vstack(data)
            baselines = baselines[:,:500000]
            for i in range(baselines.shape[0]):
                # baselines[i,:] = myfuncs.smooth(baselines[i,:], window_len=30000)
                baselines[i,:] = baselines[i,:] - baselines[i,:].mean()
        
            baseline_mean = bn.nanmean(baselines, axis=0)
            baseline_sem = bn.nanstd(baselines, axis=0) / np.sqrt(len(self.subjects))
        
            downsample_rate = 100
        
            baseline_mean_f = sp.signal.decimate(baseline_mean, downsample_rate, 1)[40:-40]
            baseline_sem_f = sp.signal.decimate(baseline_sem, downsample_rate, 1)[40:-40]
        
            from lmfit import minimize, Parameters, Parameter, report_fit
            from lmfit.models import LinearModel, ExponentialModel
            
            x = np.linspace(0,len(baseline_mean_f)*downsample_rate/1000.0,len(baseline_mean_f))
        
            mod = LinearModel()
            params = Parameters()
            params.add('slope', value=0, min=-np.inf, max=np.inf)
            params.add('intercept', value=0, min=-np.inf, max=np.inf)
            lin_fit = mod.fit(baseline_mean_f, params, x=x,)
            lin_model = lin_fit.eval()
            
            mod = ExponentialModel() + LinearModel()
            params = Parameters()
            params.add('intercept', value=1, min=-np.inf, max=np.inf)
            params.add('slope', value=0, min=-1e-10, max=1e-10)
            params.add('amplitude', value=start_value, min=min_amp, max=max_amp)
            params.add('decay', value=1, min=0.01, max=np.inf)
            exp_fit = mod.fit(baseline_mean_f, params, x=x,)
            exp_model = exp_fit.eval()
            
            fig = plt.figure(figsize=(3,3))
            # plt.fill_between(x, baseline_mean_f-baseline_sem_f, baseline_mean_f+baseline_sem_f, color='k', alpha=0.25)
            plt.plot(x, baseline_mean_f, lw=1, color='k', alpha=1, label="low pass (0.01Hz)",)
            plt.plot(x, lin_model, 'g-', label="lin. fit")
            plt.plot(x, exp_model, 'r-', label="exp. fit")
            plt.legend()
            # plt.ylabel('Pupil size\n(% signal change)')
            plt.ylabel(title)
            plt.xlabel('Time (s)')
            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(self.project_directory, 'figures', 'grand_mean_slow_{}.pdf'.format(title)))
            
    def white_noise_sim(self):
        len_signal = 1000
        perms = 5000
        r = np.zeros(perms)
        for p in range(perms):
            noise = np.random.normal(0,1,len_signal)
            noise = np.cumsum(noise)
            baselines = np.array([noise[i:i+5].mean() for i in np.linspace(100,900,41)])
            phasics = np.array([noise[i:i+5].mean() for i in np.linspace(110,910,41)]) - baselines
            r[p] = sp.stats.spearmanr(baselines, phasics)[0]
            
            
            
            
        shell()
            
    
    
    
    
    
            

