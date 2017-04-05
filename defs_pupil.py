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
        self.gaze_x = np.array(self.pupil_data[(self.eye + '_gaze_x')])
        self.gaze_y = np.array(self.pupil_data[(self.eye + '_gaze_y')])
        self.compute_omission_indices()
        self.trial_params()
        
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