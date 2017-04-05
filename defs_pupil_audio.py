#!/usr/bin/env python
# encoding: utf-8
"""
Created by Jan Willem de Gee on 2011-02-16.
Copyright (c) 2011 __MyCompanyName__. All rights reserved.
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

import hedfpy

import myfuncs

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
            os.mkdir(os.path.join(project_directory, experiment_name, self.subject))
        except OSError:
            pass
        self.project_directory = project_directory
        self.base_directory = os.path.join(self.project_directory, self.experiment_name, self.subject)
        self.create_folder_hierarchy()
        self.hdf5_filename = os.path.join(self.base_directory, 'processed', self.subject + '.hdf5')
        self.ho = hedfpy.HDFEyeOperator(self.hdf5_filename)
        self.sample_rate_new = int(sample_rate_new)
        self.downsample_rate = int(1000 / sample_rate_new)
        
    def create_folder_hierarchy(self):
        """createFolderHierarchy does... guess what."""
        this_dir = self.project_directory
        for d in [self.experiment_name, self.subject]:
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
        os.system('rm {}'.format(os.path.join(self.base_directory, 'processed', self.subject + '.hdf5')))
    
    def import_raw_data(self, edf_files, aliases):
        """import_raw_data loops across edf_files and their respective aliases and copies and renames them into the raw directory."""
        for (edf_file, alias,) in zip(edf_files, aliases):
            os.system('cp "' + edf_file + '" "' + os.path.join(self.base_directory, 'raw', alias + '.edf"'))
    
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
        # self.omission_indices_first[0] = True
        if self.experiment == 1:
            self.omission_indices_subject = np.array(self.parameters['confidence'] == -1)
            self.omission_indices_subject = np.array(self.parameters['correct'] == -1)
        else:
            self.omission_indices_subject = np.zeros(self.nr_trials, dtype=bool)
        self.omission_indices = self.omission_indices_answer + self.omission_indices_sac + self.omission_indices_blinks + self.omission_indices_rt + self.omission_indices_first + self.omission_indices_subject
    
    def trial_params(self):
        blinks_nr = np.zeros(self.nr_trials)
        number_blinks = np.zeros(self.nr_trials)
        missing_data = np.zeros(self.nr_trials)
        for t in range(self.nr_trials):
            blinks_nr[t] = sum((self.blink_start_times > self.cue_times[t] - 500) * (self.blink_start_times < self.choice_times[t] + 1500))
            missing_data[t] = np.mean(self.interpolated_timepoints[(self.time > self.cue_times[t] - 500) * (self.time < self.choice_times[t] + 1500)])
        sacs_nr = np.zeros(self.nr_trials)
        sacs_dur = np.zeros(self.nr_trials)
        sacs_vel = np.zeros(self.nr_trials)
        for t in range(self.nr_trials):
            saccades_in_trial_indices = (self.saccade_start_times > self.cue_times[t] - 500) * (self.saccade_start_times < self.choice_times[t] + 1500)
            sacs_nr[t] = sum(saccades_in_trial_indices)
            sacs_dur[t] = sum(self.saccade_durs[saccades_in_trial_indices])
            if sacs_nr[t] != 0:
                sacs_vel[t] = max(self.saccade_peak_velocities[saccades_in_trial_indices])
        present = np.array(self.parameters['signal_present'] == 1)
        correct = np.array(self.parameters['correct'] == 1)
        yes = present * correct + -present * -correct
        hit = present * yes
        fa = -present * yes
        miss = present * -yes
        cr = -present * -yes
        
        run_nr = int(self.alias.split('_')[-2])
        session_nr = int(self.alias.split('_')[-1]) + 1
        
        self.parameters['omissions'] = self.omission_indices
        self.parameters['omissions_answer'] = self.omission_indices_answer
        self.parameters['omissions_sac'] = self.omission_indices_sac
        self.parameters['omissions_blinks'] = self.omission_indices_blinks
        self.parameters['omissions_rt'] = self.omission_indices_rt
        self.parameters['rt'] = self.rt
        # self.parameters['iti'] = self.ITI
        self.parameters['yes'] = yes
        self.parameters['present'] = present
        self.parameters['hit'] = hit
        self.parameters['fa'] = fa
        self.parameters['miss'] = miss
        self.parameters['cr'] = cr
        self.parameters['missing_data'] = missing_data
        self.parameters['blinks_nr'] = blinks_nr
        self.parameters['sacs_nr'] = sacs_nr
        self.parameters['sacs_dur'] = sacs_dur
        self.parameters['sacs_vel'] = sacs_vel
        self.parameters['run'] = run_nr
        self.parameters['session'] = session_nr
        self.parameters['trial'] = np.arange(self.nr_trials)
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
        
    def process_runs(self, alias, artifact_rejection='strict', create_pupil_BOLD_regressor=False):
        print 'subject {}; {}'.format(self.subject, alias)
        print '##############################'
        
        self.artifact_rejection = artifact_rejection
        
        # shell()
        
        # load data:
        self.alias = alias
        # self.events = self.ho.read_session_data(alias, 'events')
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
        self.rt = self.choice_times - self.cue_times
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
        self.interpolated_timepoints = np.array(self.pupil_data[(self.eye + '_interpolated_timepoints')])
        self.pupil = np.array(self.pupil_data[(self.eye + '_pupil_bp_clean_psc')])
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
                confidence_times = np.array(phase_times['trial_phase_EL_timestamp'][(phase_times['trial_phase_index'] == 5)]) - session_start
                feedback_times = np.array(phase_times['trial_phase_EL_timestamp'][(phase_times['trial_phase_index'] == 6)]) - session_start
            
            # baseline pupil measures:
            bp_lp.append( np.array([np.mean(pupil_lp[(time>i-500)*(time<i)]) for i in cue_times]) )
            bp_bp.append( np.array([np.mean(pupil_bp[(time>i-500)*(time<i)]) for i in cue_times]) )
            if self.experiment == 1:
                bp_feed_lp.append( np.array([np.mean(pupil_lp[(time>i-500)*(time<i)]) for i in feedback_times]) )
                bp_feed_bp.append( np.array([np.mean(pupil_bp[(time>i-500)*(time<i)]) for i in feedback_times]) )
        
            # phasic pupil responses 
            tpr_lp.append( np.array([np.mean(pupil_lp[(time>i-500)*(time<i+1500)]) for i in choice_times]) - bp_lp[-1] )
            tpr_bp.append( np.array([np.mean(pupil_bp[(time>i-500)*(time<i+1500)]) for i in choice_times]) - bp_bp[-1] )
            if self.experiment == 1:
                tpr_feed_lp.append( np.array([np.mean(pupil_lp[(time>i+500)*(time<i+1500)]) for i in feedback_times]) - bp_feed_lp[-1]  )
                tpr_feed_bp.append( np.array([np.mean(pupil_bp[(time>i+500)*(time<i+1500)]) for i in feedback_times]) - bp_feed_bp[-1]  )
            
        # join over runs:
        parameters_joined = pd.concat(parameters)
        if create_pupil_BOLD_regressor:
            pupil_BOLD_regressors_joined = np.hstack(pupil_BOLD_regressors)
            np.save(os.path.join(self.project_directory, 'data', self.subject, 'pupil_BOLD_regressors'), pupil_BOLD_regressors_joined)
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
        parameters_joined['pupil_t'] = bp_bp + tpr_bp
        parameters_joined['pupil_b_lp'] = bp_lp
        parameters_joined['pupil_d_lp'] = tpr_lp
        parameters_joined['pupil_t_lp'] = bp_lp + tpr_lp
        parameters_joined['d_prime'] = d
        parameters_joined['criterion'] = c
        for i in range(len(aliases)):
            parameters_joined['d_prime_' + str(i)] = d_run[i]
            parameters_joined['criterion_' + str(i)] = c_run[i]
        if self.experiment == 1:
            parameters_joined['pupil_b_feed'] = bp_feed_bp
            parameters_joined['pupil_d_feed'] = tpr_feed_bp
            parameters_joined['pupil_t_feed'] = bp_feed_bp + tpr_feed_bp
            parameters_joined['pupil_b_feed_lp'] = bp_feed_lp
            parameters_joined['pupil_d_feed_lp'] = tpr_feed_lp
            parameters_joined['pupil_t_feed_lp'] = bp_feed_lp + tpr_feed_lp
        parameters_joined['subject'] = self.subject
        self.ho.data_frame_to_hdf('', 'parameters_joined', parameters_joined)
        
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
            self.ho = hedfpy.HDFEyeOperator(self.hdf5_filename)
            
            try:
                parameters.append(self.ho.read_session_data('', 'parameters_joined'))
            except:
                shell()
        self.parameters_joined = pd.concat(parameters)
        self.omissions = np.array(self.parameters_joined['omissions']) 
        self.omissions = self.omissions + (np.array(self.parameters_joined['correct']) == -1)
        self.omissions = self.omissions + (np.array(self.parameters_joined['trial']) == 0)
        # self.omissions = self.omissions + (np.array(self.parameters_joined['missing_data']) > 0.25)
        
        # regress out RT per session:
        for subj in np.unique(self.parameters_joined.subject):
            for s in np.unique(self.parameters_joined.session[self.parameters_joined.subject == subj]):
                ind = (self.parameters_joined.subject == subj) * (self.parameters_joined.session == s)
                rt = np.array(self.parameters_joined['rt'][ind]) / 1000.0
                pupil_d = np.array(self.parameters_joined['pupil_d'][ind])
                pupil_d = myfuncs.lin_regress_resid(pupil_d, [rt]) + pupil_d.mean()
                self.parameters_joined['pupil_d'][ind] = pupil_d
        
        self.parameters_joined = self.parameters_joined[-self.omissions]
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
        self.pupil_t = np.array(self.parameters_joined['pupil_t'])
        self.subj_idx = np.concatenate(np.array([np.repeat(i, sum(self.parameters_joined['subject'] == self.subjects[i])) for i in range(len(self.subjects))]))
        self.criterion = np.array([np.array(self.parameters_joined[self.parameters_joined['subject']==subj]['criterion'])[0] for subj in self.subjects])
        
        #########
        
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
        'pupil_t' : pd.Series(np.array(self.pupil_t)),
        'pupil_high' : pd.Series(self.pupil_h_ind),
        'run' : pd.Series(np.array(self.run, dtype=int)),
        'session' : pd.Series(np.array(self.session, dtype=int)),
        }
        self.df = pd.DataFrame(d)
        self.behavior = myfuncs.behavior(self.df)
        
    def behavior_choice(self):
        
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
            sns.barplot(x='measure',  y='value', units='subject', data=dft, ci=None, alpha=0.5, palette=['r', 'b'], ax=ax)
            sns.stripplot(x='measure', y='value', data=dft, jitter=False, size=2, edgecolor='black', linewidth=0.25, alpha=1, palette=['r', 'b'], ax=ax)
            values = np.vstack((dft[dft['measure'] == '{}_1'.format(t)].value, dft[dft['measure'] == '{}_0'.format(t)].value))
            ax.plot(np.array([0, 1]), values, color='black', lw=0.5, alpha=0.5)
            ax.set_title('p = {}'.format(round(myfuncs.permutationTest(values[0,:], values[1,:], paired=True)[1],3)))
            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(self.project_directory, 'figures', 'behavior_SDT_measures_{}.pdf'.format(t)))
            
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
        
    def SDT_correlation(self, bin_by='pupil_d', bins=10):
        
        for bin_by in ['pupil_d']:
            for y in ['c', 'd']:
                # model_comp = 'bayes'
                model_comp = 'seq'
                fig, rs = self.behavior.SDT_correlation(bin_by=bin_by, n_bins=bins, y1=y, model_comp=model_comp)
                # fig = self.behavior.SDT_correlation(bin_by=bin_by, n_bins=bins, y1='correct', y2='correct')
                fig.savefig(os.path.join(self.project_directory, 'figures', 'SDT_correlation_{}_{}_{}.pdf'.format(bin_by, y, model_comp)))