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

import sklearn
import mne
import statsmodels.formula.api as sm
import nibabel as nib
import nilearn

from IPython import embed as shell

import rpy2.robjects as robjects
import rpy2.rlike.container as rlc

sys.path.append(os.environ['ANALYSIS_HOME'])
from Tools.Sessions import *
from Tools.Run import *
from Tools.Operators import *

from Tools.other_scripts import functions_jw as myfuncs
from Tools.other_scripts import functions_jw_GLM as myglm

matplotlib.rcParams['pdf.fonttype'] = 42
sns.set(style='ticks', font='Arial', font_scale=1, rc={
    'axes.labelsize': 7, 
    'axes.titlesize': 7, 
    'xtick.labelsize': 6, 
    'ytick.labelsize': 6, 
    'legend.fontsize': 6, 
    'axes.linewidth': 0.25, 
    'xtick.major.width': 0.25, 
    'ytick.major.width': 0.25,
    # 'lines.linewidth': 0.5,
    'text.color': 'Black',
    'axes.labelcolor':'Black',
    'xtick.color':'Black',
    'ytick.color':'Black',} )
sns.plotting_context()

class defs_fmri_group(object):
    
    def __init__(self, subjects, nr_sessions, base_dir, split_by='pupil_d'):
        
        self.subjects = subjects
        self.nr_sessions = nr_sessions
        self.base_dir = base_dir
        self.data_folder = os.path.join(base_dir, 'data', 'across')
        self.figure_folder = os.path.join(base_dir, 'figures')
        self.mask_folder = '/home/shared/UvA/Niels_UvA/mni_masks/'
        
        self.split_by = split_by
        if 'pupil' in split_by:
            self.postfix = '_' + self.split_by.split('_')[-1]
        else:
            self.postfix = '_d'
        # self.postfix = '_d'
        
        # self.pupil_data = []
        # self.omissions = []
        # for i in range(len(self.subjects)):
        #     p = pd.read_csv(os.path.join(self.base_dir, 'data', self.subjects[i], 'pupil_data.csv'))
        #     omissions = np.array(p.omissions)
        #     # set omissions to False:
        #     # omissions[:] = False
        #     p = p[-omissions]
        #
        #     self.pupil_data.append(p)
        #     self.omissions.append(omissions)
        
        
        
        # ceiling effect / ITI control:
        self.pupil_data = []
        self.omissions_ori = []
        self.omissions_ori2 = []
        self.omissions = []
        for i in range(len(self.subjects)):
            d = pd.read_csv(os.path.join(self.base_dir, 'data', self.subjects[i], 'pupil_data.csv'))
            self.omissions_ori.append(np.array(d.omissions, dtype=bool))
            self.omissions_ori2.append(np.array(d['correct'])==-1)
            omissions = []
            for r in np.array(np.unique(d['run']), dtype=int):
                iti = np.array(d['iti'])[np.array(d.run == r)]
                iti[np.isnan(iti)] = 0
                pupil_b = np.array(d['pupil_b_lp'])[np.array(d.run == r)]
                pupil_d = np.array(d['pupil_d_lp'])[np.array(d.run == r)]
                lower_cutoff = np.percentile(pupil_b+pupil_d, 10)
                higher_cutoff = np.percentile(pupil_b+pupil_d, 90)
                
                omi = np.array(d.omissions, dtype=bool)[np.array(d.run == r)] #+ (np.array(d['correct']) == -1)[np.array(d.run == r)]
                
                # # only positive TPR:
                # omi = omi + (pupil_d < 0)
                
                # # ceiling effect control:
                # omi = omi + (pupil_b+pupil_d > higher_cutoff) + (pupil_b+pupil_d < lower_cutoff)
                
                # # ITI control:
                # omi = omi + (iti <= 10.0)
                
                # append:
                omissions.append(omi)
                
            omissions = np.concatenate(omissions)
            d.omissions = omissions
            d = d[-omissions]
            self.pupil_data.append(d)
            self.omissions.append(omissions)
            
        # # regress out RT across all runs:
        # self.pupil_l_ind = []
        # self.pupil_h_ind = []
        # for i in range(len(self.subjects)):
        #     iti = np.array(self.pupil_data[i]['iti'])
        #     iti[np.isnan(iti)] = bn.nanmean(iti)
        #     present = np.array(self.pupil_data[i]['present'])
        #     rt = np.array(self.pupil_data[i]['rt'])
        #     pupil_b = np.array(self.pupil_data[i]['pupil_b'])
        #     pupil_d = myfuncs.lin_regress_resid(np.array(self.pupil_data[i]['pupil_d']), [rt])
        #     pupil_t = myfuncs.lin_regress_resid(np.array(self.pupil_data[i]['pupil_t']), [rt])
        #     p_h = []
        #     p_l = []
        #     for r in np.array(np.unique(self.pupil_data[i]['run']), dtype=int):
        #         if split_by == 'pupil_b':
        #             pupil = pupil_b[np.array(self.pupil_data[i].run == r)]
        #         if split_by == 'pupil_d':
        #             pupil = pupil_d[np.array(self.pupil_data[i].run == r)]
        #         if split_by == 'pupil_t':
        #             pupil = pupil_t[np.array(self.pupil_data[i].run == r)]
        #         p_l.append( pupil <= np.percentile(pupil, 40) )
        #         p_h.append( pupil >= np.percentile(pupil, 60) )
        #     self.pupil_l_ind.append(np.concatenate(p_l))
        #     self.pupil_h_ind.append(np.concatenate(p_h))
        
        # # regress out RT per run:
        # self.pupil_l_ind = []
        # self.pupil_h_ind = []
        # for i in range(len(self.subjects)):
        #     p_h = []
        #     p_l = []
        #     for r in np.array(np.unique(self.pupil_data[i]['run']), dtype=int):
        #         rt = np.array(self.pupil_data[i]['rt'])[np.array(self.pupil_data[i].run == r)]
        #         pupil_b = np.array(self.pupil_data[i]['pupil_b'])[np.array(self.pupil_data[i].run == r)]
        #         pupil_d = myfuncs.lin_regress_resid(np.array(self.pupil_data[i]['pupil_d'])[np.array(self.pupil_data[i].run == r)], [rt])
        #         pupil_t = myfuncs.lin_regress_resid(np.array(self.pupil_data[i]['pupil_t'])[np.array(self.pupil_data[i].run == r)], [rt])
        #
        #         if split_by == 'pupil_b':
        #             pupil = pupil_b
        #         if split_by == 'pupil_d':
        #             pupil = pupil_d
        #         if split_by == 'pupil_t':
        #             pupil = pupil_t
        #         p_l.append( pupil <= np.percentile(pupil, 40) )
        #         p_h.append( pupil >= np.percentile(pupil, 60) )
        #     self.pupil_l_ind.append(np.concatenate(p_l))
        #     self.pupil_h_ind.append(np.concatenate(p_h))
            
        # shell()
        
        # regress out RT per session:
        if split_by == 'pupil_d':
            self.pupil_l_ind = []
            self.pupil_h_ind = []
            for i in range(len(self.subjects)):
                p_h = []
                p_l = []
                for s in np.array(np.unique(self.pupil_data[i]['session']), dtype=int):
                    rt = np.array(self.pupil_data[i]['rt'])[np.array(self.pupil_data[i].session == s)]
                    pupil = myfuncs.lin_regress_resid(np.array(self.pupil_data[i]['pupil_d'])[np.array(self.pupil_data[i].session == s)], [rt]) + np.mean(np.array(self.pupil_data[i]['pupil_d'])[np.array(self.pupil_data[i].session == s)])
                    p_l.append( pupil <= np.percentile(pupil, 40) )
                    p_h.append( pupil >= np.percentile(pupil, 60) )
                    self.pupil_data[i]['pupil_d'][np.array(self.pupil_data[i].session == s)] = pupil
                self.pupil_l_ind.append(np.concatenate(p_l))
                self.pupil_h_ind.append(np.concatenate(p_h))
        
        elif split_by == 'pupil_b':
            self.pupil_l_ind = []
            self.pupil_h_ind = []
            for i in range(len(self.subjects)):
                p_h = []
                p_l = []
                for s in np.array(np.unique(self.pupil_data[i]['session']), dtype=int):
                    pupil = np.array(self.pupil_data[i]['pupil_b'])[np.array(self.pupil_data[i].session == s)]
                    p_l.append( pupil <= np.percentile(pupil, 40) )
                    p_h.append( pupil >= np.percentile(pupil, 60) )
                # self.pupil_l_ind.append(np.concatenate(p_l))
                # self.pupil_h_ind.append(np.concatenate(p_h))
                
                self.pupil_l_ind.append(np.load(os.path.join(self.base_dir, 'data', 'across', 'baseline_splits', 'baselines_bad_{}.npy'.format(self.subjects[i]))))
                self.pupil_h_ind.append(np.load(os.path.join(self.base_dir, 'data', 'across', 'baseline_splits', 'baselines_good_{}.npy'.format(self.subjects[i]))))
                
        self.session = [np.array(self.pupil_data[i]['session'], dtype=int) for i in range(len(self.subjects))]
        self.run = [np.array(self.pupil_data[i]['run'], dtype=int) for i in range(len(self.subjects))]
        
        self.subj_idx = np.concatenate(np.array([np.repeat(i, sum(self.pupil_data[i]['subject'] == self.subjects[i])) for i in range(len(self.subjects))]))
        
        self.all = [np.array(self.pupil_data[i]['yes'], dtype=bool) + ~np.array(self.pupil_data[i]['yes'], dtype=bool) for i in range(len(self.subjects))]
        
        self.yes = [np.array(self.pupil_data[i]['yes'], dtype=bool) for i in range(len(self.subjects))]
        self.no = [-self.yes[i] for i in range(len(self.subjects))]
        self.correct = [np.array(self.pupil_data[i]['correct'], dtype=bool) for i in range(len(self.subjects))]
        self.error = [-self.correct[i] for i in range(len(self.subjects))]
        self.present = [np.array(self.pupil_data[i]['present'], dtype=bool) for i in range(len(self.subjects))]
        self.absent = [-self.present[i] for i in range(len(self.subjects))]
        
        self.hit = [self.yes[i]&self.correct[i] for i in range(len(self.subjects))]
        self.fa = [self.yes[i]&self.error[i] for i in range(len(self.subjects))]
        self.miss = [self.no[i]&self.error[i] for i in range(len(self.subjects))]
        self.cr = [self.no[i]&self.correct[i] for i in range(len(self.subjects))]
        
        self.rt = [np.array(self.pupil_data[i]['rt']) for i in range(len(self.subjects))]
        self.ITI = [np.array(self.pupil_data[i]['iti']) for i in range(len(self.subjects))]
        self.nr_blinks = [np.array(self.pupil_data[i]['nr_blinks']) for i in range(len(self.subjects))]
        
        # choice measures:
        
        self.dprime = np.array([ 0.90973303,  1.37383172,  1.38641933,  1.44943683,  0.86743776, 1.08008268,  1.19168672,  0.95251872,  1.50886329,  1.00334947, 1.56768555,  1.71320222,  1.9533004 ,  1.68061294])
        self.dprime_lo = np.array([ 0.82268369,  1.40626237,  1.26098667,  1.61744189,  0.93088421, 1.05655906,  1.21978025,  0.7371303 ,  1.70223294,  1.11477536, 1.40391272,  1.64518343,  2.18572193,  1.72808791])
        self.dprime_hi = np.array([ 0.98757241,  1.33481627,  1.52740115,  1.29288604,  0.79371887, 1.09588103,  1.16124991,  1.19340558,  1.31255077,  0.8971233 , 1.7165381 ,  1.72027617,  1.71988589,  1.58279409])
        
        self.criterion = np.array([ 0.18580038,  0.05327586,  0.11604731, -0.08920945,  0.02565914, 0.19491581,  0.22206797, -0.1028728 , -0.05852371, -0.03340808, 0.1176113 ,  0.60325401,  0.23742476,  1.03381485])
        self.criterion_lo = np.array([ 0.27163155,  0.12399902,  0.21859286, -0.01128321,  0.08012164, 0.24465616,  0.29856774,  0.00482141, -0.08247346,  0.07788203, 0.06668665,  0.65838094,  0.35043836,  1.18970496])
        self.criterion_hi = np.array([ 0.08132308, -0.03213842, -0.01733324, -0.16693737, -0.04507509, 0.12654924,  0.12847755, -0.24491844, -0.05769171, -0.15314651, 0.14916655,  0.51138239,  0.11535646,  0.84364222])
        
        # self.drift_rate = np.array([ 0.19757142,  0.09292426,  0.09920115,  0.15260826,  0.02021234, 0.1869882 , 0.24642884, 0.12784264, 0.04536516, 0.02965839, 0.09878793,  0.53726986,  0.25126305,  1.03668549])
        # self.drift_rate_lo = np.array([ 0.29041646,  0.13300271,  0.18665655,  0.00930619,  0.13775779, 0.20597982,  0.34027702,  0.05833908,  0.09266987,  0.12685303, 0.11566815,  0.6090364 ,  0.4335415 ,  1.29694321])
        # self.drift_rate_hi = np.array([ 0.01611655,  0.0892981 ,  0.09515956,  0.1612426 ,  0.06816878, 0.03312447,  0.09145396,  0.34810124,  0.02705677,  0.13701039, 0.1693657 ,  0.42089614,  0.03959779,  0.9543934 ])
        #
        # self.drift_criterion = np.array([ 0.19757142,  0.09292426,  0.09920115,  0.15260826,  0.02021234, 0.1869882 , 0.24642884, 0.12784264, 0.04536516, 0.02965839, 0.09878793,  0.53726986,  0.25126305,  1.03668549])
        # self.drift_criterion_lo = np.array([ 0.29041646,  0.13300271,  0.18665655,  0.00930619,  0.13775779, 0.20597982,  0.34027702,  0.05833908,  0.09266987,  0.12685303, 0.11566815,  0.6090364 ,  0.4335415 ,  1.29694321])
        # self.drift_criterion_hi = np.array([ 0.01611655,  0.0892981 ,  0.09515956,  0.1612426 ,  0.06816878, 0.03312447,  0.09145396,  0.34810124,  0.02705677,  0.13701039, 0.1693657 ,  0.42089614,  0.03959779,  0.9543934 ])
        
        self.liberal = (self.criterion <= np.median(self.criterion))
        
        self.right = []
        for i in range(len(self.subjects)):
            if self.subjects[i] in ['sub-03', 'sub-04', 'sub-06', 'sub-08', 'sub-09', 'sub-11', 'sub-13']:
                self.right.append(self.yes[i])
            else:
                self.right.append(self.no[i])
            
        # MEANS = (self.criterion_hi.mean(), self.criterion_lo.mean())
        # SEMS = (sp.stats.sem(self.criterion_hi), sp.stats.sem(self.criterion_lo))
        # N = 2
        # ind = np.linspace(0,N/2,N)
        # bar_width = 0.50
        # fig = plt.figure(figsize=(1.25,1.75))
        # ax = fig.add_subplot(111)
        # for i in range(N):
        #     ax.bar(ind[i], height=MEANS[i], width = bar_width, yerr = SEMS[i], color = ['r','b'][i], alpha = 1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
        # ax.set_title('p = {}'.format(round(sp.stats.ttest_rel(self.criterion_hi, self.criterion_lo)[1],3)), size=7)
        # ax.set_ylabel('drift criterion', size=7)
        # ax.set_xticks( (ind) )
        # ax.set_xticklabels( ('high', 'low') )
        # sns.despine(offset=10, trim=True)
        # plt.tight_layout()
        # fig.savefig(os.path.join(self.figure_folder, 'behavior_SDT_measures_dc.pdf'))
        
    def load_BOLD_per_subject(self, subject, data_type, roi, time_locked, type_response, stratification=False, project_out='4th_ventricle', baseline=True):
        
        import copy
        
        subj_idx = int(np.where(self.subjects == subject)[0])
        pupil_data = self.pupil_data[subj_idx]
        yes = self.yes[subj_idx]
        no = self.no[subj_idx]
        correct = self.correct[subj_idx]
        error = self.error[subj_idx]
        present = self.present[subj_idx]
        absent = self.absent[subj_idx]
        rt = self.rt[subj_idx]
        
        try:
            pupil_h_ind = self.pupil_h_ind[subj_idx]
            pupil_l_ind = self.pupil_l_ind[subj_idx]
        except:
            pass
        
        BOLD_data = np.load(os.path.join(self.data_folder, 'event_related_average', 'full_array_{}_{}_{}_{}.npy'.format(data_type, roi, time_locked, subject)))[-self.omissions[subj_idx],:]
        # if baseline:
            # BOLD_data = BOLD_data - np.atleast_2d(np.array(self.data_frame[(self.data_frame.subject == subject)][roi+'_b'])).T
        
        if stratification:
            k1 = []
            k2 = []
            for i in range(1000):
                if 'pupil' in self.split_by:
                    ind1 = copy.copy(pupil_h_ind)
                    ind2 = copy.copy(pupil_l_ind)
                elif self.split_by == 'yes':
                    ind1 = copy.copy(yes)
                    ind2 = copy.copy(no)
                elif self.split_by == 'correct':
                    ind1 = copy.copy(correct)
                    ind2 = copy.copy(error)
                elif self.split_by == 'present':
                    ind1 = copy.copy(present)
                    ind2 = copy.copy(absent)
                ind1_strat, ind2_strat = myfuncs.common_dist(rt[ind1], rt[ind2], bins=20)
                ind1[ind1] = ind1[ind1]*ind1_strat
                ind2[ind2] = ind2[ind2]*ind2_strat
                k1.append( bn.nanmean(BOLD_data[ind1], axis=0) )
                k2.append( bn.nanmean(BOLD_data[ind2], axis=0) )
            if type_response == 'mean':
                k1 = bn.nanmean(np.vstack(k1), axis=0)
                k2 = bn.nanmean(np.vstack(k2), axis=0)
            if type_response == 'std':
                k1 = bn.nanstd(np.vstack(k1), axis=0)
                k2 = bn.nanstd(np.vstack(k2), axis=0)
            print
            print 'subj: {}, roi: {}, comparison: {}'.format(subject, roi, comparison)
            print '-------------------------------------------------------'
            print 'trials thrown away: {} out of: {}'.format(sum(-ind1_strat)+sum(-ind2_strat), len(rt))
            print 'mean rt ind1 = {} s'.format(round(rt[ind1].mean(),3))
            print 'mean rt ind2 = {} s'.format(round(rt[ind2].mean(),3))
        else:
            if 'pupil' in self.split_by:
                ind1 = copy.copy(pupil_h_ind)
                ind2 = copy.copy(pupil_l_ind)
                ind1 = copy.copy(pupil_h_ind) * present
                ind2 = copy.copy(pupil_l_ind) * present
                ind3 = -(ind1+ind2)
            elif self.split_by == 'yes':
                ind1 = copy.copy(yes)
                ind2 = copy.copy(no)
            elif self.split_by == 'correct':
                ind1 = copy.copy(correct)
                ind2 = copy.copy(error)
            elif self.split_by == 'present':
                ind1 = copy.copy(present)
                ind2 = copy.copy(absent)
            if type_response == 'mean':
                k1 = bn.nanmean(BOLD_data[ind1], axis=0)
                k2 = bn.nanmean(BOLD_data[ind2], axis=0)
            elif type_response == 'std':
                k1 = bn.nanstd(BOLD_data[ind1], axis=0)
                k2 = bn.nanstd(BOLD_data[ind2], axis=0)
        
        if project_out:
            print 'project out: {}'.format(project_out)
            BOLD_data_project_out = np.load(os.path.join(self.data_folder, 'event_related_average', 'full_array_{}_{}_{}_{}.npy'.format(data_type, project_out, time_locked, subject)))[-self.omissions[subj_idx],:]
            if type_response == 'mean':
                k1 = myfuncs.lin_regress_resid(k1, bn.nanmean(BOLD_data_project_out[ind1], axis=0))
                k2 = myfuncs.lin_regress_resid(k2, bn.nanmean(BOLD_data_project_out[ind2], axis=0))
            if type_response == 'std':
                k1 = myfuncs.lin_regress_resid(k1, bn.nanstd(BOLD_data_project_out[ind1], axis=0))
                k2 = myfuncs.lin_regress_resid(k2, bn.nanstd(BOLD_data_project_out[ind2], axis=0))
        
        if 'pupil' in self.split_by:
            if type_response == 'mean':
                k3 = bn.nanmean(BOLD_data, axis=0)
            elif type_response == 'std':
                k3 = bn.nanstd(BOLD_data, axis=0)
        else:
            k3 = bn.nanmean(BOLD_data, axis=0)
        
        return k1, k2, k3
        
        
    def make_pupil_BOLD_dataframe(self, data_type, time_locked='stim_locked', regress_iti=False, regress_rt=False, regress_stimulus=False):
        
        if data_type == 'psc_False':
        
            ROIs = ['LC_JW',]
        
        if data_type == 'clean_4th_ventricle':
        
            ROIs = [
                        # BRAIN:
                        # ------
                        # 'brain_mask',
                    
                        # RETINOPY:
                        # --------
                        # 'V1',
                        # 'V1_center',
                        # 'V1_surround',
                        # 'V1_periphery',
                        # 'V2_center',
                        # 'V2_surround',
                        # 'V3_center',
                        # 'V3_surround',
                        # 'V4',
                    
                        # BRAINSTEM:
                        # ----------
                        # 'LC_standard_2_100',
                        # 'LC_standard_1_100',
                        # 'mean_fullMB_100',
                        # 'mean_SN_100',
                        # 'mean_VTA_100',
                        # 'basal_forebrain_4_100',
                        # 'basal_forebrain_123_100',
                        # 'inferior_colliculus_100',
                        # 'superior_colliculus_100',
                        # 'AAN_VTA_100',
                        # 'AAN_PAG_100',
                        # 'AAN_PBC_100',
                        # 'AAN_PO_100',
                        # 'AAN_PPN_100',
                        # 'AAN_LC_100',
                        # 'AAN_MR_100',
                        # 'AAN_MRF_100',
                        # 'AAN_DR_100',
                        'LC_standard_2',
                        'LC_standard_1',
                        'mean_fullMB',
                        'mean_SN',
                        'mean_VTA',
                        'basal_forebrain_4',
                        'basal_forebrain_123',
                        'AAN_VTA',
                        'AAN_PAG',
                        'AAN_PBC',
                        'AAN_PO',
                        'AAN_PPN',
                        'AAN_LC',
                        'AAN_MR',
                        'AAN_MRF',
                        'AAN_DR',
                        'LC_JW',
                        'LC_JW_nn',
                        'sup_col_jw',
                        'inf_col_jw',
                        '4th_ventricle',
                    
                        # # PARCELATION:
                        # # ------------
                        #
                        # # visual:
                        # 'Pole_occipital',
                        # 'G_occipital_sup',
                        # 'S_oc_sup_and_transversal',
                        # 'G_occipital_middle',
                        # 'S_occipital_ant',
                        # 'S_oc_middle_and_Lunatus',
                        # 'G_and_S_occipital_inf',
                        # 'S_collat_transv_post',
                        # 'G_oc-temp_med-Lingual',
                        # 'S_calcarine',
                        # 'G_cuneus',
                        #
                        # # temporal:
                        # 'Lat_Fis-post',
                        # 'G_temp_sup-Plan_tempo',
                        # 'S_temporal_transverse',
                        # 'G_temp_sup-G_T_transv',
                        # 'G_temp_sup-Lateral',
                        # 'S_temporal_sup',
                        # 'G_temporal_middle',
                        # 'S_temporal_inf',
                        # 'G_temporal_inf',
                        # 'S_oc-temp_lat',
                        # 'G_oc-temp_med-Parahip',
                        # 'S_collat_transv_ant',
                        # 'G_oc-temp_lat-fusifor',
                        # 'S_oc-temp_med_and_Lingual',
                        # 'G_temp_sup-Plan_polar',
                        # 'Pole_temporal',
                        #
                        # # parietal:
                        # 'S_parieto_occipital',
                        # 'S_subparietal',
                        # 'G_precuneus',
                        # 'G_parietal_sup',
                        # 'S_intrapariet_and_P_trans',
                        # 'G_pariet_inf-Angular',
                        # 'S_interm_prim-Jensen',
                        # 'G_and_S_paracentral',
                        # 'S_postcentral',
                        # 'G_postcentral',
                        # 'S_central',
                        # 'G_pariet_inf-Supramar',
                        # 'G_and_S_subcentral',
                        #
                        # # insular:
                        # 'S_circular_insula_sup',
                        # 'G_insular_short',
                        # 'S_circular_insula_inf',
                        # 'G_Ins_lg_and_S_cent_ins',
                        # 'S_circular_insula_ant',
                        #
                        # # cingulate:
                        # 'G_cingul-Post-ventral',
                        # 'S_pericallosal',
                        # 'G_cingul-Post-dorsal',
                        # 'S_cingul-Marginalis',
                        # 'G_and_S_cingul-Mid-Post',
                        # 'G_and_S_cingul-Mid-Ant',
                        # 'G_and_S_cingul-Ant',
                        #
                        # # frontal:
                        # 'G_precentral',
                        # 'S_precentral-sup-part',
                        # 'S_precentral-inf-part',
                        # 'G_front_sup',
                        # 'S_front_sup',
                        # 'G_front_middle',
                        # 'S_front_middle',
                        # 'S_front_inf',
                        # 'G_front_inf-Opercular',
                        # 'G_front_inf-Triangul',
                        # 'S_orbital_lateral',
                        # 'Lat_Fis-ant-Horizont',
                        # 'Lat_Fis-ant-Vertical',
                        # 'G_front_inf-Orbital',
                        # 'G_and_S_transv_frontopol',
                        # 'G_and_S_frontomargin',
                        # 'G_orbital',
                        # 'S_orbital-H_Shaped',
                        # 'S_orbital_med-olfact',
                        # 'G_rectus',
                        # 'S_suborbital',
                        # 'G_subcallosal',
                        #
                        # # PARCELATION LEFT:
                        # # -----------------
                        #
                        # # visual:
                        # 'lh.Pole_occipital',
                        # 'lh.G_occipital_sup',
                        # 'lh.S_oc_sup_and_transversal',
                        # 'lh.G_occipital_middle',
                        # 'lh.S_occipital_ant',
                        # 'lh.S_oc_middle_and_Lunatus',
                        # 'lh.G_and_S_occipital_inf',
                        # 'lh.S_collat_transv_post',
                        # 'lh.G_oc-temp_med-Lingual',
                        # 'lh.S_calcarine',
                        # 'lh.G_cuneus',
                        #
                        # # temporal:
                        # 'lh.Lat_Fis-post',
                        # 'lh.G_temp_sup-Plan_tempo',
                        # 'lh.S_temporal_transverse',
                        # 'lh.G_temp_sup-G_T_transv',
                        # 'lh.G_temp_sup-Lateral',
                        # 'lh.S_temporal_sup',
                        # 'lh.G_temporal_middle',
                        # 'lh.S_temporal_inf',
                        # 'lh.G_temporal_inf',
                        # 'lh.S_oc-temp_lat',
                        # 'lh.G_oc-temp_med-Parahip',
                        # 'lh.S_collat_transv_ant',
                        # 'lh.G_oc-temp_lat-fusifor',
                        # 'lh.S_oc-temp_med_and_Lingual',
                        # 'lh.G_temp_sup-Plan_polar',
                        # 'lh.Pole_temporal',
                        #
                        # # parietal:
                        # 'lh.S_parieto_occipital',
                        # 'lh.S_subparietal',
                        # 'lh.G_precuneus',
                        # 'lh.G_parietal_sup',
                        # 'lh.S_intrapariet_and_P_trans',
                        # 'lh.G_pariet_inf-Angular',
                        # 'lh.S_interm_prim-Jensen',
                        # 'lh.G_and_S_paracentral',
                        # 'lh.S_postcentral',
                        # 'lh.G_postcentral',
                        # 'lh.S_central',
                        # 'lh.G_pariet_inf-Supramar',
                        # 'lh.G_and_S_subcentral',
                        #
                        # # insular:
                        # 'lh.S_circular_insula_sup',
                        # 'lh.G_insular_short',
                        # 'lh.S_circular_insula_inf',
                        # 'lh.G_Ins_lg_and_S_cent_ins',
                        # 'lh.S_circular_insula_ant',
                        #
                        # # cingulate:
                        # 'lh.G_cingul-Post-ventral',
                        # 'lh.S_pericallosal',
                        # 'lh.G_cingul-Post-dorsal',
                        # 'lh.S_cingul-Marginalis',
                        # 'lh.G_and_S_cingul-Mid-Post',
                        # 'lh.G_and_S_cingul-Mid-Ant',
                        # 'lh.G_and_S_cingul-Ant',
                        #
                        # # frontal:
                        # 'lh.G_precentral',
                        # 'lh.S_precentral-sup-part',
                        # 'lh.S_precentral-inf-part',
                        # 'lh.G_front_sup',
                        # 'lh.S_front_sup',
                        # 'lh.G_front_middle',
                        # 'lh.S_front_middle',
                        # 'lh.S_front_inf',
                        # 'lh.G_front_inf-Opercular',
                        # 'lh.G_front_inf-Triangul',
                        # 'lh.S_orbital_lateral',
                        # 'lh.Lat_Fis-ant-Horizont',
                        # 'lh.Lat_Fis-ant-Vertical',
                        # 'lh.G_front_inf-Orbital',
                        # 'lh.G_and_S_transv_frontopol',
                        # 'lh.G_and_S_frontomargin',
                        # 'lh.G_orbital',
                        # 'lh.S_orbital-H_Shaped',
                        # 'lh.S_orbital_med-olfact',
                        # 'lh.G_rectus',
                        # 'lh.S_suborbital',
                        # 'lh.G_subcallosal',
                        #
                        # # PARCELATION RIGHT:
                        # # -----------------
                        #
                        # # visual:
                        # 'rh.Pole_occipital',
                        # 'rh.G_occipital_sup',
                        # 'rh.S_oc_sup_and_transversal',
                        # 'rh.G_occipital_middle',
                        # 'rh.S_occipital_ant',
                        # 'rh.S_oc_middle_and_Lunatus',
                        # 'rh.G_and_S_occipital_inf',
                        # 'rh.S_collat_transv_post',
                        # 'rh.G_oc-temp_med-Lingual',
                        # 'rh.S_calcarine',
                        # 'rh.G_cuneus',
                        #
                        # # temporal:
                        # 'rh.Lat_Fis-post',
                        # 'rh.G_temp_sup-Plan_tempo',
                        # 'rh.S_temporal_transverse',
                        # 'rh.G_temp_sup-G_T_transv',
                        # 'rh.G_temp_sup-Lateral',
                        # 'rh.S_temporal_sup',
                        # 'rh.G_temporal_middle',
                        # 'rh.S_temporal_inf',
                        # 'rh.G_temporal_inf',
                        # 'rh.S_oc-temp_lat',
                        # 'rh.G_oc-temp_med-Parahip',
                        # 'rh.S_collat_transv_ant',
                        # 'rh.G_oc-temp_lat-fusifor',
                        # 'rh.S_oc-temp_med_and_Lingual',
                        # 'rh.G_temp_sup-Plan_polar',
                        # 'rh.Pole_temporal',
                        #
                        # # parietal:
                        # 'rh.S_parieto_occipital',
                        # 'rh.S_subparietal',
                        # 'rh.G_precuneus',
                        # 'rh.G_parietal_sup',
                        # 'rh.S_intrapariet_and_P_trans',
                        # 'rh.G_pariet_inf-Angular',
                        # 'rh.S_interm_prim-Jensen',
                        # 'rh.G_and_S_paracentral',
                        # 'rh.S_postcentral',
                        # 'rh.G_postcentral',
                        # 'rh.S_central',
                        # 'rh.G_pariet_inf-Supramar',
                        # 'rh.G_and_S_subcentral',
                        #
                        # # insular:
                        # 'rh.S_circular_insula_sup',
                        # 'rh.G_insular_short',
                        # 'rh.S_circular_insula_inf',
                        # 'rh.G_Ins_lg_and_S_cent_ins',
                        # 'rh.S_circular_insula_ant',
                        #
                        # # cingulate:
                        # 'rh.G_cingul-Post-ventral',
                        # 'rh.S_pericallosal',
                        # 'rh.G_cingul-Post-dorsal',
                        # 'rh.S_cingul-Marginalis',
                        # 'rh.G_and_S_cingul-Mid-Post',
                        # 'rh.G_and_S_cingul-Mid-Ant',
                        # 'rh.G_and_S_cingul-Ant',
                        #
                        # # frontal:
                        # 'rh.G_precentral',
                        # 'rh.S_precentral-sup-part',
                        # 'rh.S_precentral-inf-part',
                        # 'rh.G_front_sup',
                        # 'rh.S_front_sup',
                        # 'rh.G_front_middle',
                        # 'rh.S_front_middle',
                        # 'rh.S_front_inf',
                        # 'rh.G_front_inf-Opercular',
                        # 'rh.G_front_inf-Triangul',
                        # 'rh.S_orbital_lateral',
                        # 'rh.Lat_Fis-ant-Horizont',
                        # 'rh.Lat_Fis-ant-Vertical',
                        # 'rh.G_front_inf-Orbital',
                        # 'rh.G_and_S_transv_frontopol',
                        # 'rh.G_and_S_frontomargin',
                        # 'rh.G_orbital',
                        # 'rh.S_orbital-H_Shaped',
                        # 'rh.S_orbital_med-olfact',
                        # 'rh.G_rectus',
                        # 'rh.S_suborbital',
                        # 'rh.G_subcallosal',
                    ]
                    
        elif data_type == 'clean_False':
    
            ROIs = [
                        # BRAIN:
                        # ------
                        # 'brain_mask',
                
                        # RETINOPY:
                        # --------
                        # 'V1',
                        'V1_center',
                        'V1_surround',
                        # 'V1_periphery',
                        'V2_center',
                        'V2_surround',
                        'V3_center',
                        'V3_surround',
                        'lr_aIPS',
                        'lr_PCeS',
                        'lr_M1',
                        'sl_IPL',
                        'sl_SPL1',
                        'sl_SPL2',
                        'sl_pIns',
                        # 'V4',
                
                        # BRAINSTEM:
                        # ----------
                        'LC_standard_2',
                        'LC_standard_1',
                        'mean_fullMB',
                        'mean_SN',
                        'mean_VTA',
                        'basal_forebrain_4',
                        'basal_forebrain_123',
                        'AAN_VTA',
                        'AAN_PAG',
                        'AAN_PBC',
                        'AAN_PO',
                        'AAN_PPN',
                        'AAN_LC',
                        'AAN_MR',
                        'AAN_MRF',
                        'AAN_DR',
                        'LC_JW',
                        '4th_ventricle',
                
                        # # PARCELATION:
                        # # ------------
                        #
                        # visual:
                        'Pole_occipital',
                        'G_occipital_sup',
                        'S_oc_sup_and_transversal',
                        'G_occipital_middle',
                        'S_occipital_ant',
                        'S_oc_middle_and_Lunatus',
                        'G_and_S_occipital_inf',
                        'S_collat_transv_post',
                        'G_oc-temp_med-Lingual',
                        'S_calcarine',
                        'G_cuneus',

                        # temporal:
                        'Lat_Fis-post',
                        'G_temp_sup-Plan_tempo',
                        'S_temporal_transverse',
                        'G_temp_sup-G_T_transv',
                        'G_temp_sup-Lateral',
                        'S_temporal_sup',
                        'G_temporal_middle',
                        'S_temporal_inf',
                        'G_temporal_inf',
                        'S_oc-temp_lat',
                        'G_oc-temp_med-Parahip',
                        'S_collat_transv_ant',
                        'G_oc-temp_lat-fusifor',
                        'S_oc-temp_med_and_Lingual',
                        'G_temp_sup-Plan_polar',
                        'Pole_temporal',

                        # parietal:
                        'S_parieto_occipital',
                        'S_subparietal',
                        'G_precuneus',
                        'G_parietal_sup',
                        'S_intrapariet_and_P_trans',
                        'G_pariet_inf-Angular',
                        'S_interm_prim-Jensen',
                        'G_and_S_paracentral',
                        'S_postcentral',
                        'G_postcentral',
                        'S_central',
                        'G_pariet_inf-Supramar',
                        'G_and_S_subcentral',

                        # insular:
                        'S_circular_insula_sup',
                        'G_insular_short',
                        'S_circular_insula_inf',
                        'G_Ins_lg_and_S_cent_ins',
                        'S_circular_insula_ant',

                        # cingulate:
                        'G_cingul-Post-ventral',
                        'S_pericallosal',
                        'G_cingul-Post-dorsal',
                        'S_cingul-Marginalis',
                        'G_and_S_cingul-Mid-Post',
                        'G_and_S_cingul-Mid-Ant',
                        'G_and_S_cingul-Ant',

                        # frontal:
                        'G_precentral',
                        'S_precentral-sup-part',
                        'S_precentral-inf-part',
                        'G_front_sup',
                        'S_front_sup',
                        'G_front_middle',
                        'S_front_middle',
                        'S_front_inf',
                        'G_front_inf-Opercular',
                        'G_front_inf-Triangul',
                        'S_orbital_lateral',
                        'Lat_Fis-ant-Horizont',
                        'Lat_Fis-ant-Vertical',
                        'G_front_inf-Orbital',
                        'G_and_S_transv_frontopol',
                        'G_and_S_frontomargin',
                        'G_orbital',
                        'S_orbital-H_Shaped',
                        'S_orbital_med-olfact',
                        'G_rectus',
                        'S_suborbital',
                        'G_subcallosal',

                        # PARCELATION LEFT:
                        # -----------------

                        # visual:
                        'lh.Pole_occipital',
                        'lh.G_occipital_sup',
                        'lh.S_oc_sup_and_transversal',
                        'lh.G_occipital_middle',
                        'lh.S_occipital_ant',
                        'lh.S_oc_middle_and_Lunatus',
                        'lh.G_and_S_occipital_inf',
                        'lh.S_collat_transv_post',
                        'lh.G_oc-temp_med-Lingual',
                        'lh.S_calcarine',
                        'lh.G_cuneus',

                        # temporal:
                        'lh.Lat_Fis-post',
                        'lh.G_temp_sup-Plan_tempo',
                        'lh.S_temporal_transverse',
                        'lh.G_temp_sup-G_T_transv',
                        'lh.G_temp_sup-Lateral',
                        'lh.S_temporal_sup',
                        'lh.G_temporal_middle',
                        'lh.S_temporal_inf',
                        'lh.G_temporal_inf',
                        'lh.S_oc-temp_lat',
                        'lh.G_oc-temp_med-Parahip',
                        'lh.S_collat_transv_ant',
                        'lh.G_oc-temp_lat-fusifor',
                        'lh.S_oc-temp_med_and_Lingual',
                        'lh.G_temp_sup-Plan_polar',
                        'lh.Pole_temporal',

                        # parietal:
                        'lh.S_parieto_occipital',
                        'lh.S_subparietal',
                        'lh.G_precuneus',
                        'lh.G_parietal_sup',
                        'lh.S_intrapariet_and_P_trans',
                        'lh.G_pariet_inf-Angular',
                        'lh.S_interm_prim-Jensen',
                        'lh.G_and_S_paracentral',
                        'lh.S_postcentral',
                        'lh.G_postcentral',
                        'lh.S_central',
                        'lh.G_pariet_inf-Supramar',
                        'lh.G_and_S_subcentral',

                        # insular:
                        'lh.S_circular_insula_sup',
                        'lh.G_insular_short',
                        'lh.S_circular_insula_inf',
                        'lh.G_Ins_lg_and_S_cent_ins',
                        'lh.S_circular_insula_ant',

                        # cingulate:
                        'lh.G_cingul-Post-ventral',
                        'lh.S_pericallosal',
                        'lh.G_cingul-Post-dorsal',
                        'lh.S_cingul-Marginalis',
                        'lh.G_and_S_cingul-Mid-Post',
                        'lh.G_and_S_cingul-Mid-Ant',
                        'lh.G_and_S_cingul-Ant',

                        # frontal:
                        'lh.G_precentral',
                        'lh.S_precentral-sup-part',
                        'lh.S_precentral-inf-part',
                        'lh.G_front_sup',
                        'lh.S_front_sup',
                        'lh.G_front_middle',
                        'lh.S_front_middle',
                        'lh.S_front_inf',
                        'lh.G_front_inf-Opercular',
                        'lh.G_front_inf-Triangul',
                        'lh.S_orbital_lateral',
                        'lh.Lat_Fis-ant-Horizont',
                        'lh.Lat_Fis-ant-Vertical',
                        'lh.G_front_inf-Orbital',
                        'lh.G_and_S_transv_frontopol',
                        'lh.G_and_S_frontomargin',
                        'lh.G_orbital',
                        'lh.S_orbital-H_Shaped',
                        'lh.S_orbital_med-olfact',
                        'lh.G_rectus',
                        'lh.S_suborbital',
                        'lh.G_subcallosal',

                        # PARCELATION RIGHT:
                        # -----------------

                        # visual:
                        'rh.Pole_occipital',
                        'rh.G_occipital_sup',
                        'rh.S_oc_sup_and_transversal',
                        'rh.G_occipital_middle',
                        'rh.S_occipital_ant',
                        'rh.S_oc_middle_and_Lunatus',
                        'rh.G_and_S_occipital_inf',
                        'rh.S_collat_transv_post',
                        'rh.G_oc-temp_med-Lingual',
                        'rh.S_calcarine',
                        'rh.G_cuneus',

                        # temporal:
                        'rh.Lat_Fis-post',
                        'rh.G_temp_sup-Plan_tempo',
                        'rh.S_temporal_transverse',
                        'rh.G_temp_sup-G_T_transv',
                        'rh.G_temp_sup-Lateral',
                        'rh.S_temporal_sup',
                        'rh.G_temporal_middle',
                        'rh.S_temporal_inf',
                        'rh.G_temporal_inf',
                        'rh.S_oc-temp_lat',
                        'rh.G_oc-temp_med-Parahip',
                        'rh.S_collat_transv_ant',
                        'rh.G_oc-temp_lat-fusifor',
                        'rh.S_oc-temp_med_and_Lingual',
                        'rh.G_temp_sup-Plan_polar',
                        'rh.Pole_temporal',

                        # parietal:
                        'rh.S_parieto_occipital',
                        'rh.S_subparietal',
                        'rh.G_precuneus',
                        'rh.G_parietal_sup',
                        'rh.S_intrapariet_and_P_trans',
                        'rh.G_pariet_inf-Angular',
                        'rh.S_interm_prim-Jensen',
                        'rh.G_and_S_paracentral',
                        'rh.S_postcentral',
                        'rh.G_postcentral',
                        'rh.S_central',
                        'rh.G_pariet_inf-Supramar',
                        'rh.G_and_S_subcentral',

                        # insular:
                        'rh.S_circular_insula_sup',
                        'rh.G_insular_short',
                        'rh.S_circular_insula_inf',
                        'rh.G_Ins_lg_and_S_cent_ins',
                        'rh.S_circular_insula_ant',

                        # cingulate:
                        'rh.G_cingul-Post-ventral',
                        'rh.S_pericallosal',
                        'rh.G_cingul-Post-dorsal',
                        'rh.S_cingul-Marginalis',
                        'rh.G_and_S_cingul-Mid-Post',
                        'rh.G_and_S_cingul-Mid-Ant',
                        'rh.G_and_S_cingul-Ant',

                        # frontal:
                        'rh.G_precentral',
                        'rh.S_precentral-sup-part',
                        'rh.S_precentral-inf-part',
                        'rh.G_front_sup',
                        'rh.S_front_sup',
                        'rh.G_front_middle',
                        'rh.S_front_middle',
                        'rh.S_front_inf',
                        'rh.G_front_inf-Opercular',
                        'rh.G_front_inf-Triangul',
                        'rh.S_orbital_lateral',
                        'rh.Lat_Fis-ant-Horizont',
                        'rh.Lat_Fis-ant-Vertical',
                        'rh.G_front_inf-Orbital',
                        'rh.G_and_S_transv_frontopol',
                        'rh.G_and_S_frontomargin',
                        'rh.G_orbital',
                        'rh.S_orbital-H_Shaped',
                        'rh.S_orbital_med-olfact',
                        'rh.G_rectus',
                        'rh.S_suborbital',
                        'rh.G_subcallosal',
                    ]
        if time_locked == 'stim_locked':
            step_range = [-4,12]
            time_of_interest = [2,12]
        if time_locked == 'resp_locked':
            step_range = [-6,12]
            time_of_interest = [2,12]
        time_of_interest_b = [-2,2]
        
        # dataframe:
        self.data_frame = pd.concat(self.pupil_data)
        
        # add baseline BOLD measures based on stim-locked data:
        for roi in ROIs:
            BOLD_data = []
            for i in range(len(self.subjects)):
                try:
                    BOLD_data.append( np.load(os.path.join(self.data_folder, 'event_related_average', 'full_array_{}_{}_{}_{}.npy'.format(data_type, roi, 'stim_locked', self.subjects[i])))[-self.omissions[i],:] )
                except:
                    shell()
            # length kernel:
            kernel_length = BOLD_data[0].shape[1]
            
            # step size:
            step = pd.Series(np.linspace(step_range[0], step_range[1], kernel_length), name='time from cue (s)')
            time_of_interest_b_ind = np.array(step >= time_of_interest_b[0]) * np.array(step < time_of_interest_b[1])
            
            # scalars:
            main_BOLD_b = []
            for i in range(len(self.subjects)):
                baselines = bn.nanmean(BOLD_data[i][:,time_of_interest_b_ind], axis=1)
                if regress_iti:
                    baselines = myfuncs.lin_regress_resid(baselines, np.array(self.data_frame['iti'][self.data_frame.subject == self.subjects[i]]))
                main_BOLD_b.append( baselines )
            self.data_frame[roi+'_b'] = np.concatenate(main_BOLD_b)
        
            # add phasic BOLD measures based on either stim-locked or response-locked data:
            # for roi in ROIs:
            BOLD_data = []
            for i in range(len(self.subjects)):
                try:
                    BOLD_data.append( np.load(os.path.join(self.data_folder, 'event_related_average', 'full_array_{}_{}_{}_{}.npy'.format(data_type, roi, time_locked, self.subjects[i])))[-self.omissions[i],:] )
                except:
                    shell()
            # length kernel:
            kernel_length = BOLD_data[0].shape[1]
            
            # step size:
            if time_locked == 'stim_locked':
                step = pd.Series(np.linspace(step_range[0], step_range[1], kernel_length), name='time from cue (s)')
            if time_locked == 'resp_locked':
                step = pd.Series(np.linspace(step_range[0], step_range[1], kernel_length), name='time from report (s)')
            time_of_interest_ind = np.array(step >= time_of_interest[0]) * np.array(step < time_of_interest[1])
            
            # shell()
            
            # scalars:
            main_BOLD = []
            for i in range(len(self.subjects)):
                phasics = bn.nanmean(BOLD_data[i][:,time_of_interest_ind], axis=1)
                phasics = phasics - main_BOLD_b[i]
                for s in np.array(np.unique(self.pupil_data[i]['session']), dtype=int):
                    if regress_rt & regress_stimulus:
                         phasics[np.array(self.pupil_data[i].session == s)] = myfuncs.lin_regress_resid(phasics[np.array(self.pupil_data[i].session == s)], [np.array(self.data_frame['rt'][self.data_frame.subject == self.subjects[i]])[np.array(self.pupil_data[i].session == s)], np.array(self.data_frame['present'][self.data_frame.subject == self.subjects[i]], dtype=int)[np.array(self.pupil_data[i].session == s)]] ) + phasics[np.array(self.pupil_data[i].session == s)].mean()
                    elif regress_rt:
                        phasics[np.array(self.pupil_data[i].session == s)] = myfuncs.lin_regress_resid(phasics[np.array(self.pupil_data[i].session == s)], [np.array(self.data_frame['rt'][self.data_frame.subject == self.subjects[i]])[np.array(self.pupil_data[i].session == s)],] ) + phasics[np.array(self.pupil_data[i].session == s)].mean()
                main_BOLD.append( phasics )
                
            self.data_frame[roi+'_d'] = np.concatenate(main_BOLD)
        
            # add trial-related BOLD_measures:
            self.data_frame[roi+'_t'] = self.data_frame[roi+'_b'] + self.data_frame[roi+'_d']
        
        if regress_iti:
            pupil_b = []
            for i in range(len(self.subjects)):
                pupil_b.append(myfuncs.lin_regress_resid(np.array(self.data_frame[self.data_frame.subject == self.subjects[i]]['pupil_b']), np.array(self.data_frame['iti'][self.data_frame.subject == self.subjects[i]])))
            self.data_frame['pupil_b'] = np.concatenate(pupil_b)
        if regress_rt & regress_stimulus: 
            pupil_d = []
            pupil_t = []
            for i in range(len(self.subjects)):
                 for s in np.array(np.unique(self.pupil_data[i]['session']), dtype=int):
                    rt = np.array(self.pupil_data[i]['rt'])[np.array(self.pupil_data[i].session == s)]
                    present = np.array(self.pupil_data[i]['present'], dtype=int)[np.array(self.pupil_data[i].session == s)]
                    pupil_d.append(myfuncs.lin_regress_resid(np.array(self.data_frame[self.data_frame.subject == self.subjects[i]]['pupil_d'])[np.array(self.pupil_data[i].session == s)], [rt, present] ) + np.array(self.data_frame[self.data_frame.subject == self.subjects[i]]['pupil_d'])[np.array(self.pupil_data[i].session == s)].mean())
                    pupil_t.append(myfuncs.lin_regress_resid(np.array(self.data_frame[self.data_frame.subject == self.subjects[i]]['pupil_t'])[np.array(self.pupil_data[i].session == s)], [rt, present] ) + np.array(self.data_frame[self.data_frame.subject == self.subjects[i]]['pupil_t'])[np.array(self.pupil_data[i].session == s)].mean())
            self.data_frame['pupil_d'] = np.concatenate(pupil_d)
            self.data_frame['pupil_t'] = np.concatenate(pupil_t)
        if regress_rt: 
            pupil_d = []
            pupil_t = []
            for i in range(len(self.subjects)):
                 for s in np.array(np.unique(self.pupil_data[i]['session']), dtype=int):
                    rt = np.array(self.pupil_data[i]['rt'])[np.array(self.pupil_data[i].session == s)]
                    pupil_d.append(myfuncs.lin_regress_resid(np.array(self.data_frame[self.data_frame.subject == self.subjects[i]]['pupil_d'])[np.array(self.pupil_data[i].session == s)], [rt] ) + np.array(self.data_frame[self.data_frame.subject == self.subjects[i]]['pupil_d'])[np.array(self.pupil_data[i].session == s)].mean())
                    pupil_t.append(myfuncs.lin_regress_resid(np.array(self.data_frame[self.data_frame.subject == self.subjects[i]]['pupil_t'])[np.array(self.pupil_data[i].session == s)], [rt] ) + np.array(self.data_frame[self.data_frame.subject == self.subjects[i]]['pupil_t'])[np.array(self.pupil_data[i].session == s)].mean())
            self.data_frame['pupil_d'] = np.concatenate(pupil_d)
            self.data_frame['pupil_t'] = np.concatenate(pupil_t)
        
        
        # save for DDM
        # d = {
        # 'LC_standard_1': self.data_frame['LC_standard_1'],
        # 'LC_standard_1_b': self.data_frame['LC_standard_1_b'],
        # 'LC_JW': self.data_frame['LC_JW'],
        # 'LC_JW_b': self.data_frame['LC_JW_b'],
        # 'mean_fullMB': self.data_frame['mean_fullMB'],
        # 'mean_fullMB_b': self.data_frame['mean_fullMB_b'],
        # 'omissions' : self.data_frame['omissions'],
        # 'rt_compare' : self.data_frame['rt'],
        # }
        # data_response = pd.DataFrame(d)
        # data_response.to_csv(os.path.join(self.base_dir, 'fMRI_data_frame.csv'))
    
    
    def plot_mean_bars(self, roi, data_type, smooth=False, event_average=False, type_response='mean', stratification=False, project_out=False):
        
        import copy
        
        self.subjects = np.array(self.subjects)
        
        self.make_pupil_BOLD_dataframe(data_type=data_type, regress_iti=False, regress_rt=False,)
        
        ylim = (-0.5,1.5)
        ml = 0.5
        if roi in ['LC_standard_2',
            'LC_standard_1',
            'mean_fullMB',
            'mean_SN',
            'mean_VTA',
            'basal_forebrain_4',
            'basal_forebrain_123',
            'AAN_VTA',
            'AAN_PAG',
            'AAN_PBC',
            'AAN_PO',
            'AAN_PPN',
            'AAN_LC',
            'AAN_MR',
            'AAN_MRF',
            'AAN_DR',
            '4th_ventricle',
            'LC_JW']:
            print 'hallo!'
            ylim = (-0.3,0.6)
        
        phasics_h = np.zeros(len(self.subjects))
        phasics_l = np.zeros(len(self.subjects))
        
        for i in range(len(self.subjects)):
            
            if type_response == 'mean':
                phasics_h[i] = np.array(self.data_frame[(self.data_frame.subject == self.subjects[i]) & np.concatenate(self.pupil_h_ind)][roi+'_d']).mean()
                phasics_l[i] = np.array(self.data_frame[(self.data_frame.subject == self.subjects[i]) & np.concatenate(self.pupil_l_ind)][roi+'_d']).mean()
            if type_response == 'std':
                phasics_h[i] = np.array(self.data_frame[(self.data_frame.subject == self.subjects[i]) & np.concatenate(self.pupil_h_ind)][roi+'_d']).std()
                phasics_l[i] = np.array(self.data_frame[(self.data_frame.subject == self.subjects[i]) & np.concatenate(self.pupil_l_ind)][roi+'_d']).std()
            
        values = [phasics_h, phasics_l,]
        MEANS = np.array([v.mean() for v in values])
        SEMS = np.array([sp.stats.sem(v) for v in values])
        
        # p1 = sp.stats.ttest_rel(values[0], values[1])[1]
        p1 = myfuncs.permutationTest(values[0], values[1], paired=True)[1]
        
        # shell()
        
        ind = np.arange(0,len(MEANS))
        bar_width = 0.60
    
        fig = plt.figure(figsize=(1.5,2))
        ax = fig.add_subplot(111)
        my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}
        for i in range(len(MEANS)):
            ax.bar(ind[i], MEANS[i], width = bar_width, yerr=SEMS[i], color = ['r','b',][i], alpha = [1, 1][i], align = 'center', edgecolor = 'k', linewidth = 0, capsize = 0, ecolor = 'k', error_kw={'elinewidth':0.5,})
        ax.set_ylim(ylim)
        ax.tick_params(axis='y', which='major', labelsize=6)
        ax.set_xticks(ind)
        ax.text(s=str(round(p1,3)), x=(ind[0]+ind[1])/2.0, y=ax.get_ylim()[1]-((ax.get_ylim()[1] - ax.get_ylim()[0]) / 10.0), size=7, horizontalalignment='center')
        if type_response == 'mean':
            ax.set_ylabel('fMRI response\n (% signal change)')        
        if type_response == 'std':
            ax.set_ylabel('fMRI response\nvariability (s.d.)')
        ax.set_title(roi)
        sns.despine(offset=10, trim=True)
        ax.set_xticklabels(['evoked', 'evoked'], rotation=45, size=7)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder, 'event_related_average', data_type, type_response, 'bars_{}_{}.pdf'.format(roi, self.split_by)))
    
    def plot_mean_bars_all(self, roi, data_type, smooth=False, event_average=False, type_response='mean', stratification=False, project_out=False):
        
        import copy
        
        self.subjects = np.array(self.subjects)
        
        self.make_pupil_BOLD_dataframe(data_type=data_type, regress_iti=False, regress_rt=False,)
        
        if type_response == 'mean':
            ylim = (-1, 1.5)
        elif type_response == 'std':
            ylim = (0, 5)
        
        baselines_h = np.zeros(len(self.subjects))
        baselines_l = np.zeros(len(self.subjects))
        phasics_h = np.zeros(len(self.subjects))
        phasics_l = np.zeros(len(self.subjects))
        trials_h = np.zeros(len(self.subjects))
        trials_l = np.zeros(len(self.subjects))
        
        for i in range(len(self.subjects)):
            
            if type_response == 'mean':
                baselines_h[i] = np.array(self.data_frame[(self.data_frame.subject == self.subjects[i]) & np.concatenate(self.pupil_h_ind)][roi+'_b']).mean()
                baselines_l[i] = np.array(self.data_frame[(self.data_frame.subject == self.subjects[i]) & np.concatenate(self.pupil_l_ind)][roi+'_b']).mean()
                phasics_h[i] = np.array(self.data_frame[(self.data_frame.subject == self.subjects[i]) & np.concatenate(self.pupil_h_ind)][roi+'_d']).mean()
                phasics_l[i] = np.array(self.data_frame[(self.data_frame.subject == self.subjects[i]) & np.concatenate(self.pupil_l_ind)][roi+'_d']).mean()
                trials_h[i] = np.array(self.data_frame[(self.data_frame.subject == self.subjects[i]) & np.concatenate(self.pupil_h_ind)][roi+'_t']).mean()
                trials_l[i] = np.array(self.data_frame[(self.data_frame.subject == self.subjects[i]) & np.concatenate(self.pupil_l_ind)][roi+'_t']).mean()
            if type_response == 'std':
                baselines_h[i] = np.array(self.data_frame[(self.data_frame.subject == self.subjects[i]) & np.concatenate(self.pupil_h_ind)][roi+'_b']).std()
                baselines_l[i] = np.array(self.data_frame[(self.data_frame.subject == self.subjects[i]) & np.concatenate(self.pupil_l_ind)][roi+'_b']).std()
                phasics_h[i] = np.array(self.data_frame[(self.data_frame.subject == self.subjects[i]) & np.concatenate(self.pupil_h_ind)][roi+'_d']).std()
                phasics_l[i] = np.array(self.data_frame[(self.data_frame.subject == self.subjects[i]) & np.concatenate(self.pupil_l_ind)][roi+'_d']).std()
                trials_h[i] = np.array(self.data_frame[(self.data_frame.subject == self.subjects[i]) & np.concatenate(self.pupil_h_ind)][roi+'_t']).std()
                trials_l[i] = np.array(self.data_frame[(self.data_frame.subject == self.subjects[i]) & np.concatenate(self.pupil_l_ind)][roi+'_t']).std()
        
        
        values = [baselines_h, baselines_l, trials_h, trials_l, phasics_h, phasics_l]
        MEANS = np.array([v.mean() for v in values])
        SEMS = np.array([sp.stats.sem(v) for v in values])
        
        p1 = sp.stats.ttest_rel(values[0], values[1])[1]
        p2 = sp.stats.ttest_rel(values[2], values[3])[1]
        p3 = sp.stats.ttest_rel(values[4], values[5])[1]
        
        ind = np.arange(0,len(MEANS))
        bar_width = 0.60
    
        fig = plt.figure(figsize=(2,1.75))
        ax = fig.add_subplot(111)
        my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}
        for i in range(len(MEANS)):
            ax.bar(ind[i], MEANS[i], width = bar_width, yerr=SEMS[i], color = ['r','b','r','b','r','b'][i], alpha = [0.5, 0.5, 1, 1, 1, 1][i], align = 'center', edgecolor = 'k', linewidth = 0, capsize = 0, ecolor = 'k', error_kw={'elinewidth':0.5,})
        ax.set_ylim(ylim)
        ax.tick_params(axis='y', which='major', labelsize=6)
        ax.set_xticks(ind)
        ax.text(s=str(round(p1,3)), x=(ind[0]+ind[1])/2.0, y=ax.get_ylim()[1]-((ax.get_ylim()[1] - ax.get_ylim()[0]) / 10.0), size=7, horizontalalignment='center')
        ax.text(s=str(round(p2,3)), x=(ind[2]+ind[3])/2.0, y=ax.get_ylim()[1]-((ax.get_ylim()[1] - ax.get_ylim()[0]) / 10.0), size=7, horizontalalignment='center')
        ax.text(s=str(round(p3,3)), x=(ind[4]+ind[5])/2.0, y=ax.get_ylim()[1]-((ax.get_ylim()[1] - ax.get_ylim()[0]) / 10.0), size=7, horizontalalignment='center')
        if type_response == 'mean':
            ax.set_ylabel('fMRI response\n (% signal change)')        
        if type_response == 'std':
            ax.set_ylabel('fMRI response\nvariability (s.d.)')
        sns.despine(offset=10, trim=True)
        ax.set_xticklabels(['baseline', 'baseline', 'trial', 'trial', 'evoked', 'evoked'], rotation=45, size=7)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder, 'event_related_average', data_type, type_response, 'bars_{}_{}.pdf'.format(roi, self.split_by)))
        
        
        
        
    
    def plot_event_average_all_trials(self, roi, data_type, smooth=False, event_average=False, type_response='mean', stratification=False, project_out=False):
        
        import copy
        
        self.subjects = np.array(self.subjects)
        
        self.make_pupil_BOLD_dataframe(data_type=data_type, regress_iti=False, regress_rt=False,)
        
        sample_rate = 2
        
        ylim = (-0.1,0.5)
        ml = 0.5
        if roi in [
            'LC_standard_2',
            'LC_standard_1',
            'mean_fullMB',
            'mean_SN',
            'mean_VTA',
            'basal_forebrain_4',
            'basal_forebrain_123',
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
            ]:
            print 'hallo!'
            ylim = (-0.1,0.2)
            ml = 0.25
        elif roi in [
            'sup_col_jw',
            ]:
            print 'hallo!'
            ylim = (-0.2,0.5)
            ml = 0.25
        elif roi in [
            'LC_JW',
            'LC_JW_nn',
            '4th_ventricle',
            ]:
            print 'hallo!'
            ylim = (-0.1,0.15)
            ml = 0.25
        if type_response == 'std':
            ylim = (-0.04,0.02)

        
        fig = plt.figure(figsize=(1.75,1.75))
        ax = fig.add_subplot(111)
        # for time_locked in ['stim_locked', 'resp_locked']:
        for time_locked in ['stim_locked']:
            if event_average:
                if time_locked == 'stim_locked':
                    step_lim = [-4,12]
                    xlim = [-2,12]
                    time_of_interest = [2,12]
                if time_locked == 'resp_locked':
                    step_lim = [-6,12]
                    xlim = [-4,12]
                    time_of_interest = [2,12]
                time_of_interest_b = [-2,2]
            
            # load data for plotting:
            # -----------------------
            kernels_condition1 = []
            kernels_condition2 = []
            kernels_condition3 = []
            for s in self.subjects:
                if event_average:
                    k1, k2, k3 = self.load_BOLD_per_subject(subject=s, data_type=data_type, roi=roi, time_locked=time_locked, type_response=type_response, stratification=stratification, project_out=project_out) 
                    kernels_condition1.append(k1)
                    kernels_condition2.append(k2)
                    kernels_condition3.append(k3)
                else:
                    kernels = np.load(os.path.join(self.data_folder, 'deconvolution', '{}_{}_{}_{}_{}.npy'.format(data_type, roi, c, time_locked, self.subjects[i])))
            kernels_condition1 = np.vstack(kernels_condition1)
            kernels_condition2 = np.vstack(kernels_condition2)
            kernels_condition3 = np.vstack(kernels_condition3)
            
            # length kernel:
            kernel_length = kernels_condition1.shape[1]
            
            # step size:
            if time_locked == 'stim_locked':
                step = pd.Series(np.linspace(step_lim[0], step_lim[1], kernel_length), name='Time from cue (s)')
            if time_locked == 'resp_locked':
                step = pd.Series(np.linspace(step_lim[0], step_lim[1], kernel_length), name='Time from report (s)')
            time_of_interest_ind = np.array(step >= time_of_interest[0]) * np.array(step < time_of_interest[1])
            time_of_interest_b_ind = np.array(step >= time_of_interest_b[0]) * np.array(step < time_of_interest_b[1])
            
            xlim_indices = np.array(step >= xlim[0]) * np.array(step <= xlim[1])
            
            # shell()
            
            # indices:
            if 'pupil' in self.split_by:
                ind1 = np.concatenate(copy.copy(self.pupil_h_ind))
                ind2 = np.concatenate(copy.copy(self.pupil_l_ind))
            elif self.split_by == 'yes':
                ind1 = np.concatenate(copy.copy(self.yes))
                ind2 = np.concatenate(copy.copy(self.no))
            elif self.split_by == 'correct':
                ind1 = np.concatenate(copy.copy(self.correct))
                ind2 = np.concatenate(copy.copy(self.error))
            elif self.split_by == 'present':
                ind1 = np.concatenate(copy.copy(self.present))
                ind2 = np.concatenate(copy.copy(self.absent))
            
            if 'pupil' in self.split_by:
                ind3 = np.ones(len(ind1), dtype=bool)
            else:
                ind3 = ind2
            
            # scalars:
            if type_response == 'mean':
                baselines_1 = np.array([np.array(self.data_frame[(self.data_frame.subject == s) & ind1][roi+'_b']).mean() for s in self.subjects])
                baselines_2 = np.array([np.array(self.data_frame[(self.data_frame.subject == s) & ind2][roi+'_b']).mean() for s in self.subjects])
                baselines_3 = np.array([np.array(self.data_frame[(self.data_frame.subject == s) & ind3][roi+'_b']).mean() for s in self.subjects])
            elif type_response == 'std':
                baselines_1 = np.array([kernels_condition1[i,time_of_interest_b_ind].mean() for i in range(len(self.subjects))])
                baselines_2 = np.array([kernels_condition2[i,time_of_interest_b_ind].mean() for i in range(len(self.subjects))])
                baselines_3 = np.array([kernels_condition3[i,time_of_interest_b_ind].mean() for i in range(len(self.subjects))])
                
            # baseline:
            for i in range(len(self.subjects)):
                kernels_condition1[i,:] = kernels_condition1[i,:] - baselines_1[i]
                kernels_condition2[i,:] = kernels_condition2[i,:] - baselines_2[i]
                kernels_condition3[i,:] = kernels_condition3[i,:] - baselines_3[i]
            
            if type_response == 'mean':
                phasics_1 = np.array([np.array(self.data_frame[(self.data_frame.subject == s) & ind1][roi + self.postfix]).mean() for s in self.subjects])
                phasics_2 = np.array([np.array(self.data_frame[(self.data_frame.subject == s) & ind2][roi + self.postfix]).mean() for s in self.subjects])
                phasics_3 = np.array([np.array(self.data_frame[(self.data_frame.subject == s) & ind3][roi + self.postfix]).mean() for s in self.subjects])
            elif type_response == 'std':
                phasics_1 = np.array([kernels_condition1[i,time_of_interest_ind].mean() for i in range(len(self.subjects))])
                phasics_2 = np.array([kernels_condition2[i,time_of_interest_ind].mean() for i in range(len(self.subjects))])
                phasics_3 = np.array([kernels_condition3[i,time_of_interest_ind].mean() for i in range(len(self.subjects))])
            
            scalars = np.vstack((baselines_1, baselines_2, phasics_1, phasics_2)).T
                
            sem = sp.stats.sem(kernels_condition3, axis=0)
            
            # plot:
            # -----
            # timeseries:
            
            # shell()
            ax.axvspan(np.array(step[time_of_interest_ind])[0], np.array(step[time_of_interest_ind])[-1], facecolor='k', alpha=0.1)
            ax.axvline(0, lw=0.25, alpha=0.5, color = 'k')
            ax.axhline(0, lw=0.25, alpha=0.5, color = 'k')
            colors = ['black']
            conditions = pd.Series([self.split_by], name='trial type')
            if type_response == 'std':
                ax.fill_between(np.array(step)[xlim_indices], kernels_condition3[:,xlim_indices].mean(axis=0)-sem[xlim_indices], kernels_condition3[:,xlim_indices].mean(axis=0)+sem[xlim_indices], color='m', alpha=0.25)
                sns.tsplot(kernels_condition3[:,xlim_indices], time=step[xlim_indices], condition=conditions, value='fMRI response\nvariability (s.d.)', color='m', err_style=None, lw=1, ls='-', ax=ax)
            elif type_response == 'var':
                sns.tsplot(kernels_condition3[:,xlim_indices], time=step[xlim_indices], condition=conditions, value='fMRI response\nvariability (var)', color='k', err_style=None, lw=1, ls='-', ax=ax)
            else:
                ax.fill_between(np.array(step)[xlim_indices], kernels_condition3[:,xlim_indices].mean(axis=0)-sem[xlim_indices], kernels_condition3[:,xlim_indices].mean(axis=0)+sem[xlim_indices], color='k', alpha=0.25)
                sns.tsplot(kernels_condition3[:,xlim_indices], time=step[xlim_indices], condition=conditions, value='fMRI response\n(% signal change)', color='k', err_style=None, lw=1, ls='-', ax=ax)
            
            conditions = pd.Series([self.split_by], name='trial type')
            ax.legend_ = None
            ax.set_xlabel(xlabel='Time from cue (s)')
            # ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            myfuncs.cluster_sig_bar_1samp(array=kernels_condition3[:,xlim_indices], x=np.array(step[xlim_indices]), yloc=1, color='g', ax=ax, threshold=0.05, nrand=5000)
            y_limits = ax.get_ylim()
            # ax.xaxis.set_major_locator(MultipleLocator(6))
            # if type_response == 'mean':
            #     ax.yaxis.set_major_locator(MultipleLocator(ml))
        sns.despine(offset=10, trim=True)
        # plt.setp( axes[3].xaxis.get_majorticklabels(), rotation=45 )
        # axes[3].yaxis.set_visible(False)
        plt.tight_layout()
        if event_average:
            if stratification and not project_out:
                fig.savefig(os.path.join(self.figure_folder, 'event_related_average', data_type, type_response, 'stratification', 'all_trials_{}_{}.pdf'.format(roi, self.split_by)))
            elif project_out and not stratification:
                fig.savefig(os.path.join(self.figure_folder, 'event_related_average', data_type, type_response, 'projectout', 'all_trials_{}_{}.pdf'.format(roi, self.split_by)))
            elif stratification and project_out:
                fig.savefig(os.path.join(self.figure_folder, 'event_related_average', data_type, type_response, 'stratification_projectout', 'all_trials_{}_{}.pdf'.format(roi, self.split_by)))
            else:
                fig.savefig(os.path.join(self.figure_folder, 'event_related_average', data_type, type_response, 'all_trials_{}_{}.pdf'.format(roi, self.split_by)))
        
        # shell()
        
        
    def plot_event_average(self, roi, data_type, smooth=False, event_average=False, type_response='mean', stratification=False, project_out=False):
        
        import copy
        
        self.subjects = np.array(self.subjects)
        
        self.make_pupil_BOLD_dataframe(data_type=data_type, regress_iti=False, regress_rt=False,)
        
        sample_rate = 2
        
        ylim = (-0.5,2)
        ml = 0.5
        if roi in [
            'LC_standard_2',
            'LC_standard_1',
            'mean_fullMB',
            'mean_SN',
            'mean_VTA',
            'basal_forebrain_4',
            'basal_forebrain_123',
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
            ]:
            print 'hallo!'
            ylim = (-0.1,0.2)
            ml = 0.25
        elif roi in [
            'sup_col_jw',
            ]:
            print 'hallo!'
            ylim = (-0.2,0.5)
            ml = 0.25
        elif roi in [
            'LC_JW',
            'LC_JW_nn',
            '4th_ventricle',
            ]:
            print 'hallo!'
            ylim = (-0.1,0.15)
            ml = 0.25
        if type_response == 'std':
            ylim = (-0.04,0.02)
        
        # fig_spacing = (4,14)
        # fig = plt.figure(figsize=(8.27, 5.845))
        # ax = plt.subplot2grid(fig_spacing, (0, 0), colspan=2)
        
        fig = plt.figure(figsize=(1.9,1.75))
        ax = fig.add_subplot(111)
        
        for time_locked in ['stim_locked']:
            if event_average:
                if time_locked == 'stim_locked':
                    step_lim = [-4,12]
                    xlim = [-2,12]
                    time_of_interest = [2,12]
                if time_locked == 'resp_locked':
                    step_lim = [-6,12]
                    xlim = [-4,12]
                    time_of_interest = [2,12]
                time_of_interest_b = [-2,2]
            
            # load data for plotting:
            # -----------------------
            kernels_condition1 = []
            kernels_condition2 = []
            kernels_condition3 = []
            for s in self.subjects:
                if event_average:
                    k1, k2, k3 = self.load_BOLD_per_subject(subject=s, data_type=data_type, roi=roi, time_locked=time_locked, type_response=type_response, stratification=stratification, project_out=project_out) 
                    kernels_condition1.append(k1)
                    kernels_condition2.append(k2)
                    kernels_condition3.append(k3)
                else:
                    kernels = np.load(os.path.join(self.data_folder, 'deconvolution', '{}_{}_{}_{}_{}.npy'.format(data_type, roi, c, time_locked, self.subjects[i])))
            kernels_condition1 = np.vstack(kernels_condition1)
            kernels_condition2 = np.vstack(kernels_condition2)
            kernels_condition3 = np.vstack(kernels_condition3)
            
            # length kernel:
            kernel_length = kernels_condition1.shape[1]
            
            # step size:
            if time_locked == 'stim_locked':
                step = pd.Series(np.linspace(step_lim[0], step_lim[1], kernel_length), name='Time from cue (s)')
            if time_locked == 'resp_locked':
                step = pd.Series(np.linspace(step_lim[0], step_lim[1], kernel_length), name='Time from report (s)')
            time_of_interest_ind = np.array(step >= time_of_interest[0]) * np.array(step < time_of_interest[1])
            time_of_interest_b_ind = np.array(step >= time_of_interest_b[0]) * np.array(step < time_of_interest_b[1])
            
            xlim_indices = np.array(step >= xlim[0]) * np.array(step <= xlim[1])
            
            # shell()
            
            # indices:
            if 'pupil' in self.split_by:
                ind1 = np.concatenate(copy.copy(self.pupil_h_ind))
                ind2 = np.concatenate(copy.copy(self.pupil_l_ind))
            elif self.split_by == 'yes':
                ind1 = np.concatenate(copy.copy(self.yes))
                ind2 = np.concatenate(copy.copy(self.no))
            elif self.split_by == 'correct':
                ind1 = np.concatenate(copy.copy(self.correct))
                ind2 = np.concatenate(copy.copy(self.error))
            elif self.split_by == 'present':
                ind1 = np.concatenate(copy.copy(self.present))
                ind2 = np.concatenate(copy.copy(self.absent))
            
            if 'pupil' in self.split_by:
                ind3 = -(ind1+ind2)
            else:
                ind3 = ind2
            
            # scalars:
            if type_response == 'mean':
                baselines_1 = np.array([bn.nanmean(np.array(self.data_frame[(self.data_frame.subject == s) & ind1][roi+'_b'])) for s in self.subjects])
                baselines_2 = np.array([bn.nanmean(np.array(self.data_frame[(self.data_frame.subject == s) & ind2][roi+'_b'])) for s in self.subjects])
                baselines_3 = np.array([bn.nanmean(np.array(self.data_frame[(self.data_frame.subject == s) & ind3][roi+'_b'])) for s in self.subjects])
            elif type_response == 'std':
                baselines_1 = np.array([kernels_condition1[i,time_of_interest_b_ind].mean() for i in range(len(self.subjects))])
                baselines_2 = np.array([kernels_condition2[i,time_of_interest_b_ind].mean() for i in range(len(self.subjects))])
                baselines_3 = np.array([kernels_condition3[i,time_of_interest_b_ind].mean() for i in range(len(self.subjects))])
                
            # baseline:
            for i in range(len(self.subjects)):
                kernels_condition1[i,:] = kernels_condition1[i,:] - baselines_1[i]
                kernels_condition2[i,:] = kernels_condition2[i,:] - baselines_2[i]
                kernels_condition3[i,:] = kernels_condition3[i,:] - baselines_3[i]
            
            if type_response == 'mean':
                phasics_1 = np.array([bn.nanmean(np.array(self.data_frame[(self.data_frame.subject == s) & ind1][roi + self.postfix])) for s in self.subjects])
                phasics_2 = np.array([bn.nanmean(np.array(self.data_frame[(self.data_frame.subject == s) & ind2][roi + self.postfix])) for s in self.subjects])
                phasics_3 = np.array([bn.nanmean(np.array(self.data_frame[(self.data_frame.subject == s) & ind3][roi + self.postfix])) for s in self.subjects])
            elif type_response == 'std':
                phasics_1 = np.array([kernels_condition1[i,time_of_interest_ind].mean() for i in range(len(self.subjects))])
                phasics_2 = np.array([kernels_condition2[i,time_of_interest_ind].mean() for i in range(len(self.subjects))])
                phasics_3 = np.array([kernels_condition3[i,time_of_interest_ind].mean() for i in range(len(self.subjects))])
            
            scalars = np.vstack((baselines_1, baselines_2, phasics_1, phasics_2)).T
                
            # stats:
            p_bars_b = sp.stats.ttest_rel(baselines_1, baselines_2)[1]
            p_bars = sp.stats.ttest_rel(phasics_1, phasics_2)[1]
            
            # print myfuncs.permutationTest(scalars[:,3], np.zeros(len(self.subjects)), paired=True)
            
            # baseline the kernels:
            # if self.split_by == 'pupil_d':

            
            # # stats across time series:
            # # -------------------------
            # p1 = np.zeros(kernel_length)
            # for i in range(kernel_length):
            #     means = np.vstack((kernels_condition1[:,i], kernels_condition2[:,i])).mean(axis=0)
            #     p1[i] = myfuncs.permutationTest(kernels_condition1[:,i], kernels_condition2[:,i], paired=True)[1]
            # p = mne.stats.fdr_correction(p1, 0.05,)
            # p[1][-1] = 1
            #
            # sig_indices = np.array(p[1] < 0.05, dtype=int)
            # sig_indices = np.array(p[1] < 0.05, dtype=int)
            # sig_indices[0] = 0
            # sig_indices[-1] = 0
            # s_bar = zip(np.where(np.diff(sig_indices)==1)[0]+1, np.where(np.diff(sig_indices)==-1)[0]+1)
            
            # normalize SEMS:
            meannn = (kernels_condition1+kernels_condition2)/2.0
            # sem_1 = sp.stats.sem(kernels_condition1-meannn, axis=0)
            # sem_2 = sp.stats.sem(kernels_condition1-meannn, axis=0)
            sem_1 = sp.stats.sem(kernels_condition1, axis=0)
            sem_2 = sp.stats.sem(kernels_condition1, axis=0)
            
            # plot:
            # -----
            # timeseries:
            
            # shell()
            
            # ax.axvspan(np.array(step[time_of_interest_b_ind])[0], np.array(step[time_of_interest_b_ind])[-1], facecolor='r', alpha=0.1)
            
            # shell()
            
            c1 = 'r'
            c2 = 'b'
            
            ax.axvspan(np.array(step[time_of_interest_ind])[0], np.array(step[time_of_interest_ind])[-1], facecolor='k', alpha=0.1)
            ax.axvline(0, lw=0.25, alpha=0.5, color = 'k')
            if not type_response == 'std':
                ax.axhline(0, lw=0.25, alpha=0.5, color = 'k')
            colors = ['black']
            conditions = pd.Series([self.split_by], name='trial type')
            ax.fill_between(np.array(step)[xlim_indices], kernels_condition1[:,xlim_indices].mean(axis=0)-sem_1[xlim_indices], kernels_condition1[:,xlim_indices].mean(axis=0)+sem_1[xlim_indices], color=c1, alpha=0.25)
            ax.fill_between(np.array(step)[xlim_indices], kernels_condition2[:,xlim_indices].mean(axis=0)-sem_2[xlim_indices], kernels_condition2[:,xlim_indices].mean(axis=0)+sem_2[xlim_indices], color=c2, alpha=0.25)
            if type_response == 'std':
                sns.tsplot(kernels_condition1[:,xlim_indices], time=step[xlim_indices], condition=conditions, value='fMRI response\nvariability (s.d.)', color=c1, err_style=None, lw=1, ls='-', ax=ax)
                sns.tsplot(kernels_condition2[:,xlim_indices], time=step[xlim_indices], condition=conditions, value='fMRI response\nvariability (s.d.)', color=c2, err_style=None, lw=1, ls='-', ax=ax)
            elif type_response == 'var':
                sns.tsplot(kernels_condition1[:,xlim_indices], time=step[xlim_indices], condition=conditions, value='fMRI response\nvariability (var)', color=c1, err_style=None, lw=1, ls='-', ax=ax)
                sns.tsplot(kernels_condition2[:,xlim_indices], time=step[xlim_indices], condition=conditions, value='fMRI response\nvariability (var)', color=c2, err_style=None, lw=1, ls='-', ax=ax)
            else:
                # if 'pupil' in self.split_by:
                #     ax.plot(np.array(step[xlim_indices]), kernels_condition3[:,xlim_indices].mean(axis=0), color='grey')
                sns.tsplot(kernels_condition1[:,xlim_indices], time=step[xlim_indices], condition=conditions, value='fMRI response\n(% signal change)', color=c1, err_style=None, lw=1, ls='-', ax=ax)
                sns.tsplot(kernels_condition2[:,xlim_indices], time=step[xlim_indices], condition=conditions, value='fMRI response\n(% signal change)', color=c2, err_style=None, lw=1, ls='-', ax=ax)
            
            conditions = pd.Series([self.split_by], name='trial type')
            ax.legend_ = None
            ax.set_title(roi)
            ax.set_xlabel(xlabel='Time from cue (s)')
            # ax.set_xlim(xlim)
            if not type_response == 'std':
                ax.set_ylim(ylim)
            myfuncs.cluster_sig_bar(array=[kernels_condition1[:,xlim_indices],kernels_condition2[:,xlim_indices]], x=np.array(step[xlim_indices]), yloc=1, color='g', ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)
            if not type_response == 'std':
                myfuncs.cluster_sig_bar_1samp(array=kernels_condition1[:,xlim_indices], x=np.array(step[xlim_indices]), yloc=2, color='r', ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)
                myfuncs.cluster_sig_bar_1samp(array=kernels_condition2[:,xlim_indices], x=np.array(step[xlim_indices]), yloc=3, color='b', ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)
            # shell()
            # for sig in s_bar:
            #     ax.hlines(((ax.get_ylim()[1] - ax.get_ylim()[0]) / 10.0)+ax.get_ylim()[0], step[int(sig[0])], step[int(sig[1])], lw=2, color='g')
            y_limits = ax.get_ylim()
            # ax.xaxis.set_major_locator(MultipleLocator(6))
            # if type_response == 'mean':
            #     ax.yaxis.set_major_locator(MultipleLocator(ml))
            # ax.locator_params(axis='y', nbins=6, tight=None)
            
            # shell()
            
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        if event_average:
            if stratification and not project_out:
                fig.savefig(os.path.join(self.figure_folder, 'event_related_average', data_type, type_response, 'stratification', '{}_{}.pdf'.format(roi, self.split_by)))
            elif project_out and not stratification:
                fig.savefig(os.path.join(self.figure_folder, 'event_related_average', data_type, type_response, 'projectout', '{}_{}.pdf'.format(roi, self.split_by)))
            elif stratification and project_out:
                fig.savefig(os.path.join(self.figure_folder, 'event_related_average', data_type, type_response, 'stratification_projectout', '{}_{}.pdf'.format(roi, self.split_by)))
            else:
                fig.savefig(os.path.join(self.figure_folder, 'event_related_average', data_type, type_response, '{}_{}.pdf'.format(roi, self.split_by)))
        
        if self.split_by == 'yes':
            choice_effect = scalars[:,2] - scalars[:,3]
            
            fig = plt.figure(figsize=(2,2))
            ax = fig.add_subplot(111)
            myfuncs.correlation_plot(self.criterion, choice_effect, ax=ax, line=True)
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.xaxis.set_major_locator(MaxNLocator(5))
            plt.xlabel('criterion')
            plt.ylabel('choice effect (yes-no)')
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, 'criterion_{}.pdf'.format(roi)))
    
    def ANOVA_LC(self):
        
        
        
        self.make_pupil_BOLD_dataframe(data_type='clean_4th_ventricle', regress_iti=False, regress_rt_signal=False)
        
        roi = 'LC_JW'
        ind1 = self.pupil_h_ind
        ind2 = self.pupil_l_ind
        baselines_LC_1 = np.array([np.array(self.data_frame[(self.data_frame.subject == s) & np.concatenate(ind1)][roi+'_b']).mean() for s in self.subjects])
        baselines_LC_0 = np.array([np.array(self.data_frame[(self.data_frame.subject == s) & np.concatenate(ind2)][roi+'_b']).mean() for s in self.subjects])
        phasics_LC_1 = np.array([np.array(self.data_frame[(self.data_frame.subject == s) & np.concatenate(ind1)][roi]).mean() for s in self.subjects])
        phasics_LC_0 = np.array([np.array(self.data_frame[(self.data_frame.subject == s) & np.concatenate(ind2)][roi]).mean() for s in self.subjects])
        
        self.make_pupil_BOLD_dataframe(data_type='clean_4th_ventricle', regress_iti=False, regress_rt_signal=False)
        
        roi = 'LC_standard_2_dilated_2'
        ind1 = self.pupil_h_ind
        ind2 = self.pupil_l_ind
        baselines_LC_2_1 = np.array([np.array(self.data_frame[(self.data_frame.subject == s) & np.concatenate(ind1)][roi+'_b']).mean() for s in self.subjects])
        baselines_LC_2_0 = np.array([np.array(self.data_frame[(self.data_frame.subject == s) & np.concatenate(ind2)][roi+'_b']).mean() for s in self.subjects])
        phasics_LC_2_1 = np.array([np.array(self.data_frame[(self.data_frame.subject == s) & np.concatenate(ind1)][roi]).mean() for s in self.subjects])
        phasics_LC_2_0 = np.array([np.array(self.data_frame[(self.data_frame.subject == s) & np.concatenate(ind2)][roi]).mean() for s in self.subjects])
        
        
        # ANOVA 1:
        data = np.concatenate((phasics_LC_0, phasics_LC_1, phasics_LC_2_0, phasics_LC_2_1))
        subject = np.concatenate((np.arange(len(self.subjects)), np.arange(len(self.subjects)), np.arange(len(self.subjects)), np.arange(len(self.subjects))))
        region = np.concatenate((np.ones(len(self.subjects)), np.ones(len(self.subjects)), np.zeros(len(self.subjects)), np.zeros(len(self.subjects))))
        pupil = np.concatenate((np.zeros(len(self.subjects)), np.ones(len(self.subjects)), np.zeros(len(self.subjects)), np.ones(len(self.subjects))))
        
        d = rlc.OrdDict([('region', robjects.IntVector(list(region.ravel()))), ('pupil', robjects.IntVector(list(pupil.ravel()))), ('subject', robjects.IntVector(list(subject.ravel()))), ('data', robjects.FloatVector(list(data.ravel())))])
        robjects.r.assign('dataf', robjects.DataFrame(d))
        robjects.r('attach(dataf)')
        statres = robjects.r('res = summary(aov(data ~ as.factor(region)*as.factor(pupil) + Error(as.factor(subject)), dataf))')
        p1 = statres[-1][0][4][0]    # we will log-transform and min p values
        p2 = statres[-1][0][4][1]    # we will log-transform and min p values
        p3 = statres[-1][0][4][2]    # we will log-transform and min p values
        
        print
        print statres
        
        # ANOVA 2:
        data = np.concatenate((phasics_LC_0, phasics_LC_1, phasics_LC_2_0, phasics_LC_2_1, baselines_LC_0, baselines_LC_1, baselines_LC_2_0, baselines_LC_2_1))
        subject = np.concatenate((np.arange(len(self.subjects)), np.arange(len(self.subjects)), np.arange(len(self.subjects)), np.arange(len(self.subjects)), np.arange(len(self.subjects)), np.arange(len(self.subjects)), np.arange(len(self.subjects)), np.arange(len(self.subjects))))
        region = np.concatenate((np.ones(len(self.subjects)), np.ones(len(self.subjects)), np.zeros(len(self.subjects)), np.zeros(len(self.subjects)), np.ones(len(self.subjects)), np.ones(len(self.subjects)), np.zeros(len(self.subjects)), np.zeros(len(self.subjects))))
        pupil = np.concatenate((np.zeros(len(self.subjects)), np.ones(len(self.subjects)), np.zeros(len(self.subjects)), np.ones(len(self.subjects)), np.zeros(len(self.subjects)), np.ones(len(self.subjects)), np.zeros(len(self.subjects)), np.ones(len(self.subjects))))
        interval = np.concatenate((np.ones(len(self.subjects)), np.ones(len(self.subjects)), np.ones(len(self.subjects)), np.ones(len(self.subjects)), np.zeros(len(self.subjects)), np.zeros(len(self.subjects)), np.zeros(len(self.subjects)), np.zeros(len(self.subjects))))
        
        d = rlc.OrdDict([('region', robjects.IntVector(list(region.ravel()))), ('pupil', robjects.IntVector(list(pupil.ravel()))), ('interval', robjects.IntVector(list(interval.ravel()))), ('subject', robjects.IntVector(list(subject.ravel()))), ('data', robjects.FloatVector(list(data.ravel())))])
        robjects.r.assign('dataf', robjects.DataFrame(d))
        robjects.r('attach(dataf)')
        statres = robjects.r('res = summary(aov(data ~ as.factor(region)*as.factor(pupil)*as.factor(interval) + Error(as.factor(subject)), dataf))')
        p1 = statres[-1][0][4][0]    # we will log-transform and min p values
        p2 = statres[-1][0][4][1]    # we will log-transform and min p values
        p3 = statres[-1][0][4][2]    # we will log-transform and min p values
        
        print
        print statres
    
    def sequential_effects(self, data_type):
        
        from matplotlib.pyplot import cm 
        color=cm.rainbow(np.linspace(0,1,14))
        
        self.make_pupil_BOLD_dataframe(data_type=data_type, regress_iti=True, regress_rt_signal=True)
        
        rois = ['LC_JW', 'G_and_S_cingul-Mid-Ant', 'pupil', 'mean_fullMB', 'V1']
        n_back = 3
        
        for roi in rois:
            for pred in ['yes', 'present', 'correct' ]:
                for signal_type in ['baseline', 'phasic']:
                    betas_s = []
                    for i in range(len(self.subjects)):
                        df_s = self.data_frame[self.data_frame.subject == self.subjects[i]]
                        df_s = df_s[-(np.array(df_s.correct) == -1)]
                        runs = np.unique(df_s.run)
                        betas_r = []
                        for r in runs:
                            df = df_s[np.array(df_s.run) == r]
                            if signal_type == 'baseline':
                                scalars = np.array(df[roi+'_b'])[n_back:]
                            if signal_type == 'phasic':
                                scalars = np.array(df[roi+'_d'])[n_back:]
                            type_reg_1_current = np.array(df[pred], dtype=int)[n_back:]
                            type_reg_1_past = np.vstack([np.array(df[pred], dtype=int)[n_back-n:-n] for n in range(1,n_back+1)])
                            if sum(type_reg_1_current) == type_reg_1_current.shape[0]:
                                pass
                            else:
                                # zscore:
                                scalars = (scalars - scalars.mean()) / scalars.std()
                                type_reg_1_current = (type_reg_1_current - type_reg_1_current.mean()) / type_reg_1_current.std()
                                for r in range(type_reg_1_past.shape[0]):
                                    type_reg_1_past[r,:] = (type_reg_1_past[r,:] - type_reg_1_past[r,:].mean()) / type_reg_1_past[r,:].std()
                                design_matrix = np.matrix(np.vstack((type_reg_1_current, type_reg_1_past)).T)
                                betas_r.append( np.array(((design_matrix.T * design_matrix).I * design_matrix.T) * np.matrix(scalars).T).ravel() )
                        betas_s.append(np.vstack(betas_r))
                    
                    betas_s.append(np.vstack(betas_r))
                    betas = np.vstack([betas_s[i].mean(axis=0) for i in range(len(self.subjects))])
                    subjects_repitition = betas[:,1] > 0.05
                    subjects_repulsion = betas[:,1] < -0.05
                    subjects_neutral = -subjects_repulsion*-subjects_repitition
                    
                    
                    
                    fig = plt.figure(figsize=(5,5))
                    
                    ax = fig.add_subplot(221)
                    MEANS = betas.mean(axis=0)
                    SEMS = sp.stats.sem(betas, axis=0)
                    x = np.arange(len(MEANS))
                    p = np.array([sp.stats.ttest_1samp(betas[:,i],0)[1] for i in range(betas.shape[1])])
                    # p = np.array([sp.stats.wilcoxon(betas[:,i])[1] for i in range(betas.shape[1])])
                    # p = np.array([myfuncs.permutationTest(betas[:,i], np.zeros(betas[:,i].shape))[1] for i in range(betas.shape[1])])
                    plt.axhline(0, lw=0.5, alpha=0.5)
                    # for b in range(betas.shape[0]):
                    #     ax.errorbar(x, betas[b,:], fmt='o', ms=5, color=color[b], alpha=0.3)
                    #     ax.plot(x, betas[b,:], color=color[b], lw=0.5, alpha=0.3)
                    ax.plot(x, MEANS, color='k', lw=1.5)
                    ax.errorbar(x, MEANS, yerr=SEMS, fmt='o', ms=10, color='k')
                    ax.set_xticks(x)
                    ax.set_xticklabels(-x)
                    ax.set_title(roi)
                    ax.set_ylabel('beta')
                    ax.set_xlim((-0.5, n_back+0.5))
                    # ax.set_ylim((-0.3, 0.3))
                    for j, pp in enumerate(p):
                        ax.text(x[j], ax.axis()[3] - ((ax.axis()[3]-ax.axis()[2]) / 15.0), str(round(pp,3)), color='k', size=7)
                    
                    ax = fig.add_subplot(222)
                    
                    try:
                        offset = -0.3
                        for b in np.arange(len(self.subjects))[subjects_neutral]:
                            ax.errorbar(x+offset, betas_s[b].mean(axis=0), yerr=sp.stats.sem(betas_s[b], axis=0), fmt='o', ms=5, color=color[b], elinewidth=0.5)
                            ax.plot(x+offset, betas_s[b].mean(axis=0), color=color[b], lw=0.5)
                            ax.set_xticks(x)
                            ax.set_xticklabels(-x)
                            ax.set_title('neutral')
                            ax.set_ylabel('beta')
                            ax.set_xlim((-0.5, n_back+0.5))
                            ax.set_ylim((-0.3, 0.3))
                            offset+=0.025
                            
                    except:
                        pass
                    
                    
                    ax = fig.add_subplot(223)
                    
                    try:
                        offset = -0.3
                        for b in np.arange(len(self.subjects))[subjects_repitition]:
                            ax.errorbar(x+offset, betas_s[b].mean(axis=0), yerr=sp.stats.sem(betas_s[b], axis=0), fmt='o', ms=5, color=color[b], elinewidth=0.5)
                            ax.plot(x+offset, betas_s[b].mean(axis=0), color=color[b], lw=0.5)
                            ax.set_xticks(x)
                            ax.set_xticklabels(-x)
                            ax.set_title('repitition')
                            ax.set_ylabel('beta')
                            ax.set_xlim((-0.5, n_back+0.5))
                            ax.set_ylim((-0.3, 0.3))
                            offset+=0.025
                            
                    except:
                        pass
                    
                    ax = fig.add_subplot(224)
                    
                    try:
                        offset = -0.3
                        for b in np.arange(len(self.subjects))[subjects_repulsion]:
                            ax.errorbar(x+offset, betas_s[b].mean(axis=0), yerr=sp.stats.sem(betas_s[b], axis=0), fmt='o', ms=5, color=color[b], elinewidth=0.5)
                            ax.plot(x+offset, betas_s[b].mean(axis=0), color=color[b], lw=0.5)
                            ax.set_xticks(x)
                            ax.set_xticklabels(-x)
                            ax.set_title('repulsion')
                            ax.set_ylabel('beta')
                            ax.set_xlim((-0.5, n_back+0.5))
                            ax.set_ylim((-0.3, 0.3))
                            offset+=0.025
                            
                    except:
                        pass
                        
                    
                    sns.despine(offset=10, trim=True)
                    plt.tight_layout()
                    fig.savefig(os.path.join(self.figure_folder, 'sequential_effects', 'seq_effects_{}_{}_{}.pdf'.format(signal_type, roi, pred)))
                    
                    
    def rates_across_trials(self, data_type):
        
        self.make_pupil_BOLD_dataframe(data_type=data_type)
        
        shell()
        
        rois = ['basal_forebrain', 'mean_fullMB', 'LC_standard_1', 'LC_JW', 'pupil',]
        rois = ['LC_JW',]
        
        window_len = 10
        for roi in rois:
            
            b_rt = np.zeros(len(self.subjects))
            b_acc = np.zeros(len(self.subjects))
            b_reward = np.zeros(len(self.subjects))
            b_yes = np.zeros(len(self.subjects))
            p_rt = np.zeros(len(self.subjects))
            p_acc = np.zeros(len(self.subjects))
            p_reward = np.zeros(len(self.subjects))
            p_yes = np.zeros(len(self.subjects))
            for i in range(len(self.subjects)):
                baselines = myfuncs.smooth(np.array(self.data_frame[self.data_frame.subject == self.subjects[i]][roi + '_b']),window_len)[window_len:-window_len]
                phasics = myfuncs.smooth(np.array(self.data_frame[self.data_frame.subject == self.subjects[i]][roi]),window_len)[window_len:-window_len]
                reward_rate = myfuncs.smooth(np.array(self.data_frame[self.data_frame.subject == self.subjects[i]].correct/self.data_frame[self.data_frame.subject == self.subjects[i]].rt),window_len)[window_len:-window_len]
                rt_rate = myfuncs.smooth(np.array(self.data_frame[self.data_frame.subject == self.subjects[i]].rt),window_len)[window_len:-window_len]
                acc_rate = myfuncs.smooth(np.array(self.data_frame[self.data_frame.subject == self.subjects[i]].correct),window_len)[window_len:-window_len]
                yes_rate = myfuncs.smooth(np.array(self.data_frame[self.data_frame.subject == self.subjects[i]].yes),window_len)[window_len:-window_len]
                
                # correlation seperate per subject:
                b_rt[i] = sp.stats.pearsonr(baselines, rt_rate)[0]
                b_acc[i] = sp.stats.pearsonr(baselines, acc_rate)[0]
                b_reward[i] = sp.stats.pearsonr(baselines, reward_rate)[0]
                b_yes[i] = sp.stats.pearsonr(baselines, yes_rate)[0]
                p_rt[i] = sp.stats.pearsonr(phasics, rt_rate)[0]
                p_acc[i] = sp.stats.pearsonr(phasics, acc_rate)[0]
                p_reward[i] = sp.stats.pearsonr(phasics, reward_rate)[0]
                p_yes[i] = sp.stats.pearsonr(phasics, yes_rate)[0]
                
                fig = plt.figure(figsize=(9,2))
                ax = fig.add_subplot(111)
                # ax.plot(a[window_len:-window_len], alpha=0.2, color='b')
                # ax.plot(b[window_len:-window_len], alpha=0.2, color='r')
                ax.plot(baselines, alpha=1, color='b', lw=1.5, label='baseline scalars')
                # ax.plot(myfuncs.smooth(b,window_len)[window_len:-window_len], alpha=1, color='g', lw=0.5, label='phasic interval scalars')
                ax.plot(phasics, alpha=1, color='r', lw=1.5, label='phasic scalars')
                ax.set_xlabel('trials')
                ax.set_ylabel('BOLD (% signal change)')
                plt.legend(loc=2)
                ax = ax.twinx()
                ax.plot(reward_rate, alpha=1, color='pink', ls='--', lw=1.5, label='reward rate')
                ax.plot(rt_rate, alpha=1, color='k', ls='--', lw=1.5, label='rt')
                ax.plot(acc_rate, alpha=1, color='g', ls='--', lw=1.5, label='accuracy')
                ax.set_ylabel('reward rate')
                plt.legend()
                # sns.despine(offset=10, trim=True)
                plt.tight_layout()
                fig.savefig(os.path.join(self.figure_folder, 'LC_timeseries', '{}_timeseries_{}.pdf'.format(roi, self.subjects[i])))
            
            # shell()
            
            p_b = np.array([myfuncs.permutationTest(b_yes, np.zeros(len(self.subjects)))[1], myfuncs.permutationTest(b_rt, np.zeros(len(self.subjects)))[1], myfuncs.permutationTest(b_acc, np.zeros(len(self.subjects)))[1], myfuncs.permutationTest(b_reward, np.zeros(len(self.subjects)))[1]])
            p_p = np.array([myfuncs.permutationTest(p_yes, np.zeros(len(self.subjects)))[1], myfuncs.permutationTest(p_rt, np.zeros(len(self.subjects)))[1], myfuncs.permutationTest(p_acc, np.zeros(len(self.subjects)))[1], myfuncs.permutationTest(p_reward, np.zeros(len(self.subjects)))[1]])
            # p_b = mne.stats.fdr_correction(p_b, alpha=0.05)[1]
            # p_p = mne.stats.fdr_correction(p_p, alpha=0.05)[1]
            
            
            MEANS_b = np.array([b_yes.mean(), b_rt.mean(), b_acc.mean(), b_reward.mean()])
            MEANS_p = np.array([p_yes.mean(), p_rt.mean(), p_acc.mean(), p_reward.mean()])

            SEMS_b = np.array([sp.stats.sem(b_yes), sp.stats.sem(b_rt), sp.stats.sem(b_acc), sp.stats.sem(b_reward)])
            SEMS_p = np.array([sp.stats.sem(p_yes), sp.stats.sem(p_rt), sp.stats.sem(p_acc), sp.stats.sem(p_reward)])

            # FIGURE 1
            my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}
            N = len(MEANS_b)
            ind = np.linspace(0,N,N)  # the x locations for the groups
            bar_width = 0.5   # the width of the bars
            spacing = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            fig = plt.figure(figsize=(3,2))
            ax = fig.add_subplot(111)
            ax.bar(ind, MEANS_b, width = bar_width, yerr=SEMS_b, color='b', alpha=1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'edge', error_kw={'elinewidth':0.5,})
            ax.bar(ind+0.5, MEANS_p, width = bar_width, yerr=SEMS_p, color='r', alpha=1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'edge', error_kw={'elinewidth':0.5,})
            for t in range(len(MEANS_b)):
                if round(p_b[t],3) < 0.05:
                    ax.text(ind[t], ax.axis()[3] - ((ax.axis()[3]-ax.axis()[2]) / 10.0), round(p_b[t],3))
                if round(p_p[t],3) < 0.05:
                    ax.text(ind[t]+0.5, ax.axis()[3] - ((ax.axis()[3]-ax.axis()[2]) / 15.0), round(p_p[t],3))
            ax.set_xticks(ind+bar_width)
            ax.set_xticklabels(['yes', 'rt', 'acc', 'reward rate'])
            ax.set_ylabel('correlation coefficient')
            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, 'correlation', 'reward_rates_{}.pdf'.format(roi)))
            
            
            # 
            
        
        
    
    def correlation_per_subject(self, rois, data_type):
        
        self.make_pupil_BOLD_dataframe(data_type=data_type)
        
        for roi in rois:
            fig = plt.figure(figsize=(15,9))
            for i in range(len(self.subjects)):
                varX = np.array(self.data_frame[self.data_frame.subject == self.subjects[i]][roi + '_b'])
                varY = np.array(self.data_frame[self.data_frame.subject == self.subjects[i]][roi + '_d'])
                slope, intercept, r_value, p_value, std_err = stats.linregress(varX,varY)
                (m,b) = sp.polyfit(varX, varY, 1)
                regression_line = sp.polyval([m,b], varX)
                ax = fig.add_subplot(3,5,i+1)
                ax.plot(varX, varY, 'o', color='k', marker='o', markeredgecolor='w', markeredgewidth=0.5, rasterized=True)
                ax.plot(varX,regression_line, color = 'r', linewidth = 1.5)
                ax.set_title('subj.' + str(i+1) + ' (r = ' + str(round(r_value, 3)) + ')', size = 12)
                ax.set_ylabel('phasic BOLD (% signal change)', size = 10)
                ax.set_xlabel('baseline BOLD (% signal change)', size = 10)
                plt.tick_params(axis='both', which='major', labelsize=10)
            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, 'correlation', 'correlation_per_subject_{}.pdf'.format(roi)))
            

    def single_trial_correlation_ITI(self, data_type, bins=50,):
        
        import copy
        
        time_locked = 'stim_locked'
        # rois = ['LC_JW', 'pupil']
        rois = ['mean_fullMB', 'LC_JW', 'Pole_occipital',]
        
        shell()
        
        self.make_pupil_BOLD_dataframe(data_type=data_type, time_locked=time_locked, regress_iti=False, regress_rt_signal=False)
        R2s = []
        ps = []
        for roi in rois:
            
            # for signal_type in ['baseline', 'phasic', 'baseline_phasic']:
            for signal_type in ['baseline']:
                
                if signal_type == 'baseline':
                    X = [np.array(self.data_frame[self.data_frame.subject == s]['nr_blinks']) for s in self.subjects]
                    Y = [np.array(self.data_frame[self.data_frame.subject == s][roi + '_b']) for s in self.subjects]
                    # Y = [np.array(self.data_frame[self.data_frame.subject == s][roi]) for s in self.subjects]
                
                # demean:
                for i in range(len(self.subjects)):
                    print len(X[i])
                    X[i] = X[i] - X[i].mean()
                    Y[i] = Y[i] - Y[i].mean()
                
                # append:
                r = np.zeros(len(self.subjects))
                p = np.zeros(len(self.subjects))
                for i in range(len(self.subjects)):
                    r[i] = sp.stats.pearsonr(X[i], Y[i])[0]
                    p[i] = sp.stats.pearsonr(X[i], Y[i])[1]
                    
                R2s.append(r)
                ps.append(p)
                
                
                print r
        
        MEANS = [R2s[i].mean() for i in range(len(R2s))]
        SEMS = [sp.stats.sem(R2s[i]) for i in range(len(R2s))]
        N = len(MEANS)
        ind = np.linspace(0,N,N)
        bar_width = 0.80
        fig = plt.figure(figsize=(1.25,1.75))
        ax = fig.add_subplot(111)
        for i in range(N):
            ax.bar(ind[i], height=MEANS[i], width = bar_width, yerr = SEMS[i], color = 'gray', alpha=0.75, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
        ax.set_title('N={}'.format(len(self.subjects)), size=7)
        ax.set_xticks( (ind) )
        ax.set_xticklabels( ('MB', 'LV', 'V1') )
        # ax.set_ylabel('Correlation')
        for i in range(N):
            plt.text(x=ind[i], y=ax.axis()[2] + ((ax.axis()[3]-ax.axis()[2]) / 5.0), s='p = {}'.format(round(myfuncs.permutationTest(R2s[i], np.zeros(len(self.subjects)))[1],3)), horizontalalignment='center')
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder, 'correlation', '0_correlation_ITI_{}_{}_{}_{}.pdf'.format(roi, signal_type, data_type, time_locked)))

    def single_trial_correlation(self, data_type, bins=50,):
        
        import copy
        
        # time_locked = 'resp_locked'
        time_locked = 'stim_locked'
        
        rois = ['basal_forebrain_4', 'basal_forebrain_123', 'mean_SN','mean_VTA', 'LC_JW']
        
        self.make_pupil_BOLD_dataframe(data_type=data_type, time_locked=time_locked, regress_iti=False, regress_rt=True, regress_stimulus=True)
        
        rois_project_out = ['basal_forebrain_4', 'basal_forebrain_123', 'mean_SN','mean_VTA', 'LC_JW', 'superior_colliculus', 'inferior_colliculus']
        
        for project_out in [True, False]:
            for roi in rois:
                X = [np.array(self.data_frame[self.data_frame.subject == s]['pupil' + self.postfix]) for s in self.subjects]
                Y = [np.array(self.data_frame[self.data_frame.subject == s][roi  + self.postfix]) for s in self.subjects]
                # X = [np.array(self.data_frame[self.data_frame.subject == s][roi + '_b']) for s in self.subjects]
                # Y = [np.array(self.data_frame[self.data_frame.subject == s][roi]) for s in self.subjects]
                
                # project out at single trial level:
                if project_out:
                    rois_p = copy.copy(rois_project_out)
                    rois_p.remove(roi)
                    for i in range(len(self.subjects)):
                        to_project_out = [np.array(self.data_frame[self.data_frame.subject == self.subjects[i]][r + self.postfix]) for r in rois_p]
                        Y[i] = myfuncs.lin_regress_resid(Y[i], to_project_out) + Y[i].mean()
                        X[i] = myfuncs.lin_regress_resid(X[i], to_project_out) + X[i].mean()
                    
                        # shell()
                        
                # # demean:
                # for i in range(len(self.subjects)):
                #     print len(X[i])
                #     X[i] = X[i] - X[i].mean()
                #     Y[i] = Y[i] - Y[i].mean()
                
                # shell()
                
                fig = plt.figure(figsize=(2,2))
                ax = fig.add_subplot(111)
    
                # for ss, splits in enumerate([8,10,12]):
                pupil_array = np.zeros((len(self.subjects), bins))
                BOLD_array = np.zeros((len(self.subjects), bins))
                for i in range(len(self.subjects)):
                    split_indices = np.array_split(np.argsort(X[i]), bins)
                    for b in range(bins):
                        pupil_array[i,b] = np.mean(X[i][split_indices[b]])
                        BOLD_array[i,b] = np.mean(Y[i][split_indices[b]])
                
                pupil_across = pupil_array.mean(axis=0)
                BOLD_across = BOLD_array.mean(axis=0)
                
                # plot 1:
                # -------
                ax.errorbar(pupil_across, BOLD_across, yerr=sp.stats.sem(BOLD_array, axis=0), xerr=sp.stats.sem(pupil_array, axis=0), capsize=0, ls='none', color='black', elinewidth=0.5)
                myfuncs.correlation_plot(pupil_across, BOLD_across, ax=ax)
                ax.set_xlabel('Pupil response\n(% signal change)')
                ax.set_ylabel('fMRI response\n(% signal change)')
                n_bins = 385
                # n_bins = 310
                # single trial:
                pupil_array = np.zeros((len(self.subjects), n_bins))
                BOLD_array = np.zeros((len(self.subjects), n_bins))
                for i in range(len(self.subjects)):
                    split_indices = np.array_split(np.argsort(X[i]), n_bins)
                    for b in range(n_bins):
                        pupil_array[i,b] = np.mean(X[i][split_indices[b]])
                        BOLD_array[i,b] = np.mean(Y[i][split_indices[b]])
                slope, intercept, r_value, p_value, std_err = stats.linregress(pupil_array.mean(axis=0),BOLD_array.mean(axis=0))
                (m,b) = sp.polyfit(pupil_array.mean(axis=0),BOLD_array.mean(axis=0),1)
                x_line = np.linspace(ax.axis()[0], ax.axis()[1], 100)
                regression_line = sp.polyval([m,b],x_line)
                ax.plot(x_line, regression_line, color='Black', alpha=1)
                ax.text(ax.axis()[1] - ((ax.axis()[1] - ax.axis()[0]) / 10.0), ax.axis()[2] + ((ax.axis()[3]-ax.axis()[2]) / 5.0), 'r = ' + str(round(r_value, 3)) + '\np = ' + str(p_value), size=6,)
                sns.despine(offset=10, trim=True)
                plt.tight_layout()
                if project_out:
                    fig.savefig(os.path.join(self.figure_folder, 'correlation', 'single_trial', 'correlation_{}_{}_{}_{}_{}_partial.pdf'.format(roi, self.split_by, data_type, time_locked, bins)))
                else:
                    fig.savefig(os.path.join(self.figure_folder, 'correlation', 'single_trial', 'correlation_{}_{}_{}_{}_{}.pdf'.format(roi, self.split_by, data_type, time_locked, bins)))
    
    def single_trial_correlation2(self, data_type, bins=50,):
        
        import copy
        
        # time_locked = 'resp_locked'
        time_locked = 'stim_locked'
        
        rois = ['4th_ventricle']
        
        self.make_pupil_BOLD_dataframe(data_type=data_type, time_locked=time_locked, regress_iti=False, regress_rt=True, regress_stimulus=True)
        
        rois_project_out = ['basal_forebrain_4', 'basal_forebrain_123', 'mean_SN','mean_VTA', 'LC_JW', 'superior_colliculus', 'inferior_colliculus']
        
        for project_out in [False]:
            for roi in rois:
                X = [np.array(self.data_frame[self.data_frame.subject == s]['pupil' + self.postfix]) for s in self.subjects]
                Y = [np.array(self.data_frame[self.data_frame.subject == s][roi  + self.postfix]) for s in self.subjects]
                # X = [np.array(self.data_frame[self.data_frame.subject == s][roi + '_b']) for s in self.subjects]
                # Y = [np.array(self.data_frame[self.data_frame.subject == s][roi]) for s in self.subjects]
                
                # project out at single trial level:
                if project_out:
                    rois_p = copy.copy(rois_project_out)
                    rois_p.remove(roi)
                    for i in range(len(self.subjects)):
                        to_project_out = [np.array(self.data_frame[self.data_frame.subject == self.subjects[i]][r + self.postfix]) for r in rois_p]
                        Y[i] = myfuncs.lin_regress_resid(Y[i], to_project_out) + Y[i].mean()
                        X[i] = myfuncs.lin_regress_resid(X[i], to_project_out) + X[i].mean()
                    
                        # shell()
                        
                # # demean:
                # for i in range(len(self.subjects)):
                #     print len(X[i])
                #     X[i] = X[i] - X[i].mean()
                #     Y[i] = Y[i] - Y[i].mean()
                
                # shell()
                
                fig = plt.figure(figsize=(2,2))
                ax = fig.add_subplot(111)
    
                # for ss, splits in enumerate([8,10,12]):
                pupil_array = np.zeros((len(self.subjects), bins))
                BOLD_array = np.zeros((len(self.subjects), bins))
                for i in range(len(self.subjects)):
                    split_indices = np.array_split(np.argsort(X[i]), bins)
                    for b in range(bins):
                        pupil_array[i,b] = np.mean(X[i][split_indices[b]])
                        BOLD_array[i,b] = np.mean(Y[i][split_indices[b]])
                
                pupil_across = pupil_array.mean(axis=0)
                BOLD_across = BOLD_array.mean(axis=0)
                
                # plot 1:
                # -------
                ax.errorbar(pupil_across, BOLD_across, yerr=sp.stats.sem(BOLD_array, axis=0), xerr=sp.stats.sem(pupil_array, axis=0), capsize=0, ls='none', color='black', elinewidth=0.5)
                myfuncs.correlation_plot(pupil_across, BOLD_across, ax=ax)
                ax.set_xlabel('Pupil response\n(% signal change)')
                ax.set_ylabel('fMRI response\n(% signal change)')
                n_bins = 385
                # n_bins = 310
                # single trial:
                pupil_array = np.zeros((len(self.subjects), n_bins))
                BOLD_array = np.zeros((len(self.subjects), n_bins))
                for i in range(len(self.subjects)):
                    split_indices = np.array_split(np.argsort(X[i]), n_bins)
                    for b in range(n_bins):
                        pupil_array[i,b] = np.mean(X[i][split_indices[b]])
                        BOLD_array[i,b] = np.mean(Y[i][split_indices[b]])
                slope, intercept, r_value, p_value, std_err = stats.linregress(pupil_array.mean(axis=0),BOLD_array.mean(axis=0))
                (m,b) = sp.polyfit(pupil_array.mean(axis=0),BOLD_array.mean(axis=0),1)
                x_line = np.linspace(ax.axis()[0], ax.axis()[1], 100)
                regression_line = sp.polyval([m,b],x_line)
                ax.plot(x_line, regression_line, color='Black', alpha=1)
                ax.text(ax.axis()[1] - ((ax.axis()[1] - ax.axis()[0]) / 10.0), ax.axis()[2] + ((ax.axis()[3]-ax.axis()[2]) / 5.0), 'r = ' + str(round(r_value, 3)) + '\np = ' + str(round(p_value,3)), size=6,)
                sns.despine(offset=10, trim=True)
                plt.tight_layout()
                if project_out:
                    fig.savefig(os.path.join(self.figure_folder, 'correlation', 'single_trial', 'correlation_{}_{}_{}_{}_{}_partial.pdf'.format(roi, self.split_by, data_type, time_locked, bins)))
                else:
                    fig.savefig(os.path.join(self.figure_folder, 'correlation', 'single_trial', 'correlation_{}_{}_{}_{}_{}.pdf'.format(roi, self.split_by, data_type, time_locked, bins)))    
    
    def brainstem_to_behaviour(self, data_type,):
        
        import copy
        
        time_locked = 'stim_locked'
        self.make_pupil_BOLD_dataframe(data_type=data_type, time_locked=time_locked, regress_iti=False, regress_rt_signal=True)
        
        rois_project_out = ['basal_forebrain', 'mean_fullMB', 'LC_JW']
        # rois_project_out = ['basal_forebrain', 'mean_fullMB', 'LC_standard_1']
        
        # for project_out in [True, False]:
        for project_out in [False]:
            
            for roi in rois:
                
                # for signal_type in ['baseline', 'phasic', 'baseline_phasic']:
                for signal_type in ['phasic']:
                    
                    if signal_type == 'baseline':
                        X = [np.array(self.data_frame[self.data_frame.subject == s]['pupil_b']) for s in self.subjects]
                        Y = [np.array(self.data_frame[self.data_frame.subject == s][roi + '_b']) for s in self.subjects]
                    if signal_type == 'phasic':
                        X = [np.array(self.data_frame[self.data_frame.subject == s]['pupil']) for s in self.subjects]
                        Y = [np.array(self.data_frame[self.data_frame.subject == s][roi]) for s in self.subjects]
                    
                    # project out at single trial level:
                    if project_out:
                        rois_p = copy.copy(rois_project_out)
                        rois_p.remove(roi)
                        for r in rois_p:
                            print 'projecting out: {}'.format(r)
                            for i in range(len(self.subjects)):
                                X[i] = myfuncs.lin_regress_resid(X[i], np.array(self.data_frame[self.data_frame.subject == self.subjects[i]][r]))
                                Y[i] = myfuncs.lin_regress_resid(Y[i], np.array(self.data_frame[self.data_frame.subject == self.subjects[i]][r]))
                    
                    
                    # get scalars:
                    scalars_hi = np.zeros(len(self.subjects))
                    scalars_lo = np.zeros(len(self.subjects))
                    
                    for i in range(len(self.subjects)):
                        scalars_hi[i] = np.mean(Y[i][self.pupil_h_ind[i]])
                        scalars_lo[i] = np.mean(Y[i][self.pupil_l_ind[i]])
                    
                    print sp.stats.ttest_rel(scalars_hi, scalars_lo)
                    
                    # plot:
                    fig = plt.figure(figsize=(2,2))
                    ax = fig.add_subplot(111)
                    plt.axvline(0, lw=0.5, alpha=0.5)
                    plt.axhline(0, lw=0.5, alpha=0.5)
                    myfuncs.correlation_plot(scalars_hi-scalars_lo, self.criterion_hi-self.criterion_lo, line=True, ax=ax)
                    ax.set_ylabel('delta drift criterion')
                    ax.set_xlabel('delta fMRI response')
                    sns.despine(offset=10, trim=True)
                    plt.tight_layout()
                    if project_out:
                        fig.savefig(os.path.join(self.figure_folder, 'bold_behaviour', 'roi', 'correlation_{}_{}_{}_{}_partial.pdf'.format(roi, signal_type, data_type, time_locked)))
                    else:
                        fig.savefig(os.path.join(self.figure_folder, 'bold_behaviour', 'roi', 'correlation_{}_{}_{}_{}.pdf'.format(roi, signal_type, data_type, time_locked)))
                    
                    
                    
                    
                    
                    
                    # demean:
                    for i in range(len(self.subjects)):
                        print len(X[i])
                        X[i] = X[i] - X[i].mean()
                        Y[i] = Y[i] - Y[i].mean()
    
    
    def single_trial_multiple_regression(self, data_type, bins=385,):
        
        time_locked = 'stim_locked'
        self.make_pupil_BOLD_dataframe(data_type=data_type, time_locked=time_locked, regress_iti=False, regress_rt=True, regress_stimulus=True)
        
        rois = ['basal_forebrain', 'mean_SN', 'mean_VTA', 'LC_JW']
        
        # for signal_type in ['baseline', 'phasic', 'baseline_phasic']:
        
        X_1 = [np.array(self.data_frame[self.data_frame.subject == s][rois[0] + self.postfix]) for s in self.subjects]
        X_2 = [np.array(self.data_frame[self.data_frame.subject == s][rois[1] + self.postfix]) for s in self.subjects]
        X_3 = [np.array(self.data_frame[self.data_frame.subject == s][rois[2] + self.postfix]) for s in self.subjects]
        X_4 = [np.array(self.data_frame[self.data_frame.subject == s][rois[3] + self.postfix]) for s in self.subjects]
        Y = [np.array(self.data_frame[self.data_frame.subject == s]['pupil' + self.postfix]) for s in self.subjects]
        
        # demean:
        for i in range(len(self.subjects)):
            X_1[i] = (X_1[i] - X_1[i].mean())
            X_2[i] = (X_2[i] - X_2[i].mean())
            X_3[i] = (X_3[i] - X_3[i].mean())
            X_4[i] = (X_4[i] - X_4[i].mean())
            Y[i] = (Y[i] - Y[i].mean())
        
        # bin and combine:
        Y_array = np.zeros((len(self.subjects), bins))
        X_1_array = np.zeros((len(self.subjects), bins))
        X_2_array = np.zeros((len(self.subjects), bins))
        X_3_array = np.zeros((len(self.subjects), bins))
        X_4_array = np.zeros((len(self.subjects), bins))
        for i in range(len(self.subjects)):
            split_indices = np.array_split(np.argsort(Y[i]), bins)
            for b in range(bins):
                Y_array[i,b] = np.mean(Y[i][split_indices[b]])
                X_1_array[i,b] = np.mean(X_1[i][split_indices[b]])
                X_2_array[i,b] = np.mean(X_2[i][split_indices[b]])
                X_3_array[i,b] = np.mean(X_3[i][split_indices[b]])
                X_4_array[i,b] = np.mean(X_4[i][split_indices[b]])
        Y_across = Y_array.mean(axis=0)
        X_1_across = X_1_array.mean(axis=0)
        X_2_across = X_2_array.mean(axis=0)
        X_3_across = X_3_array.mean(axis=0)
        X_4_across = X_4_array.mean(axis=0)
        
        #

        # X = [X_1_across, X_2_across, X_3_across, X_4_across]
        X = [X_4_across,]
        
        # prepare data:
        d = {'Y' : pd.Series(Y_across),}
        for i in range(len(X)):
            d['X{}'.format(i)] = pd.Series(X[i])
        data = pd.DataFrame(d)
    
        # formula:
        formula = 'Y ~ X0'
        if len(X) > 1:
            for i in range(1,len(X)):
                formula = formula + ' + X{}'.format(i)
    
        # print formula
    
        # fit:
        model = sm.ols(formula=formula, data=data)
        fitted = model.fit()
        
        print 'r = {}'.format(np.sqrt(fitted.rsquared))
        
        shell()
    
        
    def correlation_bars_single_trial(self, rois, data_type, cor_by='pupil_d', partial=False):
        
        time_locked = 'stim_locked'
        self.make_pupil_BOLD_dataframe(data_type=data_type, time_locked=time_locked, regress_rt_signal=True,)
        
        # PLOTTING:
        labels = ['SC', 'BF', 'SN/VTA', 'LC' ]
        
        for norm in [False, True]:
        
            for signal_type in ['baseline', 'phasic']:
            
                corrmat = np.zeros((len(self.subjects),len(rois)))
                p_mat = np.zeros(len(rois))
            
                for j, roi in enumerate(rois):
                    corrs = np.zeros(len(self.subjects))
                    for i in range(len(self.subjects)):
                        if signal_type == 'baseline':
                            roi_data_1 = np.array(self.data_frame[self.data_frame.subject == self.subjects[i]]['pupil_b'])
                            roi_data_2 = np.array(self.data_frame[self.data_frame.subject == self.subjects[i]][roi+'_b'])
                        if signal_type == 'phasic':
                            roi_data_1 = np.array(self.data_frame[self.data_frame.subject == self.subjects[i]]['pupil_d'])
                            roi_data_2 = np.array(self.data_frame[self.data_frame.subject == self.subjects[i]][roi])
                        corr, p_value = sp.stats.pearsonr(roi_data_1, roi_data_2)
                        corrmat[i,j] = corr
                    p_mat[j] = myfuncs.permutationTest( corrmat[:,j], np.zeros(len(self.subjects)) )[1]
            
                # fdr correction:
                p_mat = mne.stats.fdr_correction(p_mat, 0.05)[1]
            
                MEANS = corrmat.mean(axis=0)
                
                shell()
                
                if norm:
                    if signal_type == 'baseline':
                        mean_activations = np.array([np.mean([np.mean(self.data_frame[self.data_frame.subject == self.subjects[i]][roi+'_b']) for i in range(len(self.subjects))]) for roi in rois])
                        max_activation = max([np.mean([np.mean(self.data_frame[self.data_frame.subject == self.subjects[i]][roi+'_b']) for i in range(len(self.subjects))]) for roi in rois])
                    if signal_type == 'phasic':
                        # shell()
                        mean_activations = np.array([np.mean([np.mean(self.data_frame[self.data_frame.subject == self.subjects[i]][roi]) for i in range(len(self.subjects))]) for roi in rois])
                        max_activation = max([np.mean([np.mean(self.data_frame[self.data_frame.subject == self.subjects[i]][roi]) for i in range(len(self.subjects))]) for roi in rois])
                    weights = abs(mean_activations / max_activation)
                    MEANS = MEANS/weights
                    # MEANS = MEANS / max(MEANS)
            
                SEMS = sp.stats.sem(corrmat, axis=0)
                ind = np.arange(0,len(MEANS))
                bar_width = 0.90

                fig_spacing = (4,24)
                fig = plt.figure(figsize=(8.27, 5.845))
                ax1 = plt.subplot2grid(fig_spacing, (0, 0), colspan=2)
                axes = [ax1]
                plot_nr = 0
                ax = axes[plot_nr]
                my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}
                for i in range(len(MEANS)):
                    ax.bar(ind[i], MEANS[i], width = bar_width, yerr = SEMS[i], color = 'k', alpha = 0.75, align = 'center', edgecolor = 'k', linewidth = 0, capsize = 0, ecolor = 'k', error_kw={'elinewidth':0.5,})
                ax.tick_params(axis='y', which='major', labelsize=6)
                ax.set_xticks(ind)
                for i, pp in enumerate(p_mat):
                    star1 = 'n.s.'
                    if pp < 0.05:
                        star1 = '*'
                    if pp < 0.01:
                        star1 = '**'
                    if pp < 0.001:
                        star1 = '***'
                    ax.text(s=star1, x=ind[i], y=ax.get_ylim()[1]-((ax.get_ylim()[1] - ax.get_ylim()[0]) / 10.0), size=10)
                ax.set_ylabel('mean R', size=7)
                if not norm:
                    ax.set_ylim(-0.2,1.2)
                sns.despine(offset=10, trim=True)
                ax.set_xticklabels(labels, rotation=45, size=7)
                plt.tight_layout()
                if norm:
                    fig.savefig(os.path.join(self.figure_folder, 'correlation', '3_correlation_bars_{}_single_{}_norm.pdf'.format(cor_by, signal_type)))
                else:
                    fig.savefig(os.path.join(self.figure_folder, 'correlation', '3_correlation_bars_{}_single_{}.pdf'.format(cor_by, signal_type)))
    
    


    
    def correlation_bars_binned(self, rois, data_type, bin_by='pupil', partial=False, bins=50):
        
        time_locked = 'stim_locked'
        self.make_pupil_BOLD_dataframe(data_type=data_type, time_locked=time_locked, regress_iti=False, regress_rt=True, regress_stimulus=True)
            
        # PLOTTING:
        labels = ['SC', 'SN', 'VTA', 'LC', 'BF',]
        
        for norm in [False, True]:
            
            corrmat = np.zeros(len(rois))
            p_mat = np.zeros(len(rois))
    
            for j, roi in enumerate(rois):
        
                roi_data_across_1 = np.zeros((len(self.subjects),bins))
                roi_data_across_2 = np.zeros((len(self.subjects),bins))
                for i in range(len(self.subjects)):
                    bin_indices = np.array_split(np.argsort(np.array(self.data_frame[self.data_frame.subject == self.subjects[i]]['pupil'+self.postfix])), bins)
                    roi_data_1 = np.array(self.data_frame[self.data_frame.subject == self.subjects[i]]['pupil'+self.postfix])
                    roi_data_2 = np.array(self.data_frame[self.data_frame.subject == self.subjects[i]][roi+self.postfix])
                    for b in range(bins):
                        roi_data_across_1[i,b] = np.mean(roi_data_1[bin_indices[b]])
                        roi_data_across_2[i,b] = np.mean(roi_data_2[bin_indices[b]])
                roi_data_across_1 = roi_data_across_1.mean(axis=0)
                roi_data_across_2 = roi_data_across_2.mean(axis=0)
                corr, p_value = sp.stats.pearsonr(roi_data_across_1, roi_data_across_2)
                corrmat[j] = corr
                p_mat[j] = p_value
        
            # fdr correction:
            p_mat = mne.stats.fdr_correction(p_mat, 0.05)[1]
            
            MEANS = corrmat
            if norm:
                mean_activations = np.array([np.mean([np.mean(self.data_frame[self.data_frame.subject == self.subjects[i]][roi+self.postfix]) for i in range(len(self.subjects))]) for roi in rois])
                max_activation = max([np.mean([np.mean(self.data_frame[self.data_frame.subject == self.subjects[i]][roi+self.postfix]) for i in range(len(self.subjects))]) for roi in rois])
                weights = abs(mean_activations / max_activation)
                MEANS = MEANS/weights
                # MEANS = MEANS / max(MEANS)
            
            ind = np.arange(0,len(MEANS))
            bar_width = 0.90
        
            fig_spacing = (4,24)
            fig = plt.figure(figsize=(8.27, 5.845))
            ax1 = plt.subplot2grid(fig_spacing, (0, 0), colspan=2)
            axes = [ax1]
            plot_nr = 0
            ax = axes[plot_nr]
            my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}
            for i in range(len(MEANS)):
                ax.bar(ind[i], MEANS[i], width = bar_width, color = 'k', alpha = 0.75, align = 'center', edgecolor = 'k', linewidth = 0, capsize = 0, ecolor = 'k', error_kw={'elinewidth':0.5,})
            ax.tick_params(axis='y', which='major', labelsize=6)
            ax.set_xticks(ind)
            for i, pp in enumerate(p_mat):
                star1 = 'n.s.'
                if pp < 0.05:
                    star1 = '*'
                if pp < 0.01:
                    star1 = '**'
                if pp < 0.001:
                    star1 = '***'
                ax.text(s=star1, x=ind[i], y=ax.get_ylim()[1]-((ax.get_ylim()[1] - ax.get_ylim()[0]) / 10.0), size=10)
            ax.set_ylabel('mean R', size=7)
            if not norm:
                ax.set_ylim(-0.2,1.2)
            sns.despine(offset=10, trim=True)
            ax.set_xticklabels(labels, rotation=45, size=7)
            plt.tight_layout()
            if norm:
                fig.savefig(os.path.join(self.figure_folder, 'correlation', 'bars', 'bars_{}_binned_{}_{}_norm.pdf'.format(bin_by, self.split_by, bins)))
            else:
                fig.savefig(os.path.join(self.figure_folder, 'correlation', 'bars', 'bars_{}_binned_{}_{}.pdf'.format(bin_by, self.split_by, bins)))
    

    def surface_labels_to_vol(self):
        
        session = [self.runList[i] for i in self.conditionDict['task']][0].session
        if session == 1:
        
            # surface to vol:
            labelFiles = subprocess.Popen('ls ' + os.path.join('/home/shared/Niels_UvA/surface_masks/', '*.label'), shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
            for lf in labelFiles:
                lfx = os.path.split(lf)[-1]
                if 'lh' in lfx:
                    hemi = 'lh'
                elif 'rh' in lfx:
                    hemi = 'rh'
                lvo = LabelToVolOperator(lf)
                template_file = os.path.join(self.project.base_dir, self.subject.initials, 'files', 'detection_data_clean_MNI_{}_1.nii.gz'.format(self.subject.initials))
            
                lvo.configure(templateFileName = template_file, hemispheres = [hemi], register = os.path.join(os.environ["FREESURFER_HOME"], "average/mni152.register.dat"), fsSubject = 'fsaverage', outputFileName = lf[:-6]+'.nii.gz', threshold = 0.05, surfType = 'label')
                lvo.execute()
        
            # # combined parietal region:
            # regions = ['ips_anterior', 'ips_posterior', 'postcentral_sulcus']
            # b = [np.array(nib.load(os.path.join('/home/shared/Niels_UvA/surface_masks', 'rh.{}.nii.gz'.format(region))).get_data(), dtype=bool) for region in regions]
            # b = b[0]+b[1]+b[2]
            # b = nib.Nifti1Image(b, affine=nib.load(os.path.join('/home/shared/Niels_UvA/surface_masks', 'rh.{}.nii.gz'.format(region))).get_affine(), header=nib.load(os.path.join('/home/shared/Niels_UvA/surface_masks', 'rh.{}.nii.gz'.format(region))).get_header())
            # nib.save(b, os.path.join('/home/shared/Niels_UvA/surface_masks', 'rh.parietal.nii.gz'))
        
            # # copy to left hemisphere:
            # volumeFiles = subprocess.Popen('ls ' + os.path.join('/home/shared/Niels_UvA/surface_masks/', 'rh.*.nii.gz'), shell=True, stdout=PIPE).communicate()[0].split('\n')[0:-1]
            # for nifti in volumeFiles:
            #     b = nib.load(nifti).get_data()
            #     b = b[::-1,:,:]
            #     b = nib.Nifti1Image(b, affine=nib.load(nifti).get_affine(), header=nib.load(nifti).get_header())
            #     output_object = nifti.replace('rh.', 'lh.')
            #     nib.save(b, output_object)

    
    def correlation_bars_binned_all_rois(self, data_type, bin_by='pupil', partial=False, bins=50):
        
        time_locked = 'stim_locked'
        self.make_pupil_BOLD_dataframe(data_type=data_type, time_locked=time_locked)
        
        # PLOTTING:
        rois = [
                # visual:
                'Pole_occipital',
                'G_occipital_sup',
                'S_oc_sup_and_transversal',
                'G_occipital_middle',
                'S_occipital_ant',
                'S_oc_middle_and_Lunatus',
                'G_and_S_occipital_inf',
                'S_collat_transv_post',
                'G_oc-temp_med-Lingual',
                'S_calcarine',
                'G_cuneus',
    
                # temporal:
                'Lat_Fis-post',
                'G_temp_sup-Plan_tempo',
                'S_temporal_transverse',
                'G_temp_sup-G_T_transv',
                'G_temp_sup-Lateral',
                'S_temporal_sup',
                'G_temporal_middle',
                'S_temporal_inf',
                'G_temporal_inf',
                'S_oc-temp_lat',
                'G_oc-temp_med-Parahip',
                'S_collat_transv_ant',
                'G_oc-temp_lat-fusifor',
                'S_oc-temp_med_and_Lingual',
                'G_temp_sup-Plan_polar',
                'Pole_temporal',
    
                # parietal:
                'S_parieto_occipital',
                'S_subparietal',
                'G_precuneus',
                'G_parietal_sup',
                'S_intrapariet_and_P_trans',
                'G_pariet_inf-Angular',
                'S_interm_prim-Jensen',
                'G_and_S_paracentral',
                'S_postcentral',
                'G_postcentral',
                'S_central',
                'G_pariet_inf-Supramar',
                'G_and_S_subcentral',
    
                # insular:
                'S_circular_insula_sup',
                'G_insular_short',
                'S_circular_insula_inf',
                'G_Ins_lg_and_S_cent_ins',
                'S_circular_insula_ant',
                
                # cingulate:
                'G_cingul-Post-ventral',
                'S_pericallosal',
                'G_cingul-Post-dorsal',
                'S_cingul-Marginalis',
                'G_and_S_cingul-Mid-Post',
                'G_and_S_cingul-Mid-Ant',
                'G_and_S_cingul-Ant',
                
                # frontal:
                'G_precentral',
                'S_precentral-sup-part',
                'S_precentral-inf-part',
                'G_front_sup',
                'S_front_sup',
                'G_front_middle',
                'S_front_middle',
                'S_front_inf',
                'G_front_inf-Opercular',
                'G_front_inf-Triangul',
                'S_orbital_lateral',
                'Lat_Fis-ant-Horizont',
                'Lat_Fis-ant-Vertical',
                'G_front_inf-Orbital',
                'G_and_S_transv_frontopol',
                'G_and_S_frontomargin',
                'G_orbital',
                'S_orbital-H_Shaped',
                'S_orbital_med-olfact',
                'G_rectus',
                'S_suborbital',
                'G_subcallosal',
                ]
                
        for norm in [False, True]:
        # for norm in [False,]:
            
            for signal_type in ['baseline', 'phasic']:
            
                corrmat = np.zeros(len(rois))
                p_mat = np.zeros(len(rois))
        
                for j, roi in enumerate(rois):
            
                    roi_data_across_1 = np.zeros((len(self.subjects),bins))
                    roi_data_across_2 = np.zeros((len(self.subjects),bins))
                    for i in range(len(self.subjects)):
                        if signal_type == 'baseline':
                            bin_indices = np.array_split(np.argsort(np.array(self.data_frame[self.data_frame.subject == self.subjects[i]][bin_by+'_b'])), bins)
                            roi_data_1 = np.array(self.data_frame[self.data_frame.subject == self.subjects[i]][bin_by+'_b'])
                            roi_data_2 = np.array(self.data_frame[self.data_frame.subject == self.subjects[i]][roi+'_b'])
                        if signal_type == 'phasic':
                            bin_indices = np.array_split(np.argsort(np.array(self.data_frame[self.data_frame.subject == self.subjects[i]][bin_by])), bins)
                            roi_data_1 = np.array(self.data_frame[self.data_frame.subject == self.subjects[i]][bin_by])
                            roi_data_2 = np.array(self.data_frame[self.data_frame.subject == self.subjects[i]][roi])
                        for b in range(bins):
                            roi_data_across_1[i,b] = np.mean(roi_data_1[bin_indices[b]])
                            roi_data_across_2[i,b] = np.mean(roi_data_2[bin_indices[b]])
                    roi_data_across_1 = roi_data_across_1.mean(axis=0)
                    roi_data_across_2 = roi_data_across_2.mean(axis=0)
                    corr, p_value = sp.stats.pearsonr(roi_data_across_1, roi_data_across_2)
                    corrmat[j] = corr
                    p_mat[j] = p_value
            
                # fdr correction:
                p_mat = mne.stats.fdr_correction(p_mat, 0.05)[1]
                
                MEANS = corrmat
                if norm:
                    if signal_type == 'baseline':
                        mean_activations = np.array([np.mean([np.mean(self.data_frame[self.data_frame.subject == self.subjects[i]][roi+'_b']) for i in range(len(self.subjects))]) for roi in rois])
                        max_activation = max([np.mean([np.mean(self.data_frame[self.data_frame.subject == self.subjects[i]][roi+'_b']) for i in range(len(self.subjects))]) for roi in rois])
                    if signal_type == 'phasic':
                        # shell()
                        mean_activations = np.array([np.mean([np.mean(self.data_frame[self.data_frame.subject == self.subjects[i]][roi]) for i in range(len(self.subjects))]) for roi in rois])
                        max_activation = max([np.mean([np.mean(self.data_frame[self.data_frame.subject == self.subjects[i]][roi]) for i in range(len(self.subjects))]) for roi in rois])
                    weights = abs(mean_activations / max_activation)
                    MEANS = MEANS/weights
                    # MEANS = MEANS / max(MEANS)
                    
                ind = np.arange(0,len(MEANS))
                bar_width = 0.90
                
                fig_spacing = (3,1)
                fig = plt.figure(figsize=(8.3,11.7))
                ax1 = plt.subplot2grid(fig_spacing, (0, 0), colspan=2)
                axes = [ax1]
                plot_nr = 0
                ax = axes[plot_nr]
                my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}
                for i in range(len(MEANS)):
                    ax.bar(ind[i], MEANS[i], width = bar_width, color = 'k', alpha = 0.75, align = 'center', edgecolor = 'k', linewidth = 0, capsize = 0, ecolor = 'k', error_kw={'elinewidth':0.5,})
                ax.tick_params(axis='y', which='major', labelsize=6)
                ax.set_xticks(ind)
                ax.set_xlim(ind[0]-1,ind[-1]+1)
                if not norm:
                    ax.set_ylim(-0.2,0.75)
                for i, pp in enumerate(p_mat):
                    star1 = ''
                    if pp < 0.05:
                        star1 = '*'
                    if pp < 0.01:
                        star1 = '**'
                    if pp < 0.001:
                        star1 = '***'
                    ax.text(s=star1, x=ind[i], y=ax.get_ylim()[1]-((ax.get_ylim()[1] - ax.get_ylim()[0]) / 10.0), size=10, rotation=90)
                    # ax.text(s=star1, x=ind[i], y=0.025, size=10, color='white', rotation=90)
                ax.set_ylabel('mean R', size=7)
                sns.despine(offset=10, trim=True)
                ax.set_xticklabels(rois, rotation=90, size=7)
                plt.tight_layout()
                if norm:
                    fig.savefig(os.path.join(self.figure_folder, 'correlation', '6_correlation_bars_all_rois_{}_binned_{}_{}_norm.pdf'.format(bin_by, signal_type, bins)))
                else:
                    fig.savefig(os.path.join(self.figure_folder, 'correlation', '6_correlation_bars_all_rois_{}_binned_{}_{}.pdf'.format(bin_by, signal_type, bins)))
    
    
    
    def BRAINSTEM_bar_plots(self, data_type):
        
        shell()
        
        self.make_pupil_BOLD_dataframe(data_type=data_type, regress_iti=False, regress_rt=True,)
        
        # add:
        pupil = np.array(np.concatenate(self.pupil_h_ind), dtype=int)
        pupil[~(np.concatenate(self.pupil_l_ind)+np.concatenate(self.pupil_h_ind))] = 2
        self.data_frame['pupil'] = pupil
        
        titles =         [
                        'colliculi',
                        # 'colliculi_100',
                        'brainstem',
                        # 'brainstem_100',
                        'aan'
                        ]
        rois_groups =     [
                        ['sup_col_jw_d', 'inf_col_jw_d'],
                        # ['superior_colliculus_100_d', 'inferior_colliculus_100_d'],
                        ['LC_JW_d', 'mean_VTA_d', 'mean_SN_d', 'basal_forebrain_123_d', 'basal_forebrain_4_d'],
                        # ['mean_VTA_100_d', 'mean_SN_100_d', 'basal_forebrain_123_100_d', 'basal_forebrain_4_100_d'],
                        ['AAN_LC_d', 'AAN_VTA_d', 'AAN_DR_d', 'AAN_PAG_d', 'AAN_PBC_d', 'AAN_PO_d', 'AAN_PPN_d', 'AAN_MR_d', 'AAN_MRF_d',]
                        ]
        rois_names =     [
                        ['SC', 'IC'],
                        # ['SC', 'IC'],
                        ['LC', 'VTA', 'SN', 'BF-sept', 'BF-subl'],
                        # ['VTA', 'SN', 'BF-sept', 'BF-subl'],
                        ['LC', 'VTA', 'DR', 'PAG', 'PBC', 'PO', 'PPN',  'MR', 'MRF',]
                        ]
        ylabels =         [
                        ['fMRI response\n(%signal change)'],
                        # ['fMRI response\n(%signal change)'],
                        ['fMRI response\n(%signal change)'],
                        # ['fMRI response\n(%signal change)'],
                        ['fMRI response\n(%signal change)'],
                        ]
        
        for title, rois, rois_name, ylabel in zip(titles, rois_groups, rois_names, ylabels):        
            for inds, trial_type, xlabels, colors in zip([[self.pupil_h_ind, self.pupil_l_ind], [self.yes, self.no], [self.present, self.absent],], ['pupil', 'yes', 'present',], [['high', 'low'], ['yes', 'no'], ['present', 'absent',]],  [['r','b'], ['m','c',], ['orange','forestgreen']]):
                
                # get data in place:
                df = self.data_frame.ix[:,np.concatenate((np.array(rois), np.array(['subject', 'pupil', 'yes', 'present']))).ravel()]
                if trial_type == 'pupil':
                    df = df[df[trial_type] != 2]
                
                # get data in place:
                df = df.ix[:,np.concatenate((np.array(rois), np.array(['subject', trial_type])))]
                k = df.groupby(['subject', trial_type]).mean()
                k_s = k.stack().reset_index()
                k_s.columns = ['subject', trial_type, 'area', 'bold']
                
                # plot:
                locs = np.arange(0,len(rois))
                bar_width = 0.2
                fig = plt.figure(figsize=( (1+(len(rois)*0.3)),1.75))
                ax = fig.add_subplot(111)
                sns.barplot(x='area',  y='bold', units='subject', hue=trial_type, hue_order=[1,0], data=k_s, palette=colors, linewidth=0, ax=ax, alpha=0.5, ci=None)
                
                # add paired observations:
                sns.stripplot(x='area', y='bold', hue=trial_type, hue_order=[1,0], data=k_s, split=True, jitter=False, size=2, palette=colors, edgecolor='black', alpha=1, linewidth=0.25, ax=ax)
                for r in range(len(rois)):
                    values = np.vstack((k_s[(k_s['area'] == rois[r]) & (k_s[trial_type] == 1)].bold, k_s[(k_s['area'] == rois[r]) & (k_s[trial_type] == 0)].bold))
                    x = np.array([locs[r]-bar_width, locs[r]+bar_width])
                    ax.plot(x, values, color='black', lw=0.5, alpha=0.5)
                
                # add p-values:
                for r in range(len(rois)):
                    p1 = myfuncs.permutationTest(k_s[(k_s[trial_type]==0) & (k_s['area']==rois[r])].bold, k_s[(k_s[trial_type]==1) & (k_s['area']==rois[r])].bold, paired=True)[1]
                    # p2 = myfuncs.permutationTest(k_s[(k_s[trial_type]==0) & (k_s['area']==rois[r])].bold, np.zeros(len(self.subjects)), paired=True)[1]
                    # p3 = myfuncs.permutationTest(k_s[(k_s[trial_type]==1) & (k_s['area']==rois[r])].bold, np.zeros(len(self.subjects)), paired=True)[1]
                    
                    # if title == 'brainstem':
                    #     if trial_type == 'pupil':
                    #         shell()
                    
                    # p1 = sp.stats.ttest_rel(k_s[(k_s[trial_type]==0) & (k_s['area']==rois[r])].bold, k_s[(k_s[trial_type]==1) & (k_s['area']==rois[r])].bold)[1]
                    # p2 = sp.stats.ttest_rel(k_s[(k_s[trial_type]==0) & (k_s['area']==rois[r])].bold, np.zeros(len(self.subjects)))[1]
                    # p3 = sp.stats.ttest_rel(k_s[(k_s[trial_type]==1) & (k_s['area']==rois[r])].bold, np.zeros(len(self.subjects)))[1]
                    # if p1 < 0.05:
                    plt.text(s='{}'.format(round(p1, 3)), x=locs[r], y=gca().get_ylim()[1]-((gca().get_ylim()[1] - gca().get_ylim()[0]) / 10.0), size=5, horizontalalignment='center',)
                    # if p2 < 0.05:
                    #     plt.text(s='{}'.format(round(p2, 3)), x=locs[r]-(bar_width/2.0), y=gca().get_ylim()[0]+((gca().get_ylim()[1] - gca().get_ylim()[0]) / 8.0), size=5, horizontalalignment='center', rotation=90)
                    # if p3 < 0.05:
                    #     plt.text(s='{}'.format(round(p3, 3)), x=locs[r]+(bar_width/2.0), y=gca().get_ylim()[0]+((gca().get_ylim()[1] - gca().get_ylim()[0]) / 8.0), size=5, horizontalalignment='center', rotation=90)
                ax.legend_.remove()
                plt.xticks(locs, rois_name, rotation=45)
                sns.despine(offset=10, trim=True)
                plt.tight_layout()
                fig.savefig(os.path.join(self.figure_folder, 'event_related_average', data_type, 'mean', 'bars_{}_{}.pdf'.format(title, trial_type)))
    
        dfs = [self.data_frame[self.data_frame.subject == s] for s in self.subjects]

        ylims = [(-0.5, 0.5), (-0.1, 0.1), (-0.8, 0.8), (-0.8, 0.8), (-0.1, 0.1), (-0.1, 0.1)]
        for j, measure in enumerate(['mean_VTA_d']):

            # # Spearman:
            # measure_hit = [dfs[i][measure][self.hit[i]] for i in range(len(self.subjects))]
            # measure_fa = [dfs[i][measure][self.fa[i]] for i in range(len(self.subjects))]
            # measure_miss = [dfs[i][measure][self.miss[i]] for i in range(len(self.subjects))]
            # measure_cr = [dfs[i][measure][self.cr[i]] for i in range(len(self.subjects))]
            # r = np.zeros(len(self.subjects))
            # for i in range(len(self.subjects)):
            #     x = np.concatenate(( np.ones(len(measure_hit[i])), np.ones(len(measure_fa[i]))*2, np.ones(len(measure_miss[i]))*3, np.ones(len(measure_cr[i]))*4 ))
            #     y = np.concatenate(( np.array(measure_hit[i]), np.array(measure_fa[i]), np.array(measure_miss[i]), np.array(measure_cr[i])))
            #     r[i] = sp.stats.spearmanr(x, y)[0]
            # p = myfuncs.permutationTest(r, np.zeros(len(r)), paired=True)[1]

            # SDT bars:
            measure_hit = np.array([np.mean(dfs[i][measure][self.hit[i]]) for i in range(len(self.subjects))])
            measure_fa = np.array([np.mean(dfs[i][measure][self.fa[i]]) for i in range(len(self.subjects))])
            measure_miss = np.array([np.mean(dfs[i][measure][self.miss[i]]) for i in range(len(self.subjects))])
            measure_cr = np.array([np.mean(dfs[i][measure][self.cr[i]]) for i in range(len(self.subjects))])
            measure_yes = np.array([np.mean(dfs[i][measure][self.yes[i]]) for i in range(len(self.subjects))])
            measure_no = np.array([np.mean(dfs[i][measure][self.no[i]]) for i in range(len(self.subjects))])
            measure_correct = np.array([np.mean(dfs[i][measure][self.correct[i]]) for i in range(len(self.subjects))])
            measure_error = np.array([np.mean(dfs[i][measure][self.error[i]]) for i in range(len(self.subjects))])

            for measures, titles, labels, colors, alphas in zip([[measure_hit, measure_fa, measure_miss, measure_cr], [measure_yes, measure_no, measure_correct, measure_error]],
                                                        ['0', '1'],
                                                        [['H', 'FA', 'M', 'CR'], ['Yes', 'No', 'Correct', 'Error']],
                                                        [['m', 'm', 'c', 'c'], ['m', 'c', 'k', 'k']],
                                                        [[1, 0.5, 0.5, 1], [1, 1, 1, 0.5]]):

                my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}
                N = 4
                ind = np.linspace(0,N,N)  # the x locations for the groups
                bar_width = 0.9   # the width of the bars
                spacing = [0, 0, 0, 0]
                p1 = myfuncs.permutationTest(measures[0], measures[1], paired=True)[1]
                p2 = myfuncs.permutationTest(measures[1], measures[2], paired=True)[1]
                p3 = myfuncs.permutationTest(measures[2], measures[3], paired=True)[1]
                # p1 = sp.stats.ttest_rel(measures[0], measures[1],)[1]
                # p2 = sp.stats.ttest_rel(measures[1], measures[2],)[1]
                # p3 = sp.stats.ttest_rel(measures[2], measures[3],)[1]
                MEANS = np.array([np.mean(m) for m in measures])
                SEMS = np.array([sp.stats.sem(m) for m in measures])
                fig = plt.figure(figsize=(2,2))
                ax = fig.add_subplot(111)
                for i in range(N):
                    ax.bar(ind[i]+spacing[i], MEANS[i], width = bar_width, yerr=SEMS[i], color=colors[i], alpha=alphas[i], edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
                ax.set_xticks( ind )
                ax.set_xticklabels( labels )
                ax.set_ylabel(measure)
                # ax.set_ylim(ylims[j])
                ax.text(s=str(round(p1,3)), x=(ind[0]+ind[1])/2.0, y=ax.get_ylim()[1]-((ax.get_ylim()[1] - ax.get_ylim()[0]) / 10.0), size=5, horizontalalignment='center')
                ax.text(s=str(round(p2,3)), x=(ind[1]+ind[2])/2.0, y=ax.get_ylim()[1]-((ax.get_ylim()[1] - ax.get_ylim()[0]) / 10.0), size=5, horizontalalignment='center')
                ax.text(s=str(round(p3,3)), x=(ind[2]+ind[3])/2.0, y=ax.get_ylim()[1]-((ax.get_ylim()[1] - ax.get_ylim()[0]) / 10.0), size=5, horizontalalignment='center')
                # ax.set_title('r={}; p={}'.format(round(r.mean(),3), round(p,3)))
                sns.despine(offset=10, trim=True,)
                plt.tight_layout()
                fig.savefig(os.path.join(self.figure_folder, 'event_related_average', data_type, 'mean', 'bars_{}_{}.pdf'.format(measure, titles)))
    
    
    
        
    
    def BRAINSTEM_correlation_matrix_single_trial(self, rois, data_type, partial=False):
        
        time_locked = 'stim_locked'
        self.make_pupil_BOLD_dataframe(data_type=data_type, time_locked=time_locked, regress_iti=False, regress_rt=True, regress_stimulus=True)
        
        corrmats = np.zeros((len(rois), len(rois), len(self.subjects)))
        for i in range(len(self.subjects)):
            C = np.vstack([np.array(self.data_frame[self.data_frame.subject == self.subjects[i]][roi + self.postfix]) for roi in rois]).T
            if partial:
                corrmats[:,:,i], dummy = myfuncs.corr_matrix_partial(C)
            else:
                corrmats[:,:,i], dummy = myfuncs.corr_matrix(C)
        corrmat = corrmats.mean(axis=-1)
        p_mat = np.zeros(corrmat.shape)
        for i in range(p_mat.shape[0]):
            for j in range(p_mat.shape[1]):
                p_mat[i,j] = myfuncs.permutationTest( corrmats[i,j,:], np.zeros(corrmats[i,j,:].shape[0]), paired=True )[1]
        
        # fdr correction:
        p_mat = mne.stats.fdr_correction(p_mat, 0.05)[1]
        
        # plotting:
        mask =  np.tri(corrmat.shape[0], k=0) #+ (p_mat>0.05)
        corrmat_m = np.ma.masked_where(mask, corrmat)
        p_mat_m = np.ma.masked_where(mask.T, p_mat)
        roi_names = ['SC', 'IC', 'SN', 'VTA', 'LC', 'BF-sept', 'BF-subl']
        if partial:
            roi_names = ['BF', 'SN/VTA', 'LC' ]
        fig = plt.figure(figsize=(2.2,2))
        ax = fig.add_subplot(111)
        im = ax.pcolormesh(corrmat_m, cmap='bwr', vmin=-0.2, vmax=0.2)
        # if partial:
        #     im = ax.pcolormesh(corrmat_m, cmap='bwr', vmin=-0.5, vmax=0.5)
        # else:
        #     im = ax.pcolormesh(corrmat_m, cmap='gray', vmin=0, vmax=0.5)
        # ax.patch.set_hatch('x')
        ax.set_xlim(xmax=len(rois))
        ax.set_ylim(ymax=len(rois))
        ax.set_yticks(arange(0.5,len(rois)+.5))
        ax.set_xticks(arange(0.5,len(rois)+.5))
        ax.set_yticklabels(roi_names)
        ax.set_xticklabels(roi_names, rotation=270)
        ax.patch.set_hatch('x')
        for i in range(p_mat_m.shape[0]):
            for j in range(p_mat_m.shape[1]):
                if p_mat_m[i,j] or (p_mat_m[i,j] == 0):
                    star1 = ''
                    if p_mat_m[i,j] < 0.05:
                        star1 = '*'
                    if p_mat_m[i,j] < 0.01:
                        star1 = '**'
                    if p_mat_m[i,j] < 0.001:
                        star1 = '***'
                    ax.text(i+0.5,j+0.5,star1,size=8,horizontalalignment='center',verticalalignment='center',)
        fig.colorbar(im)
        plt.tight_layout()
        if partial:
            fig.savefig(os.path.join(self.figure_folder, 'correlation', 'matrix', 'correlation_matrix_single_partial_{}.pdf'.format(self.split_by)))
        else:
            fig.savefig(os.path.join(self.figure_folder, 'correlation', 'matrix', 'correlation_matrix_single_{}.pdf'.format(self.split_by)))
    
    def BRAINSTEM_correlation_bars(self):
        
        # load dataframe:
        self.make_pupil_BOLD_dataframe(data_type='clean_4th_ventricle', time_locked='stim_locked', regress_iti=False, regress_rt=True, regress_stimulus=True)
        data_brainstem = self.data_frame.copy()
        self.make_pupil_BOLD_dataframe(data_type='clean_False', time_locked='stim_locked', regress_iti=False, regress_rt=True, regress_stimulus=True)
        data_cortex = self.data_frame.copy()
        data_frame = pd.read_csv(os.path.join(self.data_folder, 'event_related_average_wb', 'MNI_dataframe_1.csv'))
        dfs_brainstem = [data_brainstem[data_brainstem.subject == s] for s in self.subjects]
        dfs_cortex = [data_cortex[data_cortex.subject == s] for s in self.subjects]
        dfs = [data_frame[data_frame.subject == s] for s in self.subjects]
        
        # add linear combinations:
        for i in range(len(self.subjects)):
            d = {'Y' : pd.Series(self.pupil_data[i]['pupil_d']),
                'X1' : pd.Series(dfs_brainstem[i]['LC_JW_d']),
                'X2' : pd.Series(dfs_cortex[i]['G_and_S_cingul-Mid-Post_d']),}
            data = pd.DataFrame(d)
            formula = 'Y ~ X1 + X2'
            model = sm.ols(formula=formula, data=data)
            fitted = model.fit()
            values = np.array(fitted.fittedvalues)
            dfs_brainstem[i]['LC_ACC'] = values
            d = {'Y' : pd.Series(self.pupil_data[i]['pupil_d']),
                'X1' : pd.Series(dfs_brainstem[i]['LC_JW_d']),
                'X2' : pd.Series(dfs_brainstem[i]['mean_SN_d']),
                'X3' : pd.Series(dfs_brainstem[i]['mean_VTA_d']),
                'X4' : pd.Series(dfs_brainstem[i]['basal_forebrain_123_d']),
                'X5' : pd.Series(dfs_brainstem[i]['basal_forebrain_4_d']),
                'X6' : pd.Series(dfs_brainstem[i]['inf_col_jw_d']),
                'X7' : pd.Series(dfs_brainstem[i]['sup_col_jw_d']),
                }
            data = pd.DataFrame(d)
            formula = 'Y ~ X1 + X2 + X3 + X4 + X5'
            model = sm.ols(formula=formula, data=data)
            fitted = model.fit()
            values = np.array(fitted.fitted_values)
            values = (values - values.mean()) / values.std()
            dfs_brainstem[i]['BS_nMod'] = values
            formula = 'Y ~ X6 + X7'
            model = sm.ols(formula=formula, data=data)
            fitted = model.fit()
            values = np.array(fitted.fitted_values)
            values = (values - values.mean()) / values.std()
            dfs_brainstem[i]['BS_C'] = values
            
        # correlations:    
        r1 = np.zeros(len(self.subjects))
        r2 = np.zeros(len(self.subjects))
        r3 = np.zeros(len(self.subjects))
        r4 = np.zeros(len(self.subjects))
        for i in range(len(self.subjects)):
            r1[i] = sp.stats.pearsonr(dfs_cortex[i]['G_and_S_cingul-Mid-Ant_d'], self.pupil_data[i]['pupil_d'])[0]
            r2[i] = sp.stats.pearsonr(dfs_cortex[i]['G_and_S_cingul-Mid-Ant_d'], dfs_brainstem[i]['LC_JW_d'])[0]
            r3[i] = sp.stats.pearsonr(dfs_cortex[i]['G_and_S_cingul-Mid-Ant_d'], dfs_brainstem[i]['BS_nMod'])[0]
            r4[i] = sp.stats.pearsonr(dfs_cortex[i]['G_and_S_cingul-Mid-Ant_d'], dfs_brainstem[i]['BS_C'])[0]
        rois = ['TPR', 'LC', 'BS_nMod', 'BS_C']
        rois_name = rois
        
        df = pd.DataFrame(np.vstack((r1, r2, r3, r4)).T, columns=rois)
        df['subject'] = np.arange(len(self.subjects))
        colors = ['grey']
        df = df.ix[:,np.concatenate((np.array(rois), np.array(['subject'])))]
        k = df.groupby(['subject']).mean()
        k_s = k.stack().reset_index()
        k_s.columns = ['subject', 'area', 'bold']
        
        # plot:
        locs = np.arange(0,len(rois))
        bar_width = 0.2
        fig = plt.figure(figsize=(2,2))
        ax = fig.add_subplot(111)
        sns.barplot(x='area',  y='bold', units='subject', data=k_s, palette=colors, ci=None, linewidth=0, alpha=0.5, ax=ax)
        sns.stripplot(x="area", y="bold", data=k_s, jitter=True, size=2, palette=colors, edgecolor='black', linewidth=0.25, ax=ax, split=True, alpha=1)
        # add p-values:
        for r in range(len(rois)):
            p1 = myfuncs.permutationTest(k_s[(k_s['area']==rois[r])].bold, np.zeros(len(self.subjects)), paired=True)[1]
            # if p1 < 0.05:
            plt.text(s='{}'.format(round(p1, 3)), x=locs[r], y=plt.gca().get_ylim()[1]-((plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]) / 10.0), size=5, horizontalalignment='center',)
        # ax.legend_.remove()
        plt.xticks(locs, rois_name, rotation=45)
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder, 'correlation', 'bars', 'bars_2.pdf'))
        
        
        # PARTIAL:
        
        rois_project_out = ['sup_col_jw_d', 'inf_col_jw_d', 'LC_JW_d', 'mean_VTA_d', 'mean_SN_d', 'basal_forebrain_123_d', 'basal_forebrain_4_d']
        import copy
        r1 = np.zeros(len(self.subjects))
        r2 = np.zeros(len(self.subjects))
        r3 = np.zeros(len(self.subjects))
        r4 = np.zeros(len(self.subjects))
        r5 = np.zeros(len(self.subjects))
        r6 = np.zeros(len(self.subjects))
        r7 = np.zeros(len(self.subjects))
        for i in range(len(self.subjects)):
            rois_p = copy.copy(rois_project_out)
            rois_p.remove('sup_col_jw_d')
            to_project_out = [np.array(dfs_brainstem[i][r]) for r in rois_p]
            r1[i] = sp.stats.pearsonr( myfuncs.lin_regress_resid(np.array(dfs_brainstem[i]['sup_col_jw_d']), to_project_out), myfuncs.lin_regress_resid(np.array(self.pupil_data[i]['pupil_d']), to_project_out))[0]
            rois_p = copy.copy(rois_project_out)
            rois_p.remove('inf_col_jw_d')
            to_project_out = [np.array(dfs_brainstem[i][r]) for r in rois_p]
            r2[i] = sp.stats.pearsonr(myfuncs.lin_regress_resid(np.array(dfs_brainstem[i]['inf_col_jw_d']), to_project_out), myfuncs.lin_regress_resid(np.array(self.pupil_data[i]['pupil_d']), to_project_out))[0]
            rois_p = copy.copy(rois_project_out)
            rois_p.remove('LC_JW_d')
            to_project_out = [np.array(dfs_brainstem[i][r]) for r in rois_p]
            r3[i] = sp.stats.pearsonr(myfuncs.lin_regress_resid(np.array(dfs_brainstem[i]['LC_JW_d']), to_project_out), myfuncs.lin_regress_resid(np.array(self.pupil_data[i]['pupil_d']), to_project_out))[0]
            rois_p = copy.copy(rois_project_out)
            rois_p.remove('mean_VTA_d')
            to_project_out = [np.array(dfs_brainstem[i][r]) for r in rois_p]
            r4[i] = sp.stats.pearsonr(myfuncs.lin_regress_resid(np.array(dfs_brainstem[i]['mean_VTA_d']), to_project_out), myfuncs.lin_regress_resid(np.array(self.pupil_data[i]['pupil_d']), to_project_out))[0]
            rois_p = copy.copy(rois_project_out)
            rois_p.remove('mean_SN_d')
            to_project_out = [np.array(dfs_brainstem[i][r]) for r in rois_p]
            r5[i] = sp.stats.pearsonr(myfuncs.lin_regress_resid(np.array(dfs_brainstem[i]['mean_SN_d']), to_project_out), myfuncs.lin_regress_resid(np.array(self.pupil_data[i]['pupil_d']), to_project_out))[0]
            rois_p = copy.copy(rois_project_out)
            rois_p.remove('basal_forebrain_123_d')
            to_project_out = [np.array(dfs_brainstem[i][r]) for r in rois_p]
            r6[i] = sp.stats.pearsonr(myfuncs.lin_regress_resid(np.array(dfs_brainstem[i]['basal_forebrain_123_d']), to_project_out), myfuncs.lin_regress_resid(np.array(self.pupil_data[i]['pupil_d']), to_project_out))[0]
            rois_p = copy.copy(rois_project_out)
            rois_p.remove('basal_forebrain_4_d')
            to_project_out = [np.array(dfs_brainstem[i][r]) for r in rois_p]
            r7[i] = sp.stats.pearsonr(myfuncs.lin_regress_resid(np.array(dfs_brainstem[i]['basal_forebrain_4_d']), to_project_out), myfuncs.lin_regress_resid(np.array(self.pupil_data[i]['pupil_d']), to_project_out))[0]
        
        rois = ['SC', 'IC', 'LC', 'VTA', 'SN', 'BF-sept', 'BF-sub']
        rois_name = rois
        df = pd.DataFrame(np.vstack((r1, r2, r3, r4, r5, r6, r7)).T, columns=rois)
        df['subject'] = np.arange(len(self.subjects))
        colors = ['grey']
        df = df.ix[:,np.concatenate((np.array(rois), np.array(['subject'])))]
        k = df.groupby(['subject']).mean()
        k_s = k.stack().reset_index()
        k_s.columns = ['subject', 'area', 'bold']
        
        # plot:
        locs = np.arange(0,len(rois))
        bar_width = 0.2
        fig = plt.figure(figsize=(2.75,2))
        ax = fig.add_subplot(111)
        sns.barplot(x='area',  y='bold', units='subject', data=k_s, palette=colors, ci=None, linewidth=0, alpha=0.5, ax=ax)
        sns.stripplot(x="area", y="bold", data=k_s, jitter=True, size=2, palette=colors, edgecolor='black', linewidth=0.25, ax=ax, split=True, alpha=1)
        # add p-values:
        for r in range(len(rois)):
            p1 = myfuncs.permutationTest(k_s[(k_s['area']==rois[r])].bold, np.zeros(len(self.subjects)), paired=True)[1]
            # if p1 < 0.05:
            plt.text(s='{}'.format(round(p1, 3)), x=locs[r], y=plt.gca().get_ylim()[1]-((plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]) / 10.0), size=5, horizontalalignment='center',)
        # ax.legend_.remove()
        plt.xticks(locs, rois_name, rotation=45)
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder, 'correlation', 'bars', 'bars_single_trial_correlation_partial.pdf'))
        
        
        
        #
        # r1 = np.zeros(len(self.subjects))
        # r2 = np.zeros(len(self.subjects))
        # r3 = np.zeros(len(self.subjects))
        # r4 = np.zeros(len(self.subjects))
        # r5 = np.zeros(len(self.subjects))
        # r6 = np.zeros(len(self.subjects))
        # r7 = np.zeros(len(self.subjects))
        # for i in range(len(self.subjects)):
        #     r1[i] = sp.stats.pearsonr(dfs_brainstem[i]['sup_col_jw_d'], self.pupil_data[i]['pupil_d'])[0]
        #     r2[i] = sp.stats.pearsonr(dfs_brainstem[i]['inf_col_jw_d'], self.pupil_data[i]['pupil_d'])[0]
        #     r3[i] = sp.stats.pearsonr(dfs_brainstem[i]['LC_JW_d'], self.pupil_data[i]['pupil_d'])[0]
        #     r4[i] = sp.stats.pearsonr(dfs_brainstem[i]['mean_VTA_d'], self.pupil_data[i]['pupil_d'])[0]
        #     r5[i] = sp.stats.pearsonr(dfs_brainstem[i]['mean_SN_d'], self.pupil_data[i]['pupil_d'])[0]
        #     r6[i] = sp.stats.pearsonr(dfs_brainstem[i]['basal_forebrain_123_d'], self.pupil_data[i]['pupil_d'])[0]
        #     r7[i] = sp.stats.pearsonr(dfs_brainstem[i]['basal_forebrain_4_d'], self.pupil_data[i]['pupil_d'])[0]
        #
        # labels = ['SC', 'IC', 'LC', 'VTA', 'SN', 'BF-sept', 'BF-sub']
        # values = [r1, r2, r3, r4, r5, r6, r7]
        # MEANS = np.array([v.mean() for v in values])
        # SEMS = np.array([sp.stats.sem(v) for v in values])
        # p_mat = np.array([myfuncs.permutationTest(v, np.zeros(len(v)), paired=True)[1] for v in values])
        # ind = np.arange(0,len(MEANS))
        # bar_width = 0.90
        # fig = plt.figure(figsize=( (1+(len(MEANS)*0.15)),1.75))
        # ax = fig.add_subplot(111)
        # axes = [ax1]
        # plot_nr = 0
        # my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}
        # for i in range(len(MEANS)):
        #     ax.bar(ind[i], MEANS[i], width = bar_width, yerr=SEMS[i], color = 'k', alpha = 0.75, align = 'center', edgecolor = 'k', linewidth = 0, capsize = 0, ecolor = 'k', error_kw={'elinewidth':0.5,})
        # ax.tick_params(axis='y', which='major', labelsize=6)
        # ax.set_xticks(ind)
        # for i, pp in enumerate(p_mat):
        #     star1 = 'n.s.'
        #     if pp < 0.05:
        #         star1 = '*'
        #     if pp < 0.01:
        #         star1 = '**'
        #     if pp < 0.001:
        #         star1 = '***'
        #     ax.text(s=star1, x=ind[i], y=ax.get_ylim()[1]-((ax.get_ylim()[1] - ax.get_ylim()[0]) / 10.0), size=10, horizontalalignment='center',)
        # ax.set_ylabel('mean R', size=7)
        # ax.set_ylim(-0.04,0.08)
        # sns.despine(offset=10, trim=True)
        # ax.set_xticklabels(labels, rotation=45, size=7)
        # plt.tight_layout()
        # fig.savefig(os.path.join(self.figure_folder, 'correlation', 'bars', 'bars_single_trial_correlation.pdf'))
    
    
    def BRAINSTEM_choice(self, data_type):
        
        
        time_locked = 'stim_locked'
        self.make_pupil_BOLD_dataframe(data_type=data_type, time_locked=time_locked, regress_iti=False, regress_rt=True, regress_stimulus=False)
        
        data_brainstem = self.data_frame.copy()
        dfs_brainstem = [data_brainstem[data_brainstem.subject == s] for s in self.subjects]
        
        
        
        # add linear combinations:
        for i in range(len(self.subjects)):
            d = {'Y' : pd.Series(self.pupil_data[i]['pupil_d']),
                'X1' : pd.Series(dfs_brainstem[i]['LC_JW_d']),
                'X2' : pd.Series(dfs_brainstem[i]['mean_SN_d']),
                'X3' : pd.Series(dfs_brainstem[i]['mean_VTA_d']),
                'X4' : pd.Series(dfs_brainstem[i]['basal_forebrain_123_d']),
                'X5' : pd.Series(dfs_brainstem[i]['basal_forebrain_4_d']),
                'X6' : pd.Series(dfs_brainstem[i]['inf_col_jw_d']),
                'X7' : pd.Series(dfs_brainstem[i]['sup_col_jw_d']),
                }
            data = pd.DataFrame(d)
            formula = 'Y ~ X1 + X2 + X3 + X4 + X5'
            model = sm.ols(formula=formula, data=data)
            fitted = model.fit()
            values = np.array(fitted.fitted_values)
            values = (values - values.mean()) / values.std()
            dfs_brainstem[i]['BS_nMod'] = values
            formula = 'Y ~ X6 + X7'
            model = sm.ols(formula=formula, data=data)
            fitted = model.fit()
            values = np.array(fitted.fitted_values)
            values = (values - values.mean()) / values.std()
            dfs_brainstem[i]['BS_C'] = values
            
        r1 = np.zeros(len(self.subjects))
        r2 = np.zeros(len(self.subjects))
        for i in range(len(self.subjects)):
            r1[i] = sp.stats.pearsonr(dfs_brainstem[i]['BS_nMod'], self.pupil_data[i]['pupil_d'])[0]
            r2[i] = sp.stats.pearsonr(dfs_brainstem[i]['BS_C'], self.pupil_data[i]['pupil_d'])[0]
        
        print np.mean(r1)
        print sp.stats.sem(r1)
        print np.mean(r2)
        print sp.stats.sem(r2)
        
        # # pupil split:
        # self.pupil_l_ind = []
        # self.pupil_h_ind = []
        # for i in range(len(self.subjects)):
        #     d = self.pupil_data[i]
        #     dd = dfs_brainstem[i]
        #     p_h = []
        #     p_l = []
        #     for s in np.array(np.unique(d['session']), dtype=int):
        #         pupil = np.array(dd['BS_COCKTAIL'])[np.array(d.session) == s]
        #         p_l.append( pupil <= np.percentile(pupil, 40) )
        #         p_h.append( pupil >= np.percentile(pupil, 60) )
        #     self.pupil_l_ind.append(np.concatenate(p_l))
        #     self.pupil_h_ind.append(np.concatenate(p_h))
        # self.pupil_l_ind = np.concatenate(self.pupil_l_ind)
        # self.pupil_h_ind = np.concatenate(self.pupil_h_ind)
        # self.pupil_rest_ind = -(self.pupil_h_ind + self.pupil_l_ind)
        
        # # initialize behavior operator:
        # d = {
        # 'subj_idx' : pd.Series(self.subj_idx),
        # 'choice_a' : pd.Series(np.array(np.concatenate(self.yes), dtype=int)),
        # 'stimulus' : pd.Series(np.array(np.concatenate(self.present), dtype=int)),
        # 'rt' : pd.Series(np.concatenate(self.rt)),
        # 'pupil_d' : pd.Series(np.array(pd.concat(dfs_brainstem)['BS_COCKTAIL'])),
        # 'pupil_high' : pd.Series(self.pupil_h_ind),
        # 'run' : pd.Series(np.concatenate(self.run)),
        # 'session' : pd.Series(np.concatenate(self.session)),
        # }
        # self.df = pd.DataFrame(d)
        # self.df = self.df[~self.pupil_rest_ind]
        # self.behavior = myfuncs.behavior(self.df)
        #
        # # measures high vs low:
        # df_h = self.behavior.choice_fractions(split_by='pupil_high', split_target=1)
        # df_l = self.behavior.choice_fractions(split_by='pupil_high', split_target=0)
        # titles = ['rt', 'acc', "d'", 'crit', 'crit_abs', 'c_a', 'rtcv']
        # ylim_max = [2.5, 1, 2, 0.6, 1, 0.8, 10]
        # ylim_min = [0.0, 0, 0, 0.0, 0, 0.2, -4]
        # for i, t in enumerate(titles):
        #     dft = pd.concat((df_h.ix[:,'{}_1'.format(t)], df_l.ix[:,'{}_0'.format(t)]), axis=1)
        #     dft = dft.stack().reset_index()
        #     dft.columns = ['subject', 'measure', 'value']
        #     fig = plt.figure(figsize=(1.5,2))
        #     ax = fig.add_subplot(111)
        #     sns.barplot(x='measure',  y='value', units='subject', data=dft, ci=None, alpha=0.5, palette=['r', 'b'], ax=ax)
        #     sns.stripplot(x='measure', y='value', data=dft, jitter=False, size=2, edgecolor='black', linewidth=0.25, alpha=1, palette=['r', 'b'], ax=ax)
        #     values = np.vstack((dft[dft['measure'] == '{}_1'.format(t)].value, dft[dft['measure'] == '{}_0'.format(t)].value))
        #     ax.plot(np.array([0, 1]), values, color='black', lw=0.5, alpha=0.5)
        #     ax.set_title('p = {}'.format(round(myfuncs.permutationTest(values[0,:], values[1,:], paired=True)[1],3)))
        #     sns.despine(offset=10, trim=True)
        #     plt.tight_layout()
        #     fig.savefig(os.path.join(self.figure_folder, 'BRAINSTEM_behavior_SDT_measures_{}.pdf'.format(t)))
            
        # initialize behavior operator:
        d = {
        'subj_idx' : pd.Series(self.subj_idx),
        'choice_a' : pd.Series(np.array(np.concatenate(self.yes), dtype=int)),
        'stimulus' : pd.Series(np.array(np.concatenate(self.present), dtype=int)),
        'rt' : pd.Series(np.concatenate(self.rt)),
        'BS_nMod' : pd.Series(np.array(pd.concat(dfs_brainstem)['BS_nMod'])),
        'BS_C' : pd.Series(np.array(pd.concat(dfs_brainstem)['BS_C'])),
        'pupil_d' : pd.Series(np.array(pd.concat(self.pupil_data)['pupil_d'])),
        'run' : pd.Series(np.concatenate(self.run)),
        'session' : pd.Series(np.concatenate(self.session)),
        }
        self.df = pd.DataFrame(d)
        self.behavior = myfuncs.behavior(self.df)
        
        # for y in ['rt',]:
        R = []
        for bin_by in ['BS_nMod', 'BS_C']:
            for y in ['d', 'c', 'choice_a']:
                # model_comp = 'bayes'
                model_comp = 'seq'
                bins = 5
                fig, rs = self.behavior.SDT_correlation(bin_by=bin_by, n_bins=bins, y1=y, model_comp=model_comp)
                fig.savefig(os.path.join(self.figure_folder, 'BRAINSTEM_behavior_SDT_correlation_{}_{}.pdf'.format(bin_by, y)))
                R.append(rs)
        
        print myfuncs.permutationTest(R[0], R[3], paired=True)
        print myfuncs.permutationTest(R[1], R[4], paired=True)
        print myfuncs.permutationTest(R[2], R[5], paired=True)
            
    def correlation_matrix_single_trial_all_rois(self, data_type, partial=False):
        
        time_locked = 'stim_locked'
        self.make_pupil_BOLD_dataframe(data_type=data_type, time_locked=time_locked,)
        
        rois = np.array([
                    # PARCELATION:
                    # ------------

                    # # visual:
                    # 'Pole_occipital',
                    # 'G_occipital_sup',
                    # 'S_oc_sup_and_transversal',
                    # 'G_occipital_middle',
                    # 'S_occipital_ant',
                    # 'S_oc_middle_and_Lunatus',
                    # 'G_and_S_occipital_inf',
                    # 'S_collat_transv_post',
                    # 'G_oc-temp_med-Lingual',
                    # 'S_calcarine',
                    # 'G_cuneus',
                    #
                    # # temporal:
                    # 'Lat_Fis-post',
                    # 'G_temp_sup-Plan_tempo',
                    # 'S_temporal_transverse',
                    # 'G_temp_sup-G_T_transv',
                    # 'G_temp_sup-Lateral',
                    # 'S_temporal_sup',
                    # 'G_temporal_middle',
                    # 'S_temporal_inf',
                    # 'G_temporal_inf',
                    # # 'S_oc-temp_lat',
                    # 'G_oc-temp_med-Parahip',
                    # 'S_collat_transv_ant',
                    # 'G_oc-temp_lat-fusifor',
                    # 'S_oc-temp_med_and_Lingual',
                    # 'G_temp_sup-Plan_polar',
                    # 'Pole_temporal',
                    #
                    # # parietal:
                    # 'S_parieto_occipital',
                    # 'S_subparietal',
                    # 'G_precuneus',
                    # 'G_parietal_sup',
                    # 'S_intrapariet_and_P_trans',
                    # 'G_pariet_inf-Angular',
                    # 'S_interm_prim-Jensen',
                    # 'G_and_S_paracentral',
                    # 'S_postcentral',
                    # 'G_postcentral',
                    # 'S_central',
                    # 'G_pariet_inf-Supramar',
                    # 'G_and_S_subcentral',
                    #
                    # # insular:
                    # 'S_circular_insula_sup',
                    # 'G_insular_short',
                    # 'S_circular_insula_inf',
                    # 'G_Ins_lg_and_S_cent_ins',
                    # 'S_circular_insula_ant',
                    #
                    # # cingulate:
                    # 'G_cingul-Post-ventral',
                    # 'S_pericallosal',
                    # 'G_cingul-Post-dorsal',
                    # 'S_cingul-Marginalis',
                    # 'G_and_S_cingul-Mid-Post',
                    # 'G_and_S_cingul-Mid-Ant',
                    # 'G_and_S_cingul-Ant',
                    #
                    # # frontal:
                    # 'G_precentral',
                    # 'S_precentral-sup-part',
                    # 'S_precentral-inf-part',
                    # 'G_front_sup',
                    # 'S_front_sup',
                    # 'G_front_middle',
                    # 'S_front_middle',
                    # 'S_front_inf',
                    # 'G_front_inf-Opercular',
                    # 'G_front_inf-Triangul',
                    # 'S_orbital_lateral',
                    # 'Lat_Fis-ant-Horizont',
                    # 'Lat_Fis-ant-Vertical',
                    # 'G_front_inf-Orbital',
                    # 'G_and_S_transv_frontopol',
                    # 'G_and_S_frontomargin',
                    # 'G_orbital',
                    # 'S_orbital-H_Shaped',
                    # 'S_orbital_med-olfact',
                    # 'G_rectus',
                    # 'S_suborbital',
                    # 'G_subcallosal',

                    # PARCELATION LEFT:
                    # -----------------

                    # visual:
                    'lh.Pole_occipital',
                    'lh.G_occipital_sup',
                    'lh.S_oc_sup_and_transversal',
                    'lh.G_occipital_middle',
                    'lh.S_occipital_ant',
                    'lh.S_oc_middle_and_Lunatus',
                    'lh.G_and_S_occipital_inf',
                    'lh.S_collat_transv_post',
                    'lh.G_oc-temp_med-Lingual',
                    'lh.S_calcarine',
                    'lh.G_cuneus',

                    # temporal:
                    'lh.Lat_Fis-post',
                    'lh.G_temp_sup-Plan_tempo',
                    'lh.S_temporal_transverse',
                    'lh.G_temp_sup-G_T_transv',
                    'lh.G_temp_sup-Lateral',
                    'lh.S_temporal_sup',
                    'lh.G_temporal_middle',
                    'lh.S_temporal_inf',
                    'lh.G_temporal_inf',
                    # 'lh.S_oc-temp_lat',
                    'lh.G_oc-temp_med-Parahip',
                    'lh.S_collat_transv_ant',
                    'lh.G_oc-temp_lat-fusifor',
                    'lh.S_oc-temp_med_and_Lingual',
                    'lh.G_temp_sup-Plan_polar',
                    'lh.Pole_temporal',

                    # parietal:
                    'lh.S_parieto_occipital',
                    'lh.S_subparietal',
                    'lh.G_precuneus',
                    'lh.G_parietal_sup',
                    'lh.S_intrapariet_and_P_trans',
                    'lh.G_pariet_inf-Angular',
                    'lh.S_interm_prim-Jensen',
                    'lh.G_and_S_paracentral',
                    'lh.S_postcentral',
                    'lh.G_postcentral',
                    'lh.S_central',
                    'lh.G_pariet_inf-Supramar',
                    'lh.G_and_S_subcentral',

                    # insular:
                    'lh.S_circular_insula_sup',
                    'lh.G_insular_short',
                    'lh.S_circular_insula_inf',
                    'lh.G_Ins_lg_and_S_cent_ins',
                    'lh.S_circular_insula_ant',

                    # cingulate:
                    'lh.G_cingul-Post-ventral',
                    'lh.S_pericallosal',
                    'lh.G_cingul-Post-dorsal',
                    'lh.S_cingul-Marginalis',
                    'lh.G_and_S_cingul-Mid-Post',
                    'lh.G_and_S_cingul-Mid-Ant',
                    'lh.G_and_S_cingul-Ant',

                    # frontal:
                    # 'lh.G_precentral',
                    'lh.S_precentral-sup-part',
                    'lh.S_precentral-inf-part',
                    'lh.G_front_sup',
                    'lh.S_front_sup',
                    'lh.G_front_middle',
                    # 'lh.S_front_middle',
                    'lh.S_front_inf',
                    'lh.G_front_inf-Opercular',
                    'lh.G_front_inf-Triangul',
                    'lh.S_orbital_lateral',
                    'lh.Lat_Fis-ant-Horizont',
                    'lh.Lat_Fis-ant-Vertical',
                    'lh.G_front_inf-Orbital',
                    'lh.G_and_S_transv_frontopol',
                    'lh.G_and_S_frontomargin',
                    'lh.G_orbital',
                    'lh.S_orbital-H_Shaped',
                    'lh.S_orbital_med-olfact',
                    'lh.G_rectus',
                    'lh.S_suborbital',
                    'lh.G_subcallosal',

                    # PARCELATION RIGHT:
                    # -----------------

                    # visual:
                    'rh.Pole_occipital',
                    'rh.G_occipital_sup',
                    'rh.S_oc_sup_and_transversal',
                    'rh.G_occipital_middle',
                    'rh.S_occipital_ant',
                    'rh.S_oc_middle_and_Lunatus',
                    'rh.G_and_S_occipital_inf',
                    'rh.S_collat_transv_post',
                    'rh.G_oc-temp_med-Lingual',
                    'rh.S_calcarine',
                    'rh.G_cuneus',

                    # temporal:
                    'rh.Lat_Fis-post',
                    'rh.G_temp_sup-Plan_tempo',
                    'rh.S_temporal_transverse',
                    'rh.G_temp_sup-G_T_transv',
                    'rh.G_temp_sup-Lateral',
                    'rh.S_temporal_sup',
                    'rh.G_temporal_middle',
                    'rh.S_temporal_inf',
                    'rh.G_temporal_inf',
                    # 'rh.S_oc-temp_lat',
                    'rh.G_oc-temp_med-Parahip',
                    'rh.S_collat_transv_ant',
                    'rh.G_oc-temp_lat-fusifor',
                    'rh.S_oc-temp_med_and_Lingual',
                    'rh.G_temp_sup-Plan_polar',
                    'rh.Pole_temporal',

                    # parietal:
                    'rh.S_parieto_occipital',
                    'rh.S_subparietal',
                    'rh.G_precuneus',
                    'rh.G_parietal_sup',
                    'rh.S_intrapariet_and_P_trans',
                    'rh.G_pariet_inf-Angular',
                    'rh.S_interm_prim-Jensen',
                    'rh.G_and_S_paracentral',
                    'rh.S_postcentral',
                    'rh.G_postcentral',
                    'rh.S_central',
                    'rh.G_pariet_inf-Supramar',
                    'rh.G_and_S_subcentral',

                    # insular:
                    'rh.S_circular_insula_sup',
                    'rh.G_insular_short',
                    'rh.S_circular_insula_inf',
                    'rh.G_Ins_lg_and_S_cent_ins',
                    'rh.S_circular_insula_ant',

                    # cingulate:
                    'rh.G_cingul-Post-ventral',
                    'rh.S_pericallosal',
                    'rh.G_cingul-Post-dorsal',
                    'rh.S_cingul-Marginalis',
                    'rh.G_and_S_cingul-Mid-Post',
                    'rh.G_and_S_cingul-Mid-Ant',
                    'rh.G_and_S_cingul-Ant',

                    # frontal:
                    # 'rh.G_precentral',
                    'rh.S_precentral-sup-part',
                    'rh.S_precentral-inf-part',
                    'rh.G_front_sup',
                    'rh.S_front_sup',
                    'rh.G_front_middle',
                    # 'rh.S_front_middle',
                    'rh.S_front_inf',
                    'rh.G_front_inf-Opercular',
                    'rh.G_front_inf-Triangul',
                    'rh.S_orbital_lateral',
                    'rh.Lat_Fis-ant-Horizont',
                    'rh.Lat_Fis-ant-Vertical',
                    'rh.G_front_inf-Orbital',
                    'rh.G_and_S_transv_frontopol',
                    'rh.G_and_S_frontomargin',
                    'rh.G_orbital',
                    'rh.S_orbital-H_Shaped',
                    'rh.S_orbital_med-olfact',
                    'rh.G_rectus',
                    'rh.S_suborbital',
                    'rh.G_subcallosal',
                ])
        
        # throw away ROIs based on min nr_voxels:
        # nr_voxels = np.array([ 1499.,  1277.,   784.,  1543.,   395.,   348.,   857.,   258.,
        #         2021.,  1057.,  1075.,   726.,   620.,   196.,   399.,  2241.,
        #         3354.,  2895.,   638.,  1839.,   546.,  1219.,   544.,  1735.,
        #         1212.,   566.,  1635.,  1126.,   555.,  2027.,  1788.,  1586.,
        #         2342.,   187.,   304.,  1549.,  1107.,  1310.,  2302.,  1011.,
        #          890.,   652.,   810.,   396.,   319.,   245.,   789.,   461.,
        #          676.,  1013.,   960.,  1765.,  1345.,   400.,   799.,  1460.,
        #          131.,  2032.,   877.,  1093.,  1240.,   782.,   196.,   198.,
        #          134.,   314.,   181.,   237.,  1826.,   604.,   342.,   203.,
        #          262.,   148.])
        # rois = rois[(nr_voxels > 500)]
        # rois = rois[np.concatenate(( (nr_voxels > 500), (nr_voxels > 500) ))]
        
        wrong_rois = np.zeros(len(rois))
        for i in range(len(self.subjects)):
            C = np.vstack([np.array(self.data_frame[(self.data_frame.subject == self.subjects[i])][roi]) for roi in rois]).T
            median_sdt = np.median(np.std(C[:,0], axis=0))
            wrong_rois = wrong_rois + np.array(np.std(C, axis=0) > (5*median_sdt), dtype=bool)
            # wrong_rois = wrong_rois + np.array(np.std(C, axis=0) > 20, dtype=bool)
            # wrong_rois = wrong_rois + np.array(np.array((C>75)+(C<-75)).sum(axis=0), dtype=bool)
        clean_rois = (wrong_rois == 0)
        # clean_rois[:71] = clean_rois[:71] * clean_rois[71:]
        # clean_rois[71:] = clean_rois[:71] * clean_rois[71:]
        rois = rois[clean_rois]
        
        print
        print
        print len(rois)
        print
        
        # boundaries for plot:
        boundaries = [11,21,34,39,45]
        # boundaries = [11,22,35,40,47,54,65,75,88,93,99]
        
        # shell()
        
        # for signal_type in ['baseline', 'phasic']:
        for signal_type in ['phasic']:
            
            contrast_across_dvs = []
            for dv in ['cor', 'mean', 'var', 'cov', 'snr']:
            # for dv in ['var',]:
                corrmats_across = []
                mean_corrmats_across = []
                
                for condition in ['high', 'low', 'all']:
                    if (signal_type == 'baseline')*(condition == 'low'):
                        print 'baseline low'
                        ind = np.concatenate(self.pupil_b_l_ind)
                    elif (signal_type == 'baseline')*(condition == 'high'):
                        print 'baseline high'
                        ind = np.concatenate(self.pupil_b_h_ind)
                    elif (signal_type == 'phasic')*(condition == 'low'):
                        print 'phasic low'
                        ind = np.concatenate(self.pupil_l_ind)
                    elif (signal_type == 'phasic')*(condition == 'high'):
                        print 'phasic high'
                        ind = np.concatenate(self.pupil_h_ind)
                    else:
                        print 'all'
                        ind = np.ones(len(self.data_frame), dtype=bool)
                    
                    corrmats = np.zeros((len(rois), len(rois), len(self.subjects)))
                    for i in range(len(self.subjects)):
                        C = np.vstack([np.array(self.data_frame[(self.data_frame.subject == self.subjects[i])*ind][roi]) for roi in rois]).T
                        if dv == 'cor':
                            if partial:
                                corrmats[:,:,i], dummy = myfuncs.corr_matrix_partial(C)
                            else:
                                corrmats[:,:,i], dummy = myfuncs.corr_matrix(C, dv='cor')
                        if dv == 'var':
                            corrmats[:,:,i], dummy = myfuncs.corr_matrix(C, dv='var')
                        if dv == 'cov':
                            corrmats[:,:,i], dummy = myfuncs.corr_matrix(C, dv='cov')
                        if dv == 'mean':
                            corrmats[:,:,i], dummy = myfuncs.corr_matrix(C, dv='mean')
                        if dv == 'snr':
                            corrmats[:,:,i], dummy = myfuncs.corr_matrix(C, dv='snr')
                    
                    corrmats_across.append(corrmats)
                    mean_corrmat = corrmats.mean(axis=-1)
                    p_mat = np.zeros(mean_corrmat.shape)
                    mean_corrmats_across.append(mean_corrmat)
                    
                    # for i in range(p_mat.shape[0]):
                    #     for j in range(p_mat.shape[1]):
                    #         p_mat[i,j] = myfuncs.permutationTest( corrmats[i,j,:], np.zeros(corrmats[i,j,:].shape[0]) )[1]
                    #
                    # # fdr correction:
                    # p_mat = mne.stats.fdr_correction(p_mat, 0.05)[1]
                    
                    # shell()
                    
                    # plot matrix:
                    mask =  np.tri(mean_corrmat.shape[0], k=0)
                    corrmat_m = np.ma.masked_where(mask, mean_corrmat)
                    p_mat_m = np.ma.masked_where(mask.T, p_mat)
                    roi_names = rois
                    fig = plt.figure(figsize=(5,4))
                    ax = fig.add_subplot(111)
                    vmax = max(abs(mean_corrmat.ravel()))
                    if (dv == 'cor') or (dv == 'mean') or (dv == 'snr'):
                        im = ax.pcolormesh(corrmat_m, cmap='bwr', vmin=-vmax, vmax=vmax)
                    else:
                        im = ax.pcolormesh(corrmat_m, cmap='Reds', vmin=0, vmax=vmax)
                    ax.set_xlim(xmax=len(rois))
                    ax.set_ylim(ymax=len(rois))
                    ax.set_yticks(arange(0.5,len(rois)+.5))
                    ax.set_xticks(arange(0.5,len(rois)+.5))
                    ax.set_yticklabels(roi_names)
                    ax.set_xticklabels(roi_names, rotation=270)
                    ax.patch.set_hatch('x')
                    ax.set_aspect(1)
                    fig.colorbar(im)
                    plt.tight_layout()
                    if partial:
                        fig.savefig(os.path.join(self.figure_folder, 'correlation', 'big_matrix', '7a_correlation_matrix_single_partial_{}_{}_{}.pdf'.format(signal_type, condition, dv)))
                    else:
                        fig.savefig(os.path.join(self.figure_folder, 'correlation', 'big_matrix', '7a_correlation_matrix_single_{}_{}_{}.pdf'.format(signal_type, condition, dv)))
                
                # add to across dvs:
                contrast_mat = mean_corrmats_across[0] - mean_corrmats_across[1]
                contrast_across_dvs.append(contrast_mat)

                # plotting:
                # ---------
                # correlation:

                shell()
                
                self.criterion = np.array([ 0.19757142,  0.09292426,  0.09920115, -0.15260826,  0.02021234, 0.1869882 ,  0.24642884, -0.12784264, -0.04536516, -0.02965839, 0.09878793,  0.53726986,  0.25126305,  1.03668549])
                
                self.acc_lo = np.array([ 0.71052632,  0.734375  ,  0.76439791,  0.76953125,  0.66875   , 0.69791667,  0.73863636,  0.70253165,  0.75789474,  0.65104167, 0.78645833,  0.73298429,  0.79428571,  0.734375  ])
                self.acc_hi = array([ 0.64736842,  0.78645833,  0.7382199 ,  0.76953125,  0.6375    , 0.69270833,  0.71590909,  0.63291139,  0.77368421,  0.77083333, 0.79166667,  0.78534031,  0.85714286,  0.6875    ])
                
                c = self.criterion_hi - self.criterion_lo
                c = self.yes_hi - self.yes_lo

                a = bn.nanmean(bn.nanmean(corrmats_across[0],axis=0),axis=0) - bn.nanmean(bn.nanmean(corrmats_across[1],axis=0),axis=0)
                a_err = sp.stats.sem(bn.nanmean(corrmats_across[0],axis=0) - bn.nanmean(corrmats_across[1],axis=0), axis=0)
                fig = plt.figure(figsize=(2,2))
                ax = fig.add_subplot(111)
                ax.errorbar(x=c, y=a, yerr=a_err, fmt='o', ms=0, ecolor='k', elinewidth=0.5, capsize=0)
                myfuncs.correlation_plot(c, a, ax=ax, line=True, stat='spearmanr')
                plt.ylabel(dv + ' (hi - lo)')
                plt.xlabel('Criterion (hi - lo)')
                plt.tight_layout()
                fig.savefig(os.path.join(self.figure_folder, 'correlation', 'big_matrix', 'correlation_criterion_{}_{}_contrast_matrix.pdf'.format(signal_type, dv)))

                a_a = np.array([bn.nanmean(corrmats_across[0][:,:,i][(corrmats_across[0][:,:,i]-corrmats_across[1][:,:,i])<0],axis=0) for i in range(len(self.subjects))]) - np.array([bn.nanmean(corrmats_across[1][:,:,i][(corrmats_across[0][:,:,i]-corrmats_across[1][:,:,i])<0],axis=0) for i in range(len(self.subjects))])
                a_a_err = sp.stats.sem(bn.nanmean(corrmats_across[0],axis=0) - bn.nanmean(corrmats_across[1],axis=0), axis=0)
                fig = plt.figure(figsize=(2,2))
                ax = fig.add_subplot(111)
                ax.errorbar(x=c, y=a_a, yerr=a_a_err, fmt='o', ms=0, ecolor='k', elinewidth=0.5, capsize=0)
                myfuncs.correlation_plot(c, a_a, ax=ax, line=True, stat='spearmanr')
                plt.ylabel(dv + ' (hi - lo)')
                plt.xlabel('Criterion (hi - lo)')
                plt.tight_layout()
                fig.savefig(os.path.join(self.figure_folder, 'correlation', 'big_matrix', 'correlation_criterion_{}_{}_contrast_matrix_neg.pdf'.format(signal_type, dv)))

                a_b = np.array([bn.nanmean(corrmats_across[0][:,:,i][(corrmats_across[0][:,:,i]-corrmats_across[1][:,:,i])>0],axis=0) for i in range(len(self.subjects))]) - np.array([bn.nanmean(corrmats_across[1][:,:,i][(corrmats_across[0][:,:,i]-corrmats_across[1][:,:,i])>0],axis=0) for i in range(len(self.subjects))])
                a_b_err = sp.stats.sem(bn.nanmean(corrmats_across[0],axis=0) - bn.nanmean(corrmats_across[1],axis=0), axis=0)
                fig = plt.figure(figsize=(2,2))
                ax = fig.add_subplot(111)
                ax.errorbar(x=c, y=a_b, yerr=a_b_err, fmt='o', ms=0, ecolor='k', elinewidth=0.5, capsize=0)
                myfuncs.correlation_plot(c, a_b, ax=ax, line=True, stat='spearmanr')
                plt.ylabel(dv + ' (hi - lo)')
                plt.xlabel('Criterion (hi - lo)')
                plt.tight_layout()
                fig.savefig(os.path.join(self.figure_folder, 'correlation', 'big_matrix', 'correlation_criterion_{}_{}_contrast_matrix_pos.pdf'.format(signal_type, dv)))

                aa = np.vstack((
                    np.array([bn.nanmean(corrmats_across[0][:,:,i][(corrmats_across[0][:,:,i]-corrmats_across[1][:,:,i])>0],axis=0) for i in range(len(self.subjects))]) - np.array([bn.nanmean(corrmats_across[1][:,:,i][(corrmats_across[0][:,:,i]-corrmats_across[1][:,:,i])>0],axis=0) for i in range(len(self.subjects))]),
                    (self.criterion_hi - self.criterion_lo),
                    )).T

                bb = np.vstack((
                    np.array([bn.nanmean(corrmats_across[0][:,:,i][(corrmats_across[0][:,:,i]-corrmats_across[1][:,:,i])<0],axis=0) for i in range(len(self.subjects))]) - np.array([bn.nanmean(corrmats_across[1][:,:,i][(corrmats_across[0][:,:,i]-corrmats_across[1][:,:,i])<0],axis=0) for i in range(len(self.subjects))]),
                    (self.criterion_hi - self.criterion_lo),
                    )).T

                print
                print
                print dv
                print myfuncs.permutationTest_correlation(aa,bb)

                #

                # spit group and make bars:
                split = c < np.median(c)
                values = [a_b[split], a_b[-split], a_a[split], a_a[-split],]
                MEANS = np.array([np.mean(v) for v in values])
                SEMS = np.array([sp.stats.sem(v) for v in values])
                p = [sp.stats.ttest_ind(values[0], values[1])[1], sp.stats.ttest_ind(values[2], values[3])[1]]
                ind = np.arange(0,len(MEANS))
                bar_width = 0.75
                fig = plt.figure(figsize=(1.75,1.75))
                ax = plt.subplot(111)
                my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}
                ax.bar(ind, MEANS, yerr=SEMS, width=bar_width, color=['orange','orange','green','green'], alpha=1, align='center', edgecolor = 'k', linewidth = 0, capsize = 0, ecolor = 'k', error_kw={'elinewidth':0.5,})
                ax.tick_params(axis='y', which='major', labelsize=6)
                ax.set_xticks(ind)
                for i, pp in enumerate(p):
                    ax.text(s=str(round(pp,3)), x=ind[::2][i], y=ax.get_ylim()[1]-((ax.get_ylim()[1] - ax.get_ylim()[0]) / 10.0), size=7)
                ax.set_ylabel(dv + ' hi - lo', size=7)
                sns.despine(offset=10, trim=True)
                ax.set_xticklabels(['+lib', '+con', '-lib', '-con'], rotation=45, size=7)
                plt.tight_layout()
                fig.savefig(os.path.join(self.figure_folder, 'correlation', 'big_matrix', 'bars_group_split_{}_{}.pdf'.format(signal_type, dv)))

                # ANOVA 1:
                data = np.concatenate((np.array([bn.nanmean(corrmats_across[0][:,:,i][(corrmats_across[0][:,:,i]-corrmats_across[1][:,:,i])<0],axis=0) for i in range(len(self.subjects))]) - np.array([bn.nanmean(corrmats_across[1][:,:,i][(corrmats_across[0][:,:,i]-corrmats_across[1][:,:,i])<0],axis=0) for i in range(len(self.subjects))]), np.array([bn.nanmean(corrmats_across[0][:,:,i][(corrmats_across[0][:,:,i]-corrmats_across[1][:,:,i])>0],axis=0) for i in range(len(self.subjects))]) - np.array([bn.nanmean(corrmats_across[1][:,:,i][(corrmats_across[0][:,:,i]-corrmats_across[1][:,:,i])>0],axis=0) for i in range(len(self.subjects))])))
                subject = np.concatenate(( np.arange(len(self.subjects)), np.arange(len(self.subjects)) ))
                region = np.concatenate(( np.zeros(len(self.subjects)), np.ones(len(self.subjects)) ))
                c_split = np.concatenate(( np.array(split, dtype=int), np.array(split, dtype=int)))

                # d = {
                #     'data':data,
                #     'subject':subject,
                #     'region':region,
                #     'split':c_split,
                # }
                # df = pd.DataFrame(d)

                d = rlc.OrdDict([('region', robjects.IntVector(list(region.ravel()))), ('split', robjects.IntVector(list(c_split.ravel()))), ('subject', robjects.IntVector(list(subject.ravel()))), ('data', robjects.FloatVector(list(data.ravel())))])
                robjects.r.assign('dataf', robjects.DataFrame(d))
                robjects.r('attach(dataf)')

                # shell()


                statres = robjects.r('res = summary(aov(data ~ (as.factor(region)*as.factor(split)) + Error(as.factor(subject)/as.factor(region)) + as.factor(split), dataf))')
                p1 = statres[-1][0][4][0]    # we will log-transform and min p values
                p2 = statres[-1][0][4][1]    # we will log-transform and min p values
                p3 = statres[-1][0][4][2]    # we will log-transform and min p values

                text_file = open(os.path.join(self.figure_folder, 'correlation', 'big_matrix', 'ANOVA_{}_{}.txt'.format(signal_type, dv)), 'w')
                for string in statres:
                    text_file.write(str(string))
                text_file.close()

                print statres
                
                # contrast matrix:
                mask =  np.tri(contrast_mat.shape[0], k=0)
                corrmat_m = np.ma.masked_where(mask, contrast_mat)
                p_mat_m = np.ma.masked_where(mask.T, p_mat)
                roi_names = rois
                fig = plt.figure(figsize=(5,4))
                # fig = plt.figure(figsize=(10,8))
                ax = fig.add_subplot(111)
                vmax = max(abs(contrast_mat.ravel()))
                im = ax.pcolormesh(corrmat_m, cmap='bwr', vmin=-vmax, vmax=vmax)
                ax.set_xlim(xmax=len(rois))
                ax.set_ylim(ymax=len(rois))
                ax.set_yticks(arange(0.5,len(rois)+.5))
                ax.set_xticks(arange(0.5,len(rois)+.5))
                ax.set_yticklabels(roi_names)
                ax.set_xticklabels(roi_names, rotation=270)
                ax.patch.set_hatch('x')
                ax.set_aspect(1)
                fig.colorbar(im)
                for l in boundaries:
                    # plt.axvline(l, lw=1)
                    # plt.axhline(l, lw=1)
                    
                    plt.vlines(l, 0, l, lw=1)
                    plt.hlines(l, l, corrmat_m.shape[0], lw=1)
                    
                plt.tight_layout()
                if partial:
                    fig.savefig(os.path.join(self.figure_folder, 'correlation', 'big_matrix', '7a_correlation_matrix_single_partial_{}_{}_{}.pdf'.format(signal_type, 'contrast', dv)))
                else:
                    fig.savefig(os.path.join(self.figure_folder, 'correlation', 'big_matrix', '7a_correlation_matrix_single_{}_{}_{}.pdf'.format(signal_type, 'contrast', dv)))

            
            # shell()
            
            # correlation of contrast matrices:
            cor = np.zeros(3)
            p = np.zeros(3)
            for i, dv in enumerate(['mean', 'var', 'cov',]):
                cor[i] = sp.stats.pearsonr(np.ma.masked_where(mask, contrast_across_dvs[i+1]).ravel(), np.ma.masked_where(mask, contrast_across_dvs[0]).ravel())[0]
                p[i] = sp.stats.pearsonr(np.ma.masked_where(mask, contrast_across_dvs[i+1]).ravel(), np.ma.masked_where(mask, contrast_across_dvs[0]).ravel())[1]
                fig = plt.figure(figsize=(1.75,1.75))
                ax = fig.add_subplot(111)
                myfuncs.correlation_plot(np.ma.masked_where(mask, contrast_across_dvs[0]).ravel(), np.ma.masked_where(mask, contrast_across_dvs[i+1]).ravel(), ax=ax, line=True)
                ax.set_xlabel('contrast correlation matrix')
                ax.set_ylabel('contrast {} matrix'.format(dv))
                fig.tight_layout()
                fig.savefig(os.path.join(self.figure_folder, 'correlation', 'big_matrix', 'contrast_correlation_{}_{}.pdf'.format(signal_type, dv)))
            labels = ['mean', 'var', 'cov', 'snr']
            MEANS = cor
            ind = np.arange(0,len(MEANS))
            bar_width = 0.75
            fig = plt.figure(figsize=(1.5,1.75))
            ax = plt.subplot(111)
            my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}
            ax.bar(ind, MEANS, width=bar_width, color='k', alpha=1, align='center', edgecolor = 'k', linewidth = 0, capsize = 0, ecolor = 'k', error_kw={'elinewidth':0.5,})
            ax.tick_params(axis='y', which='major', labelsize=6)
            ax.set_xticks(ind)
            for i, pp in enumerate(p):
                star1 = 'n.s.'
                if pp < 0.05:
                    star1 = '*'
                if pp < 0.01:
                    star1 = '**'
                if pp < 0.001:
                    star1 = '***'
                ax.text(s=star1, x=ind[i], y=ax.get_ylim()[1]-((ax.get_ylim()[1] - ax.get_ylim()[0]) / 10.0), size=10)
            ax.set_ylabel('Correlation', size=7)
            sns.despine(offset=10, trim=True)
            ax.set_xticklabels(labels, rotation=45, size=7)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, 'correlation', 'big_matrix', 'bars_{}.pdf'.format(signal_type)))

            # specific to pos or neg voxels...:

            for condition in ['high', 'low', 'all']:
                if (signal_type == 'baseline')*(condition == 'low'):
                    print 'baseline low'
                    ind = np.concatenate(self.pupil_b_l_ind)
                elif (signal_type == 'baseline')*(condition == 'high'):
                    print 'baseline high'
                    ind = np.concatenate(self.pupil_b_h_ind)
                elif (signal_type == 'phasic')*(condition == 'low'):
                    print 'phasic low'
                    ind = np.concatenate(self.pupil_l_ind)
                elif (signal_type == 'phasic')*(condition == 'high'):
                    print 'phasic high'
                    ind = np.concatenate(self.pupil_h_ind)
                else:
                    print 'all'
                    ind = np.ones(len(self.data_frame), dtype=bool)

                pos_indices = contrast_across_dvs[0] > 0
                neg_indices = contrast_across_dvs[0] < 0
                for dv, yl in zip(['mean', 'var', 'cov', 'snr'], [0.4, 7, 2.5, 0.25]):
                    corrmats = np.zeros((len(rois), len(rois), len(self.subjects)))
                    for i in range(len(self.subjects)):

                        # shell()

                        C = np.vstack([np.array(self.data_frame[(self.data_frame.subject == self.subjects[i])*ind][roi]) for roi in rois]).T
                        if dv == 'var':
                            corrmats[:,:,i], dummy = myfuncs.corr_matrix(C, dv='var')
                        if dv == 'cov':
                            corrmats[:,:,i], dummy = myfuncs.corr_matrix(C, dv='cov')
                        if dv == 'mean':
                            corrmats[:,:,i], dummy = myfuncs.corr_matrix(C, dv='mean')
                        if dv == 'snr':
                            corrmats[:,:,i], dummy = myfuncs.corr_matrix(C, dv='snr')
                    pos = np.array([corrmats[pos_indices,i].mean() for i in range(len(self.subjects))])
                    neg = np.array([corrmats[neg_indices,i].mean() for i in range(len(self.subjects))])
                    values = [pos, neg]
                    MEANS = np.array([np.mean(v) for v in values])
                    SEMS = np.array([sp.stats.sem(v) for v in values])
                    p = [sp.stats.ttest_rel(pos, neg)[1]]
                    x = np.arange(0,len(MEANS))
                    bar_width = 0.75
                    fig = plt.figure(figsize=(1.25,1.75))
                    ax = plt.subplot(111)
                    my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}
                    ax.bar(x, MEANS, yerr=SEMS, width=bar_width, color=['orange','green'], alpha=1, align='center', edgecolor = 'k', linewidth = 0, capsize = 0, ecolor = 'k', error_kw={'elinewidth':0.5,})
                    ax.tick_params(axis='y', which='major', labelsize=6)
                    ax.set_xticks(x)
                    for i, pp in enumerate(p):
                        star1 = 'n.s.'
                        if pp < 0.05:
                            star1 = '*'
                        if pp < 0.01:
                            star1 = '**'
                        if pp < 0.001:
                            star1 = '***'
                        # ax.text(s=star1, x=ind[i], y=ax.get_ylim()[1]-((ax.get_ylim()[1] - ax.get_ylim()[0]) / 10.0), size=10)
                        ax.text(s=str(round(pp,3)), x=x[i], y=ax.get_ylim()[1]-((ax.get_ylim()[1] - ax.get_ylim()[0]) / 10.0), size=7)
                    ax.set_ylim(0,yl)
                    ax.set_ylabel(dv, size=7)
                    sns.despine(offset=10, trim=True)
                    ax.set_xticklabels(['pos', 'neg'], rotation=45, size=7)
                    plt.tight_layout()
                    fig.savefig(os.path.join(self.figure_folder, 'correlation', 'big_matrix', 'bars_{}_{}_{}.pdf'.format(signal_type, dv, condition)))
                    
                    
                    
    def VISUAL_snr(self, data_type, partial=False):
        
        # time_locked = 'stim_locked'
        # self.make_pupil_BOLD_dataframe(data_type=data_type, time_locked=time_locked, regress_iti=False, regress_rt_signal=True)
        
        nr_voxels = 50
        
        dfs = []
        for i in range(len(self.subjects)):
            s = []
            for roi in ['V1', 'V2', 'V3']:
                files = sort(glob.glob(os.path.join(self.data_folder, 'noise_correlation', 'scalars{}_{}*{}_center*'.format(self.postfix, self.subjects[i], roi))))
                s.append( np.concatenate([np.load(f) for f in files]) )
                files = sort(glob.glob(os.path.join(self.data_folder, 'noise_correlation', 'scalars{}_{}*{}_surround*'.format(self.postfix, self.subjects[i], roi))))
                s.append( np.concatenate([np.load(f) for f in files]) )
            scalars_visual = np.hstack(s)
    
            center_ind = np.array(np.concatenate((np.ones(nr_voxels), np.zeros(nr_voxels), np.ones(nr_voxels), np.zeros(nr_voxels), np.ones(nr_voxels), np.zeros(nr_voxels),)), dtype=bool)
            v1_ind = np.array(np.concatenate((np.ones(nr_voxels), np.ones(nr_voxels), np.zeros(nr_voxels), np.zeros(nr_voxels), np.zeros(nr_voxels), np.zeros(nr_voxels),)), dtype=bool)
            v2_ind = np.array(np.concatenate((np.zeros(nr_voxels), np.zeros(nr_voxels), np.ones(nr_voxels), np.ones(nr_voxels), np.zeros(nr_voxels), np.zeros(nr_voxels),)), dtype=bool)
            v3_ind = np.array(np.concatenate((np.zeros(nr_voxels), np.zeros(nr_voxels), np.zeros(nr_voxels), np.zeros(nr_voxels), np.ones(nr_voxels), np.ones(nr_voxels),)), dtype=bool)
        
            scalars_center_V123 = scalars_visual[:,center_ind].mean(axis=1)
            scalars_surround_V123 = scalars_visual[:,~center_ind].mean(axis=1)
            scalars_center_V1 = scalars_visual[:,center_ind & v1_ind].mean(axis=1)
            scalars_surround_V1 = scalars_visual[:,~center_ind & v1_ind].mean(axis=1)
            scalars_center_V2 = scalars_visual[:,center_ind & v2_ind].mean(axis=1)
            scalars_surround_V2 = scalars_visual[:,~center_ind & v2_ind].mean(axis=1)
            scalars_center_V3 = scalars_visual[:,center_ind & v3_ind].mean(axis=1)
            scalars_surround_V3 = scalars_visual[:,~center_ind & v3_ind].mean(axis=1)
            
            # regress out whole brain signal, and add to dataframe:
            scalars_center_V123_clean = myfuncs.lin_regress_resid(scalars_center_V123, [scalars_surround_V123])
            scalars_center_V1_clean = myfuncs.lin_regress_resid(scalars_center_V1, [scalars_surround_V1])
            scalars_center_V2_clean = myfuncs.lin_regress_resid(scalars_center_V2, [scalars_surround_V2])
            scalars_center_V3_clean = myfuncs.lin_regress_resid(scalars_center_V3, [scalars_surround_V3])
            
            df = {}
            df['V1'] = scalars_center_V1_clean
            df['V2'] = scalars_center_V2_clean
            df['V3'] = scalars_center_V3_clean
            df['V123'] = scalars_center_V123_clean
            df['V1_center'] = scalars_center_V1
            df['V2_center'] = scalars_center_V2
            df['V3_center'] = scalars_center_V3
            df['V123_center'] = scalars_center_V123
            df['V1_surround'] = scalars_surround_V1
            df['V2_surround'] = scalars_surround_V2
            df['V3_surround'] = scalars_surround_V3
            df['V123_surround'] = scalars_surround_V123
            df['subject'] = self.subjects[i]
            
            df = pd.DataFrame(df)
            dfs.append(df)
        data_frame = pd.concat(dfs)
        
        import copy
        
        averages_av_snr_center = np.array([np.array([np.array(data_frame[data_frame.subject == s][roi]).mean() / np.array(data_frame[data_frame.subject == s][roi]).std() for i, s in enumerate(self.subjects)]) for roi in ['V1_center', 'V2_center', 'V3_center', 'V123_center']])
        averages_av_snr_surround = np.array([np.array([np.array(data_frame[data_frame.subject == s][roi]).mean() / np.array(data_frame[data_frame.subject == s][roi]).std() for i, s in enumerate(self.subjects)]) for roi in ['V1_surround', 'V2_surround', 'V3_surround', 'V123_surround']])
        
        averages_av_snr_center_h = np.array([np.array([np.array(data_frame[data_frame.subject == s][roi][self.pupil_h_ind[i]]).mean() / np.array(data_frame[data_frame.subject == s][roi][self.pupil_h_ind[i]]).std() for i, s in enumerate(self.subjects)]) for roi in ['V1_center', 'V2_center', 'V3_center', 'V123_center']])
        averages_av_snr_surround_h = np.array([np.array([np.array(data_frame[data_frame.subject == s][roi][self.pupil_h_ind[i]]).mean() / np.array(data_frame[data_frame.subject == s][roi][self.pupil_h_ind[i]]).std() for i, s in enumerate(self.subjects)]) for roi in ['V1_surround', 'V2_surround', 'V3_surround', 'V123_surround']])
        averages_av_snr_center_clean_h = np.array([np.array([np.array(data_frame[data_frame.subject == s][roi][self.pupil_h_ind[i]]).mean() / np.array(data_frame[data_frame.subject == s][roi][self.pupil_h_ind[i]]).std() for i, s in enumerate(self.subjects)]) for roi in ['V1', 'V2', 'V3', 'V123']])
        
        averages_av_snr_center_l = np.array([np.array([np.array(data_frame[data_frame.subject == s][roi][self.pupil_l_ind[i]]).mean() / np.array(data_frame[data_frame.subject == s][roi][self.pupil_l_ind[i]]).std() for i, s in enumerate(self.subjects)]) for roi in ['V1_center', 'V2_center', 'V3_center', 'V123_center']])
        averages_av_snr_surround_l = np.array([np.array([np.array(data_frame[data_frame.subject == s][roi][self.pupil_l_ind[i]]).mean() / np.array(data_frame[data_frame.subject == s][roi][self.pupil_l_ind[i]]).std() for i, s in enumerate(self.subjects)]) for roi in ['V1_surround', 'V2_surround', 'V3_surround', 'V123_surround']])
        averages_av_snr_center_clean_l = np.array([np.array([np.array(data_frame[data_frame.subject == s][roi][self.pupil_l_ind[i]]).mean() / np.array(data_frame[data_frame.subject == s][roi][self.pupil_l_ind[i]]).std() for i, s in enumerate(self.subjects)]) for roi in ['V1', 'V2', 'V3', 'V123']])
        
        averages_av_snr_center_p = np.array([np.array([np.array(data_frame[data_frame.subject == s][roi][self.present[i]]).mean() / np.array(data_frame[data_frame.subject == s][roi][self.present[i]]).std() for i, s in enumerate(self.subjects)]) for roi in ['V1_center', 'V2_center', 'V3_center', 'V123_center']])
        averages_av_snr_surround_p = np.array([np.array([np.array(data_frame[data_frame.subject == s][roi][self.present[i]]).mean() / np.array(data_frame[data_frame.subject == s][roi][self.present[i]]).std() for i, s in enumerate(self.subjects)]) for roi in ['V1_surround', 'V2_surround', 'V3_surround', 'V123_surround']])
        averages_av_snr_center_clean_p = np.array([np.array([np.array(data_frame[data_frame.subject == s][roi][self.present[i]]).mean() / np.array(data_frame[data_frame.subject == s][roi][self.present[i]]).std() for i, s in enumerate(self.subjects)]) for roi in ['V1', 'V2', 'V3', 'V123']])
        
        averages_av_snr_center_a = np.array([np.array([np.array(data_frame[data_frame.subject == s][roi][self.absent[i]]).mean() / np.array(data_frame[data_frame.subject == s][roi][self.absent[i]]).std() for i, s in enumerate(self.subjects)]) for roi in ['V1_center', 'V2_center', 'V3_center', 'V123_center']])
        averages_av_snr_surround_a = np.array([np.array([np.array(data_frame[data_frame.subject == s][roi][self.absent[i]]).mean() / np.array(data_frame[data_frame.subject == s][roi][self.absent[i]]).std() for i, s in enumerate(self.subjects)]) for roi in ['V1_surround', 'V2_surround', 'V3_surround', 'V123_surround']])
        averages_av_snr_center_clean_a = np.array([np.array([np.array(data_frame[data_frame.subject == s][roi][self.absent[i]]).mean() / np.array(data_frame[data_frame.subject == s][roi][self.absent[i]]).std() for i, s in enumerate(self.subjects)]) for roi in ['V1', 'V2', 'V3', 'V123']])
        
        # barplots of (i) overall responses, (ii) present vs absent, (iii) high vs low, (iv) present vs absent scaled by pupil:
        titles = ['average', 'pupil_contrast_center', 'pupil_contrast_surround', 'pupil_contrast_center_clean', 'signal_contrast_center', 'signal_contrast_surround', 'signal_contrast_center_clean'] 
        measures = [
                    [averages_av_snr_center, averages_av_snr_surround], 
                    [averages_av_snr_center_h, averages_av_snr_center_l],
                    [averages_av_snr_surround_h, averages_av_snr_surround_l],
                    [averages_av_snr_center_clean_h, averages_av_snr_center_clean_l],
                    [averages_av_snr_center_p, averages_av_snr_center_a],
                    [averages_av_snr_surround_p, averages_av_snr_surround_a],
                    [averages_av_snr_center_clean_p, averages_av_snr_center_clean_a],
                    ]
        locations = ['mixed', 'center', 'surround', 'center', 'center', 'surround', 'center']
        ylims = [(-0.2,0.5), (0,0.7), (-0.2,0.1), (-0.1,0.1), (0,0.7), (-0.2,0.1), (-0.1,0.1)]
        for measure, t, l, ylim in zip(measures, titles, locations, ylims):
            
            p1 = myfuncs.permutationTest(measure[0][0], measure[1][0], paired=True)[1]
            p2 = myfuncs.permutationTest(measure[0][1], measure[1][1], paired=True)[1]
            p3 = myfuncs.permutationTest(measure[0][2], measure[1][2], paired=True)[1]
            p4 = myfuncs.permutationTest(measure[0][3], measure[1][3], paired=True)[1]
            stats = [p1, p2, p3, p4]
            
            MEANS_1 = np.array([d.mean() for d in measure[0]])
            MEANS_2 = np.array([d.mean() for d in measure[1]])
            SEMS_1 = np.array([sp.stats.sem(d) for d in measure[0]])
            SEMS_2 = np.array([sp.stats.sem(d) for d in measure[1]])
            my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}
            N = len(MEANS_1)
            ind = np.linspace(0,N,N)  # the x locations for the groups
            bar_width = 0.5   # the width of the bars
            fig = plt.figure(figsize=(N*0.75,3))
            ax = fig.add_subplot(111)
            if l == 'mixed':
                ax.bar(ind, MEANS_1, width = bar_width, yerr=SEMS_1, color='orange', alpha=1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'edge')
                ax.bar(ind+0.5, MEANS_2, width = bar_width, yerr=SEMS_2, color='green', alpha=1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'edge')
            if l == 'center':
                ax.bar(ind, MEANS_1, width = bar_width, yerr=SEMS_1, color='orange', alpha=1, edgecolor = 'r', ecolor = 'k', linewidth = 2, capsize = 0, align = 'edge')
                ax.bar(ind+0.5, MEANS_2, width = bar_width, yerr=SEMS_2, color='orange', alpha=1, edgecolor = 'b', ecolor = 'k', linewidth = 2, capsize = 0, align = 'edge')
            if l == 'surround':
                ax.bar(ind, MEANS_1, width = bar_width, yerr=SEMS_1, color='green', alpha=1, edgecolor = 'r', ecolor = 'k', linewidth = 2, capsize = 0, align = 'edge')
                ax.bar(ind+0.5, MEANS_2, width = bar_width, yerr=SEMS_2, color='green', alpha=1, edgecolor = 'b', ecolor = 'k', linewidth = 2, capsize = 0, align = 'edge')
            ax.set_xticks(ind+bar_width)
            ax.set_ylim(ylim)
            sns.despine(offset=10, trim=True)
            ax.set_xticklabels(['V1', 'V2', 'V3', 'V123'], rotation=45)
            ax.set_ylabel('SNR')
            for i, pp in enumerate(stats):
                ax.text(s=str(round(pp,3)), x=ind[i]+bar_width, y=ax.get_ylim()[1]-((ax.get_ylim()[1] - ax.get_ylim()[0]) / 10.0), size=10,)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, 'V123', 'snr', 'bars_{}_{}.pdf'.format(t, self.split_by)),)
        
        # subj_ind = np.ones(14, dtype=bool)
        # subj_ind[10] = False
        #
        # self.dprime = self.dprime[subj_ind]
        # self.dprime_lo = self.dprime_lo[subj_ind]
        # self.dprime_hi = self.dprime_hi[subj_ind]
        #
        # self.criterion = self.criterion[subj_ind]
        # self.criterion_lo = self.criterion_lo[subj_ind]
        # self.criterion_hi = self.criterion_hi[subj_ind]
        #
        # for measure in [self.dprime_hi-self.dprime_lo, self.criterion_hi-self.criterion_lo]:
        #     myfuncs.correlation_plot(averages_av_snr_center_clean_h[3,:]-averages_av_snr_center_clean_l[3,:], measure, line=True)
        
        
    def VISUAL_noise_correlation_ROI(self, data_type, partial=False):
        
        time_locked = 'stim_locked'
        self.make_pupil_BOLD_dataframe(data_type=data_type, time_locked=time_locked, regress_iti=False, regress_rt_signal=True)
        
        scalars = []
        for i in range(len(self.subjects)):
            s = []
            for roi in ['V1', 'V2', 'V3']:
                files = sort(glob.glob(os.path.join(self.data_folder, 'noise_correlation', 'scalars{}_{}*{}_center*'.format(self.postfix, self.subjects[i], roi))))
                s.append( np.concatenate([np.load(f) for f in files]) )
                files = sort(glob.glob(os.path.join(self.data_folder, 'noise_correlation', 'scalars{}_{}*{}_surround*'.format(self.postfix, self.subjects[i], roi))))
                s.append( np.concatenate([np.load(f) for f in files]) )
            scalars.append(np.hstack(s))
        scalars = np.vstack(scalars)
        
        nr_voxelss = scalars.shape[1] / 6
        boundaries = np.cumsum(np.repeat(nr_voxelss,5))
        
        scalars_V1_center = scalars[:,nr_voxelss*0:nr_voxelss*1].mean(axis=1)
        scalars_V1_surround = scalars[:,nr_voxelss*1:nr_voxelss*2].mean(axis=1)
        scalars_V2_center = scalars[:,nr_voxelss*2:nr_voxelss*3].mean(axis=1)
        scalars_V2_surround = scalars[:,nr_voxelss*3:nr_voxelss*4].mean(axis=1)
        scalars_V3_center = scalars[:,nr_voxelss*4:nr_voxelss*5].mean(axis=1)
        scalars_V3_surround = scalars[:,nr_voxelss*5:nr_voxelss*6].mean(axis=1)
        
        scalars = np.vstack((scalars_V1_center, scalars_V1_surround, scalars_V2_center, scalars_V2_surround, scalars_V3_center, scalars_V3_surround)).T
        
        contrast_across_dvs = []
        # for dv in ['cor', 'mean', 'var', 'cov', 'snr']:
        for dv in ['cor',]:
            cormats = []
            for condition in [np.concatenate(self.pupil_h_ind), np.concatenate(self.pupil_l_ind)]:
                for signal_present in [np.concatenate(self.present), ~np.concatenate(self.present)]:
                    ind = condition * signal_present
                    cm = np.zeros((scalars.shape[1], scalars.shape[1], len(self.subjects)))
                    for i in range(len(self.subjects)):
                        C = scalars[np.array(self.data_frame.subject == self.subjects[i])*ind,:]
                        if dv == 'cor':
                            if partial:
                                cm[:,:,i], dummy = myfuncs.corr_matrix_partial(C)
                            else:
                                cm[:,:,i], dummy = myfuncs.corr_matrix(C, dv='cor')
                        if dv == 'var':
                            cm[:,:,i], dummy = myfuncs.corr_matrix(C, dv='var')
                        if dv == 'cov':
                            cm[:,:,i], dummy = myfuncs.corr_matrix(C, dv='cov')
                        if dv == 'mean':
                            cm[:,:,i], dummy = myfuncs.corr_matrix(C, dv='mean')
                        if dv == 'snr':
                            cm[:,:,i], dummy = myfuncs.corr_matrix(C, dv='snr')
                    cormats.append(cm)
            
            corrmats_mean = (cormats[0]+cormats[1]+cormats[2]+cormats[3]) / 4.0
            corrmats_mean_av = corrmats_mean.mean(axis=-1)
            corrmats_contrast = ((cormats[0]-cormats[2]) + (cormats[1]-cormats[3])) / 2.0
            corrmats_contrast_av = corrmats_contrast.mean(axis=-1)
            
            contrast_across_dvs.append(corrmats_contrast_av)
            
            p_mat = np.zeros(corrmats_mean_av.shape)
            for i in range(p_mat.shape[0]):
                for j in range(p_mat.shape[1]):
                    try:
                        p_mat[i,j] = sp.stats.wilcoxon(corrmats_contrast[i,j,:])[1]
                    except:
                        p_mat[i,j] = 1
            
            # fdr correction:
            p_mat = mne.stats.fdr_correction(p_mat, 0.05)[1]

            # plot all matrix:
            mask =  np.tri(corrmats_mean_av.shape[0], k=0)
            corrmat_m = np.ma.masked_where(mask, corrmats_mean_av)
            p_mat_m = np.ma.masked_where(mask.T, p_mat)
            fig = plt.figure(figsize=(5,4))
            ax = fig.add_subplot(111)
            vmax = max(abs(corrmats_mean_av.ravel()))
            if (dv == 'cor') or (dv == 'mean') or (dv == 'snr'):
                im = ax.pcolormesh(corrmat_m, cmap='bwr', vmin=-vmax, vmax=vmax)
            else:
                im = ax.pcolormesh(corrmat_m, cmap='Reds', vmin=0, vmax=vmax)
            ax.set_xlim(xmax=scalars.shape[1])
            ax.set_ylim(ymax=scalars.shape[1])
            ax.set_yticks(arange(0.5,scalars.shape[1]+.5))
            ax.set_xticks(arange(0.5,scalars.shape[1]+.5))
            ax.patch.set_hatch('x')
            ax.set_aspect(1)
            fig.colorbar(im)
            for l in boundaries:
                plt.vlines(l, 0, l, lw=1)
                plt.hlines(l, l, corrmat_m.shape[0], lw=1)
            plt.tight_layout()
            if partial:
                fig.savefig(os.path.join(self.figure_folder, 'V123', 'noise_correlation', 'matrix_single_partial_{}_{}_{}.pdf'.format('all', dv, self.split_by)))
            else:
                fig.savefig(os.path.join(self.figure_folder, 'V123', 'noise_correlation', 'matrix_single_{}_{}_{}_ROI.pdf'.format('all', dv, self.split_by)))

            # contrast matrix:
            mask =  np.tri(corrmats_contrast_av.shape[0], k=0)
            corrmat_m = np.ma.masked_where(mask, corrmats_contrast_av)
            p_mat_m = np.ma.masked_where(mask.T, p_mat)
            fig = plt.figure(figsize=(5,4))
            # fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(111)
            vmax = max(abs(corrmats_contrast_av.ravel()))
            im = ax.pcolormesh(corrmat_m, cmap='bwr', vmin=-vmax, vmax=vmax)
            ax.set_xlim(xmax=scalars.shape[1])
            ax.set_ylim(ymax=scalars.shape[1])
            ax.set_yticks(arange(0.5,scalars.shape[1]+.5))
            ax.set_xticks(arange(0.5,scalars.shape[1]+.5))
            ax.set_xticklabels(['V1_c', 'V1_s', 'V2_c', 'V2_s', 'V3_c', 'V3_s'])
            ax.set_yticklabels(['V1_c', 'V1_s', 'V2_c', 'V2_s', 'V3_c', 'V3_s'])
            ax.patch.set_hatch('x')
            ax.set_aspect(1)
            fig.colorbar(im)
            for l in boundaries:
                plt.vlines(l, 0, l, lw=1)
                plt.hlines(l, l, corrmat_m.shape[0], lw=1)
            plt.tight_layout()
            if partial:
                fig.savefig(os.path.join(self.figure_folder, 'V123', 'noise_correlation', 'matrix_single_partial_{}_{}_{}.pdf'.format('contrast', dv, self.split_by)))
            else:
                fig.savefig(os.path.join(self.figure_folder, 'V123', 'noise_correlation', 'matrix_single_{}_{}_{}_ROI.pdf'.format('contrast', dv, self.split_by)))
            
            
            center_center = [(2,0), (4,0), (4,2)]
            values = np.vstack([corrmats_contrast[ii[0],ii[1],:] for ii in center_center]).mean(axis=0)
            
            surround_surround = [(3,1), (5,1), (5,3)]
            values = np.vstack([corrmats_contrast[ii[0],ii[1],:] for ii in surround_surround]).mean(axis=0)
            
            center_surround = [(1,0), (3,0), (5,0), (2,1), (4,1), (3,2), (5,2), (4,3), (5,4)]
            values = np.vstack([corrmats_contrast[ii[0],ii[1],:] for ii in center_surround]).mean(axis=0)
            
    def VISUAL_noise_correlation(self, data_type, partial=False):
        
        nr_voxels = 100
        # self.postfix = '_d'
        
        time_locked = 'stim_locked'
        self.make_pupil_BOLD_dataframe(data_type=data_type, time_locked=time_locked, regress_iti=False, regress_rt=True, regress_stimulus=True,)
        
        scalars = []
        for i in range(len(self.subjects)):
            s = []
            for roi in ['V1', 'V2', 'V3']:
                files = sort(glob.glob(os.path.join(self.data_folder, 'noise_correlation', 'scalars{}_{}*{}_{}_center*'.format(self.postfix, self.subjects[i], roi, nr_voxels))))
                s.append( np.concatenate([np.load(f) for f in files]) )
                files = sort(glob.glob(os.path.join(self.data_folder, 'noise_correlation', 'scalars{}_{}*{}_{}_surround*'.format(self.postfix, self.subjects[i], roi, nr_voxels))))
                s.append( np.concatenate([np.load(f) for f in files]) )
            scalars.append(np.hstack(s))
        scalars = np.vstack(scalars)
        
        boundaries = np.cumsum(np.repeat(nr_voxels,5))
        
        contrast_across_dvs = []
        # for dv in ['cor', 'mean', 'var', 'cov', 'snr']:
        for dv in ['cor',]:
            cormats = []
            for condition in [np.concatenate(self.pupil_h_ind), np.concatenate(self.pupil_l_ind)]:
                for signal_present in [np.concatenate(self.present), ~np.concatenate(self.present)]:
                    ind = condition * signal_present
                    cm = np.zeros((nr_voxels*6, nr_voxels*6, len(self.subjects)))
                    for i in range(len(self.subjects)):
                        C = scalars[np.array(self.data_frame.subject == self.subjects[i])*ind,:]
                        if dv == 'cor':
                            if partial:
                                cm[:,:,i], dummy = myfuncs.corr_matrix_partial(C)
                            else:
                                cm[:,:,i], dummy = myfuncs.corr_matrix(C, dv='cor')
                        if dv == 'var':
                            cm[:,:,i], dummy = myfuncs.corr_matrix(C, dv='var')
                        if dv == 'cov':
                            cm[:,:,i], dummy = myfuncs.corr_matrix(C, dv='cov')
                        if dv == 'mean':
                            cm[:,:,i], dummy = myfuncs.corr_matrix(C, dv='mean')
                        if dv == 'snr':
                            cm[:,:,i], dummy = myfuncs.corr_matrix(C, dv='snr')
                    cormats.append(cm)
            
            corrmats_mean = (cormats[0]+cormats[1]+cormats[2]+cormats[3]) / 4.0
            corrmats_mean_av = corrmats_mean.mean(axis=-1)
            corrmats_contrast = ((cormats[0]-cormats[2]) + (cormats[1]-cormats[3])) / 2.0
            corrmats_contrast_av = corrmats_contrast.mean(axis=-1)
            
            contrast_across_dvs.append(corrmats_contrast_av)
            
            p_mat = np.zeros(corrmats_mean_av.shape)
            for i in range(p_mat.shape[0]):
                for j in range(p_mat.shape[1]):
                    try:
                        p_mat[i,j] = sp.stats.wilcoxon(corrmats_contrast[i,j,:])[1]
                    except:
                        p_mat[i,j] = 1
            
            # fdr correction:
            p_mat = mne.stats.fdr_correction(p_mat, 0.05)[1]

            # plot all matrix:
            mask =  np.tri(corrmats_mean_av.shape[0], k=0)
            corrmat_m = np.ma.masked_where(mask, corrmats_mean_av)
            p_mat_m = np.ma.masked_where(mask.T, p_mat)
            fig = plt.figure(figsize=(5,4))
            ax = fig.add_subplot(111)
            vmax = max(abs(corrmats_mean_av.ravel()))
            if (dv == 'cor') or (dv == 'mean') or (dv == 'snr'):
                im = ax.pcolormesh(corrmat_m, cmap='bwr', vmin=-vmax, vmax=vmax)
            else:
                im = ax.pcolormesh(corrmat_m, cmap='Reds', vmin=0, vmax=vmax)
            ax.set_xlim(xmax=(nr_voxels*6))
            ax.set_ylim(ymax=(nr_voxels*6))
            ax.set_yticks(arange(0.5,(nr_voxels*6)+.5))
            ax.set_xticks(arange(0.5,(nr_voxels*6)+.5))
            ax.patch.set_hatch('x')
            ax.set_aspect(1)
            fig.colorbar(im)
            for l in boundaries:
                plt.vlines(l, 0, l, lw=1)
                plt.hlines(l, l, corrmat_m.shape[0], lw=1)
            plt.tight_layout()
            if partial:
                fig.savefig(os.path.join(self.figure_folder, 'V123', 'noise_correlation', 'matrix_single_partial_{}_{}_{}.pdf'.format('all', dv, self.split_by)))
            else:
                fig.savefig(os.path.join(self.figure_folder, 'V123', 'noise_correlation', 'matrix_single_{}_{}_{}.pdf'.format('all', dv, self.split_by)))

            # contrast matrix:
            mask =  np.tri(corrmats_contrast_av.shape[0], k=0)
            corrmat_m = np.ma.masked_where(mask, corrmats_contrast_av)
            p_mat_m = np.ma.masked_where(mask.T, p_mat)
            fig = plt.figure(figsize=(5,4))
            # fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(111)
            vmax = max(abs(corrmats_contrast_av.ravel()))
            im = ax.pcolormesh(corrmat_m, cmap='bwr', vmin=-vmax, vmax=vmax)
            ax.set_xlim(xmax=(nr_voxels*6))
            ax.set_ylim(ymax=(nr_voxels*6))
            ax.set_yticks(arange(0.5,(nr_voxels*6)+.5))
            ax.set_xticks(arange(0.5,(nr_voxels*6)+.5))
            ax.patch.set_hatch('x')
            ax.set_aspect(1)
            fig.colorbar(im)
            for l in boundaries:
                plt.vlines(l, 0, l, lw=1)
                plt.hlines(l, l, corrmat_m.shape[0], lw=1)
            plt.tight_layout()
            if partial:
                fig.savefig(os.path.join(self.figure_folder, 'V123', 'noise_correlation', 'matrix_single_partial_{}_{}_{}.pdf'.format('contrast', dv, self.split_by)))
            else:
                fig.savefig(os.path.join(self.figure_folder, 'V123', 'noise_correlation', 'matrix_single_{}_{}_{}.pdf'.format('contrast', dv, self.split_by)))
            
            # # stats on cells:
            # for r in np.concatenate((np.array([0]), boundaries)):
            #
            #     r_h = np.zeros(len(self.subjects))
            #     r_l = np.zeros(len(self.subjects))
            #     for i in range(len(self.subjects)):
            #         cm_h = (cormats[0][:,:,i]+cormats[1][:,:,i])/2.0
            #         cm_l = (cormats[2][:,:,i]+cormats[3][:,:,i])/2.0
            #         r_h[i] = np.mean(cm_h[r:r+nr_voxels,r:r+nr_voxels].ravel())
            #         r_l[i] = np.mean(cm_l[r:r+nr_voxels,r:r+nr_voxels].ravel())
            #
            #     MEANS = (r_h.mean(), r_l.mean())
            #     SEMS = (sp.stats.sem(r_h), sp.stats.sem(r_l))
            #     N = 2
            #     ind = np.linspace(0,N/2,N)
            #     bar_width = 0.50
            #     fig = plt.figure(figsize=(1.25,1.75))
            #     ax = fig.add_subplot(111)
            #     for i in range(N):
            #         ax.bar(ind[i], height=MEANS[i], width = bar_width, yerr = SEMS[i], color = ['r','b'][i], alpha = 1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
            #     ax.set_title('p = {}'.format(round(myfuncs.permutationTest(r_h, r_l, paired=False)[1],3)), size=7)
            #     ax.set_ylabel('number of cells', size=7)
            #     ax.set_xticks( (ind) )
            #     ax.set_xticklabels( ('H', 'L') )
            #     sns.despine(offset=10, trim=True)
            #     plt.tight_layout()
            #     fig.savefig(os.path.join(self.figure_folder, 'V123', 'noise_correlation', 'matrix_single_{}_{}_{}_nr_cells_{}.pdf'.format('contrast', dv, self.split_by, r)))
            #
            # pos_ind = np.array(np.concatenate((np.ones(nr_voxels), np.zeros(nr_voxels), np.ones(nr_voxels), np.zeros(nr_voxels), np.ones(nr_voxels), np.zeros(nr_voxels),)), dtype=bool)
            #
            # r_h = np.zeros(len(self.subjects))
            # r_l = np.zeros(len(self.subjects))
            # for i in range(len(self.subjects)):
            #     cm_h = (cormats[0][:,:,i]+cormats[1][:,:,i])/2.0
            #     cm_l = (cormats[2][:,:,i]+cormats[3][:,:,i])/2.0
            #     r_h[i] = np.mean(cm_h[:,pos_ind][pos_ind,:].ravel())
            #     r_l[i] = np.mean(cm_l[:,pos_ind][pos_ind,:].ravel())
            #
            # MEANS = (r_h.mean(), r_l.mean())
            # SEMS = (sp.stats.sem(r_h), sp.stats.sem(r_l))
            # N = 2
            # ind = np.linspace(0,N/2,N)
            # bar_width = 0.50
            # fig = plt.figure(figsize=(1.25,1.75))
            # ax = fig.add_subplot(111)
            # for i in range(N):
            #     ax.bar(ind[i], height=MEANS[i], width = bar_width, yerr = SEMS[i], color = ['r','b'][i], alpha = 1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
            # ax.set_title('p = {}'.format(round(myfuncs.permutationTest(r_h, r_l, paired=False)[1],3)), size=7)
            # ax.set_ylabel('number of cells', size=7)
            # ax.set_xticks( (ind) )
            # ax.set_xticklabels( ('incr', 'decr') )
            # sns.despine(offset=10, trim=True)
            # plt.tight_layout()
            # fig.savefig(os.path.join(self.figure_folder, 'V123', 'noise_correlation', 'matrix_single_{}_{}_{}_nr_cells_{}.pdf'.format('contrast', dv, self.split_by, 'pos')))
            #
            # r_h = np.zeros(len(self.subjects))
            # r_l = np.zeros(len(self.subjects))
            # for i in range(len(self.subjects)):
            #     cm_h = (cormats[0][:,:,i]+cormats[1][:,:,i])/2.0
            #     cm_l = (cormats[2][:,:,i]+cormats[3][:,:,i])/2.0
            #     r_h[i] = np.mean(cm_h[:,~pos_ind][~pos_ind,:].ravel())
            #     r_l[i] = np.mean(cm_l[:,~pos_ind][~pos_ind,:].ravel())
            #
            # MEANS = (r_h.mean(), r_l.mean())
            # SEMS = (sp.stats.sem(r_h), sp.stats.sem(r_l))
            # N = 2
            # ind = np.linspace(0,N/2,N)
            # bar_width = 0.50
            # fig = plt.figure(figsize=(1.25,1.75))
            # ax = fig.add_subplot(111)
            # for i in range(N):
            #     ax.bar(ind[i], height=MEANS[i], width = bar_width, yerr = SEMS[i], color = ['r','b'][i], alpha = 1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
            # ax.set_title('p = {}'.format(round(myfuncs.permutationTest(r_h, r_l, paired=False)[1],3)), size=7)
            # ax.set_ylabel('number of cells', size=7)
            # ax.set_xticks( (ind) )
            # ax.set_xticklabels( ('incr', 'decr') )
            # sns.despine(offset=10, trim=True)
            # plt.tight_layout()
            # fig.savefig(os.path.join(self.figure_folder, 'V123', 'noise_correlation', 'matrix_single_{}_{}_{}_nr_cells_{}.pdf'.format('contrast', dv, self.split_by, 'neg')))
            
            # stats on cells:
            for r in np.concatenate((np.array([0]), boundaries)):

                nr_pos_cells = np.zeros(len(self.subjects))
                nr_neg_cells = np.zeros(len(self.subjects))
                for i in range(len(self.subjects)):
                    mask =  np.tri(corrmats_contrast_av.shape[0], k=0)
                    cm = corrmats_contrast[:,:,i]
                    nr_pos_cells[i] = np.sum(cm[r:r+nr_voxels,r:r+nr_voxels].ravel()>0)
                    nr_neg_cells[i] = np.sum(cm[r:r+nr_voxels,r:r+nr_voxels].ravel()<0)
                pos_cells = nr_pos_cells / (nr_pos_cells+nr_neg_cells) * 100
                neg_cells = nr_neg_cells / (nr_pos_cells+nr_neg_cells) * 100

                MEANS = (pos_cells.mean(), neg_cells.mean())
                SEMS = (sp.stats.sem(pos_cells), sp.stats.sem(neg_cells))
                N = 2
                ind = np.linspace(0,N/2,N)
                bar_width = 0.50
                fig = plt.figure(figsize=(1.25,1.75))
                ax = fig.add_subplot(111)
                for i in range(N):
                    ax.bar(ind[i], height=MEANS[i], width = bar_width, yerr = SEMS[i], color = ['r','b'][i], alpha = 1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
                ax.set_title('p = {}'.format(round(myfuncs.permutationTest(pos_cells, np.repeat(50,len(self.subjects)), paired=True)[1],3)), size=7)
                ax.set_ylabel('number of cells', size=7)
                ax.set_xticks( (ind) )
                ax.set_xticklabels( ('incr', 'decr') )
                sns.despine(offset=10, trim=True)
                plt.tight_layout()
                fig.savefig(os.path.join(self.figure_folder, 'V123', 'noise_correlation', 'matrix_single_{}_{}_{}_nr_cells_{}.pdf'.format('contrast', dv, self.split_by, r)))

            pos_ind = np.array(np.concatenate((np.ones(nr_voxels), np.zeros(nr_voxels), np.ones(nr_voxels), np.zeros(nr_voxels), np.ones(nr_voxels), np.zeros(nr_voxels),)), dtype=bool)

            nr_pos_cells = np.zeros(len(self.subjects))
            nr_neg_cells = np.zeros(len(self.subjects))
            for i in range(len(self.subjects)):
                mask =  np.tri(corrmats_contrast_av.shape[0], k=0)
                cm = corrmats_contrast[:,:,i]
                nr_pos_cells[i] = np.sum(cm[:,pos_ind][pos_ind,:].ravel()>0)
                nr_neg_cells[i] = np.sum(cm[:,pos_ind][pos_ind,:].ravel()<0)
            pos_cells = nr_pos_cells / (nr_pos_cells+nr_neg_cells) * 100
            neg_cells = nr_neg_cells / (nr_pos_cells+nr_neg_cells) * 100

            MEANS = (pos_cells.mean(), neg_cells.mean())
            SEMS = (sp.stats.sem(pos_cells), sp.stats.sem(neg_cells))
            N = 2
            ind = np.linspace(0,N/2,N)
            bar_width = 0.50
            fig = plt.figure(figsize=(1.25,1.75))
            ax = fig.add_subplot(111)
            for i in range(N):
                ax.bar(ind[i], height=MEANS[i], width = bar_width, yerr = SEMS[i], color = ['r','b'][i], alpha = 1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
            ax.set_title('p = {}'.format(round(myfuncs.permutationTest(pos_cells, np.repeat(50,len(self.subjects)), paired=True)[1],3)), size=7)
            ax.set_ylabel('number of cells', size=7)
            ax.set_xticks( (ind) )
            ax.set_xticklabels( ('incr', 'decr') )
            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, 'V123', 'noise_correlation', 'matrix_single_{}_{}_{}_nr_cells_{}.pdf'.format('contrast', dv, self.split_by, 'pos')))

            nr_pos_cells = np.zeros(len(self.subjects))
            nr_neg_cells = np.zeros(len(self.subjects))
            for i in range(len(self.subjects)):
                mask =  np.tri(corrmats_contrast_av.shape[0], k=0)
                cm = corrmats_contrast[:,:,i]
                nr_pos_cells[i] = np.sum(cm[:,~pos_ind][~pos_ind,:].ravel()>0)
                nr_neg_cells[i] = np.sum(cm[:,~pos_ind][~pos_ind,:].ravel()<0)
            pos_cells = nr_pos_cells / (nr_pos_cells+nr_neg_cells) * 100
            neg_cells = nr_neg_cells / (nr_pos_cells+nr_neg_cells) * 100

            MEANS = (pos_cells.mean(), neg_cells.mean())
            SEMS = (sp.stats.sem(pos_cells), sp.stats.sem(neg_cells))
            N = 2
            ind = np.linspace(0,N/2,N)
            bar_width = 0.50
            fig = plt.figure(figsize=(1.25,1.75))
            ax = fig.add_subplot(111)
            for i in range(N):
                ax.bar(ind[i], height=MEANS[i], width = bar_width, yerr = SEMS[i], color = ['r','b'][i], alpha = 1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
            ax.set_title('p = {}'.format(round(myfuncs.permutationTest(pos_cells, np.repeat(50,len(self.subjects)), paired=True)[1],3)), size=7)
            ax.set_ylabel('number of cells', size=7)
            ax.set_xticks( (ind) )
            ax.set_xticklabels( ('incr', 'decr') )
            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, 'V123', 'noise_correlation', 'matrix_single_{}_{}_{}_nr_cells_{}.pdf'.format('contrast', dv, self.split_by, 'neg')))
                
    def MULTIVARIATE_plot_patterns(self, data_type, measure='mean', prepare=False):
        
        import matplotlib.gridspec as gridspec
        
        partial = False
        time_locked = 'stim_locked'
        type_response = self.split_by.split('_')[-1]
        cortex_mask = np.array(NiftiImage('/home/shared/Niels_UvA/brainstem_masks/harvard_oxford/volume_1.nii.gz').data, dtype=bool) + np.array(NiftiImage('/home/shared/Niels_UvA/brainstem_masks/harvard_oxford/volume_12.nii.gz').data, dtype=bool)
        epi_box = np.array(NiftiImage(os.path.join('/home/shared/Niels_UvA/brainstem_masks/', 'epi_box.nii.gz')).data, dtype=bool)
        epi_box = epi_box * cortex_mask
        task_network = NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'trials_all_clean_MNI_smooth_stim_locked_mean_pupil_d_0_psc_all_tfce_corrp_tstat1.nii.gz')).data
        task_network = task_network>0.99
        
        self.make_pupil_BOLD_dataframe(data_type='clean_False', time_locked=time_locked, regress_iti=False, regress_rt=False)
        
        scores_45 = np.zeros((len(self.subjects),2))
        scores_135 = np.zeros((len(self.subjects),2))

        mean_pattern_45 = np.zeros((len(self.subjects),2))
        mean_pattern_135 = np.zeros((len(self.subjects),2))

        rois = ['V1_center']
        ind_type = 'ori'

        gs = gridspec.GridSpec(28, 2, width_ratios=[1,1], height_ratios=np.concatenate([ (7,1) for _ in range(len(self.subjects))])) 
        for orii in [45, 135]:
            fig = plt.figure(figsize=(8, 40)) 
            plt_nr = 0
            for i in range(len(self.subjects)):
                print
                print self.subjects[i]
                df = {}
                for r in rois:
                    # for nr_voxels in np.array(np.linspace(10,150,15), dtype=int):
                    for nr_voxels in [50]:
            
                        # multivariate files:
                        files = sort(glob.glob(os.path.join(self.data_folder, 'V123', 'scalars{}_{}_{}_{}_{}_*.npy'.format(self.postfix, r, ind_type, nr_voxels, self.subjects[i],))))
                        scalars_m = [np.load(f) for f in files]
                
                        # templates:
                        files = sort(glob.glob(os.path.join(self.data_folder, 'V123', 'template{}_{}_{}_{}_{}_*.npy'.format(self.postfix, r, ind_type, nr_voxels, self.subjects[i],))))
                        template_diff = [np.load(f) for f in files]
                
                        # full patterns:
                        mean_pattern = []
                        line = []
                        for s in range(2):
                
                            # pupil data:
                            pupil_data = pd.read_csv(os.path.join(self.data_folder[:-6], self.subjects[i], 'pupil_data_{}.csv'.format(s+1)))
                            pupil_data = pupil_data[-np.array(pupil_data.omissions)]
                    
                            nr_trials_in_session = scalars_m[s].shape[0]
                            ori = np.array(pupil_data['signal_orientation'])
                            present = np.array(pupil_data['present'])
                            correct = np.array(pupil_data['correct'])
                    
                            # line:
                            l = np.argmax(template_diff[s] < 0)
                            line.append( l )
                    
                            # subtract out global response:
                            a = scalars_m[s]
                            # for v in range(nr_voxels):
                            #     a[:,v] = a[:,v] / a[:,v].std() # within each voxel, normalize variance across trials
                            #     # a[:,v] = (a[:,v] - a[:,v].mean()) / a[:,v].std() # within each voxel, z-score across trials
                            for t in range(nr_trials_in_session):
                                to_subtract = np.mean((a[t,:l].mean(), a[t,l:].mean()))
                                a[t,:] = (a[t,:] - to_subtract) # within each trial, set mean across voxels to 0
                    
                            # scores:
                            scores_45[i,s] = np.sum(correct * (ori==45)) / np.sum((ori==45))
                            scores_135[i,s] = np.sum(correct * (ori==135)) / np.sum((ori==135))
                    
                            # scores:
                            mean_pattern_45[i,s] = (a.mean(axis=0)[:nr_voxels/2]).mean()
                            mean_pattern_135[i,s] = (a.mean(axis=0)[nr_voxels/2:]).mean()
                    
                            # subsample trials:
                            a = a[present&(ori==orii),:]
                    
                            # compute mean pattern:
                            # mean_pattern.append(a.std(axis=0))
                            mean_pattern.append(a.mean(axis=0))
                    
                            # plot:
                            ax = plt.subplot(gs[plt_nr])
                            ax.pcolormesh(a, cmap='bwr', vmin=-10, vmax=10)
                            plt.axvline(line[s], color='k')
                            ax.set_title('{}, session {}'.format(self.subjects[i], s+1))
                            ax.set_ylabel('trial #')
                            ax.set_xlabel('voxel #')
                            plt_nr += 1
                    
                        # mean patterns:
                        for s in range(2):
                
                            ax = plt.subplot(gs[plt_nr])
                            # im = ax.pcolormesh(np.atleast_2d(np.arange(nr_voxels+1)), np.atleast_2d(np.array([0,1])).T, np.atleast_2d(mean_pattern[s]), cmap='bwr',) #vmin=-3, vmax=3)
                            # fig.colorbar(im, ax=ax, fraction=0.1, pad=0.1)
                            if orii == 45:
                                ax.pcolormesh(np.atleast_2d(np.arange(nr_voxels+1)), np.atleast_2d(np.array([0,1,2])).T, np.vstack((template_diff[s], mean_pattern[s])), cmap='bwr', vmin=-4, vmax=4)
                            if orii == 135:
                                ax.pcolormesh(np.atleast_2d(np.arange(nr_voxels+1)), np.atleast_2d(np.array([0,1,2])).T, np.vstack((template_diff[s]*-1.0, mean_pattern[s])), cmap='bwr', vmin=-4, vmax=4)
                            plt.axvline(line[s], color='k')
                            ax.set_yticks([0.5, 1.5])
                            ax.set_yticklabels(['template', 'mean'])
                            ax.set_title('{}, session {}'.format(self.subjects[i], s+1))
                            ax.set_xlabel('voxel #')
                            plt_nr += 1
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, 'lateralization', 'patterns_{}.jpg'.format(orii)), dpi=150)
            
        sp.stats.pearsonr(scores_45[:,0]-scores_135[:,0], mean_pattern_45[:,0]-mean_pattern_135[:,0])
        sp.stats.pearsonr(scores_45[:,1]-scores_135[:,1], mean_pattern_45[:,1]-mean_pattern_135[:,1])
        
    
    def MULTIVARIATE_make_lineplot(self, data_type, measure='mean', prepare=False):
        
        partial = False
        time_locked = 'stim_locked'
        type_response = self.split_by.split('_')[-1]
        cortex_mask = np.array(NiftiImage('/home/shared/Niels_UvA/brainstem_masks/harvard_oxford/volume_1.nii.gz').data, dtype=bool) + np.array(NiftiImage('/home/shared/Niels_UvA/brainstem_masks/harvard_oxford/volume_12.nii.gz').data, dtype=bool)
        epi_box = np.array(NiftiImage(os.path.join('/home/shared/Niels_UvA/brainstem_masks/', 'epi_box.nii.gz')).data, dtype=bool)
        epi_box = epi_box * cortex_mask
        task_network = NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'trials_all_clean_MNI_smooth_stim_locked_mean_pupil_d_0_psc_all_tfce_corrp_tstat1.nii.gz')).data
        task_network = task_network>0.99
        
        self.make_pupil_BOLD_dataframe(data_type='clean_False', time_locked=time_locked, regress_iti=False, regress_rt_signal=False)
        
        # shell()
        
        # params:
        v = 3
        # rois = ['V1', 'V2', 'V3']
        # rois = ['V1', 'V2',]
        rois = ['V1_center', 'V2_center', 'V3_center']
        # rois = ['V1',]
        nr_voxels_for_V123 = 50
        
        # create data_frame:
        # ------------------
        dfs = []
        
        ind_type = 'ori'
        
        
        for i in range(len(self.subjects)):
            print
            print self.subjects[i]
            df = {}
            
            for r in rois:
                for nr_voxels in np.array(np.linspace(10,150,15), dtype=int):
                # for nr_voxels in [30]:
                    
                    # multivariate files:
                    files = sort(glob.glob(os.path.join(self.data_folder, 'V123', 'scalars{}_{}_{}_{}_{}_*.npy'.format(self.postfix, r, ind_type, nr_voxels, self.subjects[i],))))
                    scalars_m = [np.load(f) for f in files]
                    
                    # # templates:
                    # files = sort(glob.glob(os.path.join(self.data_folder, 'V123', 'template{}_{}_{}_{}_{}_*.npy'.format(self.postfix, r, ind_type, nr_voxels, self.subjects[i],))))
                    # template_diff = [np.load(f) for f in files]
                    
                    # project onto template:
                    mean_output = []
                    weigthed_mean_output = []
                    signal_orientation = []
                    session = []
                    trials = 0
                    for s in range(len(scalars_m)):
                        
                        # pupil data:
                        pupil_data = pd.read_csv(os.path.join(self.data_folder[:-6], self.subjects[i], 'pupil_data_{}.csv'.format(s+1)))
                        pupil_data = pupil_data[-np.array(pupil_data.omissions)]
                        
                        nr_trials_in_session = scalars_m[s].shape[0]
                        sig_orientation = np.array(pupil_data['signal_orientation'])
                        present = np.array(pupil_data['present'])
                        
                        # line:
                        # line = np.argmax(template_diff[s] < 0)
                        line = scalars_m[s].shape[1]/2
                        
                        # normalize:
                        # scalars_m[s][:,:] = np.array( (scalars_m[s][:,:] - np.matrix(scalars_m[s][:,:].mean(axis=0))) / np.matrix(scalars_m[s][:,:].std(axis=0)) )
                        # for t in range(nr_trials_in_session):
                            # to_subtract = np.mean((scalars_m[s][t,:line].mean(), scalars_m[s][t,line:].mean()))
                            # scalars_m[s][t,:] = scalars_m[s][t,:] - to_subtract
                            
                        # template_cw[s] = (template_cw[s] - template_cw[s].mean() ) #/ template_cw[s].std()
                        # template_ccw[s] = (template_ccw[s] - template_ccw[s].mean() ) #/ template_ccw[s].std()
                        # template_diff[s] = (template_diff[s] - template_diff[s].mean() ) #/ template_diff[s].std()
                        # for vv in range(nr_voxels):
                            # scalars_m[s][:,vv] = (scalars_m[s][:,vv] - scalars_m[s][:,vv].mean()) / scalars_m[s][:,vv].std()
                        # template_cw[s] = (template_cw[s] - template_cw[s].mean() ) / template_cw[s].std()
                        
                        # MEAN SCALARS:
                        output_s_mean = np.zeros(nr_trials_in_session)
                        for t in range(nr_trials_in_session):
                            output_s_mean[t] = scalars_m[s][t,:line].mean() - scalars_m[s][t,line:].mean()
                            if sig_orientation[t] == 45:
                                output_s_mean[t] = scalars_m[s][t,:line].mean() - scalars_m[s][t,line:].mean()
                            else:
                                output_s_mean[t] = scalars_m[s][t,line:].mean() - scalars_m[s][t,:line].mean()
                                
                        # MULTIVARIATE SCALARS:
                        output_s = np.zeros(nr_trials_in_session)
                        for t in range(nr_trials_in_session):
                            
                            trials_for_template = np.ones(nr_trials_in_session, dtype=bool)
                            trials_for_template[t] = False
                            trials_for_template[~present] = False
                            template = scalars_m[s][trials_for_template&(sig_orientation==45),:].mean(axis=0) - scalars_m[s][trials_for_template&(sig_orientation==135),:].mean(axis=0)
                            
                            if sig_orientation[t] == 45:
                                scalars = scalars_m[s][t,:]
                                # output_s[t] = np.dot( template, scalars) / len( template )
                                # output_s[t] = np.dot( template, scalars) / np.dot( template, template)
                                output_s[t] = sp.stats.pearsonr(template, scalars)[0]
                            if sig_orientation[t] == 135:
                                scalars = scalars_m[s][t,:] * -1.0
                                # output_s[t] = np.dot( template, scalars) / len( template )
                                # output_s[t] = np.dot( template, scalars) / np.dot( template, template)
                                output_s[t] = sp.stats.pearsonr(template, scalars)[0]
                        
                        weigthed_mean_output.append(output_s)
                        mean_output.append(output_s_mean)
                        signal_orientation.append(sig_orientation)
                        session.append(np.ones(len(sig_orientation))*s)
                        trials = trials+nr_trials_in_session
                        
                    weigthed_mean_output = np.concatenate(weigthed_mean_output)
                    mean_output = np.concatenate(mean_output)
                    
                    
                    df['{}_{}_info'.format(r, nr_voxels)] = weigthed_mean_output
                    df['{}_{}_mean'.format(r, nr_voxels)] = mean_output
                    df['ori'.format(r, nr_voxels)] = np.concatenate(signal_orientation)
                    df['session'.format(r, nr_voxels)] = np.concatenate(session)
                    
            df = pd.DataFrame(df)
            df['subject'] = self.subjects[i]
            
            dfs.append(df)
        data_frame = pd.concat(dfs)
        
        
        # for s in self.subjects:
        #     print
        #     print s
        #     print 'session1: {}'.format(data_frame['V1_center_100_info'][np.array(data_frame.subject == s) & np.array(data_frame.ori == 135) & np.array(data_frame.session == 0)].mean())
        #     print 'session2: {}'.format(data_frame['V1_center_100_info'][np.array(data_frame.subject == s) & np.array(data_frame.ori == 45) & np.array(data_frame.session == 1)].mean())
        #     # print 'session1: {}'.format(data_frame['V1_100_mean'][np.array(data_frame.subject == s) & np.array(data_frame.session == 0)].mean())
        #     # print 'session2: {}'.format(data_frame['V1_100_mean'][np.array(data_frame.subject == s) & np.array(data_frame.session == 1)].mean())
            
        # shell()    
            
        
        fig = plt.figure(figsize=(6,4))
        plot_nr = 1
        # for ori in [45, 135]:
        for type_signal in ['mean', 'info']:
            
            projections_per_nr_voxels_r_present = []
            projections_per_nr_voxels_r_absent = []
            for r in rois:
                projections_per_nr_voxels_present = np.zeros((len(self.subjects), len(np.array(np.linspace(10,150,15), dtype=int)))) 
                projections_per_nr_voxels_absent = np.zeros((len(self.subjects), len(np.array(np.linspace(10,150,15), dtype=int)))) 
                for j, nr_voxels in enumerate(np.array(np.linspace(10,150,15), dtype=int)):
                    for i in range(len(self.subjects)):
                        s_ori = np.array(self.pupil_data[i]['signal_orientation'])
                        # ind_1 = self.present[i]
                        # ind_2 = ~self.present[i]
                        ind_1 = self.present[i] # * (s_ori==ori)
                        ind_2 = ~self.present[i] # * (s_ori==ori)
                        projections_per_nr_voxels_present[i,j] = np.array(data_frame[(data_frame.subject == self.subjects[i])]['{}_{}_{}'.format(r, nr_voxels, type_signal)])[ind_1].mean()
                        projections_per_nr_voxels_absent[i,j] = np.array(data_frame[(data_frame.subject == self.subjects[i])]['{}_{}_{}'.format(r, nr_voxels, type_signal)])[ind_2].mean()
                        # projections_per_nr_voxels_present[i,j] = np.array(data_frame[(data_frame.subject == self.subjects[i])]['{}_{}_mean'.format(r, nr_voxels)])[ind_1].mean()
                        # projections_per_nr_voxels_absent[i,j] = np.array(data_frame[(data_frame.subject == self.subjects[i])]['{}_{}_mean'.format(r, nr_voxels)])[ind_2].mean()

                projections_per_nr_voxels_r_present.append(projections_per_nr_voxels_present)
                projections_per_nr_voxels_r_absent.append(projections_per_nr_voxels_absent)
        
            ax = fig.add_subplot(1,2,plot_nr)
            x = np.array(np.linspace(10,150,15), dtype=int)
            # p1 = np.zeros(len(x))
            # p2 = np.zeros(len(x))
            # p3 = np.zeros(len(x))
            # for i in range(len(p1)):
            #     p1[i] = sp.stats.ttest_rel(projections_per_nr_voxels_r_present[0][:,i], projections_per_nr_voxels_r_absent[0][:,i])[1]
            #     p2[i] = sp.stats.ttest_rel(projections_per_nr_voxels_r_present[1][:,i], projections_per_nr_voxels_r_absent[1][:,i])[1]
            # sig_level = 0.05
            # sig_indices = np.array(p1 < sig_level, dtype=int)
            # sig_indices[0] = 0
            # sig_indices[-1] = 0
            # s_bar_p1 = zip(np.where(np.diff(sig_indices)==1)[0]+1, np.where(np.diff(sig_indices)==-1)[0]+1)
            # sig_indices = np.array(p2 < sig_level, dtype=int)
            # sig_indices[0] = 0
            # sig_indices[-1] = 0
            # s_bar_p2 = zip(np.where(np.diff(sig_indices)==1)[0]+1, np.where(np.diff(sig_indices)==-1)[0]+1)
            # sig_indices = np.array(p3 < sig_level, dtype=int)
            # sig_indices[0] = 0
            # sig_indices[-1] = 0
            # s_bar_p3 = zip(np.where(np.diff(sig_indices)==1)[0]+1, np.where(np.diff(sig_indices)==-1)[0]+1)
            # s_bar = [s_bar_p1, s_bar_p2, s_bar_p3]
            # s_bar = [s_bar_p1, s_bar_p2]
            for data, r, color in zip(projections_per_nr_voxels_r_absent, ['V1', 'V2', 'V3'], ['b', 'g', 'r']):
                # for d in data:
                    # ax.plot(x, d, label=r, alpha=0.2, ls='--')
                ax.plot(x, data.mean(axis=0), label=r + ' - absent', color=color, ls='--')
                ax.fill_between(x, data.mean(axis=0)-sp.stats.sem(data, axis=0), data.mean(axis=0)+sp.stats.sem(data, axis=0), alpha=0.2, color=color)
            for data, r, color in zip(projections_per_nr_voxels_r_present, ['V1', 'V2', 'V3'], ['b', 'g', 'r']):
                # for d in data:
                    # ax.plot(x, d, label=r, alpha=0.2)
                ax.plot(x, data.mean(axis=0), label=r + ' - present', color=color)
                ax.fill_between(x, data.mean(axis=0)-sp.stats.sem(data, axis=0), data.mean(axis=0)+sp.stats.sem(data, axis=0), alpha=0.2, color=color)
            # for sigs, yloc, color in zip(s_bar, [-0.01, -0.015, -0.02], ['b', 'g', 'r']):
            #     for sig in sigs:
            #         ax.hlines(yloc, x[int(sig[0])], x[int(sig[1])], lw=2, color=color)
            plt.axhline(0, color='k')
            plt.xlabel('nr voxels')
            plt.ylabel(type_signal)
            plt.title(type_signal)
            plt.legend()
            plot_nr += 1
            
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder, 'lateralization', 'lineplot.pdf'))
        
        
    def MULTIVARIATE_make_dataframe(self, data_type, prepare=False):
        
        from sklearn import preprocessing
        from sklearn import svm
        from sklearn.cross_validation import LeaveOneOut
        from sklearn.cross_validation import KFold
        
        partial = False
        time_locked = 'stim_locked'
        type_response = self.split_by.split('_')[-1]
        cortex_mask = np.array(nib.load('/home/shared/UvA/Niels_UvA/mni_masks/harvard_oxford/volume_1.nii.gz').get_data(), dtype=bool) + np.array(nib.load('/home/shared/UvA/Niels_UvA/mni_masks/harvard_oxford/volume_12.nii.gz').get_data(), dtype=bool)
        epi_box = np.array(nib.load(os.path.join('/home/shared/UvA/Niels_UvA/mni_masks/', '2014_fMRI_yesno_epi_box.nii.gz')).get_data(), dtype=bool)
        epi_box = epi_box * cortex_mask
        # task_network = NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'trials_all_clean_MNI_smooth_stim_locked_snr_pupil_d_0_psc_all_tfce_corrp_tstat1.nii.gz')).data
        # task_network = task_network>0.99
        
        self.make_pupil_BOLD_dataframe(data_type='clean_False', time_locked=time_locked, regress_iti=False, regress_rt=False)
        
        # params:
        rois = ['lr_aIPS', 'lr_PCeS', 'lr_M1',]
        nr_voxels_for_lat = 50 # means 50 per hemisphere, so 100 in total.
        nr_voxels_for_V123 = 100 # same N-voxels as choice selective
        nr_voxels_for_multivariate = 100 # to really only look at the most selective voxels.
        
        v = 1
        ind_type = 'ori'
        
        if prepare:
            for i in range(len(self.subjects)):
                print self.subjects[i]
                scalars = np.array(nib.load(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_{}_{}_{}_{}.nii.gz'.format('clean_MNI', time_locked, self.subjects[i], type_response))).get_data())
                scalars_flipped = scalars[::-1,:,:,:]
                scalars_control = (scalars[np.array(nib.load(os.path.join('/home/shared/UvA/Niels_UvA/surface_masks/', 'rh.precentral_sulcus_control.nii.gz')).get_data(), dtype=bool)+np.array(nib.load(os.path.join('/home/shared/UvA/Niels_UvA/surface_masks/', 'lh.precentral_sulcus_control.nii.gz')).get_data(), dtype=bool),:].mean(axis=0)) / 2.0
                # scalars_control = scalars[:,epi_box*cortex_mask].mean(axis=1)
                # scalars_whole_brain = vstack([scalars[:,np.array(NiftiImage(os.path.join('/home/shared/Niels_UvA/surface_masks/', 'rh.'+r+'.nii.gz')).data, dtype=bool)+np.array(NiftiImage(os.path.join('/home/shared/Niels_UvA/surface_masks/', 'lh.'+r+'.nii.gz')).data, dtype=bool)].mean(axis=1) for r in rois]).mean(axis=0)
                np.save(os.path.join(self.data_folder, 'choice_areas', 'scalars_control_{}.npy'.format(self.subjects[i])), scalars_control)
                for r in ['lr_aIPS', 'lr_PCeS', 'lr_M1',]:
                    roi_rh = np.array(nib.load(os.path.join('/home/shared/UvA/Niels_UvA/surface_masks/', 'rh.'+r+'.nii.gz')).get_data(), dtype=bool)
                    if self.subjects[i] in ['sub-03', 'sub-04', 'sub-06', 'sub-08', 'sub-09', 'sub-11', 'sub-13']:
                        scalars_rh = scalars[roi_rh,:]
                        scalars_lh = scalars_flipped[roi_rh,:]
                    else:
                        scalars_rh = scalars_flipped[roi_rh,:]
                        scalars_lh = scalars[roi_rh,:]
                    np.save(os.path.join(self.data_folder, 'choice_areas', '{}_{}_lh.npy'.format(r, self.subjects[i])), scalars_lh)
                    np.save(os.path.join(self.data_folder, 'choice_areas', '{}_{}_rh.npy'.format(r, self.subjects[i])), scalars_rh)
                
                for r in ['sl_IPL', 'sl_SPL1', 'sl_SPL2', 'sl_pIns', 'sl_PCeS_IFG', 'sl_MFG']:
                    try:
                        subtract = np.sum(np.vstack([np.array(nib.load(os.path.join('/home/shared/UvA/Niels_UvA/surface_masks/', 'lh.'+rr+'.nii.gz')).get_data(), dtype=bool)[np.newaxis,...] for rr in ['lr_aIPS', 'lr_PCeS', 'lr_M1']]), axis=0, dtype=bool)
                        roi_lh = np.array(nib.load(os.path.join('/home/shared/UvA/Niels_UvA/surface_masks/', 'lh.'+r+'.nii.gz')).get_data(), dtype=bool) & ~subtract
                        scalars_lh = scalars[roi_lh,:]
                        np.save(os.path.join(self.data_folder, 'choice_areas', '{}_{}_lh.npy'.format(r, self.subjects[i])), scalars_lh)
                    except:
                        pass
                    subtract = np.sum(np.vstack([np.array(nib.load(os.path.join('/home/shared/UvA/Niels_UvA/surface_masks/', 'rh.'+rr+'.nii.gz')).get_data(), dtype=bool)[np.newaxis,...] for rr in ['lr_aIPS', 'lr_PCeS', 'lr_M1']]), axis=0, dtype=bool)
                    roi_rh = np.array(nib.load(os.path.join('/home/shared/UvA/Niels_UvA/surface_masks/', 'rh.'+r+'.nii.gz')).get_data(), dtype=bool)
                    scalars_rh = scalars[roi_rh,:]
                    np.save(os.path.join(self.data_folder, 'choice_areas', '{}_{}_rh.npy'.format(r, self.subjects[i])), scalars_rh)
        else:
        
            # create data_frame:
            # ------------------
        
            dfs = []
            for i in range(len(self.subjects)):
                print self.subjects[i]
                
                # shell()
                
                # params:
                rt = np.array(self.pupil_data[i]['rt'])
                yes = np.array(self.pupil_data[i]['yes'])
                session = np.array(self.pupil_data[i]['session'])
                
                scalars_control = np.load(os.path.join(self.data_folder, 'choice_areas', 'scalars_control_{}.npy'.format(self.subjects[i])))
                
                df = {}
                
                # add lateralization:
                # -------------
                
                for r in rois:
                
                    # compute:
                    scalars_lh = np.load(os.path.join(self.data_folder, 'choice_areas', '{}_{}_lh.npy'.format(r, self.subjects[i]))).T
                    scalars_rh = np.load(os.path.join(self.data_folder, 'choice_areas', '{}_{}_rh.npy'.format(r, self.subjects[i]))).T
                    values_lh = []
                    values_rh = []
                    for s in np.unique(session):
                        session_ind = (session == s)
                        t_values_lh = (sp.stats.ttest_1samp(scalars_lh[yes & session_ind], 0)[0] + (sp.stats.ttest_1samp(scalars_lh[~yes & session_ind], 0)[0]*-1.0) ) / 2.0
                        t_values_rh = (sp.stats.ttest_1samp(scalars_rh[~yes & session_ind], 0)[0] + (sp.stats.ttest_1samp(scalars_rh[yes & session_ind], 0)[0]*-1.0) ) / 2.0
                        lh_ind = np.argsort(t_values_lh)[::-1][:nr_voxels_for_lat]
                        rh_ind = np.argsort(t_values_rh)[::-1][:nr_voxels_for_lat]
                        values_lh.append(scalars_lh[session_ind,:][:,lh_ind].mean(axis=1))
                        values_rh.append(scalars_rh[session_ind,:][:,rh_ind].mean(axis=1))
                    values_lh = np.concatenate(values_lh)
                    values_rh = np.concatenate(values_rh)
                    values_mean = (values_lh + values_rh) / 2.0
                    values_mean = myfuncs.lin_regress_resid(values_mean, [scalars_control], project_out=True)
                    values_lat = values_lh - values_rh
                    
                    # add:
                    df[r] = values_mean
                    df[r+'_lat'] = values_lat
                    df[r+'_lh'] = values_lh
                    df[r+'_rh'] = values_rh
                
                
                # add parietal:
                # -------------
                
                for r in ['sl_IPL', 'sl_SPL1', 'sl_SPL2', 'sl_pIns', 'sl_PCeS_IFG', 'sl_MFG']:
                    try:
                        scalars_lh = np.load(os.path.join(self.data_folder, 'choice_areas', '{}_{}_lh.npy'.format(r, self.subjects[i]))).T
                        scalars_rh = np.load(os.path.join(self.data_folder, 'choice_areas', '{}_{}_rh.npy'.format(r, self.subjects[i]))).T
                        scalars = np.hstack((scalars_lh, scalars_rh))
                    except:
                        scalars = np.load(os.path.join(self.data_folder, 'choice_areas', '{}_{}_rh.npy'.format(r, self.subjects[i]))).T
                    
                    info_signal = []
                    SVM_signal = []
                    for s in np.unique(session):
                        session_ind = (session == s)
                        t_values = sp.stats.ttest_1samp(scalars[yes & session_ind,:], 0)[0] - sp.stats.ttest_1samp(scalars[~yes & session_ind,:], 0)[0]
                        # choice_ind = np.argsort(t_values)
                        choice_ind = np.concatenate((np.argsort(t_values)[-nr_voxels_for_lat:][::-1], np.argsort(t_values)[:nr_voxels_for_lat:][::-1]))
                        
                        y = yes[session_ind]
                        data = scalars[session_ind,:]
                        cv = LeaveOneOut(y.size)
                        info = np.zeros(sum(session_ind))
                        svm = np.zeros(sum(session_ind))
                        for train, test in cv:
                        
                            # data:
                            train_data = np.atleast_2d(data[train,:][:,choice_ind])
                            test_data = np.atleast_2d(data[test,:][:,choice_ind])
                            scaler = preprocessing.StandardScaler().fit(train_data)
                            train_data = scaler.transform(train_data)
                            test_data = scaler.transform(test_data)
                            
                            # projection:
                            # template = train_data[y[train],:].mean(axis=0) - train_data[~y[train],:].mean(axis=0)
                            template = sp.stats.ttest_1samp(train_data[y[train],:], 0)[0] - sp.stats.ttest_1samp(train_data[~y[train],:], 0)[0]
                            info[test] = sp.stats.pearsonr(template, test_data.ravel())[0]
                            
                            # # SVM:
                            # clf = svm.SVC(kernel='linear', class_weight='balanced',) # verbose=1,)
                            # clf.fit(train_data, y[train],)
                            # svm[test] = clf.decision_function(test_data)
                        
                        info_signal.append(info)
                        SVM_signal.append(svm)
                            
                    # add:
                    info_signal = np.concatenate(info_signal)
                    SVM_signal = np.concatenate(SVM_signal)
                    df['{}_mean'.format(r,)] = scalars.mean(axis=1)
                    df['{}_info'.format(r,)] = info_signal
                    df['{}_SVM'.format(r,)] = SVM_signal
                
                
                # add V1-3:
                # --------

                s = []
                for roi in ['V1', 'V2', 'V3']:
                    files = sort(glob.glob(os.path.join(self.data_folder, 'V123', 'univariate', 'scalars{}_{}*{}_{}_center*'.format(self.postfix, self.subjects[i], roi, nr_voxels_for_V123))))
                    s.append( np.concatenate([np.load(f) for f in files]) )
                    files = sort(glob.glob(os.path.join(self.data_folder, 'V123', 'univariate', 'scalars{}_{}*{}_{}_surround*'.format(self.postfix, self.subjects[i], roi, nr_voxels_for_V123))))
                    s.append( np.concatenate([np.load(f) for f in files]) )
                scalars_visual = np.hstack(s)

                center_ind = np.array(np.concatenate((np.ones(nr_voxels_for_V123), np.zeros(nr_voxels_for_V123), np.ones(nr_voxels_for_V123), np.zeros(nr_voxels_for_V123), np.ones(nr_voxels_for_V123), np.zeros(nr_voxels_for_V123),)), dtype=bool)
                v1_ind = np.array(np.concatenate((np.ones(nr_voxels_for_V123), np.ones(nr_voxels_for_V123), np.zeros(nr_voxels_for_V123), np.zeros(nr_voxels_for_V123), np.zeros(nr_voxels_for_V123), np.zeros(nr_voxels_for_V123),)), dtype=bool)
                v2_ind = np.array(np.concatenate((np.zeros(nr_voxels_for_V123), np.zeros(nr_voxels_for_V123), np.ones(nr_voxels_for_V123), np.ones(nr_voxels_for_V123), np.zeros(nr_voxels_for_V123), np.zeros(nr_voxels_for_V123),)), dtype=bool)
                v3_ind = np.array(np.concatenate((np.zeros(nr_voxels_for_V123), np.zeros(nr_voxels_for_V123), np.zeros(nr_voxels_for_V123), np.zeros(nr_voxels_for_V123), np.ones(nr_voxels_for_V123), np.ones(nr_voxels_for_V123),)), dtype=bool)

                scalars_center_V123 = scalars_visual[:,center_ind].mean(axis=1)
                scalars_surround_V123 = scalars_visual[:,~center_ind].mean(axis=1)
                scalars_center_V1 = scalars_visual[:,center_ind & v1_ind].mean(axis=1)
                scalars_surround_V1 = scalars_visual[:,~center_ind & v1_ind].mean(axis=1)
                scalars_center_V2 = scalars_visual[:,center_ind & v2_ind].mean(axis=1)
                scalars_surround_V2 = scalars_visual[:,~center_ind & v2_ind].mean(axis=1)
                scalars_center_V3 = scalars_visual[:,center_ind & v3_ind].mean(axis=1)
                scalars_surround_V3 = scalars_visual[:,~center_ind & v3_ind].mean(axis=1)

                # regress out rt:
                for s in np.unique(session):
                    scalars_center_V123[session==s] = myfuncs.lin_regress_resid(scalars_center_V123[session==s], [rt[session==s]], project_out=False) + scalars_center_V123[session==s].mean()
                    scalars_surround_V123[session==s] = myfuncs.lin_regress_resid(scalars_surround_V123[session==s], [rt[session==s]], project_out=False) + scalars_surround_V123[session==s].mean()
                    scalars_center_V1[session==s] = myfuncs.lin_regress_resid(scalars_center_V1[session==s], [rt[session==s]], project_out=False) + scalars_center_V1[session==s].mean()
                    scalars_surround_V1[session==s] = myfuncs.lin_regress_resid(scalars_surround_V1[session==s], [rt[session==s]], project_out=False) + scalars_surround_V1[session==s].mean()
                    scalars_center_V2[session==s] = myfuncs.lin_regress_resid(scalars_center_V2[session==s], [rt[session==s]], project_out=False) + scalars_center_V2[session==s].mean()
                    scalars_surround_V2[session==s] = myfuncs.lin_regress_resid(scalars_surround_V2[session==s], [rt[session==s]], project_out=False) + scalars_surround_V2[session==s].mean()
                    scalars_center_V3[session==s] = myfuncs.lin_regress_resid(scalars_center_V3[session==s], [rt[session==s]], project_out=False) + scalars_center_V3[session==s].mean()
                    scalars_surround_V3[session==s] = myfuncs.lin_regress_resid(scalars_surround_V3[session==s], [rt[session==s]], project_out=False) + scalars_surround_V3[session==s].mean()

                # regress out surround
                scalars_center_V123_clean = scalars_center_V123.copy()
                scalars_center_V1_clean = scalars_center_V1.copy()
                scalars_center_V2_clean = scalars_center_V2.copy()
                scalars_center_V3_clean = scalars_center_V3.copy()
                for s in np.unique(session):
                    scalars_center_V123_clean[session==s] = myfuncs.lin_regress_resid(scalars_center_V123[session==s], [scalars_surround_V123[session==s]], project_out=False) + scalars_center_V123_clean[session==s].mean()
                    scalars_center_V1_clean[session==s] = myfuncs.lin_regress_resid(scalars_center_V1[session==s], [scalars_surround_V1[session==s]], project_out=False) + scalars_center_V1_clean[session==s].mean()
                    scalars_center_V2_clean[session==s] = myfuncs.lin_regress_resid(scalars_center_V2[session==s], [scalars_surround_V2[session==s]], project_out=False) + scalars_center_V2_clean[session==s].mean()
                    scalars_center_V3_clean[session==s] = myfuncs.lin_regress_resid(scalars_center_V3[session==s], [scalars_surround_V3[session==s]], project_out=False) + scalars_center_V3_clean[session==s].mean()

                # add to dataframe:
                df['V1'] = scalars_center_V1_clean
                df['V2'] = scalars_center_V2_clean
                df['V3'] = scalars_center_V3_clean
                df['V123'] = scalars_center_V123_clean
                df['V1_center'] = scalars_center_V1
                df['V2_center'] = scalars_center_V2
                df['V3_center'] = scalars_center_V3
                df['V123_center'] = scalars_center_V123
                df['V1_surround'] = scalars_surround_V1
                df['V2_surround'] = scalars_surround_V2
                df['V3_surround'] = scalars_surround_V3
                df['V123_surround'] = scalars_surround_V123

                # MULTIVARIATE STUFF:
                # -------------------

                for r in ['V1_center', 'V2_center', 'V3_center']:
                    files = sort(glob.glob(os.path.join(self.data_folder, 'V123', 'multivariate', 'scalars{}_{}_{}_{}_{}_*.npy'.format(self.postfix, r, ind_type, 'all', self.subjects[i],))))
                    scalars_m = [np.load(f) for f in files]

                    stim_info_signal_sessions = []
                    stim_ori_info_signal_sessions = []
                    choice_info_signal_sessions = []

                    SVM_stim_sessions = []
                    SVM_stim_ori_sessions = []
                    SVM_choice_sessions = []

                    signal_orientation = []
                    trials = 0
                    for s in range(len(scalars_m)):

                        # regress out rt:
                        for v in range(scalars_m[s].shape[1]):
                            scalars_m[s][:,v] = myfuncs.lin_regress_resid(scalars_m[s][:,v], [rt[(session-1)==s]], project_out=False) + scalars_m[s][:,v].mean()

                        # params:
                        pupil_data = pd.read_csv(os.path.join(self.data_folder[:-6], self.subjects[i], 'pupil_data_{}.csv'.format(s+1)))
                        pupil_data = pupil_data[-np.array(pupil_data.omissions)]
                        nr_trials_in_session = scalars_m[s].shape[0]
                        sig_orientation = np.array(pupil_data['signal_orientation'])
                        present = np.array(pupil_data['present'], dtype=bool)
                        yess = np.array(pupil_data['yes'], dtype=bool)

                        t_values = sp.stats.ttest_1samp(scalars_m[s][present,:], 0)[0] - sp.stats.ttest_1samp(scalars_m[s][~present,:], 0)[0]
                        # stim_ind = np.argsort(t_values)
                        stim_ind = np.concatenate((np.argsort(t_values)[-nr_voxels_for_multivariate/2:][::-1], np.argsort(t_values)[:nr_voxels_for_multivariate/2:][::-1]))

                        t_values = sp.stats.ttest_1samp(scalars_m[s][present&(sig_orientation==45),:], 0)[0] - sp.stats.ttest_1samp(scalars_m[s][present&(sig_orientation==135),:], 0)[0]
                        # stim_ind_ori = np.argsort(t_values)
                        stim_ind_ori = np.concatenate((np.argsort(t_values)[-nr_voxels_for_multivariate/2:][::-1], np.argsort(t_values)[:nr_voxels_for_multivariate/2:][::-1]))

                        t_values = sp.stats.ttest_1samp(scalars_m[s][yess,:], 0)[0] - sp.stats.ttest_1samp(scalars_m[s][~yess,:], 0)[0]
                        # choice_ind = np.argsort(t_values)
                        choice_ind = np.concatenate((np.argsort(t_values)[-nr_voxels_for_multivariate/2:][::-1], np.argsort(t_values)[:nr_voxels_for_multivariate/2:][::-1]))

                        data = scalars_m[s]
                        info_signal_stim = np.zeros(nr_trials_in_session)
                        info_signal_stim_ori = np.zeros(nr_trials_in_session)
                        info_signal_choice = np.zeros(nr_trials_in_session)
                        SVM_stim = np.zeros(nr_trials_in_session)
                        SVM_stim_ori = np.zeros(nr_trials_in_session)
                        SVM_choice = np.zeros(nr_trials_in_session)

                        cv = LeaveOneOut(nr_trials_in_session)
                        # cv = KFold(y.size, n_folds=8)
                        for train, test in cv:

                            # stim:
                            train_data = np.atleast_2d(data[train,:][:,stim_ind])
                            test_data = np.atleast_2d(data[test,:][:,stim_ind])
                            scaler = preprocessing.StandardScaler().fit(train_data)
                            train_data = scaler.transform(train_data)
                            test_data = scaler.transform(test_data)

                            template = sp.stats.ttest_1samp(train_data[present[train],:], 0)[0] - sp.stats.ttest_1samp(train_data[~present[train],:], 0)[0]
                            info_signal_stim[test] = sp.stats.pearsonr(template, test_data.ravel())[0]

                            # clf = svm.SVC(kernel='linear', class_weight='balanced',) # verbose=1,)
                            # clf.fit(train_data, np.array(present, dtype=int)[train],)
                            # SVM_stim[test] = clf.decision_function(test_data)

                            # stim ori:
                            train_data = np.atleast_2d(data[train,:][:,stim_ind_ori])
                            test_data = np.atleast_2d(data[test,:][:,stim_ind_ori])
                            scaler = preprocessing.StandardScaler().fit(train_data)
                            train_data = scaler.transform(train_data)
                            test_data = scaler.transform(test_data)

                            template = sp.stats.ttest_1samp(train_data[present[train]&(sig_orientation[train]==45),:], 0)[0] - sp.stats.ttest_1samp(train_data[present[train]&(sig_orientation[train]==135),:], 0)[0]
                            if sig_orientation[test] == 45:
                                info_signal_stim_ori[test] = sp.stats.pearsonr(template, test_data.ravel())[0]
                            elif sig_orientation[test] == 135:
                                info_signal_stim_ori[test] = sp.stats.pearsonr(template, test_data.ravel())[0] * -1.0

                            # clf = svm.SVC(kernel='linear', class_weight='balanced',) # verbose=1,)
                            # clf.fit(train_data[present[train],:], sig_orientation[train][present[train]],)
                            # if sig_orientation[test] == 45:
                            #     SVM_stim_ori[test] = clf.decision_function(test_data) * -1.0
                            # elif sig_orientation[test] == 135:
                            #     SVM_stim_ori[test] = clf.decision_function(test_data)

                            # choice:
                            train_data = np.atleast_2d(data[train,:][:,choice_ind])
                            test_data = np.atleast_2d(data[test,:][:,choice_ind])
                            scaler = preprocessing.StandardScaler().fit(train_data)
                            train_data = scaler.transform(train_data)
                            test_data = scaler.transform(test_data)

                            template = sp.stats.ttest_1samp(train_data[yess[train],:], 0)[0] - sp.stats.ttest_1samp(train_data[~yess[train],:], 0)[0]
                            info_signal_choice[test] = sp.stats.pearsonr(template, test_data.ravel())[0]

                            # clf = svm.SVC(kernel='linear', class_weight='balanced',) # verbose=1,)
                            # clf.fit(train_data, np.array(yess, dtype=int)[train],)
                            # SVM_choice[test] = clf.decision_function(test_data)

                        stim_info_signal_sessions.append(info_signal_stim)
                        stim_ori_info_signal_sessions.append(info_signal_stim_ori)
                        choice_info_signal_sessions.append(info_signal_choice)

                        SVM_stim_sessions.append(SVM_stim)
                        SVM_stim_ori_sessions.append(SVM_stim_ori)
                        SVM_choice_sessions.append(SVM_choice)

                        signal_orientation.append(sig_orientation)
                        trials = trials+nr_trials_in_session
                    stim_info_signal_sessions = np.concatenate(stim_info_signal_sessions)
                    stim_ori_info_signal_sessions = np.concatenate(stim_ori_info_signal_sessions)
                    choice_info_signal_sessions = np.concatenate(choice_info_signal_sessions)

                    SVM_stim_sessions = np.concatenate(SVM_stim_sessions)
                    SVM_stim_ori_sessions = np.concatenate(SVM_stim_ori_sessions)
                    SVM_choice_sessions = np.concatenate(SVM_choice_sessions)

                    df['{}_info_stim'.format(r,)] = stim_info_signal_sessions
                    df['{}_info_stim_ori'.format(r,)] = stim_ori_info_signal_sessions
                    df['{}_info_choice'.format(r,)] = choice_info_signal_sessions
                    df['{}_SVM_stim'.format(r,)] = SVM_stim_sessions
                    df['{}_SVM_stim_ori'.format(r,)] = SVM_stim_ori_sessions
                    df['{}_SVM_choice'.format(r,)] = SVM_choice_sessions
                    df['ori'] = np.concatenate(signal_orientation)
                    df['session'] = session
                    
                # add subject:
                # ------------
                df['subject'] = self.subjects[i]
            
                # make dataframe and append:
                df = pd.DataFrame(df)
                dfs.append(df)
        
            # combine across subjects and save:    
            data_frame = pd.concat(dfs)
            data_frame.to_csv(os.path.join(self.data_folder, 'event_related_average_wb', 'MNI_dataframe_1.csv'))
    
    def MULTIVARIATE_add_combined_signal(self):
        
        data_frame = pd.read_csv(os.path.join(self.data_folder, 'event_related_average_wb', 'MNI_dataframe_1.csv'))
        dfs = [data_frame[data_frame.subject == s] for s in self.subjects]
        
        # add linear combinations:
        for i in range(len(self.subjects)):
            d = {
                'Y1' : pd.Series(np.array(self.present[i], dtype=int)),
                'Y2' : pd.Series(np.array(self.yes[i], dtype=int)),
                'X0' : np.array(pd.Series(dfs[i]['V1_center_info_stim_ori'])),
                'X1' : np.array(pd.Series(dfs[i]['V2_center_info_stim_ori'])),
                'X2' : np.array(pd.Series(dfs[i]['V3_center_info_stim_ori'])),
                'X3' : np.array(pd.Series(dfs[i]['V1_center_info_choice'])),
                'X4' : np.array(pd.Series(dfs[i]['V2_center_info_choice'])),
                'X5' : np.array(pd.Series(dfs[i]['V3_center_info_choice'])),
                'X6' : np.array(pd.Series(dfs[i]['sl_IPL_info'])),
                'X7' : np.array(pd.Series(dfs[i]['sl_SPL1_info'])),
                'X8' : np.array(pd.Series(dfs[i]['lr_PCeS_lat'])),
                'X9' : np.array(pd.Series(dfs[i]['sl_pIns_info'])),
                'X10' : np.array(pd.Series(dfs[i]['lr_aIPS_lat'])),
                'X11' : np.array(pd.Series(dfs[i]['sl_SPL2_info'])),
                'X12' : np.array(pd.Series(dfs[i]['sl_PCeS_IFG_info'])),
                'X13' : np.array(pd.Series(dfs[i]['sl_MFG_info'])),
                'X14' : np.array(pd.Series(dfs[i]['lr_M1_lat'])),
                }
            data = pd.DataFrame(d)
            
            # only stim:
            formula = 'Y1 ~ X0 + X1 + X2'
            model = sm.logit(formula=formula, data=data)
            fitted = model.fit()
            values = np.array(fitted.fittedvalues)
            dfs[i]['combined_stim_V123'] = values - values.mean()
            
            # only stim:
            formula = 'Y2 ~ X0 + X1 + X2'
            model = sm.logit(formula=formula, data=data)
            fitted = model.fit()
            values = np.array(fitted.fittedvalues)
            dfs[i]['combined_stim_V123_to_choice'] = values - values.mean()
            
            # only stim:
            formula = 'Y2 ~ X3 + X4 + X5'
            model = sm.logit(formula=formula, data=data)
            fitted = model.fit()
            values = np.array(fitted.fittedvalues)
            dfs[i]['combined_choice_V123'] = values - values.mean()
            
            # choice regions:
            formula = 'Y2 ~ X8 + X10 + X6 + X7 + X9 + X11 + X12 + X13'
            model = sm.logit(formula=formula, data=data)
            fitted = model.fit()
            values = np.array(fitted.fittedvalues)
            dfs[i]['combined_choice_parietal_all'] = values - values.mean()
            
            # choice regions:
            formula = 'Y2 ~ X8 + X10'
            model = sm.logit(formula=formula, data=data)
            fitted = model.fit()
            values = np.array(fitted.fittedvalues)
            dfs[i]['combined_choice_parietal_lat'] = values - values.mean()
            
            # choice regions:
            formula = 'Y2 ~ X6 + X7 + X9 + X11 + X12 + X13'
            model = sm.logit(formula=formula, data=data)
            fitted = model.fit()
            values = np.array(fitted.fittedvalues)
            dfs[i]['combined_choice_parietal_sl'] = values - values.mean()
            
            for r in ['lr_M1_lat', 'combined_choice_parietal_all', 'combined_choice_parietal_lat', 'combined_choice_parietal_sl']:
                dfs[i][r+'_res'] = myfuncs.lin_regress_resid(np.array(dfs[i][r]), [np.array(self.present[i], dtype=int)]) + dfs[i][r].mean()
            
        # combine across subjects and save:    
        data_frame = pd.concat(dfs)
        data_frame.to_csv(os.path.join(self.data_folder, 'event_related_average_wb', 'MNI_dataframe_1.csv'))    
        
    def CHOICE_SIGNALS_plots(self):
        
        # load dataframe:
        data_frame = pd.read_csv(os.path.join(self.data_folder, 'event_related_average_wb', 'MNI_dataframe_1.csv'))
        data_frame['pupil_h_ind'] = np.concatenate(self.pupil_h_ind)
        dfs = [data_frame[data_frame.subject == s] for s in self.subjects]
        
        decoding_type = 'info'
        
        titles =         [
                        # 'V123_info_stim',
                        # 'V123_info_stim_ori',
                        # 'V123_info_choice',
                        # 'V123_spat_spec',
                        # 'V123_spat_center',
                        # 'V123_spat_surround',
                        'choice_lat',
                        'choice_info',
                        'combined',
                        ]
        rois_groups =     [
                        # ['V1_center_'+decoding_type+'_stim', 'V2_center_'+decoding_type+'_stim', 'V3_center_'+decoding_type+'_stim'],
                        # ['V1_center_'+decoding_type+'_stim_ori', 'V2_center_'+decoding_type+'_stim_ori', 'V3_center_'+decoding_type+'_stim_ori'],
                        # ['V1_center_'+decoding_type+'_choice', 'V2_center_'+decoding_type+'_choice', 'V3_center_'+decoding_type+'_choice'],
                        # ['V1', 'V2', 'V3',],
                        # ['V1_center', 'V2_center', 'V3_center',],
                        # ['V1_surround', 'V2_surround', 'V3_surround',],
                        ['lr_aIPS_lat', 'lr_PCeS_lat', 'lr_M1_lat',],
                        ['sl_IPL_'+decoding_type, 'sl_SPL1_'+decoding_type, 'sl_SPL2_'+decoding_type, 'sl_pIns_'+decoding_type, 'sl_PCeS_IFG_'+decoding_type, 'sl_MFG_'+decoding_type],
                        ['combined_stim_V123', 'combined_stim_V123_to_choice', 'combined_choice_V123'],
                        ]
        rois_names =     [
                        # ['V1', 'V2', 'V3',],
                        # ['V1', 'V2', 'V3',],
                        # ['V1', 'V2', 'V3',],
                        # ['V1', 'V2', 'V3',],
                        # ['V1', 'V2', 'V3',],
                        # ['V1', 'V2', 'V3',],
                        # ['V1', 'V2', 'V3',],
                        ['aIPS1', 'IPS/PostCeS', 'M1',],
                        ['IPL', 'SPL', 'aIPS2', 'pIns', 'PCeS_IFG', 'MFG'],
                        ['stim V123', 'stim V123 choice', 'choice V123'],
                        ]
        ylabels =         [
                        # 'Correlation coefficient',
                        # 'Correlation coefficient',
                        # 'Correlation coefficient',
                        # 'fMRI response\n(%signal change)',
                        # 'fMRI response\n(%signal change)',
                        # 'fMRI response\n(%signal change)',
                        'Lateralization\n(%signal change)',
                        'Correlation coefficient',
                        'Choice selective response (a.u.)',
                        ]
        # ylims = [(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.1, 0.1), (-0.8, 0.8), (-0.8, 0.8), (-0.1, 0.1), (-0.1, 0.1)]
                        
                        
        for title, rois, rois_name, ylabel in zip(titles, rois_groups, rois_names, ylabels):        
            for inds, trial_type, xlabels, colors in zip([[self.pupil_h_ind, self.pupil_l_ind], [self.yes, self.no], [self.present, self.absent],], ['pupil', 'yes', 'present',], [['high', 'low'], ['yes', 'no'], ['present', 'absent',]],  [['r','b'], ['m','c',], ['orange','forestgreen']]):
                fig = plt.figure(figsize=( (1+(len(rois)*0.3)),1.75))
                ax = fig.add_subplot(111)
                locs = np.arange(0,len(rois))
                for r, roi in enumerate(rois):
                    lat = np.zeros(len(self.subjects))
                    lat_1 = np.zeros(len(self.subjects))
                    lat_2 = np.zeros(len(self.subjects))
                    for i in range(len(self.subjects)):
                        phasics = np.array(dfs[i][roi])
                        ind_1 = inds[0][i]
                        ind_2 = inds[1][i]
                        if title == 'V123_info_stim' or title == 'V123_info_stim_ori':
                            if trial_type == 'pupil':
                                ind_1 = inds[0][i] * self.present[i]
                                ind_2 = inds[1][i] * self.present[i]
                        # ind_1 = inds[0][i]
                        # ind_2 = inds[1][i]
                        lat_1[i] = np.nanmean(phasics[ind_1])
                        lat_2[i] = np.nanmean(phasics[ind_2])
                        lat[i] = np.nanmean(phasics)
                    p1 = myfuncs.permutationTest(lat_1, lat_2, paired=True)[1]
                    p2 = myfuncs.permutationTest(lat_1, np.zeros(len(lat_1)), paired=True)[1]
                    p3 = myfuncs.permutationTest(lat_2, np.zeros(len(lat_2)), paired=True)[1]
                    # p1 = sp.stats.ttest_rel(lat_1, lat_2,)[1]
                    # p2 = sp.stats.ttest_rel(lat_1, np.zeros(len(lat_1)),)[1]
                    # p3 = sp.stats.ttest_rel(lat_2, np.zeros(len(lat_2)),)[1]
                    # if title == 'choice_lat_lr':
                    #     if roi == 'lr_insula2_lat':
                    #         if trial_type == 'pupil':
                    #             bias = (lat_2*-1.0) - abs(lat_1)
                    #             # shell()
                    #
                    #             dc = (np.load(os.path.join(self.data_folder, 'params_fMRI.npy'))[2,:] + np.load(os.path.join(self.data_folder, 'params_fMRI.npy'))[3,:] / 2.0)
                    #             delta_dc = np.load(os.path.join(self.data_folder, 'params_fMRI.npy'))[2,:] - np.load(os.path.join(self.data_folder, 'params_fMRI.npy'))[3,:]
                    #             delta_sp = np.load(os.path.join(self.data_folder, 'params_fMRI.npy'))[0,:] - np.load(os.path.join(self.data_folder, 'params_fMRI.npy'))[1,:]
                        
                    bar_width = 0.4
                    my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}
                    ax.bar(locs[r]-(bar_width/2.0), lat_1.mean(), width = bar_width, yerr = sp.stats.sem(lat_1), color = colors[0], alpha = 1, align = 'center', edgecolor = 'k', linewidth = 0, capsize = 0, ecolor = 'k', error_kw={'elinewidth':0.5,})
                    ax.bar(locs[r]+(bar_width/2.0), lat_2.mean(), width = bar_width, yerr = sp.stats.sem(lat_2), color = colors[1], alpha = 1, align = 'center', edgecolor = 'k', linewidth = 0, capsize = 0, ecolor = 'k', error_kw={'elinewidth':0.5,})
                    
                    # plt.ylim(-0.6,0.6)
                    if p1 < 0.05:
                        plt.text(s='{}'.format(round(p1, 3)), x=locs[r], y=gca().get_ylim()[1]-((gca().get_ylim()[1] - gca().get_ylim()[0]) / 10.0), size=5, horizontalalignment='center',)
                    if p2 < 0.05:
                        plt.text(s='{}'.format(round(p2, 3)), x=locs[r]-(bar_width/2.0), y=gca().get_ylim()[0]+((gca().get_ylim()[1] - gca().get_ylim()[0]) / 8.0), size=5, horizontalalignment='center', rotation=90)
                    if p3 < 0.05:
                        plt.text(s='{}'.format(round(p3, 3)), x=locs[r]+(bar_width/2.0), y=gca().get_ylim()[0]+((gca().get_ylim()[1] - gca().get_ylim()[0]) / 8.0), size=5, horizontalalignment='center', rotation=90)
                plt.title(title)
                plt.ylabel(ylabel)
                sns.despine(offset=10, trim=True)
                plt.xticks(locs, rois_name, rotation=45)
                plt.tight_layout()
                fig.savefig(os.path.join(self.figure_folder, 'lateralization', 'bars_{}_{}.pdf'.format(title, trial_type)))
                
        # # ANOVA:
        # rois = ['aIPS_lat', 'M1_lat',]
        # lat_1_h = np.zeros(len(self.subjects))
        # lat_1_l = np.zeros(len(self.subjects))
        # lat_2_h = np.zeros(len(self.subjects))
        # lat_2_l = np.zeros(len(self.subjects))
        # for i in range(len(self.subjects)):
        #     lat_1_h[i] = np.array(dfs[i]['aIPS_lat'])[self.pupil_h_ind[i]].mean()
        #     lat_1_l[i] = np.array(dfs[i]['aIPS_lat'])[self.pupil_l_ind[i]].mean()
        #     lat_2_h[i] = np.array(dfs[i]['M1_lat'])[self.pupil_h_ind[i]].mean()
        #     lat_2_l[i] = np.array(dfs[i]['M1_lat'])[self.pupil_l_ind[i]].mean()
        #
        # dv = np.concatenate( (lat_1_h, lat_1_l, lat_2_h, lat_2_l) )
        # pupil = np.concatenate( (np.ones(len(self.subjects)), np.zeros(len(self.subjects)), np.ones(len(self.subjects)), np.zeros(len(self.subjects))) )
        # area = np.concatenate( (np.ones(len(self.subjects)), np.ones(len(self.subjects)), np.zeros(len(self.subjects)), np.zeros(len(self.subjects))) )
        # subject = np.concatenate( (np.arange(len(self.subjects)), np.arange(len(self.subjects)), np.arange(len(self.subjects)), np.arange(len(self.subjects))) )
        # d = rlc.OrdDict([('pupil', robjects.IntVector(list(pupil.ravel()))), ('area', robjects.IntVector(list(area.ravel()))), ('subject', robjects.IntVector(list(subject.ravel()))), ('data', robjects.FloatVector(list(dv.ravel())))])
        # robjects.r.assign('dataf', robjects.DataFrame(d))
        # robjects.r('attach(dataf)')
        # statres = robjects.r('res = summary(aov(data ~ as.factor(pupil)*as.factor(area) + Error(as.factor(subject)/(as.factor(pupil)*as.factor(area))), dataf))')
        # # statres = robjects.r('res = summary(aov(data ~ as.factor(pupil)*as.factor(area) + Error(as.factor(subject)), dataf))')
        #
        #
        #
        # print myfuncs.permutationTest(lat_1_h-lat_1_l, lat_2_h-lat_2_l, paired=True)
        # print sp.stats.ttest_rel(lat_1_h-lat_1_l, lat_2_h-lat_2_l)
        
    
    def CHOICE_SIGNALS_plots_2(self):
        
        # load dataframe:
        data_frame = pd.read_csv(os.path.join(self.data_folder, 'event_related_average_wb', 'MNI_dataframe_1.csv'))
        dfs = [data_frame[data_frame.subject == s] for s in self.subjects]
        data_frame = pd.concat(dfs)
        
        # add:
        pupil = np.array(np.concatenate(self.pupil_h_ind), dtype=int)
        pupil[~(np.concatenate(self.pupil_l_ind)+np.concatenate(self.pupil_h_ind))] = 2
        data_frame['pupil'] = pupil
        data_frame['present'] = np.concatenate(self.present)
        data_frame['yes'] = np.concatenate(self.yes)
        
        decoding_type = 'info'
        
        titles =        [
                        # 'V123_info_stim',
                        # 'V123_info_stim_ori',
                        # 'V123_info_choice',
                        # 'V123_spat_spec',
                        # 'V123_spat_center',
                        # 'V123_spat_surround',
                        # 'choice_lat',
                        # 'choice_info',
                        # 'combined',
                        'combined_all'
                        'combined1',
                        'combined2',
                        ]
        rois_groups =   [
                        # ['V1_center_'+decoding_type+'_stim', 'V2_center_'+decoding_type+'_stim', 'V3_center_'+decoding_type+'_stim'],
                        # ['V1_center_'+decoding_type+'_stim_ori', 'V2_center_'+decoding_type+'_stim_ori', 'V3_center_'+decoding_type+'_stim_ori'],
                        # ['V1_center_'+decoding_type+'_choice', 'V2_center_'+decoding_type+'_choice', 'V3_center_'+decoding_type+'_choice'],
                        # ['V1', 'V2', 'V3',],
                        # ['V1_center', 'V2_center', 'V3_center',],
                        # ['V1_surround', 'V2_surround', 'V3_surround',],
                        # ['lr_aIPS_lat', 'lr_PCeS_lat', 'lr_M1_lat',],
                        # ['sl_IPL_'+decoding_type, 'sl_SPL1_'+decoding_type, 'sl_SPL2_'+decoding_type, 'sl_pIns_'+decoding_type, 'sl_PCeS_IFG_'+decoding_type, 'sl_MFG_'+decoding_type],
                        # ['combined_stim_V123', 'combined_stim_V123_to_choice', 'combined_choice_V123'],
                        ['lr_M1_lat', 'combined_choice_parietal_lat', 'combined_choice_parietal_sl'],
                        ['lr_M1_lat', 'combined_choice_parietal_lat'],
                        ['combined_choice_parietal_sl'],
                        ]
        rois_names =    [
                        # ['V1', 'V2', 'V3',],
                        # ['V1', 'V2', 'V3',],
                        # ['V1', 'V2', 'V3',],
                        # ['V1', 'V2', 'V3',],
                        # ['V1', 'V2', 'V3',],
                        # ['V1', 'V2', 'V3',],
                        # ['aIPS1', 'IPS/PostCeS', 'M1',],
                        # ['IPL', 'SPL', 'aIPS2', 'pIns', 'PreCeS/IFG', 'MFG'],
                        # ['stim V123', 'stim V123 choice', 'choice V123'],
                        ['M1', 'Combined_lat', 'Combined_sl'],
                        ['M1', 'Combined'],
                        ['Combined'],
                        ]
        ylabels =       [
                        # 'Correlation coefficient',
                        # 'Correlation coefficient',
                        # 'Correlation coefficient',
                        # 'fMRI response\n(%signal change)',
                        # 'fMRI response\n(%signal change)',
                        # 'fMRI response\n(%signal change)',
                        # 'Lateralization\n(%signal change)',
                        # 'Correlation coefficient',
                        # 'Choice selective response (a.u.)',
                        'Choice selective response (a.u.)',
                        'Choice selective response (a.u.)',
                        'Choice selective response (a.u.)',
                        ]
        
        for title, rois, rois_name, ylabel in zip(titles, rois_groups, rois_names, ylabels):        
            for inds, trial_type, xlabels, colors in zip([[self.pupil_h_ind, self.pupil_l_ind], [self.yes, self.no], [self.present, self.absent],], ['pupil', 'yes', 'present',], [['high', 'low'], ['yes', 'no'], ['present', 'absent',]],  [['r','b'], ['m','c',], ['orange','forestgreen']]):
                
                
                shell()
                
                # get data in place:
                df = data_frame.ix[:,np.concatenate((np.array(rois), np.array(['subject', 'pupil', 'yes', 'present']))).ravel()]
                if trial_type == 'pupil':
                    df = df[df[trial_type] != 2]
                if title == 'V123_info_stim' or title == 'V123_info_stim_ori':
                    if trial_type == 'pupil':
                        df = df[df['present'] == 1]
                
                # get data in place:
                df = df.ix[:,np.concatenate((np.array(rois), np.array(['subject', trial_type])))]
                k = df.groupby(['subject', trial_type]).mean()
                k_s = k.stack().reset_index()
                k_s.columns = ['subject', trial_type, 'area', 'bold']
                
                # plot:
                locs = np.arange(0,len(rois))
                bar_width = 0.2
                fig = plt.figure(figsize=( (1+(len(rois)*0.3)),1.75))
                ax = fig.add_subplot(111)
                sns.barplot(x='area',  y='bold', units='subject', hue=trial_type, hue_order=[1,0], data=k_s, palette=colors, ci=None, linewidth=0, alpha=0.5, ax=ax)
                
                # add paired observations:
                sns.stripplot(x="area", y="bold", hue=trial_type, hue_order=[1,0], data=k_s, jitter=False, size=2, palette=colors, edgecolor='black', linewidth=0.25, ax=ax, split=True, alpha=1)
                for r in range(len(rois)):
                    values = np.vstack((k_s[(k_s['area'] == rois[r]) & (k_s[trial_type] == 1)].bold, k_s[(k_s['area'] == rois[r]) & (k_s[trial_type] == 0)].bold))
                    x = np.array([locs[r]-bar_width, locs[r]+bar_width])
                    ax.plot(x, values, color='black', lw=0.5, alpha=0.5)
                
                # add p-values:
                for r in range(len(rois)):
                    p1 = myfuncs.permutationTest(k_s[(k_s[trial_type]==0) & (k_s['area']==rois[r])].bold, k_s[(k_s[trial_type]==1) & (k_s['area']==rois[r])].bold, paired=True)[1]
                    # if p1 < 0.05:
                    plt.text(s='{}'.format(round(p1, 3)), x=locs[r], y=gca().get_ylim()[1]-((gca().get_ylim()[1] - gca().get_ylim()[0]) / 10.0), size=5, horizontalalignment='center',)
                ax.legend_.remove()
                plt.xticks(locs, rois_name, rotation=45)
                sns.despine(offset=10, trim=True)
                plt.tight_layout()
                fig.savefig(os.path.join(self.figure_folder, 'lateralization', 'bars_{}_{}.pdf'.format(title, trial_type)))
                
                
                
                
                # anova and planned comparison:
                myfuncs.permutationTest(np.array(k_s.query('(pupil==1) & (area == "combined_choice_parietal_lat")')['bold'])-np.array(k_s.query('(pupil==0) & (area == "combined_choice_parietal_lat")')['bold']),
                                        np.array(k_s.query('(pupil==1) & (area == "lr_M1_lat")')['bold'])-np.array(k_s.query('(pupil==0) & (area == "lr_M1_lat")')['bold']), paired=True)
                myfuncs.permutationTest(np.array(k_s.query('(pupil==1) & (area == "combined_choice_parietal_sl")')['bold'])-np.array(k_s.query('(pupil==0) & (area == "combined_choice_parietal_sl")')['bold']),
                                        np.array(k_s.query('(pupil==1) & (area == "lr_M1_lat")')['bold'])-np.array(k_s.query('(pupil==0) & (area == "lr_M1_lat")')['bold']), paired=True)
                
                
                # ANOVA:
                data = np.concatenate((np.array(k_s.query('(pupil==1) & (area == "lr_M1_lat")')['bold']), np.array(k_s.query('(pupil==0) & (area == "lr_M1_lat")')['bold']), np.array(k_s.query('(pupil==1) & (area == "combined_choice_parietal_lat")')['bold']), np.array(k_s.query('(pupil==0) & (area == "combined_choice_parietal_lat")')['bold']), np.array(k_s.query('(pupil==1) & (area == "combined_choice_parietal_sl")')['bold']), np.array(k_s.query('(pupil==0) & (area == "combined_choice_parietal_sl")')['bold'])))
                subject = np.concatenate((np.arange(len(self.subjects)), np.arange(len(self.subjects)), np.arange(len(self.subjects)), np.arange(len(self.subjects)), np.arange(len(self.subjects)), np.arange(len(self.subjects))))
                component = np.concatenate((np.zeros(len(self.subjects)), np.zeros(len(self.subjects)), np.ones(len(self.subjects)), np.ones(len(self.subjects)), 2*np.ones(len(self.subjects)), 2*np.ones(len(self.subjects))))
                pupil = np.concatenate((np.zeros(len(self.subjects)), np.ones(len(self.subjects)), np.zeros(len(self.subjects)), np.ones(len(self.subjects)), np.zeros(len(self.subjects)), np.ones(len(self.subjects))))
            
                d = rlc.OrdDict([('component', robjects.FactorVector(list(component.ravel()))), ('pupil', robjects.FactorVector(list(pupil.ravel()))), ('subject', robjects.FactorVector(list(subject.ravel()))), ('data', robjects.FloatVector(list(data.ravel())))])
                robjects.r.assign('dataf', robjects.DataFrame(d))
                robjects.r('attach(dataf)')
                statres = robjects.r('res = summary(aov(data ~ component*pupil + Error(subject/(component*pupil)), dataf))')
                # statres = robjects.r('res = summary(aov(data ~ component*pupil + Error(subject), dataf))')
            
                # text_file = open(os.path.join(self.project_directory, 'figures', 'ANOVA_{}.txt'.format(int(1))), 'w')
                # for string in statres:
                #     text_file.write(str(string))
                # text_file.close()
            
            
                print statres
                
                
                
    def CHOICE_SIGNALS_plots_stratified(self, perms=10000):
        
        # load dataframe:
        data_frame = pd.read_csv(os.path.join(self.data_folder, 'event_related_average_wb', 'MNI_dataframe_1.csv'))
        data_frame = data_frame[~np.concatenate(self.omissions_ori2)[~np.concatenate(self.omissions_ori)]]
        dfs = [data_frame[data_frame.subject == s] for s in self.subjects]
        data_frame = pd.concat(dfs)
        
        # add:
        pupil = np.array(np.concatenate(self.pupil_h_ind), dtype=int)
        pupil[~(np.concatenate(self.pupil_l_ind)+np.concatenate(self.pupil_h_ind))] = 2
        data_frame['pupil'] = pupil
        data_frame['present'] = np.concatenate(self.present)
        data_frame['yes'] = np.concatenate(self.yes)
        
        decoding_type = 'info'
        
        titles =        [
                        # 'V123_info_stim',
                        # 'V123_info_stim_ori',
                        # 'V123_info_choice',
                        # 'V123_spat_spec',
                        # 'V123_spat_center',
                        # 'V123_spat_surround',
                        'choice_lat',
                        'choice_info',
                        # 'combined',
                        ]
        rois_groups =   [
                        # ['V1_center_'+decoding_type+'_stim', 'V2_center_'+decoding_type+'_stim', 'V3_center_'+decoding_type+'_stim'],
                        # ['V1_center_'+decoding_type+'_stim_ori', 'V2_center_'+decoding_type+'_stim_ori', 'V3_center_'+decoding_type+'_stim_ori'],
                        # ['V1_center_'+decoding_type+'_choice', 'V2_center_'+decoding_type+'_choice', 'V3_center_'+decoding_type+'_choice'],
                        # ['V1', 'V2', 'V3',],
                        # ['V1_center', 'V2_center', 'V3_center',],
                        # ['V1_surround', 'V2_surround', 'V3_surround',],
                        ['lr_aIPS_lat', 'lr_PCeS_lat', 'lr_M1_lat',],
                        ['sl_IPL_'+decoding_type, 'sl_SPL1_'+decoding_type, 'sl_SPL2_'+decoding_type, 'sl_pIns_'+decoding_type, 'sl_PCeS_IFG_'+decoding_type, 'sl_MFG_'+decoding_type],
                        # ['combined_stim_V123', 'combined_stim_V123_to_choice', 'combined_choice_V123'],
                        ]
        rois_names =    [
                        # ['V1', 'V2', 'V3',],
                        # ['V1', 'V2', 'V3',],
                        # ['V1', 'V2', 'V3',],
                        # ['V1', 'V2', 'V3',],
                        # ['V1', 'V2', 'V3',],
                        # ['V1', 'V2', 'V3',],
                        ['aIPS1', 'IPS/PostCeS', 'M1',],
                        ['IPL', 'SPL', 'aIPS2', 'pIns', 'PreCeS/IFG', 'MFG'],
                        # ['stim V123', 'stim V123 choice', 'choice V123'],
                        ]
        ylabels =       [
                        # 'Correlation coefficient',
                        # 'Correlation coefficient',
                        # 'Correlation coefficient',
                        # 'fMRI response\n(%signal change)',
                        # 'fMRI response\n(%signal change)',
                        # 'fMRI response\n(%signal change)',
                        'Lateralization\n(%signal change)',
                        'Correlation coefficient',
                        # 'Choice selective response (a.u.)',
                        ]
        
        for title, rois, rois_name, ylabel in zip(titles, rois_groups, rois_names, ylabels):        
            for inds, trial_type, xlabels, colors in zip([[self.pupil_h_ind, self.pupil_l_ind],], ['pupil',], [['high', 'low'],],  [['r','b'],]):
                
                # get data in place:
                df = data_frame.ix[:,np.concatenate((np.array(rois), np.array(['subject', 'pupil', 'yes', 'present']))).ravel()]
                if trial_type == 'pupil':
                    df = df[df[trial_type] != 2]
                if title == 'V123_info_stim' or title == 'V123_info_stim_ori':
                    if trial_type == 'pupil':
                        df = df[df['present'] == 1]
                
                # shell()
                
                indds = np.load(os.path.join(self.figure_folder, 'trial_balance_indices.npy'))
                
                # get data in place:
                df = df.ix[:,np.concatenate((np.array(rois), np.array(['subject', trial_type])))]
                
                k = []
                for r in range(indds.shape[0]):
                    d = df[indds[r]]
                    k.append( np.array(d.groupby(['subject', trial_type]).mean().stack().reset_index()) )
                
                k_s = pd.DataFrame({
                    'subject' : df.groupby(['subject', trial_type]).mean().stack().reset_index()['subject'],
                    trial_type: df.groupby(['subject', trial_type]).mean().stack().reset_index()[trial_type],
                    'area' : df.groupby(['subject', trial_type]).mean().stack().reset_index()['level_2'],
                    'bold': pd.Series(np.array(k)[:,:,-1].mean(axis=0)),
                    })
                
                # plot:
                locs = np.arange(0,len(rois))
                bar_width = 0.2
                fig = plt.figure(figsize=( (1+(len(rois)*0.3)),1.75))
                ax = fig.add_subplot(111)
                sns.barplot(x='area',  y='bold', units='subject', hue=trial_type, hue_order=[1,0], data=k_s, palette=colors, ci=None, linewidth=0, alpha=0.5, ax=ax)
                
                # add paired observations:
                sns.stripplot(x="area", y="bold", hue=trial_type, hue_order=[1,0], data=k_s, jitter=False, size=2, palette=colors, edgecolor='black', linewidth=0.25, ax=ax, split=True, alpha=1)
                for r in range(len(rois)):
                    values = np.vstack((k_s[(k_s['area'] == rois[r]) & (k_s[trial_type] == 1)].bold, k_s[(k_s['area'] == rois[r]) & (k_s[trial_type] == 0)].bold))
                    x = np.array([locs[r]-bar_width, locs[r]+bar_width])
                    ax.plot(x, values, color='black', lw=0.5, alpha=0.5)
                
                # add p-values:
                for r in range(len(rois)):
                    p1 = myfuncs.permutationTest(k_s[(k_s[trial_type]==0) & (k_s['area']==rois[r])].bold, k_s[(k_s[trial_type]==1) & (k_s['area']==rois[r])].bold, paired=True)[1]
                    # if p1 < 0.05:
                    plt.text(s='{}'.format(round(p1, 3)), x=locs[r], y=gca().get_ylim()[1]-((gca().get_ylim()[1] - gca().get_ylim()[0]) / 10.0), size=5, horizontalalignment='center',)
                ax.legend_.remove()
                plt.xticks(locs, rois_name, rotation=45)
                sns.despine(offset=10, trim=True)
                plt.tight_layout()
                fig.savefig(os.path.join(self.figure_folder, 'lateralization', 'bars_stratified_{}_{}.pdf'.format(title, trial_type)))       
                
    
    def CHOICE_SIGNALS_stim_TPR_interactions(self):
        
        import rpy2.robjects as robjects
        import rpy2.rlike.container as rlc
        
        # load dataframe:
        data_frame = pd.read_csv(os.path.join(self.data_folder, 'event_related_average_wb', 'MNI_dataframe_1.csv'))
        dfs = [data_frame[data_frame.subject == s] for s in self.subjects]
        
        main_effect = 'stimulus'
        
        # shell()
        
        # for source in ['V1', 'V2', 'V3', 'V1_center', 'V2_center', 'V3_center', 'V1_surround', 'V2_surround', 'V3_surround', 'V1_center_info_choice', 'V2_center_info_choice', 'V3_center_info_choice', 'V1_center_info_stim_ori', 'V2_center_info_stim_ori', 'V3_center_info_stim_ori', 'combined_stim_V123', 'combined_choice_V123', 'combined_choice_parietal']:
        for source in ['V1_center_info_stim_ori', 'V2_center_info_stim_ori', 'V3_center_info_stim_ori',]:
            
            measure_present_h = np.zeros(len(self.subjects))
            measure_present_l = np.zeros(len(self.subjects))
            measure_absent_h = np.zeros(len(self.subjects))
            measure_absent_l = np.zeros(len(self.subjects))
            for i in range(len(self.subjects)):
                subj_ind = np.array((data_frame.subject == self.subjects[i]), dtype=bool)
                
                if main_effect == 'stimulus':
                    measure_present_h[i] = np.array(data_frame[subj_ind][source])[self.present[i]*self.pupil_h_ind[i]].mean()
                    measure_present_l[i] = np.array(data_frame[subj_ind][source])[self.present[i]*self.pupil_l_ind[i]].mean()
                    measure_absent_h[i] = np.array(data_frame[subj_ind][source])[self.absent[i]*self.pupil_h_ind[i]].mean()
                    measure_absent_l[i] = np.array(data_frame[subj_ind][source])[self.absent[i]*self.pupil_l_ind[i]].mean()
                elif main_effect == 'choice':
                    measure_present_h[i] = np.array(data_frame[subj_ind][source])[self.yes[i]*self.pupil_h_ind[i]].mean()
                    measure_present_l[i] = np.array(data_frame[subj_ind][source])[self.yes[i]*self.pupil_l_ind[i]].mean()
                    measure_absent_h[i] = np.array(data_frame[subj_ind][source])[self.no[i]*self.pupil_h_ind[i]].mean()
                    measure_absent_l[i] = np.array(data_frame[subj_ind][source])[self.no[i]*self.pupil_l_ind[i]].mean()
            
            measures = [measure_present_h, measure_present_l, measure_absent_h, measure_absent_l]
            
            # ANOVA:
            data = np.concatenate(( measure_present_h, measure_present_l, measure_absent_h, measure_absent_l ))
            subject = np.concatenate(( np.arange(len(self.subjects)), np.arange(len(self.subjects)), np.arange(len(self.subjects)), np.arange(len(self.subjects)) ))
            pupil = np.concatenate(( np.ones(len(self.subjects)), np.zeros(len(self.subjects)), np.ones(len(self.subjects)), np.zeros(len(self.subjects)) ))
            present = np.concatenate(( np.ones(len(self.subjects)), np.ones(len(self.subjects)), np.zeros(len(self.subjects)), np.zeros(len(self.subjects)) ))
            d = rlc.OrdDict([('present', robjects.FactorVector(list(present.ravel()))), ('pupil', robjects.FactorVector(list(pupil.ravel()))), ('subject', robjects.FactorVector(list(subject.ravel()))), ('data', robjects.FloatVector(list(data.ravel())))])
            robjects.r.assign('dataf', robjects.DataFrame(d))
            robjects.r('attach(dataf)')
            statres = robjects.r('res = summary(aov(data ~ (present * pupil) + Error(subject/(present*pupil)), dataf))')
            print statres
            p1 = round(statres[1][0][4][0],4)
            p2 = round(statres[2][0][4][0],4)
            p3 = round(statres[3][0][4][0],4)
            
            
            
            data = pd.DataFrame(np.array([measures[i] for i in range(len(measures))]).T)
            data.columns = ['present', 'present', 'absent', 'absent',]
            dft = data.stack().reset_index()
            dft.columns = ['subject', 'present', 'value']
            dft['pupil'] = np.concatenate([np.array([1,0,1,0]) for _ in range(len(self.subjects))])
            
            fig = plt.figure(figsize=(2,2))
            ax = fig.add_subplot(111)
            # sns.pointplot(x='pupil',  y='value', hue='present', order=[1,0], units='subject', data=dft, ci=66, alpha=0.5, linewidth=0.5, saturation=0.8, ax=ax)
            sns.stripplot(x='pupil', y='value', hue='present', data=dft, order=[1, 0], hue_order=['present', 'absent'], jitter=False, size=2, edgecolor='black', linewidth=0.25, alpha=1, palette=['orange','forestgreen','orange','forestgreen'], ax=ax)
            values = np.vstack((dft[(dft['pupil'] == 1) & (dft['present'] == 'present')].value, dft[(dft['pupil'] == 0) & (dft['present'] == 'present')].value))
            ax.plot(np.array([0,1]), values, color='black', lw=0.5, alpha=0.5)
            values = np.vstack((dft[(dft['pupil'] == 1) & (dft['present'] == 'absent')].value, dft[(dft['pupil'] == 0) & (dft['present'] == 'absent')].value))
            ax.plot(np.array([0,1]), values, color='black', lw=0.5, alpha=0.5)
            ax.set_ylim(-0.1,0.32)
            sns.despine(offset=10, trim=True)
            # plt.title('{}\n{}, pupil, interaction\np={}, p={}, p={}'.format(source, main_effect, p1, p2, p3))
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, 'V123', 'interaction', 'interaction_pupil_{}_{}.pdf'.format(main_effect, source)))
            
            # shell()
            
            
    
    
    def CHOICE_SIGNALS_coupling_2(self):
        
        # load dataframe:
        data_frame = pd.read_csv(os.path.join(self.data_folder, 'event_related_average_wb', 'MNI_dataframe_1.csv'))
        
        # omissions:
        remove_trials = np.concatenate(self.omissions)[~np.concatenate(self.omissions_ori)]
        data_frame = data_frame[~remove_trials]
        
        dfs = [data_frame[data_frame.subject == s] for s in self.subjects]
                
        coupling_rois_h_present = []
        coupling_rois_l_present = []
        
        n_bins = 7
        # for binned in [True, False]:
        for binned in [False,]:
            for source in ['V1_center_info_stim_ori',]:
            # for source in ['V1','V2','V3','V1_center_info','V2_center_info','V3_center_info']:
            # for source in ['combined_stim_V123']:
            # for source in ['lr_M1_lat']:
            # for source in ['combined_choice_V123']:
                for inds, trial_type in zip([self.all, self.yes, self.no, self.present, self.absent], ['all', 'yes', 'no', 'present', 'absent']):
                    # rois = ['aIPS_lh', 'aIPS_rh', 'aIPS_lat', 'M1_lh', 'M1_rh', 'M1_lat',]
                    # rois = ['aIPS_lat', 'M1_lat', 'aIPS', 'M1',]
                    rois = ['sl_IPL_info', 'sl_SPL1_info', 'lr_aIPS_lat', 'sl_SPL2_info', 'lr_PCeS_lat', 'sl_pIns_info', 'sl_PCeS_IFG_info', 'sl_MFG_info', 'combined_choice_parietal',]
                    roi_names = ['IPL', 'SPL', 'aIPS1', 'aIPS2', 'IPS/PCeS', 'pIns', 'PCeS/IFG', 'MFG', 'Combined']
                    coupling_rois_h = []
                    coupling_rois_l = []
                    ps_h = []
                    ps_l = []
                    ps_hvsl = []
                    for roi in rois:
                        coupling_h = np.zeros(len(self.subjects))
                        coupling_l = np.zeros(len(self.subjects))
                        
                        for i in range(len(self.subjects)):
                            subj_ind = np.array((data_frame.subject == self.subjects[i]), dtype=bool)
                        
                            measure_1_h_hit = np.array(data_frame[subj_ind][source])[self.hit[i]*self.pupil_h_ind[i]]
                            measure_2_h_hit = np.array(data_frame[subj_ind][roi])[self.hit[i]*self.pupil_h_ind[i]]
                            measure_1_h_fa = np.array(data_frame[subj_ind][source])[self.fa[i]*self.pupil_h_ind[i]]
                            measure_2_h_fa = np.array(data_frame[subj_ind][roi])[self.fa[i]*self.pupil_h_ind[i]]
                            measure_1_h_miss = np.array(data_frame[subj_ind][source])[self.miss[i]*self.pupil_h_ind[i]]
                            measure_2_h_miss = np.array(data_frame[subj_ind][roi])[self.miss[i]*self.pupil_h_ind[i]]
                            measure_1_h_cr = np.array(data_frame[subj_ind][source])[self.cr[i]*self.pupil_h_ind[i]]
                            measure_2_h_cr = np.array(data_frame[subj_ind][roi])[self.cr[i]*self.pupil_h_ind[i]]
                        
                            measure_1_l_hit = np.array(data_frame[subj_ind][source])[self.hit[i]*self.pupil_l_ind[i]]
                            measure_2_l_hit = np.array(data_frame[subj_ind][roi])[self.hit[i]*self.pupil_l_ind[i]]
                            measure_1_l_fa = np.array(data_frame[subj_ind][source])[self.fa[i]*self.pupil_l_ind[i]]
                            measure_2_l_fa = np.array(data_frame[subj_ind][roi])[self.fa[i]*self.pupil_l_ind[i]]
                            measure_1_l_miss = np.array(data_frame[subj_ind][source])[self.miss[i]*self.pupil_l_ind[i]]
                            measure_2_l_miss = np.array(data_frame[subj_ind][roi])[self.miss[i]*self.pupil_l_ind[i]]
                            measure_1_l_cr = np.array(data_frame[subj_ind][source])[self.cr[i]*self.pupil_l_ind[i]]
                            measure_2_l_cr = np.array(data_frame[subj_ind][roi])[self.cr[i]*self.pupil_l_ind[i]]
                        
                            # if trial_type == 'present':
                            #     coupling_h[i] = (sp.stats.pearsonr(measure_1_h_hit, measure_2_h_hit)[0] + sp.stats.pearsonr(measure_1_h_miss, measure_2_h_miss)[0]) / 2.0
                            #     coupling_l[i] = (sp.stats.pearsonr(measure_1_l_hit, measure_2_l_hit)[0] + sp.stats.pearsonr(measure_1_l_miss, measure_2_l_miss)[0]) / 2.0
                            #
                            # elif trial_type == 'absent':
                            #     coupling_h[i] = (sp.stats.pearsonr(measure_1_h_fa, measure_2_h_fa)[0] + sp.stats.pearsonr(measure_1_h_cr, measure_2_h_cr)[0]) / 2.0
                            #     coupling_l[i] = (sp.stats.pearsonr(measure_1_l_fa, measure_2_l_fa)[0] + sp.stats.pearsonr(measure_1_l_cr, measure_2_l_cr)[0]) / 2.0
                            #
                            # elif trial_type == 'yes':
                            #     coupling_h[i] = (sp.stats.pearsonr(measure_1_h_fa, measure_2_h_fa)[0] + sp.stats.pearsonr(measure_1_h_hit, measure_2_h_hit)[0]) / 2.0
                            #     coupling_l[i] = (sp.stats.pearsonr(measure_1_l_fa, measure_2_l_fa)[0] + sp.stats.pearsonr(measure_1_l_hit, measure_2_l_hit)[0]) / 2.0
                            #
                            # elif trial_type == 'no':
                            #     coupling_h[i] = (sp.stats.pearsonr(measure_1_h_miss, measure_2_h_miss)[0] + sp.stats.pearsonr(measure_1_h_cr, measure_2_h_cr)[0]) / 2.0
                            #     coupling_l[i] = (sp.stats.pearsonr(measure_1_l_miss, measure_2_l_miss)[0] + sp.stats.pearsonr(measure_1_l_cr, measure_2_l_cr)[0]) / 2.0
                            #
                            # else:
                            measure_1_h = np.array(data_frame[subj_ind][source])[inds[i]*self.pupil_h_ind[i]]
                            measure_2_h = np.array(data_frame[subj_ind][roi])[inds[i]*self.pupil_h_ind[i]]
                            measure_1_l = np.array(data_frame[subj_ind][source])[inds[i]*self.pupil_l_ind[i]]
                            measure_2_l = np.array(data_frame[subj_ind][roi])[inds[i]*self.pupil_l_ind[i]]
                            coupling_h[i] = sp.stats.pearsonr(measure_1_h, measure_2_h)[0]
                            coupling_l[i] = sp.stats.pearsonr(measure_1_l, measure_2_l)[0]
                        
                        # if trial_type == 'present' or trial_type == 'absent':
                        if trial_type == 'yes' or trial_type == 'no':
                            coupling_rois_h_present.append( coupling_h ) 
                            coupling_rois_l_present.append( coupling_l ) 
                            
                        
                        coupling_rois_h.append(coupling_h)
                        coupling_rois_l.append(coupling_l)
                        # ps_h.append( myfuncs.permutationTest(coupling_h, np.zeros(len(coupling_h)), paired=True)[1] )
                        # ps_l.append( myfuncs.permutationTest(coupling_l, np.zeros(len(coupling_l)), paired=True)[1] )
                        # ps_hvsl.append( myfuncs.permutationTest(coupling_h, coupling_l, paired=True)[1] )
                        ps_h.append( sp.stats.wilcoxon(coupling_h, np.zeros(len(coupling_h)),)[1] )
                        ps_l.append(sp.stats.wilcoxon(coupling_l, np.zeros(len(coupling_l)),)[1] )
                        ps_hvsl.append( sp.stats.wilcoxon(coupling_h, coupling_l,)[1] )
                    MEANS_h = [coupling_rois_h[i].mean() for i in range(len(coupling_rois_h))]
                    SEMS_h = [sp.stats.sem(coupling_rois_h[i]) for i in range(len(coupling_rois_h))]
                    MEANS_l = [coupling_rois_l[i].mean() for i in range(len(coupling_rois_l))]
                    SEMS_l = [sp.stats.sem(coupling_rois_l[i]) for i in range(len(coupling_rois_l))]
                    locs = np.arange(0,len(MEANS_h))
                    bar_width = 0.45
                    fig = plt.figure(figsize=( (1+(len(rois)*0.3)),1.75))
                    my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}
                    for i in range(len(MEANS_h)):
                        plt.bar(locs[i]-(bar_width/2.0), MEANS_h[i], width = bar_width, yerr = SEMS_h[i], color = 'r', alpha = 1, align = 'center', edgecolor = 'k', linewidth = 0, capsize = 0, ecolor = 'k', error_kw={'elinewidth':0.5,})
                    for i in range(len(MEANS_l)):
                        plt.bar(locs[i]+(bar_width/2.0), MEANS_l[i], width = bar_width, yerr = SEMS_l[i], color = 'b', alpha = 1, align = 'center', edgecolor = 'k', linewidth = 0, capsize = 0, ecolor = 'k', error_kw={'elinewidth':0.5,})
                    # plt.ylim(-0.05,0.1)
                    for i, p in enumerate(ps_hvsl):
                        # if p < 0.05:
                        plt.text(s='{}'.format(round(p, 3)), x=locs[i], y=gca().get_ylim()[1]-((gca().get_ylim()[1] - gca().get_ylim()[0]) / 20.0), size=5, color='k', horizontalalignment='center', )
                    for i, p in enumerate(ps_h):
                        if p < 0.05:
                            plt.text(s='{}'.format(round(p, 3)), x=locs[i]-(bar_width/2.0), y=0, size=5, color='k', horizontalalignment='center', rotation=90)
                    # if xlabels[0] != 'all':
                        for i, p in enumerate(ps_l):
                            if p < 0.05:
                                plt.text(s='{}'.format(round(p, 3)), x=locs[i]+(bar_width/2.0), y=0, size=5, color='k', horizontalalignment='center', rotation=90 )
                    plt.title('{}; {}'.format(source, trial_type))
                    plt.ylabel('Coupling')
                    sns.despine(offset=10, trim=True)
                    plt.xticks(locs, roi_names, rotation=45)
                    plt.tight_layout()
                    fig.savefig(os.path.join(self.figure_folder, 'lateralization', 'couplings_{}_{}_{}.pdf'.format(source, trial_type, int(binned))))
    
        # # ANOVA:
        # data = np.concatenate(( coupling_rois_h_present[0], coupling_rois_h_present[1], coupling_rois_l_present[0], coupling_rois_l_present[1] ))
        # subject = np.concatenate(( np.arange(len(self.subjects)), np.arange(len(self.subjects)), np.arange(len(self.subjects)), np.arange(len(self.subjects)) ))
        # pupil = np.concatenate(( np.ones(len(self.subjects)), np.ones(len(self.subjects)), np.zeros(len(self.subjects)), np.zeros(len(self.subjects)) ))
        # present = np.concatenate(( np.ones(len(self.subjects)), np.zeros(len(self.subjects)), np.ones(len(self.subjects)), np.zeros(len(self.subjects)) ))
        # d = rlc.OrdDict([('present', robjects.FactorVector(list(present.ravel()))), ('pupil', robjects.FactorVector(list(pupil.ravel()))), ('subject', robjects.FactorVector(list(subject.ravel()))), ('data', robjects.FloatVector(list(data.ravel())))])
        # robjects.r.assign('dataf', robjects.DataFrame(d))
        # robjects.r('attach(dataf)')
        #
        # statres = robjects.r('res = summary(aov(data ~ (present * pupil) + Error(subject/(present*pupil)), dataf))')
        # print statres
        #
        # myfuncs.permutationTest(coupling_rois_h_present[1], coupling_rois_l_present[1], paired=True)
        
    def CHOICE_SIGNALS_coupling(self):
        
        # load dataframe:
        data_frame = pd.read_csv(os.path.join(self.data_folder, 'event_related_average_wb', 'MNI_dataframe_1.csv'))
        dfs = [data_frame[data_frame.subject == s] for s in self.subjects]
        
        shell()
        
        n_bins = 7
        # for binned in [True, False]:
        for binned in [False,]:
            # for source in ['V123_center_info',]:
            # for source in ['V1','V2','V3','V1_center_info','V2_center_info','V3_center_info']:
            for source in ['combined_stim']:
                for inds, xlabels, colors, alphas, trial_type in zip([[self.all, self.all], [self.pupil_h_ind, self.pupil_l_ind], [self.yes, self.no]], [['all', 'all'], ['high', 'low'], ['yes', 'no']], [['k', 'k'], ['r', 'b'], ['m', 'c'],], [[1,1], [1,1], [1,1],], ['all', 'pupil', 'choice']):
                    # rois = ['aIPS_lh', 'aIPS_rh', 'aIPS_lat', 'M1_lh', 'M1_rh', 'M1_lat',]
                    # rois = ['aIPS_lat', 'M1_lat', 'aIPS', 'M1',]
                    rois = ['combined_all_2',]
                    roi_names = rois
                    coupling_rois_h = []
                    coupling_rois_l = []
                    ps_h = []
                    ps_l = []
                    ps_hvsl = []
                    for roi in rois:
                        # if binned:
                        #     measure_1_b_h = np.zeros((len(self.subjects), n_bins))
                        #     measure_2_b_h = np.zeros((len(self.subjects), n_bins))
                        #     measure_1_b_l = np.zeros((len(self.subjects), n_bins))
                        #     measure_2_b_l = np.zeros((len(self.subjects), n_bins))
                        #     for i in range(len(self.subjects)):
                        #         subj_ind = np.array((data_frame.subject == self.subjects[i]), dtype=bool)
                        #         measure_1_h = np.array(data_frame[subj_ind][source])[inds[0][i]]
                        #         measure_2_h = np.array(data_frame[subj_ind][roi])[inds[0][i]]
                        #         split_indices_h = np.array_split(np.argsort(measure_1_h), n_bins)
                        #         measure_1_l = np.array(data_frame[subj_ind][source])[inds[1][i]]
                        #         measure_2_l = np.array(data_frame[subj_ind][roi])[inds[1][i]]
                        #         split_indices_l = np.array_split(np.argsort(measure_1_l), n_bins)
                        #         # split_indices = np.array_split(np.argsort(self.pupil_data[i]['pupil_d']), n_bins)
                        #         for b in range(n_bins):
                        #             measure_1_b_h[i,b] = np.mean(measure_1_h[split_indices_h[b]])
                        #             measure_2_b_h[i,b] = np.mean(measure_2_h[split_indices_h[b]])
                        #             measure_1_b_l[i,b] = np.mean(measure_1_l[split_indices_l[b]])
                        #             measure_2_b_l[i,b] = np.mean(measure_2_l[split_indices_l[b]])
                        #     measure_1_h = measure_1_b_h.mean(axis=0)
                        #     measure_2_h = measure_2_b_h.mean(axis=0)
                        #     measure_1_l = measure_1_b_l.mean(axis=0)
                        #     measure_2_l = measure_2_b_l.mean(axis=0)
                        #     r_h, p_h = sp.stats.pearsonr(measure_1_h, measure_2_h)
                        #     r_l, p_l = sp.stats.pearsonr(measure_1_l, measure_2_l)
                        #     coupling_rois_h.append(np.repeat(r_h, len(self.subjects)))
                        #     coupling_rois_l.append(np.repeat(r_l, len(self.subjects)))
                        #     ps_h.append(p_h)
                        #     ps_l.append(p_l)
                        # else:
                        coupling_h = np.zeros(len(self.subjects))
                        coupling_l = np.zeros(len(self.subjects))
                        for i in range(len(self.subjects)):
                            subj_ind = np.array((data_frame.subject == self.subjects[i]), dtype=bool)
                            
                            if trial_type=='choice':
                                measure_1_h = np.array(data_frame[subj_ind][source])[self.hit[i]]
                                measure_2_h = np.array(data_frame[subj_ind][roi])[self.hit[i]]
                                measure_1_fa = np.array(data_frame[subj_ind][source])[self.fa[i]]
                                measure_2_fa = np.array(data_frame[subj_ind][roi])[self.fa[i]]
                                measure_1_miss = np.array(data_frame[subj_ind][source])[self.miss[i]]
                                measure_2_miss = np.array(data_frame[subj_ind][roi])[self.miss[i]]
                                measure_1_cr = np.array(data_frame[subj_ind][source])[self.cr[i]]
                                measure_2_cr = np.array(data_frame[subj_ind][roi])[self.cr[i]]
                                coupling_h[i] = (sp.stats.pearsonr(measure_1_h, measure_2_h)[0] + sp.stats.pearsonr(measure_1_fa, measure_2_fa)[0]) / 2.0
                                coupling_l[i] = (sp.stats.pearsonr(measure_1_miss, measure_2_miss)[0] + sp.stats.pearsonr(measure_1_cr, measure_2_cr)[0]) / 2.0
                                # coupling_h[i] = np.cov(measure_1_h, measure_2_h)[0,1]
                                # coupling_l[i] = np.cov(measure_1_l, measure_2_l)[0,1]
                            else:
                                measure_1_h = np.array(data_frame[subj_ind][source])[inds[0][i]]
                                measure_2_h = np.array(data_frame[subj_ind][roi])[inds[0][i]]
                                measure_1_l = np.array(data_frame[subj_ind][source])[inds[1][i]]
                                measure_2_l = np.array(data_frame[subj_ind][roi])[inds[1][i]]
                                coupling_h[i] = sp.stats.pearsonr(measure_1_h, measure_2_h)[0]
                                coupling_l[i] = sp.stats.pearsonr(measure_1_l, measure_2_l)[0]
                                # coupling_h[i] = np.cov(measure_1_h, measure_2_h)[0,1]
                                # coupling_l[i] = np.cov(measure_1_l, measure_2_l)[0,1]
                        coupling_rois_h.append(coupling_h)
                        coupling_rois_l.append(coupling_l)
                        ps_h.append( myfuncs.permutationTest(coupling_h, np.zeros(len(coupling_h)), paired=True)[1] )
                        ps_l.append( myfuncs.permutationTest(coupling_l, np.zeros(len(coupling_l)), paired=True)[1] )
                        ps_hvsl.append( myfuncs.permutationTest(coupling_h, coupling_l, paired=True)[1] )
                        # ps_h.append( sp.stats.wilcoxon(coupling_h, np.zeros(len(coupling_h)),)[1] )
                        # ps_l.append(sp.stats.wilcoxon(coupling_l, np.zeros(len(coupling_l)),)[1] )
                        # ps_hvsl.append( sp.stats.wilcoxon(coupling_h, coupling_l,)[1] )
                    MEANS_h = [coupling_rois_h[i].mean() for i in range(len(coupling_rois_h))]
                    SEMS_h = [sp.stats.sem(coupling_rois_h[i]) for i in range(len(coupling_rois_h))]
                    MEANS_l = [coupling_rois_l[i].mean() for i in range(len(coupling_rois_l))]
                    SEMS_l = [sp.stats.sem(coupling_rois_l[i]) for i in range(len(coupling_rois_l))]
                    locs = np.arange(0,len(MEANS_h))
                    bar_width = 0.45
                    fig = plt.figure(figsize=(1.75,2))
                    my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}
                    for i in range(len(MEANS_h)):
                        plt.bar(locs[i]-(bar_width/2.0), MEANS_h[i], width = bar_width, yerr = SEMS_h[i], color = colors[0], alpha = alphas[0], align = 'center', edgecolor = 'k', linewidth = 0, capsize = 0, ecolor = 'k', error_kw={'elinewidth':0.5,})
                    if xlabels[0] != 'all':
                        for i in range(len(MEANS_l)):
                            plt.bar(locs[i]+(bar_width/2.0), MEANS_l[i], width = bar_width, yerr = SEMS_l[i], color = colors[1], alpha = alphas[1], align = 'center', edgecolor = 'k', linewidth = 0, capsize = 0, ecolor = 'k', error_kw={'elinewidth':0.5,})
                    # plt.ylim(-0.05,0.1)
                    for i, p in enumerate(ps_hvsl):
                        if p < 0.05:
                            plt.text(s='{}'.format(round(p, 3)), x=locs[i], y=gca().get_ylim()[1]-((gca().get_ylim()[1] - gca().get_ylim()[0]) / 20.0), size=5, color='k', horizontalalignment='center', )
                    for i, p in enumerate(ps_h):
                        if p < 0.05:
                            plt.text(s='{}'.format(round(p, 3)), x=locs[i]-(bar_width/2.0), y=0, size=5, color='k', horizontalalignment='center', rotation=90)
                    if xlabels[0] != 'all':
                        for i, p in enumerate(ps_l):
                            if p < 0.05:
                                plt.text(s='{}'.format(round(p, 3)), x=locs[i]+(bar_width/2.0), y=0, size=5, color='k', horizontalalignment='center', rotation=90 )
                    plt.title('{}; {}'.format(source, trial_type))
                    plt.ylabel('Coupling')
                    sns.despine(offset=10, trim=True)
                    plt.xticks(locs, roi_names, rotation=45)
                    plt.tight_layout()
                    fig.savefig(os.path.join(self.figure_folder, 'lateralization', 'couplings_{}_{}_{}.pdf'.format(source, trial_type, int(binned))))
        
        # # COUPLING split by trial type:
        # for source in ['V1_center_info', 'V2_center_info', 'V3_center_info', 'V123_center_info', 'V123',]:
        #     for inds, trial_type, in zip([self.pupil_h_ind, self.yes, self.present,], ['pupil', 'yes', 'present']):
        #         # rois = ['ips_anterior_lat', 'ips_posterior_lat', 'postcentral_sulcus_lat', 'parietal_lat', 'precentral_sulcus_lat', 'insula_lat',]
        #         # rois = ['lh.ips_anterior', 'lh.ips_posterior', 'lh.postcentral_sulcus', 'lh.parietal', 'lh.precentral_sulcus', 'lh.insula',]
        #         # rois = ['rh.ips_anterior', 'rh.ips_posterior', 'rh.postcentral_sulcus', 'rh.parietal', 'rh.precentral_sulcus', 'rh.insula',]
        #         # roi_names = ['IPS_ant', 'IPS_post', 'post_CS', 'combined', 'pre_CS', 'insula',]
        #
        #         rois = ['postcentral_sulcus_lat', 'parietal_lat', 'precentral_sulcus_lat',]
        #         # rois = ['lh.postcentral_sulcus', 'lh.parietal', 'lh.precentral_sulcus',]
        #         # rois = ['rh.postcentral_sulcus', 'rh.parietal', 'rh.precentral_sulcus',]
        #         roi_names = ['post_CS', 'combined', 'pre_CS',]
        #
        #
        #
        #         coupling_rois_1 = []
        #         ps_1 = []
        #         for roi in rois:
        #             coupling_1 = np.zeros(len(self.subjects))
        #             for i in range(len(self.subjects)):
        #                 subj_ind = np.array((data_frame.subject == self.subjects[i]), dtype=bool)
        #                 phasics = np.array(data_frame[subj_ind][roi])[inds[i]]
        #                 phasics_V1 = np.array(data_frame[subj_ind][source])[inds[i]]
        #                 coupling_1[i] = sp.stats.pearsonr(phasics_V1, phasics)[0]
        #                 # coupling_1[i] = np.cov(phasics_V1, phasics)[0][1]
        #             coupling_rois_1.append(coupling_1)
        #             ps_1.append( myfuncs.permutationTest(coupling_1, np.zeros(len(coupling_1)), paired=True)[1] )
        #
        #         coupling_rois_0 = []
        #         ps_0 = []
        #         for roi in rois:
        #             coupling_0 = np.zeros(len(self.subjects))
        #             for i in range(len(self.subjects)):
        #                 subj_ind = np.array((data_frame.subject == self.subjects[i]), dtype=bool)
        #                 phasics = np.array(data_frame[subj_ind][roi])[~inds[i]]
        #                 phasics_V1 = np.array(data_frame[subj_ind][source])[~inds[i]]
        #                 coupling_0[i] = sp.stats.pearsonr(phasics_V1, phasics)[0]
        #                 # coupling_0[i] = np.cov(phasics_V1, phasics)[0][1]
        #             coupling_rois_0.append(coupling_0)
        #             ps_0.append( myfuncs.permutationTest(coupling_0, np.zeros(len(coupling_0)), paired=True)[1] )
        #
        #         ps = []
        #         for i in range(len(coupling_rois_0)):
        #             ps.append(myfuncs.permutationTest(coupling_rois_0[i], coupling_rois_1[i], paired=True)[1])
        #
        #         MEANS_0 = [coupling_rois_0[i].mean() for i in range(len(coupling_rois_0))]
        #         SEMS_0 = [sp.stats.sem(coupling_rois_0[i]) for i in range(len(coupling_rois_0))]
        #         MEANS_1 = [coupling_rois_1[i].mean() for i in range(len(coupling_rois_1))]
        #         SEMS_1 = [sp.stats.sem(coupling_rois_1[i]) for i in range(len(coupling_rois_1))]
        #
        #         locs = np.arange(0,len(MEANS_0))
        #         bar_width = 0.4
        #         fig = plt.figure(figsize=(2,2))
        #         my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}
        #         for i in range(len(MEANS_1)):
        #             plt.bar(locs[i]-(bar_width/2), MEANS_1[i], width = bar_width, yerr = SEMS_1[i], color = 'k', alpha = 1, align = 'center', edgecolor = 'k', linewidth = 0, capsize = 0, ecolor = 'k', error_kw={'elinewidth':0.5,})
        #         for i in range(len(MEANS_0)):
        #             plt.bar(locs[i]+(bar_width/2), MEANS_0[i], width = bar_width, yerr = SEMS_0[i], color = 'k', alpha = 0.5, align = 'center', edgecolor = 'k', linewidth = 0, capsize = 0, ecolor = 'k', error_kw={'elinewidth':0.5,})
        #
        #         for i, p in enumerate(ps):
        #             if p < 0.05:
        #                 plt.text(s='{}'.format(round(p, 3)), x=locs[i], y=gca().get_ylim()[1]-((gca().get_ylim()[1] - gca().get_ylim()[0]) / 10.0), size=5)
        #
        #         plt.title('{}\n{}'.format(source, trial_type))
        #         plt.ylabel('Coupling')
        #         sns.despine(offset=10, trim=True)
        #         plt.xticks(locs, roi_names, rotation=90)
        #         plt.tight_layout()
        #         fig.savefig(os.path.join(self.figure_folder, 'lateralization', 'couplings_{}_{}.pdf'.format(source, trial_type)))
        
    def CHOICE_SIGNALS_correlation_matrix(self):
        
        data_frame = pd.read_csv(os.path.join(self.data_folder, 'event_related_average_wb', 'MNI_dataframe_1.csv'))
        
        # omissions:
        remove_trials = np.concatenate(self.omissions)[~np.concatenate(self.omissions_ori)]
        data_frame = data_frame[~remove_trials]
        
        dfs = [data_frame[data_frame.subject == s] for s in self.subjects]
        
        # title = 'V123_stim'
        # rois = ['V1_center_info_stim_ori', 'V2_center_info_stim_ori', 'V3_center_info_stim_ori',]
        # roi_names = ['V1', 'V2', 'V3',]
        
        title = 'V123_choice'
        rois = ['V1_center_info_choice', 'V2_center_info_choice', 'V3_center_info_choice',]
        roi_names = ['V1', 'V2', 'V3',]
        
        partial=False
        # shell()
        
        # make correlation matrix:
        # ------------------------
        contrast_across_dvs = []
        # for dv in ['cor', 'mean', 'var', 'cov', 'snr']:
        for dv in ['cor',]:
            for inds, trial_type in zip([np.concatenate(self.all), np.concatenate(self.yes), np.concatenate(self.no), np.concatenate(self.present), np.concatenate(self.absent)], ['all', 'yes', 'no', 'present', 'absent']):
                cormats = []
                for condition in [np.concatenate(self.pupil_h_ind), np.concatenate(self.pupil_l_ind)]:
                    ind = condition * inds
                    cm = np.zeros((len(rois), len(rois), len(self.subjects)))
                    cm_p = np.zeros((len(rois), len(rois), len(self.subjects)))
                    for i in range(len(self.subjects)):
                        C = np.vstack([np.array(data_frame[(data_frame.subject == self.subjects[i])*ind][roi]) for roi in rois]).T
                        if dv == 'cor':
                            if partial:
                                cm[:,:,i], cm_p[:,:,i] = myfuncs.corr_matrix_partial(C)
                            else:
                                cm[:,:,i], cm_p[:,:,i] = myfuncs.corr_matrix(C, dv='cor')
                        if dv == 'var':
                            cm[:,:,i], cm_p[:,:,i] = myfuncs.corr_matrix(C, dv='var')
                        if dv == 'cov':
                            cm[:,:,i], cm_p[:,:,i] = myfuncs.corr_matrix(C, dv='cov')
                        if dv == 'mean':
                            cm[:,:,i], cm_p[:,:,i] = myfuncs.corr_matrix(C, dv='mean')
                        if dv == 'snr':
                            cm[:,:,i], cm_p[:,:,i] = myfuncs.corr_matrix(C, dv='snr')
                    cormats.append(cm)

                corrmats_mean = (cormats[0]+cormats[1]) / 2.0
                corrmats_mean_av = corrmats_mean.mean(axis=-1)
                corrmats_contrast = cormats[0]-cormats[1]
                corrmats_contrast_av = corrmats_contrast.mean(axis=-1)

                contrast_across_dvs.append(corrmats_contrast_av)

                p_mat_contrast = np.zeros(corrmats_mean_av.shape)
                p_mat_mean = np.zeros(corrmats_mean_av.shape)
                for i in range(p_mat_mean.shape[0]):
                    for j in range(i, p_mat_mean.shape[1]):
                        try:
                            # p_mat_mean[i,j] = myfuncs.permutationTest(corrmats_mean[i,j,:],np.zeros(corrmats_mean[i,j,:].shape[0]), paired=True)[1]
                            # p_mat_contrast[i,j] = myfuncs.permutationTest(corrmats_contrast[i,j,:],np.zeros(corrmats_contrast[i,j,:].shape[0]), paired=True)[1]
                            p_mat_mean[i,j] = sp.stats.wilcoxon(corrmats_mean[i,j,:],np.zeros(corrmats_mean[i,j,:].shape[0]),)[1]
                            p_mat_contrast[i,j] = sp.stats.wilcoxon(corrmats_contrast[i,j,:],np.zeros(corrmats_contrast[i,j,:].shape[0]),)[1]
                        except:
                            p_mat[i,j] = 1

                # # fdr correction:
                # mask =  np.array(np.tri(corrmats_mean_av.shape[0], k=-1), dtype=bool)
                # p = p_mat[mask].ravel()
                # p = mne.stats.fdr_correction(p, 0.05)[1]
                
                # plot matrix:
                if dv == 'mean':
                    mask =  np.tri(corrmats_mean_av.shape[0], k=-1)
                    mask = mask + (p_mat_mean>0.05)
                else:
                    mask = np.tri(corrmats_mean_av.shape[0], k=0)
                    mask = mask + (p_mat_mean>0.05)
                corrmat_m = np.ma.masked_where(mask, corrmats_mean_av)
                p_mat_m = np.ma.masked_where(mask.T, p_mat_mean)
                fig = plt.figure(figsize=(3,2.5))
                ax = fig.add_subplot(111)
                if dv == 'mean':
                    vmax = 0.5
                elif dv == 'snr':
                    vmax = 0.2
                else:
                    vmax = max(abs(corrmats_mean_av.ravel()))
                if (dv == 'cor') or (dv == 'mean') or (dv == 'snr'):
                    im = ax.pcolormesh(corrmat_m, cmap='bwr', vmin=-vmax, vmax=vmax)
                else:
                    im = ax.pcolormesh(corrmat_m, cmap='Reds', vmin=0, vmax=vmax)
                ax.set_xlim(xmax=len(rois))
                ax.set_ylim(ymax=len(rois))
                ax.set_yticks(arange(0.5,len(rois)+.5))
                ax.set_xticks(arange(0.5,len(rois)+.5))
                ax.set_yticklabels(roi_names)
                ax.set_xticklabels(roi_names, rotation=270)
                ax.set_title(trial_type)
                ax.patch.set_hatch('x')
                ax.set_aspect(1)
                fig.colorbar(im)
                # for l in boundaries:
                #     plt.vlines(l, 0, l, lw=1)
                #     plt.hlines(l, l, corrmat_m.shape[0], lw=1)
                plt.tight_layout()
                # fig.savefig(os.path.join(self.figure_folder, 'lateralization', 'matrix_single_{}_{}_{}_{}.pdf'.format(self.split_by, 'all', dv, trial_type)))
                fig.savefig(os.path.join(self.figure_folder, 'lateralization', 'matrix_single_{}_{}_{}_{}_{}2.pdf'.format(title, self.split_by, 'all', dv, trial_type)))

                # contrast matrix:
                if dv == 'mean':
                    mask =  np.tri(corrmats_mean_av.shape[0], k=-1)
                    # mask = mask + (p_mat_contrast>0.05)
                else:
                    mask =  np.tri(corrmats_mean_av.shape[0], k=0)
                    # mask = mask + (p_mat_contrast>0.05)
                corrmat_m = np.ma.masked_where(mask, corrmats_contrast_av)
                p_mat_m = np.ma.masked_where(mask.T, p_mat_contrast)
                fig = plt.figure(figsize=(3,2.5))
                # fig = plt.figure(figsize=(10,8))
                ax = fig.add_subplot(111)
                vmax = max(abs(corrmats_contrast_av.ravel()))
                im = ax.pcolormesh(corrmat_m, cmap='bwr', vmin=-vmax, vmax=vmax)
                ax.set_xlim(xmax=len(rois))
                ax.set_ylim(ymax=len(rois))
                ax.set_yticks(arange(0.5,len(rois)+.5))
                ax.set_xticks(arange(0.5,len(rois)+.5))
                ax.set_yticklabels(roi_names)
                ax.set_xticklabels(roi_names, rotation=270)
                ax.set_title(trial_type)
                ax.patch.set_hatch('x')
                ax.set_aspect(1)
                fig.colorbar(im)
                # for l in boundaries:
                #     plt.vlines(l, 0, l, lw=1)
                #     plt.hlines(l, l, corrmat_m.shape[0], lw=1)
                plt.tight_layout()
                # fig.savefig(os.path.join(self.figure_folder, 'lateralization', 'matrix_single_{}_{}_{}_{}.pdf'.format(self.split_by, 'contrast', dv, trial_type)))
                fig.savefig(os.path.join(self.figure_folder, 'lateralization', 'matrix_single_{}_{}_{}_{}_{}2.pdf'.format(title, self.split_by, 'contrast', dv, trial_type)))
                
                # shell()
                
                
                # # stats on cells:
                # nr_pos_cells = np.zeros(len(self.subjects))
                # nr_neg_cells = np.zeros(len(self.subjects))
                # for i in range(len(self.subjects)):
                #     nr_pos_cells[i] = np.sum(corrmats_contrast[:,:,i].ravel()>0) / 2.0
                #     nr_neg_cells[i] = np.sum(corrmats_contrast[:,:,i].ravel()<0) / 2.0
                #
                # nr_pos_cells = nr_pos_cells / (((len(rois)*len(rois)) - len(rois)) / 2) * 100
                # nr_neg_cells = nr_neg_cells / (((len(rois)*len(rois)) - len(rois)) / 2) * 100
                #
                # MEANS = (nr_pos_cells.mean(), nr_neg_cells.mean())
                # SEMS = (sp.stats.sem(nr_pos_cells), sp.stats.sem(nr_neg_cells))
                # N = 2
                # ind = np.linspace(0,N/2,N)
                # bar_width = 0.50
                # fig = plt.figure(figsize=(1.25,1.75))
                # ax = fig.add_subplot(111)
                # for i in range(N):
                #     ax.bar(ind[i], height=MEANS[i], width = bar_width, yerr = SEMS[i], color = ['r','b'][i], alpha = 1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
                # ax.set_title('N={}'.format(len(self.subjects)), size=7)
                # ax.set_ylabel('number of cells', size=7)
                # ax.set_xticks( (ind) )
                # ax.set_xticklabels( ('pos', 'neg') )
                # plt.text(x=np.mean((ind[0], ind[1])), y=ax.axis()[3] - ((ax.axis()[3]-ax.axis()[2]) / 10.0), s='p = {}'.format(round(myfuncs.permutationTest(nr_pos_cells, nr_neg_cells)[1],3)), horizontalalignment='center')
                # ax.set_ylim(ymax=75)
                # locator = plt.MaxNLocator(nbins=3) # with 3 bins you will have 4 ticks
                # ax.yaxis.set_major_locator(locator)
                # sns.despine(offset=10, trim=True)
                # plt.tight_layout()
                # fig.savefig(os.path.join(self.figure_folder, 'lateralization', 'bars_single_{}_{}_{}_{}_{}2.pdf'.format(title, self.split_by, 'contrast', dv, trial_type)))
                #
                #
                #
                # # stats on cells:
                # nr_pos_cells = np.zeros(len(self.subjects))
                # nr_neg_cells = np.zeros(len(self.subjects))
                # for i in range(len(self.subjects)):
                #     nr_pos_cells[i] = np.mean(corrmats_contrast[:,:,i].ravel())
                #     nr_neg_cells[i] = np.mean(corrmats_contrast[:,:,i].ravel())
                #
                # MEANS = (nr_pos_cells.mean(), nr_neg_cells.mean())
                # SEMS = (sp.stats.sem(nr_pos_cells), sp.stats.sem(nr_neg_cells))
                # N = 2
                # ind = np.linspace(0,N/2,N)
                # bar_width = 0.50
                # fig = plt.figure(figsize=(1.25,1.75))
                # ax = fig.add_subplot(111)
                # for i in range(N):
                #     ax.bar(ind[i], height=MEANS[i], width = bar_width, yerr = SEMS[i], color = ['r','b'][i], alpha = 1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
                # ax.set_title('N={}'.format(len(self.subjects)), size=7)
                # ax.set_ylabel('number of cells', size=7)
                # ax.set_xticks( (ind) )
                # ax.set_xticklabels( ('pos', 'neg') )
                # plt.text(x=np.mean((ind[0], ind[1])), y=ax.axis()[3] - ((ax.axis()[3]-ax.axis()[2]) / 10.0), s='p = {}'.format(round(myfuncs.permutationTest(nr_pos_cells, np.zeros(len(nr_pos_cells)), paired=True)[1],3)), horizontalalignment='center')
                # # ax.set_ylim(ymax=75)
                # locator = plt.MaxNLocator(nbins=3) # with 3 bins you will have 4 ticks
                # ax.yaxis.set_major_locator(locator)
                # sns.despine(offset=10, trim=True)
                # plt.tight_layout()
                # # fig.savefig(os.path.join(self.figure_folder, 'lateralization', 'bars_single_{}_{}_{}_{}.pdf'.format(self.split_by, 'contrast', dv, trial_type)))
                # fig.savefig(os.path.join(self.figure_folder, 'lateralization', 'bars_single_{}_{}_{}_{}2.pdf'.format(self.split_by, 'contrast', dv, trial_type)))
                
                
            
    def CHOICE_SIGNALS_SDT(self):
        
        # load dataframe:
        data_frame = pd.read_csv(os.path.join(self.data_folder, 'event_related_average_wb', 'MNI_dataframe_1.csv'))
        dfs = [data_frame[data_frame.subject == s] for s in self.subjects]
        
        ylims = [(-0.5, 0.5), (-0.1, 0.1), (-0.8, 0.8), (-0.8, 0.8), (-0.1, 0.1), (-0.1, 0.1)]
        
        # for j, measure in enumerate(['V1_center_info', 'V2_center_info', 'V3_center_info', 'V1', 'V2', 'V3', 'lr_aIPS_lat', 'lr_PCeS_lat', 'lr_M1_lat']):
        # for j, measure in enumerate(['V1_SVM', 'V2_SVM', 'V3_SVM',]):
        for j, measure in enumerate(['sl_IPL_SVM', 'sl_SPL1_SVM', 'sl_SPL2_SVM', 'sl_pIns_SVM']):
            
            # # Spearman:
            # measure_hit = [dfs[i][measure][self.hit[i]] for i in range(len(self.subjects))]
            # measure_fa = [dfs[i][measure][self.fa[i]] for i in range(len(self.subjects))]
            # measure_miss = [dfs[i][measure][self.miss[i]] for i in range(len(self.subjects))]
            # measure_cr = [dfs[i][measure][self.cr[i]] for i in range(len(self.subjects))]
            # r = np.zeros(len(self.subjects))
            # for i in range(len(self.subjects)):
            #     x = np.concatenate(( np.ones(len(measure_hit[i])), np.ones(len(measure_fa[i]))*2, np.ones(len(measure_miss[i]))*3, np.ones(len(measure_cr[i]))*4 ))
            #     y = np.concatenate(( np.array(measure_hit[i]), np.array(measure_fa[i]), np.array(measure_miss[i]), np.array(measure_cr[i])))
            #     r[i] = sp.stats.spearmanr(x, y)[0]
            # p = myfuncs.permutationTest(r, np.zeros(len(r)), paired=True)[1]

            # SDT bars:
            measure_hit = np.array([np.mean(dfs[i][measure][self.hit[i]]) for i in range(len(self.subjects))])
            measure_fa = np.array([np.mean(dfs[i][measure][self.fa[i]]) for i in range(len(self.subjects))])
            measure_miss = np.array([np.mean(dfs[i][measure][self.miss[i]]) for i in range(len(self.subjects))])
            measure_cr = np.array([np.mean(dfs[i][measure][self.cr[i]]) for i in range(len(self.subjects))])
            measure_yes = np.array([np.mean(dfs[i][measure][self.yes[i]]) for i in range(len(self.subjects))])
            measure_no = np.array([np.mean(dfs[i][measure][self.no[i]]) for i in range(len(self.subjects))])
            measure_correct = np.array([np.mean(dfs[i][measure][self.correct[i]]) for i in range(len(self.subjects))])
            measure_error = np.array([np.mean(dfs[i][measure][self.error[i]]) for i in range(len(self.subjects))])

            for measures, titles, labels, colors, alphas in zip([[measure_hit, measure_fa, measure_miss, measure_cr], [measure_yes, measure_no, measure_correct, measure_error]],
                                                        ['0', '1'],
                                                        [['H', 'FA', 'M', 'CR'], ['Yes', 'No', 'Correct', 'Error']],
                                                        [['m', 'm', 'c', 'c'], ['m', 'c', 'k', 'k']],
                                                        [[1, 0.5, 0.5, 1], [1, 1, 1, 0.5]]):

                my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}
                N = 4
                ind = np.linspace(0,N,N)  # the x locations for the groups
                bar_width = 0.9   # the width of the bars
                spacing = [0, 0, 0, 0]
                # p1 = myfuncs.permutationTest(measures[0], measures[1], paired=True)[1]
                # p2 = myfuncs.permutationTest(measures[1], measures[2], paired=True)[1]
                # p3 = myfuncs.permutationTest(measures[2], measures[3], paired=True)[1]
                p1 = sp.stats.ttest_rel(measures[0], measures[1],)[1]
                p2 = sp.stats.ttest_rel(measures[1], measures[2],)[1]
                p3 = sp.stats.ttest_rel(measures[2], measures[3],)[1]
                MEANS = np.array([np.mean(m) for m in measures])
                SEMS = np.array([sp.stats.sem(m) for m in measures])
                fig = plt.figure(figsize=(2,2))
                ax = fig.add_subplot(111)
                for i in range(N):
                    ax.bar(ind[i]+spacing[i], MEANS[i], width = bar_width, yerr=SEMS[i], color=colors[i], alpha=alphas[i], edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
                ax.set_xticks( ind )
                ax.set_xticklabels( labels )
                ax.set_ylabel(measure)
                # ax.set_ylim(ylims[j])
                ax.text(s=str(round(p1,3)), x=(ind[0]+ind[1])/2.0, y=ax.get_ylim()[1]-((ax.get_ylim()[1] - ax.get_ylim()[0]) / 10.0), size=5, horizontalalignment='center')
                ax.text(s=str(round(p2,3)), x=(ind[1]+ind[2])/2.0, y=ax.get_ylim()[1]-((ax.get_ylim()[1] - ax.get_ylim()[0]) / 10.0), size=5, horizontalalignment='center')
                ax.text(s=str(round(p3,3)), x=(ind[2]+ind[3])/2.0, y=ax.get_ylim()[1]-((ax.get_ylim()[1] - ax.get_ylim()[0]) / 10.0), size=5, horizontalalignment='center')
                # ax.set_title('r={}; p={}'.format(round(r.mean(),3), round(p,3)))
                sns.despine(offset=10, trim=True,)
                plt.tight_layout()
                fig.savefig(os.path.join(self.figure_folder, 'V123', 'cp', 'SDT_measure_{}_{}_{}.pdf'.format(measure, titles, self.split_by)))
    
    def CHOICE_SIGNALS_variability(self):
        
        shell()
        
        # load dataframe:
        self.make_pupil_BOLD_dataframe(data_type='clean_4th_ventricle', time_locked='stim_locked', regress_iti=False, regress_rt=True, regress_stimulus=False)
        data_brainstem = self.data_frame.copy()
        self.make_pupil_BOLD_dataframe(data_type='clean_False', time_locked='stim_locked', regress_iti=False, regress_rt=True, regress_stimulus=False)
        data_cortex = self.data_frame.copy()
        data_frame = pd.read_csv(os.path.join(self.data_folder, 'event_related_average_wb', 'MNI_dataframe_1.csv'))
        dfs_brainstem = [data_brainstem[data_brainstem.subject == s] for s in self.subjects]
        dfs_cortex = [data_cortex[data_cortex.subject == s] for s in self.subjects]
        dfs = [data_frame[data_frame.subject == s] for s in self.subjects]
        
        rois = ['V1_center_info_choice', 'V2_center_info_choice', 'sl_IPL_info', 'sl_SPL1_info', 'lr_PCeS_lat', 'sl_pIns_info', 'combined_2']
        roi_names = ['V1', 'V2', 'IPL', 'SPL', 'IPS/PCeS', 'pIns', 'Combined']
        
        for r in rois:
        
        
            x1 = []
            x2 = []
        
            for s in range(len(self.subjects)):
            
                X1 = np.array(self.present[s], dtype=int)
                X2 = np.array(self.pupil_data[s].pupil_d)
                Y = np.array(dfs[s][r])
                
                d = {'X1' : pd.Series(X1),
                    'X2' : pd.Series(X2),
                    'Y' : pd.Series(Y),
                    }
                data = pd.DataFrame(d)
                
                model = 'ols'
                
                # direct:
                f = 'Y ~ X1 + X2'
                if model == 'ols':
                    m = sm.ols(formula=f, data=data)
                if model == 'logit':
                    m = sm.logit(formula=f, data=data)
                fit = m.fit()
            
                x1.append(fit.params['X1'])
                x2.append(fit.params['X2'])

            x1 = np.array(x1)
            x2 = np.array(x2)
            
            print r
            print myfuncs.permutationTest(x2, np.zeros(len(self.subjects)), paired=True)
        
    
    def CHOICE_SIGNALS_logistic_regressions(self):
        
        # load dataframe:
        data_frame = pd.read_csv(os.path.join(self.data_folder, 'event_related_average_wb', 'MNI_dataframe_1.csv'))
        dfs = [data_frame[data_frame.subject == s] for s in self.subjects]
        
        logistic = sklearn.linear_model.LogisticRegression(C=1e5, fit_intercept=True)
        
        # for j, measure in enumerate(['sl_IPL_info', 'sl_SPL1_info', 'lr_aIPS_lat', 'sl_SPL2_info', 'lr_PCeS_lat', 'sl_pIns_info', 'sl_PCeS_IFG_info', 'sl_MFG_info', 'lr_M1_lat']):
        for j, measure in enumerate(['combined_choice_parietal',]):

            # logistic regression with choice:
            import statsmodels.api as sm

            xline = np.linspace(-3, 3, 25)
            r_choice = np.zeros(len(self.subjects))
            r_present = np.zeros(len(self.subjects))
            line_choice = []
            line_present = []
            for i in range(len(self.subjects)):
                
                if measure == 'pupil_d':
                    values = np.array(self.pupil_data[i][measure])
                else:
                    values = np.array(dfs[i][measure])
                nan_ind = np.isnan(values)
                values = values[~nan_ind]
                values = (values - values.mean()) / values.std()
                
                # d = {
                #     'measure_h' : values[self.pupil_h_ind[i]],
                #     'choice_h' : np.array(self.yes[i], dtype=int)[self.present[i]],
                #     'present_h' : np.array(self.present[i], dtype=int)[self.yes[i]],
                #     'measure_l' : values[self.pupil_l_ind[i]],
                #     'choice_l' : np.array(self.yes[i], dtype=int)[self.absent[i]],
                #     'present_l' : np.array(self.present[i], dtype=int)[self.no[i]],
                #     }
                # d = pd.DataFrame(d)

                choice = np.array(self.yes[i], dtype=int)[~nan_ind]
                measure_choice = values

                present = np.array(self.present[i], dtype=int)[~nan_ind]
                measure_present = values

                logistic.fit(np.atleast_2d(measure_choice).T, choice)
                r_choice[i] = np.exp(logistic.coef_) / (1 + np.exp(logistic.coef_))
                line_choice.append( logistic.predict_proba(np.atleast_2d(xline).T)[:,1] )

                logistic.fit(np.atleast_2d(measure_present).T, present)
                r_present[i] = np.exp(logistic.coef_) / (1 + np.exp(logistic.coef_))
                line_present.append( logistic.predict_proba(np.atleast_2d(xline).T)[:,1] )
                
            line_choice = np.vstack(line_choice)
            line_present = np.vstack(line_present)
            p_choice = myfuncs.permutationTest(r_choice, np.ones(len(r_choice))/2.0, paired=True)[1]
            p_present = myfuncs.permutationTest(r_present, np.ones(len(r_choice))/2.0, paired=True)[1]

            fig = plt.figure(figsize=(2,2))
            ax = fig.add_subplot(111)
            MEAN = line_present.mean(axis=0)
            SEM = sp.stats.sem(line_present, axis=0)
            ax.plot(xline, MEAN, 'c', label='presence')
            ax.fill_between(xline, MEAN-SEM, MEAN+SEM, color='c', alpha=0.2)
            MEAN = line_choice.mean(axis=0)
            SEM = sp.stats.sem(line_choice, axis=0)
            ax.plot(xline, MEAN, 'm', label='choice')
            ax.fill_between(xline, MEAN-SEM, MEAN+SEM, color='m', alpha=0.2)
            ax.set_xlabel('{} (Z)'.format(measure))
            ax.set_ylabel('Probability')
            ax.set_title('r={}, p={}\nr={}, p={}'.format(round(r_present.mean(),3), round(p_present,3), round(r_choice.mean(),3), round(p_choice,3),))
            ax.set_ylim((0,1))
            plt.legend()
            sns.despine(offset=10, trim=True,)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, 'V123', 'cp', 'Logistic_measure_simple_{}.pdf'.format(measure)))
            
            # save:
            np.save(os.path.join(self.data_folder, 'cp', 'lr_stim_{}.npy'.format(measure)), r_present)
            np.save(os.path.join(self.data_folder, 'cp', 'lr_choice_{}.npy'.format(measure)), r_choice)
            
            # with stim / choice factored out:
            xline = np.linspace(-3, 3, 25)
            r_choice = np.zeros(len(self.subjects))
            r_choice_h = np.zeros(len(self.subjects))
            r_choice_l = np.zeros(len(self.subjects))
            r_present = np.zeros(len(self.subjects))
            r_present_h = np.zeros(len(self.subjects))
            r_present_l = np.zeros(len(self.subjects))
            line_choice = []
            line_choice_h = []
            line_choice_l = []
            line_present = []
            line_present_h = []
            line_present_l = []
            for i in range(len(self.subjects)):

                if measure == 'pupil_d':
                    values = np.array(self.pupil_data[i][measure])
                else:
                    values = np.array(dfs[i][measure])
                nan_ind = np.isnan(values)
                values = values[~nan_ind]
                values = (values - values.mean()) / values.std()

                # d = {
                #     'measure_h' : values[self.pupil_h_ind[i]],
                #     'choice_h' : np.array(self.yes[i], dtype=int)[self.present[i]],
                #     'present_h' : np.array(self.present[i], dtype=int)[self.yes[i]],
                #     'measure_l' : values[self.pupil_l_ind[i]],
                #     'choice_l' : np.array(self.yes[i], dtype=int)[self.absent[i]],
                #     'present_l' : np.array(self.present[i], dtype=int)[self.no[i]],
                #     }
                # d = pd.DataFrame(d)

                choice_h = np.array(self.yes[i], dtype=int)[self.present[i] & ~nan_ind]
                choice_l = np.array(self.yes[i], dtype=int)[self.absent[i] & ~nan_ind]
                measure_choice_h = values[self.present[i][~nan_ind]]
                measure_choice_l = values[self.absent[i][~nan_ind]]

                present_h = np.array(self.present[i], dtype=int)[self.yes[i] & ~nan_ind]
                present_l = np.array(self.present[i], dtype=int)[self.no[i] & ~nan_ind]
                measure_present_h = values[self.yes[i][~nan_ind]]
                measure_present_l = values[self.no[i][~nan_ind]]

                logistic.fit(np.atleast_2d(measure_choice_h).T, choice_h)
                r_choice_h[i] = np.exp(logistic.coef_) / (1 + np.exp(logistic.coef_))
                line_choice_h.append( logistic.predict_proba(np.atleast_2d(xline).T)[:,1] )

                logistic.fit(np.atleast_2d(measure_choice_l).T, choice_l)
                r_choice_l[i] = np.exp(logistic.coef_) / (1 + np.exp(logistic.coef_))
                line_choice_l.append( logistic.predict_proba(np.atleast_2d(xline).T)[:,1] )

                r_choice[i] =  (r_choice_h[i]+r_choice_l[i]) / 2.0
                line_choice.append( (line_choice_h[i]+line_choice_l[i]) / 2.0 )

                logistic.fit(np.atleast_2d(measure_present_h).T, present_h)
                r_present_h[i] = np.exp(logistic.coef_) / (1 + np.exp(logistic.coef_))
                line_present_h.append( logistic.predict_proba(np.atleast_2d(xline).T)[:,1] )

                logistic.fit(np.atleast_2d(measure_present_l).T, present_l)
                r_present_l[i] = np.exp(logistic.coef_) / (1 + np.exp(logistic.coef_))
                line_present_l.append( logistic.predict_proba(np.atleast_2d(xline).T)[:,1] )

                r_present[i] =  (r_present_h[i]+r_present_l[i]) / 2.0
                line_present.append( (line_present_h[i]+line_present_l[i]) / 2.0 )
            
            line_choice = np.vstack(line_choice)
            line_choice_h = np.vstack(line_choice_h)
            line_choice_l = np.vstack(line_choice_l)
            line_present = np.vstack(line_present)
            line_present_h = np.vstack(line_present_h)
            line_present_l = np.vstack(line_present_l)
            
            
            # # save:
            # np.save(os.path.join(self.data_folder, 'cp', 'lr_stim_{}.npy'.format(measure)), r_present)
            # np.save(os.path.join(self.data_folder, 'cp', 'lr_choice_{}.npy'.format(measure)), r_choice)
            
            p_choice = myfuncs.permutationTest(r_choice, np.ones(len(r_choice_h))/2.0, paired=True)[1]
            p_choice_h = myfuncs.permutationTest(r_choice_h, np.ones(len(r_choice_h))/2.0, paired=True)[1]
            p_choice_l = myfuncs.permutationTest(r_choice_l, np.ones(len(r_choice_l))/2.0, paired=True)[1]
            p_present = myfuncs.permutationTest(r_present, np.ones(len(r_choice_h))/2.0, paired=True)[1]
            p_present_h = myfuncs.permutationTest(r_present_h, np.ones(len(r_present_h))/2.0, paired=True)[1]
            p_present_l = myfuncs.permutationTest(r_present_l, np.ones(len(r_present_l))/2.0, paired=True)[1]

            fig = plt.figure(figsize=(2,2))
            ax = fig.add_subplot(111)
            MEAN = line_present.mean(axis=0)
            SEM = sp.stats.sem(line_present, axis=0)
            ax.plot(xline, MEAN, 'c', label='presence')
            ax.fill_between(xline, MEAN-SEM, MEAN+SEM, color='c', alpha=0.2)
            MEAN = line_choice.mean(axis=0)
            SEM = sp.stats.sem(line_choice, axis=0)
            ax.plot(xline, MEAN, 'm', label='choice')
            ax.fill_between(xline, MEAN-SEM, MEAN+SEM, color='m', alpha=0.2)
            ax.set_xlabel('{} (Z)'.format(measure))
            ax.set_ylabel('Probability')
            ax.set_title('r={}, p={}\nr={}, p={}'.format(round(r_present.mean(),3), round(p_present,3), round(r_choice.mean(),3), round(p_choice,3),))
            ax.set_ylim((0,1))
            plt.legend()
            sns.despine(offset=10, trim=True,)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, 'V123', 'cp', 'Logistic_measure_{}.pdf'.format(measure)))
            
            # split by pupil:
            xline = np.linspace(-3, 3, 25)
            r_choice_h = np.zeros(len(self.subjects))
            r_choice_l = np.zeros(len(self.subjects))
            r_present_h = np.zeros(len(self.subjects))
            r_present_l = np.zeros(len(self.subjects))
            i_choice_h = np.zeros(len(self.subjects))
            i_choice_l = np.zeros(len(self.subjects))
            i_present_h = np.zeros(len(self.subjects))
            i_present_l = np.zeros(len(self.subjects))
            line_choice = []
            line_choice_h = []
            line_choice_l = []
            line_present = []
            line_present_h = []
            line_present_l = []
            for i in range(len(self.subjects)):

                if measure == 'pupil_d':
                    values = np.array(self.pupil_data[i][measure])
                else:
                    values = np.array(dfs[i][measure])
                nan_ind = np.isnan(values)
                values = values[~nan_ind]
                values = (values - values.mean()) / values.std()

                # d = {
                #     'measure_h' : values[self.pupil_h_ind[i]],
                #     'choice_h' : np.array(self.yes[i], dtype=int)[self.present[i]],
                #     'present_h' : np.array(self.present[i], dtype=int)[self.yes[i]],
                #     'measure_l' : values[self.pupil_l_ind[i]],
                #     'choice_l' : np.array(self.yes[i], dtype=int)[self.absent[i]],
                #     'present_l' : np.array(self.present[i], dtype=int)[self.no[i]],
                #     }
                # d = pd.DataFrame(d)

                choice_h = np.array(self.yes[i], dtype=int)[self.pupil_h_ind[i] & ~nan_ind]
                choice_l = np.array(self.yes[i], dtype=int)[self.pupil_l_ind[i] & ~nan_ind]
                measure_choice_h = values[self.pupil_h_ind[i][~nan_ind]]
                measure_choice_l = values[self.pupil_l_ind[i][~nan_ind]]

                present_h = np.array(self.present[i], dtype=int)[self.pupil_h_ind[i] & ~nan_ind]
                present_l = np.array(self.present[i], dtype=int)[self.pupil_l_ind[i] & ~nan_ind]
                measure_present_h = values[self.pupil_h_ind[i][~nan_ind]]
                measure_present_l = values[self.pupil_l_ind[i][~nan_ind]]

                logistic.fit(np.atleast_2d(measure_choice_h).T, choice_h)
                r_choice_h[i] = np.exp(logistic.coef_) / (1 + np.exp(logistic.coef_))
                i_choice_h[i] = np.exp(logistic.intercept_) / (1 + np.exp(logistic.intercept_))
                line_choice_h.append( logistic.predict_proba(np.atleast_2d(xline).T)[:,1] )

                logistic.fit(np.atleast_2d(measure_choice_l).T, choice_l)
                r_choice_l[i] = np.exp(logistic.coef_) / (1 + np.exp(logistic.coef_))
                i_choice_l[i] = np.exp(logistic.intercept_) / (1 + np.exp(logistic.intercept_))
                line_choice_l.append( logistic.predict_proba(np.atleast_2d(xline).T)[:,1] )
                
                logistic.fit(np.atleast_2d(measure_present_h).T, present_h)
                r_present_h[i] = np.exp(logistic.coef_) / (1 + np.exp(logistic.coef_))
                i_present_h[i] = np.exp(logistic.intercept_) / (1 + np.exp(logistic.intercept_))
                line_present_h.append( logistic.predict_proba(np.atleast_2d(xline).T)[:,1] )

                logistic.fit(np.atleast_2d(measure_present_l).T, present_l)
                r_present_l[i] = np.exp(logistic.coef_) / (1 + np.exp(logistic.coef_))
                i_present_l[i] = np.exp(logistic.intercept_) / (1 + np.exp(logistic.intercept_))
                line_present_l.append( logistic.predict_proba(np.atleast_2d(xline).T)[:,1] )
                
            line_choice_h = np.vstack(line_choice_h)
            line_choice_l = np.vstack(line_choice_l)
            line_present_h = np.vstack(line_present_h)
            line_present_l = np.vstack(line_present_l)
            
            # p_choice_h = myfuncs.permutationTest(r_choice_h, np.ones(len(r_choice_h))/2.0, paired=True)[1]
            # p_choice_l = myfuncs.permutationTest(r_choice_l, np.ones(len(r_choice_l))/2.0, paired=True)[1]
            # p_present_h = myfuncs.permutationTest(r_present_h, np.ones(len(r_present_h))/2.0, paired=True)[1]
            # p_present_l = myfuncs.permutationTest(r_present_l, np.ones(len(r_present_l))/2.0, paired=True)[1]
            
            p_present = myfuncs.permutationTest(r_present_h, r_present_l, paired=True)[1]
            p_choice = myfuncs.permutationTest(r_choice_h, r_choice_l, paired=True)[1]
            int_p_present = myfuncs.permutationTest(i_present_h, i_present_l, paired=True)[1]
            int_p_choice = myfuncs.permutationTest(i_choice_h, i_choice_l, paired=True)[1]
            
            fig = plt.figure(figsize=(2,4))

            ax = fig.add_subplot(211)
            MEAN = line_present_h.mean(axis=0)
            SEM = sp.stats.sem(line_present_h, axis=0)
            ax.plot(xline, MEAN, 'r')
            ax.fill_between(xline, MEAN-SEM, MEAN+SEM, color='r', alpha=0.2)
            MEAN = line_present_l.mean(axis=0)
            SEM = sp.stats.sem(line_present_l, axis=0)
            ax.plot(xline, MEAN, 'b',)
            ax.fill_between(xline, MEAN-SEM, MEAN+SEM, color='b', alpha=0.2)
            ax.hlines(y=i_present_h.mean(), xmin=-3, xmax=0, color='r', linestyle=':', lw=0.5)
            ax.hlines(y=i_present_l.mean(), xmin=-3, xmax=0, color='b', linestyle=':', lw=0.5)
            ax.set_xlabel('{} (Z)'.format(measure))
            ax.set_ylabel('P(present)', color='c')
            ax.set_title('diff r={}, p={}\ndiff i={}, p={}'.format(round(r_present_h.mean()-r_present_l.mean(),3), round(p_present,3), round(i_present_h.mean()-i_present_l.mean(),3), round(int_p_present,3)))
            ax.set_ylim((0,1))

            ax = fig.add_subplot(212)
            MEAN = line_choice_h.mean(axis=0)
            SEM = sp.stats.sem(line_choice_h, axis=0)
            ax.plot(xline, MEAN, 'r')
            ax.fill_between(xline, MEAN-SEM, MEAN+SEM, color='r', alpha=0.2)
            MEAN = line_choice_l.mean(axis=0)
            SEM = sp.stats.sem(line_choice_l, axis=0)
            ax.plot(xline, MEAN, 'b')
            ax.fill_between(xline, MEAN-SEM, MEAN+SEM, color='b', alpha=0.2)
            ax.hlines(y=i_choice_h.mean(), xmin=-3, xmax=0, color='r', linestyle=':', lw=0.5)
            ax.hlines(y=i_choice_l.mean(), xmin=-3, xmax=0, color='b', linestyle=':', lw=0.5)
            ax.set_xlabel('{} (Z)'.format(measure))
            ax.set_ylabel('P(yes)', color='m')
            ax.set_title('diff r={}, p={}\ndiff i={}, p={}'.format(round(r_choice_h.mean()-r_choice_l.mean(),3), round(p_choice,3), round(i_choice_h.mean()-i_choice_l.mean(),3), round(int_p_choice,3)))
            ax.set_ylim((0,1))
            sns.despine(offset=10, trim=True,)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, 'V123', 'cp', 'Logistic_measure_{}_{}.pdf'.format(measure, self.split_by)))
            
            
            # # logistic regression split by pupil:
            #
            # xline = np.linspace(-3, 3, 25)
            # r_choice_h = np.zeros(len(self.subjects))
            # r_present_h = np.zeros(len(self.subjects))
            # i_choice_h = np.zeros(len(self.subjects))
            # i_present_h = np.zeros(len(self.subjects))
            # line_choice_h = []
            # line_present_h = []
            # r_choice_l = np.zeros(len(self.subjects))
            # r_present_l = np.zeros(len(self.subjects))
            # i_choice_l = np.zeros(len(self.subjects))
            # i_present_l = np.zeros(len(self.subjects))
            # line_choice_l = []
            # line_present_l = []
            # for i in range(len(self.subjects)):
            #
            #     if measure == 'pupil_d':
            #         values = np.array(self.pupil_data[i][measure])
            #     else:
            #         values = np.array(dfs[i][measure])
            #     nan_ind = np.isnan(values)
            #     values = values[~nan_ind]
            #     values = (values - values.mean()) / values.std()
            #
            #     # d = {
            #     #     'measure_h' : values[self.pupil_h_ind[i]],
            #     #     'choice_h' : np.array(self.yes[i], dtype=int)[self.pupil_h_ind[i]],
            #     #     'present_h' : np.array(self.present[i], dtype=int)[self.pupil_h_ind[i]],
            #     #     'measure_l' : values[self.pupil_l_ind[i]],
            #     #     'choice_l' : np.array(self.yes[i], dtype=int)[self.pupil_l_ind[i]],
            #     #     'present_l' : np.array(self.present[i], dtype=int)[self.pupil_l_ind[i]],
            #     #     }
            #     # d = pd.DataFrame(d)
            #
            #     choice_h_present = np.array(self.yes[i], dtype=int)[self.pupil_h_ind[i] & self.present[i] & ~nan_ind]
            #     choice_l_present = np.array(self.yes[i], dtype=int)[self.pupil_l_ind[i] & self.present[i] & ~nan_ind]
            #     choice_h_absent = np.array(self.yes[i], dtype=int)[self.pupil_h_ind[i] & self.absent[i] & ~nan_ind]
            #     choice_l_absent = np.array(self.yes[i], dtype=int)[self.pupil_l_ind[i] & self.absent[i] & ~nan_ind]
            #     measure_choice_h_present = values[(self.pupil_h_ind[i] & self.present[i])[~nan_ind]]
            #     measure_choice_l_present = values[(self.pupil_l_ind[i] & self.present[i])[~nan_ind]]
            #     measure_choice_h_absent = values[(self.pupil_h_ind[i] & self.absent[i])[~nan_ind]]
            #     measure_choice_l_absent = values[(self.pupil_l_ind[i] & self.absent[i])[~nan_ind]]
            #
            #     present_h_yes = np.array(self.present[i], dtype=int)[self.pupil_h_ind[i] & self.yes[i] & ~nan_ind]
            #     present_l_yes = np.array(self.present[i], dtype=int)[self.pupil_l_ind[i] & self.yes[i] & ~nan_ind]
            #     present_h_no = np.array(self.present[i], dtype=int)[self.pupil_h_ind[i] & self.no[i] & ~nan_ind]
            #     present_l_no = np.array(self.present[i], dtype=int)[self.pupil_l_ind[i] & self.no[i] & ~nan_ind]
            #     measure_present_h_yes = values[(self.pupil_h_ind[i] & self.yes[i])[~nan_ind]]
            #     measure_present_l_yes = values[(self.pupil_l_ind[i] & self.yes[i])[~nan_ind]]
            #     measure_present_h_no = values[(self.pupil_h_ind[i] & self.no[i])[~nan_ind]]
            #     measure_present_l_no = values[(self.pupil_l_ind[i] & self.no[i])[~nan_ind]]
            #
            #     logistic.fit(np.atleast_2d(measure_choice_h_present).T, choice_h_present)
            #     r_1 = np.exp(logistic.coef_) / (1 + np.exp(logistic.coef_))
            #     i_1 = np.exp(logistic.intercept_) / (1 + np.exp(logistic.intercept_))
            #     line_1 = logistic.predict_proba(np.atleast_2d(xline).T)[:,1]
            #     logistic.fit(np.atleast_2d(measure_choice_h_absent).T, choice_h_absent)
            #     r_2 = np.exp(logistic.coef_) / (1 + np.exp(logistic.coef_))
            #     i_2 = np.exp(logistic.intercept_) / (1 + np.exp(logistic.intercept_))
            #     line_2 = logistic.predict_proba(np.atleast_2d(xline).T)[:,1]
            #     r_choice_h[i] = (r_1 + r_2) / 2.0
            #     i_choice_h[i] = (i_1 + i_2) / 2.0
            #     line_choice_h.append( (line_1+line_2) / 2.0 )
            #
            #     logistic.fit(np.atleast_2d(measure_choice_l_present).T, choice_l_present)
            #     r_1 = np.exp(logistic.coef_) / (1 + np.exp(logistic.coef_))
            #     i_1 = np.exp(logistic.intercept_) / (1 + np.exp(logistic.intercept_))
            #     line_1 = logistic.predict_proba(np.atleast_2d(xline).T)[:,1]
            #     logistic.fit(np.atleast_2d(measure_choice_l_absent).T, choice_l_absent)
            #     r_2 = np.exp(logistic.coef_) / (1 + np.exp(logistic.coef_))
            #     i_2 = np.exp(logistic.intercept_) / (1 + np.exp(logistic.intercept_))
            #     line_2 = logistic.predict_proba(np.atleast_2d(xline).T)[:,1]
            #     r_choice_l[i] = (r_1 + r_2) / 2.0
            #     i_choice_l[i] = (i_1 + i_2) / 2.0
            #     line_choice_l.append( (line_1+line_2) / 2.0 )
            #
            #     logistic.fit(np.atleast_2d(measure_present_h_yes).T, present_h_yes)
            #     r_1 = np.exp(logistic.coef_) / (1 + np.exp(logistic.coef_))
            #     i_1 = np.exp(logistic.intercept_) / (1 + np.exp(logistic.intercept_))
            #     line_1 = logistic.predict_proba(np.atleast_2d(xline).T)[:,1]
            #     logistic.fit(np.atleast_2d(measure_present_h_no).T, present_h_no)
            #     r_2 = np.exp(logistic.coef_) / (1 + np.exp(logistic.coef_))
            #     i_2 = np.exp(logistic.intercept_) / (1 + np.exp(logistic.intercept_))
            #     line_2 = logistic.predict_proba(np.atleast_2d(xline).T)[:,1]
            #     r_present_h[i] = (r_1 + r_2) / 2.0
            #     i_present_h[i] = (i_1 + i_2) / 2.0
            #     line_present_h.append( (line_1+line_2) / 2.0 )
            #
            #     logistic.fit(np.atleast_2d(measure_present_l_yes).T, present_l_yes)
            #     r_1 = np.exp(logistic.coef_) / (1 + np.exp(logistic.coef_))
            #     i_1 = np.exp(logistic.intercept_) / (1 + np.exp(logistic.intercept_))
            #     line_1 = logistic.predict_proba(np.atleast_2d(xline).T)[:,1]
            #     logistic.fit(np.atleast_2d(measure_present_l_no).T, present_l_no)
            #     r_2 = np.exp(logistic.coef_) / (1 + np.exp(logistic.coef_))
            #     i_2 = np.exp(logistic.intercept_) / (1 + np.exp(logistic.intercept_))
            #     line_2 = logistic.predict_proba(np.atleast_2d(xline).T)[:,1]
            #     r_present_l[i] = (r_1 + r_2) / 2.0
            #     i_present_l[i] = (i_1 + i_2) / 2.0
            #     line_present_l.append( (line_1+line_2) / 2.0 )
            #
            #
            # line_choice_h = np.vstack(line_choice_h)
            # line_present_h = np.vstack(line_present_h)
            # # p_choice_h = myfuncs.permutationTest(r_choice_h, np.ones(len(r_choice_h))/2.0, paired=True)[1]
            # # p_present_h = myfuncs.permutationTest(r_present_h, np.ones(len(r_present_h))/2.0, paired=True)[1]
            #
            # line_choice_l = np.vstack(line_choice_l)
            # line_present_l = np.vstack(line_present_l)
            # # p_choice_l = myfuncs.permutationTest(r_choice_l, np.ones(len(r_choice_l))/2.0, paired=True)[1]
            # # p_present_l = myfuncs.permutationTest(r_present_l, np.ones(len(r_present_l))/2.0, paired=True)[1]
            #
            # np.save(os.path.join(self.data_folder, 'cp', 'lr_stim_h_{}.npy'.format(measure)), r_present_h)
            # np.save(os.path.join(self.data_folder, 'cp', 'lr_stim_l_{}.npy'.format(measure)), r_present_l)
            # np.save(os.path.join(self.data_folder, 'cp', 'lr_choice_h_{}.npy'.format(measure)), r_choice_h)
            # np.save(os.path.join(self.data_folder, 'cp', 'lr_choice_l_{}.npy'.format(measure)), r_choice_l)
            # np.save(os.path.join(self.data_folder, 'cp', 'lr_line_stim_h_{}.npy'.format(measure)), line_present_h)
            # np.save(os.path.join(self.data_folder, 'cp', 'lr_line_stim_l_{}.npy'.format(measure)), line_present_l)
            # np.save(os.path.join(self.data_folder, 'cp', 'lr_line_choice_h_{}.npy'.format(measure)), line_choice_h)
            # np.save(os.path.join(self.data_folder, 'cp', 'lr_line_choice_l_{}.npy'.format(measure)), line_choice_l)
            #
            # p_present = myfuncs.permutationTest(r_present_h, r_present_l, paired=True)[1]
            # p_choice = myfuncs.permutationTest(r_choice_h, r_choice_l, paired=True)[1]
            #
            # int_p_present = myfuncs.permutationTest(i_present_h, i_present_l, paired=True)[1]
            # int_p_choice = myfuncs.permutationTest(i_choice_h, i_choice_l, paired=True)[1]
            #
            # fig = plt.figure(figsize=(2,4))
            #
            # ax = fig.add_subplot(211)
            # MEAN = line_present_h.mean(axis=0)
            # SEM = sp.stats.sem(line_present_h, axis=0)
            # ax.plot(xline, MEAN, 'r')
            # ax.fill_between(xline, MEAN-SEM, MEAN+SEM, color='r', alpha=0.2)
            # MEAN = line_present_l.mean(axis=0)
            # SEM = sp.stats.sem(line_present_l, axis=0)
            # ax.plot(xline, MEAN, 'b',)
            # ax.fill_between(xline, MEAN-SEM, MEAN+SEM, color='b', alpha=0.2)
            # ax.hlines(y=i_present_h.mean(), xmin=-3, xmax=0, color='r', linestyle=':', lw=0.5)
            # ax.hlines(y=i_present_l.mean(), xmin=-3, xmax=0, color='b', linestyle=':', lw=0.5)
            # ax.set_xlabel('{} (Z)'.format(measure))
            # ax.set_ylabel('P(present)', color='c')
            # ax.set_title('diff r={}, p={}\ndiff i={}, p={}'.format(round(r_present_h.mean()-r_present_l.mean(),3), round(p_present,3), round(i_present_h.mean()-i_present_l.mean(),3), round(int_p_present,3)))
            # ax.set_ylim((0,1))
            #
            # ax = fig.add_subplot(212)
            # MEAN = line_choice_h.mean(axis=0)
            # SEM = sp.stats.sem(line_choice_h, axis=0)
            # ax.plot(xline, MEAN, 'r')
            # ax.fill_between(xline, MEAN-SEM, MEAN+SEM, color='r', alpha=0.2)
            # MEAN = line_choice_l.mean(axis=0)
            # SEM = sp.stats.sem(line_choice_l, axis=0)
            # ax.plot(xline, MEAN, 'b')
            # ax.fill_between(xline, MEAN-SEM, MEAN+SEM, color='b', alpha=0.2)
            # ax.hlines(y=i_choice_h.mean(), xmin=-3, xmax=0, color='r', linestyle=':', lw=0.5)
            # ax.hlines(y=i_choice_l.mean(), xmin=-3, xmax=0, color='b', linestyle=':', lw=0.5)
            # ax.set_xlabel('{} (Z)'.format(measure))
            # ax.set_ylabel('P(yes)', color='m')
            # ax.set_title('diff r={}, p={}\ndiff i={}, p={}'.format(round(r_choice_h.mean()-r_choice_l.mean(),3), round(p_choice,3), round(i_choice_h.mean()-i_choice_l.mean(),3), round(int_p_choice,3)))
            # ax.set_ylim((0,1))
            # sns.despine(offset=10, trim=True,)
            # plt.tight_layout()
            # fig.savefig(os.path.join(self.figure_folder, 'V123', 'cp', 'Logistic_measure_{}_{}.pdf'.format(measure, self.split_by)))
            
    def CHOICE_SIGNALS_choice_probability(self, prepare=False):
        
        nrand = 10000
        
        # load dataframe:
        data_frame = pd.read_csv(os.path.join(self.data_folder, 'event_related_average_wb', 'MNI_dataframe_1.csv'))
        dfs = [data_frame[data_frame.subject == s] for s in self.subjects]
        
        for roi in [
                    'V1_center_info_stim_ori', 'V2_center_info_stim_ori', 'V3_center_info_stim_ori',
                    'V1_center_info_stim', 'V2_center_info_stim', 'V3_center_info_stim',
                    'V1_center_info_choice', 'V2_center_info_choice', 'V3_center_info_choice',
                    # 'V1_center_SVM_stim', 'V2_center_SVM_stim', 'V3_center_SVM_stim',
                    # 'V1_center_SVM_stim_ori', 'V2_center_SVM_stim_ori', 'V3_center_SVM_stim_ori',
                    # 'V1_center_SVM_choice', 'V2_center_SVM_choice', 'V3_center_SVM_choice',
                    'V1', 'V2', 'V3',
                    'lr_aIPS_lat', 'lr_PCeS_lat', 'lr_M1_lat',
                    'sl_IPL_info', 'sl_SPL1_info', 'sl_SPL2_info', 'sl_pIns_info', 'sl_PCeS_IFG_info', 'sl_MFG_info',
                    # 'combined_choice_V123', 'combined_stim_V123', 'combined_choice_parietal'
                    # 'combined_choice_parietal'
                    'lr_M1_lat', 'combined_choice_parietal_all', 'combined_choice_parietal_lat', 'combined_choice_parietal_sl',
                    'lr_M1_lat_res', 'combined_choice_parietal_all_res', 'combined_choice_parietal_lat_res', 'combined_choice_parietal_sl_res',
                    ]:
            
            # Choice probability:
            measure_yes = []
            measure_no = []
            measure_present = []
            measure_absent = []
            measure_hit = []
            measure_fa = []
            measure_miss = []
            measure_cr = []
            for i in range(len(self.subjects)):
                dummy = dfs[i][roi][self.hit[i]] 
                dummy = dummy[-np.isnan(dummy)]
                measure_hit.append( dummy )
                dummy = dfs[i][roi][self.fa[i]] 
                dummy = dummy[-np.isnan(dummy)]
                measure_fa.append( dummy )
                dummy = dfs[i][roi][self.miss[i]] 
                dummy = dummy[-np.isnan(dummy)]
                measure_miss.append( dummy )
                dummy = dfs[i][roi][self.cr[i]] 
                dummy = dummy[-np.isnan(dummy)]
                measure_cr.append( dummy )
                dummy = dfs[i][roi][self.yes[i]] 
                dummy = dummy[-np.isnan(dummy)]
                measure_yes.append( dummy )
                dummy = dfs[i][roi][self.no[i]] 
                dummy = dummy[-np.isnan(dummy)]
                measure_no.append( dummy )
                dummy = dfs[i][roi][self.present[i]] 
                dummy = dummy[-np.isnan(dummy)]
                measure_present.append( dummy )
                dummy = dfs[i][roi][self.absent[i]] 
                dummy = dummy[-np.isnan(dummy)]
                measure_absent.append( dummy )
                
            sp_ = np.zeros(len(self.subjects))
            cp_ = np.zeros(len(self.subjects))
            sp_yes = np.zeros(len(self.subjects))
            sp_no = np.zeros(len(self.subjects))
            cp_present = np.zeros(len(self.subjects))
            cp_absent = np.zeros(len(self.subjects))
            for i in range(len(self.subjects)):
                sp_[i] = myfuncs.roc_analysis(measure_present[i], measure_absent[i], nrand=nrand)[0]
                cp_[i] = myfuncs.roc_analysis(measure_yes[i], measure_no[i], nrand=nrand)[0]
                sp_yes[i] = myfuncs.roc_analysis(measure_hit[i], measure_fa[i], nrand=nrand)[0]
                sp_no[i] = myfuncs.roc_analysis(measure_miss[i], measure_cr[i], nrand=nrand)[0]
                cp_present[i] = myfuncs.roc_analysis(measure_hit[i], measure_miss[i], nrand=nrand)[0]
                cp_absent[i] = myfuncs.roc_analysis(measure_fa[i], measure_cr[i], nrand=nrand)[0]
            sp_without_choice = (sp_yes + sp_no) / 2.0
            cp_without_stim = (cp_present + cp_absent) / 2.0
        
            np.save(os.path.join(self.data_folder, 'cp', 'sp_{}.npy'.format(roi)), sp_)
            np.save(os.path.join(self.data_folder, 'cp', 'cp_{}.npy'.format(roi)), cp_)
            np.save(os.path.join(self.data_folder, 'cp', 'sp_without_choice_{}.npy'.format(roi)), sp_without_choice)
            np.save(os.path.join(self.data_folder, 'cp', 'cp_without_stim_{}.npy'.format(roi)), cp_without_stim)
            
            # # Choice split by pupil:
            # measure_yes_h = []
            # measure_yes_l = []
            # measure_no_h = []
            # measure_no_l = []
            # for i in range(len(self.subjects)):
            #     dummy = dfs[i][roi][self.yes[i] * self.pupil_h_ind[i]]
            #     dummy = dummy[-np.isnan(dummy)]
            #     measure_yes_h.append( dummy )
            #     dummy = dfs[i][roi][self.yes[i] * self.pupil_l_ind[i]]
            #     dummy = dummy[-np.isnan(dummy)]
            #     measure_yes_l.append( dummy )
            #     dummy = dfs[i][roi][self.no[i] * self.pupil_h_ind[i]]
            #     dummy = dummy[-np.isnan(dummy)]
            #     measure_no_h.append( dummy )
            #     dummy = dfs[i][roi][self.no[i] * self.pupil_l_ind[i]]
            #     dummy = dummy[-np.isnan(dummy)]
            #     measure_no_l.append( dummy )
            #
            # cp_h = np.zeros(len(self.subjects))
            # cp_l = np.zeros(len(self.subjects))
            # for i in range(len(self.subjects)):
            #     cp_h[i] = myfuncs.roc_analysis(measure_yes_h[i], measure_no_h[i], nrand=nrand)[0]
            #     cp_l[i] = myfuncs.roc_analysis(measure_yes_l[i], measure_no_l[i], nrand=nrand)[0]
            # np.save(os.path.join(self.data_folder, 'cp', 'cp_pupil_h_{}.npy'.format(roi)), cp_h)
            # np.save(os.path.join(self.data_folder, 'cp', 'cp_pupil_l_{}.npy'.format(roi)), cp_l)
            
    def CHOICE_SIGNALS_choice_probability_plot(self, prepare=False):
        
        # load dataframe:
        data_frame = pd.read_csv(os.path.join(self.data_folder, 'event_related_average_wb', 'MNI_dataframe_1.csv'))
        dfs = [data_frame[data_frame.subject == s] for s in self.subjects]
        
        decoding_type = 'info'
        
        titles =         [
                        # 'V123_info_stim',
                        # 'V123_info_stim_ori',
                        # 'V123_info_choice',
                        # 'V123',
                        # 'choice',
                        'combined_choice',
                        'combined_choice_res',
                        ]
        rois_groups =     [
                        # ['V1_center_'+decoding_type+'_stim', 'V2_center_'+decoding_type+'_stim', 'V3_center_'+decoding_type+'_stim'],
                        # ['V1_center_'+decoding_type+'_stim_ori', 'V2_center_'+decoding_type+'_stim_ori', 'V3_center_'+decoding_type+'_stim_ori'],
                        # ['V1_center_'+decoding_type+'_choice', 'V2_center_'+decoding_type+'_choice', 'V3_center_'+decoding_type+'_choice'],
                        # ['V1', 'V2', 'V3'],
                        # ['lr_M1_lat', 'sl_IPL_'+decoding_type, 'sl_SPL1_'+decoding_type, 'lr_aIPS_lat', 'sl_SPL2_'+decoding_type, 'sl_pIns_'+decoding_type, 'lr_PCeS_lat', 'sl_PCeS_IFG_'+decoding_type, 'sl_MFG_'+decoding_type, ],
                        ['lr_M1_lat', 'combined_choice_parietal_lat', 'combined_choice_parietal_sl',],
                        ['lr_M1_lat_res', 'combined_choice_parietal_lat_res', 'combined_choice_parietal_sl_res',],
                        ]
        rois_names =     [
                        # ['V1', 'V2', 'V3'],
                        # ['V1', 'V2', 'V3'],
                        # ['V1', 'V2', 'V3'],
                        # ['V1', 'V2', 'V3'],
                        # ['M1', 'IPL', 'SPL', 'aIPS1', 'aIPS2', 'pIns', 'IPS/PostCeS', 'PreCeS/IFG', 'MFG'],
                        ['M1', 'Comb_lat', 'Comb_sl'],
                        ['M1', 'Comb_lat', 'Comb_sl'],
                        ]
        
        for kind in ['cp']: # alternatively logistic regression...
            for title, rois, rois_name, in zip(titles, rois_groups, rois_names,):        
                for p in ['sp', 'cp', 'sp_without_choice', 'cp_without_stim']:
                    roi_values = []
                    for r, roi in enumerate(rois):
                        if kind == 'cp':
                            prob = np.load(os.path.join(self.data_folder, 'cp', '{}_{}.npy'.format(p, roi)))
                        roi_values.append(prob)
                    roi_values = pd.DataFrame(np.vstack(roi_values).T, columns=rois_name)
                    
                    dft = roi_values.stack().reset_index()
                    dft.columns = ['subject', 'measure', 'value']
                    fig = plt.figure(figsize=( (1+(len(rois)*0.15)),1.75))
                    ax = fig.add_subplot(111)
                    sns.barplot(x='measure',  y='value', units='subject', data=dft, ci=None, ax=ax, color='grey', alpha=0.5)
                    sns.stripplot(x='measure', y='value', data=dft, jitter=True, size=2, edgecolor='black', linewidth=0.25, color='black', alpha=1, ax=ax)
                    plt.ylabel(p)
                    for i, pp in enumerate(ax.patches):
                        p1 = myfuncs.permutationTest(roi_values[rois_name[i]], np.ones(roi_values.shape[0])/2.0, paired=True)[1]
                        if p1 < 0.055:
                            plt.text(pp.get_x(), 0, str(round(p1,3)), size=6, horizontalalignment='center')
                    plt.ylim(ymin=0.5, ymax=1)
                    sns.despine(offset=10, trim=True)
                    for item in ax.get_xticklabels():
                        item.set_rotation(45)
                    plt.tight_layout()
                    fig.savefig(os.path.join(self.figure_folder, 'V123', 'cp', 'bars_{}_{}_{}.pdf'.format(title, p, kind)))
                
                
                    if title == 'choice':   
                        nr_subjects = len(self.subjects)
                        dv = np.concatenate((roi_values['M1'], roi_values['IPL'], roi_values['SPL'],roi_values['aIPS1'], roi_values['aIPS2'], roi_values['pIns'], roi_values['IPS/PostCeS'], roi_values['PreCeS/IFG'], roi_values['MFG']))
                        area = np.concatenate((np.zeros(nr_subjects), np.ones(nr_subjects)*1, np.ones(nr_subjects)*2, np.ones(nr_subjects)*3, np.ones(nr_subjects)*4, np.ones(nr_subjects)*5, np.ones(nr_subjects)*6, np.ones(nr_subjects)*7, np.ones(nr_subjects)*8,))
                        subject = np.concatenate((np.arange(nr_subjects), np.arange(nr_subjects), np.arange(nr_subjects),np.arange(nr_subjects), np.arange(nr_subjects), np.arange(nr_subjects), np.arange(nr_subjects), np.arange(nr_subjects), np.arange(nr_subjects)))
                        d = rlc.OrdDict([('area', robjects.FactorVector(list(area.ravel()))), ('subject', robjects.FactorVector(list(subject.ravel()))), ('data', robjects.FloatVector(list(dv.ravel())))])
                        robjects.r.assign('dataf', robjects.DataFrame(d))
                        robjects.r('attach(dataf)')
                        statres = robjects.r('res = summary(aov(data ~ area + Error(subject/area), dataf))')
                        # statres = robjects.r('res = summary(aov(data ~ area, dataf))')
                        print statres
                
                    if title == 'combined_choice':
                        nr_subjects = len(self.subjects)
                        dv = np.concatenate((roi_values['M1'], roi_values['Comb_lat'], roi_values['Comb_sl'],))
                        area = np.concatenate((np.zeros(nr_subjects), np.ones(nr_subjects)*1, np.ones(nr_subjects)*2,))
                        subject = np.concatenate((np.arange(nr_subjects), np.arange(nr_subjects), np.arange(nr_subjects),))
                        d = rlc.OrdDict([('area', robjects.FactorVector(list(area.ravel()))), ('subject', robjects.FactorVector(list(subject.ravel()))), ('data', robjects.FloatVector(list(dv.ravel())))])
                        robjects.r.assign('dataf', robjects.DataFrame(d))
                        robjects.r('attach(dataf)')
                        statres = robjects.r('res = summary(aov(data ~ area + Error(subject/area), dataf))')
                        # statres = robjects.r('res = summary(aov(data ~ area, dataf))')
                        print statres
                    
                    if title == 'combined_choice_res':
                        nr_subjects = len(self.subjects)
                        dv = np.concatenate((roi_values['M1'], roi_values['Comb_lat'], roi_values['Comb_sl'],))
                        area = np.concatenate((np.zeros(nr_subjects), np.ones(nr_subjects)*1, np.ones(nr_subjects)*2,))
                        subject = np.concatenate((np.arange(nr_subjects), np.arange(nr_subjects), np.arange(nr_subjects),))
                        d = rlc.OrdDict([('area', robjects.FactorVector(list(area.ravel()))), ('subject', robjects.FactorVector(list(subject.ravel()))), ('data', robjects.FloatVector(list(dv.ravel())))])
                        robjects.r.assign('dataf', robjects.DataFrame(d))
                        robjects.r('attach(dataf)')
                        statres = robjects.r('res = summary(aov(data ~ area + Error(subject/area), dataf))')
                        # statres = robjects.r('res = summary(aov(data ~ area, dataf))')
                        print statres
                    
                
                # shell()
            
        # shell()
        #
        #
        # cp_0 = np.load(os.path.join(self.data_folder, 'cp', 'cp_{}.npy'.format('lr_pInsula_cv_lat')))
        # cp_1 = np.load(os.path.join(self.data_folder, 'cp', 'cp_{}.npy'.format('lr_aIPS_cv_lat')))
        # cp_2 = np.load(os.path.join(self.data_folder, 'cp', 'cp_{}.npy'.format('lr_PCeS_cv_lat')))
        # cp_3 = np.load(os.path.join(self.data_folder, 'cp', 'cp_{}.npy'.format('lr_M1_cv_lat')))
        # print myfuncs.permutationTest(cp_0, cp_1, paired=True)
        # print myfuncs.permutationTest(cp_1, cp_2, paired=True)
        # print myfuncs.permutationTest(cp_2, cp_3, paired=True)
        
        
        # shell()
        #
        # r_choice_h = np.load(os.path.join(self.data_folder, 'cp', 'lr_choice_h_{}.npy'.format(roi)), )
        # r_choice_l = np.load(os.path.join(self.data_folder, 'cp', 'lr_choice_l_{}.npy'.format(roi)), )
        # line_present_h = np.load(os.path.join(self.data_folder, 'cp', 'lr_line_stim_h_{}.npy'.format(roi)), )
        # line_present_l = np.load(os.path.join(self.data_folder, 'cp', 'lr_line_stim_l_{}.npy'.format(roi)), )
        # line_choice_h = np.load(os.path.join(self.data_folder, 'cp', 'lr_line_choice_h_{}.npy'.format(roi)), )
        # line_choice_l = np.load(os.path.join(self.data_folder, 'cp', 'lr_line_choice_l_{}.npy'.format(roi)), )
        #
        # bias_h = (line_choice_h[:,-1] + line_choice_h[:,0]) / 2.0
        # bias_l = (line_choice_l[:,-1] + line_choice_l[:,0]) / 2.0
        # bias = bias_h - bias_l
        #
        # intercept_h = line_choice_h[:,12]
        # intercept_l = line_choice_l[:,12]
        # bias_i = (intercept_h+intercept_l) / 2.0
        # delta_i = intercept_h - intercept_l
        #
        #
        # bias = (r_choice_h+r_choice_l)/2.0
        # bias_r = r_choice_h-r_choice_l
        #
        # dc = (np.load(os.path.join(self.data_folder, 'params_fMRI.npy'))[2,:] + np.load(os.path.join(self.data_folder, 'params_fMRI.npy'))[3,:] / 2.0)
        # delta_dc = np.load(os.path.join(self.data_folder, 'params_fMRI.npy'))[2,:] - np.load(os.path.join(self.data_folder, 'params_fMRI.npy'))[3,:]
        # delta_sp = np.load(os.path.join(self.data_folder, 'params_fMRI.npy'))[0,:] - np.load(os.path.join(self.data_folder, 'params_fMRI.npy'))[1,:]
    
    def CHOICE_SIGNALS_choice_probability_pupil_plot(self, prepare=False):
        
        # load dataframe:
        data_frame = pd.read_csv(os.path.join(self.data_folder, 'event_related_average_wb', 'MNI_dataframe_1.csv'))
        dfs = [data_frame[data_frame.subject == s] for s in self.subjects]
        
        # titles = ['parietal_cv', 'parietal', 'IPS_mean', 'IPS_info', 'stimulus_info', 'stimulus', 'all']
        # rois_groups = [['lr_pInsula_cv_lat', 'lr_aIPS_cv_lat', 'lr_PCeS_cv_lat', 'lr_M1_cv_lat',], ['lr_aIPS_lat', 'lr_PCeS_lat', 'lr_M1_lat',], ['FS_S_intrapariet_and_P_trans_mean'], ['FS_S_intrapariet_and_P_trans_info'], ['V1_center_info', 'V2_center_info', 'V3_center_info'], ['V1', 'V2', 'V3'], ['pupil_d', 'V1', 'V2', 'V3', 'lr_aIPS_lat', 'lr_PCeS_lat', 'lr_M1_lat']]
        # rois_names = [['pInsula', 'aIPS', 'IPS/PostCeS', 'M1'], ['aIPS', 'IPS/PostCeS', 'M1'], ['IPS_mean'], ['IPS_info'], ['V1', 'V2', 'V3'], ['V1', 'V2', 'V3'], ['TPR', 'V1', 'V2', 'V3', 'aIPS', 'IPS/PostCeS', 'M1']]
        
        # titles = ['all']
        # rois_groups = [['FS_S_intrapariet_and_P_trans_mean', 'FS_S_intrapariet_and_P_trans_info', 'lr_aIPS_lat', 'lr_PCeS_lat', 'lr_M1_lat',],]
        # rois_names = [['IPS_m', 'IPS_i', 'aIPS', 'IPS/PostCeS', 'M1'],]
        
        # titles = ['all']
        # rois_groups = [['V1_center_info_choice', 'V2_center_info_choice', 'sl_IPL_info', 'sl_SPL1_info', 'lr_PCeS_lat', 'sl_pIns_info', 'combined_2']]
        # rois_names = [['V1', 'V2', 'IPL', 'SPL', 'IPS/PCeS', 'pIns', 'combined']]
        
        
        
        titles = ['combined_choice_parietal']
        rois_groups = [['sl_IPL_info', 'sl_SPL1_info', 'lr_aIPS_lat', 'sl_SPL2_info', 'lr_PCeS_lat', 'sl_pIns_info', 'sl_PCeS_IFG_info', 'sl_MFG_info', 'combined_choice_parietal']]
        rois_names = [['IPL', 'SPL', 'aIPS1', 'aIPS2', 'IPS/PCeS', 'pIns', 'PCeS/IFG', 'MFG', 'Combined']]
        
        
        
        for kind in ['cp']:
            for title, rois, rois_name, in zip(titles, rois_groups, rois_names,):        
                fig = plt.figure(figsize=( (1+(len(rois)*0.3)),1.75))
                plt.ylim(ymin=0.5,)
                locs = np.arange(0,len(rois))
                for r, roi in enumerate(rois):
                    if kind == 'cp':
                        prob_h = np.load(os.path.join(self.data_folder, 'cp', 'cp_pupil_h_{}.npy'.format(roi)))
                        prob_l = np.load(os.path.join(self.data_folder, 'cp', 'cp_pupil_l_{}.npy'.format(roi)))
                    else:
                        prob = np.load(os.path.join(self.data_folder, 'cp', 'lr_choice_{}.npy'.format(roi)))
                    
                    # p1 = sp.stats.ttest_rel(prob_h, prob_l,)[1]
                    p1 = myfuncs.permutationTest(prob_h, prob_l, paired=True)[1]
                    bar_width = 0.3
                    my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}
                    plt.bar(locs[r]-(bar_width/2), prob_h.mean(), width = bar_width, yerr = sp.stats.sem(prob_h), color = 'r', alpha = 1, align = 'center', edgecolor = 'k', linewidth = 0, capsize = 0, ecolor = 'k', error_kw={'elinewidth':0.5,})
                    plt.bar(locs[r]+(bar_width/2), prob_l.mean(), width = bar_width, yerr = sp.stats.sem(prob_l), color = 'b', alpha = 1, align = 'center', edgecolor = 'k', linewidth = 0, capsize = 0, ecolor = 'k', error_kw={'elinewidth':0.5,})
                    plt.ylabel('Choice-predictive index (a.u.)')
                    if p1 < 0.05:
                        plt.text(s='{}'.format(round(p1, 3)), x=locs[r], y=gca().get_ylim()[0]+((gca().get_ylim()[1] - gca().get_ylim()[0]) / 8.0), size=5, horizontalalignment='center', rotation=0)
                plt.title(title)
                sns.despine(offset=10, trim=True)
                plt.xticks(locs, rois_name, rotation=45)
                plt.tight_layout()
                fig.savefig(os.path.join(self.figure_folder, 'V123', 'cp', 'bars_{}_pupil_{}.pdf'.format(title, kind)))
                
    def CHOICE_SIGNALS_ROC_curve(self):
        
        from scipy.integrate import cumtrapz
        
        # load dataframe:
        data_frame = pd.read_csv(os.path.join(self.data_folder, 'event_related_average_wb', 'MNI_dataframe_1.csv'))
        dfs = [data_frame[data_frame.subject == s] for s in self.subjects]
        
        shell()
        # for roi in ['V1_center_info', 'V2_center_info', 'V3_center_info']:
        # for roi in ['lr_aIPS_cv_lat', 'lr_PCeS_cv_lat', 'lr_M1_cv_lat', 'lr_pInsula_cv_lat']:
        for roi in ['V1_center_info_stim_ori']:
            
            # Choice probability:
            measure_yes = []
            measure_no = []
            measure_hit = []
            measure_fa = []
            measure_miss = []
            measure_cr = []
            for i in range(len(self.subjects)):
                dummy = dfs[i][roi][self.hit[i]] 
                dummy = dummy[-np.isnan(dummy)]
                measure_hit.append( dummy )
                dummy = dfs[i][roi][self.fa[i]] 
                dummy = dummy[-np.isnan(dummy)]
                measure_fa.append( dummy )
                dummy = dfs[i][roi][self.miss[i]] 
                dummy = dummy[-np.isnan(dummy)]
                measure_miss.append( dummy )
                dummy = dfs[i][roi][self.cr[i]] 
                dummy = dummy[-np.isnan(dummy)]
                measure_cr.append( dummy )
                
                dummy = dfs[i][roi][self.yes[i]] 
                dummy = dummy[-np.isnan(dummy)]
                measure_yes.append( dummy )
                dummy = dfs[i][roi][self.no[i]] 
                dummy = dummy[-np.isnan(dummy)]
                measure_no.append( dummy )
            
            sp_yes = np.zeros(len(self.subjects))
            sp_no = np.zeros(len(self.subjects))
            cp_present = np.zeros(len(self.subjects))
            cp_absent = np.zeros(len(self.subjects))
            for i in range(len(self.subjects)):
                sp_yes[i] = myfuncs.roc_analysis(measure_hit[i], measure_fa[i])[0]
                sp_no[i] = myfuncs.roc_analysis(measure_miss[i], measure_cr[i])[0]
                cp_present[i] = myfuncs.roc_analysis(measure_hit[i], measure_miss[i])[0]
                cp_absent[i] = myfuncs.roc_analysis(measure_fa[i], measure_cr[i])[0]
            stim_p = (sp_yes + sp_no) / 2.0
            cp = (cp_present + cp_absent) / 2.0
            
            # i = 12
            i = 8
            x = measure_hit[i]
            y = measure_miss[i]
            nx = len(x)
            ny = len(y)
            z = np.concatenate((x,y)) 
            c = np.sort(z) 
            det = np.zeros((c.shape[0],2))
            for ic in range(c.shape[0]):
                det[ic,0] = (x > c[ic]).sum() / float(nx)
                det[ic,1] = (y > c[ic]).sum() / float(ny)
            t1 = np.sort(det[:,0])
            t2 = np.argsort(det[:,0])
            roc = np.vstack(( [0,0],det[t2,:],[1,1] ))
            t1 = sp.integrate.cumtrapz(roc[:,0],roc[:,1])
        
            # x = measure_fa[i]
            # y = measure_cr[i][:-2]
            # nx = len(x)
            # ny = len(y)
            # z = np.concatenate((x,y))
            # c = np.sort(z)
            # det = np.zeros((c.shape[0],2))
            # for ic in range(c.shape[0]):
            #     det[ic,0] = (x > c[ic]).sum() / float(nx)
            #     det[ic,1] = (y > c[ic]).sum() / float(ny)
            # t1 = np.sort(det[:,0])
            # t2 = np.argsort(det[:,0])
            # roc2 = np.vstack(( [0,0],det[t2,:],[1,1] ))
            # t1 = sp.integrate.cumtrapz(roc[:,0],roc[:,1])
            #
            # roc = (roc1+roc2) / 2.0
            
            # shell()
            
            fig = plt.figure(figsize=(2,2))
            plt.plot(np.linspace(0,1,len(roc[:,0])), np.linspace(0,1,len(roc[:,0])))
            plt.plot(roc[:,1],roc[:,0])
            plt.title('AUC = {}'.format(round(t1[-1], 4)))
            plt.ylabel('True positive rate')
            plt.xlabel('False positive rate')
            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, 'V123', 'cp', 'ROC.pdf'))
            
            fig = plt.figure(figsize=(2,2))
            ax = fig.add_subplot(111)
            # plt.hist(np.array(measure_hit[i]), bins=10, color='m', alpha=1, lw=0.25)
            # plt.hist(np.array(measure_miss[i]), bins=10, color='c', alpha=1, lw=0.25)
            plt.hist(np.array(measure_hit[i]), bins=10, color='m', alpha=1, lw=0.25)
            plt.hist(np.array(measure_fa[i]), bins=10, color='c', alpha=1, lw=0.25)
            x_grid = np.linspace(min(np.array(measure_hit[i]).min(), np.array(measure_miss[i]).min()), max(np.array(measure_hit[i]).max(), np.array(measure_miss[i]).max()), 50)
            a = sp.histogram(np.array(measure_hit[i]), bins=10,)
            kde = myfuncs.kde_sklearn(x=np.array(measure_hit[i]), x_grid=x_grid, bandwidth=0.5) 
            kde = kde / max(kde) * a[0].max()
            # plt.plot(x_grid, kde, lw=1.5, color='m', label='yes')
            a = sp.histogram(np.array(measure_miss[i]), bins=10,)
            kde = myfuncs.kde_sklearn(x=np.array(measure_miss[i]), x_grid=x_grid, bandwidth=0.5)
            kde = kde / max(kde) * a[0].max()
            # plt.plot(x_grid, kde, lw=1.5, color='c', label='no')
            plt.legend()
            plt.xlabel('Lateralization\n(% signal change)')
            plt.ylabel('Trial count')
            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, 'V123', 'cp', 'ROC3.pdf'))
            
            fig = plt.figure(figsize=(2,2))
            ax = fig.add_subplot(111)
            # plt.hist(np.array(measure_cr[i]), bins=10, color='c', alpha=1, lw=0.25)
            # plt.hist(np.array(measure_fa[i]), bins=10, color='m', alpha=1, lw=0.25)
            
            plt.hist(np.array(measure_cr[i]), bins=10, color='c', alpha=1, lw=0.25)
            plt.hist(np.array(measure_miss[i]), bins=10, color='m', alpha=1, lw=0.25)
            
            x_grid = np.linspace(min(np.array(measure_fa[i]).min(), np.array(measure_cr[i]).min()), max(np.array(measure_fa[i]).max(), np.array(measure_cr[i]).max()), 50)
            a = sp.histogram(np.array(measure_fa[i]), bins=10,)
            kde = myfuncs.kde_sklearn(x=np.array(measure_fa[i]), x_grid=x_grid, bandwidth=0.5) 
            kde = kde / max(kde) * a[0].max()
            # plt.plot(x_grid, kde, lw=1.5, color='m', label='yes')
            a = sp.histogram(np.array(measure_cr[i]), bins=10,)
            kde = myfuncs.kde_sklearn(x=np.array(measure_cr[i]), x_grid=x_grid, bandwidth=0.5)
            kde = kde / max(kde) * a[0].max()
            # plt.plot(x_grid, kde, lw=1.5, color='c', label='no')
            plt.legend()
            plt.xlabel('Lateralization\n(% signal change)')
            plt.ylabel('Trial count')
            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, 'V123', 'cp', 'ROC4.pdf'))
            
            shell()
            
    def CHOICE_SIGNALS_bars_to_criterion(self):
        
        n_bins = 10
        
        # load dataframe:
        data_frame = pd.read_csv(os.path.join(self.data_folder, 'event_related_average_wb', 'MNI_dataframe_1.csv'))
        dfs = [data_frame[data_frame.subject == s] for s in self.subjects]
        
        shell()
        
        titles = ['parietal',]
        # rois_groups = [['pupil_d', 'V1', 'V2', 'V3', 'V1_center_info', 'V2_center_info', 'V3_center_info', 'lr_aIPS_lat', 'lr_PCeS_lat', 'lr_M1_lat',]]
        # rois_names = [['TPR', 'V1_tp', 'V2_tp', 'V3_tp', 'V1_s', 'V2_s', 'V3_s', 'aIPS', 'IPS/PostCeS', 'M1'],]
        rois_groups = [[ 'V1', 'V2', 'V3',]]
        rois_names = [['V1_tp', 'V2_tp', 'V3_tp',],]
        
        # ylims = [(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.1, 0.1), (-0.8, 0.8), (-0.8, 0.8), (-0.1, 0.1), (-0.1, 0.1)]
        
        for title, rois, rois_name, in zip(titles, rois_groups, rois_names,):        
            
            fig = plt.figure(figsize=( (1+(len(rois)*0.15)),1.75))
            # plt.ylim(ymin=0.45, ymax=0.8)
            locs = np.arange(0,len(rois))
            for r, roi in enumerate(rois):
            
                var_X = roi 
                bin_by = var_X
                var_Y = 'behaviour'
            
                bins = []
                BOLD_subjects = []
                for i in range(len(self.subjects)):
                
                    # regress RT per session:
                    # -------------------
                    inds_s = np.zeros((len(self.pupil_data[i]), n_bins), dtype=bool)
                    trial_nr = 0
                    if roi == 'pupil_d':
                        BOLD = np.array(self.pupil_data[i][roi])
                    else:
                        BOLD = np.array(dfs[i][var_X])
                    for s in np.array(np.unique(self.pupil_data[i]['session']), dtype=int):
                        # rt = np.array(self.pupil_data[i]['rt'])[np.array(self.pupil_data[i].session == s)]
                        # present = np.array(self.pupil_data[i]['present'], dtype=int)[np.array(self.pupil_data[i].session == s)]
                        if bin_by == 'pupil_d':
                            bin_measure = np.array(self.pupil_data[i][bin_by])[np.array(self.pupil_data[i].session == s)]
                            # bin_measure = myfuncs.lin_regress_resid(bin_measure, [rt])
                        else:
                            bin_measure = np.array(dfs[i][var_X])[np.array(self.pupil_data[i].session == s)]
                        nr_trials_in_run = len(bin_measure)
                        inds = np.array_split(np.argsort(bin_measure), n_bins)
                        for b in range(n_bins):
                            ind = np.zeros(len(bin_measure), dtype=bool)
                            ind[inds[b]] = True
                            inds_s[trial_nr:trial_nr+nr_trials_in_run, b] = ind
                        # if 'lat' not in roi:
                            # BOLD[np.array(self.pupil_data[i].session == s)] = myfuncs.lin_regress_resid(BOLD[np.array(self.pupil_data[i].session == s)], [rt])
                        trial_nr += nr_trials_in_run
                    BOLD_subjects.append(BOLD)
                    bins.append(inds_s)
                
                # compute criterion values:
                varX = np.zeros((len(self.subjects), n_bins))
                varY = np.zeros((len(self.subjects), n_bins))
                varY2 = np.zeros((len(self.subjects), n_bins))
                for i in range(len(self.subjects)):
                    for b in range(n_bins):
                        data = self.pupil_data[i][bins[i][:,b]]
                        varX[i,b] = BOLD_subjects[i][bins[i][:,b]].mean()
                        if var_Y == 'behaviour':
                            d, c = myfuncs.SDT_measures(target=np.array(data.present), hit=np.array(data.hit), fa=np.array(data.fa))
                            # c = np.array(data.yes).sum() / float(len(np.array(data.yes)))
                            varY[i,b] = c
                            varY2[i,b] = d
                        else:
                            varY[i,b] = np.array(dfs[i][var_Y][bins[i][:,b]]).mean()
            

                rv = np.zeros(len(self.subjects))
                for i in range(len(self.subjects)):
                    x = varX[i,:]
                    y = varY2[i,:]
                    slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x,y)
                    (m,b) = sp.polyfit(x,y,1)
                    rv[i] = r_value
            
                p1 = sp.stats.ttest_rel(rv, np.zeros(len(rv)),)[1]
                
                print myfuncs.permutationTest(rv, np.zeros(len(rv)), paired=True)[1]
                
                bar_width = 0.5
                my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}
                plt.bar(locs[r], rv.mean(), width = bar_width, yerr = sp.stats.sem(rv), color = 'k', alpha = 1, align = 'center', edgecolor = 'k', linewidth = 0, capsize = 0, ecolor = 'k', error_kw={'elinewidth':0.5,})
                plt.ylabel('Correlation to criterion')
                plt.text(s='{}'.format(round(p1, 3)), x=locs[r], y=gca().get_ylim()[0]+((gca().get_ylim()[1] - gca().get_ylim()[0]) / 8.0), size=5, horizontalalignment='center', rotation=90)
            plt.title(title)
            sns.despine(offset=10, trim=True)
            plt.xticks(locs, rois_name, rotation=45)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, 'V123', 'cp', 'bars_criterion_{}.pdf'.format(title)))
        
        
        
        
        
        
        # roi_measure = []
        # roi_r = []
        #
        #
        # shell()
        #
        # var_Y = 'behaviour'
        # for var_X in ['lr_aIPS_lat', 'lr_PCeS_lat', 'lr_M1_lat']:
        #     bin_by = var_X
        #     R_varY = []
        #     R_varY2 = []
        #     for n_bins in [10]:
        #
        
        
        
        
    
    def CHOICE_SIGNALS_behavioural_correlation(self):
        
        random_effects = True
        
        # load dataframe:
        data_frame = pd.read_csv(os.path.join(self.data_folder, 'event_related_average_wb', 'MNI_dataframe_1.csv'))
        dfs = [data_frame[data_frame.subject == s] for s in self.subjects]
        
        
        roi_measure = []
        roi_r = []
        
        var_Y = 'behaviour'
        # for var_X in ['V1', 'V2', 'V3', 'V123_center_info', 'aIPS_lat', 'M1_lat', 'aIPS', 'M1',]:
        # for var_X in ['aIPS_lat', 'M1_lat']:
        # for var_X in ['lr_IPS_lat', 'lr_aIPS_lat', 'lr_insula1_lat', 'lr_insula2_lat', 'lr_M1_lat']:
        
        color1 = 'mediumslateblue'
        color2 = 'k'
        
        # for var_X in ['lr_PCeS_lat', 'lr_aIPS_lat', 'lr_M1_lat']:
        for var_X in ['lr_PCeS_lat']:
            bin_by = 'pupil_d'
            # bin_by = var_X
            
            R_varY = []
            R_varY2 = []
            
            # for n_bins in np.arange(5,11):
            # for n_bins in [8,10,12]:
            for n_bins in [10]:
                
                bins = []
                BOLD_subjects = []
                for i in range(len(self.subjects)):
                    
                    # regress RT per session:
                    # -------------------
                    inds_s = np.zeros((len(self.pupil_data[i]), n_bins), dtype=bool)
                    trial_nr = 0
                    BOLD = np.array(dfs[i][var_X])
                    for s in np.array(np.unique(self.pupil_data[i]['session']), dtype=int):
                        rt = np.array(self.pupil_data[i]['rt'])[np.array(self.pupil_data[i].session == s)]
                        present = np.array(self.pupil_data[i]['present'], dtype=int)[np.array(self.pupil_data[i].session == s)]
                        if bin_by == 'pupil_d':
                            bin_measure = np.array(self.pupil_data[i][bin_by])[np.array(self.pupil_data[i].session == s)]
                            bin_measure = myfuncs.lin_regress_resid(bin_measure, [rt]) + np.mean(bin_measure)
                        else:
                            bin_measure = np.array(dfs[i][var_X])[np.array(self.pupil_data[i].session == s)]
                        nr_trials_in_run = len(bin_measure)
                        inds = np.array_split(np.argsort(bin_measure), n_bins)
                        for b in range(n_bins):
                            ind = np.zeros(len(bin_measure), dtype=bool)
                            ind[inds[b]] = True
                            inds_s[trial_nr:trial_nr+nr_trials_in_run, b] = ind
                        # if 'lat' not in var_X:
                            # BOLD[np.array(self.pupil_data[i].session == s)] = myfuncs.lin_regress_resid(BOLD[np.array(self.pupil_data[i].session == s)], [rt])
                        trial_nr += nr_trials_in_run
                    BOLD_subjects.append(BOLD)
                    bins.append(inds_s)
                    
                # compute criterion values:
                varX = np.zeros((len(self.subjects), n_bins))
                varY = np.zeros((len(self.subjects), n_bins))
                varY2 = np.zeros((len(self.subjects), n_bins))
                for i in range(len(self.subjects)):
                    for b in range(n_bins):
                        data = self.pupil_data[i][bins[i][:,b]]
                        varX[i,b] = BOLD_subjects[i][bins[i][:,b]].mean()
                        varX[i,b] = data[bin_by].mean()
                        if var_Y == 'behaviour':
                            d, c = myfuncs.SDT_measures(target=np.array(data.present), hit=np.array(data.hit), fa=np.array(data.fa))
                            # c = np.array(data.yes).sum() / float(len(np.array(data.yes)))
                            varY[i,b] = d
                            varY2[i,b] = c
                        else:
                            varY[i,b] = np.array(dfs[i][var_Y][bins[i][:,b]]).mean()
                
                # plot mean traces as a function of pupil:
                fig = plt.figure(figsize=(2.1,2))
                x = np.arange(n_bins)+1
                
                ax = fig.add_subplot(111)
                ax.errorbar(varX.mean(axis=0), varY.mean(axis=0), xerr=sp.stats.sem(varX, axis=0), yerr=sp.stats.sem(varY, axis=0), fmt='o', markersize=6, color=color1, alpha=1, capsize=0, elinewidth=0.5, markeredgecolor='w', markeredgewidth=0.5)
                # ax.set_xticks(x)
                # ax.set_ylim((1, 2))
                ax.set_ylabel(var_Y, color=color1)
                ax.set_xlabel('{}'.format(var_X))
                
                if random_effects:
                    regression_line = []
                    r1 = np.zeros(len(self.subjects))
                    for i in range(len(self.subjects)):
                        x = varX[i,:]
                        y = varY[i,:]
                        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x,y)
                        (m,b) = sp.polyfit(x,y,1)
                        x_line = np.linspace(min(varX[:,:].mean(axis=0))-((max(varX[:,:].mean(axis=0))-min(varX[:,:].mean(axis=0))) / 8.0), max(varX[:,:].mean(axis=0))+((max(varX[:,:].mean(axis=0))-min(varX[:,:].mean(axis=0))) / 8.0), 50)
                        regression_line.append(sp.polyval([m,b],x_line))
                        r1[i] = r_value
                    # p1 = myfuncs.permutationTest(r1, np.zeros(len(r1)), paired=True)[1]
                    p1 = sp.stats.wilcoxon(r1, np.zeros(len(r1)),)[1]
                    R_varY.append(r1)
                    regression_line = np.vstack(regression_line)
                    if p1 < 0.05:
                        ax.plot(x_line, regression_line.mean(axis=0), color=color1, alpha=1)
                        ax.fill_between(x_line, regression_line.mean(axis=0)-sp.stats.sem(regression_line, axis=0), regression_line.mean(axis=0)+sp.stats.sem(regression_line, axis=0), color=color1, alpha=0.2)
                else:
                    x = varX[:,:].mean(axis=0)
                    y = varY[:,:].mean(axis=0)
                    slope, intercept, r_value, p1, std_err = sp.stats.linregress(x,y)
                    (m,b) = sp.polyfit(x,y,1)
                    x_line = np.linspace(min(varX[:,:].mean(axis=0))-((max(varX[:,:].mean(axis=0))-min(varX[:,:].mean(axis=0))) / 8.0), max(varX[:,:].mean(axis=0))+((max(varX[:,:].mean(axis=0))-min(varX[:,:].mean(axis=0))) / 8.0), 50)
                    regression_line = sp.polyval([m,b],x_line)
                    r1 = r_value
                    if p1 < 0.05:
                        ax.plot(x_line, regression_line, color=color1, alpha=1)
                ax.set_xlim(min(varX[:,:].mean(axis=0))-((max(varX[:,:].mean(axis=0))-min(varX[:,:].mean(axis=0))) / 4.0), max(varX[:,:].mean(axis=0))+((max(varX[:,:].mean(axis=0))-min(varX[:,:].mean(axis=0))) / 4.0))
                for tl in ax.get_yticklabels():
                    tl.set_color(color1)
            
                if var_Y == 'behaviour':
                    ax2 = ax.twinx()
                    ax2.errorbar(varX.mean(axis=0), varY2.mean(axis=0), xerr=sp.stats.sem(varX, axis=0), yerr=sp.stats.sem(varY2, axis=0), fmt='o', markersize=6, color=color2, alpha=1, capsize=0,  elinewidth=0.5, markeredgecolor='w', markeredgewidth=0.5)
                    # ax2.set_ylim((-0.2, 0.8))
                    ax2.set_ylabel(var_Y, color=color2)
                    
                    if random_effects:
                        regression_line = []
                        r2 = np.zeros(len(self.subjects))
                        for i in range(len(self.subjects)):
                            x = varX[i,:]
                            y = varY2[i,:]
                            slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x,y)
                            (m,b) = sp.polyfit(x,y,1)
                            x_line = np.linspace(min(varX[:,:].mean(axis=0))-((max(varX[:,:].mean(axis=0))-min(varX[:,:].mean(axis=0))) / 8.0), max(varX[:,:].mean(axis=0))+((max(varX[:,:].mean(axis=0))-min(varX[:,:].mean(axis=0))) / 8.0), 50)
                            regression_line.append(sp.polyval([m,b],x_line))
                            r2[i] = r_value
                        # p2 = myfuncs.permutationTest(r2, np.zeros(len(r1)), paired=True)[1]
                        p2 = sp.stats.wilcoxon(r2, np.zeros(len(r1)),)[1]
                        R_varY2.append(r2)
                        regression_line = np.vstack(regression_line)
                        if p2 < 0.05:
                            ax2.plot(x_line, regression_line.mean(axis=0), color=color2, alpha=1)
                            ax2.fill_between(x_line, regression_line.mean(axis=0)-sp.stats.sem(regression_line, axis=0), regression_line.mean(axis=0)+sp.stats.sem(regression_line, axis=0), color=color2, alpha=0.2)
                        if (var_X == 'lr_PCeS_lat') or (var_X == 'lr_M1_lat'):
                            roi_r.append(r2)
                    else:
                        x = varX[:,:].mean(axis=0)
                        y = varY2[:,:].mean(axis=0)
                        slope, intercept, r_value, p2, std_err = sp.stats.linregress(x,y)
                        (m,b) = sp.polyfit(x,y,1)
                        x_line = np.linspace(min(varX[:,:].mean(axis=0))-((max(varX[:,:].mean(axis=0))-min(varX[:,:].mean(axis=0))) / 8.0), max(varX[:,:].mean(axis=0))+((max(varX[:,:].mean(axis=0))-min(varX[:,:].mean(axis=0))) / 8.0), 50)
                        regression_line = sp.polyval([m,b],x_line)
                        r2 = r_value
                        if p2 < 0.05:
                            ax2.plot(x_line, regression_line, color=color2, alpha=1)
                        
                        if (var_X == 'lr_PCeS_lat') or (var_X == 'lr_M1_lat'):
                            roi_measure.append(np.vstack((x,y)).T)
                    ax.set_xlim(min(varX[:,:].mean(axis=0))-((max(varX[:,:].mean(axis=0))-min(varX[:,:].mean(axis=0))) / 4.0), max(varX[:,:].mean(axis=0))+((max(varX[:,:].mean(axis=0))-min(varX[:,:].mean(axis=0))) / 4.0))
                    for tl in ax2.get_yticklabels():
                        tl.set_color(color2)
                if var_Y == 'behaviour':
                    ax.set_title('r = {}, p = {}\nr = {}, p = {}'.format(round(r1.mean(), 3), round(p1, 3), round(r2.mean(), 3), round(p2, 3)))
                else:
                    ax.set_title('r = {}, p = {}'.format(round(r1.mean(), 3), round(p1, 3),))
            
                sns.despine(offset=10, trim=True, right=False)
                plt.tight_layout()
                fig.savefig(os.path.join(self.figure_folder, 'V123', 'cp', 'behavioural_corr_{}_{}_{}_{}_{}.pdf'.format(n_bins, var_X, var_Y, bin_by, self.split_by)))    
                
                # shell()
                
        print myfuncs.permutationTest_correlation(roi_measure[0], roi_measure[1])
        
        # print myfuncs.permutationTest(roi_r[0], roi_r[1], paired=True)
    
    def CHOICE_SIGNALS_M1_correlation(self):
        
        random_effects = True
        
        # load dataframe:
        data_frame = pd.read_csv(os.path.join(self.data_folder, 'event_related_average_wb', 'MNI_dataframe_1.csv'))
        dfs = [data_frame[data_frame.subject == s] for s in self.subjects]
        
        
        roi_measure = []
        roi_r = []
        
        var_Y = 'behaviour'
        # for var_X in ['V1', 'V2', 'V3', 'V123_center_info', 'aIPS_lat', 'M1_lat', 'aIPS', 'M1',]:
        # for var_X in ['aIPS_lat', 'M1_lat']:
        # for var_X in ['lr_IPS_lat', 'lr_aIPS_lat', 'lr_insula1_lat', 'lr_insula2_lat', 'lr_M1_lat']:
        
        color1 = 'mediumslateblue'
        color2 = 'k'
        
        for var_X in ['lr_PCeS_lat', 'lr_aIPS_lat', 'lr_M1_lat']:
        # for var_X in ['V1', 'V2', 'V3']:
            # bin_by = 'pupil_d'
            bin_by = var_X
            
            R_varY = []
            R_varY2 = []
            
            # for n_bins in np.arange(5,11):
            # for n_bins in [8,10,12]:
            for n_bins in [10]:
                
                bins = []
                BOLD_subjects = []
                for i in range(len(self.subjects)):
                    
                    # regress RT per session:
                    # -------------------
                    inds_s = np.zeros((len(self.pupil_data[i]), n_bins), dtype=bool)
                    trial_nr = 0
                    BOLD = np.array(dfs[i][var_X])
                    for s in np.array(np.unique(self.pupil_data[i]['session']), dtype=int):
                        rt = np.array(self.pupil_data[i]['rt'])[np.array(self.pupil_data[i].session == s)]
                        present = np.array(self.pupil_data[i]['present'], dtype=int)[np.array(self.pupil_data[i].session == s)]
                        if bin_by == 'pupil_d':
                            bin_measure = np.array(self.pupil_data[i][bin_by])[np.array(self.pupil_data[i].session == s)]
                            bin_measure = myfuncs.lin_regress_resid(bin_measure, [rt])
                        else:
                            bin_measure = np.array(dfs[i][var_X])[np.array(self.pupil_data[i].session == s)]
                        nr_trials_in_run = len(bin_measure)
                        inds = np.array_split(np.argsort(bin_measure), n_bins)
                        for b in range(n_bins):
                            ind = np.zeros(len(bin_measure), dtype=bool)
                            ind[inds[b]] = True
                            inds_s[trial_nr:trial_nr+nr_trials_in_run, b] = ind
                        # if 'lat' not in var_X:
                            # BOLD[np.array(self.pupil_data[i].session == s)] = myfuncs.lin_regress_resid(BOLD[np.array(self.pupil_data[i].session == s)], [rt])
                        trial_nr += nr_trials_in_run
                    BOLD_subjects.append(BOLD)
                    bins.append(inds_s)
                    
                # compute criterion values:
                varX = np.zeros((len(self.subjects), n_bins))
                varY = np.zeros((len(self.subjects), n_bins))
                varY2 = np.zeros((len(self.subjects), n_bins))
                for i in range(len(self.subjects)):
                    for b in range(n_bins):
                        data = self.pupil_data[i][bins[i][:,b]]
                        varX[i,b] = BOLD_subjects[i][bins[i][:,b]].mean()
                        # varX[i,b] = data[bin_by].mean()
                        varY[i,b] = np.array(dfs[i]['lr_M1_lat'][bins[i][:,b]]).mean()
                
                # plot mean traces as a function of pupil:
                fig = plt.figure(figsize=(2.1,2))
                x = np.arange(n_bins)+1
                
                ax = fig.add_subplot(111)
                ax.errorbar(varX.mean(axis=0), varY.mean(axis=0), xerr=sp.stats.sem(varX, axis=0), yerr=sp.stats.sem(varY, axis=0), fmt='o', markersize=6, color=color2, alpha=1, capsize=0,  elinewidth=0.5, markeredgecolor='w', markeredgewidth=0.5)
                # ax2.set_ylim((-0.2, 0.8))
                ax.set_ylabel(var_Y, color=color2)
                
                if random_effects:
                    regression_line = []
                    r2 = np.zeros(len(self.subjects))
                    for i in range(len(self.subjects)):
                        x = varX[i,:]
                        y = varY[i,:]
                        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x,y)
                        (m,b) = sp.polyfit(x,y,1)
                        x_line = np.linspace(min(varX[:,:].mean(axis=0))-((max(varX[:,:].mean(axis=0))-min(varX[:,:].mean(axis=0))) / 8.0), max(varX[:,:].mean(axis=0))+((max(varX[:,:].mean(axis=0))-min(varX[:,:].mean(axis=0))) / 8.0), 50)
                        regression_line.append(sp.polyval([m,b],x_line))
                        r2[i] = r_value
                    # p2 = myfuncs.permutationTest(r2, np.zeros(len(r1)), paired=True)[1]
                    p2 = sp.stats.wilcoxon(r2, np.zeros(len(r2)),)[1]
                    R_varY2.append(r2)
                    regression_line = np.vstack(regression_line)
                    if p2 < 0.05:
                        ax.plot(x_line, regression_line.mean(axis=0), color=color2, alpha=1)
                        ax.fill_between(x_line, regression_line.mean(axis=0)-sp.stats.sem(regression_line, axis=0), regression_line.mean(axis=0)+sp.stats.sem(regression_line, axis=0), color=color2, alpha=0.2)
                    if (var_X == 'lr_PCeS_lat') or (var_X == 'lr_M1_lat'):
                        roi_r.append(r2)
                else:
                    x = varX[:,:].mean(axis=0)
                    y = varY[:,:].mean(axis=0)
                    slope, intercept, r_value, p2, std_err = sp.stats.linregress(x,y)
                    (m,b) = sp.polyfit(x,y,1)
                    x_line = np.linspace(min(varX[:,:].mean(axis=0))-((max(varX[:,:].mean(axis=0))-min(varX[:,:].mean(axis=0))) / 8.0), max(varX[:,:].mean(axis=0))+((max(varX[:,:].mean(axis=0))-min(varX[:,:].mean(axis=0))) / 8.0), 50)
                    regression_line = sp.polyval([m,b],x_line)
                    r2 = r_value
                    if p2 < 0.05:
                        ax.plot(x_line, regression_line, color=color2, alpha=1)
                    
                    if (var_X == 'lr_PCeS_lat') or (var_X == 'lr_M1_lat'):
                        roi_measure.append(np.vstack((x,y)).T)
                ax.set_xlim(min(varX[:,:].mean(axis=0))-((max(varX[:,:].mean(axis=0))-min(varX[:,:].mean(axis=0))) / 4.0), max(varX[:,:].mean(axis=0))+((max(varX[:,:].mean(axis=0))-min(varX[:,:].mean(axis=0))) / 4.0))
                for tl in ax.get_yticklabels():
                    tl.set_color(color2)
                ax.set_title('r = {}, p = {}'.format(round(r2.mean(), 3), round(p2, 3),))
            
                sns.despine(offset=10, trim=True, right=False)
                plt.tight_layout()
                fig.savefig(os.path.join(self.figure_folder, 'V123', 'cp', 'behavioural_M1_corr_{}_{}_{}_{}_{}.pdf'.format(n_bins, var_X, var_Y, bin_by, self.split_by)))    
                
                # shell()
                
        print myfuncs.permutationTest_correlation(roi_measure[0], roi_measure[1])
    
    def CHOICE_SIGNALS_mediation_analysis(self):
        
        data_type='clean_4th_ventricle'
        time_locked = 'stim_locked'
        self.make_pupil_BOLD_dataframe(data_type=data_type, time_locked=time_locked, regress_iti=False, regress_rt=True, regress_stimulus=False)
        
        data_brainstem = self.data_frame.copy()
        dfs_brainstem = [data_brainstem[data_brainstem.subject == s] for s in self.subjects]
        
        data_frame = pd.read_csv(os.path.join(self.data_folder, 'event_related_average_wb', 'MNI_dataframe_1.csv'))
        data_frame = pd.concat([data_frame[data_frame.subject == s] for s in self.subjects])
        
        # rois = ['sl_IPL_info', 'sl_SPL1_info', 'lr_aIPS_lat', 'sl_SPL2_info', 'sl_pIns_info', 'lr_PCeS_lat', 'lr_M1_lat', 'sl_PCeS_IFG_info', 'sl_MFG_info','combined_choice_parietal1a', 'combined_choice_parietal1b', 'combined_choice_parietal2']
        # roi_names = ['IPL', 'SPL', 'aIPS1', 'aIPS2', 'pIns', 'IPS/PostCeS', 'M1', 'PreCeS/IFG', 'MFG', 'comb1a', 'comb1b', 'comb2',]
        
        rois = ['lr_M1_lat', 'combined_choice_parietal_lat', 'combined_choice_parietal_sl']
        roi_names = ['M1', 'comb_lat', 'comb_sl',]
        
        # shell()
        
        # fit the path models:
        ######################
        import rpy2.robjects as robjects
        import pandas.rpy.common as com
        nr_subjects = len(self.subjects)
        nr_rois = 1
        
        robjects.r('library(lavaan)')
        
        for X in ['pupil_d']:
            
            stim = []
            choice = []
            rs_across = []
            for M in rois:
                
                d = {
                    'pupil_d' : pd.Series(np.array(pd.concat(self.pupil_data)['pupil_d']), dtype=float),
                    'vta' : pd.Series(np.array(pd.concat(dfs_brainstem)['mean_VTA_d']), dtype=float),
                    M : pd.Series(np.array(data_frame[M], dtype=float)),
                    'choice_a' : pd.Series(np.array(pd.concat(self.pupil_data)['yes']), dtype=int),
                    'subj_idx' : pd.Series(self.subj_idx),
                    'stimulus' : pd.Series(np.array(pd.concat(self.pupil_data)['present']), dtype=int),
                    'session' : pd.Series(np.concatenate(self.session)),
                    'run' : pd.Series(np.concatenate(self.run)),
                    'pupil_high' : pd.Series(np.concatenate(self.pupil_h_ind)),
                    'rt' : pd.Series(np.concatenate(self.rt)),
                    }
                data = pd.DataFrame(d)
                
                a = np.zeros(nr_subjects)
                for i, s in enumerate(self.subjects):
                    a[i] = sp.stats.pearsonr(data[(data['subj_idx'] == i)][M], data[(data['subj_idx'] == i)]['stimulus'])[0]
                stim.append(a)
                
                # take out stimulus:
                for i in range(len(self.subjects)):
                    ind = (data['subj_idx'] == i)
                    # for measure in ['pupil_d', M]:
                    for measure in [M]:
                        data.ix[ind,measure] = myfuncs.lin_regress_resid( np.array(data.ix[ind,measure]), [np.array(data.ix[ind,'stimulus'], dtype=int)]) + data.ix[ind,measure].mean()

                # initialize behavior operator:
                self.behavior = myfuncs.behavior(data)
                for y in [M]:
                    # model_comp = 'bayes'
                    model_comp = 'seq'
                    bin_by = X
                    bins = 5
                    fig, rs = self.behavior.SDT_correlation(bin_by=bin_by, n_bins=bins, y1=y, model_comp=model_comp)
                    fig.savefig(os.path.join(self.figure_folder, 'V123', 'cp', 'a_path_{}_{}.pdf'.format(X,M)))
                    rs_across.append(rs)
                    
                # # z-score:
                # for i in range(len(self.subjects)):
                #     for s in np.unique(data.query('subj_idx=={}'.format(i))['session']):
                #         ind = (data['subj_idx'] == i) & (data['session'] == s)
                #         for measure in [M, 'pupil_d']:
                #             data.ix[ind,measure] = (data.ix[ind,measure] - data.ix[ind,measure].mean()) / data.ix[ind,measure].std()

                # # z-score:
                # for i in range(len(self.subjects)):
                #     ind = (data['subj_idx'] == i)
                #     for measure in [M, 'pupil_d']:
                #         data.ix[ind,measure] = (data.ix[ind,measure] - data.ix[ind,measure].mean()) / data.ix[ind,measure].std()
                
                # MODEL 0
                a = np.zeros(nr_subjects)
                a2 = np.zeros(nr_subjects)
                for i, s in enumerate(self.subjects):
                    # robjects.globalenv["data_s"] = com.convert_to_r_dataframe(data[(data['subj_idx'] == i)], strings_as_factors=True)
                    # # robjects.globalenv["model"] = "{} ~ a*{} + cov1*stimulus".format(M, X)
                    # robjects.globalenv["model"] = "{} ~ a*{}".format(M, X)
                    # # robjects.globalenv["model"] = "choice_a ~ cp*{} + b*{} + cov2*stimulus\n{} ~ a*{} + cov3*stimulus".format(X,M,M,X)
                    # res = robjects.r("fit = sem(model, data=data_s, estimator='ML',)")
                    # coefs = robjects.r('coef(fit)')
                    # a[i] = coefs[0]

                    # formula = "{} ~ {}".format(M, X)
                    # model = sm.ols(formula=formula, data=data[(data['subj_idx'] == i)])
                    # fitted = model.fit()
                    # a[i] = fitted.rsquared
                    
                    a[i] = sp.stats.pearsonr(data[(data['subj_idx'] == i)][M], data[(data['subj_idx'] == i)][X])[0]
                choice.append(a)
                
                # behavior = myfuncs.behavior(data[(np.concatenate(self.pupil_l_ind)+np.concatenate(self.pupil_h_ind))])
                # df_h = behavior.choice_fractions(split_by='pupil_high', split_target=1)
                # df_l = behavior.choice_fractions(split_by='pupil_high', split_target=0)
                # delta = np.array(df_h.c_a_1) - np.array(df_l.c_a_0)
                
                # params = pd.read_csv(os.path.join(self.base_dir, 'DDM_params.csv'))
                # params = params[np.array(params['subject'] >= 22)]
                # delta = np.array(params['dc'][params['pupil']==1]) - np.array(params['dc'][params['pupil']==0])
                #
                #
                # fig = plt.figure(figsize=(2,2))
                # ax = fig.add_subplot(111)
                # myfuncs.correlation_plot(a, delta, line=True, ax=ax)
                # plt.ylabel('Change in drift criterion')
                # plt.xlabel('Correlation TPR - {}'.format(M))
                # fig.savefig(os.path.join(self.figure_folder, 'V123', 'cp', 'across_subjects_cor_{}_{}.pdf'.format(X,M)))
                
                
                # shell()
                
                
                
            choice = pd.DataFrame(np.vstack(choice).T, columns=roi_names)
            dft = choice.stack().reset_index()
            dft.columns = ['subject', 'measure', 'value']
            fig = plt.figure(figsize=( (1+(len(np.unique(dft.measure))*0.15)),2.5))
            ax = fig.add_subplot(111)
            sns.barplot(x='measure',  y='value', units='subject', data=dft, ci=None, alpha=0.5, palette=['black'], ax=ax)
            sns.stripplot(x='measure', y='value', data=dft, jitter=False, size=2, edgecolor='black', linewidth=0.25, alpha=1, palette=['black'], ax=ax)
            for i, p in enumerate(ax.patches):
                values = dft[dft['measure'] == roi_names[i]].value
                if 'bic' in np.unique(dft.measure)[i]:
                    p1 = round(np.mean(values), 1)
                else:
                    # p1 = myfuncs.permutationTest(values, np.zeros(values.shape[0]), paired=True)[1]
                    p1 = sp.stats.wilcoxon(values, np.zeros(values.shape[0]),)[1]
                plt.text(p.get_x(), -0.0, str(round(p1,3)), size=5, rotation=90, color='r')
            sns.despine(offset=10, trim=True)
            plt.xticks(rotation=90)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, 'V123', 'cp', 'a_path_{}_{}.pdf'.format(X,'all')))
            
            
            
            
            stim = pd.DataFrame(np.vstack(stim).T, columns=roi_names)
            dft = stim.stack().reset_index()
            dft.columns = ['subject', 'measure', 'value']
            fig = plt.figure(figsize=( (1+(len(np.unique(dft.measure))*0.15)),2.5))
            ax = fig.add_subplot(111)
            sns.barplot(x='measure',  y='value', units='subject', data=dft, ci=None, alpha=0.5, palette=['black'], ax=ax)
            sns.stripplot(x='measure', y='value', data=dft, jitter=False, size=2, edgecolor='black', linewidth=0.25, alpha=1, palette=['black'], ax=ax)
            for i, p in enumerate(ax.patches):
                values = dft[dft['measure'] == roi_names[i]].value
                if 'bic' in np.unique(dft.measure)[i]:
                    p1 = round(np.mean(values), 1)
                else:
                    # p1 = myfuncs.permutationTest(values, np.zeros(values.shape[0]), paired=True)[1]
                    p1 = sp.stats.wilcoxon(values, np.zeros(values.shape[0]),)[1]
                plt.text(p.get_x(), -0.0, str(round(p1,3)), size=5, rotation=90, color='r')
            sns.despine(offset=10, trim=True)
            plt.xticks(rotation=90)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, 'V123', 'cp', 'stim_{}_{}.pdf'.format(X,'all')))
            
            
            for r in roi_names:
                print myfuncs.permutationTest(choice[r], np.zeros(len(self.subjects)), paired=True) 
            
            print myfuncs.permutationTest(rs_across[1], rs_across[0], paired=True) 
            print myfuncs.permutationTest(rs_across[2], rs_across[0], paired=True) 
            print myfuncs.permutationTest(rs_across[3], rs_across[0], paired=True) 
            
    def CHOICE_SIGNALS_to_choice(self):
        
        data_type='clean_4th_ventricle'
        time_locked = 'stim_locked'
        # self.make_pupil_BOLD_dataframe(data_type=data_type, time_locked=time_locked, regress_iti=False, regress_rt=True, regress_stimulus=False)
        
        data_frame = pd.read_csv(os.path.join(self.data_folder, 'event_related_average_wb', 'MNI_dataframe_1.csv'))
        dfs = [data_frame[data_frame.subject == s] for s in self.subjects]
        data_frame = pd.concat(dfs)
        
        rois = ['lr_M1_lat', 'combined_choice_parietal_all', 'combined_choice_parietal1a', 'combined_choice_parietal2']
        roi_names = ['M1', 'comb_all', 'comb1a', 'comb2',]
        
        # rois = ['sl_PCeS_IFG_info']
        # roi_names = ['PostCeS/IFG',]
        
        stim = []
        choice1 = []
        choice2 = []
        rs_across = []
        for M in rois:
            
            d = {
                'pupil_d' : pd.Series(np.array(pd.concat(self.pupil_data)['pupil_d']), dtype=float),
                M : pd.Series(np.array(data_frame[M], dtype=float)),
                'choice_a' : pd.Series(np.array(pd.concat(self.pupil_data)['yes']), dtype=int),
                'subj_idx' : pd.Series(self.subj_idx),
                'stimulus' : pd.Series(np.array(pd.concat(self.pupil_data)['present']), dtype=int),
                'session' : pd.Series(np.concatenate(self.session)),
                'run' : pd.Series(np.concatenate(self.run)),
                'rt' : pd.Series(np.concatenate(self.rt)),
                }
            data = pd.DataFrame(d)
            
            a = np.zeros(len(self.subjects))
            b = np.zeros(len(self.subjects))
            for i, s in enumerate(self.subjects):
                a[i] = sp.stats.pearsonr(data[(data['subj_idx'] == i)][M], data[(data['subj_idx'] == i)]['stimulus'])[0]
                b[i] = sp.stats.pearsonr(data[(data['subj_idx'] == i)][M], data[(data['subj_idx'] == i)]['choice_a'])[0]
            stim.append(a)
            choice1.append(b)
            
            # # initialize behavior operator:
            # self.behavior = myfuncs.behavior(data)
            # for y in ['rt', 'stimulus',]:
            #     # model_comp = 'bayes'
            #     model_comp = 'seq'
            #     bin_by = M
            #     bins = 5
            #     fig, rs = self.behavior.SDT_correlation(bin_by=bin_by, n_bins=bins, y1=y, model_comp=model_comp)
            #     fig.savefig(os.path.join(self.figure_folder, 'V123', 'cp', 'ctx_to_choicevar_{}_{}.pdf'.format(y,M)))
            #     rs_across.append(rs)
            
            
            # take out stimulus:
            for i in range(len(self.subjects)):
                ind = (data['subj_idx'] == i)
                # for measure in ['pupil_d', M]:
                for measure in [M]:
                    data.ix[ind,measure] = myfuncs.lin_regress_resid( np.array(data.ix[ind,measure]), [np.array(data.ix[ind,'stimulus'], dtype=int)]) + data.ix[ind,measure].mean()
            
            b = np.zeros(len(self.subjects))
            for i, s in enumerate(self.subjects):
                b[i] = sp.stats.pearsonr(data[(data['subj_idx'] == i)][M], data[(data['subj_idx'] == i)]['choice_a'])[0]
            choice2.append(b)
            
            # # initialize behavior operator:
            # self.behavior = myfuncs.behavior(data)
            # for y in ['choice_a',]:
            #     # model_comp = 'bayes'
            #     model_comp = 'seq'
            #     bin_by = M
            #     bins = 5
            #     fig, rs = self.behavior.SDT_correlation(bin_by=bin_by, n_bins=bins, y1=y, model_comp=model_comp)
            #     fig.savefig(os.path.join(self.figure_folder, 'V123', 'cp', 'ctx_to_choicevar_{}_{}.pdf'.format(y,M)))
         
        for measure, title in zip([stim, choice1, choice2], ['stim', 'choice1', 'choice2']):
        
            dft = pd.DataFrame(np.vstack(measure).T, columns=roi_names)
            dft = dft.stack().reset_index()
            dft.columns = ['subject', 'measure', 'value']
            fig = plt.figure(figsize=( (1+(len(np.unique(dft.measure))*0.15)),2.5))
            ax = fig.add_subplot(111)
            sns.barplot(x='measure',  y='value', units='subject', data=dft, ci=None, alpha=0.5, palette=['black'], ax=ax)
            sns.stripplot(x='measure', y='value', data=dft, jitter=False, size=2, edgecolor='black', linewidth=0.25, alpha=1, palette=['black'], ax=ax)
            for i, p in enumerate(ax.patches):
                values = dft[dft['measure'] == roi_names[i]].value
                if 'bic' in np.unique(dft.measure)[i]:
                    p1 = round(np.mean(values), 1)
                else:
                    p1 = myfuncs.permutationTest(values, np.zeros(values.shape[0]), paired=True)[1]
                    # p1 = sp.stats.wilcoxon(values, np.zeros(values.shape[0]),)[1]
                plt.text(p.get_x(), -0.0, str(round(p1,3)), size=5, rotation=90, color='r')
            plt.ylim(0,0.75)
            sns.despine(offset=10, trim=True)
            plt.xticks(rotation=90)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, 'V123', 'cp', 'stim_{}_{}.pdf'.format(title, 'all')))
            
    def CHOICE_SIGNALS_mediation_analysis_(self):
        
        data_type='clean_4th_ventricle'
        time_locked = 'stim_locked'
        self.make_pupil_BOLD_dataframe(data_type=data_type, time_locked=time_locked, regress_iti=False, regress_rt=True, regress_stimulus=False)
        
        data_brainstem = self.data_frame.copy()
        dfs_brainstem = [data_brainstem[data_brainstem.subject == s] for s in self.subjects]
        
        # add linear combinations:
        for i in range(len(self.subjects)):
            d = {'Y' : pd.Series(self.pupil_data[i]['pupil_d']),
                'X1' : pd.Series(dfs_brainstem[i]['LC_JW_d']),
                'X2' : pd.Series(dfs_brainstem[i]['mean_SN_d']),
                'X3' : pd.Series(dfs_brainstem[i]['mean_VTA_d']),
                'X4' : pd.Series(dfs_brainstem[i]['basal_forebrain_123_d']),
                'X5' : pd.Series(dfs_brainstem[i]['basal_forebrain_4_d']),
                'X6' : pd.Series(dfs_brainstem[i]['inf_col_jw_d']),
                'X7' : pd.Series(dfs_brainstem[i]['sup_col_jw_d']),
                }
            data = pd.DataFrame(d)
            formula = 'Y ~ X1 + X2 + X3 + X4 + X5 + X6 + X7'
            model = sm.ols(formula=formula, data=data)
            fitted = model.fit()
            # values = np.array(fitted.fittedvalues)
            values = np.array(fitted.fitted_values)
            dfs_brainstem[i]['BS_COCKTAIL'] = values
        
        data_frame = pd.read_csv(os.path.join(self.data_folder, 'event_related_average_wb', 'MNI_dataframe_1.csv'))
        # data_frame = data_frame[~np.concatenate(self.omissions_ori2)[~np.concatenate(self.omissions_ori)]]
        dfs = [data_frame[data_frame.subject == s] for s in self.subjects]
        rois = ['sl_IPL_info', 'sl_SPL1_info', 'lr_aIPS_lat', 'sl_SPL2_info', 'sl_pIns_info', 'lr_PCeS_lat', 'lr_M1_lat', 'sl_PCeS_IFG_info', 'sl_MFG_info', 'combined_choice_parietal1a']
        roi_names = ['IPL', 'SPL', 'aIPS1', 'aIPS2', 'pIns', 'IPS/PostCeS', 'M1', 'PreCeS/IFG', 'MFG', 'Combined']
        
        d = {
            'pupil_d' : pd.Series(np.array(pd.concat(self.pupil_data)['pupil_d']), dtype=float),
            'brainstem_combined' : pd.Series(np.array(pd.concat(dfs_brainstem)['BS_COCKTAIL']), dtype=float),
            'combined_choice_parietal1a' : pd.Series(np.array(data_frame['combined_choice_parietal1a']), dtype=float),
            'combined_choice_parietal1b' : pd.Series(np.array(data_frame['combined_choice_parietal1b']), dtype=float),
            'combined_choice_parietal2' : pd.Series(np.array(data_frame['combined_choice_parietal2']), dtype=float),
            'Y' : pd.Series(np.array(pd.concat(self.pupil_data)['yes']), dtype=int),
            's' : pd.Series(np.array(pd.concat(self.pupil_data)['subject']),),
            'stim' : pd.Series(np.array(pd.concat(self.pupil_data)['present']), dtype=int),
            }
        data = pd.DataFrame(d)
        
        # z-score:
        for i in range(len(self.subjects)):
            ind = (data['s'] == self.subjects[i])
            for roi in ['pupil_d', 'brainstem_combined', 'combined_choice_parietal1a', 'combined_choice_parietal1b', 'combined_choice_parietal2']:
                data.ix[ind,roi] = (data.ix[ind,roi] - data.ix[ind,roi].mean()) / data.ix[ind,roi].std()
        
        # shell()
        
        # fit the path models:
        ######################
        import rpy2.robjects as robjects
        import pandas.rpy.common as com
        nr_subjects = len(self.subjects)
        nr_rois = 1
        
        robjects.r('library(lavaan)')
        
        rois = ['combined_choice_parietal1a', 'combined_choice_parietal1b', 'combined_choice_parietal2']
        for X in ['pupil_d', 'brainstem_combined']:
            for M in rois:
                
                # MODEL 0
                c0 = np.zeros((nr_subjects, nr_rois))
                aic0 = np.zeros((nr_subjects, nr_rois))
                bic0 = np.zeros((nr_subjects, nr_rois))
                for i, s in enumerate(self.subjects):
                    for j, roi in enumerate([M]):
                        data['M'] = data[roi].values
                        robjects.globalenv["data_s"] = com.convert_to_r_dataframe(data[(data['s'] == s)], strings_as_factors=True)
                        robjects.globalenv["model"] = "Y ~ c*{} + cov1*stim".format(X)
                        res = robjects.r("fit = sem(model, data=data_s, estimator='ML',)")
                        coefs = robjects.r('coef(fit)')
                        c0[i] = coefs[0]
                        aic0[i] = robjects.r('AIC(fit)')[0]
                        bic0[i] = robjects.r('BIC(fit)')[0]
            
                # MODEL 1
                a1 = np.zeros((nr_subjects, nr_rois))
                b1 = np.zeros((nr_subjects, nr_rois))
                c_p1 = np.zeros((nr_subjects, nr_rois))
                indirect1 = np.zeros((nr_subjects, nr_rois))
                aic1 = np.zeros((nr_subjects, nr_rois))
                bic1 = np.zeros((nr_subjects, nr_rois))
                for i, s in enumerate(self.subjects):
                    for j, roi in enumerate([M]):
                        data['M'] = data[roi].values
                        robjects.globalenv["data_s"] = com.convert_to_r_dataframe(data[(data['s'] == s)], strings_as_factors=True)
                        robjects.globalenv["model"] = "Y ~ cp*{} + b*{} + cov2*stim\n{} ~ a*{} + cov3*stim".format(X,M,M,X)
                        res = robjects.r("fit = sem(model, data=data_s, estimator='ML')")
                        coefs = robjects.r('coef(fit)')
                        c_p1[i,j] = coefs[0]
                        b1[i,j] = coefs[1]
                        a1[i,j] = coefs[3]
                        indirect1[i,j] = a1[i,j] * b1[i,j]
                        aic1[i,j] = robjects.r('AIC(fit)')[0]
                        bic1[i,j] = robjects.r('BIC(fit)')[0]
        
                # MODEL 2
                b2 = np.zeros((nr_subjects, nr_rois))
                c_p2 = np.zeros((nr_subjects, nr_rois))
                aic2 = np.zeros((nr_subjects, nr_rois))
                bic2 = np.zeros((nr_subjects, nr_rois))
                for i, s in enumerate(self.subjects):
                    for j, roi in enumerate([M]):
                        data['M'] = data[roi].values
                        robjects.globalenv["data_s"] = com.convert_to_r_dataframe(data[(data['s'] == s)], strings_as_factors=True)
                        robjects.globalenv["model"] = "Y ~ cp*{} + b*{} + cov2*stim\n{} ~ cov3*stim".format(X,M,M)
                        res = robjects.r("fit = sem(model, data=data_s, estimator='ML')")
                        coefs = robjects.r('coef(fit)')
                        c_p2[i,j] = coefs[0]
                        b2[i,j] = coefs[1]
                        aic2[i,j] = robjects.r('AIC(fit)')[0]
                        bic2[i,j] = robjects.r('BIC(fit)')[0]
        
                # MODEL 3
                a3 = np.zeros((nr_subjects, nr_rois))
                b3 = np.zeros((nr_subjects, nr_rois))
                c_p3 = np.zeros((nr_subjects, nr_rois))
                indirect3 = np.zeros((nr_subjects, nr_rois))
                aic3 = np.zeros((nr_subjects, nr_rois))
                bic3 = np.zeros((nr_subjects, nr_rois))
                for i, s in enumerate(self.subjects):
                    for j, roi in enumerate([M]):
                        data['M'] = data[roi].values
                        robjects.globalenv["data_s"] = com.convert_to_r_dataframe(data[(data['s'] == s)], strings_as_factors=True)
                        robjects.globalenv["model"] = "Y ~ cp*{} + b*{}\n{} ~ a*{} + cov3*stim".format(X,M,M,X)
                        res = robjects.r("fit = sem(model, data=data_s, estimator='ML')")
                        coefs = robjects.r('coef(fit)')
                        c_p3[i,j] = coefs[0]
                        b3[i,j] = coefs[1]
                        a3[i,j] = coefs[2]
                        indirect3[i,j] = a3[i,j] * b3[i,j]
                        aic3[i,j] = robjects.r('AIC(fit)')[0]
                        bic3[i,j] = robjects.r('BIC(fit)')[0]
        
        
                # plot:
                #######
                measures1 = pd.DataFrame(np.hstack([a1, b1, c0, c_p1, indirect1]), columns=['a', 'b', 'c', 'cp', 'indirect',])
                measures2 = pd.DataFrame(np.hstack([bic1, bic2, bic3]), columns=['bic1', 'bic2', 'bic3'])
        
                for j, m in enumerate([measures1, measures2]):
        
                    dft = m.stack().reset_index()
                    dft.columns = ['subject', 'measure', 'value']
                    fig = plt.figure(figsize=( (1+(len(np.unique(dft.measure))*0.15)),2.5))
                    ax = fig.add_subplot(111)
                    sns.barplot(x='measure',  y='value', units='subject', data=dft, ci=None, alpha=0.5, palette=['black'], ax=ax)
                    sns.stripplot(x='measure', y='value', data=dft, jitter=False, size=2, edgecolor='black', linewidth=0.25, alpha=1, palette=['black'], ax=ax)
                    values = np.vstack((dft[dft['measure'] == '{}_1'.format(t)].value, dft[dft['measure'] == '{}_0'.format(t)].value))
                    for i, p in enumerate(ax.patches):
                        values = dft[dft['measure'] == roi_names[i]].value
                        if 'bic' in np.unique(dft.measure)[i]:
                            p1 = round(np.mean(values), 1)
                        else:
                            p1 = myfuncs.permutationTest(values, np.zeros(values.shape[0]), paired=True)[1]
                            # p1 = sp.stats.wilcoxon(values, np.zeros(measure.shape[0]),)[1]
                        plt.text(p.get_x(), 0, str(round(p1,3)), size=5, rotation=90, color='r')
                    sns.despine(offset=10, trim=True)
                    plt.tight_layout()
                    fig.savefig(os.path.join(self.figure_folder, 'V123', 'cp', 'mediation_{}_{}_{}.pdf'.format(X,M,j)))
            
            
    def CHOICE_SIGNALS_behavioural_correlation_old(self):
        
        random_effects = True
        
        # load dataframe:
        data_frame = pd.read_csv(os.path.join(self.data_folder, 'event_related_average_wb', 'MNI_dataframe_1.csv'))
        dfs = [data_frame[data_frame.subject == s] for s in self.subjects]
        
        var_Y = 'behaviour'
        # for var_X in ['V1', 'V2', 'V3', 'V123_center_info', 'aIPS_lat', 'M1_lat', 'aIPS', 'M1',]:
        # for var_X in ['aIPS_lat', 'M1_lat']:
        for var_X in ['lr_aIPS_lat', 'lr_insula2_lat']:
        # for var_X in ['lr_IPS_lat',]:
            bin_by = 'pupil_d'
            # bin_by = var_X
            
            R_varY = []
            R_varY2 = []
            
            # for n_bins in np.arange(5,11):
            for n_bins in [7]:
                # n_bins = 8
                
                bins = []
                for i in range(len(self.subjects)):
                    
                    # # regress RT all runs together:
                    # # -----------------------------
                    # inds_s = np.zeros((len(self.pupil_data[i]), n_bins), dtype=bool)
                    # trial_nr = 0
                    # present = np.array(self.pupil_data[i]['present'], dtype=int)
                    # rt = np.array(self.pupil_data[i]['rt'])
                    # if bin_by == 'pupil_d':
                    #     bin_measure = np.array(self.pupil_data[i][bin_by])
                    #     bin_measure = myfuncs.lin_regress_resid(bin_measure, [rt])
                    #     # for s in np.unique(self.session[i]):
                    #     #     bin_measure[self.session[i]==s] = myfuncs.lin_regress_resid(bin_measure[self.session[i]==s], [rt[self.session[i]==s]])
                    # else:
                    #     bin_measure = np.array(dfs[i][var_X])
                    # for r in np.array(np.unique(self.pupil_data[i]['run']), dtype=int):
                    #     nr_trials_in_run = len(bin_measure[np.array(self.pupil_data[i].run == r)])
                    #     inds = np.array_split(np.argsort(bin_measure[np.array(self.pupil_data[i].run == r)]), n_bins)
                    #     for b in range(n_bins):
                    #         ind = np.zeros(len(bin_measure[np.array(self.pupil_data[i].run == r)]), dtype=bool)
                    #         ind[inds[b]] = True
                    #         inds_s[trial_nr:trial_nr+nr_trials_in_run, b] = ind
                    #     trial_nr += nr_trials_in_run
                    # bins.append(inds_s)
                    
                    # regress RT per session:
                    # -------------------
                    inds_s = np.zeros((len(self.pupil_data[i]), n_bins), dtype=bool)
                    trial_nr = 0
                    for s in np.array(np.unique(self.pupil_data[i]['session']), dtype=int):
                        rt = np.array(self.pupil_data[i]['rt'])[np.array(self.pupil_data[i].session == s)]
                        present = np.array(self.pupil_data[i]['present'], dtype=int)[np.array(self.pupil_data[i].session == s)]
                        if bin_by == 'pupil_d':
                            bin_measure = np.array(self.pupil_data[i][bin_by])[np.array(self.pupil_data[i].session == s)]
                            bin_measure = myfuncs.lin_regress_resid(bin_measure, [rt])
                        else:
                            bin_measure = np.array(dfs[i][var_X])[np.array(self.pupil_data[i].session == s)]
                        nr_trials_in_run = len(bin_measure)
                        inds = np.array_split(np.argsort(bin_measure), n_bins)
                        for b in range(n_bins):
                            ind = np.zeros(len(bin_measure), dtype=bool)
                            ind[inds[b]] = True
                            inds_s[trial_nr:trial_nr+nr_trials_in_run, b] = ind
                        trial_nr += nr_trials_in_run
                    bins.append(inds_s)
                    
                    # # regress RT per run:
                    # # -------------------
                    # inds_s = np.zeros((len(self.pupil_data[i]), n_bins), dtype=bool)
                    # trial_nr = 0
                    # for r in np.array(np.unique(self.pupil_data[i]['run']), dtype=int):
                    #     rt = np.array(self.pupil_data[i]['rt'])[np.array(self.pupil_data[i].run == r)]
                    #     present = np.array(self.pupil_data[i]['present'], dtype=int)[np.array(self.pupil_data[i].run == r)]
                    #     if bin_by == 'pupil_d':
                    #         bin_measure = np.array(self.pupil_data[i][bin_by])[np.array(self.pupil_data[i].run == r)]
                    #         bin_measure = myfuncs.lin_regress_resid(bin_measure, [rt])
                    #     else:
                    #         bin_measure = np.array(dfs[i][var_X])[np.array(self.pupil_data[i].run == r)]
                    #     nr_trials_in_run = len(bin_measure)
                    #     inds = np.array_split(np.argsort(bin_measure), n_bins)
                    #     for b in range(n_bins):
                    #         ind = np.zeros(len(bin_measure), dtype=bool)
                    #         ind[inds[b]] = True
                    #         inds_s[trial_nr:trial_nr+nr_trials_in_run, b] = ind
                    #     trial_nr += nr_trials_in_run
                    # bins.append(inds_s)
                    
                    # # bin across all runs:
                    # # --------------------
                    # inds_s = np.zeros((len(self.pupil_data[i]), n_bins), dtype=bool)
                    # iti = np.array(self.pupil_data[i]['iti'])
                    # iti[np.isnan(iti)] = bn.nanmean(iti)
                    # present = np.array(self.pupil_data[i]['present'])
                    # rt = np.array(self.pupil_data[i]['rt'])
                    # if bin_by == 'pupil_d':
                    #     bin_measure = np.array(self.pupil_data[i][bin_by])
                    #     bin_measure = myfuncs.lin_regress_resid(bin_measure, [rt])
                    #     # bin_measure = rt
                    # else:
                    #     bin_measure = np.array(dfs[i][var_X])
                    # inds = np.array_split(np.argsort(bin_measure), n_bins)
                    # for b in range(n_bins):
                    #     ind = np.zeros(len(bin_measure), dtype=bool)
                    #     ind[inds[b]] = True
                    #     inds_s[:, b] = ind
                    # bins.append(inds_s)
                
                # compute criterion values:
                varX = np.zeros((len(self.subjects), n_bins))
                varY = np.zeros((len(self.subjects), n_bins))
                varY2 = np.zeros((len(self.subjects), n_bins))
                for i in range(len(self.subjects)):
                    for b in range(n_bins):
                        data = self.pupil_data[i][bins[i][:,b]]
                        varX[i,b] = np.array(dfs[i][var_X][bins[i][:,b]]).mean()
                        if var_Y == 'behaviour':
                            d, c = myfuncs.SDT_measures(target=np.array(data.present), hit=np.array(data.hit), fa=np.array(data.fa))
                            # c = np.array(data.yes).sum() / float(len(np.array(data.yes)))
                            varY[i,b] = d
                            varY2[i,b] = c
                        else:
                            varY[i,b] = np.array(dfs[i][var_Y][bins[i][:,b]]).mean()
                
                # plot mean traces as a function of pupil:
                fig = plt.figure(figsize=(6,2))
                x = np.arange(n_bins)+1
                
                if not bin_by == var_X:
                    ax = fig.add_subplot(131)
                    ax.errorbar(x-0.1, varY.mean(axis=0), yerr=sp.stats.sem(varY, axis=0), fmt='o', markersize=6, color='purple', capsize=0, elinewidth=0.5, markeredgecolor='w', markeredgewidth=0.5)
                    ax.set_xticks(x)
                    # ax.set_ylim((0, 2))
                    ax.set_ylabel(var_Y, color='purple')
                    ax.set_xlabel('Bin')
                    regression_line = []
                    r1 = np.zeros(len(self.subjects))
                    for i in range(len(self.subjects)):
                        y = varY[i,:]
                        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x,y)
                        (m,b) = sp.polyfit(x,y,1)
                        regression_line.append(sp.polyval([m,b],x))
                        r1[i] = r_value
                    p1 = myfuncs.permutationTest(r1, np.zeros(len(r1)), paired=True)[1]
                    regression_line = np.vstack(regression_line)
                    ax.plot(x, regression_line.mean(axis=0), color='purple', alpha=1)
                    ax.fill_between(x, regression_line.mean(axis=0)-sp.stats.sem(regression_line, axis=0), regression_line.mean(axis=0)+sp.stats.sem(regression_line, axis=0), color='purple', alpha=0.2)
                    for tl in ax.get_yticklabels():
                        tl.set_color('purple')
            
                    if var_Y == 'behaviour':
                        ax2 = ax.twinx()
                        ax2.errorbar(x+0.1, varY2.mean(axis=0), yerr=sp.stats.sem(varY2, axis=0), fmt='o', markersize=6, color='blue', capsize=0,  elinewidth=0.5, markeredgecolor='w', markeredgewidth=0.5)
                        # ax2.set_ylim((-0.2, 0.4))
                        ax2.set_ylabel(var_Y, color='blue')
                        regression_line = []
                        r2 = np.zeros(len(self.subjects))
                        for i in range(len(self.subjects)):
                            y = varY2[i,:]
                            slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x,y)
                            (m,b) = sp.polyfit(x,y,1)
                            regression_line.append(sp.polyval([m,b],x))
                            r2[i] = r_value
                        p2 = myfuncs.permutationTest(r2, np.zeros(len(r2)), paired=True)[1]
                        regression_line = np.vstack(regression_line)
                        ax2.plot(x, regression_line.mean(axis=0), color='blue', alpha=1)
                        ax2.fill_between(x, regression_line.mean(axis=0)-sp.stats.sem(regression_line, axis=0), regression_line.mean(axis=0)+sp.stats.sem(regression_line, axis=0), color='blue', alpha=0.2)
                        for tl in ax2.get_yticklabels():
                            tl.set_color('blue')
                    if var_Y == 'behaviour':
                        ax.set_title('r = {}, p = {}\nr = {}, p = {}'.format(round(r1.mean(), 3), round(p1, 3), round(r2.mean(), 3), round(p2, 3)))
                    else:
                        ax.set_title('r = {}, p = {}'.format(round(r1.mean(), 3), round(p1, 3),))
                
                ax = fig.add_subplot(132)
                # ax.plot(x, V123.mean(axis=0), ls='o', color='k', ms=10 )
                ax.errorbar(x, varX.mean(axis=0), yerr=sp.stats.sem(varX, axis=0), fmt='o', markersize=6, color='k', capsize=0, elinewidth=0.5, markeredgecolor='w', markeredgewidth=0.5)
                ax.set_xticks(x)
                ax.set_ylabel('{}'.format(var_X))
                ax.set_xlabel('Bin')
                if random_effects:
                    regression_line = []
                    r = np.zeros(len(self.subjects))
                    for i in range(len(self.subjects)):
                        y = varX[i,:]
                        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x,y)
                        (m,b) = sp.polyfit(x,y,1)
                        regression_line.append(sp.polyval([m,b],x))
                        r[i] = r_value
                    p = myfuncs.permutationTest(r, np.zeros(len(r)), paired=True)[1]
                    regression_line = np.vstack(regression_line)
                    ax.plot(x, regression_line.mean(axis=0), color='k', alpha=1)
                    ax.fill_between(x, regression_line.mean(axis=0)-sp.stats.sem(regression_line, axis=0), regression_line.mean(axis=0)+sp.stats.sem(regression_line, axis=0), color='k', alpha=0.2)
                else:
                    y = varX[:,:].mean(axis=0)
                    slope, intercept, r_value, p, std_err = sp.stats.linregress(x,y)
                    (m,b) = sp.polyfit(x,y,1)
                    regression_line = sp.polyval([m,b],x)
                    r = r_value
                    ax.plot(x, regression_line, color='k', alpha=1)
                ax.set_title('r = {}\np = {}'.format(round(r.mean(), 3), round(p, 3)))
                    
                
                ax = fig.add_subplot(133)
                ax.errorbar(varX.mean(axis=0), varY.mean(axis=0), xerr=sp.stats.sem(varX, axis=0), yerr=sp.stats.sem(varY, axis=0), fmt='o', markersize=6, color='purple', capsize=0, elinewidth=0.5, markeredgecolor='w', markeredgewidth=0.5)
                # ax.set_xticks(x)
                # ax.set_ylim((0, 2))
                ax.set_ylabel(var_Y, color='purple')
                ax.set_xlabel('{}'.format(var_X))
                
                if random_effects:
                    regression_line = []
                    r1 = np.zeros(len(self.subjects))
                    for i in range(len(self.subjects)):
                        x = varX[i,:]
                        y = varY[i,:]
                        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x,y)
                        (m,b) = sp.polyfit(x,y,1)
                        x_line = np.linspace(min(varX[:,:].mean(axis=0))-((max(varX[:,:].mean(axis=0))-min(varX[:,:].mean(axis=0))) / 4), max(varX[:,:].mean(axis=0))+((max(varX[:,:].mean(axis=0))-min(varX[:,:].mean(axis=0))) / 4), 50)
                        regression_line.append(sp.polyval([m,b],x_line))
                        r1[i] = r_value
                    p1 = myfuncs.permutationTest(r1, np.zeros(len(r1)), paired=True)[1]
                    R_varY.append(r1)
                    regression_line = np.vstack(regression_line)
                    ax.plot(x_line, regression_line.mean(axis=0), color='purple', alpha=1)
                    ax.fill_between(x_line, regression_line.mean(axis=0)-sp.stats.sem(regression_line, axis=0), regression_line.mean(axis=0)+sp.stats.sem(regression_line, axis=0), color='purple', alpha=0.2)
                else:
                    x = varX[:,:].mean(axis=0)
                    y = varY[:,:].mean(axis=0)
                    slope, intercept, r_value, p1, std_err = sp.stats.linregress(x,y)
                    (m,b) = sp.polyfit(x,y,1)
                    x_line = np.linspace(min(varX[:,:].mean(axis=0))-((max(varX[:,:].mean(axis=0))-min(varX[:,:].mean(axis=0))) / 1), max(varX[:,:].mean(axis=0))+((max(varX[:,:].mean(axis=0))-min(varX[:,:].mean(axis=0))) / 1), 50)
                    regression_line = sp.polyval([m,b],x_line)
                    r1 = r_value
                    ax.plot(x_line, regression_line, color='purple', alpha=1)
                for tl in ax.get_yticklabels():
                    tl.set_color('purple')
            
                if var_Y == 'behaviour':
                    ax2 = ax.twinx()
                    ax2.errorbar(varX.mean(axis=0), varY2.mean(axis=0), xerr=sp.stats.sem(varX, axis=0), yerr=sp.stats.sem(varY2, axis=0), fmt='o', markersize=6, color='blue', capsize=0,  elinewidth=0.5, markeredgecolor='w', markeredgewidth=0.5)
                    # ax2.set_ylim((-0.2, 0.4))
                    ax2.set_ylabel(var_Y, color='blue')
                    
                    if random_effects:
                        regression_line = []
                        r2 = np.zeros(len(self.subjects))
                        for i in range(len(self.subjects)):
                            x = varX[i,:]
                            y = varY2[i,:]
                            slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x,y)
                            (m,b) = sp.polyfit(x,y,1)
                            x_line = np.linspace(min(varX[:,:].mean(axis=0))-((max(varX[:,:].mean(axis=0))-min(varX[:,:].mean(axis=0))) / 4), max(varX[:,:].mean(axis=0))+((max(varX[:,:].mean(axis=0))-min(varX[:,:].mean(axis=0))) / 4), 50)
                            regression_line.append(sp.polyval([m,b],x_line))
                            r2[i] = r_value
                        p2 = myfuncs.permutationTest(r2, np.zeros(len(r1)), paired=True)[1]
                        R_varY2.append(r2)
                        regression_line = np.vstack(regression_line)
                        ax2.plot(x_line, regression_line.mean(axis=0), color='blue', alpha=1)
                        ax2.fill_between(x_line, regression_line.mean(axis=0)-sp.stats.sem(regression_line, axis=0), regression_line.mean(axis=0)+sp.stats.sem(regression_line, axis=0), color='blue', alpha=0.2)
                        # shell()
                    else:
                        x = varX[:,:].mean(axis=0)
                        y = varY2[:,:].mean(axis=0)
                        slope, intercept, r_value, p2, std_err = sp.stats.linregress(x,y)
                        (m,b) = sp.polyfit(x,y,1)
                        x_line = np.linspace(min(varX[:,:].mean(axis=0))-((max(varX[:,:].mean(axis=0))-min(varX[:,:].mean(axis=0))) / 1), max(varX[:,:].mean(axis=0))+((max(varX[:,:].mean(axis=0))-min(varX[:,:].mean(axis=0))) / 1), 50)
                        regression_line = sp.polyval([m,b],x_line)
                        r2 = r_value
                        ax2.plot(x_line, regression_line, color='blue', alpha=1)
                    
                    for tl in ax2.get_yticklabels():
                        tl.set_color('blue')
                if var_Y == 'behaviour':
                    ax.set_title('r = {}, p = {}\nr = {}, p = {}'.format(round(r1.mean(), 3), round(p1, 3), round(r2.mean(), 3), round(p2, 3)))
                else:
                    ax.set_title('r = {}, p = {}'.format(round(r1.mean(), 3), round(p1, 3),))
            
                sns.despine(offset=10, trim=True, right=False)
                plt.tight_layout()
                fig.savefig(os.path.join(self.figure_folder, 'V123', 'cp', 'SDT_pupil_bins_{}_{}_{}_{}_{}.pdf'.format(n_bins, var_X, var_Y, bin_by, self.split_by)))
                
            # for measure, title in zip([R_dprime, R_criterion], ['dprime', 'criterion']):
            #     MEANS = [r.mean() for r in measure]
            #     SEMS = [sp.stats.sem(r) for r in measure]
            #     ps = [myfuncs.permutationTest(measure[i], np.zeros(len(measure[i])), paired=True, nrand=10000)[1] for i in range(len(measure))]
            #     locs = np.arange(3,11)
            #     bar_width = 0.90
            #     fig = plt.figure(figsize=(2,2))
            #     my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}
            #     for i in range(len(MEANS)):
            #         plt.bar(locs[i], MEANS[i], width = bar_width, yerr = SEMS[i], color = 'k', alpha = 1, align = 'center', edgecolor = 'k', linewidth = 0, capsize = 0, ecolor = 'k', error_kw={'elinewidth':0.5,})
            #
            #     for i, p in enumerate(ps):
            #         if p < 0.05:
            #             # plt.text(s='{}'.format(round(p, 3)), x=locs[i], y=gca().get_ylim()[1]-((gca().get_ylim()[1] - gca().get_ylim()[0]) / 10.0), size=5, horizontalalignment='center')
            #             plt.text(s='{}'.format('*'), x=locs[i], y=gca().get_ylim()[1]-((gca().get_ylim()[1] - gca().get_ylim()[0]) / 10.0), size=5, horizontalalignment='center')
            #     plt.xticks(locs)
            #     plt.ylabel('Correlation\nV123 - {}'.format(title))
            #     # plt.ylabel("Correlation\nV123 - d'")
            #     plt.xlabel('Number of bins')
            #     sns.despine(offset=10, trim=True)
            #     plt.tight_layout()
            #     fig.savefig(os.path.join(self.figure_folder, 'V123', 'snr', 'correlations_{}_{}_{}.pdf'.format(area, title, self.split_by)))



    #
    # def WHOLEBRAIN_stim_TPR_interaction(self, data_type='clean_MNI'):
    #
    #     session = [self.runList[i] for i in self.conditionDict['task']][0].session
    #     nr_runs = len([self.runList[i] for i in self.conditionDict['task']])
    #     time_locked = 'stim_locked'
    #     if session == 2:
    #
    #         self.load_concatenated_data(data_type=data_type, bold=False)
    #         omissions = np.array(self.pupil_data.omissions, dtype=bool)
    #         self.pupil_data = self.pupil_data[-omissions]
    #
    #         brain_mask = nib.load('/home/shared/UvA/Niels_UvA/mni_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz')
    #         mask = np.array(brain_mask.get_data(), dtype=bool)
    #
    #         scalars_d = np.array(nib.load(os.path.join(self.project.base_dir, 'across', 'event_related_average_wb', 'scalars_{}_{}_{}_d.nii.gz'.format(data_type, time_locked, self.subject.initials))).get_data())[mask,:]
    #
    #         # params // pupil split:
    #         present = np.array(self.pupil_data['present'])
    #         pupil_l_ind = []
    #         pupil_h_ind = []
    #         for s in np.array(np.unique(self.pupil_data['session']), dtype=int):
    #             pupil = np.array(self.pupil_data['pupil_d'])[np.array(self.pupil_data.session) == s]
    #             rt = np.array(self.pupil_data['rt'])[np.array(self.pupil_data.session) == s]
    #             pupil = myfuncs.lin_regress_resid(pupil, [rt]) + pupil.mean()
    #             pupil_l_ind.append( pupil <= np.percentile(pupil, 40) )
    #             pupil_h_ind.append( pupil >= np.percentile(pupil, 60) )
    #         pupil_l_ind = np.concatenate(pupil_l_ind)
    #         pupil_h_ind = np.concatenate(pupil_h_ind)
    #         pupil_rest_ind = -(pupil_h_ind + pupil_l_ind)
    #
    #         # interaction stimulus vs. TPR:
    #         # ----------------------------
    #
    #         interactions = np.zeros(scalars_d.shape[0])
    #         for v in range(scalars_d.shape[0]):
    #             BOLD = scalars_d[v,:]
    #             BOLD = (BOLD-BOLD.mean()) / BOLD.std()
    #
    #             measure_present_h = BOLD[present*pupil_h_ind].mean()
    #             measure_present_l = BOLD[present*pupil_l_ind].mean()
    #             measure_absent_h = BOLD[~present*pupil_h_ind].mean()
    #             measure_absent_l = BOLD[~present*pupil_l_ind].mean()
    #
    #
    #
    #
    #
    #         # save:
    #         results = np.zeros((mask.shape[0],mask.shape[1],mask.shape[2]))
    #         results[mask] = corrs_stim
    #         res_nii_file = nib.Nifti1Image(results, affine=brain_mask.get_affine(), header=brain_mask.get_header())
    #         res_nii_file.set_data_dtype(np.float32)
    #         nib.save(res_nii_file, os.path.join(self.project.base_dir, 'across', 'correlation', 'whole_brain_{}_{}_{}_{}_BOLD_present_s{}.nii.gz'.format('mean', data_type, time_locked, self.subject.initials, session)))
    #
    #         results = np.zeros((mask.shape[0],mask.shape[1],mask.shape[2]))
    #         results[mask] = corrs_choice
    #         res_nii_file = nib.Nifti1Image(results, affine=brain_mask.get_affine(), header=brain_mask.get_header())
    #         res_nii_file.set_data_dtype(np.float32)
    #         nib.save(res_nii_file, os.path.join(self.project.base_dir, 'across', 'correlation', 'whole_brain_{}_{}_{}_{}_BOLD_choice_s{}.nii.gz'.format('mean', data_type, time_locked, self.subject.initials, session)))
    #
    #






            
    def WHOLEBRAIN_event_related_average_prepare(self, data_type, measure='mean'):
        
        brain_mask = nib.load('/home/shared/UvA/Niels_UvA/mni_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz')
        mask = np.array(brain_mask.get_data(), dtype=bool)
        time_locked = 'stim_locked'
        type_response = self.split_by.split('_')[-1]
        
        for i in range(len(self.subjects)):
            print self.subjects[i]
            
            rt = np.array(self.pupil_data[i]['rt'])
            yes = np.array(self.pupil_data[i]['yes'])
            session = np.array(self.pupil_data[i]['session'])
            
            # load:
            scalars = np.array(nib.load(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_{}_{}_{}_{}.nii.gz'.format(data_type, time_locked, self.subjects[i], type_response))).get_data())
            
            # all trials:
            scalars_a = scalars.mean(axis=-1)
            scalars_a_std = sp.stats.sem(scalars, axis=-1)
            res_nii_file = nib.Nifti1Image(np.stack((scalars_a, scalars_a_std,), axis=-1), affine=brain_mask.get_affine(), header=brain_mask.get_header())
            res_nii_file.set_data_dtype(np.float32)
            nib.save(res_nii_file, os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_all_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, self.subjects[i], self.split_by)))
            
            # split by pupil:
            scalars_a = scalars[:,:,:,self.pupil_h_ind[i]].mean(axis=-1)
            scalars_b = scalars[:,:,:,self.pupil_l_ind[i]].mean(axis=-1)
            scalars_a_std = sp.stats.sem(scalars[:,:,:,self.pupil_h_ind[i]], axis=-1)
            scalars_b_std = sp.stats.sem(scalars[:,:,:,self.pupil_l_ind[i]], axis=-1)
            res_nii_file = nib.Nifti1Image(np.stack((scalars_a, scalars_b, scalars_a_std, scalars_b_std), axis=-1), affine=brain_mask.get_affine(), header=brain_mask.get_header())
            res_nii_file.set_data_dtype(np.float32)
            nib.save(res_nii_file, os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_pupil_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, self.subjects[i], self.split_by)))
            
            # split signal presence:
            scalars_a = scalars[:,:,:,self.present[i]].mean(axis=-1)
            scalars_b = scalars[:,:,:,~self.present[i]].mean(axis=-1)
            scalars_a_std = sp.stats.sem(scalars[:,:,:,self.present[i]], axis=-1)
            scalars_b_std = sp.stats.sem(scalars[:,:,:,~self.present[i]], axis=-1)
            res_nii_file = nib.Nifti1Image(np.stack((scalars_a, scalars_b, scalars_a_std, scalars_b_std), axis=-1), affine=brain_mask.get_affine(), header=brain_mask.get_header())
            res_nii_file.set_data_dtype(np.float32)
            nib.save(res_nii_file, os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_signal_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, self.subjects[i], self.split_by)))
            
            # interaction:
            scalars_a = scalars[:,:,:,self.present[i]&self.pupil_h_ind[i]].mean(axis=-1)
            scalars_b = scalars[:,:,:,~self.present[i]&self.pupil_h_ind[i]].mean(axis=-1)
            scalars_c = scalars[:,:,:,self.present[i]&self.pupil_l_ind[i]].mean(axis=-1)
            scalars_d = scalars[:,:,:,~self.present[i]&self.pupil_l_ind[i]].mean(axis=-1)
            scalars_a_std = sp.stats.sem(scalars[:,:,:,self.present[i]&self.pupil_h_ind[i]], axis=-1)
            scalars_b_std = sp.stats.sem(scalars[:,:,:,~self.present[i]&self.pupil_h_ind[i]], axis=-1)
            scalars_c_std = sp.stats.sem(scalars[:,:,:,self.present[i]&self.pupil_l_ind[i]], axis=-1)
            scalars_d_std = sp.stats.sem(scalars[:,:,:,~self.present[i]&self.pupil_l_ind[i]], axis=-1)
            scalars_a = (scalars_a-scalars_b) - (scalars_c-scalars_d)
            scalars_a_std = (scalars_a_std-scalars_b_std) - (scalars_c_std-scalars_d_std) 
            res_nii_file = nib.Nifti1Image(np.stack((scalars_a, scalars_a_std,), axis=-1), affine=brain_mask.get_affine(), header=brain_mask.get_header())
            res_nii_file.set_data_dtype(np.float32)
            nib.save(res_nii_file, os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_interaction_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, self.subjects[i], self.split_by)))
            
            # split by choice:
            scalars_a = scalars[:,:,:,self.yes[i]].mean(axis=-1)
            scalars_b = scalars[:,:,:,~self.yes[i]].mean(axis=-1)
            scalars_a_std = sp.stats.sem(scalars[:,:,:,self.yes[i]], axis=-1)
            scalars_b_std = sp.stats.sem(scalars[:,:,:,~self.yes[i]], axis=-1)
            res_nii_file = nib.Nifti1Image(np.stack((scalars_a, scalars_b, scalars_a_std, scalars_b_std), axis=-1), affine=brain_mask.get_affine(), header=brain_mask.get_header())
            res_nii_file.set_data_dtype(np.float32)
            nib.save(res_nii_file, os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_choice_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, self.subjects[i], self.split_by)))

            # split by right button:
            scalars_a = scalars[:,:,:,self.right[i]].mean(axis=-1)
            scalars_b = scalars[:,:,:,~self.right[i]].mean(axis=-1)
            scalars_a_std = sp.stats.sem(scalars[:,:,:,self.right[i]], axis=-1)
            scalars_b_std = sp.stats.sem(scalars[:,:,:,~self.right[i]], axis=-1)
            res_nii_file = nib.Nifti1Image(np.stack((scalars_a, scalars_b, scalars_a_std, scalars_b_std), axis=-1), affine=brain_mask.get_affine(), header=brain_mask.get_header())
            res_nii_file.set_data_dtype(np.float32)
            nib.save(res_nii_file, os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_right_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, self.subjects[i], self.split_by)))
            
    def WHOLEBRAIN_event_related_average_conditions(self, data_type, measure='mean'):
        
        brain_mask = nib.load('/home/shared/UvA/Niels_UvA/mni_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz')
        mask = np.array(brain_mask.get_data(), dtype=bool)
        time_locked = 'stim_locked'
        type_response = self.split_by.split('_')[-1]
        
        # all trials:
        scalars_all_a = np.stack([np.array(nib.load(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_all_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).get_data())[:,:,:,0] for subj in self.subjects], axis=-1)
        scalars_all_a_std = np.stack([np.array(nib.load(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_all_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).get_data())[:,:,:,1] for subj in self.subjects], axis=-1)
        
        # interaction:
        scalars_interaction_a = np.stack([np.array(nib.load(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_interaction_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).get_data())[:,:,:,0] for subj in self.subjects], axis=-1)
        scalars_interaction_a_std = np.stack([np.array(nib.load(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_interaction_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).get_data())[:,:,:,1] for subj in self.subjects], axis=-1)
        
        # split by pupil:
        scalars_pupil_a = np.stack([np.array(nib.load(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_pupil_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).get_data())[:,:,:,0] for subj in self.subjects], axis=-1)
        scalars_pupil_b = np.stack([np.array(nib.load(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_pupil_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).get_data())[:,:,:,1] for subj in self.subjects], axis=-1)
        scalars_pupil_a_std = np.stack([np.array(nib.load(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_pupil_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).get_data())[:,:,:,2] for subj in self.subjects], axis=-1)
        scalars_pupil_b_std = np.stack([np.array(nib.load(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_pupil_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).get_data())[:,:,:,3] for subj in self.subjects], axis=-1)
        
        # split by signal presence:
        scalars_signal_a = np.stack([np.array(nib.load(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_signal_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).get_data())[:,:,:,0] for subj in self.subjects], axis=-1)
        scalars_signal_b = np.stack([np.array(nib.load(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_signal_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).get_data())[:,:,:,1] for subj in self.subjects], axis=-1)
        scalars_signal_a_std = np.stack([np.array(nib.load(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_signal_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).get_data())[:,:,:,2] for subj in self.subjects], axis=-1)
        scalars_signal_b_std = np.stack([np.array(nib.load(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_signal_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).get_data())[:,:,:,3] for subj in self.subjects], axis=-1)
        
        # split by choice:
        scalars_choice_a = np.stack([np.array(nib.load(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_choice_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).get_data())[:,:,:,0] for subj in self.subjects], axis=-1)
        scalars_choice_b = np.stack([np.array(nib.load(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_choice_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).get_data())[:,:,:,1] for subj in self.subjects], axis=-1)
        scalars_choice_a_std = np.stack([np.array(nib.load(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_choice_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).get_data())[:,:,:,2] for subj in self.subjects], axis=-1)
        scalars_choice_b_std = np.stack([np.array(nib.load(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_choice_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).get_data())[:,:,:,3] for subj in self.subjects], axis=-1)
        
        # split by right button:
        scalars_right_a = np.stack([np.array(nib.load(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_right_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).get_data())[:,:,:,0] for subj in self.subjects], axis=-1)
        scalars_right_b = np.stack([np.array(nib.load(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_right_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).get_data())[:,:,:,1] for subj in self.subjects], axis=-1)
        scalars_right_a_std = np.stack([np.array(nib.load(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_right_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).get_data())[:,:,:,2] for subj in self.subjects], axis=-1)
        scalars_right_b_std = np.stack([np.array(nib.load(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_right_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).get_data())[:,:,:,3] for subj in self.subjects], axis=-1)
        
        import copy
        if measure == 'mean':
            s_all_a = copy.copy(scalars_all_a)
            s_interaction_a = copy.copy(scalars_interaction_a)
            s_pupil_a = copy.copy(scalars_pupil_a)
            s_pupil_b = copy.copy(scalars_pupil_b)
            s_signal_a = copy.copy(scalars_signal_a)
            s_signal_b = copy.copy(scalars_signal_b)
            s_choice_a = copy.copy(scalars_choice_a)
            s_choice_b = copy.copy(scalars_choice_b)
            s_right_a = copy.copy(scalars_right_a)
            s_right_b = copy.copy(scalars_right_b)
        if measure == 'snr':
            s_all_a = copy.copy(scalars_all_a) / copy.copy(scalars_all_a_std)
            s_interaction_a = copy.copy(scalars_interaction_a) / copy.copy(scalars_interaction_a_std)
            s_pupil_a = copy.copy(scalars_pupil_a) / copy.copy(scalars_pupil_a_std)
            s_pupil_b = copy.copy(scalars_pupil_b) / copy.copy(scalars_pupil_b_std)
            s_signal_a = copy.copy(scalars_signal_a) / copy.copy(scalars_signal_a_std)
            s_signal_b = copy.copy(scalars_signal_b) / copy.copy(scalars_signal_b_std)
            s_choice_a = copy.copy(scalars_choice_a) / copy.copy(scalars_choice_a_std)
            s_choice_b = copy.copy(scalars_choice_b) / copy.copy(scalars_choice_b_std)
            s_right_a = copy.copy(scalars_right_a) / copy.copy(scalars_right_a_std)
            s_right_b = copy.copy(scalars_right_b) / copy.copy(scalars_right_b_std)
        if measure == 'std':
            s_all_a = copy.copy(scalars_all_a_std)
            s_interaction_a = copy.copy(scalars_interaction_a_std)
            s_pupil_a = copy.copy(scalars_pupil_a_std)
            s_pupil_b = copy.copy(scalars_pupil_b_std)
            s_signal_a = copy.copy(scalars_signal_a_std)
            s_signal_b = copy.copy(scalars_signal_b_std)
            s_choice_a = copy.copy(scalars_choice_a_std)
            s_choice_b = copy.copy(scalars_choice_b_std)
            s_right_a = copy.copy(scalars_right_a_std)
            s_right_b = copy.copy(scalars_right_b_std)
        
        # all:
        trials_all = s_all_a
        interaction_all = s_interaction_a
        
        # pupil maps:
        pupil_hi = s_pupil_a
        pupil_lo = s_pupil_b
        pupil_contrast = pupil_hi - pupil_lo
        
        # signal maps:
        signal_present = s_signal_a
        signal_absent = s_signal_b
        signal_contrast = signal_present - signal_absent
        
        # choice maps:
        yes = s_choice_a
        no = s_choice_b
        choice_contrast = yes - no
        
        # button maps:
        right = s_right_a
        left = s_right_b
        right_contrast = right - left
        
        # maps:
        # maps = [trials_all, pupil_hi, pupil_lo, pupil_contrast, signal_present, signal_absent, signal_contrast, yes, no, choice_contrast, left, right, right_contrast, interaction_all]
        # titles = ['trials_all', 'pupil_hi', 'pupil_lo', 'pupil_contrast', 'signal_present', 'signal_absent', 'signal_contrast', 'yes', 'no', 'choice_contrast', 'left', 'right', 'right_contrast', 'interaction_all']
        
        # maps = [trials_all, pupil_contrast,]
        # titles = ['trials_all', 'pupil_contrast',]
        
        maps = [pupil_contrast,]
        titles = ['pupil_contrast',]
        
        for m, t in zip(maps, titles):
            for flip in [False, True]:
                if flip:
                    for i in range(len(self.subjects)):
                        if self.subjects[i] in ['sub-03', 'sub-04', 'sub-06', 'sub-08', 'sub-09', 'sub-11', 'sub-13']:
                            pass
                        else:
                            m[:,:,:,i] = m[::-1,:,:,i]
                
                # save:
                res_nii_file = nib.Nifti1Image(m, affine=brain_mask.get_affine(), header=brain_mask.get_header())
                res_nii_file.set_data_dtype(np.float32)
                nib.save(res_nii_file, os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_{}_{}_psc_all.nii.gz'.format(t, data_type, time_locked, measure, self.split_by, int(flip))))
                
                # smooth:
                FWHM = 8.0 # mm
                conversion_factor = 2.35482004503
                inputObject = os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_{}_{}_psc_all.nii.gz'.format(t, data_type, time_locked, measure, self.split_by, int(flip)))
                outputFileName = os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_{}_{}_psc_all.nii.gz'.format(t, data_type, time_locked, measure, self.split_by, int(flip)))
                fO = FSLMathsOperator(inputObject=inputObject)
                fO.configureSmooth(outputFileName=outputFileName, smoothing_sd=FWHM/conversion_factor)
                fO.execute()
                
                # save rev:
                res_nii_file = nib.Nifti1Image(m*-1.0, affine=brain_mask.get_affine(), header=brain_mask.get_header())
                res_nii_file.set_data_dtype(np.float32)
                nib.save(res_nii_file, os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_{}_{}_psc_all_rev.nii.gz'.format(t, data_type, time_locked, measure, self.split_by, int(flip))))
                
                # smooth:
                FWHM = 8.0 # mm
                conversion_factor = 2.35482004503
                inputObject = os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_{}_{}_psc_all_rev.nii.gz'.format(t, data_type, time_locked, measure, self.split_by, int(flip)))
                outputFileName = os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_{}_{}_psc_all_rev.nii.gz'.format(t, data_type, time_locked, measure, self.split_by, int(flip)))
                fO = FSLMathsOperator(inputObject=inputObject)
                fO.configureSmooth(outputFileName=outputFileName, smoothing_sd=FWHM/conversion_factor)
                fO.execute()
                
        # cluster correction:
        cluster_correction = True
        if cluster_correction:
            for t in titles:
                
                input_object_1 = os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_{}_{}_psc_all'.format(t, data_type, time_locked, measure, self.split_by, 0))
                input_object_2 = os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_{}_{}_psc_all_rev'.format(t, data_type, time_locked, measure, self.split_by, 0))
                input_object_3 = os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_{}_{}_psc_all'.format(t, data_type, time_locked, measure, self.split_by, 1))
                input_object_4 = os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_{}_{}_psc_all_rev'.format(t, data_type, time_locked, measure, self.split_by, 1))
                mask = os.path.join('/home/shared/UvA/Niels_UvA/mni_masks/', '2014_fMRI_yesno_epi_box.nii.gz')

                cmdline = 'randomise -i {} -o {} -m {} -1 -T -n 10000'.format(input_object_1, input_object_1, mask)
                cmdline = cmdline + '& randomise -i {} -o {} -m {} -1 -T -n 10000'.format(input_object_2, input_object_2, mask)
                cmdline = cmdline + '& randomise -i {} -o {} -m {} -1 -T -n 10000'.format(input_object_3, input_object_3, mask)
                cmdline = cmdline + '& randomise -i {} -o {} -m {} -1 -T -n 10000'.format(input_object_4, input_object_4, mask)
                subprocess.call( cmdline, shell=True, bufsize=0,)
                
    def WHOLEBRAIN_lateralization_per_session(self, data_type, measure='snr', prepare=False):
        
        brain_mask = np.array(NiftiImage('/home/shared/Niels_UvA/brainstem_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz').data, dtype=bool)
        time_locked = 'resp_locked'
        type_response = self.split_by.split('_')[-1]
        
        if prepare:
            for i in range(len(self.subjects)):
                print self.subjects[i]
                
                # load:
                scalars = np.array(NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_{}_{}_{}_{}.nii.gz'.format(data_type, time_locked, self.subjects[i], type_response))).data)
                
                for s in [1,2]:
                    
                    # split by pupil x right button:
                    scalars_a = scalars[self.pupil_h_ind[i] * self.right[i] * (self.session[i]==s)].mean(axis=0)
                    scalars_b = scalars[self.pupil_h_ind[i] * ~self.right[i] * (self.session[i]==s)].mean(axis=0)
                    scalars_c = scalars[self.pupil_l_ind[i] * self.right[i] * (self.session[i]==s)].mean(axis=0)
                    scalars_d = scalars[self.pupil_l_ind[i] * ~self.right[i] * (self.session[i]==s)].mean(axis=0)
                    scalars_a_std = scalars[self.pupil_h_ind[i] * self.right[i] * (self.session[i]==s)].std(axis=0)
                    scalars_b_std = scalars[self.pupil_h_ind[i] * ~self.right[i] * (self.session[i]==s)].std(axis=0)
                    scalars_c_std = scalars[self.pupil_l_ind[i] * self.right[i] * (self.session[i]==s)].std(axis=0)
                    scalars_d_std = scalars[self.pupil_l_ind[i] * ~self.right[i] * (self.session[i]==s)].std(axis=0)
                    res_nii_file = NiftiImage(np.vstack((scalars_a[np.newaxis,...], scalars_b[np.newaxis,...], scalars_c[np.newaxis,...], scalars_d[np.newaxis,...], scalars_a_std[np.newaxis,...], scalars_b_std[np.newaxis,...], scalars_c_std[np.newaxis,...], scalars_d_std[np.newaxis,...])))
                    res_nii_file.header = NiftiImage('/home/shared/Niels_UvA/brainstem_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz').header
                    res_nii_file.save(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_right_{}_{}_{}_session_{}_split_{}.nii.gz'.format(data_type, time_locked, self.subjects[i], s, self.split_by)))
        
        else:
            
            for s in [1,2]:
            
                # split by pupil x right button:
                scalars_right_a = np.vstack([np.array(NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_right_{}_{}_{}_session_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, s, self.split_by))).data)[0,brain_mask] for subj in self.subjects])
                scalars_right_b = np.vstack([np.array(NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_right_{}_{}_{}_session_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, s, self.split_by))).data)[1,brain_mask] for subj in self.subjects])
                scalars_right_c = np.vstack([np.array(NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_right_{}_{}_{}_session_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, s, self.split_by))).data)[2,brain_mask] for subj in self.subjects])
                scalars_right_d = np.vstack([np.array(NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_right_{}_{}_{}_session_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, s, self.split_by))).data)[3,brain_mask] for subj in self.subjects])
                scalars_right_a_std = np.vstack([np.array(NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_right_{}_{}_{}_session_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, s, self.split_by))).data)[4,brain_mask] for subj in self.subjects])
                scalars_right_b_std = np.vstack([np.array(NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_right_{}_{}_{}_session_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, s, self.split_by))).data)[5,brain_mask] for subj in self.subjects])
                scalars_right_c_std = np.vstack([np.array(NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_right_{}_{}_{}_session_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, s, self.split_by))).data)[6,brain_mask] for subj in self.subjects])
                scalars_right_d_std = np.vstack([np.array(NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_right_{}_{}_{}_session_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, s, self.split_by))).data)[7,brain_mask] for subj in self.subjects])
        
                import copy
                if measure == 'mean':
                    s_right_a = copy.copy(scalars_right_a)
                    s_right_b = copy.copy(scalars_right_b)
                    s_right_c = copy.copy(scalars_right_c)
                    s_right_d = copy.copy(scalars_right_d)
                if measure == 'snr':
                    s_right_a = copy.copy(scalars_right_a) / copy.copy(scalars_right_a_std)
                    s_right_b = copy.copy(scalars_right_b) / copy.copy(scalars_right_b_std)
                    s_right_c = copy.copy(scalars_right_c) / copy.copy(scalars_right_c_std)
                    s_right_d = copy.copy(scalars_right_d) / copy.copy(scalars_right_d_std)
                if measure == 'std':
                    s_right_a = copy.copy(scalars_right_a_std)
                    s_right_b = copy.copy(scalars_right_b_std)
                    s_right_c = copy.copy(scalars_right_c_std)
                    s_right_d = copy.copy(scalars_right_d_std)
        
                # button maps:
                right = np.vstack((s_right_a[np.newaxis,...], s_right_c[np.newaxis,...])).mean(axis=0)
                left = np.vstack((s_right_b[np.newaxis,...], s_right_d[np.newaxis,...])).mean(axis=0)
                right_contrast = right - left
                right_contrast_contrast = (s_right_a - s_right_b) - (s_right_c - s_right_d)
        
                # lateralization map:
                # -------------------
                right_contrast_not_flipped = np.zeros((len(self.subjects),brain_mask.shape[0],brain_mask.shape[1],brain_mask.shape[2]))
                right_contrast_flipped = np.zeros((len(self.subjects),brain_mask.shape[0],brain_mask.shape[1],brain_mask.shape[2]))
                right_contrast_not_flipped[:,brain_mask] = right_contrast
                right_contrast_flipped[:,brain_mask] = right_contrast
                for i in range(len(self.subjects)):
                    right_contrast_flipped[i,:,:,:] = right_contrast_flipped[i,:,:,::-1]
                lateralization = right_contrast_not_flipped - right_contrast_flipped
        
                # save:
                res_nii_file = NiftiImage(lateralization)
                res_nii_file.header = NiftiImage('/home/shared/Niels_UvA/brainstem_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz').header
                res_nii_file.save(os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_{}_{}_{}_psc_all.nii.gz'.format('lateralization', data_type, time_locked, measure, s, self.split_by, int(0))))
    
                # save rev:
                res_nii_file = NiftiImage(lateralization*-1.0)
                res_nii_file.header = NiftiImage('/home/shared/Niels_UvA/brainstem_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz').header
                res_nii_file.save(os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_{}_{}_{}_psc_all_rev.nii.gz'.format('lateralization', data_type, time_locked, measure, s, self.split_by, int(0))))
        
                # cluster correction:
                cluster_correction = True
                if cluster_correction:
                    input_object_1 = os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_{}_{}_{}_psc_all'.format('lateralization', data_type, time_locked, measure, s, self.split_by, 0))
                    input_object_2 = os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_{}_{}_{}_psc_all_rev'.format('lateralization', data_type, time_locked, measure, s, self.split_by, 0))
                    mask = os.path.join('/home/shared/Niels_UvA/brainstem_masks/', 'epi_box_half')
            
                    cmdline = 'randomise -i {} -o {} -m {} -1 -T -n 10000'.format(input_object_1, input_object_1, mask)
                    cmdline = cmdline + '& randomise -i {} -o {} -m {} -1 -T -n 10000'.format(input_object_2, input_object_2, mask)
                    subprocess.call( cmdline, shell=True, bufsize=0,)
    
    def WHOLEBRAIN_combine_searchlight(self, data_type):
        
        # shell()
        
        brain_mask = np.array(NiftiImage('/home/shared/Niels_UvA/brainstem_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz').data, dtype=bool)
        measure = 'mean'
        time_locked = 'stim_locked'
        
        for s in [1,2]:
        # for s in [2]:
            scalars = np.vstack([np.array(NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'searchlight_choice_{}_{}_{}_s{}.nii.gz'.format(data_type, time_locked, subj, s))).data)[brain_mask]-0.5 for subj in self.subjects])
            # scalars = np.vstack([np.array(NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'searchlight_stimulus_{}_{}_{}_s{}.nii.gz'.format(data_type, time_locked, subj, s))).data)[brain_mask]-0.5 for subj in self.subjects])
            
            # maps:
            maps = [scalars,]
            titles = ['searchlight_choice',]
            # titles = ['searchlight_stimulus',]
        
            for m, t in zip(maps, titles):
                for flip in [False, True]:
                    res = np.zeros((len(self.subjects),brain_mask.shape[0],brain_mask.shape[1],brain_mask.shape[2]))
                    res[:,brain_mask] = m
                    if flip:
                        for i in range(len(self.subjects)):
                            if self.subjects[i] in ['sub-03', 'sub-04', 'sub-06', 'sub-08', 'sub-09', 'sub-11', 'sub-13']:
                                pass
                            else:
                                res[i,:,:,:] = res[i,:,:,::-1]
            
                    # save:
                    res_nii_file = NiftiImage(res)
                    res_nii_file.header = NiftiImage('/home/shared/Niels_UvA/brainstem_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz').header
                    res_nii_file.save(os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_{}_{}_psc_all_s{}.nii.gz'.format(t, data_type, time_locked, measure, self.split_by, int(flip), s)))
            
                    # save rev:
                    res_nii_file = NiftiImage(res*-1.0)
                    res_nii_file.header = NiftiImage('/home/shared/Niels_UvA/brainstem_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz').header
                    res_nii_file.save(os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_{}_{}_psc_all_rev_s{}.nii.gz'.format(t, data_type, time_locked, measure, self.split_by, int(flip), s)))
        
            # cluster correction:
            cluster_correction = True
            if cluster_correction:
                for t in titles:

                    input_object_1 = os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_{}_{}_psc_all_s{}'.format(t, data_type, time_locked, measure, self.split_by, 0, s))
                    input_object_2 = os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_{}_{}_psc_all_rev_s{}'.format(t, data_type, time_locked, measure, self.split_by, 0, s))
                    input_object_3 = os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_{}_{}_psc_all_s{}'.format(t, data_type, time_locked, measure, self.split_by, 1, s))
                    input_object_4 = os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_{}_{}_psc_all_rev_s{}'.format(t, data_type, time_locked, measure, self.split_by, 1, s))
                    mask = os.path.join('/home/shared/Niels_UvA/brainstem_masks/', 'epi_box')

                    # cmdline = 'randomise -i {} -o {} -m {} -1 -T -n 10000 -v 4'.format(input_object_1, input_object_1, mask)
                    # cmdline = cmdline + '& randomise -i {} -o {} -m {} -1 -T -n 10000 -v 4'.format(input_object_2, input_object_2, mask)
                    # cmdline = cmdline + '& randomise -i {} -o {} -m {} -1 -T -n 10000 -v 4'.format(input_object_3, input_object_3, mask)
                    # cmdline = cmdline + '& randomise -i {} -o {} -m {} -1 -T -n 10000 -v 4'.format(input_object_4, input_object_4, mask)
                    
                    cmdline = 'randomise -i {} -o {} -m {} -1 -T -n 1000'.format(input_object_1, input_object_1, mask)
                    cmdline = cmdline + '& randomise -i {} -o {} -m {} -1 -T -n 1000'.format(input_object_2, input_object_2, mask)
                    cmdline = cmdline + '& randomise -i {} -o {} -m {} -1 -T -n 1000'.format(input_object_3, input_object_3, mask)
                    cmdline = cmdline + '& randomise -i {} -o {} -m {} -1 -T -n 1000'.format(input_object_4, input_object_4, mask)
                    
                    subprocess.call( cmdline, shell=True, bufsize=0,)
    
    
    def WHOLEBRAIN_event_related_average_plots(self, data_type, measure='mean'):
        
        brain_mask = np.array(NiftiImage('/home/shared/Niels_UvA/brainstem_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz').data, dtype=bool)
        time_locked = 'stim_locked'
        type_response = self.split_by.split('_')[-1]
        
        # load data:
        corrs_all = np.vstack([np.load(os.path.join(self.data_folder, 'correlation', 'whole_brain_{}_{}_{}_{}_{}.npy'.format('mean', data_type, time_locked, subj, self.split_by))) for subj in self.subjects])
        
        # convert to t-statistics:
        for i in range(len(self.subjects)):
            df = len(self.all[i]) - 1 - 1
            for j in range(len(corrs_all[i,:])):
                corrs_all[i,j] = myfuncs.r_to_t(corrs_all[i,j], df)
        
        # shell()
        
        scalars_all_a = np.vstack([np.array(NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_all_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).data)[0,brain_mask] for subj in self.subjects])
        scalars_all_a_std = np.vstack([np.array(NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_all_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).data)[1,brain_mask] for subj in self.subjects])
        
        scalars_pupil_a = np.vstack([np.array(NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_pupil_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).data)[0,brain_mask] for subj in self.subjects])
        scalars_pupil_b = np.vstack([np.array(NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_pupil_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).data)[1,brain_mask] for subj in self.subjects])
        scalars_pupil_a_std = np.vstack([np.array(NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_pupil_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).data)[2,brain_mask] for subj in self.subjects])
        scalars_pupil_b_std = np.vstack([np.array(NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_pupil_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).data)[3,brain_mask] for subj in self.subjects])
        
        scalars_signal_a = np.vstack([np.array(NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_signal_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).data)[0,brain_mask] for subj in self.subjects])
        scalars_signal_b = np.vstack([np.array(NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_signal_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).data)[1,brain_mask] for subj in self.subjects])
        scalars_signal_a_std = np.vstack([np.array(NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_signal_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).data)[2,brain_mask] for subj in self.subjects])
        scalars_signal_b_std = np.vstack([np.array(NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_signal_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).data)[3,brain_mask] for subj in self.subjects])
        
        scalars_choice_a = np.vstack([np.array(NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_choice_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).data)[0,brain_mask] for subj in self.subjects])
        scalars_choice_b = np.vstack([np.array(NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_choice_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).data)[1,brain_mask] for subj in self.subjects])
        scalars_choice_a_std = np.vstack([np.array(NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_choice_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).data)[2,brain_mask] for subj in self.subjects])
        scalars_choice_b_std = np.vstack([np.array(NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_choice_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).data)[3,brain_mask] for subj in self.subjects])
        
        scalars_right_a = np.vstack([np.array(NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_right_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).data)[0,brain_mask] for subj in self.subjects])
        scalars_right_b = np.vstack([np.array(NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_right_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).data)[1,brain_mask] for subj in self.subjects])
        scalars_right_a_std = np.vstack([np.array(NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_right_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).data)[2,brain_mask] for subj in self.subjects])
        scalars_right_b_std = np.vstack([np.array(NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_right_{}_{}_{}_split_{}.nii.gz'.format(data_type, time_locked, subj, self.split_by))).data)[3,brain_mask] for subj in self.subjects])
        
        import copy
        if measure == 'mean':
            s_all_a = copy.copy(scalars_all_a)
            s_pupil_a = copy.copy(scalars_pupil_a)
            s_pupil_b = copy.copy(scalars_pupil_b)
            s_signal_a = copy.copy(scalars_signal_a)
            s_signal_b = copy.copy(scalars_signal_b)
            s_choice_a = copy.copy(scalars_choice_a)
            s_choice_b = copy.copy(scalars_choice_b)
            s_right_a = copy.copy(scalars_right_a)
            s_right_b = copy.copy(scalars_right_b)
        if measure == 'snr':
            s_all_a = copy.copy(scalars_all_a) / copy.copy(scalars_all_a_std)
            s_pupil_a = copy.copy(scalars_pupil_a) / copy.copy(scalars_pupil_a_std)
            s_pupil_b = copy.copy(scalars_pupil_b) / copy.copy(scalars_pupil_b_std)
            s_signal_a = copy.copy(scalars_signal_a) / copy.copy(scalars_signal_a_std)
            s_signal_b = copy.copy(scalars_signal_b) / copy.copy(scalars_signal_b_std)
            s_choice_a = copy.copy(scalars_choice_a) / copy.copy(scalars_choice_a_std)
            s_choice_b = copy.copy(scalars_choice_b) / copy.copy(scalars_choice_b_std)
            s_right_a = copy.copy(scalars_right_a) / copy.copy(scalars_right_a_std)
            s_right_b = copy.copy(scalars_right_b) / copy.copy(scalars_right_b_std)
        if measure == 'std':
            s_all_a = copy.copy(scalars_all_a_std)
            s_pupil_a = copy.copy(scalars_pupil_a_std)
            s_pupil_b = copy.copy(scalars_pupil_b_std)
            s_signal_a = copy.copy(scalars_signal_a_std)
            s_signal_b = copy.copy(scalars_signal_b_std)
            s_choice_a = copy.copy(scalars_choice_a_std)
            s_choice_b = copy.copy(scalars_choice_b_std)
            s_right_a = copy.copy(scalars_right_a_std)
            s_right_b = copy.copy(scalars_right_b_std)
        
        # for masking:
        # ------------
        cortex_mask = np.array(NiftiImage('/home/shared/Niels_UvA/brainstem_masks/harvard_oxford/volume_1.nii.gz').data, dtype=bool)[brain_mask] + np.array(NiftiImage('/home/shared/Niels_UvA/brainstem_masks/harvard_oxford/volume_12.nii.gz').data, dtype=bool)[brain_mask]
        epi_box = np.array(NiftiImage(os.path.join('/home/shared/Niels_UvA/brainstem_masks', 'epi_box.nii.gz')).data, dtype=bool)[brain_mask]
        std_all = scalars_all_a_std
        bad_voxels = np.zeros(std_all.shape[1], dtype=bool)
        for i in range(len(self.subjects)):
            # bad_voxels = bad_voxels + (std_all[i] > 25)
            bad_voxels = bad_voxels + myfuncs.is_outlier(std_all[i,:])
            plt.hist(std_all[i][-bad_voxels], bins=100)
        
        pos_regions_mean = np.array(NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'trials_all_clean_MNI_smooth_{}_{}_{}_{}_psc_all_tfce_corrp_tstat1.nii.gz'.format(time_locked, 'mean', self.split_by, 0))).data)[brain_mask]
        # pos_regions_mean = pos_regions_mean > 0
        pos_regions_mean = pos_regions_mean > 0.99
        neg_regions_mean = np.array(NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'trials_all_clean_MNI_smooth_{}_{}_{}_{}_psc_all_rev_tfce_corrp_tstat1.nii.gz'.format(time_locked, 'mean', self.split_by, 0))).data)[brain_mask]
        # neg_regions_mean = neg_regions_mean > 0
        neg_regions_mean = neg_regions_mean > 0.99
        significant_regions_mean = pos_regions_mean + neg_regions_mean
        
        pos_regions_snr = np.array(NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'trials_all_clean_MNI_smooth_{}_{}_{}_{}_psc_all_tfce_corrp_tstat1.nii.gz'.format(time_locked, 'snr', self.split_by, 0))).data)[brain_mask]
        # pos_regions_snr = pos_regions_snr > 0
        pos_regions_snr = pos_regions_snr > 0.99
        neg_regions_snr = np.array(NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', 'trials_all_clean_MNI_smooth_{}_{}_{}_{}_psc_all_rev_tfce_corrp_tstat1.nii.gz'.format(time_locked, 'snr', self.split_by, 0))).data)[brain_mask]
        # neg_regions_snr = neg_regions_snr > 0
        neg_regions_snr = neg_regions_snr > 0.99
        significant_regions_snr = pos_regions_snr + neg_regions_snr
        
        ind = (cortex_mask & epi_box & -bad_voxels)
        
        # lateral = np.array(NiftiImage('/home/shared/Niels_UvA/surface_masks/rh.lateral.nii.gz').data, dtype=bool)[brain_mask] + np.array(NiftiImage('/home/shared/Niels_UvA/surface_masks/lh.lateral.nii.gz').data, dtype=bool)[brain_mask]
        # medial = np.array(NiftiImage('/home/shared/Niels_UvA/surface_masks/rh.medial.nii.gz').data, dtype=bool)[brain_mask] + np.array(NiftiImage('/home/shared/Niels_UvA/surface_masks/lh.medial.nii.gz').data, dtype=bool)[brain_mask]
        
        
        
        # # PLOTTING!!
        # # ----------
        #
        # Ys = [
        #     [scalars_1_snr[:,ind & pos_regions_mean] - scalars_2_snr[:,ind & pos_regions_mean], scalars_1_snr[:,ind & neg_regions_mean] - scalars_2_snr[:,ind & neg_regions_mean]],
        #     [corrs_all[:,ind & pos_regions_snr], corrs_all[:,ind & neg_regions_snr]],
        #     ]
        #
        # labels_x = ['pos', 'neg']
        # labels_y = ['Change in SNR', 'Correlation to pupil',]
        # titles = ['mean_dSNR', 'SNR_cor_pupil',]
        #
        # for s in range(len(Ys)):
        #
        #     MEANS = (Ys[s][0].mean(axis=1).mean(), Ys[s][1].mean(axis=1).mean())
        #     SEMS = (sp.stats.sem(Ys[s][0].mean(axis=1)), sp.stats.sem(Ys[s][1].mean(axis=1)))
        #     p = [sp.stats.ttest_rel(Ys[s][0].mean(axis=1), Ys[s][1].mean(axis=1))[1]]
        #     # p = [myfuncs.permutationTest(Ys[s][0].mean(axis=1), Ys[s][1].mean(axis=1))[1]]
        #     x = np.arange(0,len(MEANS))
        #     bar_width = 0.75
        #     fig = plt.figure(figsize=(1.25,1.75))
        #     ax = plt.subplot(111)
        #     my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}
        #     ax.bar(x, MEANS, yerr=SEMS, width=bar_width, color=['orange','green'], alpha=1, align='center', edgecolor = 'k', linewidth = 0, capsize = 0, ecolor = 'k', error_kw={'elinewidth':0.5,})
        #     ax.tick_params(axis='y', which='major', labelsize=6)
        #     ax.set_xticks(x)
        #     for i, pp in enumerate(p):
        #         star1 = 'n.s.'
        #         if pp < 0.05:
        #             star1 = '*'
        #         if pp < 0.01:
        #             star1 = '**'
        #         if pp < 0.001:
        #             star1 = '***'
        #         # ax.text(s=star1, x=ind[i], y=ax.get_ylim()[1]-((ax.get_ylim()[1] - ax.get_ylim()[0]) / 10.0), size=10)
        #         ax.text(s=str(round(pp,3)), x=x[i], y=ax.get_ylim()[1]-((ax.get_ylim()[1] - ax.get_ylim()[0]) / 10.0), size=7)
        #     # ax.set_ylim(0,yl)
        #     ax.set_ylabel(labels_y[s], size=7)
        #     sns.despine(offset=10, trim=True)
        #     ax.set_xticklabels(['pos', 'neg'], rotation=45, size=7)
        #     plt.tight_layout()
        #     fig.savefig(os.path.join(self.figure_folder, 'cortex', 'cor_map_{}_{}_bars.pdf'.format(titles[s], self.split_by)),)
            
        # SNR -- change in SNR correlation:
        # --------------------------------
        
        # shell()
        
        
        from matplotlib.pyplot import cm 
        color=cm.rainbow(np.linspace(0,1,len(self.subjects)))
        
        
        # pupil_map =
        # task_map =
        
        # shell()
        
        
        Xs = [
            s_all_a[:,ind],
            # (s_signal_a-s_signal_b)[:,ind],
            # scalars_all_mean[:,ind],
            # scalars_all_mean[:,ind & pos_regions_mean],
            # scalars_all_mean[:,ind & neg_regions_mean],
            # corrs_all[:,ind],
            # scalars_all_snr[:,ind],
            # scalars_all_snr[:,ind & pos_regions_snr],
            # scalars_all_snr[:,ind & neg_regions_snr],
            # scalars_all_snr[:,ind],
            # signal_contrast[:,ind],
            # choice_contrast[:,ind],
            # scalars_all_snr[:,ind & medial],
            # scalars_all_snr[:,ind & lateral],
            ]
        Ys = [
            # corrs_all[:,ind],
            s_pupil_a[:,ind] - s_pupil_b[:,ind]
            # corrs_all[:,ind],
            # scalars_1_snr[:,ind] - scalars_2_snr[:,ind],
            # scalars_1_snr[:,ind & pos_regions_mean] - scalars_2_snr[:,ind & pos_regions_mean],
            # scalars_1_snr[:,ind & neg_regions_mean] - scalars_2_snr[:,ind & neg_regions_mean],
            # scalars_1_snr[:,ind] - scalars_2_snr[:,ind],
            # corrs_all[:,ind],
            # corrs_all[:,ind & pos_regions_snr],
            # corrs_all[:,ind & neg_regions_snr],
            # scalars_all_mean[:,ind],
            # signal_contrast_pupil_h[:,ind] - signal_contrast_pupil_l[:,ind],
            # scalars_1_snr[:,ind] - scalars_2_snr[:,ind],
            # scalars_1_snr[:,ind] - scalars_2_snr[:,ind],
            # corrs_all[:,ind & medial],
            # corrs_all[:,ind & lateral],
            ]
        
        # labels_x = ['fMRI response\n(% signal change)', 'fMRI response\n(% signal change)', ]
        labels_x = ['fMRI response\n(t-value)', 'fMRI response\n(t-value)', ]
        labels_y = ['High - low TPR\n(t-value)', 'Correlation to TPR\n(t-value)']
        titles = ['mean_cor_pupil', 'stimulus_contrast_cor_pupil', ]
        draw_axhline = [False, False, ]
        draw_axvline = [True, True, ]
        for s in range(len(Xs)):
        # for s in [9, 10]:
            X = Xs[s]
            Y = Ys[s]
            
            # for i in range(len(self.subjects)):
            #     # Y[i,:] = (Y[i,:]-np.percentile(Y[i,:], 5))/(np.percentile(Y[i,:], 95)-np.percentile(Y[i,:], 5))
            #     X[i,:] = (X[i,:]-X[i,:].mean())/X[i,:].std()
            #     Y[i,:] = (Y[i,:]-Y[i,:].mean())/Y[i,:].std()
            
            
            
            # r = np.zeros(len(self.subjects))
            # for i in range(len(self.subjects)):
            #     r[i] = sp.stats.pearsonr(scalars_all_mean[i,ind], scalars_1_snr[i,ind]-scalars_2_snr[i,ind])[0]
            # p = myfuncs.permutationTest(r, np.zeros(len(self.subjects)))[1]
            
            fig = plt.figure(figsize=(2,2))
            # ax = sns.kdeplot(X, Y, shade=True)
            ax = fig.add_subplot(111)
            regression_line = []
            r = np.zeros(len(self.subjects))
            inter = np.zeros(len(self.subjects))
            slope = np.zeros(len(self.subjects))
            for i in range(len(self.subjects)):
                x = X[i,:]
                y = Y[i,:]
                sl, intercept, r_value, p_value, std_err = stats.linregress(x,y)
                (m,b) = sp.polyfit(x,y,1)
                x_line = np.linspace(min(X[:,:].mean(axis=0)), max(X[:,:].mean(axis=0)), 50)
                regression_line.append(sp.polyval([m,b],x_line))
                r[i] = r_value
                inter[i] = intercept
                slope[i] = sl
                # ax.plot(x_line, regression_line[i], color=color[i], alpha=0.2)
                
                # popt, pcov = curve_fit(sigmoid, X[i,:], Y[i,:], maxfev=10000000)
                # yyy = sigmoid(x_line, *popt)
                # ax.plot(x_line,yyy)
            
            p1 = round(myfuncs.permutationTest(r, np.zeros(len(self.subjects)), paired=True)[1],3)
            p2 = round(myfuncs.permutationTest(slope, np.zeros(len(self.subjects)), paired=True)[1],3)
            p3 = round(myfuncs.permutationTest(inter, np.zeros(len(self.subjects)), paired=True)[1],3)
            
            regression_line = np.vstack(regression_line)
            ax.set_title('r={}, p={}\nslope={}, p={}\ni={}, p={}'.format(round(r.mean(),3), p1, round(slope.mean(),3), p2, round(inter.mean(),3), p3))
            ax.plot(x_line, regression_line.mean(axis=0), color='Black', alpha=1)
            ax.fill_between(x_line, regression_line.mean(axis=0)-sp.stats.sem(regression_line, axis=0), regression_line.mean(axis=0)+sp.stats.sem(regression_line, axis=0), color='Black', alpha=0.2)
            # slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)
            # (m,b) = sp.polyfit(X,Y,1)
            # x_line = np.linspace(ax.axis()[0], ax.axis()[1], 100)
            # regression_line = sp.polyval([m,b],x_line)
            # ax.plot(x_line, regression_line, color='Black', alpha=1)
            # ax.set_title('r={}, p={}'.format(round(r.mean(),5), round(p,5)))
            # ax.set_ylim(0.4,1.6)
            ax.set_xlabel(labels_x[s])
            ax.set_ylabel(labels_y[s])
            sns.despine(offset=10, trim=True)
            if draw_axvline[s]:
                plt.axvline(0, lw=0.5, color='k')
            if draw_axhline[s]:
                plt.axhline(0, lw=0.5, color='k')
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, 'cortex', 'cor_map_{}_{}.pdf'.format(titles[s], self.split_by)),)
        
        # pos_hi = np.zeros(len(self.subjects))
        # pos_lo = np.zeros(len(self.subjects))
        # neg_hi = np.zeros(len(self.subjects))
        # neg_lo = np.zeros(len(self.subjects))
        # for i in range(len(self.subjects)):
        #     pos_indices = ((scalars_all[i,:] > 0).ravel() & ind & significant_regions)
        #     neg_indices = ((scalars_all[i,:] < 0).ravel() & ind & significant_regions)
        #     pos_hi[i] = np.mean(scalars_2[i, pos_indices])
        #     pos_lo[i] = np.mean(scalars_1[i, pos_indices])
        #     neg_hi[i] = np.mean(scalars_2[i, neg_indices])
        #     neg_lo[i] = np.mean(scalars_1[i, neg_indices])
        #
        # N = 4
        # ticks = np.linspace(0,2,N)  # the x locations for the groups
        # bar_width = 0.6   # the width of the bars
        # spacing = [0, 0, 0, 0]
        #
        # # FIGURE 1
        # values = [pos_hi, pos_lo, neg_hi, neg_lo]
        # MEANS = np.array([np.mean(v) for v in values])
        # SEMS = np.array([sp.stats.sem(v) for v in values])
        # p_values = np.array([sp.stats.ttest_rel(pos_hi, pos_lo)[1], sp.stats.ttest_rel(neg_hi, neg_lo)[1]])
        # fig = plt.figure(figsize=(3,3))
        # ax = fig.add_subplot(111)
        # for i in range(N):
        #     ax.bar(ticks[i]+spacing[i], MEANS[i], width = bar_width, yerr=SEMS[i], color=['r','b','r','b'][i], alpha=1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
        # ax.set_xticklabels( ('pos_H','pos_L','neg_H','neg_L') )
        # ax.set_xticks( ticks )
        # ax.tick_params(axis='x', which='major', labelsize=7)
        # ax.tick_params(axis='y', which='major', labelsize=7)
        # plt.ylabel('fMRI response\n(% signal change)')
        # plt.text(x=np.mean((ticks[0],ticks[1])), y=0, s='p = {}'.format(round(p_values[0],3)), horizontalalignment='center')
        # plt.text(x=np.mean((ticks[2],ticks[3])), y=0, s='p = {}'.format(round(p_values[1],3)), horizontalalignment='center')
        # sns.despine(offset=10, trim=True)
        # plt.tight_layout()
        # fig.savefig(os.path.join(self.figure_folder, 'cortex', 'double_dis_{}.pdf'.format(self.split_by)),)
        #
        #
        # delta_SNR_pos = np.zeros(len(self.subjects))
        # delta_SNR_neg = np.zeros(len(self.subjects))
        # for i in range(len(self.subjects)):
        #     pos_indices = ((scalars_all[i,:] > 0).ravel() & ind & significant_regions)
        #     neg_indices = ((scalars_all[i,:] < 0).ravel() & ind & significant_regions)
        #     delta_SNR_pos[i] = np.mean(scalars_1[i, pos_indices]) - np.mean(scalars_2[i, pos_indices])
        #     delta_SNR_neg[i] = np.mean(scalars_1[i, neg_indices]) - np.mean(scalars_2[i, neg_indices])
        #
        # N = 2
        # ticks = np.linspace(0,1,N)  # the x locations for the groups
        # bar_width = 0.6   # the width of the bars
        # spacing = [0, 0, 0, 0]
        #
        # # FIGURE 1
        # values = [delta_SNR_pos, delta_SNR_neg]
        # MEANS = np.array([np.mean(v) for v in values])
        # SEMS = np.array([sp.stats.sem(v) for v in values])
        # p_values = sp.stats.ttest_rel(values[0], values[1])[1],
        # fig = plt.figure(figsize=(3,3))
        # ax = fig.add_subplot(111)
        # for i in range(N):
        #     ax.bar(ticks[i]+spacing[i], MEANS[i], width = bar_width, yerr=SEMS[i], color=['r','b','r','b'][i], alpha=1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
        # ax.set_xticklabels( ('pos', 'neg') )
        # ax.set_xticks( ticks )
        # ax.tick_params(axis='x', which='major', labelsize=7)
        # ax.tick_params(axis='y', which='major', labelsize=7)
        # plt.ylabel('fMRI response\n(% signal change)')
        # plt.text(x=np.mean((ticks[0],ticks[1])), y=0, s='p = {}'.format(round(p_values[0],3)), horizontalalignment='center')
        # sns.despine(offset=10, trim=True)
        # plt.tight_layout()
        # fig.savefig(os.path.join(self.figure_folder, 'cortex', 'double_dis_{}_2.pdf'.format(self.split_by)),)


        # # SNR -- correlation to pupil correlation:
        # # ----------------------------------------
        #
        # corrs_all = np.vstack([np.load(os.path.join(self.data_folder, 'correlation', 'whole_brain_{}_{}_{}_{}_{}.npy'.format('mean', data_type, time_locked, subj, c))) for subj in self.subjects])
        #
        # R1 = []
        # R2 = []
        # R3 = []
        # for type_response in ['mean', 'snr']:
        #     if type_response == 'snr':
        #         phasics_all = np.vstack([np.load(os.path.join(self.data_folder, 'event_related_average_wb', 'whole_brain_{}_{}_{}_{}_{}.npy'.format('mean', data_type, time_locked, subj, c)))[3] for subj in self.subjects]) / np.vstack([np.load(os.path.join(self.data_folder, 'event_related_average_wb', 'whole_brain_{}_{}_{}_{}_{}.npy'.format('std', data_type, time_locked, subj, c)))[3] for subj in self.subjects])
        #         phasics_1 = np.vstack([np.load(os.path.join(self.data_folder, 'event_related_average_wb', 'whole_brain_{}_{}_{}_{}_{}.npy'.format('mean', data_type, time_locked, subj, c)))[4] for subj in self.subjects]) / np.vstack([np.load(os.path.join(self.data_folder, 'event_related_average_wb', 'whole_brain_{}_{}_{}_{}_{}.npy'.format('std', data_type, time_locked, subj, c)))[4] for subj in self.subjects])
        #         phasics_2 = np.vstack([np.load(os.path.join(self.data_folder, 'event_related_average_wb', 'whole_brain_{}_{}_{}_{}_{}.npy'.format('mean', data_type, time_locked, subj, c)))[5] for subj in self.subjects]) / np.vstack([np.load(os.path.join(self.data_folder, 'event_related_average_wb', 'whole_brain_{}_{}_{}_{}_{}.npy'.format('std', data_type, time_locked, subj, c)))[5] for subj in self.subjects])
        #     else:
        #         phasics_all = np.vstack([np.load(os.path.join(self.data_folder, 'event_related_average_wb', 'whole_brain_{}_{}_{}_{}_{}.npy'.format(type_response, data_type, time_locked, subj, c)))[3] for subj in self.subjects])
        #         phasics_1 = np.vstack([np.load(os.path.join(self.data_folder, 'event_related_average_wb', 'whole_brain_{}_{}_{}_{}_{}.npy'.format(type_response, data_type, time_locked, subj, c)))[4] for subj in self.subjects])
        #         phasics_2 = np.vstack([np.load(os.path.join(self.data_folder, 'event_related_average_wb', 'whole_brain_{}_{}_{}_{}_{}.npy'.format(type_response, data_type, time_locked, subj, c)))[5] for subj in self.subjects])
        #
        #     # correlation to signal_type:
        #     Rs = np.zeros(len(self.subjects))
        #     for i in range(len(self.subjects)):
        #         Rs[i] = sp.stats.pearsonr(phasics_all[i,ind], corrs_all[i,ind])[0]
        #     R1.append(Rs)
        #
        #     # correlation to signal_type contrast:
        #     Rs = np.zeros(len(self.subjects))
        #     for i in range(len(self.subjects)):
        #         Rs[i] = sp.stats.pearsonr(phasics_2[i,ind]-phasics_1[i,ind], corrs_all[i,ind])[0]
        #     R2.append(Rs)
        #
        #
        #     # correlation to signal_type contrast:
        #     Rs = np.zeros(len(self.subjects))
        #     for i in range(len(self.subjects)):
        #         Rs[i] = sp.stats.pearsonr(phasics_2[i,ind]-phasics_1[i,ind], phasics_all[i,ind])[0]
        #     R3.append(Rs)
        #
        # MEANS = [R1[i].mean() for i in range(len(R1))]
        # SEMS = [sp.stats.sem(R1[i]) for i in range(len(R1))]
        # # p = [sp.stats.ttest_1samp(R1[i],0)[1] for i in range(len(R1))]
        # p = [myfuncs.permutationTest(R1[i],np.zeros(len(self.subjects)))[1] for i in range(len(R1))]
        # N = len(MEANS)
        # ind = np.linspace(0,N,N)
        # bar_width = 0.80
        # fig = plt.figure(figsize=(1.25,1.75))
        # ax = fig.add_subplot(111)
        # for i in range(N):
        #     ax.bar(ind[i], height=MEANS[i], width = bar_width, yerr = SEMS[i], color = 'k', alpha = 1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
        # ax.set_title('N={}'.format(len(self.subjects)), size=8)
        # ax.set_xticks( (ind) )
        # ax.set_xticklabels( ['mean', 'snr'] )
        # for i in range(N):
        #     ax.text(ind[i],0,'{}'.format(round(p[i],3)))
        # plt.gca().spines["bottom"].set_linewidth(.5)
        # plt.gca().spines["left"].set_linewidth(.5)
        # sns.despine(offset=10, trim=True)
        # plt.tight_layout()
        # fig.savefig(os.path.join(self.figure_folder, 'cortex', 'correlation_bars_1_{}_{}.pdf'.format(c, type_response)),)
        #
        # MEANS = [R2[i].mean() for i in range(len(R2))]
        # SEMS = [sp.stats.sem(R2[i]) for i in range(len(R2))]
        # # p = [sp.stats.ttest_1samp(R2[i],0)[1] for i in range(len(R2))]
        # p = [myfuncs.permutationTest(R2[i],np.zeros(len(self.subjects)))[1] for i in range(len(R2))]
        # N = len(MEANS)
        # ind = np.linspace(0,N,N)
        # bar_width = 0.80
        # fig = plt.figure(figsize=(1.25,1.75))
        # ax = fig.add_subplot(111)
        # for i in range(N):
        #     ax.bar(ind[i], height=MEANS[i], width = bar_width, yerr = SEMS[i], color = 'k', alpha = 1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
        # ax.set_title('N={}'.format(len(self.subjects)), size=8)
        # ax.set_xticks( (ind) )
        # ax.set_xticklabels( ['mean', 'snr'] )
        # for i in range(N):
        #     ax.text(ind[i],0,'{}'.format(round(p[i],3)))
        # plt.gca().spines["bottom"].set_linewidth(.5)
        # plt.gca().spines["left"].set_linewidth(.5)
        # sns.despine(offset=10, trim=True)
        # plt.tight_layout()
        # fig.savefig(os.path.join(self.figure_folder, 'cortex', 'correlation_bars_2_{}_{}.pdf'.format(c, type_response)),)
        #
        # results_psc = np.zeros((len(self.subjects),brain_mask.shape[0],brain_mask.shape[1],brain_mask.shape[2]))
        # results_psc[:,brain_mask] = bad_voxels
        # res_nii_file = NiftiImage(results_psc)
        # res_nii_file.header = NiftiImage('/home/shared/Niels_UvA/brainstem_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz').header
        # res_nii_file.save(os.path.join(self.data_folder, 'event_related_average_wb', 'bad_voxels.nii.gz'.format(data_type, c, time_locked, type_response)))
        
        
    def WHOLEBRAIN_correlation(self, data_type, measure='mean', source='pupil_d'):
        
        brain_mask = nib.load(os.path.join(self.mask_folder, 'MNI152_T1_2mm_brain_mask_bin.nii.gz'))
        mask = np.array(brain_mask.get_data(), dtype=bool)
        LC_mask = np.array(nib.load(os.path.join(self.mask_folder, 'brainstem', 'LC_standard_1.nii.gz')).get_data(), dtype=bool)
        
        for time_locked in ['stim_locked']:
            
            if source == 'BOLD_present' or source == 'BOLD_choice':
                corrs_all = np.vstack([np.load(os.path.join(self.data_folder, 'correlation', 'whole_brain_{}_{}_{}_{}_{}.npy'.format(measure, data_type, time_locked, subj, source))).ravel() - 0.50 for subj in self.subjects])
            else:
                corrs_all = np.stack([nib.load(os.path.join(self.data_folder, 'correlation', 'whole_brain_{}_{}_{}_{}_{}.nii.gz'.format(measure, data_type, time_locked, subj, source))).get_data() for subj in self.subjects], axis=-1)
            
            # save across:
            res_nii_file = nib.Nifti1Image(corrs_all, affine=brain_mask.get_affine(), header=brain_mask.get_header())
            res_nii_file.set_data_dtype(np.float32)
            nib.save(res_nii_file, os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_phasic_all.nii.gz'.format(data_type, source, time_locked, 'corr')))
            res_nii_file = nib.Nifti1Image(corrs_all*-1.0, affine=brain_mask.get_affine(), header=brain_mask.get_header())
            res_nii_file.set_data_dtype(np.float32)
            nib.save(res_nii_file, os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_phasic_all_rev.nii.gz'.format(data_type, source, time_locked, 'corr')))
            
            # save uncorrected T-stats:            
            results_T = np.zeros(mask.shape)
            results_p = np.zeros(mask.shape)
            results_T[mask], results_p[mask] = sp.stats.ttest_1samp(corrs_all[mask,:], 0, axis=-1)
            res_nii_file = nib.Nifti1Image(results_T, affine=brain_mask.get_affine(), header=brain_mask.get_header())
            res_nii_file.set_data_dtype(np.float32)
            nib.save(res_nii_file, os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_combined.nii.gz'.format(data_type, source, time_locked, 'corr')))
            
            # run FSL:
            run_FSL_ttest = True
            # mask = os.path.join(self.data_folder, 'event_related_average_wb', '/home/shared/Niels_UvA/brainstem_masks/brainstem_box_bin.nii.gz')
            mask = os.path.join(self.data_folder, 'event_related_average_wb', '/home/shared/Niels_UvA/brainstem_masks/brainstem_bin_2.nii.gz')
            if run_FSL_ttest:
                cmdline = ''
                cmdline = cmdline + 'randomise -i {} -o {} -m {} -1 -T -n 10000'.format( os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_phasic_all'.format(data_type, source, time_locked, 'corr')), os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_phasic_all'.format(data_type, source, time_locked, 'corr')), os.path.join(self.mask_folder, '2014_fMRI_yesno_epi_box'))
                cmdline = cmdline + ' & randomise -i {} -o {} -m {} -1 -T -n 10000'.format( os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_phasic_all_rev'.format(data_type, source, time_locked, 'corr')), os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_phasic_all_rev'.format(data_type, source, time_locked, 'corr')), os.path.join(self.mask_folder, '2014_fMRI_yesno_epi_box'))
                # cmdline = cmdline + ' & randomise -i {} -o {} -m {} -1 -T -n 10000'.format( os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_phasic_all'.format(data_type, source, time_locked, 'corr')), os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_phasic_all_brainstemMask'.format(data_type, source, time_locked, 'corr')), mask)
                # cmdline = cmdline + ' & randomise -i {} -o {} -m {} -1 -T -n 10000'.format( os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_phasic_all_rev'.format(data_type, source, time_locked, 'corr')), os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_phasic_all_rev_brainstemMask'.format(data_type, source, time_locked, 'corr')), mask)
                #
                # cmdline = cmdline + 'randomise -i {} -o {} -m {} -1 -T -n 10000'.format( os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_phasic_all'.format(data_type, source, time_locked, 'corr')), os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_lat_{}_{}_phasic_all'.format(data_type, source, time_locked, 'corr')), os.path.join('/home/shared/Niels_UvA/brainstem_masks/', 'MNI152_T1_2mm_brain_mask_bin_half'))
                # cmdline = cmdline + ' & randomise -i {} -o {} -m {} -1 -T -n 10000'.format( os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_phasic_all_rev'.format(data_type, source, time_locked, 'corr')), os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_lat_{}_{}_phasic_all_rev'.format(data_type, source, time_locked, 'corr')), os.path.join('/home/shared/Niels_UvA/brainstem_masks/', 'MNI152_T1_2mm_brain_mask_bin_half'))
                subprocess.call( cmdline, shell=True, bufsize=0,)

            if data_type == 'clean_MNI':
                
                # plot:
                volume_file1 = os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_combined.nii.gz'.format(data_type, source, time_locked, 'corr'))
                # threshold_file1 = NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_phasic_all_brainstemMask_tfce_corrp_tstat1.nii.gz'.format(data_type, source, time_locked, 'corr'))).data
                threshold_file1 = nib.load(os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_phasic_all_tfce_corrp_tstat1.nii.gz'.format(data_type, source, time_locked, 'corr'))).get_data()
                
                dummy1 = nib.load(volume_file1).get_data()
                # dummy1 = dummy1.mean(axis=0)
                dummy1[threshold_file1<0.95] = 0
                # dummy1[~np.array(NiftiImage(mask).data, dtype=bool)] = 0
                dummy1 = nib.Nifti1Image(dummy1, affine=brain_mask.get_affine(), header=brain_mask.get_header())
                dummy1.set_data_dtype(np.float32)
                nib.save(dummy1, os.path.join('/home/shared/UvA/Niels_UvA/Visual_UvA/surface_plots', "dummy1.nii.gz"))
                dummy_file1 = os.path.join('/home/shared/UvA/Niels_UvA/Visual_UvA/surface_plots', "dummy1.nii.gz")
                
                from nilearn.image.resampling import coord_transform
                from nilearn import plotting

                cuts = [(42, 44, 20,), (45, 49, 34), (45, 53, 29), (45, 53, 30)]
                bg_img = nib.load(os.path.join(self.mask_folder, 'MNI_standard_2x2x2.nii.gz'))

                for i, cc in enumerate(cuts):
                    
                    c = coord_transform(cc[0], cc[1], cc[2], bg_img.get_affine())
                    kwargs = {'ls': ':', 'lw': 0.5}
                    fig = plt.figure(figsize=(9,3))
                    display = plotting.plot_anat(bg_img, cut_coords=c, figure=fig)
                    display.add_overlay(dummy_file1, colorbar=False, cmap="YlOrRd_r",)
                    # display.add_contours(mask, levels=[0.01], colors='r', alpha=0.75)
                    display.add_contours(os.path.join(self.mask_folder, 'brainstem', 'LC_standard_1.nii.gz'), levels=[0.05], colors='yellow', alpha=1, **kwargs)
                    display.add_contours(os.path.join(self.mask_folder, 'brainstem', 'mean_VTA.nii.gz'), levels=[0.45], colors='yellow', alpha=1, **kwargs)
                    display.add_contours(os.path.join(self.mask_folder, 'brainstem', 'mean_SN.nii.gz'), levels=[0.45], colors='yellow', alpha=1, **kwargs)
                    display.add_contours(os.path.join(self.mask_folder, 'brainstem', 'sup_col_jw.nii.gz'), levels=[0.1], colors='yellow', alpha=1, **kwargs)
                    display.add_contours(os.path.join(self.mask_folder, 'brainstem', 'inf_col_jw.nii.gz'), levels=[0.1], colors='yellow', alpha=1, **kwargs)
                    display.add_contours(os.path.join(self.mask_folder, 'brainstem', 'basal_forebrain_123.nii.gz'), levels=[0.2], colors='yellow', alpha=1, **kwargs)
                    display.add_contours(os.path.join(self.mask_folder, 'brainstem', 'basal_forebrain_4.nii.gz'), levels=[0.2], colors='yellow', alpha=1, **kwargs)
                    display.savefig(os.path.join(self.figure_folder, 'cortex', 'pupil_correlation_{}_{}_{}_b.pdf'.format(i, data_type, source)))

                    # fig = plt.figure(figsize=(9,3))
                    # display = plotting.plot_anat(bg_img, cut_coords=c, figure=fig)
                    # # display.add_contours(mask, levels=[0.01], colors='r', alpha=0.75)
                    # display.savefig(os.path.join(self.figure_folder, 'cortex', 'pupil_correlation_{}_{}_{}_a.pdf'.format(i, data_type, source)))
    
    def WHOLEBRAIN_correlation_per_session(self, data_type, measure='mean', source='pupil_d'):
        
        brain_mask = np.array(NiftiImage('/home/shared/Niels_UvA/brainstem_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz').data, dtype=bool)
        LC_mask = np.array(NiftiImage('/home/shared/Niels_UvA/brainstem_masks/LC_standard_1.nii.gz').data, dtype=bool)
        
        for s in [1,2]:
        
            for time_locked in ['stim_locked']:
            
                if source == 'BOLD_present' or source == 'BOLD_choice':
                    corrs_all = np.vstack([np.load(os.path.join(self.data_folder, 'correlation', 'whole_brain_{}_{}_{}_{}_{}_s{}.npy'.format(measure, data_type, time_locked, subj, source, s))).ravel() - 0.50 for subj in self.subjects])
                else:
                    corrs_all = np.vstack([np.load(os.path.join(self.data_folder, 'correlation', 'whole_brain_{}_{}_{}_{}_{}.npy'.format(measure, data_type, time_locked, subj, source))).ravel() for subj in self.subjects])
            
                results_T = np.zeros((1,brain_mask.shape[0],brain_mask.shape[1],brain_mask.shape[2]))
                results_p = np.zeros((1,brain_mask.shape[0],brain_mask.shape[1],brain_mask.shape[2]))
                results_T[0,brain_mask], results_p[0,brain_mask] = sp.stats.ttest_1samp(corrs_all, 0, axis=0)
            
                # save uncorrected T-stats:
                res_nii_file = NiftiImage(results_T)
                res_nii_file.header = NiftiImage('/home/shared/Niels_UvA/brainstem_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz').header
                res_nii_file.save(os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_combined_s{}.nii.gz'.format(data_type, source, time_locked, 'corr', s)))
            
                # save all trials (psc):
                results_psc = np.zeros((len(self.subjects),brain_mask.shape[0],brain_mask.shape[1],brain_mask.shape[2]))
                results_psc[:,brain_mask] = corrs_all
            
                # flip:
                for i in range(len(self.subjects)):
                    if self.subjects[i] in ['sub-03', 'sub-04', 'sub-06', 'sub-08', 'sub-09', 'sub-11', 'sub-13']:
                        pass
                    else:
                        results_psc[i,:,:,:] = results_psc[i,:,:,::-1]
                
                    # make lateralization:
                    results_psc[i,:,:,0:45] = results_psc[i,:,:,-45:][:,:,::-1] - results_psc[i,:,:,0:45]
            
                res_nii_file = NiftiImage(results_psc)
                res_nii_file.header = NiftiImage('/home/shared/Niels_UvA/brainstem_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz').header
                res_nii_file.save(os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_phasic_all_s{}.nii.gz'.format(data_type, source, time_locked, 'corr', s)))
            
                res_nii_file = NiftiImage(results_psc*-1.0)
                res_nii_file.header = NiftiImage('/home/shared/Niels_UvA/brainstem_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz').header
                res_nii_file.save(os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_phasic_all_rev_s{}.nii.gz'.format(data_type, source, time_locked, 'corr', s)))
            
                # run FSL:
                run_FSL_ttest = True
                # mask = os.path.join(self.data_folder, 'event_related_average_wb', '/home/shared/Niels_UvA/brainstem_masks/brainstem_box_bin.nii.gz')
                mask = os.path.join(self.data_folder, 'event_related_average_wb', '/home/shared/Niels_UvA/brainstem_masks/brainstem_bin_2.nii.gz')
                if run_FSL_ttest:
                    cmdline = ''
                    # cmdline = cmdline + 'randomise -i {} -o {} -m {} -1 -T -n 10000'.format( os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_phasic_all'.format(data_type, source, time_locked, 'corr')), os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_phasic_all_brainstemMask'.format(data_type, source, time_locked, 'corr')), mask)
                    # cmdline = cmdline + ' & randomise -i {} -o {} -m {} -1 -T -n 10000'.format( os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_phasic_all_rev'.format(data_type, source, time_locked, 'corr')), os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_phasic_all_rev_brainstemMask'.format(data_type, source, time_locked, 'corr')), mask)
                    # cmdline = cmdline + ' & randomise -i {} -o {} -m {} -1 -T -n 10000'.format( os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_phasic_all'.format(data_type, source, time_locked, 'corr')), os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_phasic_all'.format(data_type, source, time_locked, 'corr')), os.path.join('/home/shared/Niels_UvA/brainstem_masks/', 'epi_box'))
                    # cmdline = cmdline + ' & randomise -i {} -o {} -m {} -1 -T -n 10000'.format( os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_phasic_all_rev'.format(data_type, source, time_locked, 'corr')), os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_phasic_all_rev'.format(data_type, source, time_locked, 'corr')), os.path.join('/home/shared/Niels_UvA/brainstem_masks/', 'epi_box'))
                    cmdline = cmdline + 'randomise -i {} -o {} -m {} -1 -T -n 10000'.format( os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_phasic_all_s{}'.format(data_type, source, time_locked, 'corr', s)), os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_lat_{}_{}_phasic_all_s{}'.format(data_type, source, time_locked, 'corr', s)), os.path.join('/home/shared/Niels_UvA/brainstem_masks/', 'epi_box_half'))
                    cmdline = cmdline + ' & randomise -i {} -o {} -m {} -1 -T -n 10000'.format( os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_phasic_all_rev_s{}'.format(data_type, source, time_locked, 'corr', s)), os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_lat_{}_{}_phasic_all_rev_s{}'.format(data_type, source, time_locked, 'corr', s)), os.path.join('/home/shared/Niels_UvA/brainstem_masks/', 'epi_box_half'))
                    subprocess.call( cmdline, shell=True, bufsize=0,)
    
    
    def WHOLEBRAIN_noise_correlation_make_dataframe(self, data_type, prepare=False):
        
        shell()
        
        time_locked = 'stim_locked'
        type_response = self.split_by.split('_')[-1]
        cortex_mask = np.array(nib.load(os.path.join(self.mask_folder, 'harvard_oxford/volume_1.nii.gz')).get_data(), dtype=bool) + np.array(nib.load(os.path.join(self.mask_folder, 'harvard_oxford/volume_12.nii.gz')).get_data(), dtype=bool)
        epi_box = np.array(nib.load(os.path.join(self.mask_folder, '2014_fMRI_yesno_epi_box.nii.gz')).get_data(), dtype=bool)
        epi_box = epi_box & cortex_mask
        
        rois = np.concatenate([
                    # PARCELATION LEFT:
                    # -----------------
                    
                    [
                    # visual:
                    'lh.Pole_occipital',
                    'lh.G_occipital_sup',
                    'lh.S_oc_sup_and_transversal',
                    'lh.G_occipital_middle',
                    'lh.S_occipital_ant',
                    'lh.S_oc_middle_and_Lunatus',
                    'lh.G_and_S_occipital_inf',
                    'lh.S_collat_transv_post',
                    'lh.G_oc-temp_med-Lingual',
                    'lh.S_calcarine',
                    'lh.G_cuneus',
                    ],
                    [
                    # temporal:
                    'lh.Lat_Fis-post',
                    'lh.G_temp_sup-Plan_tempo',
                    'lh.S_temporal_transverse',
                    'lh.G_temp_sup-G_T_transv',
                    'lh.G_temp_sup-Lateral',
                    'lh.S_temporal_sup',
                    'lh.G_temporal_middle',
                    'lh.S_temporal_inf',
                    'lh.G_temporal_inf',
                    # 'lh.S_oc-temp_lat',
                    'lh.G_oc-temp_med-Parahip',
                    'lh.S_collat_transv_ant',
                    'lh.G_oc-temp_lat-fusifor',
                    'lh.S_oc-temp_med_and_Lingual',
                    'lh.G_temp_sup-Plan_polar',
                    'lh.Pole_temporal',
                    ],
                    [
                    # parietal:
                    'lh.S_parieto_occipital',
                    'lh.S_subparietal',
                    'lh.G_precuneus',
                    'lh.G_parietal_sup',
                    'lh.S_intrapariet_and_P_trans',
                    'lh.G_pariet_inf-Angular',
                    'lh.S_interm_prim-Jensen',
                    'lh.G_and_S_paracentral',
                    'lh.S_postcentral',
                    'lh.G_postcentral',
                    'lh.S_central',
                    'lh.G_pariet_inf-Supramar',
                    'lh.G_and_S_subcentral',
                    ],
                    [
                    # insular:
                    'lh.S_circular_insula_sup',
                    'lh.G_insular_short',
                    'lh.S_circular_insula_inf',
                    'lh.G_Ins_lg_and_S_cent_ins',
                    'lh.S_circular_insula_ant',
                    ],
                    [
                    # cingulate:
                    'lh.G_cingul-Post-ventral',
                    'lh.S_pericallosal',
                    'lh.G_cingul-Post-dorsal',
                    'lh.S_cingul-Marginalis',
                    'lh.G_and_S_cingul-Mid-Post',
                    'lh.G_and_S_cingul-Mid-Ant',
                    'lh.G_and_S_cingul-Ant',
                    ],
                    [
                    # frontal:
                    # 'lh.G_precentral',
                    'lh.S_precentral-sup-part',
                    'lh.S_precentral-inf-part',
                    'lh.G_front_sup',
                    'lh.S_front_sup',
                    'lh.G_front_middle',
                    # 'lh.S_front_middle',
                    'lh.S_front_inf',
                    'lh.G_front_inf-Opercular',
                    'lh.G_front_inf-Triangul',
                    'lh.S_orbital_lateral',
                    'lh.Lat_Fis-ant-Horizont',
                    'lh.Lat_Fis-ant-Vertical',
                    'lh.G_front_inf-Orbital',
                    'lh.G_and_S_transv_frontopol',
                    'lh.G_and_S_frontomargin',
                    'lh.G_orbital',
                    'lh.S_orbital-H_Shaped',
                    'lh.S_orbital_med-olfact',
                    'lh.G_rectus',
                    'lh.S_suborbital',
                    'lh.G_subcallosal',
                    ],
                    # PARCELATION RIGHT:
                    # -----------------
                    [
                    # visual:
                    'rh.Pole_occipital',
                    'rh.G_occipital_sup',
                    'rh.S_oc_sup_and_transversal',
                    'rh.G_occipital_middle',
                    'rh.S_occipital_ant',
                    'rh.S_oc_middle_and_Lunatus',
                    'rh.G_and_S_occipital_inf',
                    'rh.S_collat_transv_post',
                    'rh.G_oc-temp_med-Lingual',
                    'rh.S_calcarine',
                    'rh.G_cuneus',
                    ],
                    [
                    # temporal:
                    'rh.Lat_Fis-post',
                    'rh.G_temp_sup-Plan_tempo',
                    'rh.S_temporal_transverse',
                    'rh.G_temp_sup-G_T_transv',
                    'rh.G_temp_sup-Lateral',
                    'rh.S_temporal_sup',
                    'rh.G_temporal_middle',
                    'rh.S_temporal_inf',
                    'rh.G_temporal_inf',
                    # 'rh.S_oc-temp_lat',
                    'rh.G_oc-temp_med-Parahip',
                    'rh.S_collat_transv_ant',
                    'rh.G_oc-temp_lat-fusifor',
                    'rh.S_oc-temp_med_and_Lingual',
                    'rh.G_temp_sup-Plan_polar',
                    'rh.Pole_temporal',
                    ],
                    [
                    # parietal:
                    'rh.S_parieto_occipital',
                    'rh.S_subparietal',
                    'rh.G_precuneus',
                    'rh.G_parietal_sup',
                    'rh.S_intrapariet_and_P_trans',
                    'rh.G_pariet_inf-Angular',
                    'rh.S_interm_prim-Jensen',
                    'rh.G_and_S_paracentral',
                    'rh.S_postcentral',
                    'rh.G_postcentral',
                    'rh.S_central',
                    'rh.G_pariet_inf-Supramar',
                    'rh.G_and_S_subcentral',
                    ],
                    [
                    # insular:
                    'rh.S_circular_insula_sup',
                    'rh.G_insular_short',
                    'rh.S_circular_insula_inf',
                    'rh.G_Ins_lg_and_S_cent_ins',
                    'rh.S_circular_insula_ant',
                    ],
                    [
                    # cingulate:
                    'rh.G_cingul-Post-ventral',
                    'rh.S_pericallosal',
                    'rh.G_cingul-Post-dorsal',
                    'rh.S_cingul-Marginalis',
                    'rh.G_and_S_cingul-Mid-Post',
                    'rh.G_and_S_cingul-Mid-Ant',
                    'rh.G_and_S_cingul-Ant',
                    ],
                    [
                    # frontal:
                    # 'rh.G_precentral',
                    'rh.S_precentral-sup-part',
                    'rh.S_precentral-inf-part',
                    'rh.G_front_sup',
                    'rh.S_front_sup',
                    'rh.G_front_middle',
                    # 'rh.S_front_middle',
                    'rh.S_front_inf',
                    'rh.G_front_inf-Opercular',
                    'rh.G_front_inf-Triangul',
                    'rh.S_orbital_lateral',
                    'rh.Lat_Fis-ant-Horizont',
                    'rh.Lat_Fis-ant-Vertical',
                    'rh.G_front_inf-Orbital',
                    'rh.G_and_S_transv_frontopol',
                    'rh.G_and_S_frontomargin',
                    'rh.G_orbital',
                    'rh.S_orbital-H_Shaped',
                    'rh.S_orbital_med-olfact',
                    'rh.G_rectus',
                    'rh.S_suborbital',
                    'rh.G_subcallosal',
                    ]
                ])
                
        dfs = []
        for i in range(len(self.subjects)):
            print self.subjects[i]
            df = pd.DataFrame()
            scalars = np.array(nib.load(os.path.join(self.data_folder, 'event_related_average_wb', 'scalars_{}_{}_{}_RT_{}.nii.gz'.format('clean_MNI', time_locked, self.subjects[i], type_response))).get_data())
            for r in rois:
                roi = np.array(nib.load(os.path.join(os.path.join(self.mask_folder, 'destrieux/', r+'.nii.gz'))).get_data(), dtype=bool) & epi_box
                if sum(roi.ravel()) < 25:
                    df[r] = np.repeat(np.NaN, scalars.shape[-1])
                else:
                    df[r] = scalars[roi,:].mean(axis=0)
            df['subject'] = self.subjects[i]
            dfs.append(df)
        data_frame = pd.concat(dfs)
        data_frame.to_csv(os.path.join(self.data_folder, 'dataframes', 'destrieux.csv'))
        
    def WHOLEBRAIN_noise_correlation(self, data_type, partial=False,):
        
        # text_file = open(os.path.join(self.data_folder, 'dataframes', 'destrieux.txt'), 'w')
        # for string in rois:
        #     text_file.write(str(string))
        #     text_file.write('\n')
        # text_file.close()
        
        rois = np.loadtxt(os.path.join(self.data_folder, 'dataframes', 'destrieux.txt'), dtype=str)
                
        df = pd.read_csv(os.path.join(self.data_folder, 'dataframes', 'destrieux.csv'))
        df = df.drop('Unnamed: 0', 1)
        # rois = np.array(df.columns)
        
        # omissions:
        remove_trials = np.concatenate(self.omissions)[~np.concatenate(self.omissions_ori)]
        df = df[~remove_trials]
        
        # detect bad ROIs based on size (<25 voxels --> activations == NaN)
        bad_rois = rois[np.isnan(np.array(df.ix[:,rois])).sum(axis=0) > 1]
        for r in bad_rois:
            hemi, roi = r.split('.')
            if hemi == 'rh':
                bad_rois = np.concatenate((bad_rois, np.array(['lh.'+roi])))
            else:
                bad_rois = np.concatenate((bad_rois, np.array(['rh.'+roi])))
            
            try:
                rois = np.delete(rois, np.where(rois == 'rh.'+roi)[0][0])
            except:
                pass
            try:
                rois = np.delete(rois, np.where(rois == 'lh.'+roi)[0][0])
            except:
                pass
        
        # detect bad ROIs based on standard devation activations across trials
        wrong_rois = np.zeros((len(self.subjects), len(rois)), dtype=bool)
        for i in range(len(self.subjects)):
            C = np.vstack([np.array(df[(df.subject == self.subjects[i])][roi]) for roi in rois]).T
            sdts = np.std(C, axis=0)
            wrong_rois[i,:] = myfuncs.is_outlier(sdts, thresh=3.5)
        bad_rois = rois[(np.sum(wrong_rois, axis=0) > 4)]
        for r in bad_rois:
            hemi, roi = r.split('.')
            if hemi == 'rh':
                bad_rois = np.concatenate((bad_rois, np.array(['lh.'+roi])))
            else:
                bad_rois = np.concatenate((bad_rois, np.array(['rh.'+roi])))
            
            try:
                rois = np.delete(rois, np.where(rois == 'rh.'+roi)[0][0])
            except:
                pass
            try:
                rois = np.delete(rois, np.where(rois == 'lh.'+roi)[0][0])
            except:
                pass
        
        # flip rois:
        for i in range(len(self.subjects)):
            if self.subjects[i] in ['sub-03', 'sub-04', 'sub-06', 'sub-08', 'sub-09', 'sub-11', 'sub-13']:
                ind = np.array(df.subject == self.subjects[i])
                for roi in rois:
                    if roi.split('.')[0] == 'lh':
                        df.loc[ind,['lh.{}'.format(roi.split('.')[1]), 'rh.{}'.format(roi.split('.')[1])]] = df.loc[ind,['rh.{}'.format(roi.split('.')[1]), 'lh.{}'.format(roi.split('.')[1])]].values
        
        shell()
              
        # compute boundaries:
        boundaries = np.array([
            np.where(rois == 'lh.Lat_Fis-post')[0][0],
            np.where(rois == 'lh.S_parieto_occipital')[0][0],
            np.where(rois == 'lh.S_circular_insula_sup')[0][0],
            np.where(rois == 'lh.S_pericallosal')[0][0],
            np.where(rois == 'lh.G_front_sup')[0][0],
            np.where(rois == 'rh.Pole_occipital')[0][0],
            np.where(rois == 'rh.Lat_Fis-post')[0][0],
            np.where(rois == 'rh.S_parieto_occipital')[0][0],
            np.where(rois == 'rh.S_circular_insula_sup')[0][0],
            np.where(rois == 'rh.S_pericallosal')[0][0],
            np.where(rois == 'rh.G_front_sup')[0][0],])
        
        # plot histogram of mean activations:
        C = np.zeros((len(self.subjects),len(rois)))
        for i in range(len(self.subjects)):
            C[i,:] = np.vstack([np.array(df[(df.subject == self.subjects[i])][roi]) for roi in rois]).T.mean(axis=0)
        C = C.mean(axis=0)
        fig = plt.figure(figsize=(3,3))
        plt.hist(C, bins=20)
        plt.xlabel('fMRI response\n(%signal change)')
        plt.ylabel('number of ROIs')
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder, 'noise_correlation', 'whole_brain', 'hist_activations_{}.pdf'.format(self.split_by)))
        
        contrast_across_dvs = []
        # for dv in ['cor', 'mean', 'var', 'cov', 'snr']:
        for dv in ['cor',]:
            cormats = []
            for condition in [np.concatenate(self.pupil_h_ind), np.concatenate(self.pupil_l_ind)]:
                for signal_present in [np.concatenate(self.present), ~np.concatenate(self.present)]:
                    ind = condition * signal_present
                    cm = np.zeros((len(rois), len(rois), len(self.subjects)))
                    cm_p = np.zeros((len(rois), len(rois), len(self.subjects)))
                    for i in range(len(self.subjects)):
                        # if self.subjects[i] in ['sub-03', 'sub-04', 'sub-06', 'sub-08', 'sub-09', 'sub-11', 'sub-13']:
                        C = np.vstack([np.array(df[(df.subject == self.subjects[i])*ind][roi]) for roi in rois]).T
                        # else:
                            # rois_rev = np.concatenate((rois[56:], rois[:56]))
                            # C = np.vstack([np.array(data_frame[(data_frame.subject == self.subjects[i])*ind][roi]) for roi in rois_rev]).T
                        if dv == 'cor':
                            if partial:
                                cm[:,:,i], cm_p[:,:,i] = myfuncs.corr_matrix_partial(C)
                            else:
                                cm[:,:,i], cm_p[:,:,i] = myfuncs.corr_matrix(C, dv='cor')
                        if dv == 'var':
                            cm[:,:,i], cm_p[:,:,i] = myfuncs.corr_matrix(C, dv='var')
                        if dv == 'cov':
                            cm[:,:,i], cm_p[:,:,i] = myfuncs.corr_matrix(C, dv='cov')
                        if dv == 'mean':
                            cm[:,:,i], cm_p[:,:,i] = myfuncs.corr_matrix(C, dv='mean')
                        if dv == 'snr':
                            cm[:,:,i], cm_p[:,:,i] = myfuncs.corr_matrix(C, dv='snr')
                    cormats.append(cm)
                    
            corrmats_mean = (cormats[0]+cormats[1]+cormats[2]+cormats[3]) / 4.0
            corrmats_mean_av = corrmats_mean.mean(axis=-1)
            corrmats_contrast = ((cormats[0]-cormats[2]) + (cormats[1]-cormats[3])) / 2.0
            corrmats_contrast_av = corrmats_contrast.mean(axis=-1)

            contrast_across_dvs.append(corrmats_contrast_av)

            p_mat = np.zeros(corrmats_mean_av.shape)
            for i in range(p_mat.shape[0]):
                for j in range(p_mat.shape[1]):
                    try:
                        p_mat[i,j] = sp.stats.wilcoxon(corrmats_contrast[i,j,:])[1]
                    except:
                        p_mat[i,j] = 1

            # # fdr correction:
            # p_mat = mne.stats.fdr_correction(p_mat, 0.10)[1]

            # plot matrix:
            mask =  np.tri(corrmats_mean_av.shape[0], k=0)
            corrmat_m = np.ma.masked_where(mask, corrmats_mean_av)
            p_mat_m = np.ma.masked_where(mask.T, p_mat)
            roi_names = rois
            fig = plt.figure(figsize=(5,4))
            ax = fig.add_subplot(111)
            vmax = max(abs(corrmats_mean_av.ravel()))
            if (dv == 'cor') or (dv == 'mean') or (dv == 'snr'):
                im = ax.pcolormesh(corrmat_m, cmap='bwr', vmin=-vmax, vmax=vmax)
            else:
                im = ax.pcolormesh(corrmat_m, cmap='Reds', vmin=0, vmax=vmax)
            ax.set_xlim(xmax=len(rois))
            ax.set_ylim(ymax=len(rois))
            ax.set_yticks(arange(0.5,len(rois)+.5))
            ax.set_xticks(arange(0.5,len(rois)+.5))
            ax.set_yticklabels(roi_names)
            ax.set_xticklabels(roi_names, rotation=270)
            ax.patch.set_hatch('x')
            ax.set_aspect(1)
            fig.colorbar(im)
            for l in boundaries:
                plt.vlines(l, 0, l, lw=1)
                plt.hlines(l, l, corrmat_m.shape[0], lw=1)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, 'noise_correlation', 'whole_brain', 'matrix_single_{}_{}_{}.pdf'.format(self.split_by, 'all', dv,)))

            # contrast matrix:
            mask =  np.tri(corrmats_contrast_av.shape[0], k=0)
            corrmat_m = np.ma.masked_where(mask, corrmats_contrast_av)
            p_mat_m = np.ma.masked_where(mask.T, p_mat)
            roi_names = rois
            fig = plt.figure(figsize=(5,4))
            # fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(111)
            vmax = max(abs(corrmats_contrast_av.ravel()))
            im = ax.pcolormesh(corrmat_m, cmap='bwr', vmin=-vmax, vmax=vmax)
            ax.set_xlim(xmax=len(rois))
            ax.set_ylim(ymax=len(rois))
            ax.set_yticks(arange(0.5,len(rois)+.5))
            ax.set_xticks(arange(0.5,len(rois)+.5))
            ax.set_yticklabels(roi_names)
            ax.set_xticklabels(roi_names, rotation=270)
            ax.patch.set_hatch('x')
            ax.set_aspect(1)
            fig.colorbar(im)
            for l in boundaries:
                plt.vlines(l, 0, l, lw=1)
                plt.hlines(l, l, corrmat_m.shape[0], lw=1)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, 'noise_correlation', 'whole_brain', 'matrix_single_{}_{}_{}.pdf'.format(self.split_by, 'contrast', dv,)))

            mask = (p_mat > 0.05)+np.array(np.tri(corrmats_contrast_av.shape[0], k=0), dtype=bool)
            corrmat_m = np.ma.masked_where(mask, corrmats_contrast_av)

            fig = plt.figure(figsize=(5,4))
            ax = fig.add_subplot(111)
            vmax = max(abs(corrmats_contrast_av.ravel()))
            im = ax.pcolormesh(corrmat_m, cmap='bwr', vmin=-vmax, vmax=vmax)
            ax.set_xlim(xmax=len(rois))
            ax.set_ylim(ymax=len(rois))
            ax.set_yticks(arange(0.5,len(rois)+.5))
            ax.set_xticks(arange(0.5,len(rois)+.5))
            ax.set_yticklabels(roi_names)
            ax.set_xticklabels(roi_names, rotation=270)
            ax.patch.set_hatch('x')
            ax.set_aspect(1)
            fig.colorbar(im)
            for l in boundaries:
                plt.vlines(l, 0, l, lw=1)
                plt.hlines(l, l, corrmat_m.shape[0], lw=1)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, 'noise_correlation', 'whole_brain', 'matrix_single_{}_{}_{}_significant.pdf'.format(self.split_by, 'contrast', dv,)))

            # roi_data = pd.read_csv('/home/shared/Niels_UvA/brainstem_masks/destrieux/rois.csv')
            # x = (np.array([float(roi_data.x[np.array(roi_data.roi == roi)]) for roi in rois]) - 45) * 2.0
            # y = (np.array([float(roi_data.y[np.array(roi_data.roi == roi)]) for roi in rois]) - 54) * 2.0
            # z = (np.array([float(roi_data.z[np.array(roi_data.roi == roi)]) for roi in rois]) - 22.5) * 2.0
            #
            # coords = np.vstack((x, y, z)).T
            
            
            # collapse across brain region:
            
            boundaries2 = np.concatenate((np.array([0]), boundaries, np.array([len(rois)])))
            corrmats_contrast_av2 = corrmats_contrast_av.copy()
            for b in range(len(boundaries2)-1):
                for bb in range(len(boundaries2)-1):
                    corrmats_contrast_av2[boundaries2[b]:boundaries2[b+1],boundaries2[bb]:boundaries2[bb+1]] = corrmats_contrast_av2[boundaries2[b]:boundaries2[b+1],boundaries2[bb]:boundaries2[bb+1]].mean()
            
            
            
            mask =  np.tri(corrmats_contrast_av2.shape[0], k=0)
            corrmat_m = np.ma.masked_where(mask, corrmats_contrast_av2)
            p_mat_m = np.ma.masked_where(mask.T, p_mat)
            roi_names = rois
            fig = plt.figure(figsize=(5,4))
            # fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(111)
            vmax = max(abs(corrmats_contrast_av2.ravel()))
            im = ax.pcolormesh(corrmat_m, cmap='bwr', vmin=-vmax, vmax=vmax)
            ax.set_xlim(xmax=len(rois))
            ax.set_ylim(ymax=len(rois))
            ax.set_yticks(arange(0.5,len(rois)+.5))
            ax.set_xticks(arange(0.5,len(rois)+.5))
            ax.set_yticklabels(roi_names)
            ax.set_xticklabels(roi_names, rotation=270)
            ax.patch.set_hatch('x')
            ax.set_aspect(1)
            fig.colorbar(im)
            for l in boundaries:
                plt.vlines(l, 0, l, lw=1)
                plt.hlines(l, l, corrmat_m.shape[0], lw=1)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, 'noise_correlation', 'whole_brain', 'matrix_single_{}_{}_{}_mean.pdf'.format(self.split_by, 'contrast', dv,)))
            
            
            
            
            # stats on cells:
            nr_pos_cells = np.zeros(len(self.subjects))
            nr_neg_cells = np.zeros(len(self.subjects))
            for i in range(len(self.subjects)):
                nr_pos_cells[i] = np.sum(corrmats_contrast[:,:,i].ravel()>0) / 2.0
                nr_neg_cells[i] = np.sum(corrmats_contrast[:,:,i].ravel()<0) / 2.0

            nr_pos_cells = nr_pos_cells / ((len(rois)*len(rois)) / 2) * 100
            nr_neg_cells = nr_neg_cells / ((len(rois)*len(rois)) / 2) * 100

            MEANS = (nr_pos_cells.mean(), nr_neg_cells.mean())
            SEMS = (sp.stats.sem(nr_pos_cells), sp.stats.sem(nr_neg_cells))
            N = 2
            ind = np.linspace(0,N/2,N)
            bar_width = 0.50
            fig = plt.figure(figsize=(1.25,1.75))
            ax = fig.add_subplot(111)
            for i in range(N):
                ax.bar(ind[i], height=MEANS[i], width = bar_width, yerr = SEMS[i], color = ['r','b'][i], alpha = 1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
            ax.set_title('N={}'.format(len(self.subjects)), size=7)
            ax.set_ylabel('number of cells', size=7)
            ax.set_xticks( (ind) )
            ax.set_xticklabels( ('pos', 'neg') )
            plt.text(x=np.mean((ind[0], ind[1])), y=ax.axis()[3] - ((ax.axis()[3]-ax.axis()[2]) / 10.0), s='p = {}'.format(round(myfuncs.permutationTest(nr_pos_cells, nr_neg_cells)[1],3)), horizontalalignment='center')
            ax.set_ylim(ymax=75)
            locator = plt.MaxNLocator(nbins=3) # with 3 bins you will have 4 ticks
            ax.yaxis.set_major_locator(locator)
            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, 'noise_correlation', 'whole_brain', 'nr_cells_{}_{}_{}.pdf'.format(self.split_by, 'contrast', dv,)))

            # plot correlation to criterion:
            fig = plt.figure(figsize=(2,2))
            ax = fig.add_subplot(111)
            myfuncs.correlation_plot(nr_pos_cells, self.criterion_hi-self.criterion_lo, line=True, ax=ax)
            sns.despine(offset=10, trim=True)
            ax.set_xlabel('Number of increases')
            ax.set_ylabel('Drift criterion')
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, 'noise_correlation', 'whole_brain', 'nr_cells_drift_criterion_{}_{}_{}_.pdf'.format(self.split_by, 'contrast', dv,)))
            
            if dv == 'cor':
                
                # plot connectome:
                # ----------------
                
                # shell()
                
                import copy
                from nilearn import image
                from nilearn import plotting

                atlas_rois = np.zeros((len(rois), 91, 109, 91))
                for i, roi in enumerate(rois):
                    atlas_rois[i,:,:,:] = NiftiImage(os.path.join(self.mask_folder, 'destrieux/' + roi + '.nii.gz')).data

                atlas_rois = NiftiImage(atlas_rois)
                atlas_rois.header = NiftiImage(os.path.join(self.mask_folder, 'MNI152_T1_2mm_brain_mask_bin.nii.gz')).header
                atlas_rois.save(os.path.join(os.path.join(self.mask_folder, 'destrieux', "atlas.nii.gz")))

                atlas = nib.load(os.path.join(os.path.join(self.mask_folder, 'destrieux', "atlas.nii.gz")))
                atlas_imgs = image.iter_img(atlas)
                atlas_region_coords = [nilearn.plotting.find_xyz_cut_coords(img) for img in atlas_imgs]
                
                dummy_file1 = os.path.join('/home/shared/UvA/Niels_UvA/Visual_UvA/surface_plots', "dummy1.nii.gz")
                dummy_file2 = os.path.join('/home/shared/UvA/Niels_UvA/Visual_UvA/surface_plots', "dummy2.nii.gz")
                
                contrast_mat_m = copy.copy(corrmats_contrast_av)
                contrast_mat_m[contrast_mat_m<0] = 0
                cut_off = np.sort(np.ravel(p_mat[corrmats_contrast_av>0]))[~np.isnan(np.sort(np.ravel(p_mat[corrmats_contrast_av>0])))][100]
                # cut_off = 0.05
                contrast_mat_m[p_mat>cut_off] = 0
                
                filename=os.path.join(self.figure_folder, 'noise_correlation', 'whole_brain', 'connectome_pos_{}.pdf'.format(self.split_by))
                plotting.plot_connectome(contrast_mat_m, atlas_region_coords, edge_threshold=0, colorbar=True, output_file=filename)
                
                contrast_mat_m = copy.copy(corrmats_contrast_av)
                contrast_mat_m[contrast_mat_m>0] = 0
                cut_off = np.sort(np.ravel(p_mat[corrmats_contrast_av<0]))[~np.isnan(np.sort(np.ravel(p_mat[corrmats_contrast_av<0])))][100]
                # cut_off = 0.05
                contrast_mat_m[p_mat>cut_off] = 0
                filename=os.path.join(self.figure_folder, 'noise_correlation', 'whole_brain', 'connectome_neg_{}.pdf'.format(self.split_by))
                plotting.plot_connectome(contrast_mat_m, atlas_region_coords, edge_threshold=0, colorbar=True, output_file=filename)
                
                
                shell()
                
                
                lr = np.zeros((len(rois), len(rois)))
                fb = np.zeros((len(rois), len(rois)))
                up = np.zeros((len(rois), len(rois)))
                
                for r1 in range(len(rois)):
                    for r2 in range(len(rois)):
                       lr[r1,r2] = abs(atlas_region_coords[r1][0] - atlas_region_coords[r2][0])
                       fb[r1,r2] = abs(atlas_region_coords[r1][1] - atlas_region_coords[r2][1])
                       up[r1,r2] = abs(atlas_region_coords[r1][2] - atlas_region_coords[r2][2])
                
                lr = lr * abs(corrmats_contrast_av)
                fb = fb * abs(corrmats_contrast_av)
                up = up * abs(corrmats_contrast_av)
                
                lr[corrmats_contrast_av<0].mean()
                        
                
                
                
                
                
                
                
                
                
                
        
        # specific to pos or neg voxels...:
        # ---------------------------------

        pos_indices = contrast_across_dvs[0] > 0
        neg_indices = contrast_across_dvs[0] < 0
        # for dv, yl in zip(['mean', 'var', 'cov', 'snr'], [0.4, 7, 2.5, 0.25]):
        for dv, yl in zip(['snr',], [0.25,]):
            cormats = []
            for condition in [np.concatenate(self.pupil_h_ind), np.concatenate(self.pupil_l_ind)]:
                for signal_present in [np.concatenate(self.present), ~np.concatenate(self.present)]:
                    ind = condition * signal_present
                    cm = np.zeros((len(rois), len(rois), len(self.subjects)))
                    cm_p = np.zeros((len(rois), len(rois), len(self.subjects)))
                    for i in range(len(self.subjects)):
                        C = np.vstack([np.array(df[(df.subject == self.subjects[i])*ind][roi]) for roi in rois]).T
                        if dv == 'cor':
                            if partial:
                                cm[:,:,i], cm_p[:,:,i] = myfuncs.corr_matrix_partial(C)
                            else:
                                cm[:,:,i], cm_p[:,:,i] = myfuncs.corr_matrix(C, dv='cor')
                        if dv == 'var':
                            cm[:,:,i], cm_p[:,:,i] = myfuncs.corr_matrix(C, dv='var')
                        if dv == 'cov':
                            cm[:,:,i], cm_p[:,:,i] = myfuncs.corr_matrix(C, dv='cov')
                        if dv == 'mean':
                            cm[:,:,i], cm_p[:,:,i] = myfuncs.corr_matrix(C, dv='mean')
                        if dv == 'snr':
                            cm[:,:,i], cm_p[:,:,i] = myfuncs.corr_matrix(C, dv='snr')
                    cormats.append(cm)
            corrmats_mean = (cormats[0]+cormats[1]+cormats[2]+cormats[3]) / 4.0
            corrmats_mean_av = corrmats_mean.mean(axis=-1)
            corrmats_contrast = ((cormats[0]-cormats[2]) + (cormats[1]-cormats[3])) / 2.0
            corrmats_contrast_av = corrmats_contrast.mean(axis=-1)

            mask =  np.array(np.tri(corrmats_contrast_av.shape[0], k=0), dtype=bool)

            pos = np.array([corrmats_mean[pos_indices & mask,i].mean() for i in range(len(self.subjects))])
            neg = np.array([corrmats_mean[neg_indices & mask,i].mean() for i in range(len(self.subjects))])
            values = [pos, neg]
            MEANS = np.array([np.mean(v) for v in values])
            SEMS = np.array([sp.stats.sem(v) for v in values])
            p = [sp.stats.ttest_rel(pos, neg)[1]]
            x = np.arange(0,len(MEANS))
            bar_width = 0.75
            fig = plt.figure(figsize=(1.25,1.75))
            ax = plt.subplot(111)
            my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}
            ax.bar(x, MEANS, yerr=SEMS, width=bar_width, color=['orange','green'], alpha=1, align='center', edgecolor = 'k', linewidth = 0, capsize = 0, ecolor = 'k', error_kw={'elinewidth':0.5,})
            ax.tick_params(axis='y', which='major', labelsize=6)
            ax.set_xticks(x)
            for i, pp in enumerate(p):
                star1 = 'n.s.'
                if pp < 0.05:
                    star1 = '*'
                if pp < 0.01:
                    star1 = '**'
                if pp < 0.001:
                    star1 = '***'
                # ax.text(s=star1, x=ind[i], y=ax.get_ylim()[1]-((ax.get_ylim()[1] - ax.get_ylim()[0]) / 10.0), size=10)
                ax.text(s=str(round(pp,3)), x=x[i], y=ax.get_ylim()[1]-((ax.get_ylim()[1] - ax.get_ylim()[0]) / 10.0), size=7)
            # ax.set_ylim(0,yl)
            ax.set_ylabel(dv, size=7)
            sns.despine(offset=10, trim=True)
            ax.set_xticklabels(['pos', 'neg'], rotation=45, size=7)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, 'noise_correlation', 'whole_brain', 'bars_{}_{}_{}.pdf'.format('cor', dv, self.split_by)))

        
        
        
        
    def WHOLEBRAIN_noise_correlation_old(self, data_type, partial=False,):
        
        self.postfix = '_d'
        time_locked = 'stim_locked'
        self.make_pupil_BOLD_dataframe(data_type=data_type, time_locked=time_locked, regress_iti=False, regress_rt=True, regress_stimulus=True,)
        
        rois_list = [
                    # PARCELATION LEFT:
                    # -----------------
                    
                    [
                    # visual:
                    'lh.Pole_occipital',
                    'lh.G_occipital_sup',
                    'lh.S_oc_sup_and_transversal',
                    'lh.G_occipital_middle',
                    'lh.S_occipital_ant',
                    'lh.S_oc_middle_and_Lunatus',
                    'lh.G_and_S_occipital_inf',
                    'lh.S_collat_transv_post',
                    'lh.G_oc-temp_med-Lingual',
                    'lh.S_calcarine',
                    'lh.G_cuneus',
                    ],
                    [
                    # temporal:
                    'lh.Lat_Fis-post',
                    'lh.G_temp_sup-Plan_tempo',
                    'lh.S_temporal_transverse',
                    'lh.G_temp_sup-G_T_transv',
                    'lh.G_temp_sup-Lateral',
                    'lh.S_temporal_sup',
                    'lh.G_temporal_middle',
                    'lh.S_temporal_inf',
                    'lh.G_temporal_inf',
                    # 'lh.S_oc-temp_lat',
                    'lh.G_oc-temp_med-Parahip',
                    'lh.S_collat_transv_ant',
                    'lh.G_oc-temp_lat-fusifor',
                    'lh.S_oc-temp_med_and_Lingual',
                    'lh.G_temp_sup-Plan_polar',
                    'lh.Pole_temporal',
                    ],
                    [
                    # parietal:
                    'lh.S_parieto_occipital',
                    'lh.S_subparietal',
                    'lh.G_precuneus',
                    'lh.G_parietal_sup',
                    'lh.S_intrapariet_and_P_trans',
                    'lh.G_pariet_inf-Angular',
                    'lh.S_interm_prim-Jensen',
                    'lh.G_and_S_paracentral',
                    'lh.S_postcentral',
                    'lh.G_postcentral',
                    'lh.S_central',
                    'lh.G_pariet_inf-Supramar',
                    'lh.G_and_S_subcentral',
                    ],
                    [
                    # insular:
                    'lh.S_circular_insula_sup',
                    'lh.G_insular_short',
                    'lh.S_circular_insula_inf',
                    'lh.G_Ins_lg_and_S_cent_ins',
                    'lh.S_circular_insula_ant',
                    ],
                    [
                    # cingulate:
                    'lh.G_cingul-Post-ventral',
                    'lh.S_pericallosal',
                    'lh.G_cingul-Post-dorsal',
                    'lh.S_cingul-Marginalis',
                    'lh.G_and_S_cingul-Mid-Post',
                    'lh.G_and_S_cingul-Mid-Ant',
                    'lh.G_and_S_cingul-Ant',
                    ],
                    [
                    # frontal:
                    # 'lh.G_precentral',
                    'lh.S_precentral-sup-part',
                    'lh.S_precentral-inf-part',
                    'lh.G_front_sup',
                    'lh.S_front_sup',
                    'lh.G_front_middle',
                    # 'lh.S_front_middle',
                    'lh.S_front_inf',
                    'lh.G_front_inf-Opercular',
                    'lh.G_front_inf-Triangul',
                    'lh.S_orbital_lateral',
                    'lh.Lat_Fis-ant-Horizont',
                    'lh.Lat_Fis-ant-Vertical',
                    'lh.G_front_inf-Orbital',
                    'lh.G_and_S_transv_frontopol',
                    'lh.G_and_S_frontomargin',
                    'lh.G_orbital',
                    'lh.S_orbital-H_Shaped',
                    'lh.S_orbital_med-olfact',
                    'lh.G_rectus',
                    'lh.S_suborbital',
                    'lh.G_subcallosal',
                    ],
                    # PARCELATION RIGHT:
                    # -----------------
                    [
                    # visual:
                    'rh.Pole_occipital',
                    'rh.G_occipital_sup',
                    'rh.S_oc_sup_and_transversal',
                    'rh.G_occipital_middle',
                    'rh.S_occipital_ant',
                    'rh.S_oc_middle_and_Lunatus',
                    'rh.G_and_S_occipital_inf',
                    'rh.S_collat_transv_post',
                    'rh.G_oc-temp_med-Lingual',
                    'rh.S_calcarine',
                    'rh.G_cuneus',
                    ],
                    [
                    # temporal:
                    'rh.Lat_Fis-post',
                    'rh.G_temp_sup-Plan_tempo',
                    'rh.S_temporal_transverse',
                    'rh.G_temp_sup-G_T_transv',
                    'rh.G_temp_sup-Lateral',
                    'rh.S_temporal_sup',
                    'rh.G_temporal_middle',
                    'rh.S_temporal_inf',
                    'rh.G_temporal_inf',
                    # 'rh.S_oc-temp_lat',
                    'rh.G_oc-temp_med-Parahip',
                    'rh.S_collat_transv_ant',
                    'rh.G_oc-temp_lat-fusifor',
                    'rh.S_oc-temp_med_and_Lingual',
                    'rh.G_temp_sup-Plan_polar',
                    'rh.Pole_temporal',
                    ],
                    [
                    # parietal:
                    'rh.S_parieto_occipital',
                    'rh.S_subparietal',
                    'rh.G_precuneus',
                    'rh.G_parietal_sup',
                    'rh.S_intrapariet_and_P_trans',
                    'rh.G_pariet_inf-Angular',
                    'rh.S_interm_prim-Jensen',
                    'rh.G_and_S_paracentral',
                    'rh.S_postcentral',
                    'rh.G_postcentral',
                    'rh.S_central',
                    'rh.G_pariet_inf-Supramar',
                    'rh.G_and_S_subcentral',
                    ],
                    [
                    # insular:
                    'rh.S_circular_insula_sup',
                    'rh.G_insular_short',
                    'rh.S_circular_insula_inf',
                    'rh.G_Ins_lg_and_S_cent_ins',
                    'rh.S_circular_insula_ant',
                    ],
                    [
                    # cingulate:
                    'rh.G_cingul-Post-ventral',
                    'rh.S_pericallosal',
                    'rh.G_cingul-Post-dorsal',
                    'rh.S_cingul-Marginalis',
                    'rh.G_and_S_cingul-Mid-Post',
                    'rh.G_and_S_cingul-Mid-Ant',
                    'rh.G_and_S_cingul-Ant',
                    ],
                    [
                    # frontal:
                    # 'rh.G_precentral',
                    'rh.S_precentral-sup-part',
                    'rh.S_precentral-inf-part',
                    'rh.G_front_sup',
                    'rh.S_front_sup',
                    'rh.G_front_middle',
                    # 'rh.S_front_middle',
                    'rh.S_front_inf',
                    'rh.G_front_inf-Opercular',
                    'rh.G_front_inf-Triangul',
                    'rh.S_orbital_lateral',
                    'rh.Lat_Fis-ant-Horizont',
                    'rh.Lat_Fis-ant-Vertical',
                    'rh.G_front_inf-Orbital',
                    'rh.G_and_S_transv_frontopol',
                    'rh.G_and_S_frontomargin',
                    'rh.G_orbital',
                    'rh.S_orbital-H_Shaped',
                    'rh.S_orbital_med-olfact',
                    'rh.G_rectus',
                    'rh.S_suborbital',
                    'rh.G_subcallosal',
                    ]
                ]
        
        rois = np.concatenate(rois_list)
        
        wrong_rois = np.zeros(len(rois))
        for i in range(len(self.subjects)):
            C = np.vstack([np.array(self.data_frame[(self.data_frame.subject == self.subjects[i])][roi+self.postfix]) for roi in rois]).T
            
            sdts = np.std(C[:,:], axis=0)
            wrong_rois = wrong_rois = myfuncs.is_outlier(sdts)
            
            # median_sdt = np.median(np.std(C[:,0], axis=0))
            # wrong_rois = wrong_rois + np.array(np.std(C, axis=0) > (5*median_sdt), dtype=bool)
            # wrong_rois = wrong_rois + np.array(np.array((C>75)+(C<-75)).sum(axis=0), dtype=bool)
        clean_rois = (wrong_rois == 0)
        # clean_rois[:71] = clean_rois[:71] * clean_rois[71:]
        # clean_rois[71:] = clean_rois[:71] * clean_rois[71:]
        wrong_rois = rois[wrong_rois]
        rois = rois[clean_rois]
        
        # compute boundaries:
        for region in range(len(rois_list)):
            for r in wrong_rois:
                try:
                    rois_list[region].remove(r)
                except:
                     pass
        boundaries = np.cumsum([len(region) for region in rois_list])
        boundaries = boundaries[:-1]
        
        # plot histogram of mean activations:
        C = np.zeros((len(self.subjects),len(rois)))
        for i in range(len(self.subjects)):
            C[i,:] = np.vstack([np.array(self.data_frame[(self.data_frame.subject == self.subjects[i])][roi+self.postfix]) for roi in rois]).T.mean(axis=0)
        C = C.mean(axis=0)
        fig = plt.figure(figsize=(3,3))
        plt.hist(C, bins=20)
        plt.xlabel('fMRI response\n(%signal change)')
        plt.ylabel('number of ROIs')
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder, 'noise_correlation', 'whole_brain', 'hist_activations_{}.pdf'.format(self.split_by)))
        
        contrast_across_dvs = []
        # for dv in ['cor', 'mean', 'var', 'cov', 'snr']:
        for dv in ['cor',]:
            cormats = []
            for condition in [np.concatenate(self.pupil_h_ind), np.concatenate(self.pupil_l_ind)]:
                for signal_present in [np.concatenate(self.present), ~np.concatenate(self.present)]:
                    ind = condition * signal_present
                    cm = np.zeros((len(rois), len(rois), len(self.subjects)))
                    cm_p = np.zeros((len(rois), len(rois), len(self.subjects)))
                    for i in range(len(self.subjects)):
                        
                        if self.subjects[i] in ['sub-03', 'sub-04', 'sub-06', 'sub-08', 'sub-09', 'sub-11', 'sub-13']:
                            C = np.vstack([np.array(self.data_frame[(self.data_frame.subject == self.subjects[i])*ind][roi+self.postfix]) for roi in rois]).T
                        else:
                            rois_rev = np.concatenate((rois[56:], rois[:56]))
                            C = np.vstack([np.array(self.data_frame[(self.data_frame.subject == self.subjects[i])*ind][roi+self.postfix]) for roi in rois_rev]).T
                        
                        if dv == 'cor':
                            if partial:
                                cm[:,:,i], cm_p[:,:,i] = myfuncs.corr_matrix_partial(C)
                            else:
                                cm[:,:,i], cm_p[:,:,i] = myfuncs.corr_matrix(C, dv='cor')
                        if dv == 'var':
                            cm[:,:,i], cm_p[:,:,i] = myfuncs.corr_matrix(C, dv='var')
                        if dv == 'cov':
                            cm[:,:,i], cm_p[:,:,i] = myfuncs.corr_matrix(C, dv='cov')
                        if dv == 'mean':
                            cm[:,:,i], cm_p[:,:,i] = myfuncs.corr_matrix(C, dv='mean')
                        if dv == 'snr':
                            cm[:,:,i], cm_p[:,:,i] = myfuncs.corr_matrix(C, dv='snr')
                    cormats.append(cm)

            corrmats_mean = (cormats[0]+cormats[1]+cormats[2]+cormats[3]) / 4.0
            corrmats_mean_av = corrmats_mean.mean(axis=-1)
            corrmats_contrast = ((cormats[0]-cormats[2]) + (cormats[1]-cormats[3])) / 2.0
            corrmats_contrast_av = corrmats_contrast.mean(axis=-1)

            contrast_across_dvs.append(corrmats_contrast_av)

            p_mat = np.zeros(corrmats_mean_av.shape)
            for i in range(p_mat.shape[0]):
                for j in range(p_mat.shape[1]):
                    try:
                        p_mat[i,j] = sp.stats.wilcoxon(corrmats_contrast[i,j,:])[1]
                    except:
                        p_mat[i,j] = 1

            # # fdr correction:
            # p_mat = mne.stats.fdr_correction(p_mat, 0.10)[1]

            # plot matrix:
            mask =  np.tri(corrmats_mean_av.shape[0], k=0)
            corrmat_m = np.ma.masked_where(mask, corrmats_mean_av)
            p_mat_m = np.ma.masked_where(mask.T, p_mat)
            roi_names = rois
            fig = plt.figure(figsize=(5,4))
            ax = fig.add_subplot(111)
            if dv == 'mean':
                vmax = 1.0
            else:
                vmax = max(abs(corrmats_mean_av.ravel()))
            if (dv == 'cor') or (dv == 'mean') or (dv == 'snr'):
                im = ax.pcolormesh(corrmat_m, cmap='bwr', vmin=-vmax, vmax=vmax)
            else:
                im = ax.pcolormesh(corrmat_m, cmap='Reds', vmin=0, vmax=vmax)
            ax.set_xlim(xmax=len(rois))
            ax.set_ylim(ymax=len(rois))
            ax.set_yticks(arange(0.5,len(rois)+.5))
            ax.set_xticks(arange(0.5,len(rois)+.5))
            ax.set_yticklabels(roi_names)
            ax.set_xticklabels(roi_names, rotation=270)
            ax.patch.set_hatch('x')
            ax.set_aspect(1)
            fig.colorbar(im)
            for l in boundaries:
                plt.vlines(l, 0, l, lw=1)
                plt.hlines(l, l, corrmat_m.shape[0], lw=1)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, 'noise_correlation', 'whole_brain', 'matrix_single_{}_{}_{}.pdf'.format(self.split_by, 'all', dv,)))

            # contrast matrix:
            mask =  np.tri(corrmats_contrast_av.shape[0], k=0)
            corrmat_m = np.ma.masked_where(mask, corrmats_contrast_av)
            p_mat_m = np.ma.masked_where(mask.T, p_mat)
            roi_names = rois
            fig = plt.figure(figsize=(5,4))
            # fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(111)
            vmax = max(abs(corrmats_contrast_av.ravel()))
            im = ax.pcolormesh(corrmat_m, cmap='bwr', vmin=-vmax, vmax=vmax)
            ax.set_xlim(xmax=len(rois))
            ax.set_ylim(ymax=len(rois))
            ax.set_yticks(arange(0.5,len(rois)+.5))
            ax.set_xticks(arange(0.5,len(rois)+.5))
            ax.set_yticklabels(roi_names)
            ax.set_xticklabels(roi_names, rotation=270)
            ax.patch.set_hatch('x')
            ax.set_aspect(1)
            fig.colorbar(im)
            for l in boundaries:
                plt.vlines(l, 0, l, lw=1)
                plt.hlines(l, l, corrmat_m.shape[0], lw=1)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, 'noise_correlation', 'whole_brain', 'matrix_single_{}_{}_{}.pdf'.format(self.split_by, 'contrast', dv,)))

            mask = (p_mat > 0.05)+np.array(np.tri(corrmats_contrast_av.shape[0], k=0), dtype=bool)
            corrmat_m = np.ma.masked_where(mask, corrmats_contrast_av)

            fig = plt.figure(figsize=(5,4))
            ax = fig.add_subplot(111)
            vmax = max(abs(corrmats_contrast_av.ravel()))
            im = ax.pcolormesh(corrmat_m, cmap='bwr', vmin=-vmax, vmax=vmax)
            ax.set_xlim(xmax=len(rois))
            ax.set_ylim(ymax=len(rois))
            ax.set_yticks(arange(0.5,len(rois)+.5))
            ax.set_xticks(arange(0.5,len(rois)+.5))
            ax.set_yticklabels(roi_names)
            ax.set_xticklabels(roi_names, rotation=270)
            ax.patch.set_hatch('x')
            ax.set_aspect(1)
            fig.colorbar(im)
            for l in boundaries:
                plt.vlines(l, 0, l, lw=1)
                plt.hlines(l, l, corrmat_m.shape[0], lw=1)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, 'noise_correlation', 'whole_brain', 'matrix_single_{}_{}_{}_significant.pdf'.format(self.split_by, 'contrast', dv,)))

            # roi_data = pd.read_csv('/home/shared/Niels_UvA/brainstem_masks/destrieux/rois.csv')
            # x = (np.array([float(roi_data.x[np.array(roi_data.roi == roi)]) for roi in rois]) - 45) * 2.0
            # y = (np.array([float(roi_data.y[np.array(roi_data.roi == roi)]) for roi in rois]) - 54) * 2.0
            # z = (np.array([float(roi_data.z[np.array(roi_data.roi == roi)]) for roi in rois]) - 22.5) * 2.0
            #
            # coords = np.vstack((x, y, z)).T

            # stats on cells:
            nr_pos_cells = np.zeros(len(self.subjects))
            nr_neg_cells = np.zeros(len(self.subjects))
            for i in range(len(self.subjects)):
                nr_pos_cells[i] = np.sum(corrmats_contrast[:,:,i].ravel()>0) / 2.0
                nr_neg_cells[i] = np.sum(corrmats_contrast[:,:,i].ravel()<0) / 2.0

            nr_pos_cells = nr_pos_cells / ((len(rois)*len(rois)) / 2) * 100
            nr_neg_cells = nr_neg_cells / ((len(rois)*len(rois)) / 2) * 100

            MEANS = (nr_pos_cells.mean(), nr_neg_cells.mean())
            SEMS = (sp.stats.sem(nr_pos_cells), sp.stats.sem(nr_neg_cells))
            N = 2
            ind = np.linspace(0,N/2,N)
            bar_width = 0.50
            fig = plt.figure(figsize=(1.25,1.75))
            ax = fig.add_subplot(111)
            for i in range(N):
                ax.bar(ind[i], height=MEANS[i], width = bar_width, yerr = SEMS[i], color = ['r','b'][i], alpha = 1, edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
            ax.set_title('N={}'.format(len(self.subjects)), size=7)
            ax.set_ylabel('number of cells', size=7)
            ax.set_xticks( (ind) )
            ax.set_xticklabels( ('pos', 'neg') )
            plt.text(x=np.mean((ind[0], ind[1])), y=ax.axis()[3] - ((ax.axis()[3]-ax.axis()[2]) / 10.0), s='p = {}'.format(round(myfuncs.permutationTest(nr_pos_cells, nr_neg_cells)[1],3)), horizontalalignment='center')
            ax.set_ylim(ymax=75)
            locator = plt.MaxNLocator(nbins=3) # with 3 bins you will have 4 ticks
            ax.yaxis.set_major_locator(locator)
            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, 'noise_correlation', 'whole_brain', 'nr_cells_{}_{}_{}.pdf'.format(self.split_by, 'contrast', dv,)))

            # plot correlation to criterion:
            fig = plt.figure(figsize=(2,2))
            ax = fig.add_subplot(111)
            myfuncs.correlation_plot(nr_pos_cells, self.criterion_hi-self.criterion_lo, line=True, ax=ax)
            sns.despine(offset=10, trim=True)
            ax.set_xlabel('Number of increases')
            ax.set_ylabel('Drift criterion')
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, 'noise_correlation', 'whole_brain', 'nr_cells_drift_criterion_{}_{}_{}_.pdf'.format(self.split_by, 'contrast', dv,)))
            
            if dv == 'cor':
                
                # plot connectome:
                # ----------------
                
                import copy
                from nilearn import image
                from nilearn import plotting

                atlas_rois = np.zeros((len(rois), 91, 109, 91))
                for i, roi in enumerate(rois):
                    atlas_rois[i,:,:,:] = NiftiImage('/home/shared/UvA/Niels_UvA/brainstem_masks/destrieux/' + roi + '.nii.gz').data

                atlas_rois = NiftiImage(atlas_rois)
                atlas_rois.header = NiftiImage('/home/shared/UvA/Niels_UvA/brainstem_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz').header
                atlas_rois.save(os.path.join('/home/shared/UvA/Niels_UvA/brainstem_masks/destrieux', "atlas.nii.gz"))

                atlas = nib.load(os.path.join('/home/shared/UvA/Niels_UvA/brainstem_masks/destrieux', "atlas.nii.gz"))
                atlas_imgs = image.iter_img(atlas)
                atlas_region_coords = [nilearn.plotting.find_xyz_cut_coords(img) for img in atlas_imgs]
                
                # volume_file1 = os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_psc_all.nii.gz'.format('clean_MNI_smooth', 'stim_locked', 'mean', self.split_by))
                # threshold_file1 = NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_psc_all_tfce_corrp_tstat1.nii.gz'.format('clean_MNI_smooth', 'stim_locked', 'mean', self.split_by))).data
                # dummy1 = NiftiImage(volume_file1).data
                # dummy1 = dummy1.mean(axis=0)
                # dummy1[threshold_file1<0.99] = 0
                # dummy1 = NiftiImage(dummy1)
                # dummy1.header = NiftiImage('/home/shared/Niels_UvA/brainstem_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz').header
                # dummy1.save(os.path.join('/home/shared/Niels_UvA/Visual_UvA/surface_plots', "dummy1.nii.gz"))
                dummy_file1 = os.path.join('/home/shared/UvA/Niels_UvA/Visual_UvA/surface_plots', "dummy1.nii.gz")
                #
                # volume_file2 = os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_psc_all_rev.nii.gz'.format('clean_MNI_smooth', 'stim_locked', 'mean', self.split_by))
                # threshold_file2 = NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_psc_all_rev_tfce_corrp_tstat1.nii.gz'.format('clean_MNI_smooth', 'stim_locked', 'mean', self.split_by))).data
                # dummy2 = NiftiImage(volume_file2).data
                # dummy2 = dummy2.mean(axis=0)
                # dummy2[threshold_file2<0.99] = 0
                # dummy2 = NiftiImage(dummy2)
                # dummy2.header = NiftiImage('/home/shared/Niels_UvA/brainstem_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz').header
                # dummy2.save(os.path.join('/home/shared/Niels_UvA/Visual_UvA/surface_plots', "dummy2.nii.gz"))
                dummy_file2 = os.path.join('/home/shared/UvA/Niels_UvA/Visual_UvA/surface_plots', "dummy2.nii.gz")

                # # params:
                # thresh1 = 0.001
                # thresh2 = 1.5
                # percentile = 99
                # strongest = 50 # nr of cells
                #
                # # negative connections:
                # contrast_mat_m = copy.copy(corrmats_contrast_av)
                # contrast_mat_m = contrast_mat_m * -1.0
                # threshold_values = contrast_mat_m.ravel()[-(contrast_mat_m.ravel()<=0)]
                # # threshold = np.percentile(threshold_values, percentile)
                # # threshold = np.sort(threshold_values)[-strongest]
                # # contrast_mat_m[contrast_mat_m<threshold] = 0
                # contrast_mat_m[p_mat>0.01] = 0
                # threshold = min(contrast_mat_m[contrast_mat_m>0])
                # contrast_mat_m[contrast_mat_m<threshold] = 0
                # fig = plt.figure(figsize=(3, 1))
                # display = plotting.plot_glass_brain(dummy_file1, threshold=100000, figure=fig)
                # # display.add_overlay(dummy_file1, threshold=thresh1, colorbar=False, cmap="YlOrRd_r", **{'vmax': thresh2,'alpha': 0.2})
                # # display.add_overlay(dummy_file2, threshold=thresh1, colorbar=False, cmap="Blues_r", **{'vmax': thresh2,'alpha': 0.2})
                # display.add_graph(contrast_mat_m, atlas_region_coords, edge_threshold=threshold, edge_cmap='bwr_r', edge_kwargs={'lw':1}, node_size=7)
                # display.savefig(os.path.join(self.figure_folder, 'noise_correlation', 'whole_brain', 'connectome_neg_{}.pdf'.format(self.split_by)))
                #
                # # positive connections:
                # contrast_mat_m = copy.copy(corrmats_contrast_av)
                # contrast_mat_m = contrast_mat_m
                # threshold_values = contrast_mat_m.ravel()[-(contrast_mat_m.ravel()<=0)]
                # # threshold = np.percentile(threshold_values, percentile)
                # # threshold = np.sort(threshold_values)[-strongest]
                # # contrast_mat_m[contrast_mat_m<threshold] = 0
                # contrast_mat_m[p_mat>0.01] = 0
                # threshold = min(contrast_mat_m[contrast_mat_m>0])
                # contrast_mat_m[contrast_mat_m<threshold] = 0
                # fig = plt.figure(figsize=(3, 1))
                # display = plotting.plot_glass_brain(dummy_file1, threshold=100000, figure=fig)
                # # display.add_overlay(dummy_file1, threshold=thresh1, colorbar=False, cmap="YlOrRd_r", **{'vmax': thresh2,'alpha': 0.2})
                # # display.add_overlay(dummy_file2, threshold=thresh1, colorbar=False, cmap="Blues_r", **{'vmax': thresh2,'alpha': 0.2})
                # display.add_graph(contrast_mat_m, atlas_region_coords, edge_threshold=threshold, edge_cmap='bwr', edge_kwargs={'lw':1}, node_size=7)
                # display.savefig(os.path.join(self.figure_folder, 'noise_correlation', 'whole_brain', 'connectome_pos_{}.pdf'.format(self.split_by)))
                
                
                contrast_mat_m = copy.copy(corrmats_contrast_av)
                contrast_mat_m[contrast_mat_m<0] = 0
                contrast_mat_m[p_mat>0.05] = 0
                filename=os.path.join(self.figure_folder, 'noise_correlation', 'whole_brain', 'connectome_pos_{}.pdf'.format(self.split_by))
                plotting.plot_connectome(contrast_mat_m, atlas_region_coords, edge_threshold=0, colorbar=True, output_file=filename)
                
                
                contrast_mat_m = copy.copy(corrmats_contrast_av)
                contrast_mat_m[contrast_mat_m>0] = 0
                contrast_mat_m[p_mat>0.005] = 0
                filename=os.path.join(self.figure_folder, 'noise_correlation', 'whole_brain', 'connectome_neg_{}.pdf'.format(self.split_by))
                plotting.plot_connectome(contrast_mat_m, atlas_region_coords, edge_threshold=0, colorbar=True, output_file=filename)
                
                
            # # plot cells that correlate to drift criterion:
            # # ---------------------------------------------
            #
            # r_matrix = np.zeros((corrmats_contrast.shape[0], corrmats_contrast.shape[1]))
            # p_matrix = np.zeros((corrmats_contrast.shape[0], corrmats_contrast.shape[1]))
            # for i in range(corrmats_contrast.shape[0]):
            #     for j in range(corrmats_contrast.shape[1]):
            #         r_matrix[i,j] = sp.stats.spearmanr(corrmats_contrast[i,j,:], self.criterion_hi-self.criterion_lo)[0]
            #         p_matrix[i,j] = sp.stats.spearmanr(corrmats_contrast[i,j,:], self.criterion_hi-self.criterion_lo)[1]
            #
            #         # if p_matrix[i,j] < 0.001:
            #             # fig = plt.figure()
            #             # ax = fig.add_subplot(111)
            #             # myfuncs.correlation_plot(contrast_matrix_across[i,j,:], self.criterion_hi-self.criterion_lo, line=True, ax=ax)
            #     #         fig.savefig(os.path.join(self.figure_folder, 'correlation', 'big_matrix_2', 'drift_criterion_{}_{}.pdf'.format(i,j)))
            #     #         break
            #     #     else:
            #     #         continue
            #     #     break
            #     # break
            #
            # # fdr correct:
            # r_matrix[np.isnan(r_matrix)] = 0
            # p_matrix[np.isnan(p_matrix)] = 1
            # p_matrix = mne.stats.fdr_correction(p_matrix, 0.05)[1]
            #
            # mask = np.array(np.tri(corrmats_contrast_av.shape[0], k=0), dtype=bool)
            # r_matrix_m = np.ma.masked_where(mask, r_matrix)
            # fig = plt.figure(figsize=(5,4))
            # ax = fig.add_subplot(111)
            # vmax = max(abs(r_matrix.ravel()))
            # im = ax.pcolormesh(r_matrix_m, cmap='bwr', vmin=-vmax, vmax=vmax)
            # ax.set_xlim(xmax=len(rois))
            # ax.set_ylim(ymax=len(rois))
            # ax.set_yticks(arange(0.5,len(rois)+.5))
            # ax.set_xticks(arange(0.5,len(rois)+.5))
            # ax.set_yticklabels(roi_names)
            # ax.set_xticklabels(roi_names, rotation=270)
            # ax.patch.set_hatch('x')
            # ax.set_aspect(1)
            # fig.colorbar(im)
            # plt.tight_layout()
            # fig.savefig(os.path.join(self.figure_folder, 'correlation', 'big_matrix_2', 'drift_criterion_{}_{}_{}.pdf'.format('contrast', dv, self.split_by)))
            #
            # mask = (p_matrix > 0.05)+np.array(np.tri(corrmats_contrast_av.shape[0], k=0), dtype=bool)
            # r_matrix_m = np.ma.masked_where(mask, r_matrix)
            # fig = plt.figure(figsize=(5,4))
            # ax = fig.add_subplot(111)
            # vmax = max(abs(r_matrix.ravel()))
            # im = ax.pcolormesh(r_matrix_m, cmap='bwr', vmin=-vmax, vmax=vmax)
            # ax.set_xlim(xmax=len(rois))
            # ax.set_ylim(ymax=len(rois))
            # ax.set_yticks(arange(0.5,len(rois)+.5))
            # ax.set_xticks(arange(0.5,len(rois)+.5))
            # ax.set_yticklabels(roi_names)
            # ax.set_xticklabels(roi_names, rotation=270)
            # ax.patch.set_hatch('x')
            # ax.set_aspect(1)
            # fig.colorbar(im)
            # plt.tight_layout()
            # fig.savefig(os.path.join(self.figure_folder, 'correlation', 'big_matrix_2', 'drift_criterion_{}_{}_{}_masked_p.pdf'.format('contrast', dv, self.split_by)))
            #
            # mask = (p_matrix > 0.05)+np.array(np.tri(corrmats_contrast_av.shape[0], k=0), dtype=bool)+(corrmats_contrast_av < 0)
            # r_matrix_m = np.ma.masked_where(mask, r_matrix)
            # fig = plt.figure(figsize=(5,4))
            # ax = fig.add_subplot(111)
            # vmax = max(abs(r_matrix.ravel()))
            # im = ax.pcolormesh(r_matrix_m, cmap='bwr', vmin=-vmax, vmax=vmax)
            # ax.set_xlim(xmax=len(rois))
            # ax.set_ylim(ymax=len(rois))
            # ax.set_yticks(arange(0.5,len(rois)+.5))
            # ax.set_xticks(arange(0.5,len(rois)+.5))
            # ax.set_yticklabels(roi_names)
            # ax.set_xticklabels(roi_names, rotation=270)
            # ax.patch.set_hatch('x')
            # ax.set_aspect(1)
            # fig.colorbar(im)
            # plt.tight_layout()
            # fig.savefig(os.path.join(self.figure_folder, 'correlation', 'big_matrix_2', 'drift_criterion_{}_{}_{}_masked_p_masked_pos.pdf'.format('contrast', dv, self.split_by)))
            
            
    def WHOLEBRAIN_noise_correlation_bias(self, data_type, partial=False,):
 
        self.postfix = '_d'
        time_locked = 'stim_locked'
        self.make_pupil_BOLD_dataframe(data_type=data_type, time_locked=time_locked, regress_iti=False, regress_rt=True, regress_stimulus=True,)
 
        rois_list = [
                     # PARCELATION LEFT:
                     # -----------------
             
                     [
                     # visual:
                     'lh.Pole_occipital',
                     'lh.G_occipital_sup',
                     'lh.S_oc_sup_and_transversal',
                     'lh.G_occipital_middle',
                     'lh.S_occipital_ant',
                     'lh.S_oc_middle_and_Lunatus',
                     'lh.G_and_S_occipital_inf',
                     'lh.S_collat_transv_post',
                     'lh.G_oc-temp_med-Lingual',
                     'lh.S_calcarine',
                     'lh.G_cuneus',
                     ],
                     [
                     # temporal:
                     'lh.Lat_Fis-post',
                     'lh.G_temp_sup-Plan_tempo',
                     'lh.S_temporal_transverse',
                     'lh.G_temp_sup-G_T_transv',
                     'lh.G_temp_sup-Lateral',
                     'lh.S_temporal_sup',
                     'lh.G_temporal_middle',
                     'lh.S_temporal_inf',
                     'lh.G_temporal_inf',
                     # 'lh.S_oc-temp_lat',
                     'lh.G_oc-temp_med-Parahip',
                     'lh.S_collat_transv_ant',
                     'lh.G_oc-temp_lat-fusifor',
                     'lh.S_oc-temp_med_and_Lingual',
                     'lh.G_temp_sup-Plan_polar',
                     'lh.Pole_temporal',
                     ],
                     [
                     # parietal:
                     'lh.S_parieto_occipital',
                     'lh.S_subparietal',
                     'lh.G_precuneus',
                     'lh.G_parietal_sup',
                     'lh.S_intrapariet_and_P_trans',
                     'lh.G_pariet_inf-Angular',
                     'lh.S_interm_prim-Jensen',
                     'lh.G_and_S_paracentral',
                     'lh.S_postcentral',
                     'lh.G_postcentral',
                     'lh.S_central',
                     'lh.G_pariet_inf-Supramar',
                     'lh.G_and_S_subcentral',
                     ],
                     [
                     # insular:
                     'lh.S_circular_insula_sup',
                     'lh.G_insular_short',
                     'lh.S_circular_insula_inf',
                     'lh.G_Ins_lg_and_S_cent_ins',
                     'lh.S_circular_insula_ant',
                     ],
                     [
                     # cingulate:
                     'lh.G_cingul-Post-ventral',
                     'lh.S_pericallosal',
                     'lh.G_cingul-Post-dorsal',
                     'lh.S_cingul-Marginalis',
                     'lh.G_and_S_cingul-Mid-Post',
                     'lh.G_and_S_cingul-Mid-Ant',
                     'lh.G_and_S_cingul-Ant',
                     ],
                     [
                     # frontal:
                     # 'lh.G_precentral',
                     'lh.S_precentral-sup-part',
                     'lh.S_precentral-inf-part',
                     'lh.G_front_sup',
                     'lh.S_front_sup',
                     'lh.G_front_middle',
                     # 'lh.S_front_middle',
                     'lh.S_front_inf',
                     'lh.G_front_inf-Opercular',
                     'lh.G_front_inf-Triangul',
                     'lh.S_orbital_lateral',
                     'lh.Lat_Fis-ant-Horizont',
                     'lh.Lat_Fis-ant-Vertical',
                     'lh.G_front_inf-Orbital',
                     'lh.G_and_S_transv_frontopol',
                     'lh.G_and_S_frontomargin',
                     'lh.G_orbital',
                     'lh.S_orbital-H_Shaped',
                     'lh.S_orbital_med-olfact',
                     'lh.G_rectus',
                     'lh.S_suborbital',
                     'lh.G_subcallosal',
                     ],
                     # PARCELATION RIGHT:
                     # -----------------
                     [
                     # visual:
                     'rh.Pole_occipital',
                     'rh.G_occipital_sup',
                     'rh.S_oc_sup_and_transversal',
                     'rh.G_occipital_middle',
                     'rh.S_occipital_ant',
                     'rh.S_oc_middle_and_Lunatus',
                     'rh.G_and_S_occipital_inf',
                     'rh.S_collat_transv_post',
                     'rh.G_oc-temp_med-Lingual',
                     'rh.S_calcarine',
                     'rh.G_cuneus',
                     ],
                     [
                     # temporal:
                     'rh.Lat_Fis-post',
                     'rh.G_temp_sup-Plan_tempo',
                     'rh.S_temporal_transverse',
                     'rh.G_temp_sup-G_T_transv',
                     'rh.G_temp_sup-Lateral',
                     'rh.S_temporal_sup',
                     'rh.G_temporal_middle',
                     'rh.S_temporal_inf',
                     'rh.G_temporal_inf',
                     # 'rh.S_oc-temp_lat',
                     'rh.G_oc-temp_med-Parahip',
                     'rh.S_collat_transv_ant',
                     'rh.G_oc-temp_lat-fusifor',
                     'rh.S_oc-temp_med_and_Lingual',
                     'rh.G_temp_sup-Plan_polar',
                     'rh.Pole_temporal',
                     ],
                     [
                     # parietal:
                     'rh.S_parieto_occipital',
                     'rh.S_subparietal',
                     'rh.G_precuneus',
                     'rh.G_parietal_sup',
                     'rh.S_intrapariet_and_P_trans',
                     'rh.G_pariet_inf-Angular',
                     'rh.S_interm_prim-Jensen',
                     'rh.G_and_S_paracentral',
                     'rh.S_postcentral',
                     'rh.G_postcentral',
                     'rh.S_central',
                     'rh.G_pariet_inf-Supramar',
                     'rh.G_and_S_subcentral',
                     ],
                     [
                     # insular:
                     'rh.S_circular_insula_sup',
                     'rh.G_insular_short',
                     'rh.S_circular_insula_inf',
                     'rh.G_Ins_lg_and_S_cent_ins',
                     'rh.S_circular_insula_ant',
                     ],
                     [
                     # cingulate:
                     'rh.G_cingul-Post-ventral',
                     'rh.S_pericallosal',
                     'rh.G_cingul-Post-dorsal',
                     'rh.S_cingul-Marginalis',
                     'rh.G_and_S_cingul-Mid-Post',
                     'rh.G_and_S_cingul-Mid-Ant',
                     'rh.G_and_S_cingul-Ant',
                     ],
                     [
                     # frontal:
                     # 'rh.G_precentral',
                     'rh.S_precentral-sup-part',
                     'rh.S_precentral-inf-part',
                     'rh.G_front_sup',
                     'rh.S_front_sup',
                     'rh.G_front_middle',
                     # 'rh.S_front_middle',
                     'rh.S_front_inf',
                     'rh.G_front_inf-Opercular',
                     'rh.G_front_inf-Triangul',
                     'rh.S_orbital_lateral',
                     'rh.Lat_Fis-ant-Horizont',
                     'rh.Lat_Fis-ant-Vertical',
                     'rh.G_front_inf-Orbital',
                     'rh.G_and_S_transv_frontopol',
                     'rh.G_and_S_frontomargin',
                     'rh.G_orbital',
                     'rh.S_orbital-H_Shaped',
                     'rh.S_orbital_med-olfact',
                     'rh.G_rectus',
                     'rh.S_suborbital',
                     'rh.G_subcallosal',
                     ]
                ]
 
        rois = np.concatenate(rois_list)
 
        wrong_rois = np.zeros(len(rois))
        for i in range(len(self.subjects)):
            C = np.vstack([np.array(self.data_frame[(self.data_frame.subject == self.subjects[i])][roi+self.postfix]) for roi in rois]).T
            sdts = np.std(C[:,:], axis=0)
            wrong_rois = wrong_rois = myfuncs.is_outlier(sdts)
        clean_rois = (wrong_rois == 0)
        wrong_rois = rois[wrong_rois]
        rois = rois[clean_rois] 
        # compute boundaries:
        for region in range(len(rois_list)):
            for r in wrong_rois:
                try:
                    rois_list[region].remove(r)
                except:
                    pass
        boundaries = np.cumsum([len(region) for region in rois_list])
        boundaries = boundaries[:-1]
        
        run = False
        if run:
            perms = 10000
            criterions = np.zeros((len(self.subjects), perms))
            d_primes = np.zeros((len(self.subjects), perms))
            indices = []
            for p in range(perms):
            
                inds = []
                for i in range(len(self.subjects)):
                    nr_trials = self.pupil_data[i].shape[0]
                    ind = np.zeros(nr_trials)
                    ind[:nr_trials/2] = 1
                    ind = ind[np.argsort(np.random.rand(nr_trials))]
                    inds.append( (ind==1) )
                indices.append(inds)
            
                for i in range(len(self.subjects)):
                    d_primes[i,p], criterions[i,p] = myfuncs.SDT_measures(self.present[i][indices[p][i]], self.hit[i][indices[p][i]], self.fa[i][indices[p][i]])
        
            print criterions[:,criterions.mean(axis=0)<np.percentile(criterions.mean(axis=0),10)].mean()
            print criterions[:,criterions.mean(axis=0)>np.percentile(criterions.mean(axis=0),90)].mean()
            print d_primes[:,criterions.mean(axis=0)<np.percentile(criterions.mean(axis=0),10)].mean()
            print d_primes[:,criterions.mean(axis=0)>np.percentile(criterions.mean(axis=0),90)].mean()
        
            inds_0 = [indices[ii] for ii in np.where(criterions.mean(axis=0)<np.percentile(criterions.mean(axis=0),10))[0]]
            inds_1 = [indices[ii] for ii in np.where(criterions.mean(axis=0)>np.percentile(criterions.mean(axis=0),90))[0]]
        
            corrmats_contrast_av_perms = []
            for p in range(len(inds_0)):
                contrast_across_dvs = []
                for dv in ['cor',]:
                    cormats = []
                    for ind in [np.concatenate(inds_1[p]), np.concatenate(inds_0[p])]:
                        cm = np.zeros((len(rois), len(rois), len(self.subjects)))
                        cm_p = np.zeros((len(rois), len(rois), len(self.subjects)))
                        for i in range(len(self.subjects)):
                            C = np.vstack([np.array(self.data_frame[(self.data_frame.subject == self.subjects[i])*ind][roi+self.postfix]) for roi in rois]).T
                            if dv == 'cor':
                                if partial:
                                    cm[:,:,i], cm_p[:,:,i] = myfuncs.corr_matrix_partial(C)
                                else:
                                    cm[:,:,i], cm_p[:,:,i] = myfuncs.corr_matrix(C, dv='cor')
                            if dv == 'var':
                                cm[:,:,i], cm_p[:,:,i] = myfuncs.corr_matrix(C, dv='var')
                            if dv == 'cov':
                                cm[:,:,i], cm_p[:,:,i] = myfuncs.corr_matrix(C, dv='cov')
                            if dv == 'mean':
                                cm[:,:,i], cm_p[:,:,i] = myfuncs.corr_matrix(C, dv='mean')
                            if dv == 'snr':
                                cm[:,:,i], cm_p[:,:,i] = myfuncs.corr_matrix(C, dv='snr')
                        cormats.append(cm)
                    
                    corrmats_mean = (cormats[0]+cormats[1]) / 2.0
                    corrmats_mean_av = corrmats_mean.mean(axis=-1)
                    corrmats_contrast = cormats[0]-cormats[1]
                    corrmats_contrast_av = corrmats_contrast.mean(axis=-1)
                    corrmats_contrast_av_perms.append(corrmats_contrast_av)
            corrmats_contrast_av_perms = np.stack(corrmats_contrast_av_perms)
            
            # save:
            np.save(os.path.join(self.data_folder, 'brute_force', 'contrast_matrices.npy'), corrmats_contrast_av_perms)
            np.save(os.path.join(self.data_folder, 'brute_force', 'inds_0.npy'), inds_0)
            np.save(os.path.join(self.data_folder, 'brute_force', 'inds_1.npy'), inds_1)
            
        else:
            
            shell()
            
            corrmats_contrast_av_perms = np.load(os.path.join(self.data_folder, 'brute_force', 'contrast_matrices.npy'), )
            inds_0 = np.load(os.path.join(self.data_folder, 'brute_force', 'inds_0.npy'), )
            inds_1 = np.load(os.path.join(self.data_folder, 'brute_force', 'inds_1.npy'), )
            
            
            corrmats_contrast_av = corrmats_contrast_av_perms.mean(axis=0)
            
            p_mat = np.zeros(corrmats_mean_av.shape)
            # for i in range(p_mat.shape[0]):
            #     for j in range(p_mat.shape[1]):
            #         try:
            #             p_mat[i,j] = sp.stats.wilcoxon(corrmats_contrast[i,j,:])[1]
            #         except:
            #             p_mat[i,j] = 1

            # # fdr correction:
            # p_mat = mne.stats.fdr_correction(p_mat, 0.10)[1]
            
            dv = 'cor'
            
            # contrast matrix:
            mask =  np.tri(corrmats_contrast_av.shape[0], k=0)
            corrmat_m = np.ma.masked_where(mask, corrmats_contrast_av)
            p_mat_m = np.ma.masked_where(mask.T, p_mat)
            roi_names = rois
            fig = plt.figure(figsize=(5,4))
            # fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(111)
            vmax = max(abs(corrmats_contrast_av.ravel()))
            im = ax.pcolormesh(corrmat_m, cmap='bwr', vmin=-vmax, vmax=vmax)
            ax.set_xlim(xmax=len(rois))
            ax.set_ylim(ymax=len(rois))
            ax.set_yticks(arange(0.5,len(rois)+.5))
            ax.set_xticks(arange(0.5,len(rois)+.5))
            ax.set_yticklabels(roi_names)
            ax.set_xticklabels(roi_names, rotation=270)
            ax.patch.set_hatch('x')
            ax.set_aspect(1)
            fig.colorbar(im)
            for l in boundaries:
                plt.vlines(l, 0, l, lw=1)
                plt.hlines(l, l, corrmat_m.shape[0], lw=1)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, 'noise_correlation', 'whole_brain', 'matrix_single_{}_{}_{}.pdf'.format(self.split_by, 'contrast', dv,)))

            mask = (p_mat > 0.05)+np.array(np.tri(corrmats_contrast_av.shape[0], k=0), dtype=bool)
            corrmat_m = np.ma.masked_where(mask, corrmats_contrast_av)

            # fig = plt.figure(figsize=(5,4))
            # ax = fig.add_subplot(111)
            # vmax = max(abs(corrmats_contrast_av.ravel()))
            # im = ax.pcolormesh(corrmat_m, cmap='bwr', vmin=-vmax, vmax=vmax)
            # ax.set_xlim(xmax=len(rois))
            # ax.set_ylim(ymax=len(rois))
            # ax.set_yticks(arange(0.5,len(rois)+.5))
            # ax.set_xticks(arange(0.5,len(rois)+.5))
            # ax.set_yticklabels(roi_names)
            # ax.set_xticklabels(roi_names, rotation=270)
            # ax.patch.set_hatch('x')
            # ax.set_aspect(1)
            # fig.colorbar(im)
            # for l in boundaries:
            #     plt.vlines(l, 0, l, lw=1)
            #     plt.hlines(l, l, corrmat_m.shape[0], lw=1)
            # plt.tight_layout()
            # fig.savefig(os.path.join(self.figure_folder, 'noise_correlation', 'whole_brain', 'matrix_single_{}_{}_{}_significant.pdf'.format(self.split_by, 'contrast', dv,)))

            if dv == 'cor':

                # plot connectome:
                # ----------------
                
                import copy
                from nilearn import image
                from nilearn import plotting

                atlas_rois = np.zeros((len(rois), 91, 109, 91))
                for i, roi in enumerate(rois):
                    atlas_rois[i,:,:,:] = NiftiImage('/home/shared/UvA/Niels_UvA/brainstem_masks/destrieux/' + roi + '.nii.gz').data

                atlas_rois = NiftiImage(atlas_rois)
                atlas_rois.header = NiftiImage('/home/shared/UvA/Niels_UvA/brainstem_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz').header
                atlas_rois.save(os.path.join('/home/shared/UvA/Niels_UvA/brainstem_masks/destrieux', "atlas.nii.gz"))

                atlas = nib.load(os.path.join('/home/shared/UvA/Niels_UvA/brainstem_masks/destrieux', "atlas.nii.gz"))
                atlas_imgs = image.iter_img(atlas)
                atlas_region_coords = [nilearn.plotting.find_xyz_cut_coords(img) for img in atlas_imgs]
                
                # volume_file1 = os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_psc_all.nii.gz'.format('clean_MNI_smooth', 'stim_locked', 'mean', self.split_by))
                # threshold_file1 = NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_psc_all_tfce_corrp_tstat1.nii.gz'.format('clean_MNI_smooth', 'stim_locked', 'mean', self.split_by))).data
                # dummy1 = NiftiImage(volume_file1).data
                # dummy1 = dummy1.mean(axis=0)
                # dummy1[threshold_file1<0.99] = 0
                # dummy1 = NiftiImage(dummy1)
                # dummy1.header = NiftiImage('/home/shared/Niels_UvA/brainstem_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz').header
                # dummy1.save(os.path.join('/home/shared/Niels_UvA/Visual_UvA/surface_plots', "dummy1.nii.gz"))
                dummy_file1 = os.path.join('/home/shared/UvA/Niels_UvA/Visual_UvA/surface_plots', "dummy1.nii.gz")
                #
                # volume_file2 = os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_psc_all_rev.nii.gz'.format('clean_MNI_smooth', 'stim_locked', 'mean', self.split_by))
                # threshold_file2 = NiftiImage(os.path.join(self.data_folder, 'event_related_average_wb', '{}_{}_{}_{}_psc_all_rev_tfce_corrp_tstat1.nii.gz'.format('clean_MNI_smooth', 'stim_locked', 'mean', self.split_by))).data
                # dummy2 = NiftiImage(volume_file2).data
                # dummy2 = dummy2.mean(axis=0)
                # dummy2[threshold_file2<0.99] = 0
                # dummy2 = NiftiImage(dummy2)
                # dummy2.header = NiftiImage('/home/shared/Niels_UvA/brainstem_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz').header
                # dummy2.save(os.path.join('/home/shared/Niels_UvA/Visual_UvA/surface_plots', "dummy2.nii.gz"))
                dummy_file2 = os.path.join('/home/shared/UvA/Niels_UvA/Visual_UvA/surface_plots', "dummy2.nii.gz")

                # # params:
                # thresh1 = 0.001
                # thresh2 = 1.5
                # percentile = 99
                # strongest = 50 # nr of cells
                #
                # # negative connections:
                # contrast_mat_m = copy.copy(corrmats_contrast_av)
                # contrast_mat_m = contrast_mat_m * -1.0
                # threshold_values = contrast_mat_m.ravel()[-(contrast_mat_m.ravel()<=0)]
                # # threshold = np.percentile(threshold_values, percentile)
                # # threshold = np.sort(threshold_values)[-strongest]
                # # contrast_mat_m[contrast_mat_m<threshold] = 0
                # contrast_mat_m[p_mat>0.01] = 0
                # threshold = min(contrast_mat_m[contrast_mat_m>0])
                # contrast_mat_m[contrast_mat_m<threshold] = 0
                # fig = plt.figure(figsize=(3, 1))
                # display = plotting.plot_glass_brain(dummy_file1, threshold=100000, figure=fig)
                # # display.add_overlay(dummy_file1, threshold=thresh1, colorbar=False, cmap="YlOrRd_r", **{'vmax': thresh2,'alpha': 0.2})
                # # display.add_overlay(dummy_file2, threshold=thresh1, colorbar=False, cmap="Blues_r", **{'vmax': thresh2,'alpha': 0.2})
                # display.add_graph(contrast_mat_m, atlas_region_coords, edge_threshold=threshold, edge_cmap='bwr_r', edge_kwargs={'lw':1}, node_size=7)
                # display.savefig(os.path.join(self.figure_folder, 'noise_correlation', 'whole_brain', 'connectome_neg_{}.pdf'.format(self.split_by)))
                #
                # # positive connections:
                # contrast_mat_m = copy.copy(corrmats_contrast_av)
                # contrast_mat_m = contrast_mat_m
                # threshold_values = contrast_mat_m.ravel()[-(contrast_mat_m.ravel()<=0)]
                # # threshold = np.percentile(threshold_values, percentile)
                # # threshold = np.sort(threshold_values)[-strongest]
                # # contrast_mat_m[contrast_mat_m<threshold] = 0
                # contrast_mat_m[p_mat>0.01] = 0
                # threshold = min(contrast_mat_m[contrast_mat_m>0])
                # contrast_mat_m[contrast_mat_m<threshold] = 0
                # fig = plt.figure(figsize=(3, 1))
                # display = plotting.plot_glass_brain(dummy_file1, threshold=100000, figure=fig)
                # # display.add_overlay(dummy_file1, threshold=thresh1, colorbar=False, cmap="YlOrRd_r", **{'vmax': thresh2,'alpha': 0.2})
                # # display.add_overlay(dummy_file2, threshold=thresh1, colorbar=False, cmap="Blues_r", **{'vmax': thresh2,'alpha': 0.2})
                # display.add_graph(contrast_mat_m, atlas_region_coords, edge_threshold=threshold, edge_cmap='bwr', edge_kwargs={'lw':1}, node_size=7)
                # display.savefig(os.path.join(self.figure_folder, 'noise_correlation', 'whole_brain', 'connectome_pos_{}.pdf'.format(self.split_by)))
                
                
                contrast_mat_m = copy.copy(corrmats_contrast_av)
                contrast_mat_m[contrast_mat_m<0] = 0
                contrast_mat_m[p_mat>0.05] = 0
                filename=os.path.join(self.figure_folder, 'noise_correlation', 'whole_brain', 'connectome_pos_{}_PERMS.pdf'.format(self.split_by))
                plotting.plot_connectome(contrast_mat_m, atlas_region_coords, edge_threshold='99%', colorbar=True, output_file=filename)
                
                
                contrast_mat_m = copy.copy(corrmats_contrast_av)
                contrast_mat_m[contrast_mat_m>0] = 0
                contrast_mat_m[p_mat>0.005] = 0
                filename=os.path.join(self.figure_folder, 'noise_correlation', 'whole_brain', 'connectome_neg_{}_PERMS.pdf'.format(self.split_by))
                plotting.plot_connectome(contrast_mat_m, atlas_region_coords, edge_threshold='99%', colorbar=True, output_file=filename)

            
            
    
    def DDM_dataframe(self,):
        
        time_locked = 'stim_locked'
        self.make_pupil_BOLD_dataframe(data_type='clean_4th_ventricle', time_locked=time_locked, regress_iti=False, regress_rt=True, regress_stimulus=False)
        
        data_brainstem = self.data_frame.copy()
        dfs_brainstem = [data_brainstem[data_brainstem.subject == s] for s in self.subjects]
        
        data_frame = pd.read_csv(os.path.join(self.data_folder, 'event_related_average_wb', 'MNI_dataframe_1.csv'))
        data_frame = pd.concat([data_frame[data_frame.subject == s] for s in self.subjects])
        
        # add linear combinations:
        for i in range(len(self.subjects)):
            d = {'Y' : pd.Series(self.pupil_data[i]['pupil_d']),
                'X1' : pd.Series(dfs_brainstem[i]['LC_JW_d']),
                'X2' : pd.Series(dfs_brainstem[i]['mean_SN_d']),
                'X3' : pd.Series(dfs_brainstem[i]['mean_VTA_d']),
                'X4' : pd.Series(dfs_brainstem[i]['basal_forebrain_123_d']),
                'X5' : pd.Series(dfs_brainstem[i]['basal_forebrain_4_d']),
                'X6' : pd.Series(dfs_brainstem[i]['inf_col_jw_d']),
                'X7' : pd.Series(dfs_brainstem[i]['sup_col_jw_d']),
                }
            
            data = pd.DataFrame(d)
            formula = 'Y ~ X1 + X2 + X3 + X4 + X5'
            model = sm.ols(formula=formula, data=data)
            fitted = model.fit()
            values = np.array(fitted.fittedvalues)
            values = (values - values.mean()) / values.std()
            dfs_brainstem[i]['BS_nMod'] = values
            formula = 'Y ~ X6 + X7'
            model = sm.ols(formula=formula, data=data)
            fitted = model.fit()
            values = np.array(fitted.fittedvalues)
            values = (values - values.mean()) / values.std()
            dfs_brainstem[i]['BS_C'] = values
            
        d = {
        'subj_idx' : pd.Series(self.subj_idx),
        'response' : pd.Series(np.array(np.concatenate(self.yes), dtype=int)),
        'stimulus' : pd.Series(np.array(np.concatenate(self.present), dtype=int)),
        'rt' : pd.Series(np.concatenate(self.rt)),
        'BS_nMod' : pd.Series(np.array(pd.concat(dfs_brainstem)['BS_nMod'])),
        'BS_C' : pd.Series(np.array(pd.concat(dfs_brainstem)['BS_C'])),
        'combined_choice_all' : pd.Series(np.array(data_frame['combined_choice_parietal_all'])),
        'combined_choice_lat' : pd.Series(np.array(data_frame['combined_choice_parietal_lat'])),
        'combined_choice_sl' : pd.Series(np.array(data_frame['combined_choice_parietal_sl'])),
        'M1' : pd.Series(np.array(data_frame['lr_M1_lat'])),
        'pupil_d' : pd.Series(np.array(pd.concat(self.pupil_data)['pupil_d'])),
        'run' : pd.Series(np.concatenate(self.run)),
        'session' : pd.Series(np.concatenate(self.session)),
        }
        self.df = pd.DataFrame(d)
        self.df.to_csv(os.path.join(self.figure_folder, '2014_fMRI_data_response_brainstem.csv'))    
    
    
    def correlation_matrix_binned(self, rois, data_type, bin_by='all', bins=50, partial=False):
        
        time_locked = 'stim_locked'
        self.make_pupil_BOLD_dataframe(data_type=data_type, time_locked=time_locked, regress_iti=False, regress_rt_signal=True)
        
        for signal_type in ['baseline', 'phasic']:
            
            if bin_by == 'all':
                corrmat = np.zeros((len(rois),len(rois)))
                p_mat = np.zeros((len(rois),len(rois)))
                for j, roi_1 in enumerate(rois):
                    for k, roi_2 in enumerate(rois):
                        roi_data_across_1 = np.zeros((len(self.subjects),bins))
                        roi_data_across_2 = np.zeros((len(self.subjects),bins))
                        for i in range(len(self.subjects)):
                            if signal_type == 'baseline':
                                bin_indices = np.array_split(np.argsort(np.array(self.data_frame[self.data_frame.subject == self.subjects[i]][roi_1+'_b'])), bins)
                                roi_data_1 = np.array(self.data_frame[self.data_frame.subject == self.subjects[i]][roi_1+'_b'])
                                roi_data_2 = np.array(self.data_frame[self.data_frame.subject == self.subjects[i]][roi_2+'_b'])
                            if signal_type == 'phasic':
                                bin_indices = np.array_split(np.argsort(np.array(self.data_frame[self.data_frame.subject == self.subjects[i]][roi_1])), bins)
                                roi_data_1 = np.array(self.data_frame[self.data_frame.subject == self.subjects[i]][roi_1])
                                roi_data_2 = np.array(self.data_frame[self.data_frame.subject == self.subjects[i]][roi_2])
                            
                            for b in range(bins):
                                roi_data_across_1[i,b] = np.mean(roi_data_1[bin_indices[b]])
                                roi_data_across_2[i,b] = np.mean(roi_data_2[bin_indices[b]])
                        roi_data_across_1 = roi_data_across_1.mean(axis=0)
                        roi_data_across_2 = roi_data_across_2.mean(axis=0)
                        corr, p_value = sp.stats.pearsonr(roi_data_across_1, roi_data_across_2)
                        corrmat[j,k] = corr
                        p_mat[j,k] = p_value
            else:
                C = np.zeros( (len(self.subjects), len(rois),bins) )
                for i in range(len(self.subjects)):
                    if signal_type == 'baseline':
                        bin_indices = np.array_split(np.argsort(np.array(self.data_frame[self.data_frame.subject == self.subjects[i]][bin_by+'_b'])), bins)
                    if signal_type == 'phasic':
                        bin_indices = np.array_split(np.argsort(np.array(self.data_frame[self.data_frame.subject == self.subjects[i]][bin_by])), bins)
                    for j, roi in enumerate(rois):
                        if signal_type == 'baseline':
                            roi_data = np.array(self.data_frame[self.data_frame.subject == self.subjects[i]][roi+'_b'])
                        if signal_type == 'phasic':
                            roi_data = np.array(self.data_frame[self.data_frame.subject == self.subjects[i]][roi])
                        for b in range(bins):
                            C[i,j,b] = np.mean(roi_data[bin_indices[b]])
                C = np.mean(C, axis=0).T
                if partial:
                    corrmat, p_mat = myfuncs.corr_matrix_partial(C)
                else:
                    corrmat, p_mat = myfuncs.corr_matrix(C)
            
            # fdr correction:
            p_mat = mne.stats.fdr_correction(p_mat, 0.05)[1]
            
            # plotting:
            mask =  np.tri(corrmat.shape[0], k=0)
            corrmat = np.ma.masked_where(mask, corrmat)
            p_mat = np.ma.masked_where(mask.T, p_mat)
            roi_names = ['ACC1', 'SC', 'BF', 'SN/VTA', 'LC',]
            if partial:
                roi_names = ['BF', 'SN/VTA', 'LC',]
            fig_spacing = (3,9)
            fig = plt.figure(figsize=(8.27, 5.845))
            ax1 = plt.subplot2grid(fig_spacing, (0, 0), colspan=2)
            axes = [ax1]
            plot_nr = 0
            ax = axes[plot_nr]
            if bins > 50:
                im = ax.pcolormesh(corrmat, cmap='bwr', vmin=-0.5, vmax=0.5)
            else:
                im = ax.pcolormesh(corrmat, cmap='bwr', vmin=-1, vmax=1)
            # if partial:
            #     im = ax.pcolormesh(corrmat_m, cmap='bwr', vmin=-0.5, vmax=0.5)
            # else:
            #     im = ax.pcolormesh(corrmat_m, cmap='hot', vmin=0, vmax=0.5)
            ax.patch.set_hatch('x')
            ax.set_xlim(xmax=len(rois))
            ax.set_ylim(ymax=len(rois))
            ax.set_yticks(arange(0.5,len(rois)+.5))
            ax.set_xticks(arange(0.5,len(rois)+.5))
            ax.set_yticklabels(roi_names)
            ax.set_xticklabels(roi_names, rotation=270)
            ax.patch.set_hatch('x')
            for i in range(p_mat.shape[0]):
                for j in range(p_mat.shape[1]):
                    if p_mat[i,j] or (p_mat[i,j] == 0):
                        star1 = ''
                        if p_mat[i,j] < 0.05:
                            star1 = '*'
                        if p_mat[i,j] < 0.01:
                            star1 = '**'
                        if p_mat[i,j] < 0.001:
                            star1 = '***'
                        ax.text(i+0.5,j+0.5,star1,size=12,horizontalalignment='center',verticalalignment='center',)
        
            fig.colorbar(im)
            plt.tight_layout()
            if partial:
                fig.savefig(os.path.join(self.figure_folder, 'correlation', '0_correlation_matrix_{}_binned_partial_{}_{}.pdf'.format(bin_by, signal_type, bins)))
            else:
                fig.savefig(os.path.join(self.figure_folder, 'correlation', '0_correlation_matrix_{}_binned_{}_{}.pdf'.format(bin_by, signal_type, bins)))
    
    def correlation_matrix_binned_all_rois(self, data_type, bin_by='all', bins=50, partial=False):
        
        # bins = int(floor((bins / 100.0 * 40)))
        
        time_locked = 'stim_locked'
        self.make_pupil_BOLD_dataframe(data_type=data_type, time_locked=time_locked)
        
        
        print 'mean baseline pupil on low baseline trials: {}'.format(np.round(np.mean(np.array(self.data_frame.pupil_b)[np.concatenate(self.pupil_b_l_ind)]),3))
        print 'mean baseline pupil on high baseline trials: {}'.format(np.round(np.mean(np.array(self.data_frame.pupil_b)[np.concatenate(self.pupil_b_h_ind)]),3))
        print 'mean phasic pupil on low phasic trials: {}'.format(np.round(np.mean(np.array(self.data_frame.pupil)[np.concatenate(self.pupil_l_ind)]),3))
        print 'mean phasic pupil on high phasic trials: {}'.format(np.round(np.mean(np.array(self.data_frame.pupil)[np.concatenate(self.pupil_h_ind)]),3))
        
        
        # shell()
        
        # myfuncs.correlation_plot(np.array(np.concatenate(self.pupil_b_l_ind),dtype=int), np.array(np.concatenate(self.pupil_l_ind),dtype=int))
        # myfuncs.correlation_plot(np.array(np.concatenate(self.pupil_b_h_ind),dtype=int), np.array(np.concatenate(self.pupil_h_ind),dtype=int))
        
        # PLOTTING:
        rois = [
                # visual:
                'Pole_occipital',
                'G_occipital_sup',
                'S_oc_sup_and_transversal',
                'G_occipital_middle',
                'S_occipital_ant',
                'S_oc_middle_and_Lunatus',
                'G_and_S_occipital_inf',
                'S_collat_transv_post',
                'G_oc-temp_med-Lingual',
                'S_calcarine',
                'G_cuneus',
    
                # temporal:
                'Lat_Fis-post',
                'G_temp_sup-Plan_tempo',
                'S_temporal_transverse',
                'G_temp_sup-G_T_transv',
                'G_temp_sup-Lateral',
                'S_temporal_sup',
                'G_temporal_middle',
                'S_temporal_inf',
                'G_temporal_inf',
                'S_oc-temp_lat',
                'G_oc-temp_med-Parahip',
                'S_collat_transv_ant',
                'G_oc-temp_lat-fusifor',
                'S_oc-temp_med_and_Lingual',
                'G_temp_sup-Plan_polar',
                'Pole_temporal',
    
                # parietal:
                'S_parieto_occipital',
                'S_subparietal',
                'G_precuneus',
                'G_parietal_sup',
                'S_intrapariet_and_P_trans',
                'G_pariet_inf-Angular',
                'S_interm_prim-Jensen',
                'G_and_S_paracentral',
                'S_postcentral',
                'G_postcentral',
                'S_central',
                'G_pariet_inf-Supramar',
                'G_and_S_subcentral',
    
                # insular:
                'S_circular_insula_sup',
                'G_insular_short',
                'S_circular_insula_inf',
                'G_Ins_lg_and_S_cent_ins',
                'S_circular_insula_ant',
                
                # cingulate:
                'G_cingul-Post-ventral',
                'S_pericallosal',
                'G_cingul-Post-dorsal',
                'S_cingul-Marginalis',
                'G_and_S_cingul-Mid-Post',
                'G_and_S_cingul-Mid-Ant',
                'G_and_S_cingul-Ant',
                
                # frontal:
                'G_precentral',
                'S_precentral-sup-part',
                'S_precentral-inf-part',
                'G_front_sup',
                'S_front_sup',
                'G_front_middle',
                'S_front_middle',
                'S_front_inf',
                'G_front_inf-Opercular',
                'G_front_inf-Triangul',
                'S_orbital_lateral',
                'Lat_Fis-ant-Horizont',
                'Lat_Fis-ant-Vertical',
                'G_front_inf-Orbital',
                'G_and_S_transv_frontopol',
                'G_and_S_frontomargin',
                'G_orbital',
                'S_orbital-H_Shaped',
                'S_orbital_med-olfact',
                'G_rectus',
                'S_suborbital',
                'G_subcallosal',
                ]
    
        for dv in ['var', 'cov', 'cor', 'mean']:
            for signal_type in ['baseline', 'phasic']:
                for condition in ['high', 'low', 'all']: 
                

                    if (signal_type == 'baseline')*(condition == 'low'):
                        print 'baseline low'
                        ind = np.concatenate(self.pupil_b_l_ind)
                    elif (signal_type == 'baseline')*(condition == 'high'):
                        print 'baseline high'
                        ind = np.concatenate(self.pupil_b_h_ind)
                    elif (signal_type == 'phasic')*(condition == 'low'):
                        print 'phasic low'
                        ind = np.concatenate(self.pupil_l_ind)
                    elif (signal_type == 'phasic')*(condition == 'high'):
                        print 'phasic high'
                        ind = np.concatenate(self.pupil_h_ind)
                    else:
                        print 'all'
                        ind = np.ones(len(self.data_frame), dtype=bool)

                    if bin_by == 'all':
                        corrmat = np.zeros((len(rois),len(rois)))
                        p_mat = np.zeros((len(rois),len(rois)))
                        for j in range(len(rois)):
                            roi_1 = rois[j]

                            corrmat[j,j] = 1
                            p_mat[j,j] = 0

                            for k in range(j+1, len(rois)):

                                roi_2 = rois[k]

                                roi_data_across_1 = np.zeros((len(self.subjects),bins))
                                roi_data_across_2 = np.zeros((len(self.subjects),bins))
                                for i in range(len(self.subjects)):
                                    
                                    subj_ind = np.array((self.data_frame.subject == self.subjects[i]), dtype=bool)

                                    if signal_type == 'baseline':
                                        bin_indices = np.array_split(np.argsort(np.array(self.data_frame[subj_ind*ind][roi_1+'_b'])), bins)
                                        roi_data_1 = np.array(self.data_frame[subj_ind*ind][roi_1+'_b'])
                                        roi_data_2 = np.array(self.data_frame[subj_ind*ind][roi_2+'_b'])
                                    if signal_type == 'phasic':
                                        bin_indices = np.array_split(np.argsort(np.array(self.data_frame[subj_ind*ind][roi_1])), bins)
                                        roi_data_1 = np.array(self.data_frame[subj_ind*ind][roi_1])
                                        roi_data_2 = np.array(self.data_frame[subj_ind*ind][roi_2])
                                    for b in range(bins):
                                        try:
                                            roi_data_across_1[i,b] = np.mean(roi_data_1[bin_indices[b]])
                                            roi_data_across_2[i,b] = np.mean(roi_data_2[bin_indices[b]])
                                        except:
                                            shell()
                                roi_data_across_1 = roi_data_across_1.mean(axis=0)
                                roi_data_across_2 = roi_data_across_2.mean(axis=0)
                            
                                # variance:
                                if dv == 'var':
                                    corr = np.mean(np.var(roi_data_across_1), np.var(roi_data_across_2))
                            
                                # covariance:
                                if dv == 'cov':
                                    corr = np.cov(roi_data_across_1, roi_data_across_2)
                            
                                # correlation:
                                if dv == 'cor':
                                    corr, p_value = sp.stats.pearsonr(roi_data_across_1, roi_data_across_2)
                            
                                # mean activation:
                                if dv == 'mean':
                                    corr = np.mean(np.mean(roi_data_across_1), np.mean(roi_data_across_2))
                            
                                corrmat[j,k] = corr
                                p_mat[j,k] = p_value
                            
                            
                    else:
                        C = np.zeros( (len(self.subjects), len(rois),bins) )
                        for i in range(len(self.subjects)):
                        
                            subj_ind = np.array((self.data_frame.subject == self.subjects[i]), dtype=bool)
                        
                            if signal_type == 'baseline':
                                bin_indices = np.array_split(np.argsort(np.array(self.data_frame[subj_ind*ind][bin_by+'_b'])), bins)
                            if signal_type == 'phasic':
                                bin_indices = np.array_split(np.argsort(np.array(self.data_frame[subj_ind*ind][bin_by])), bins)
                            for j, roi in enumerate(rois):
                                if signal_type == 'baseline':
                                    roi_data = np.array(self.data_frame[subj_ind*ind][roi+'_b'])
                                if signal_type == 'phasic':
                                    roi_data = np.array(self.data_frame[subj_ind*ind][roi])
                                for b in range(bins):
                                    C[i,j,b] = np.mean(roi_data[bin_indices[b]])
                        C = np.mean(C, axis=0).T
                        if partial:
                            corrmat, p_mat = myfuncs.corr_matrix_partial(C)
                        else:
                            corrmat, p_mat = myfuncs.corr_matrix(C)

                    # fdr correction:
                    p_mat = mne.stats.fdr_correction(p_mat, 0.05)[1]
                
                    # save:
                    np.save(os.path.join(self.data_folder, 'correlation', 'cormat_{}_{}_{}_{}.npy'.format(signal_type, condition, dv, bin_by)), corrmat)
                    np.save(os.path.join(self.data_folder, 'correlation', 'p_mat_{}_{}_{}_{}.npy'.format(signal_type, condition, dv, bin_by)), p_mat)
                
                    # load:
                    corrmat = np.load(os.path.join(self.data_folder, 'correlation', 'cormat_{}_{}_{}_{}.npy'.format(signal_type, condition, dv, bin_by)))
                    p_mat = np.load(os.path.join(self.data_folder, 'correlation', 'p_mat_{}_{}_{}_{}.npy'.format(signal_type, condition, dv, bin_by)))
                
                    # plotting:
                    mask =  np.tri(corrmat.shape[0], k=0)
                    # corrmat = np.ma.masked_where(mask+(p_mat>0.05), corrmat)
                    corrmat = np.ma.masked_where(mask, corrmat)
                    p_mat = np.ma.masked_where(mask.T, p_mat)
                    fig_spacing = (2,1)
                    fig = plt.figure(figsize=(8.3,11.7))
                    ax1 = plt.subplot2grid(fig_spacing, (0, 0), colspan=2)
                    axes = [ax1]
                    plot_nr = 0
                    ax = axes[plot_nr]
                    if bins > 50:
                        im = ax.pcolormesh(corrmat, cmap='bwr',)
                    else:
                        im = ax.pcolormesh(corrmat, cmap='bwr',)
                    # if partial:
                    #     im = ax.pcolormesh(corrmat_m, cmap='bwr', vmin=-0.5, vmax=0.5)
                    # else:
                    #     im = ax.pcolormesh(corrmat_m, cmap='hot', vmin=0, vmax=0.5)
                    ax.patch.set_hatch('x')
                    ax.set_xlim(xmax=len(rois))
                    ax.set_ylim(ymax=len(rois))
                    ax.set_yticks(arange(0.5,len(rois)+.5))
                    ax.set_xticks(arange(0.5,len(rois)+.5))
                    ax.set_yticklabels(rois,)
                    ax.set_xticklabels(rois,rotation=90)
                    ax.patch.set_hatch('x')
                    # for i in range(p_mat.shape[0]):
                    #     for j in range(p_mat.shape[1]):
                    #         if p_mat[i,j] or (p_mat[i,j] == 0):
                    #             star1 = ''
                    #             if p_mat[i,j] < 0.05:
                    #                 star1 = '*'
                    #             if p_mat[i,j] < 0.01:
                    #                 star1 = '**'
                    #             if p_mat[i,j] < 0.001:
                    #                 star1 = '***'
                    #             ax.text(i+0.5,j+0.5,star1,size=12,horizontalalignment='center',verticalalignment='center',)

                    fig.colorbar(im)
                    plt.tight_layout()
                    if partial:
                        fig.savefig(os.path.join(self.figure_folder, 'correlation', 'big_matrix', '7_correlation_matrix_all_rois_{}_binned_partial_{}_{}_{}_{}.pdf'.format(bin_by, signal_type, condition, dv, bins)))
                    else:
                        fig.savefig(os.path.join(self.figure_folder, 'correlation', 'big_matrix', '7_correlation_matrix_all_rois_{}_binned_{}_{}_{}_{}.pdf'.format(bin_by, signal_type, condition, dv, bins)))
            
                corrs = []
                for condition in ['high', 'low']:
                    # load:
                    corrmat = np.load(os.path.join(self.data_folder, 'correlation', 'cormat_{}_{}_{}_{}.npy'.format(signal_type, condition, dv, bin_by)))
                    p_mat = np.load(os.path.join(self.data_folder, 'correlation', 'p_mat_{}_{}_{}_{}.npy'.format(signal_type, condition, dv, bin_by)))
                    corrs.append(corrmat)
        
                corrmat = abs(corrs[0])-abs(corrs[1])
                # corrmat = corrs[0]-corrs[1]
        
                # plotting:
                mask =  np.tri(corrmat.shape[0], k=0)
                # corrmat = np.ma.masked_where(mask+(p_mat>0.05), corrmat)
                corrmat = np.ma.masked_where(mask, corrmat)
                # p_mat = np.ma.masked_where(mask.T, p_mat)
                fig_spacing = (2,1)
                fig = plt.figure(figsize=(8.3,11.7))
                ax1 = plt.subplot2grid(fig_spacing, (0, 0), colspan=2)
                axes = [ax1]
                plot_nr = 0
                ax = axes[plot_nr]
                im = ax.pcolormesh(corrmat, cmap='bwr',)
                # if partial:
                #     im = ax.pcolormesh(corrmat_m, cmap='bwr', vmin=-0.5, vmax=0.5)
                # else:
                #     im = ax.pcolormesh(corrmat_m, cmap='hot', vmin=0, vmax=0.5)
                ax.patch.set_hatch('x')
                ax.set_xlim(xmax=len(rois))
                ax.set_ylim(ymax=len(rois))
                ax.set_yticks(arange(0.5,len(rois)+.5))
                ax.set_xticks(arange(0.5,len(rois)+.5))
                ax.set_yticklabels(rois,)
                ax.set_xticklabels(rois,rotation=90)
                ax.patch.set_hatch('x')
                # for i in range(p_mat.shape[0]):
                #     for j in range(p_mat.shape[1]):
                #         if p_mat[i,j] or (p_mat[i,j] == 0):
                #             star1 = ''
                #             if p_mat[i,j] < 0.05:
                #                 star1 = '*'
                #             if p_mat[i,j] < 0.01:
                #                 star1 = '**'
                #             if p_mat[i,j] < 0.001:
                #                 star1 = '***'
                #             ax.text(i+0.5,j+0.5,star1,size=12,horizontalalignment='center',verticalalignment='center',)

                fig.colorbar(im)
                plt.tight_layout()
                if partial:
                    fig.savefig(os.path.join(self.figure_folder, 'correlation', 'big_matrix', '7_correlation_matrix_all_rois_{}_binned_partial_{}_contrast_{}_{}.pdf'.format(bin_by, signal_type, dv, bins)))
                else:
                    fig.savefig(os.path.join(self.figure_folder, 'correlation', 'big_matrix', '7_correlation_matrix_all_rois_{}_binned_{}_contrast_{}_{}.pdf'.format(bin_by, signal_type, dv, bins)))
    
    def multivariate_localizer(self):
        
        import matplotlib.gridspec as gridspec
        
        nr_voxels = 100
        line = 50
        start = 3
        end = 8
        
        time_array_cw_sorted = []
        time_array_ccw_sorted = []
        diff_sorted = []
        
        for i, s in enumerate(self.subjects):
            
            matrices = glob.glob(os.path.join(self.base_dir, 'data', 'across', 'multivariate', 'matrix_cw_{}*'.format(s)))
            time_array_cw_sorted.append( np.dstack([np.load(m) for m in matrices]).mean(axis=-1) )
            matrices = glob.glob(os.path.join(self.base_dir, 'data', 'across', 'multivariate', 'matrix_ccw_{}*'.format(s)))
            time_array_ccw_sorted.append( np.dstack([np.load(m) for m in matrices]).mean(axis=-1) )
            matrices = glob.glob(os.path.join(self.base_dir, 'data', 'across', 'multivariate', 'matrix_diff_{}*'.format(s)))
            diff_sorted.append( np.dstack([np.load(m) for m in matrices]).mean(axis=-1) )
        
        time_array_cw_sorted = np.dstack(time_array_cw_sorted).mean(axis=-1)
        time_array_ccw_sorted = np.dstack(time_array_ccw_sorted).mean(axis=-1)
        diff_sorted = np.dstack(diff_sorted).mean(axis=-1)
        
        x = np.arange(0,24,2)
        
        fig = plt.figure(figsize=(9, 4)) 
        gs = gridspec.GridSpec(1, 6, width_ratios=[3,1,3,1,3,1]) 
        ax = plt.subplot(gs[0])
        im = ax.pcolormesh(x, np.arange(nr_voxels), time_array_cw_sorted.T, cmap='bwr', vmin=-3, vmax=3)
        plt.axhline(line)
        plt.xlim(x[0],x[-1])
        plt.title('clock-wise')
        plt.xlabel('Time (s)')
        plt.ylabel('Voxel # (sorted)')
        fig.colorbar(im, ax=ax, fraction=0.25, pad=0.1)
        ax = plt.subplot(gs[1])
        ax.plot(time_array_cw_sorted.T[:,start:end].mean(axis=1), np.arange(nr_voxels))
        ax.set_xlabel('signal')
        
        ax = plt.subplot(gs[2])
        im = ax.pcolormesh(x, np.arange(nr_voxels), time_array_ccw_sorted.T, cmap='bwr', vmin=-3, vmax=3)
        plt.axhline(line)
        plt.xlim(x[0],x[-1])
        plt.title('counter-clock-wise')
        plt.xlabel('Time (s)')
        # plt.ylabel('Voxel # (sorted)')
        fig.colorbar(im, ax=ax, fraction=0.25, pad=0.1)
        ax = plt.subplot(gs[3])
        ax.plot(time_array_ccw_sorted.T[:,start:end].mean(axis=1), np.arange(nr_voxels))
        ax.set_xlabel('signal')
        
        ax = plt.subplot(gs[4])
        im = ax.pcolormesh(x, np.arange(nr_voxels), diff_sorted.T, cmap='bwr', vmin=-3, vmax=3)
        plt.axhline(line)
        plt.xlim(x[0],x[-1])
        plt.title('cw - ccw')
        plt.xlabel('Time (s)')
        # plt.ylabel('Voxel # (sorted)')
        fig.colorbar(im, ax=ax, fraction=0.25, pad=0.1)
        ax = plt.subplot(gs[5])
        plt.axhline(line)
        plt.axvline(0, lw=0.5)
        ax.plot(diff_sorted.T[:,start:end].mean(axis=1), np.arange(nr_voxels))
        ax.set_xlabel('signal')        
        
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder, 'multivariate', 'across_loc_matrices.pdf'))
        
    def wholebrain_cortex_behaviour(self, data_type, comparisons=['yes'], type_response='mean'):
        
        brain_mask = np.array(NiftiImage('/home/shared/Niels_UvA/brainstem_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz').data, dtype=bool)
        cortex_mask = np.array(NiftiImage('/home/shared/Niels_UvA/brainstem_masks/harvard_oxford/volume_1.nii.gz').data, dtype=bool) + np.array(NiftiImage('/home/shared/Niels_UvA/brainstem_masks/harvard_oxford/volume_12.nii.gz').data, dtype=bool)
        LC_mask = np.array(NiftiImage('/home/shared/Niels_UvA/brainstem_masks/LC_standard_1.nii.gz').data, dtype=bool)
        
        for c in comparisons:
            for time_locked in ['stim_locked']:
                
                phasics_all = np.vstack([np.load(os.path.join(self.data_folder, 'event_related_average_wb', 'whole_brain_{}_{}_{}_{}_{}.npy'.format('std', data_type, time_locked, subj, c)))[3] for subj in self.subjects])
                bad_voxels = np.zeros(phasics_all.shape[1], dtype=bool)
                for i in range(len(self.subjects)):
                    bad_voxels = bad_voxels + (phasics_all[i] > 25)
                    plt.hist(phasics_all[i][-bad_voxels], bins=100)
                
                if type_response == 'snr':
                    baselines_all = np.vstack([np.load(os.path.join(self.data_folder, 'event_related_average_wb', 'whole_brain_{}_{}_{}_{}_{}.npy'.format('mean', data_type, time_locked, subj, c)))[0] for subj in self.subjects]) / np.vstack([np.load(os.path.join(self.data_folder, 'event_related_average_wb', 'whole_brain_{}_{}_{}_{}_{}.npy'.format('std', data_type, time_locked, subj, c)))[0] for subj in self.subjects])
                    baselines_1 = np.vstack([np.load(os.path.join(self.data_folder, 'event_related_average_wb', 'whole_brain_{}_{}_{}_{}_{}.npy'.format('mean', data_type, time_locked, subj, c)))[1] for subj in self.subjects]) / np.vstack([np.load(os.path.join(self.data_folder, 'event_related_average_wb', 'whole_brain_{}_{}_{}_{}_{}.npy'.format('std', data_type, time_locked, subj, c)))[1] for subj in self.subjects])
                    baselines_2 = np.vstack([np.load(os.path.join(self.data_folder, 'event_related_average_wb', 'whole_brain_{}_{}_{}_{}_{}.npy'.format('mean', data_type, time_locked, subj, c)))[2] for subj in self.subjects]) / np.vstack([np.load(os.path.join(self.data_folder, 'event_related_average_wb', 'whole_brain_{}_{}_{}_{}_{}.npy'.format('std', data_type, time_locked, subj, c)))[2] for subj in self.subjects])
                    phasics_all = np.vstack([np.load(os.path.join(self.data_folder, 'event_related_average_wb', 'whole_brain_{}_{}_{}_{}_{}.npy'.format('mean', data_type, time_locked, subj, c)))[3] for subj in self.subjects]) / np.vstack([np.load(os.path.join(self.data_folder, 'event_related_average_wb', 'whole_brain_{}_{}_{}_{}_{}.npy'.format('std', data_type, time_locked, subj, c)))[3] for subj in self.subjects])
                    phasics_1 = np.vstack([np.load(os.path.join(self.data_folder, 'event_related_average_wb', 'whole_brain_{}_{}_{}_{}_{}.npy'.format('mean', data_type, time_locked, subj, c)))[4] for subj in self.subjects]) / np.vstack([np.load(os.path.join(self.data_folder, 'event_related_average_wb', 'whole_brain_{}_{}_{}_{}_{}.npy'.format('std', data_type, time_locked, subj, c)))[4] for subj in self.subjects])
                    phasics_2 = np.vstack([np.load(os.path.join(self.data_folder, 'event_related_average_wb', 'whole_brain_{}_{}_{}_{}_{}.npy'.format('mean', data_type, time_locked, subj, c)))[5] for subj in self.subjects]) / np.vstack([np.load(os.path.join(self.data_folder, 'event_related_average_wb', 'whole_brain_{}_{}_{}_{}_{}.npy'.format('std', data_type, time_locked, subj, c)))[5] for subj in self.subjects])
                else:
                    baselines_all = np.vstack([np.load(os.path.join(self.data_folder, 'event_related_average_wb', 'whole_brain_{}_{}_{}_{}_{}.npy'.format(type_response, data_type, time_locked, subj, c)))[0] for subj in self.subjects])
                    baselines_1 = np.vstack([np.load(os.path.join(self.data_folder, 'event_related_average_wb', 'whole_brain_{}_{}_{}_{}_{}.npy'.format(type_response, data_type, time_locked, subj, c)))[1] for subj in self.subjects])
                    baselines_2 = np.vstack([np.load(os.path.join(self.data_folder, 'event_related_average_wb', 'whole_brain_{}_{}_{}_{}_{}.npy'.format(type_response, data_type, time_locked, subj, c)))[2] for subj in self.subjects])
                    phasics_all = np.vstack([np.load(os.path.join(self.data_folder, 'event_related_average_wb', 'whole_brain_{}_{}_{}_{}_{}.npy'.format(type_response, data_type, time_locked, subj, c)))[3] for subj in self.subjects])
                    phasics_1 = np.vstack([np.load(os.path.join(self.data_folder, 'event_related_average_wb', 'whole_brain_{}_{}_{}_{}_{}.npy'.format(type_response, data_type, time_locked, subj, c)))[4] for subj in self.subjects])
                    phasics_2 = np.vstack([np.load(os.path.join(self.data_folder, 'event_related_average_wb', 'whole_brain_{}_{}_{}_{}_{}.npy'.format(type_response, data_type, time_locked, subj, c)))[5] for subj in self.subjects])
                
                # correlate voxel by voxel:
                
                phasics_d = phasics_2-phasics_1
                
                c_d = self.criterion_hi-self.criterion_lo
                r = np.array([sp.stats.pearsonr(phasics_d[:,i], c_d)[0] for i in range(phasics_d.shape[1])])
                p = np.array([sp.stats.pearsonr(phasics_d[:,i], c_d)[1] for i in range(phasics_d.shape[1])])
                p = 1-p
                
                results = np.zeros((1,brain_mask.shape[0],brain_mask.shape[1],brain_mask.shape[2]))
                results[0,brain_mask] = r
                res_nii_file = NiftiImage(results)
                res_nii_file.header = NiftiImage('/home/shared/Niels_UvA/brainstem_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz').header
                res_nii_file.save(os.path.join(self.data_folder, 'event_related_average_wb', 'behaviour_corr_{}_{}_{}_{}_combined.nii.gz'.format(data_type, c, time_locked, type_response)))
                
                results = np.zeros((1,brain_mask.shape[0],brain_mask.shape[1],brain_mask.shape[2]))
                results[0,brain_mask] = p
                res_nii_file = NiftiImage(results)
                res_nii_file.header = NiftiImage('/home/shared/Niels_UvA/brainstem_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz').header
                res_nii_file.save(os.path.join(self.data_folder, 'event_related_average_wb', 'behaviour_corr_{}_{}_{}_{}_combined_p.nii.gz'.format(data_type, c, time_locked, type_response)))
                
                # correlation to signal_type contrast:
                cortex_voxels = cortex_mask[brain_mask] * brain_mask[brain_mask]
                epi_box = os.path.join('/home/shared/Niels_UvA/brainstem_masks', 'epi_box.nii.gz')
                epi_box = np.array(NiftiImage(epi_box).data, dtype=bool)
                
                # correlate to SNR map:
                
                # shell()
                
                ind = np.isnan(phasics_all[:,cortex_voxels].mean(axis=0)) + np.isnan(phasics_d[:,cortex_voxels].mean(axis=0)) + np.isnan(r[cortex_voxels]) + bad_voxels[cortex_voxels] + (-epi_box[brain_mask][cortex_voxels])
                print sp.stats.pearsonr(phasics_all.mean(axis=0)[cortex_voxels][-ind], r[cortex_voxels][-ind])
                print sp.stats.pearsonr(phasics_d.mean(axis=0)[cortex_voxels][-ind], r[cortex_voxels][-ind])
                
                shell()
                
                fig = plt.figure(figsize=(2,2))
                ax = fig.add_subplot(111)
                myfuncs.correlation_plot(r[cortex_voxels][-ind], phasics_all.mean(axis=0)[cortex_voxels][-ind], ax=ax, dots=False, line=True)
                ax.set_ylabel('Correlation\nd SNR to d drift criterion')
                ax.set_xlabel('SNR')
                sns.despine(offset=10, trim=True)
                plt.tight_layout()
                fig.savefig(os.path.join(self.figure_folder, 'cortex', 'SNR_drift_correlation_1.pdf'))
                
                
                # R2.append(Rs)
                #
                #
                #
                #
                # # correlate SNR in task postive voxels:
                #
                #
                # shell()
                #
                #
                # for i in range(9):
                #
                #     occ_cortex = NiftiImage(os.path.join('/home/shared/Niels_UvA/brainstem_masks', 'MNI-prob-2mm.nii.gz')).data
                #     occ_cortex = occ_cortex[i,:,:,:]
                #     occ_cortex[occ_cortex>1] = 1
                #     occ_cortex = np.array(occ_cortex, dtype=bool)
                #     occ_cortex = occ_cortex[brain_mask]
                #
                #     threshold_file = NiftiImage(os.path.join('/home/shared/Niels_UvA/Visual_UvA/data/across/event_related_average_wb', 'clean_MNI_pupil_stim_locked_mean_phasic_all_tfce_corrp_tstat1.nii.gz')).data
                #     threshold_file = threshold_file[brain_mask]
                #     ind = (threshold_file>0.9995) * (phasics_d.mean(axis=0)>0) * occ_cortex
                #
                #     values = phasics_d[:,ind].mean(axis=1)
                #
                #     myfuncs.correlation_plot(values, self.criterion_hi-self.criterion_lo, line=True)
                #     plt.show()
                
                    
                
                
                
                
                
                
                
                
                
                
                
                
                
                

        
        
        
    
    def unpack_atlas(self, threshold=50):
        
        # nifti = NiftiImage('/home/shared/Niels_UvA/brainstem_masks/harvard_oxford/HarvardOxford-sub-prob-2mm.nii.gz')
        # a = nifti.data
        # for i in range(a.shape[0]):
        #     aa = a[i,:,:,:]
        #     aa[aa<threshold] = 0
        #     aa[aa>=threshold] = 1
        #     b = NiftiImage(aa)
        #     b.header = nifti.header
        #     b.save('/home/shared/Niels_UvA/brainstem_masks/harvard_oxford/volume_{}.nii.gz'.format(i))
        #
        #
        #
        # nifti_1 = NiftiImage('/home/shared/Niels_UvA/brainstem_masks/brainstem_bin.nii.gz')
        # nifti_2 = NiftiImage('/home/shared/Niels_UvA/brainstem_masks/mean_fullMB.nii.gz')
        # a = np.array(nifti_1.data, dtype=bool) + np.array(nifti_2.data, dtype=bool)
        # a = sp.ndimage.binary_dilation(np.array(a, dtype=int), iterations=2)
        #
        # b = NiftiImage(np.array(a, dtype=int))
        # b.header = nifti.header
        # b.save('/home/shared/Niels_UvA/brainstem_masks/brainstem_bin_2.nii.gz'.format(i))
        
        # shell()
        
        shell()
        
        nifti = '/home/shared/Niels_UvA/brainstem_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz'
        a = nib.load(nifti).get_data()
        top = (38,70)
        front = (95,47)
        bottom = (84,21)
        line_1 = np.linspace(top[1],front[1],front[0]-top[0])
        line_2 = np.linspace(front[1],bottom[1],front[0]-bottom[0])
        for i in range(front[0]-top[0]):
            a[:,top[0]+i,floor(line_1[i]):] = 0
        for i in range(front[0]-bottom[0]):
            a[:,front[0]-i,:floor(line_2[i])] = 0
        a[:,front[0]:,:] = 0
        
        b = nib.Nifti1Image(a, affine=nib.load(nifti).get_affine(), header=nib.load(nifti).get_header())
        nib.save(b, '/home/shared/Niels_UvA/brainstem_masks/epi_box.nii.gz')
        
        a[47:,:,:] = 0
        b = nib.Nifti1Image(a, affine=nib.load(nifti).get_affine(), header=nib.load(nifti).get_header())
        nib.save(b, '/home/shared/Niels_UvA/brainstem_masks/epi_box_half.nii.gz')
        
        nifti = '/home/shared/Niels_UvA/brainstem_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz'
        a = nib.load(nifti).get_data()
        a[47:,:,:] = 0
        b = nib.Nifti1Image(a, affine=nib.load(nifti).get_affine(), header=nib.load(nifti).get_header())
        nib.save(b, '/home/shared/Niels_UvA/brainstem_masks/MNI152_T1_2mm_brain_mask_bin_half.nii.gz')
        
        # nifti = '/home/shared/Niels_UvA/brainstem_masks/MNI152_T1_2mm_brain_mask_bin.nii.gz'
        # a = NiftiImage(nifti).data
        #
        # top = (38,70)
        # front = (95,45)
        # bottom = (84,21)
        # line_1 = np.linspace(top[1],front[1],front[0]-top[0])
        # line_2 = np.linspace(front[1],bottom[1],front[0]-bottom[0])
        # for i in range(front[0]-top[0]):
        #     a[floor(line_1[i]):,top[0]+i,:] = 0
        # for i in range(front[0]-bottom[0]):
        #     a[:floor(line_2[i]),front[0]-i,:] = 0
        # a[:,front[0]:,:] = 0
        #
        # b = NiftiImage(np.array(a, dtype=int))
        # b.header = nifti.header
        # b.save('/home/shared/Niels_UvA/brainstem_masks/epi_box.nii.gz')
        
        a = np.ones(nifti_1.data.shape, dtype=bool)
        for i in range(len(self.subjects)):
            a = a * np.array(NiftiImage(os.path.join('/home/shared/Niels_UvA/Visual_UvA/data', self.subjects[i], 'FSL_files', 'masks', 'brain_mask_MNI_1.nii.gz')).data, dtype=bool)
            a = a * np.array(NiftiImage(os.path.join('/home/shared/Niels_UvA/Visual_UvA/data', self.subjects[i], 'FSL_files', 'masks', 'brain_mask_MNI_2.nii.gz')).data, dtype=bool)
        b = NiftiImage(np.array(a, dtype=int))
        b.header = nifti.header
        b.save('/home/shared/Niels_UvA/brainstem_masks/epi_box_subjects.nii.gz')
        
    def combine_pupil_BOLD_correlation(self,):
        
        brain_mask = np.array(NiftiImage('/home/shared/Niels_UvA/LC/MNI152_T1_2mm_brain_mask_bin.nii.gz').data, dtype=bool)
        
        for k in range(3):
            
            r = []
            p = []
            for i in range(len(self.subjects)):
            
                r.append( np.expand_dims(NiftiImage(os.path.join(self.data_folder, 'correlation', 'pupil_correlation_r_{}.nii.gz'.format(self.subjects[i]))).data[k,:,:,:],0) )
                p.append( np.expand_dims(NiftiImage(os.path.join(self.data_folder, 'correlation', 'pupil_correlation_p_{}.nii.gz'.format(self.subjects[i]))).data[k,:,:,:],0) )
            
            r_combined = np.mean(np.vstack(r), axis=0)
            p_combined = np.mean(np.vstack(p), axis=0)
            r_combined[-brain_mask] = 0
            p_combined[-brain_mask] = 0
        
            # save:
            res_nii_file = NiftiImage(r_combined)
            res_nii_file.header = NiftiImage(os.path.join(self.data_folder, 'correlation', 'pupil_correlation_p_{}.nii.gz'.format(self.subjects[i]))).header
            res_nii_file.save(os.path.join(self.data_folder, 'correlation', '{}_across_correlation_r.nii.gz'.format(k)))
        
            # t stats:
            r_T, r_p = sp.stats.ttest_1samp(np.vstack(r), 0)
            r_T[-brain_mask] = 0
        
            # save:
            res_nii_file = NiftiImage(r_T)
            res_nii_file.header = NiftiImage(os.path.join(self.data_folder, 'correlation', 'pupil_correlation_p_{}.nii.gz'.format(self.subjects[i]))).header
            res_nii_file.save(os.path.join(self.data_folder, 'correlation', '{}_across_correlation_r_T.nii.gz'.format(k)))
    
    def GLM_level3(self):
        
        thisFeatFile = '/home/shared/Niels_UvA/Visual_UvA/analysis/feat_tasks/level3.fsf'
        cope_names = ['hit', 'hit_h', 'hit_l', 'fa', 'fa_h', 'fa_l', 'miss', 'miss_h', 'miss_l', 'cr', 'cr_h', 'cr_l', 'yes', 'yes_h', 'yes_l', 'no', 'no_h', 'no_l', 'correct', 'correct_h', 'correct_l', 'error', 'error_h', 'error_l', 'present', 'present_h', 'present_l', 'absent', 'absent_h', 'absent_l',  'yesVSno', 'yesVSno_h', 'yesVSno_l', 'noVSyes', 'noVSyes_h', 'noVSyes_l', 'correctVSerror', 'correctVSerror_h', 'correctVSerror_l', 'errorVScorrect', 'errorVScorrect_h', 'errorVScorrect_l', 'presentVSabsent', 'presentVSabsent_h', 'presentVSabsent_l', 'absentVSpresent', 'absentVSpresent_h', 'absentVSpresent_l']
        for i, cope in enumerate(range(1,49)):
            # remove previous feat directories
            try:
                os.system('rm -rf ' + os.path.join(self.base_dir, 'data', 'across', 'glm', 'level3_{}_{}.fsf'.format(cope, cope_names[i])))
                os.system('rm -rf ' + os.path.join(self.base_dir, 'data', 'across', 'glm', 'level3_{}_{}.gfeat'.format(cope, cope_names[i])))
            except OSError:
                pass
            
            # this is where we start up fsl feat analysis after creating the feat .fsf file and the like
            REDict = {
            '---OUTPUT---':'{}_{}'.format(cope, cope_names[i]),
            '---COPE_NR---':str(cope),
            }
                
            featFileName = os.path.join(self.base_dir, 'data', 'across', 'glm', 'level3_{}_{}.fsf'.format(cope, cope_names[i]))
            featOp = FEATOperator(inputObject = thisFeatFile)
            featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = True )
            # run feat
            featOp.execute()
    
    def GLM_level3_pupil(self):
        
        thisFeatFile = '/home/shared/Niels_UvA/Visual_UvA/analysis/feat_tasks_pupil/level3.fsf'
        
        # cope_names = ['pupil_hVS_pupil_l', 'pupil_lVS_pupil_h']
        # for i, cope in enumerate([4,5]):

        cope_names = ['pupil_h', 'pupil_l', 'pupil_hVS_pupil_l', 'pupil_h_b', 'pupil_l_b', 'pupil_h_bVS_pupil_l_b']
        for i, cope in enumerate([1,2,4,6,7,9]):
        
            # remove previous feat directories
            try:
                os.system('rm -rf ' + os.path.join(self.base_dir, 'data', 'across', 'glm', 'boxcar2', 'level3_pupil_{}_{}.fsf'.format(cope, cope_names[i])))
                os.system('rm -rf ' + os.path.join(self.base_dir, 'data', 'across', 'glm', 'boxcar2', 'level3_pupil_{}_{}.gfeat'.format(cope, cope_names[i])))
            except OSError:
                pass
            
            # this is where we start up fsl feat analysis after creating the feat .fsf file and the like
            REDict = {
            '---OUTPUT---':'boxcar2/level3_pupil_{}_{}'.format(cope, cope_names[i]),
            '---COPE_NR---':str(cope),
            '---BOXCAR---':str(2),
            }
            
            featFileName = os.path.join(self.base_dir, 'data', 'across', 'glm', 'boxcar2', 'level3_pupil_{}_{}.fsf'.format(cope, cope_names[i]))
            featOp = FEATOperator(inputObject = thisFeatFile)
            featOp.configure( REDict = REDict, featFileName = featFileName, waitForExecute = True )
            # run feat
            featOp.execute()
    
    