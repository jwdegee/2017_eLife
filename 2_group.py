#!/usr/bin/env python
# encoding: utf-8
"""
================================================
Created by Jan Willem de Gee on 2014-06-01.     
Copyright (c) 2009 jwdegee. All rights reserved.
================================================
"""

import os, sys, datetime
import subprocess, logging

import scipy as sp
import scipy.stats as stats
import numpy as np
import matplotlib.pylab as pl

from IPython import embed as shell

this_raw_folder = '/home/raw_data/UvA/Donner_lab/2017_eLife/1_fMRI_yesno_visual/'
this_project_folder = '/home/shared/UvA/Niels_UvA/Visual_UvA2/'

analysisFolder = os.path.join(this_project_folder, 'analysis')
sys.path.append( analysisFolder )
sys.path.append( os.environ['ANALYSIS_HOME'] )

from Tools.Sessions import *
from Tools.Subjects.Subject import *
from Tools.Run import *
from Tools.Projects.Project import *

from defs_fmri_group import defs_fmri_group
import defs_pupil

# SUBJECTS:
# ---------
subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10', 'sub-11', 'sub-12', 'sub-13', 'sub-14']
nr_sessions = [2,2,2,3,2,2,2,2,2,2,2,2,2,2]

# PUPIL:
# ------
# pupilAnalysisSessionAcross = defs_pupil.pupilAnalysesAcross(subjects=subjects, experiment_name='pupil_yes_no', sample_rate_new=20, project_directory=this_project_folder)
# pupilAnalysisSessionAcross.behavior_choice()
# pupilAnalysisSessionAcross.behavior_normalized(prepare=False)
# pupilAnalysisSessionAcross.SDT_correlation(bins=5)
# pupilAnalysisSessionAcross.rt_distributions()
# pupilAnalysisSessionAcross.drift_diffusion()
# pupilAnalysisSessionAcross.average_pupil_responses()
# pupilAnalysisSessionAcross.grand_average_pupil_response()
# pupilAnalysisSessionAcross.SDT_across_time()
# pupilAnalysisSessionAcross.correlation_PPRa_BPD()
# pupilAnalysisSessionAcross.GLM_betas()

# fMRI:
# -----
for split_by in ['pupil_d',]:
# for split_by in ['yes',]:
    fMRI_across = defs_fmri_group(subjects=subjects, nr_sessions=nr_sessions, base_dir=os.path.join(this_project_folder), split_by=split_by)
    rois = [
        'V1_center',
        'V1_surround',
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
        
        # 'S_intrapariet_and_P_trans',
        # 'G_and_S_cingul-Mid-Ant',
        # 'S_temporal_sup',
        # 'G_precuneus',
        
        # 'S_front_inf',
        # 'S_orbital_med-olfact',
        
        # cortex:
        # 'Pole_occipital',
        # 'cortex_dilated_mask',
        # 'G_front_middle',
        # 'G_and_S_cingul-Ant',
        # 'S_circular_insula_ant',
        # 'G_orbital',
        # 'S_orbital-H_Shaped',
        # 'S_orbital_lateral',
        # 'G_front_inf-Orbital',
        # 'S_orbital_med-olfact',

        # # brainstem:
        # 'LC_standard_2',
        # 'LC_standard_1',
        # 'mean_SN',
        # 'mean_VTA',
        # 'basal_forebrain_4',
        # 'basal_forebrain_123',
        # 'sup_col_jw',
        # 'inf_col_jw',
        # '4th_ventricle',
        # 'LC_JW',
        # 'LC_JW_nn',
        # 'AAN_VTA',
        # 'AAN_PAG',
        # 'AAN_PBC',
        # 'AAN_PO',
        # 'AAN_PPN',
        # 'AAN_LC',
        # 'AAN_MR',
        # 'AAN_MRF',
        # 'AAN_DR',
        ]
    
    # fMRI_across.surface_labels_to_vol()
    # fMRI_across.correlation_per_subject(rois=rois, data_type='clean_False')
    # fMRI_across.single_trial_correlation_ITI(data_type='clean_False')
    # fMRI_across.single_trial_correlation(data_type='clean_4th_ventricle')
    # fMRI_across.single_trial_correlation2(data_type='clean_False')
    # fMRI_across.brainstem_to_behaviour(data_type='clean_4th_ventricle')
    # fMRI_across.single_trial_multiple_regression(data_type='clean_4th_ventricle')
    # fMRI_across.rates_across_trials(data_type='clean_False')
    # fMRI_across.sequential_effects(data_type='clean_False')
    # fMRI_across.correlation_bars()
    # fMRI_across.correlation_bars_single_trial(rois=['superior_colliculus', 'basal_forebrain', 'mean_fullMB', 'LC_JW',], data_type='clean_4th_ventricle')
    # fMRI_across.correlation_bars_binned_all_rois(bins=385, data_type='clean_False')
    # fMRI_across.BRAINSTEM_correlation_bars()
    # fMRI_across.BRAINSTEM_choice(data_type='clean_4th_ventricle')
    # fMRI_across.BRAINSTEM_bar_plots(data_type='clean_4th_ventricle')
    # fMRI_across.BRAINSTEM_correlation_matrix_single_trial(rois=['sup_col_jw', 'inf_col_jw', 'mean_SN', 'mean_VTA', 'LC_JW', 'basal_forebrain_123', 'basal_forebrain_4'], data_type='clean_4th_ventricle', partial=False)
    # fMRI_across.correlation_matrix_single_trial_all_rois(data_type='clean_False', partial=False)
    
    # fMRI_across.M1_connectivity(data_type='clean_False')
    # fMRI_across.multivariate_localizer()
    # fMRI_across.multivariate_task(data_type='clean_False')
    # fMRI_across.ANOVA_LC()
    bins_by = ['all', 'pupil']
    bins = [385, 50]
    partials = [True, False]
    for b in bins:
        # fMRI_across.correlation_bars_binned(bins=b, rois=['superior_colliculus', 'mean_SN', 'mean_VTA', 'LC_JW', 'basal_forebrain'], data_type='clean_4th_ventricle')
        for bin_by in bins_by:
            # for partial in partials:
                # fMRI_across.correlation_matrix_binned(bin_by=bin_by, bins=b, partial=partial, rois=['G_and_S_cingul-Mid-Ant', 'superior_colliculus', 'basal_forebrain', 'mean_fullMB', 'LC_JW',], data_type='clean_4th_ventricle',)
                pass
    
    # fMRI_across.unpack_atlas()
    # comparisons = ['pupil', 'pupil_b', 'yes', 'correct', 'present',]
    # data_types = ['clean_False', 'clean_4th_ventricle', 'psc_False', 'psc_4th_ventricle']
    # stratifications = [False]
    stratifications = [False,]
    # data_types = ['clean_4th_ventricle']
    data_types = ['clean_False']
    # data_types = ['psc_False']
    for roi in rois:
        for stratification in stratifications:
            for data_type in data_types:
                # fMRI_across.plot_mean_bars(roi=roi, data_type=data_type, type_response='mean',)
                # fMRI_across.plot_event_average_all_trials(roi=roi, data_type=data_type, smooth=False, event_average=True, type_response='mean', stratification=stratification, project_out=False)
                # fMRI_across.plot_event_average(roi=roi, data_type=data_type, smooth=False, event_average=True, type_response='mean', stratification=stratification, project_out=False)
                pass
    
    # fMRI_across.WHOLEBRAIN_event_related_average_prepare(data_type='clean_MNI', measure='mean',)
    # fMRI_across.WHOLEBRAIN_event_related_average_conditions(data_type='clean_MNI', measure='mean',)
    # fMRI_across.WHOLEBRAIN_correlation(data_type='clean_MNI', measure='mean', source='pupil_d')
    # fMRI_across.WHOLEBRAIN_combine_searchlight(data_type='clean_MNI')
    # fMRI_across.WHOLEBRAIN_lateralization_per_session(data_type='clean_MNI_smooth', measure='snr', prepare=False)
    # fMRI_across.WHOLEBRAIN_event_related_average_plots(data_type='clean_MNI_smooth', measure='snr')
    # fMRI_across.WHOLEBRAIN_correlation(data_type='clean_MNI_smooth', measure='mean', source='pupil_d')
    # fMRI_across.WHOLEBRAIN_correlation(data_type='clean_MNI_smooth', measure='mean', source='V123')
    # fMRI_across.WHOLEBRAIN_correlation(data_type='clean_MNI_smooth', measure='mean', source='V3')
    # fMRI_across.WHOLEBRAIN_correlation(data_type='clean_MNI_smooth', measure='mean', source='V123_center_info')
    # fMRI_across.WHOLEBRAIN_correlation(data_type='clean_MNI_smooth', measure='mean', source='pupil_criterion')
    # fMRI_across.WHOLEBRAIN_correlation(data_type='clean_MNI_smooth', measure='mean', source='pupil_dprime')
    # fMRI_across.WHOLEBRAIN_correlation(data_type='clean_MNI_smooth', measure='mean', source='BOLD_criterion')
    # fMRI_across.WHOLEBRAIN_correlation(data_type='clean_MNI_smooth', measure='mean', source='BOLD_dprime')
    # fMRI_across.WHOLEBRAIN_correlation(data_type='clean_MNI_smooth', measure='mean', source='BOLD_present')
    # fMRI_across.WHOLEBRAIN_correlation(data_type='clean_MNI_smooth', measure='mean', source='BOLD_choice')
    
    # fMRI_across.WHOLEBRAIN_correlation(data_type='clean_MNI_smooth', measure='mean', source='BOLD_moderated_mediation')
    # fMRI_across.WHOLEBRAIN_correlation(data_type='clean_MNI_smooth', measure='mean', source='BOLD_PPI')
    
    # fMRI_across.WHOLEBRAIN_correlation_per_session(data_type='clean_MNI_smooth', measure='mean', source='BOLD_choice')
    # fMRI_across.WHOLEBRAIN_noise_correlation_make_dataframe(data_type='clean_False')
    # fMRI_across.WHOLEBRAIN_noise_correlation(data_type='clean_False', partial=False)
    # fMRI_across.WHOLEBRAIN_noise_correlation_bias(data_type='clean_False', partial=False)
    # fMRI_across.mutual_inhibition(data_type='clean_MNI', measure='mean', prepare=False)
    
    # fMRI_across.VISUAL_snr(data_type='clean_False')
    # fMRI_across.VISUAL_noise_correlation(data_type='clean_False', partial=False)
    # fMRI_across.MULTIVARIATE_make_lineplot(data_type='clean_MNI')
    # fMRI_across.MULTIVARIATE_plot_patterns(data_type='clean_MNI')
    # fMRI_across.MULTIVARIATE_make_dataframe(data_type='clean_MNI', prepare=False)
    # fMRI_across.MULTIVARIATE_add_combined_signal()
    # fMRI_across.CHOICE_SIGNALS_plots_2()
    # fMRI_across.CHOICE_SIGNALS_stim_TPR_interactions()
    # fMRI_across.CHOICE_SIGNALS_plots_stratified()
    # fMRI_across.CHOICE_SIGNALS_SDT()
    # fMRI_across.CHOICE_SIGNALS_logistic_regressions()
    # fMRI_across.CHOICE_SIGNALS_choice_probability()
    # fMRI_across.CHOICE_SIGNALS_choice_probability_plot()
    # fMRI_across.CHOICE_SIGNALS_choice_probability_pupil_plot()
    # fMRI_across.CHOICE_SIGNALS_ROC_curve()
    # fMRI_across.CHOICE_SIGNALS_correlation_matrix()
    # fMRI_across.CHOICE_SIGNALS_coupling_2()
    # fMRI_across.CHOICE_SIGNALS_bars_to_criterion()
    # fMRI_across.CHOICE_SIGNALS_behavioural_correlation()
    # fMRI_across.CHOICE_SIGNALS_M1_correlation()
    # fMRI_across.CHOICE_SIGNALS_PPI_analysis()
    # fMRI_across.CHOICE_SIGNALS_mediation_analysis()
    # fMRI_across.CHOICE_SIGNALS_to_choice()
    # fMRI_across.CHOICE_SIGNALS_variability()
    # fMRI_across.DDM_dataframe()
    # fMRI_across.simulation()
    