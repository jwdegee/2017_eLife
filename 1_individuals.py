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
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
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

import defs_pupil
from defs_fmri_individuals import defs_fmri_individuals

# -----------------
# Comments:       -
# -----------------

# subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10', 'sub-11', 'sub-12', 'sub-13', 'sub-14']
subjects = ['sub-01']
for which_subject in subjects:
    
    if which_subject == 'sub-04':
        sessions = [0,1,2]
    else:
        sessions = [1,2]
        
    edfs = []
    for s in sessions:

        def runWholeSession( rDA, session ):

            # get some variables in place:
            for r in rDA:
                thisRun = Run( **r )
                presentSession.addRun(thisRun)
            session.parcelateConditions()
            session.parallelize = True

            # get some more variables in place:
            global edfs
            edfs.append( [rDA[i]['eyeLinkFilePath'] for i in range(len(rDA)) if rDA[i]['condition'] == 'task'] )
            if s == 2:
                edfs = list(np.concatenate(edfs))
                session_nr = [int(f.split('ses-')[-1][:2]) for f in edfs]
                aliases = []
                for i in range(len(edfs)):
                    aliases.append('detection_{}_{}'.format(i+1, session_nr[i]))

            # some convenience functions:
            # ---------------------------
            # session.rename(conditions = ['loc', 'task'], postFix_old = ['NB', 'mcf'], postFix_new = ['B0', 'mcf'])
            # session.remove(conditions = ['loc', 'task'], postFix = ['mcf', 'phys'])
            # session.copy_files()
            # session.register_SPM_anatomy()
            # session.copy_freesurfer_labels()
            # nr_voxels.append( session.number_voxels(roi='LC_JW') )

            # ----------------------------
            # Pupil:                     -
            # ----------------------------

            if s == 2:
                pupilPreprocessSession = defs_pupil.pupilPreprocessSession(subject=Subject(which_subject, '?', None, None, None), experiment_name='pupil_yes_no', experiment_nr=2, version=3, sample_rate_new=50, project_directory=this_project_folder)
                pupilPreprocessSession.import_raw_data(edf_files=edfs, aliases=aliases)
                pupilPreprocessSession.delete_hdf5()
                pupilPreprocessSession.import_all_data(aliases)
                for alias in aliases:
                    pupilPreprocessSession.process_runs(alias, artifact_rejection='not_strict', create_pupil_BOLD_regressor=False)
                    pass
                pupilPreprocessSession.process_across_runs(aliases, create_pupil_BOLD_regressor=False)

                # within subjects stats:
                pupilAnalysisSession = defs_pupil.pupilAnalyses(subject=Subject(which_subject, '?', None, None, None), experiment_name='pupil_yes_no', experiment_nr=2, sample_rate_new=50, project_directory=this_project_folder, aliases=aliases)
                pupilAnalysisSession.trial_wise_pupil()

            # ----------------------------
            # fMRI:                      -
            # ----------------------------

            # preprocessing first steps:
            # --------------------------
            # session.setupFiles(rawBase=presentSubject.initials, process_eyelink_file=False)
            # session.registerSession(contrast='t2', deskull=False)
            # session.B0_unwarping(conditions=['loc', 'task'], wfs=12.223, etl=35.0, acceleration=3.0)
            # session.motionCorrectFunctionals(postFix=['B0'], further_args = ' -dof 7') # session.motionCorrectFunctionals(postFix=['NB'], further_args=' -dof 7')
            # session.resample_epis2(conditions=['task',], postFix=['B0', 'mcf'])
            # session.rescaleFunctionals(operations = ['sgtf'], filterFreqs={'highpass': 50.0, 'lowpass': -1.0}, funcPostFix = ['B0', 'mcf'], mask_file = None) # 0.01 Hz = 1 / (2.0 [TR] x 50 [samples])
            # session.rename(conditions = ['loc', 'task'], postFix_old = ['B0', 'mcf', 'sgtf'], postFix_new = ['B0', 'mcf', 'sgtf', '0.01'])
            # session.rescaleFunctionals(operations = ['percentsignalchange'], funcPostFix = ['B0', 'mcf', 'sgtf', '0.01'], mask_file = None)

            # stimulus timings:
            # -----------------
            # session.all_timings()

            # ROIs:
            # -----
            # session.createMasksFromFreeSurferLabels(annot=False, annotFile='aparc.a2009s', cortex=True)
            # session.create_dilated_cortical_mask(dilation_sd=0.25, label='cortex')
            # session.create_brain_masks()
            # session.transform_standard_brainmasks(min_nr_voxels=12)
            # session.register_TSE_epi()
            # session.grab_LC_masks() # --> requires LCs and Ventricles to be drawn.
            # session.transform_LC_mask_TSE2Func(min_nr_voxels=12)
            
            # preprocessing first steps continued:
            # ------------------------------------
            # session.retroicorFSL(conditions=['task', 'loc'], postFix=['B0', 'mcf'], threshold=1.5, nr_dummies=8, sample_rate=496, gradient_direction='y', prepare=True, run=False)
            # session.concatenate_data_runs()
            # session.GLM_nuisance()
            # session.clean_to_MNI()

            # MASKS:
            # ------

            rois = np.array([

                    # RETINOPY:
                    # --------
                    'V1_center',
                    'V1_surround',
                    'V2_center',
                    'V2_surround',
                    'V3_center',
                    'V3_surround',

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
                    'LC_JW_nn',
                    'inf_col_jw',
                    'sup_col_jw',
                    '4th_ventricle',

                    # CHOICE areas:
                    # ------------
                    'lr_aIPS',
                    'lr_PCeS',
                    'lr_M1',
                    'sl_IPL',
                    'sl_SPL1',
                    'sl_SPL2',
                    'sl_pIns',

                    # PARCELATION:
                    # ------------

                    # # visual:
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

                    # # visual:
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
                ])

            # session.combine_rois_across_hemispheres(rois=rois)
            # session.transfrom_destrieux_to_MNI(rois=rois)
            # session.create_session_rois(rois=rois)

            # analysis of interest:
            # ---------------------
            
            # localizer:
            # session.GLM_localizer()
            
            # V123:
            # session.V123_univariate() # selects voxels based on localizer
            # session.V123_multivariate() # selects voxels based on task
            
            # rois:
            # session.ROI_event_related_average(rois=rois, data_type='clean')
            
            # whole brain:
            # data_type = 'clean_MNI'
            # session.WHOLEBRAIN_event_related_average(data_type=data_type,)
            # session.WHOLEBRAIN_correlation(data_type=data_type,)
            # session.WHOLEBRAIN_searchlight_decoding(data_type=data_type,)
            
        # for testing;
        if __name__ == '__main__':

########################################################################################################################################################################################################

            if which_subject == 'sub-01':
                # subject information
                initials = 'sub-01'
                firstName = 'sub-01'
                standardFSID = 'AV_120414'
                birthdate = datetime.date( 1900, 01, 01 )
                labelFolderOfPreference = '2014_custom'
                presentSubject = Subject( initials, firstName, birthdate, standardFSID, labelFolderOfPreference )
                presentProject = Project( 'yes_no_fmri', subject = presentSubject, base_dir = os.path.join(this_project_folder, 'data') )
                sessionID = 'yes_no_fmri' + presentSubject.initials

                if s == 1:
                    sessionDate = datetime.date(2014, 6, 15)
                    sj_session1 = 'ses-01'
                if s == 2:
                    sessionDate = datetime.date(2014, 6, 26)
                    sj_session2 = 'ses-02'

                presentSession = defs_fmri_individuals(sessionID, sessionDate, presentProject, presentSubject)

                try:
                    os.mkdir(os.path.join(this_project_folder, 'data', initials))
                except OSError:
                    presentSession.logger.debug('output folders already exist')

                # ----------------------
                # Decision tasks:      -
                # ----------------------

                if s == 1:
                    runDecisionArray = [
                        # B0:
                        {'ID' : 1, 'scanType': 'inplane_anat', 'condition': 'B0_anat_mag', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'fmap', 'sub-01_ses-01_magnitude.nii.gz' ),},
                        {'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'B0_anat_phs', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'fmap', 'sub-01_ses-01_phasediff.nii.gz' ),},
                        # Whole brain TSE:
                        {'ID' : 3, 'scanType': 'inplane_anat', 'condition': 'TSE_anat_whole', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-01_ses-01_TSEwb.nii.gz' ),},
                        # Neuromalanin:
                        {'ID' : 4, 'scanType': 'inplane_anat', 'condition': 'TSE_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-01_ses-01_TSEori.nii.gz' ),},
                        # T1:
                        {'ID' : 5, 'scanType': 'inplane_anat', 'condition': 'T1_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-01_ses-01_T1w.nii.gz' ),},
                        # T2:
                        {'ID' : 6, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-01_ses-01_T2w.nii.gz' ),},
                        # Localizer:
                        {'ID' : 7, 'scanType': 'epi_bold', 'condition': 'loc', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-01_ses-01_task-localizer_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-01_ses-01_task-localizer_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-01_ses-01_task-localizer_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-01_ses-01_task-localizer_run-1_eyedata.edf' ),
                            },
                        # Decision tasks:
                        {'ID' : 8, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-01_ses-01_task-yesno_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-01_ses-01_task-yesno_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-01_ses-01_task-yesno_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-01_ses-01_task-yesno_run-1_eyedata.edf' ),
                            },
                        {'ID' : 9, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-01_ses-01_task-yesno_run-2_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-01_ses-01_task-yesno_run-2_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-01_ses-01_task-yesno_run-2_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-01_ses-01_task-yesno_run-2_eyedata.edf' ),
                            },
                        {'ID' : 10, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-01_ses-01_task-yesno_run-3_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-01_ses-01_task-yesno_run-3_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-01_ses-01_task-yesno_run-3_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-01_ses-01_task-yesno_run-3_eyedata.edf' ),
                            },
                        {'ID' : 11, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-01_ses-01_task-yesno_run-4_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-01_ses-01_task-yesno_run-4_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-01_ses-01_task-yesno_run-4_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-01_ses-01_task-yesno_run-4_eyedata.edf' ),
                            },
                        {'ID' : 12, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-01_ses-01_task-yesno_run-5_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-01_ses-01_task-yesno_run-5_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-01_ses-01_task-yesno_run-5_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-01_ses-01_task-yesno_run-5_eyedata.edf' ),
                            },
                        {'ID' : 13, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-01_ses-01_task-yesno_run-6_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-01_ses-01_task-yesno_run-6_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-01_ses-01_task-yesno_run-6_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-01_ses-01_task-yesno_run-6_eyedata.edf' ),
                            },
                        ]
                if s == 2:
                    runDecisionArray = [
                        # B0:
                        {'ID' : 1, 'scanType': 'inplane_anat', 'condition': 'B0_anat_mag', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'fmap', 'sub-01_ses-02_magnitude.nii.gz' ),},
                        {'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'B0_anat_phs', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'fmap', 'sub-01_ses-02_phasediff.nii.gz' ),},
                        # Whole brain TSE:
                        {'ID' : 3, 'scanType': 'inplane_anat', 'condition': 'TSE_anat_whole', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-01_ses-02_TSEwb.nii.gz' ),},
                        # Neuromalanin:
                        {'ID' : 4, 'scanType': 'inplane_anat', 'condition': 'TSE_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-01_ses-02_TSEori.nii.gz' ),},
                        # T1:
                        {'ID' : 5, 'scanType': 'inplane_anat', 'condition': 'T1_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-01_ses-02_T1w.nii.gz' ),},
                        # T2:
                        {'ID' : 6, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-01_ses-02_T2w.nii.gz' ),},
                        # Localizer:
                        {'ID' : 7, 'scanType': 'epi_bold', 'condition': 'loc', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-01_ses-02_task-localizer_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-01_ses-02_task-localizer_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-01_ses-02_task-localizer_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-01_ses-02_task-localizer_run-1_eyedata.edf' ),
                            },
                        # Decision tasks:
                        {'ID' : 8, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-01_ses-02_task-yesno_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-01_ses-02_task-yesno_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-01_ses-02_task-yesno_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-01_ses-02_task-yesno_run-1_eyedata.edf' ),
                            },
                        {'ID' : 9, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-01_ses-02_task-yesno_run-2_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-01_ses-02_task-yesno_run-2_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-01_ses-02_task-yesno_run-2_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-01_ses-02_task-yesno_run-2_eyedata.edf' ),
                            },
                        {'ID' : 10, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-01_ses-02_task-yesno_run-3_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-01_ses-02_task-yesno_run-3_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-01_ses-02_task-yesno_run-3_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-01_ses-02_task-yesno_run-3_eyedata.edf' ),
                            },
                        {'ID' : 11, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-01_ses-02_task-yesno_run-4_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-01_ses-02_task-yesno_run-4_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-01_ses-02_task-yesno_run-4_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-01_ses-02_task-yesno_run-4_eyedata.edf' ),
                            },
                        {'ID' : 12, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-01_ses-02_task-yesno_run-5_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-01_ses-02_task-yesno_run-5_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-01_ses-02_task-yesno_run-5_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-01_ses-02_task-yesno_run-5_eyedata.edf' ),
                            },
                        {'ID' : 13, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-01_ses-02_task-yesno_run-6_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-01_ses-02_task-yesno_run-6_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-01_ses-02_task-yesno_run-6_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-01_ses-02_task-yesno_run-6_eyedata.edf' ),
                            },
                        ]

                # ----------------------
                # Initialise session   -
                # ----------------------

                runWholeSession( runDecisionArray, presentSession )

    ########################################################################################################################################################################################################

            if which_subject == 'sub-02':
                # subject information
                initials = 'sub-02'
                firstName = 'sub-02'
                standardFSID = 'BL_120514'
                birthdate = datetime.date( 1900, 01, 01 )
                labelFolderOfPreference = '2014_custom'
                presentSubject = Subject( initials, firstName, birthdate, standardFSID, labelFolderOfPreference )
                presentProject = Project( 'yes_no_fmri', subject = presentSubject, base_dir = os.path.join(this_project_folder, 'data') )
                sessionID = 'yes_no_fmri' + presentSubject.initials

                if s == 1:
                    sessionDate = datetime.date(2014, 6, 4)
                    sj_session1 = 'ses-01'
                if s == 2:
                    sessionDate = datetime.date(2014, 6, 14)
                    sj_session2 = 'ses-02'

                presentSession = defs_fmri_individuals(sessionID, sessionDate, presentProject, presentSubject)

                try:
                    os.mkdir(os.path.join(this_project_folder, 'data', initials))
                except OSError:
                    presentSession.logger.debug('output folders already exist')

                # ----------------------
                # Decision tasks:      -
                # ----------------------

                if s == 1:
                    runDecisionArray = [
                        # B0:
                        {'ID' : 1, 'scanType': 'inplane_anat', 'condition': 'B0_anat_mag', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'fmap', 'sub-02_ses-01_magnitude.nii.gz' ),},
                        {'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'B0_anat_phs', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'fmap', 'sub-02_ses-01_phasediff.nii.gz' ),},
                        # Whole brain TSE:
                        {'ID' : 3, 'scanType': 'inplane_anat', 'condition': 'TSE_anat_whole', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-02_ses-01_TSEwb.nii.gz' ),},
                        # Neuromalanin:
                        {'ID' : 4, 'scanType': 'inplane_anat', 'condition': 'TSE_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-02_ses-01_TSEori.nii.gz' ),},
                        # T1:
                        {'ID' : 5, 'scanType': 'inplane_anat', 'condition': 'T1_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-02_ses-01_T1w.nii.gz' ),},
                        # T2:
                        {'ID' : 6, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-02_ses-01_T2w.nii.gz' ),},
                        # Localizer:
                        {'ID' : 7, 'scanType': 'epi_bold', 'condition': 'loc', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-02_ses-01_task-localizer_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-02_ses-01_task-localizer_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-02_ses-01_task-localizer_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-02_ses-01_task-localizer_run-1_eyedata.edf' ),
                            },
                        # Decision tasks:
                        {'ID' : 8, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-02_ses-01_task-yesno_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-02_ses-01_task-yesno_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-02_ses-01_task-yesno_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-02_ses-01_task-yesno_run-1_eyedata.edf' ),
                            },
                        {'ID' : 9, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-02_ses-01_task-yesno_run-2_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-02_ses-01_task-yesno_run-2_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-02_ses-01_task-yesno_run-2_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-02_ses-01_task-yesno_run-2_eyedata.edf' ),
                            },
                        {'ID' : 10, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-02_ses-01_task-yesno_run-3_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-02_ses-01_task-yesno_run-3_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-02_ses-01_task-yesno_run-3_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-02_ses-01_task-yesno_run-3_eyedata.edf' ),
                            },
                        {'ID' : 11, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-02_ses-01_task-yesno_run-4_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-02_ses-01_task-yesno_run-4_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-02_ses-01_task-yesno_run-4_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-02_ses-01_task-yesno_run-4_eyedata.edf' ),
                            },
                        {'ID' : 12, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-02_ses-01_task-yesno_run-5_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-02_ses-01_task-yesno_run-5_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-02_ses-01_task-yesno_run-5_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-02_ses-01_task-yesno_run-5_eyedata.edf' ),
                            },
                        {'ID' : 13, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-02_ses-01_task-yesno_run-6_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-02_ses-01_task-yesno_run-6_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-02_ses-01_task-yesno_run-6_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-02_ses-01_task-yesno_run-6_eyedata.edf' ),
                            },
                        ]
                if s == 2:
                    runDecisionArray = [
                        # B0:
                        {'ID' : 1, 'scanType': 'inplane_anat', 'condition': 'B0_anat_mag', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'fmap', 'sub-02_ses-02_magnitude.nii.gz' ),},
                        {'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'B0_anat_phs', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'fmap', 'sub-02_ses-02_phasediff.nii.gz' ),},
                        # Whole brain TSE:
                        {'ID' : 3, 'scanType': 'inplane_anat', 'condition': 'TSE_anat_whole', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-02_ses-02_TSEwb.nii.gz' ),},
                        # Neuromalanin:
                        {'ID' : 4, 'scanType': 'inplane_anat', 'condition': 'TSE_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-02_ses-02_TSEori.nii.gz' ),},
                        # T1:
                        {'ID' : 5, 'scanType': 'inplane_anat', 'condition': 'T1_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-02_ses-02_T1w.nii.gz' ),},
                        # T2:
                        {'ID' : 6, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-02_ses-02_T2w.nii.gz' ),},
                        # Localizer:
                        {'ID' : 7, 'scanType': 'epi_bold', 'condition': 'loc', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-02_ses-02_task-localizer_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-02_ses-02_task-localizer_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-02_ses-02_task-localizer_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-02_ses-02_task-localizer_run-1_eyedata.edf' ),
                            },
                        # Decision tasks:
                        {'ID' : 8, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-02_ses-02_task-yesno_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-02_ses-02_task-yesno_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-02_ses-02_task-yesno_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-02_ses-02_task-yesno_run-1_eyedata.edf' ),
                            },
                        {'ID' : 9, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-02_ses-02_task-yesno_run-2_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-02_ses-02_task-yesno_run-2_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-02_ses-02_task-yesno_run-2_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-02_ses-02_task-yesno_run-2_eyedata.edf' ),
                            },
                        {'ID' : 10, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-02_ses-02_task-yesno_run-3_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-02_ses-02_task-yesno_run-3_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-02_ses-02_task-yesno_run-3_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-02_ses-02_task-yesno_run-3_eyedata.edf' ),
                            },
                        {'ID' : 11, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-02_ses-02_task-yesno_run-4_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-02_ses-02_task-yesno_run-4_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-02_ses-02_task-yesno_run-4_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-02_ses-02_task-yesno_run-4_eyedata.edf' ),
                            },
                        {'ID' : 12, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-02_ses-02_task-yesno_run-5_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-02_ses-02_task-yesno_run-5_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-02_ses-02_task-yesno_run-5_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-02_ses-02_task-yesno_run-5_eyedata.edf' ),
                            },
                        {'ID' : 13, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-02_ses-02_task-yesno_run-6_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-02_ses-02_task-yesno_run-6_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-02_ses-02_task-yesno_run-6_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-02_ses-02_task-yesno_run-6_eyedata.edf' ),
                            },
                        ]
                # ----------------------
                # Initialise session   -
                # ----------------------

                runWholeSession( runDecisionArray, presentSession )


    ########################################################################################################################################################################################################

            if which_subject == 'sub-03':
                # subject information
                initials = 'sub-03'
                firstName = 'sub-03'
                standardFSID = 'DE_110412'
                birthdate = datetime.date( 1900, 01, 01 )
                labelFolderOfPreference = '2014_custom'
                presentSubject = Subject( initials, firstName, birthdate, standardFSID, labelFolderOfPreference )
                presentProject = Project( 'yes_no_fmri', subject = presentSubject, base_dir = os.path.join(this_project_folder, 'data') )
                sessionID = 'yes_no_fmri' + presentSubject.initials

                if s == 1:
                    sessionDate = datetime.date(2015, 2, 26)
                    sj_session1 = 'ses-01'
                if s == 2:
                    sessionDate = datetime.date(2015, 3, 2)
                    sj_session2 = 'ses-02'

                presentSession = defs_fmri_individuals(sessionID, sessionDate, presentProject, presentSubject)

                try:
                    os.mkdir(os.path.join(this_project_folder, 'data', initials))
                except OSError:
                    presentSession.logger.debug('output folders already exist')

                # ----------------------
                # Decision tasks:      -
                # ----------------------

                if s == 1:
                    runDecisionArray = [
                        # B0:
                        {'ID' : 1, 'scanType': 'inplane_anat', 'condition': 'B0_anat_mag', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'fmap', 'sub-03_ses-01_magnitude.nii.gz' ),},
                        {'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'B0_anat_phs', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'fmap', 'sub-03_ses-01_phasediff.nii.gz' ),},
                        # Whole brain TSE:
                        {'ID' : 3, 'scanType': 'inplane_anat', 'condition': 'TSE_anat_whole', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-03_ses-01_TSEwb.nii.gz' ),},
                        # Neuromalanin:
                        {'ID' : 4, 'scanType': 'inplane_anat', 'condition': 'TSE_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-03_ses-01_TSEori.nii.gz' ),},
                        # T1:
                        {'ID' : 5, 'scanType': 'inplane_anat', 'condition': 'T1_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-03_ses-01_T1w.nii.gz' ),},
                        # T2:
                        {'ID' : 6, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-03_ses-01_T2w.nii.gz' ),},
                        # Localizer:
                        {'ID' : 7, 'scanType': 'epi_bold', 'condition': 'loc', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-03_ses-01_task-localizer_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-03_ses-01_task-localizer_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-03_ses-01_task-localizer_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-03_ses-01_task-localizer_run-1_eyedata.edf' ),
                            },
                        # Decision tasks:
                        {'ID' : 8, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-03_ses-01_task-yesno_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-03_ses-01_task-yesno_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-03_ses-01_task-yesno_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-03_ses-01_task-yesno_run-1_eyedata.edf' ),
                            },
                        {'ID' : 9, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-03_ses-01_task-yesno_run-2_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-03_ses-01_task-yesno_run-2_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-03_ses-01_task-yesno_run-2_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-03_ses-01_task-yesno_run-2_eyedata.edf' ),
                            },
                        {'ID' : 10, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-03_ses-01_task-yesno_run-3_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-03_ses-01_task-yesno_run-3_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-03_ses-01_task-yesno_run-3_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-03_ses-01_task-yesno_run-3_eyedata.edf' ),
                            },
                        {'ID' : 11, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-03_ses-01_task-yesno_run-4_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-03_ses-01_task-yesno_run-4_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-03_ses-01_task-yesno_run-4_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-03_ses-01_task-yesno_run-4_eyedata.edf' ),
                            },
                        {'ID' : 12, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-03_ses-01_task-yesno_run-5_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-03_ses-01_task-yesno_run-5_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-03_ses-01_task-yesno_run-5_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-03_ses-01_task-yesno_run-5_eyedata.edf' ),
                            },
                        {'ID' : 13, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-03_ses-01_task-yesno_run-6_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-03_ses-01_task-yesno_run-6_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-03_ses-01_task-yesno_run-6_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-03_ses-01_task-yesno_run-6_eyedata.edf' ),
                            },
                        ]
                if s == 2:
                    runDecisionArray = [
                        # B0:
                        {'ID' : 1, 'scanType': 'inplane_anat', 'condition': 'B0_anat_mag', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'fmap', 'sub-03_ses-02_magnitude.nii.gz' ),},
                        {'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'B0_anat_phs', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'fmap', 'sub-03_ses-02_phasediff.nii.gz' ),},
                        # Whole brain TSE:
                        {'ID' : 3, 'scanType': 'inplane_anat', 'condition': 'TSE_anat_whole', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-03_ses-02_TSEwb.nii.gz' ),},
                        # Neuromalanin:
                        {'ID' : 4, 'scanType': 'inplane_anat', 'condition': 'TSE_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-03_ses-02_TSEori.nii.gz' ),},
                        # # T1:
                        # {'ID' : 5, 'scanType': 'inplane_anat', 'condition': 'T1_anat', 'session' : 2,
                        #     'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-03_ses-02_T1w.nii.gz' ),},
                        # T2:
                        {'ID' : 6, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-03_ses-02_T2w.nii.gz' ),},
                        # Localizer:
                        {'ID' : 7, 'scanType': 'epi_bold', 'condition': 'loc', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-03_ses-02_task-localizer_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-03_ses-02_task-localizer_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-03_ses-02_task-localizer_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-03_ses-02_task-localizer_run-1_eyedata.edf' ),
                            },
                        # Decision tasks:
                        {'ID' : 8, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-03_ses-02_task-yesno_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-03_ses-02_task-yesno_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-03_ses-02_task-yesno_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-03_ses-02_task-yesno_run-1_eyedata.edf' ),
                            },
                        {'ID' : 9, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-03_ses-02_task-yesno_run-2_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-03_ses-02_task-yesno_run-2_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-03_ses-02_task-yesno_run-2_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-03_ses-02_task-yesno_run-2_eyedata.edf' ),
                            },
                        {'ID' : 10, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-03_ses-02_task-yesno_run-3_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-03_ses-02_task-yesno_run-3_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-03_ses-02_task-yesno_run-3_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-03_ses-02_task-yesno_run-3_eyedata.edf' ),
                            },
                        {'ID' : 11, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-03_ses-02_task-yesno_run-4_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-03_ses-02_task-yesno_run-4_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-03_ses-02_task-yesno_run-4_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-03_ses-02_task-yesno_run-4_eyedata.edf' ),
                            },
                        {'ID' : 12, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-03_ses-02_task-yesno_run-5_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-03_ses-02_task-yesno_run-5_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-03_ses-02_task-yesno_run-5_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-03_ses-02_task-yesno_run-5_eyedata.edf' ),
                            },
                        {'ID' : 13, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-03_ses-02_task-yesno_run-6_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-03_ses-02_task-yesno_run-6_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-03_ses-02_task-yesno_run-6_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-03_ses-02_task-yesno_run-6_eyedata.edf' ),
                            },
                        ]

                # ----------------------
                # Initialise session   -
                # ----------------------

                runWholeSession( runDecisionArray, presentSession )

    ########################################################################################################################################################################################################

            if which_subject == 'sub-04':

                # subject information
                initials = 'sub-04'
                firstName = 'sub-04'
                standardFSID = 'DL_190414'
                birthdate = datetime.date( 1900, 01, 01 )
                labelFolderOfPreference = '2014_custom'
                presentSubject = Subject( initials, firstName, birthdate, standardFSID, labelFolderOfPreference )
                presentProject = Project( 'yes_no_fmri', subject = presentSubject, base_dir = os.path.join(this_project_folder, 'data') )
                sessionID = 'yes_no_fmri' + presentSubject.initials

                if s == 0:
                    sessionDate = datetime.date(2014, 5, 25)
                    sj_session1 = 'ses-01'
                if s == 1:
                    sessionDate = datetime.date(2014, 6, 6)
                    sj_session2 = 'ses-02'
                if s == 2:
                    sessionDate = datetime.date(2014, 6, 13)
                    sj_session3 = 'ses-03'

                presentSession = defs_fmri_individuals(sessionID, sessionDate, presentProject, presentSubject)

                try:
                    os.mkdir(os.path.join(this_project_folder, 'data', initials))
                except OSError:
                    presentSession.logger.debug('output folders already exist')

                # ----------------------
                # Decision tasks:      -
                # ----------------------

                if s == 1:
                    runDecisionArray = [
                        # B0:
                        {'ID' : 1, 'scanType': 'inplane_anat', 'condition': 'B0_anat_mag', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'fmap', 'sub-04_ses-01_magnitude.nii.gz' ),},
                        {'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'B0_anat_phs', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'fmap', 'sub-04_ses-01_phasediff.nii.gz' ),},
                        # Whole brain TSE:
                        {'ID' : 3, 'scanType': 'inplane_anat', 'condition': 'TSE_anat_whole', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-04_ses-01_TSEwb.nii.gz' ),},
                        # Neuromalanin:
                        {'ID' : 4, 'scanType': 'inplane_anat', 'condition': 'TSE_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-04_ses-01_TSEori.nii.gz' ),},
                        # T1:
                        {'ID' : 5, 'scanType': 'inplane_anat', 'condition': 'T1_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-04_ses-01_T1w.nii.gz' ),},
                        # T2:
                        {'ID' : 6, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-04_ses-01_T2w.nii.gz' ),},
                        # Localizer:
                        {'ID' : 7, 'scanType': 'epi_bold', 'condition': 'loc', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-04_ses-01_task-localizer_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-04_ses-01_task-localizer_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-04_ses-01_task-localizer_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-04_ses-01_task-localizer_run-1_eyedata.edf' ),
                            },
                        # Decision tasks:
                        {'ID' : 8, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-04_ses-01_task-yesno_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-04_ses-01_task-yesno_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-04_ses-01_task-yesno_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-04_ses-01_task-yesno_run-1_eyedata.edf' ),
                            },
                        {'ID' : 9, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-04_ses-01_task-yesno_run-2_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-04_ses-01_task-yesno_run-2_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-04_ses-01_task-yesno_run-2_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-04_ses-01_task-yesno_run-2_eyedata.edf' ),
                            },
                        {'ID' : 10, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-04_ses-01_task-yesno_run-3_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-04_ses-01_task-yesno_run-3_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-04_ses-01_task-yesno_run-3_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-04_ses-01_task-yesno_run-3_eyedata.edf' ),
                            },
                        {'ID' : 11, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-04_ses-01_task-yesno_run-4_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-04_ses-01_task-yesno_run-4_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-04_ses-01_task-yesno_run-4_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-04_ses-01_task-yesno_run-4_eyedata.edf' ),
                            },
                        {'ID' : 12, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-04_ses-01_task-yesno_run-5_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-04_ses-01_task-yesno_run-5_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-04_ses-01_task-yesno_run-5_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-04_ses-01_task-yesno_run-5_eyedata.edf' ),
                            },
                        {'ID' : 13, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-04_ses-01_task-yesno_run-6_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-04_ses-01_task-yesno_run-6_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-04_ses-01_task-yesno_run-6_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-04_ses-01_task-yesno_run-6_eyedata.edf' ),
                            },
                        ]
                if s == 2:
                    runDecisionArray = [
                        # B0:
                        {'ID' : 1, 'scanType': 'inplane_anat', 'condition': 'B0_anat_mag', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'fmap', 'sub-04_ses-02_magnitude.nii.gz' ),},
                        {'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'B0_anat_phs', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'fmap', 'sub-04_ses-02_phasediff.nii.gz' ),},
                        # Whole brain TSE:
                        {'ID' : 3, 'scanType': 'inplane_anat', 'condition': 'TSE_anat_whole', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-04_ses-02_TSEwb.nii.gz' ),},
                        # Neuromalanin:
                        {'ID' : 4, 'scanType': 'inplane_anat', 'condition': 'TSE_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-04_ses-02_TSEori.nii.gz' ),},
                        # T1:
                        {'ID' : 5, 'scanType': 'inplane_anat', 'condition': 'T1_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-04_ses-02_T1w.nii.gz' ),},
                        # T2:
                        {'ID' : 6, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-04_ses-02_T2w.nii.gz' ),},
                        # Localizer:
                        {'ID' : 7, 'scanType': 'epi_bold', 'condition': 'loc', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-02_task-localizer_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-02_task-localizer_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-02_task-localizer_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-02_task-localizer_run-1_eyedata.edf' ),
                            },
                        # Decision tasks:
                        {'ID' : 8, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-02_task-yesno_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-02_task-yesno_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-02_task-yesno_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-02_task-yesno_run-1_eyedata.edf' ),
                            },
                        {'ID' : 9, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-02_task-yesno_run-2_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-02_task-yesno_run-2_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-02_task-yesno_run-2_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-02_task-yesno_run-2_eyedata.edf' ),
                            },
                        {'ID' : 10, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-02_task-yesno_run-3_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-02_task-yesno_run-3_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-02_task-yesno_run-3_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-02_task-yesno_run-3_eyedata.edf' ),
                            },
                        {'ID' : 11, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-02_task-yesno_run-4_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-02_task-yesno_run-4_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-02_task-yesno_run-4_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-02_task-yesno_run-4_eyedata.edf' ),
                            },
                        {'ID' : 12, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-02_task-yesno_run-5_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-02_task-yesno_run-5_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-02_task-yesno_run-5_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-02_task-yesno_run-5_eyedata.edf' ),
                            },
                        {'ID' : 13, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-02_task-yesno_run-6_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-02_task-yesno_run-6_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-02_task-yesno_run-6_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-02_task-yesno_run-6_eyedata.edf' ),
                            },
                        ]

                if s == 3:
                    runDecisionArray = [
                        # B0:
                        {'ID' : 1, 'scanType': 'inplane_anat', 'condition': 'B0_anat_mag', 'session' : 3,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'fmap', 'sub-04_ses-03_magnitude.nii.gz' ),},
                        {'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'B0_anat_phs', 'session' : 3,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'fmap', 'sub-04_ses-03_phasediff.nii.gz' ),},
                        # Whole brain TSE:
                        {'ID' : 3, 'scanType': 'inplane_anat', 'condition': 'TSE_anat_whole', 'session' : 3,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-04_ses-03_TSEwb.nii.gz' ),},
                        # Neuromalanin:
                        {'ID' : 4, 'scanType': 'inplane_anat', 'condition': 'TSE_anat', 'session' : 3,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-04_ses-03_TSEori.nii.gz' ),},
                        # T1:
                        {'ID' : 5, 'scanType': 'inplane_anat', 'condition': 'T1_anat', 'session' : 3,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-04_ses-03_T1w.nii.gz' ),},
                        # T2:
                        {'ID' : 6, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 'session' : 3,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-04_ses-03_T2w.nii.gz' ),},
                        # Localizer:
                        {'ID' : 7, 'scanType': 'epi_bold', 'condition': 'loc', 'session' : 3,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-03_task-localizer_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-03_task-localizer_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-03_task-localizer_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-03_task-localizer_run-1_eyedata.edf' ),
                            },
                        # Decision tasks:
                        {'ID' : 8, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 3,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-03_task-yesno_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-03_task-yesno_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-03_task-yesno_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-03_task-yesno_run-1_eyedata.edf' ),
                            },
                        {'ID' : 9, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 3,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-03_task-yesno_run-2_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-03_task-yesno_run-2_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-03_task-yesno_run-2_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-03_task-yesno_run-2_eyedata.edf' ),
                            },
                        {'ID' : 10, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 3,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-03_task-yesno_run-3_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-03_task-yesno_run-3_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-03_task-yesno_run-3_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-03_task-yesno_run-3_eyedata.edf' ),
                            },
                        {'ID' : 11, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 3,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-03_task-yesno_run-4_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-03_task-yesno_run-4_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-03_task-yesno_run-4_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-04_ses-03_task-yesno_run-4_eyedata.edf' ),
                            },
                        ]

                # ----------------------
                # Initialise session   -
                # ----------------------

                runWholeSession( runDecisionArray, presentSession )

    ########################################################################################################################################################################################################

            if which_subject == 'sub-05':
                # subject information
                initials = 'sub-05'
                firstName = 'sub-05'
                standardFSID = 'EP_230414'
                birthdate = datetime.date( 1900, 01, 01 )
                labelFolderOfPreference = '2014_custom'
                presentSubject = Subject( initials, firstName, birthdate, standardFSID, labelFolderOfPreference )
                presentProject = Project( 'yes_no_fmri', subject = presentSubject, base_dir = os.path.join(this_project_folder, 'data') )
                sessionID = 'yes_no_fmri' + presentSubject.initials

                if s == 1:
                    sessionDate = datetime.date(2014, 5, 25)
                    sj_session1 = 'ses-01'
                if s == 2:
                    sessionDate = datetime.date(2014, 6, 9)
                    sj_session2 = 'ses-02'

                presentSession = defs_fmri_individuals(sessionID, sessionDate, presentProject, presentSubject)

                try:
                    os.mkdir(os.path.join(this_project_folder, 'data', initials))
                except OSError:
                    presentSession.logger.debug('output folders already exist')

                # ----------------------
                # Decision tasks:      -
                # ----------------------

                if s == 1:
                    runDecisionArray = [
                        # B0:
                        {'ID' : 1, 'scanType': 'inplane_anat', 'condition': 'B0_anat_mag', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'fmap', 'sub-05_ses-01_magnitude.nii.gz' ),},
                        {'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'B0_anat_phs', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'fmap', 'sub-05_ses-01_phasediff.nii.gz' ),},
                        # Whole brain TSE:
                        {'ID' : 3, 'scanType': 'inplane_anat', 'condition': 'TSE_anat_whole', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-05_ses-01_TSEwb.nii.gz' ),},
                        # Neuromalanin:
                        {'ID' : 4, 'scanType': 'inplane_anat', 'condition': 'TSE_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-05_ses-01_TSEori.nii.gz' ),},
                        # T1:
                        {'ID' : 5, 'scanType': 'inplane_anat', 'condition': 'T1_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-05_ses-01_T1w.nii.gz' ),},
                        # T2:
                        {'ID' : 6, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-05_ses-01_T2w.nii.gz' ),},
                        # Localizer:
                        {'ID' : 7, 'scanType': 'epi_bold', 'condition': 'loc', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-05_ses-01_task-localizer_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-05_ses-01_task-localizer_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-05_ses-01_task-localizer_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-05_ses-01_task-localizer_run-1_eyedata.edf' ),
                            },
                        # Decision tasks:
                        {'ID' : 8, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-05_ses-01_task-yesno_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-05_ses-01_task-yesno_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-05_ses-01_task-yesno_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-05_ses-01_task-yesno_run-1_eyedata.edf' ),
                            },
                        {'ID' : 9, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-05_ses-01_task-yesno_run-2_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-05_ses-01_task-yesno_run-2_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-05_ses-01_task-yesno_run-2_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-05_ses-01_task-yesno_run-2_eyedata.edf' ),
                            },
                        {'ID' : 10, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-05_ses-01_task-yesno_run-3_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-05_ses-01_task-yesno_run-3_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-05_ses-01_task-yesno_run-3_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-05_ses-01_task-yesno_run-3_eyedata.edf' ),
                            },
                        {'ID' : 11, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-05_ses-01_task-yesno_run-4_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-05_ses-01_task-yesno_run-4_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-05_ses-01_task-yesno_run-4_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-05_ses-01_task-yesno_run-4_eyedata.edf' ),
                            },
                        {'ID' : 12, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-05_ses-01_task-yesno_run-5_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-05_ses-01_task-yesno_run-5_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-05_ses-01_task-yesno_run-5_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-05_ses-01_task-yesno_run-5_eyedata.edf' ),
                            },
                        {'ID' : 13, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-05_ses-01_task-yesno_run-6_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-05_ses-01_task-yesno_run-6_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-05_ses-01_task-yesno_run-6_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-05_ses-01_task-yesno_run-6_eyedata.edf' ),
                            },
                        ]
                if s == 2:
                    runDecisionArray = [
                        # B0:
                        {'ID' : 1, 'scanType': 'inplane_anat', 'condition': 'B0_anat_mag', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'fmap', 'sub-05_ses-02_magnitude.nii.gz' ),},
                        {'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'B0_anat_phs', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'fmap', 'sub-05_ses-02_phasediff.nii.gz' ),},
                        # Whole brain TSE:
                        {'ID' : 3, 'scanType': 'inplane_anat', 'condition': 'TSE_anat_whole', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-05_ses-02_TSEwb.nii.gz' ),},
                        # Neuromalanin:
                        {'ID' : 4, 'scanType': 'inplane_anat', 'condition': 'TSE_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-05_ses-02_TSEori.nii.gz' ),},
                        # T1:
                        {'ID' : 5, 'scanType': 'inplane_anat', 'condition': 'T1_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-05_ses-02_T1w.nii.gz' ),},
                        # T2:
                        {'ID' : 6, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-05_ses-02_T2w.nii.gz' ),},
                        # Localizer:
                        {'ID' : 7, 'scanType': 'epi_bold', 'condition': 'loc', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-05_ses-02_task-localizer_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-05_ses-02_task-localizer_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-05_ses-02_task-localizer_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-05_ses-02_task-localizer_run-1_eyedata.edf' ),
                            },
                        # Decision tasks:
                        {'ID' : 8, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-05_ses-02_task-yesno_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-05_ses-02_task-yesno_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-05_ses-02_task-yesno_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-05_ses-02_task-yesno_run-1_eyedata.edf' ),
                            },
                        {'ID' : 9, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-05_ses-02_task-yesno_run-2_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-05_ses-02_task-yesno_run-2_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-05_ses-02_task-yesno_run-2_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-05_ses-02_task-yesno_run-2_eyedata.edf' ),
                            },
                        {'ID' : 10, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-05_ses-02_task-yesno_run-3_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-05_ses-02_task-yesno_run-3_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-05_ses-02_task-yesno_run-3_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-05_ses-02_task-yesno_run-3_eyedata.edf' ),
                            },
                        {'ID' : 11, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-05_ses-02_task-yesno_run-4_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-05_ses-02_task-yesno_run-4_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-05_ses-02_task-yesno_run-4_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-05_ses-02_task-yesno_run-4_eyedata.edf' ),
                            },
                        ]

                # ----------------------
                # Initialise session   -
                # ----------------------

                runWholeSession( runDecisionArray, presentSession )

    ########################################################################################################################################################################################################

            if which_subject == 'sub-06':
                # subject information
                initials = 'sub-06'
                firstName = 'sub-06'
                standardFSID = 'JW_310312'
                birthdate = datetime.date( 1900, 01, 01 )
                labelFolderOfPreference = '2014_custom'
                presentSubject = Subject( initials, firstName, birthdate, standardFSID, labelFolderOfPreference )
                presentProject = Project( 'yes_no_fmri', subject = presentSubject, base_dir = os.path.join(this_project_folder, 'data') )
                sessionID = 'yes_no_fmri' + presentSubject.initials

                if s == 1:
                    sessionDate = datetime.date(2014, 5, 31)
                    sj_session1 = 'ses-01'
                if s == 2:
                    sessionDate = datetime.date(2014, 6, 30)
                    sj_session2 = 'ses-02'

                presentSession = defs_fmri_individuals(sessionID, sessionDate, presentProject, presentSubject)

                try:
                    os.mkdir(os.path.join(this_project_folder, 'data', initials))
                except OSError:
                    presentSession.logger.debug('output folders already exist')

                # ----------------------
                # Decision tasks:      -
                # ----------------------

                if s == 1:
                    runDecisionArray = [
                        # B0:
                        {'ID' : 1, 'scanType': 'inplane_anat', 'condition': 'B0_anat_mag', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'fmap', 'sub-06_ses-01_magnitude.nii.gz' ),},
                        {'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'B0_anat_phs', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'fmap', 'sub-06_ses-01_phasediff.nii.gz' ),},
                        # Whole brain TSE:
                        {'ID' : 3, 'scanType': 'inplane_anat', 'condition': 'TSE_anat_whole', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-06_ses-01_TSEwb.nii.gz' ),},
                        # Neuromalanin:
                        {'ID' : 4, 'scanType': 'inplane_anat', 'condition': 'TSE_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-06_ses-01_TSEori.nii.gz' ),},
                        # T1:
                        {'ID' : 5, 'scanType': 'inplane_anat', 'condition': 'T1_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-06_ses-01_T1w.nii.gz' ),},
                        # T2:
                        {'ID' : 6, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-06_ses-01_T2w.nii.gz' ),},
                        # Localizer:
                        {'ID' : 7, 'scanType': 'epi_bold', 'condition': 'loc', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-06_ses-01_task-localizer_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-06_ses-01_task-localizer_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-06_ses-01_task-localizer_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-06_ses-01_task-localizer_run-1_eyedata.edf' ),
                            },
                        # Decision tasks:
                        {'ID' : 8, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-06_ses-01_task-yesno_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-06_ses-01_task-yesno_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-06_ses-01_task-yesno_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-06_ses-01_task-yesno_run-1_eyedata.edf' ),
                            },
                        {'ID' : 9, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-06_ses-01_task-yesno_run-2_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-06_ses-01_task-yesno_run-2_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-06_ses-01_task-yesno_run-2_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-06_ses-01_task-yesno_run-2_eyedata.edf' ),
                            },
                        {'ID' : 10, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-06_ses-01_task-yesno_run-3_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-06_ses-01_task-yesno_run-3_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-06_ses-01_task-yesno_run-3_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-06_ses-01_task-yesno_run-3_eyedata.edf' ),
                            },
                        {'ID' : 11, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-06_ses-01_task-yesno_run-4_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-06_ses-01_task-yesno_run-4_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-06_ses-01_task-yesno_run-4_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-06_ses-01_task-yesno_run-4_eyedata.edf' ),
                            },
                        {'ID' : 12, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-06_ses-01_task-yesno_run-5_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-06_ses-01_task-yesno_run-5_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-06_ses-01_task-yesno_run-5_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-06_ses-01_task-yesno_run-5_eyedata.edf' ),
                            },
                        {'ID' : 13, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-06_ses-01_task-yesno_run-6_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-06_ses-01_task-yesno_run-6_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-06_ses-01_task-yesno_run-6_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-06_ses-01_task-yesno_run-6_eyedata.edf' ),
                            },
                        ]
                if s == 2:
                    runDecisionArray = [
                        # B0:
                        {'ID' : 1, 'scanType': 'inplane_anat', 'condition': 'B0_anat_mag', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'fmap', 'sub-06_ses-02_magnitude.nii.gz' ),},
                        {'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'B0_anat_phs', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'fmap', 'sub-06_ses-02_phasediff.nii.gz' ),},
                        # Whole brain TSE:
                        {'ID' : 3, 'scanType': 'inplane_anat', 'condition': 'TSE_anat_whole', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-06_ses-02_TSEwb.nii.gz' ),},
                        # Neuromalanin:
                        {'ID' : 4, 'scanType': 'inplane_anat', 'condition': 'TSE_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-06_ses-02_TSEori.nii.gz' ),},
                        # T1:
                        {'ID' : 5, 'scanType': 'inplane_anat', 'condition': 'T1_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-06_ses-02_T1w.nii.gz' ),},
                        # T2:
                        {'ID' : 6, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-06_ses-02_T2w.nii.gz' ),},
                        # Localizer:
                        {'ID' : 7, 'scanType': 'epi_bold', 'condition': 'loc', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-06_ses-02_task-localizer_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-06_ses-02_task-localizer_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-06_ses-02_task-localizer_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-06_ses-02_task-localizer_run-1_eyedata.edf' ),
                            },
                        # Decision tasks:
                        {'ID' : 8, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-06_ses-02_task-yesno_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-06_ses-02_task-yesno_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-06_ses-02_task-yesno_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-06_ses-02_task-yesno_run-1_eyedata.edf' ),
                            },
                        {'ID' : 9, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-06_ses-02_task-yesno_run-2_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-06_ses-02_task-yesno_run-2_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-06_ses-02_task-yesno_run-2_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-06_ses-02_task-yesno_run-2_eyedata.edf' ),
                            },
                        {'ID' : 10, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-06_ses-02_task-yesno_run-3_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-06_ses-02_task-yesno_run-3_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-06_ses-02_task-yesno_run-3_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-06_ses-02_task-yesno_run-3_eyedata.edf' ),
                            },
                        {'ID' : 11, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-06_ses-02_task-yesno_run-4_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-06_ses-02_task-yesno_run-4_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-06_ses-02_task-yesno_run-4_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-06_ses-02_task-yesno_run-4_eyedata.edf' ),
                            },
                        {'ID' : 12, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-06_ses-02_task-yesno_run-5_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-06_ses-02_task-yesno_run-5_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-06_ses-02_task-yesno_run-5_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-06_ses-02_task-yesno_run-5_eyedata.edf' ),
                            },
                        {'ID' : 13, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-06_ses-02_task-yesno_run-6_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-06_ses-02_task-yesno_run-6_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-06_ses-02_task-yesno_run-6_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-06_ses-02_task-yesno_run-6_eyedata.edf' ),
                            },
                        ]

                # ----------------------
                # Initialise session   -
                # ----------------------

                runWholeSession( runDecisionArray, presentSession )

    ########################################################################################################################################################################################################

            if which_subject == 'sub-07':
                # subject information
                initials = 'sub-07'
                firstName = 'sub-07'
                standardFSID = 'JVS_091014'
                birthdate = datetime.date( 1900, 01, 01 )
                labelFolderOfPreference = '2014_custom'
                presentSubject = Subject( initials, firstName, birthdate, standardFSID, labelFolderOfPreference )
                presentProject = Project( 'yes_no_fmri', subject = presentSubject, base_dir = os.path.join(this_project_folder, 'data') )
                sessionID = 'yes_no_fmri' + presentSubject.initials

                if s == 1:
                    sessionDate = datetime.date(2015, 2, 17)
                    sj_session1 = 'ses-01'
                if s == 2:
                    sessionDate = datetime.date(2015, 2, 27)
                    sj_session2 = 'ses-02'

                presentSession = defs_fmri_individuals(sessionID, sessionDate, presentProject, presentSubject)

                try:
                    os.mkdir(os.path.join(this_project_folder, 'data', initials))
                except OSError:
                    presentSession.logger.debug('output folders already exist')

                # ----------------------
                # Decision tasks:      -
                # ----------------------

                if s == 1:
                    runDecisionArray = [
                        # B0:
                        {'ID' : 1, 'scanType': 'inplane_anat', 'condition': 'B0_anat_mag', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'fmap', 'sub-07_ses-01_magnitude.nii.gz' ),},
                        {'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'B0_anat_phs', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'fmap', 'sub-07_ses-01_phasediff.nii.gz' ),},
                        # Whole brain TSE:
                        {'ID' : 3, 'scanType': 'inplane_anat', 'condition': 'TSE_anat_whole', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-07_ses-01_TSEwb.nii.gz' ),},
                        # Neuromalanin:
                        {'ID' : 4, 'scanType': 'inplane_anat', 'condition': 'TSE_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-07_ses-01_TSEori.nii.gz' ),},
                        # T1:
                        {'ID' : 5, 'scanType': 'inplane_anat', 'condition': 'T1_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-07_ses-01_T1w.nii.gz' ),},
                        # T2:
                        {'ID' : 6, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-07_ses-01_T2w.nii.gz' ),},
                        # Localizer:
                        {'ID' : 7, 'scanType': 'epi_bold', 'condition': 'loc', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-07_ses-01_task-localizer_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-07_ses-01_task-localizer_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-07_ses-01_task-localizer_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-07_ses-01_task-localizer_run-1_eyedata.edf' ),
                            },
                        # Decision tasks:
                        {'ID' : 8, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-07_ses-01_task-yesno_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-07_ses-01_task-yesno_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-07_ses-01_task-yesno_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-07_ses-01_task-yesno_run-1_eyedata.edf' ),
                            },
                        {'ID' : 9, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-07_ses-01_task-yesno_run-2_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-07_ses-01_task-yesno_run-2_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-07_ses-01_task-yesno_run-2_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-07_ses-01_task-yesno_run-2_eyedata.edf' ),
                            },
                        {'ID' : 10, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-07_ses-01_task-yesno_run-3_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-07_ses-01_task-yesno_run-3_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-07_ses-01_task-yesno_run-3_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-07_ses-01_task-yesno_run-3_eyedata.edf' ),
                            },
                        {'ID' : 11, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-07_ses-01_task-yesno_run-4_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-07_ses-01_task-yesno_run-4_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-07_ses-01_task-yesno_run-4_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-07_ses-01_task-yesno_run-4_eyedata.edf' ),
                            },
                        {'ID' : 12, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-07_ses-01_task-yesno_run-5_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-07_ses-01_task-yesno_run-5_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-07_ses-01_task-yesno_run-5_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-07_ses-01_task-yesno_run-5_eyedata.edf' ),
                            },
                        {'ID' : 13, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-07_ses-01_task-yesno_run-6_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-07_ses-01_task-yesno_run-6_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-07_ses-01_task-yesno_run-6_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-07_ses-01_task-yesno_run-6_eyedata.edf' ),
                            },
                        ]
                if s == 2:
                    runDecisionArray = [
                        # B0:
                        {'ID' : 1, 'scanType': 'inplane_anat', 'condition': 'B0_anat_mag', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'fmap', 'sub-07_ses-02_magnitude.nii.gz' ),},
                        {'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'B0_anat_phs', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'fmap', 'sub-07_ses-02_phasediff.nii.gz' ),},
                        # Whole brain TSE:
                        {'ID' : 3, 'scanType': 'inplane_anat', 'condition': 'TSE_anat_whole', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-07_ses-02_TSEwb.nii.gz' ),},
                        # Neuromalanin:
                        {'ID' : 4, 'scanType': 'inplane_anat', 'condition': 'TSE_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-07_ses-02_TSEori.nii.gz' ),},
                        # # T1:
                        # {'ID' : 5, 'scanType': 'inplane_anat', 'condition': 'T1_anat', 'session' : 2,
                        #     'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-07_ses-02_T1w.nii.gz' ),},
                        # T2:
                        {'ID' : 6, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-07_ses-02_T2w.nii.gz' ),},
                        # Localizer:
                        {'ID' : 7, 'scanType': 'epi_bold', 'condition': 'loc', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-07_ses-02_task-localizer_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-07_ses-02_task-localizer_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-07_ses-02_task-localizer_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-07_ses-02_task-localizer_run-1_eyedata.edf' ),
                            },
                        # Decision tasks:
                        {'ID' : 8, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-07_ses-02_task-yesno_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-07_ses-02_task-yesno_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-07_ses-02_task-yesno_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-07_ses-02_task-yesno_run-1_eyedata.edf' ),
                            },
                        {'ID' : 9, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-07_ses-02_task-yesno_run-2_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-07_ses-02_task-yesno_run-2_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-07_ses-02_task-yesno_run-2_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-07_ses-02_task-yesno_run-2_eyedata.edf' ),
                            },
                        {'ID' : 10, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-07_ses-02_task-yesno_run-3_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-07_ses-02_task-yesno_run-3_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-07_ses-02_task-yesno_run-3_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-07_ses-02_task-yesno_run-3_eyedata.edf' ),
                            },
                        {'ID' : 11, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-07_ses-02_task-yesno_run-4_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-07_ses-02_task-yesno_run-4_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-07_ses-02_task-yesno_run-4_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-07_ses-02_task-yesno_run-4_eyedata.edf' ),
                            },
                        {'ID' : 12, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-07_ses-02_task-yesno_run-5_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-07_ses-02_task-yesno_run-5_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-07_ses-02_task-yesno_run-5_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-07_ses-02_task-yesno_run-5_eyedata.edf' ),
                            },
                        ]

                # ----------------------
                # Initialise session   -
                # ----------------------

                runWholeSession( runDecisionArray, presentSession )

    ########################################################################################################################################################################################################

            if which_subject == 'sub-08':
                # subject information
                initials = 'sub-08'
                firstName = 'sub-08'
                standardFSID = 'LH_250514'
                birthdate = datetime.date( 1900, 01, 01 )
                labelFolderOfPreference = '2014_custom'
                presentSubject = Subject( initials, firstName, birthdate, standardFSID, labelFolderOfPreference )
                presentProject = Project( 'yes_no_fmri', subject = presentSubject, base_dir = os.path.join(this_project_folder, 'data') )
                sessionID = 'yes_no_fmri' + presentSubject.initials

                if s == 1:
                    sessionDate = datetime.date(2014, 5, 25)
                    sj_session1 = 'ses-01'
                if s == 2:
                    sessionDate = datetime.date(2014, 6, 14)
                    sj_session2 = 'ses-02'

                presentSession = defs_fmri_individuals(sessionID, sessionDate, presentProject, presentSubject)

                try:
                    os.mkdir(os.path.join(this_project_folder, 'data', initials))
                except OSError:
                    presentSession.logger.debug('output folders already exist')

                # ----------------------
                # Decision tasks:      -
                # ----------------------

                if s == 1:
                    runDecisionArray = [
                        # B0:
                        {'ID' : 1, 'scanType': 'inplane_anat', 'condition': 'B0_anat_mag', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'fmap', 'sub-08_ses-01_magnitude.nii.gz' ),},
                        {'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'B0_anat_phs', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'fmap', 'sub-08_ses-01_phasediff.nii.gz' ),},
                        # Whole brain TSE:
                        {'ID' : 3, 'scanType': 'inplane_anat', 'condition': 'TSE_anat_whole', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-08_ses-01_TSEwb.nii.gz' ),},
                        # Neuromalanin:
                        {'ID' : 4, 'scanType': 'inplane_anat', 'condition': 'TSE_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-08_ses-01_TSEori.nii.gz' ),},
                        # T1:
                        {'ID' : 5, 'scanType': 'inplane_anat', 'condition': 'T1_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-08_ses-01_T1w.nii.gz' ),},
                        # T2:
                        {'ID' : 6, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-08_ses-01_T2w.nii.gz' ),},
                        # Localizer:
                        {'ID' : 7, 'scanType': 'epi_bold', 'condition': 'loc', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-08_ses-01_task-localizer_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-08_ses-01_task-localizer_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-08_ses-01_task-localizer_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-08_ses-01_task-localizer_run-1_eyedata.edf' ),
                            },
                        # Decision tasks:
                        {'ID' : 8, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-08_ses-01_task-yesno_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-08_ses-01_task-yesno_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-08_ses-01_task-yesno_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-08_ses-01_task-yesno_run-1_eyedata.edf' ),
                            },
                        {'ID' : 9, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-08_ses-01_task-yesno_run-2_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-08_ses-01_task-yesno_run-2_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-08_ses-01_task-yesno_run-2_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-08_ses-01_task-yesno_run-2_eyedata.edf' ),
                            },
                        {'ID' : 10, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-08_ses-01_task-yesno_run-3_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-08_ses-01_task-yesno_run-3_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-08_ses-01_task-yesno_run-3_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-08_ses-01_task-yesno_run-3_eyedata.edf' ),
                            },
                        {'ID' : 11, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-08_ses-01_task-yesno_run-4_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-08_ses-01_task-yesno_run-4_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-08_ses-01_task-yesno_run-4_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-08_ses-01_task-yesno_run-4_eyedata.edf' ),
                            },
                        ]
                if s == 2:
                    runDecisionArray = [
                        # B0:
                        {'ID' : 1, 'scanType': 'inplane_anat', 'condition': 'B0_anat_mag', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'fmap', 'sub-08_ses-02_magnitude.nii.gz' ),},
                        {'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'B0_anat_phs', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'fmap', 'sub-08_ses-02_phasediff.nii.gz' ),},
                        # Whole brain TSE:
                        {'ID' : 3, 'scanType': 'inplane_anat', 'condition': 'TSE_anat_whole', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-08_ses-02_TSEwb.nii.gz' ),},
                        # Neuromalanin:
                        {'ID' : 4, 'scanType': 'inplane_anat', 'condition': 'TSE_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-08_ses-02_TSEori.nii.gz' ),},
                        # T1:
                        {'ID' : 5, 'scanType': 'inplane_anat', 'condition': 'T1_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-08_ses-02_T1w.nii.gz' ),},
                        # T2:
                        {'ID' : 6, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-08_ses-02_T2w.nii.gz' ),},
                        # Localizer:
                        {'ID' : 7, 'scanType': 'epi_bold', 'condition': 'loc', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-08_ses-02_task-localizer_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-08_ses-02_task-localizer_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-08_ses-02_task-localizer_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-08_ses-02_task-localizer_run-1_eyedata.edf' ),
                            },
                        # Decision tasks:
                        {'ID' : 8, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-08_ses-02_task-yesno_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-08_ses-02_task-yesno_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-08_ses-02_task-yesno_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-08_ses-02_task-yesno_run-1_eyedata.edf' ),
                            },
                        {'ID' : 9, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-08_ses-02_task-yesno_run-2_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-08_ses-02_task-yesno_run-2_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-08_ses-02_task-yesno_run-2_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-08_ses-02_task-yesno_run-2_eyedata.edf' ),
                            },
                        {'ID' : 10, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-08_ses-02_task-yesno_run-3_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-08_ses-02_task-yesno_run-3_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-08_ses-02_task-yesno_run-3_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-08_ses-02_task-yesno_run-3_eyedata.edf' ),
                            },
                        {'ID' : 11, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-08_ses-02_task-yesno_run-4_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-08_ses-02_task-yesno_run-4_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-08_ses-02_task-yesno_run-4_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-08_ses-02_task-yesno_run-4_eyedata.edf' ),
                            },
                        {'ID' : 12, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-08_ses-02_task-yesno_run-5_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-08_ses-02_task-yesno_run-5_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-08_ses-02_task-yesno_run-5_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-08_ses-02_task-yesno_run-5_eyedata.edf' ),
                            },
                        {'ID' : 13, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-08_ses-02_task-yesno_run-6_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-08_ses-02_task-yesno_run-6_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-08_ses-02_task-yesno_run-6_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-08_ses-02_task-yesno_run-6_eyedata.edf' ),
                            },
                        ]

                # ----------------------
                # Initialise session   -
                # ----------------------

                runWholeSession( runDecisionArray, presentSession )

    ########################################################################################################################################################################################################

            if which_subject == 'sub-09':
                # subject information
                initials = 'sub-09'
                firstName = 'sub-09'
                standardFSID = 'LP_040514'
                birthdate = datetime.date( 1900, 01, 01 )
                labelFolderOfPreference = '2014_custom'
                presentSubject = Subject( initials, firstName, birthdate, standardFSID, labelFolderOfPreference )
                presentProject = Project( 'yes_no_fmri', subject = presentSubject, base_dir = os.path.join(this_project_folder, 'data') )
                sessionID = 'yes_no_fmri' + presentSubject.initials

                if s == 1:
                    sessionDate = datetime.date(2014, 5, 25)
                    sj_session1 = 'ses-01'
                if s == 2:
                    sessionDate = datetime.date(2014, 6, 14)
                    sj_session2 = 'ses-02'

                presentSession = defs_fmri_individuals(sessionID, sessionDate, presentProject, presentSubject)

                try:
                    os.mkdir(os.path.join(this_project_folder, 'data', initials))
                except OSError:
                    presentSession.logger.debug('output folders already exist')

                # ----------------------
                # Decision tasks:      -
                # ----------------------

                if s == 1:
                    runDecisionArray = [
                        # B0:
                        {'ID' : 1, 'scanType': 'inplane_anat', 'condition': 'B0_anat_mag', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'fmap', 'sub-09_ses-01_magnitude.nii.gz' ),},
                        {'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'B0_anat_phs', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'fmap', 'sub-09_ses-01_phasediff.nii.gz' ),},
                        # Whole brain TSE:
                        {'ID' : 3, 'scanType': 'inplane_anat', 'condition': 'TSE_anat_whole', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-09_ses-01_TSEwb.nii.gz' ),},
                        # Neuromalanin:
                        {'ID' : 4, 'scanType': 'inplane_anat', 'condition': 'TSE_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-09_ses-01_TSEori.nii.gz' ),},
                        # T1:
                        {'ID' : 5, 'scanType': 'inplane_anat', 'condition': 'T1_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-09_ses-01_T1w.nii.gz' ),},
                        # T2:
                        {'ID' : 6, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-09_ses-01_T2w.nii.gz' ),},
                        # Localizer:
                        {'ID' : 7, 'scanType': 'epi_bold', 'condition': 'loc', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-09_ses-01_task-localizer_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-09_ses-01_task-localizer_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-09_ses-01_task-localizer_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-09_ses-01_task-localizer_run-1_eyedata.edf' ),
                            },
                        # Decision tasks:
                        {'ID' : 8, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-09_ses-01_task-yesno_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-09_ses-01_task-yesno_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-09_ses-01_task-yesno_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-09_ses-01_task-yesno_run-1_eyedata.edf' ),
                            },
                        {'ID' : 9, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-09_ses-01_task-yesno_run-2_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-09_ses-01_task-yesno_run-2_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-09_ses-01_task-yesno_run-2_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-09_ses-01_task-yesno_run-2_eyedata.edf' ),
                            },
                        {'ID' : 10, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-09_ses-01_task-yesno_run-3_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-09_ses-01_task-yesno_run-3_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-09_ses-01_task-yesno_run-3_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-09_ses-01_task-yesno_run-3_eyedata.edf' ),
                            },
                        {'ID' : 11, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-09_ses-01_task-yesno_run-4_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-09_ses-01_task-yesno_run-4_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-09_ses-01_task-yesno_run-4_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-09_ses-01_task-yesno_run-4_eyedata.edf' ),
                            },
                        {'ID' : 12, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-09_ses-01_task-yesno_run-5_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-09_ses-01_task-yesno_run-5_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-09_ses-01_task-yesno_run-5_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-09_ses-01_task-yesno_run-5_eyedata.edf' ),
                            },
                        {'ID' : 13, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-09_ses-01_task-yesno_run-6_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-09_ses-01_task-yesno_run-6_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-09_ses-01_task-yesno_run-6_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-09_ses-01_task-yesno_run-6_eyedata.edf' ),
                            },
                        ]
                if s == 2:
                    runDecisionArray = [
                        # B0:
                        {'ID' : 1, 'scanType': 'inplane_anat', 'condition': 'B0_anat_mag', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'fmap', 'sub-09_ses-02_magnitude.nii.gz' ),},
                        {'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'B0_anat_phs', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'fmap', 'sub-09_ses-02_phasediff.nii.gz' ),},
                        # Whole brain TSE:
                        {'ID' : 3, 'scanType': 'inplane_anat', 'condition': 'TSE_anat_whole', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-09_ses-02_TSEwb.nii.gz' ),},
                        # Neuromalanin:
                        {'ID' : 4, 'scanType': 'inplane_anat', 'condition': 'TSE_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-09_ses-02_TSEori.nii.gz' ),},
                        # T1:
                        {'ID' : 5, 'scanType': 'inplane_anat', 'condition': 'T1_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-09_ses-02_T1w.nii.gz' ),},
                        # T2:
                        {'ID' : 6, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-09_ses-02_T2w.nii.gz' ),},
                        # Localizer:
                        {'ID' : 7, 'scanType': 'epi_bold', 'condition': 'loc', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-09_ses-02_task-localizer_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-09_ses-02_task-localizer_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-09_ses-02_task-localizer_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-09_ses-02_task-localizer_run-1_eyedata.edf' ),
                            },
                        # Decision tasks:
                        {'ID' : 8, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-09_ses-02_task-yesno_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-09_ses-02_task-yesno_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-09_ses-02_task-yesno_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-09_ses-02_task-yesno_run-1_eyedata.edf' ),
                            },
                        {'ID' : 9, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-09_ses-02_task-yesno_run-2_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-09_ses-02_task-yesno_run-2_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-09_ses-02_task-yesno_run-2_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-09_ses-02_task-yesno_run-2_eyedata.edf' ),
                            },
                        {'ID' : 10, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-09_ses-02_task-yesno_run-3_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-09_ses-02_task-yesno_run-3_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-09_ses-02_task-yesno_run-3_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-09_ses-02_task-yesno_run-3_eyedata.edf' ),
                            },
                        {'ID' : 11, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-09_ses-02_task-yesno_run-4_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-09_ses-02_task-yesno_run-4_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-09_ses-02_task-yesno_run-4_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-09_ses-02_task-yesno_run-4_eyedata.edf' ),
                            },
                        {'ID' : 12, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-09_ses-02_task-yesno_run-5_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-09_ses-02_task-yesno_run-5_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-09_ses-02_task-yesno_run-5_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-09_ses-02_task-yesno_run-5_eyedata.edf' ),
                            },
                        {'ID' : 13, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-09_ses-02_task-yesno_run-6_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-09_ses-02_task-yesno_run-6_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-09_ses-02_task-yesno_run-6_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-09_ses-02_task-yesno_run-6_eyedata.edf' ),
                            },
                        ]

                # ----------------------
                # Initialise session   -
                # ----------------------

                runWholeSession( runDecisionArray, presentSession )

    ########################################################################################################################################################################################################

            if which_subject == 'sub-10':
                # subject information
                initials = 'sub-10'
                firstName = 'sub-10'
                standardFSID = 'MG_190414'
                birthdate = datetime.date( 1900, 01, 01 )
                labelFolderOfPreference = '2014_custom'
                presentSubject = Subject( initials, firstName, birthdate, standardFSID, labelFolderOfPreference )
                presentProject = Project( 'yes_no_fmri', subject = presentSubject, base_dir = os.path.join(this_project_folder, 'data') )
                sessionID = 'yes_no_fmri' + presentSubject.initials

                if s == 1:
                    sessionDate = datetime.date(2014, 6, 9)
                    sj_session1 = 'ses-01'
                if s == 2:
                    sessionDate = datetime.date(2014, 6, 14)
                    sj_session2 = 'ses-02'

                presentSession = defs_fmri_individuals(sessionID, sessionDate, presentProject, presentSubject)

                try:
                    os.mkdir(os.path.join(this_project_folder, 'data', initials))
                except OSError:
                    presentSession.logger.debug('output folders already exist')

                # ----------------------
                # Decision tasks:      -
                # ----------------------

                if s == 1:
                    runDecisionArray = [
                        # B0:
                        {'ID' : 1, 'scanType': 'inplane_anat', 'condition': 'B0_anat_mag', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'fmap', 'sub-10_ses-01_magnitude.nii.gz' ),},
                        {'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'B0_anat_phs', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'fmap', 'sub-10_ses-01_phasediff.nii.gz' ),},
                        # Whole brain TSE:
                        {'ID' : 3, 'scanType': 'inplane_anat', 'condition': 'TSE_anat_whole', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-10_ses-01_TSEwb.nii.gz' ),},
                        # Neuromalanin:
                        {'ID' : 4, 'scanType': 'inplane_anat', 'condition': 'TSE_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-10_ses-01_TSEori.nii.gz' ),},
                        # T1:
                        {'ID' : 5, 'scanType': 'inplane_anat', 'condition': 'T1_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-10_ses-01_T1w.nii.gz' ),},
                        # T2:
                        {'ID' : 6, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-10_ses-01_T2w.nii.gz' ),},
                        # Localizer:
                        {'ID' : 7, 'scanType': 'epi_bold', 'condition': 'loc', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-10_ses-01_task-localizer_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-10_ses-01_task-localizer_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-10_ses-01_task-localizer_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-10_ses-01_task-localizer_run-1_eyedata.edf' ),
                            },
                        # Decision tasks:
                        {'ID' : 8, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-10_ses-01_task-yesno_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-10_ses-01_task-yesno_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-10_ses-01_task-yesno_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-10_ses-01_task-yesno_run-1_eyedata.edf' ),
                            },
                        {'ID' : 9, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-10_ses-01_task-yesno_run-2_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-10_ses-01_task-yesno_run-2_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-10_ses-01_task-yesno_run-2_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-10_ses-01_task-yesno_run-2_eyedata.edf' ),
                            },
                        {'ID' : 10, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-10_ses-01_task-yesno_run-3_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-10_ses-01_task-yesno_run-3_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-10_ses-01_task-yesno_run-3_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-10_ses-01_task-yesno_run-3_eyedata.edf' ),
                            },
                        {'ID' : 11, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-10_ses-01_task-yesno_run-4_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-10_ses-01_task-yesno_run-4_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-10_ses-01_task-yesno_run-4_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-10_ses-01_task-yesno_run-4_eyedata.edf' ),
                            },
                        {'ID' : 12, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-10_ses-01_task-yesno_run-5_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-10_ses-01_task-yesno_run-5_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-10_ses-01_task-yesno_run-5_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-10_ses-01_task-yesno_run-5_eyedata.edf' ),
                            },
                        {'ID' : 13, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-10_ses-01_task-yesno_run-6_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-10_ses-01_task-yesno_run-6_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-10_ses-01_task-yesno_run-6_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-10_ses-01_task-yesno_run-6_eyedata.edf' ),
                            },
                        ]
                if s == 2:
                    runDecisionArray = [
                        # B0:
                        {'ID' : 1, 'scanType': 'inplane_anat', 'condition': 'B0_anat_mag', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'fmap', 'sub-10_ses-02_magnitude.nii.gz' ),},
                        {'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'B0_anat_phs', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'fmap', 'sub-10_ses-02_phasediff.nii.gz' ),},
                        # Whole brain TSE:
                        {'ID' : 3, 'scanType': 'inplane_anat', 'condition': 'TSE_anat_whole', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-10_ses-02_TSEwb.nii.gz' ),},
                        # Neuromalanin:
                        {'ID' : 4, 'scanType': 'inplane_anat', 'condition': 'TSE_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-10_ses-02_TSEori.nii.gz' ),},
                        # T1:
                        {'ID' : 5, 'scanType': 'inplane_anat', 'condition': 'T1_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-10_ses-02_T1w.nii.gz' ),},
                        # T2:
                        {'ID' : 6, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-10_ses-02_T2w.nii.gz' ),},
                        # Localizer:
                        {'ID' : 7, 'scanType': 'epi_bold', 'condition': 'loc', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-10_ses-02_task-localizer_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-10_ses-02_task-localizer_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-10_ses-02_task-localizer_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-10_ses-02_task-localizer_run-1_eyedata.edf' ),
                            },
                        # Decision tasks:
                        {'ID' : 8, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-10_ses-02_task-yesno_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-10_ses-02_task-yesno_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-10_ses-02_task-yesno_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-10_ses-02_task-yesno_run-1_eyedata.edf' ),
                            },
                        {'ID' : 9, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-10_ses-02_task-yesno_run-2_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-10_ses-02_task-yesno_run-2_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-10_ses-02_task-yesno_run-2_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-10_ses-02_task-yesno_run-2_eyedata.edf' ),
                            },
                        {'ID' : 10, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-10_ses-02_task-yesno_run-3_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-10_ses-02_task-yesno_run-3_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-10_ses-02_task-yesno_run-3_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-10_ses-02_task-yesno_run-3_eyedata.edf' ),
                            },
                        {'ID' : 11, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-10_ses-02_task-yesno_run-4_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-10_ses-02_task-yesno_run-4_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-10_ses-02_task-yesno_run-4_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-10_ses-02_task-yesno_run-4_eyedata.edf' ),
                            },
                        {'ID' : 12, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-10_ses-02_task-yesno_run-5_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-10_ses-02_task-yesno_run-5_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-10_ses-02_task-yesno_run-5_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-10_ses-02_task-yesno_run-5_eyedata.edf' ),
                            },
                        {'ID' : 13, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-10_ses-02_task-yesno_run-6_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-10_ses-02_task-yesno_run-6_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-10_ses-02_task-yesno_run-6_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-10_ses-02_task-yesno_run-6_eyedata.edf' ),
                            },
                        ]

                # ----------------------
                # Initialise session   -
                # ----------------------

                runWholeSession( runDecisionArray, presentSession )
     ########################################################################################################################################################################################################

            if which_subject == 'sub-11':
                # subject information
                initials = 'sub-11'
                firstName = 'sub-11'
                standardFSID = 'NS_030215'
                birthdate = datetime.date( 1900, 01, 01 )
                labelFolderOfPreference = '2014_custom'
                presentSubject = Subject( initials, firstName, birthdate, standardFSID, labelFolderOfPreference )
                presentProject = Project( 'yes_no_fmri', subject = presentSubject, base_dir = os.path.join(this_project_folder, 'data') )
                sessionID = 'yes_no_fmri' + presentSubject.initials

                if s == 1:
                    sessionDate = datetime.date(2015, 2, 3)
                    sj_session1 = 'ses-01'
                if s == 2:
                    sessionDate = datetime.date(2015, 2, 6)
                    sj_session2 = 'ses-02'

                presentSession = defs_fmri_individuals(sessionID, sessionDate, presentProject, presentSubject)

                try:
                    os.mkdir(os.path.join(this_project_folder, 'data', initials))
                except OSError:
                    presentSession.logger.debug('output folders already exist')

                # ----------------------
                # Decision tasks:      -
                # ----------------------

                if s == 1:
                    runDecisionArray = [
                        # B0:
                        {'ID' : 1, 'scanType': 'inplane_anat', 'condition': 'B0_anat_mag', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'fmap', 'sub-11_ses-01_magnitude.nii.gz' ),},
                        {'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'B0_anat_phs', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'fmap', 'sub-11_ses-01_phasediff.nii.gz' ),},
                        # Whole brain TSE:
                        {'ID' : 3, 'scanType': 'inplane_anat', 'condition': 'TSE_anat_whole', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-11_ses-01_TSEwb.nii.gz' ),},
                        # Neuromalanin:
                        {'ID' : 4, 'scanType': 'inplane_anat', 'condition': 'TSE_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-11_ses-01_TSEori.nii.gz' ),},
                        # T1:
                        {'ID' : 5, 'scanType': 'inplane_anat', 'condition': 'T1_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-11_ses-01_T1w.nii.gz' ),},
                        # T2:
                        {'ID' : 6, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-11_ses-01_T2w.nii.gz' ),},
                        # Localizer:
                        {'ID' : 7, 'scanType': 'epi_bold', 'condition': 'loc', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-11_ses-01_task-localizer_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-11_ses-01_task-localizer_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-11_ses-01_task-localizer_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-11_ses-01_task-localizer_run-1_eyedata.edf' ),
                            },
                        # Decision tasks:
                        {'ID' : 8, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-11_ses-01_task-yesno_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-11_ses-01_task-yesno_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-11_ses-01_task-yesno_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-11_ses-01_task-yesno_run-1_eyedata.edf' ),
                            },
                        {'ID' : 9, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-11_ses-01_task-yesno_run-2_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-11_ses-01_task-yesno_run-2_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-11_ses-01_task-yesno_run-2_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-11_ses-01_task-yesno_run-2_eyedata.edf' ),
                            },
                        {'ID' : 10, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-11_ses-01_task-yesno_run-3_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-11_ses-01_task-yesno_run-3_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-11_ses-01_task-yesno_run-3_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-11_ses-01_task-yesno_run-3_eyedata.edf' ),
                            },
                        {'ID' : 11, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-11_ses-01_task-yesno_run-4_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-11_ses-01_task-yesno_run-4_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-11_ses-01_task-yesno_run-4_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-11_ses-01_task-yesno_run-4_eyedata.edf' ),
                            },
                        {'ID' : 12, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-11_ses-01_task-yesno_run-5_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-11_ses-01_task-yesno_run-5_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-11_ses-01_task-yesno_run-5_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-11_ses-01_task-yesno_run-5_eyedata.edf' ),
                            },
                        {'ID' : 13, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-11_ses-01_task-yesno_run-6_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-11_ses-01_task-yesno_run-6_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-11_ses-01_task-yesno_run-6_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-11_ses-01_task-yesno_run-6_eyedata.edf' ),
                            },
                        ]
                if s == 2:
                    runDecisionArray = [
                        # B0:
                        {'ID' : 1, 'scanType': 'inplane_anat', 'condition': 'B0_anat_mag', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'fmap', 'sub-11_ses-02_magnitude.nii.gz' ),},
                        {'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'B0_anat_phs', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'fmap', 'sub-11_ses-02_phasediff.nii.gz' ),},
                        # Whole brain TSE:
                        {'ID' : 3, 'scanType': 'inplane_anat', 'condition': 'TSE_anat_whole', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-11_ses-02_TSEwb.nii.gz' ),},
                        # Neuromalanin:
                        {'ID' : 4, 'scanType': 'inplane_anat', 'condition': 'TSE_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-11_ses-02_TSEori.nii.gz' ),},
                        # T1:
                        {'ID' : 5, 'scanType': 'inplane_anat', 'condition': 'T1_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-11_ses-02_T1w.nii.gz' ),},
                        # T2:
                        {'ID' : 6, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-11_ses-02_T2w.nii.gz' ),},
                        # Localizer:
                        {'ID' : 7, 'scanType': 'epi_bold', 'condition': 'loc', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-11_ses-02_task-localizer_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-11_ses-02_task-localizer_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-11_ses-02_task-localizer_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-11_ses-02_task-localizer_run-1_eyedata.edf' ),
                            },
                        # Decision tasks:
                        {'ID' : 8, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-11_ses-02_task-yesno_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-11_ses-02_task-yesno_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-11_ses-02_task-yesno_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-11_ses-02_task-yesno_run-1_eyedata.edf' ),
                            },
                        {'ID' : 9, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-11_ses-02_task-yesno_run-2_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-11_ses-02_task-yesno_run-2_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-11_ses-02_task-yesno_run-2_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-11_ses-02_task-yesno_run-2_eyedata.edf' ),
                            },
                        {'ID' : 10, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-11_ses-02_task-yesno_run-3_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-11_ses-02_task-yesno_run-3_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-11_ses-02_task-yesno_run-3_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-11_ses-02_task-yesno_run-3_eyedata.edf' ),
                            },
                        {'ID' : 11, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-11_ses-02_task-yesno_run-4_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-11_ses-02_task-yesno_run-4_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-11_ses-02_task-yesno_run-4_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-11_ses-02_task-yesno_run-4_eyedata.edf' ),
                            },
                        {'ID' : 12, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-11_ses-02_task-yesno_run-5_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-11_ses-02_task-yesno_run-5_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-11_ses-02_task-yesno_run-5_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-11_ses-02_task-yesno_run-5_eyedata.edf' ),
                            },
                        {'ID' : 13, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-11_ses-02_task-yesno_run-6_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-11_ses-02_task-yesno_run-6_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-11_ses-02_task-yesno_run-6_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-11_ses-02_task-yesno_run-6_eyedata.edf' ),
                            },
                        ]

                # ----------------------
                # Initialise session   -
                # ----------------------

                runWholeSession( runDecisionArray, presentSession )


    ########################################################################################################################################################################################################

            if which_subject == 'sub-12':
                # subject information
                initials = 'sub-12'
                firstName = 'sub-12'
                standardFSID = 'OC_250711'
                birthdate = datetime.date( 1900, 01, 01 )
                labelFolderOfPreference = '2014_custom'
                presentSubject = Subject( initials, firstName, birthdate, standardFSID, labelFolderOfPreference )
                presentProject = Project( 'yes_no_fmri', subject = presentSubject, base_dir = os.path.join(this_project_folder, 'data') )
                sessionID = 'yes_no_fmri' + presentSubject.initials

                if s == 1:
                    sessionDate = datetime.date(2015, 1, 28)
                    sj_session1 = 'ses-01'
                if s == 2:
                    sessionDate = datetime.date(2015, 1, 29)
                    sj_session2 = 'ses-02'

                presentSession = defs_fmri_individuals(sessionID, sessionDate, presentProject, presentSubject)

                try:
                    os.mkdir(os.path.join(this_project_folder, 'data', initials))
                except OSError:
                    presentSession.logger.debug('output folders already exist')

                # ----------------------
                # Decision tasks:      -
                # ----------------------

                if s == 1:
                    runDecisionArray = [
                        # B0:
                        {'ID' : 1, 'scanType': 'inplane_anat', 'condition': 'B0_anat_mag', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'fmap', 'sub-12_ses-01_magnitude.nii.gz' ),},
                        {'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'B0_anat_phs', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'fmap', 'sub-12_ses-01_phasediff.nii.gz' ),},
                        # Whole brain TSE:
                        {'ID' : 3, 'scanType': 'inplane_anat', 'condition': 'TSE_anat_whole', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-12_ses-01_TSEwb.nii.gz' ),},
                        # Neuromalanin:
                        {'ID' : 4, 'scanType': 'inplane_anat', 'condition': 'TSE_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-12_ses-01_TSEori.nii.gz' ),},
                        # # T1:
                        {'ID' : 5, 'scanType': 'inplane_anat', 'condition': 'T1_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-12_ses-01_T1w.nii.gz' ),},
                        # T2:
                        {'ID' : 6, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-12_ses-01_T2w.nii.gz' ),},
                        # Localizer:
                        {'ID' : 7, 'scanType': 'epi_bold', 'condition': 'loc', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-12_ses-01_task-localizer_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-12_ses-01_task-localizer_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-12_ses-01_task-localizer_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-12_ses-01_task-localizer_run-1_eyedata.edf' ),
                            },
                        # Decision tasks:
                        {'ID' : 8, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-12_ses-01_task-yesno_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-12_ses-01_task-yesno_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-12_ses-01_task-yesno_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-12_ses-01_task-yesno_run-1_eyedata.edf' ),
                            },
                        {'ID' : 9, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-12_ses-01_task-yesno_run-2_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-12_ses-01_task-yesno_run-2_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-12_ses-01_task-yesno_run-2_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-12_ses-01_task-yesno_run-2_eyedata.edf' ),
                            },
                        {'ID' : 10, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-12_ses-01_task-yesno_run-3_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-12_ses-01_task-yesno_run-3_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-12_ses-01_task-yesno_run-3_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-12_ses-01_task-yesno_run-3_eyedata.edf' ),
                            },
                        {'ID' : 11, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-12_ses-01_task-yesno_run-4_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-12_ses-01_task-yesno_run-4_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-12_ses-01_task-yesno_run-4_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-12_ses-01_task-yesno_run-4_eyedata.edf' ),
                            },
                        {'ID' : 12, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-12_ses-01_task-yesno_run-5_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-12_ses-01_task-yesno_run-5_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-12_ses-01_task-yesno_run-5_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-12_ses-01_task-yesno_run-5_eyedata.edf' ),
                            },
                        {'ID' : 13, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-12_ses-01_task-yesno_run-6_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-12_ses-01_task-yesno_run-6_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-12_ses-01_task-yesno_run-6_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-12_ses-01_task-yesno_run-6_eyedata.edf' ),
                            },
                        ]
                if s == 2:
                    runDecisionArray = [
                        # B0:
                        {'ID' : 1, 'scanType': 'inplane_anat', 'condition': 'B0_anat_mag', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'fmap', 'sub-12_ses-02_magnitude.nii.gz' ),},
                        {'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'B0_anat_phs', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'fmap', 'sub-12_ses-02_phasediff.nii.gz' ),},
                        # Whole brain TSE:
                        {'ID' : 3, 'scanType': 'inplane_anat', 'condition': 'TSE_anat_whole', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-12_ses-02_TSEwb.nii.gz' ),},
                        # Neuromalanin:
                        {'ID' : 4, 'scanType': 'inplane_anat', 'condition': 'TSE_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-12_ses-02_TSEori.nii.gz' ),},
                        # # T1:
                        # {'ID' : 5, 'scanType': 'inplane_anat', 'condition': 'T1_anat', 'session' : 2,
                        #     'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-12_ses-02_T1w.nii.gz' ),},
                        # T2:
                        {'ID' : 6, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-12_ses-02_T2w.nii.gz' ),},
                        # Localizer:
                        {'ID' : 7, 'scanType': 'epi_bold', 'condition': 'loc', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-12_ses-02_task-localizer_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-12_ses-02_task-localizer_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-12_ses-02_task-localizer_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-12_ses-02_task-localizer_run-1_eyedata.edf' ),
                            },
                        # Decision tasks:
                        {'ID' : 8, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-12_ses-02_task-yesno_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-12_ses-02_task-yesno_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-12_ses-02_task-yesno_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-12_ses-02_task-yesno_run-1_eyedata.edf' ),
                            },
                        {'ID' : 9, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-12_ses-02_task-yesno_run-2_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-12_ses-02_task-yesno_run-2_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-12_ses-02_task-yesno_run-2_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-12_ses-02_task-yesno_run-2_eyedata.edf' ),
                            },
                        {'ID' : 10, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-12_ses-02_task-yesno_run-3_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-12_ses-02_task-yesno_run-3_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-12_ses-02_task-yesno_run-3_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-12_ses-02_task-yesno_run-3_eyedata.edf' ),
                            },
                        {'ID' : 11, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-12_ses-02_task-yesno_run-4_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-12_ses-02_task-yesno_run-4_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-12_ses-02_task-yesno_run-4_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-12_ses-02_task-yesno_run-4_eyedata.edf' ),
                            },
                        {'ID' : 12, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-12_ses-02_task-yesno_run-5_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-12_ses-02_task-yesno_run-5_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-12_ses-02_task-yesno_run-5_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-12_ses-02_task-yesno_run-5_eyedata.edf' ),
                            },
                        {'ID' : 13, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-12_ses-02_task-yesno_run-6_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-12_ses-02_task-yesno_run-6_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-12_ses-02_task-yesno_run-6_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-12_ses-02_task-yesno_run-6_eyedata.edf' ),
                            },
                        ]
                # ----------------------
                # Initialise session   -
                # ----------------------

                runWholeSession( runDecisionArray, presentSession )

    ########################################################################################################################################################################################################

            if which_subject == 'sub-13':
                # subject information
                initials = 'sub-13'
                firstName = 'sub-13'
                standardFSID = 'TK_091009tk'
                birthdate = datetime.date( 1900, 01, 01 )
                labelFolderOfPreference = '2014_custom'
                presentSubject = Subject( initials, firstName, birthdate, standardFSID, labelFolderOfPreference )
                presentProject = Project( 'yes_no_fmri', subject = presentSubject, base_dir = os.path.join(this_project_folder, 'data') )
                sessionID = 'yes_no_fmri' + presentSubject.initials

                if s == 1:
                    sessionDate = datetime.date(2015, 1, 20)
                    sj_session1 = 'ses-01'
                if s == 2:
                    sessionDate = datetime.date(2015, 1, 22)
                    sj_session2 = 'ses-02'

                presentSession = defs_fmri_individuals(sessionID, sessionDate, presentProject, presentSubject)

                try:
                    os.mkdir(os.path.join(this_project_folder, 'data', initials))
                except OSError:
                    presentSession.logger.debug('output folders already exist')

                # ----------------------
                # Decision tasks:      -
                # ----------------------

                if s == 1:
                    runDecisionArray = [
                        # B0:
                        {'ID' : 1, 'scanType': 'inplane_anat', 'condition': 'B0_anat_mag', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'fmap', 'sub-13_ses-01_magnitude.nii.gz' ),},
                        {'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'B0_anat_phs', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'fmap', 'sub-13_ses-01_phasediff.nii.gz' ),},
                        # Whole brain TSE:
                        {'ID' : 3, 'scanType': 'inplane_anat', 'condition': 'TSE_anat_whole', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-13_ses-01_TSEwb.nii.gz' ),},
                        # Neuromalanin:
                        {'ID' : 4, 'scanType': 'inplane_anat', 'condition': 'TSE_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-13_ses-01_TSEori.nii.gz' ),},
                        # T1:
                        {'ID' : 5, 'scanType': 'inplane_anat', 'condition': 'T1_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-13_ses-01_T1w.nii.gz' ),},
                        # T2:
                        {'ID' : 6, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-13_ses-01_T2w.nii.gz' ),},
                        # Localizer:
                        {'ID' : 7, 'scanType': 'epi_bold', 'condition': 'loc', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-13_ses-01_task-localizer_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-13_ses-01_task-localizer_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-13_ses-01_task-localizer_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-13_ses-01_task-localizer_run-1_eyedata.edf' ),
                            },
                        # Decision tasks:
                        {'ID' : 8, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-13_ses-01_task-yesno_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-13_ses-01_task-yesno_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-13_ses-01_task-yesno_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-13_ses-01_task-yesno_run-1_eyedata.edf' ),
                            },
                        {'ID' : 9, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-13_ses-01_task-yesno_run-2_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-13_ses-01_task-yesno_run-2_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-13_ses-01_task-yesno_run-2_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-13_ses-01_task-yesno_run-2_eyedata.edf' ),
                            },
                        {'ID' : 10, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-13_ses-01_task-yesno_run-3_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-13_ses-01_task-yesno_run-3_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-13_ses-01_task-yesno_run-3_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-13_ses-01_task-yesno_run-3_eyedata.edf' ),
                            },
                        {'ID' : 11, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-13_ses-01_task-yesno_run-4_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-13_ses-01_task-yesno_run-4_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-13_ses-01_task-yesno_run-4_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-13_ses-01_task-yesno_run-4_eyedata.edf' ),
                            },
                        {'ID' : 12, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-13_ses-01_task-yesno_run-5_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-13_ses-01_task-yesno_run-5_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-13_ses-01_task-yesno_run-5_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-13_ses-01_task-yesno_run-5_eyedata.edf' ),
                            },
                        ]
                if s == 2:
                    runDecisionArray = [
                        # B0:
                        {'ID' : 1, 'scanType': 'inplane_anat', 'condition': 'B0_anat_mag', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'fmap', 'sub-13_ses-02_magnitude.nii.gz' ),},
                        {'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'B0_anat_phs', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'fmap', 'sub-13_ses-02_phasediff.nii.gz' ),},
                        # Whole brain TSE:
                        {'ID' : 3, 'scanType': 'inplane_anat', 'condition': 'TSE_anat_whole', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-13_ses-02_TSEwb.nii.gz' ),},
                        # Neuromalanin:
                        {'ID' : 4, 'scanType': 'inplane_anat', 'condition': 'TSE_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-13_ses-02_TSEori.nii.gz' ),},
                        # T1:
                        {'ID' : 5, 'scanType': 'inplane_anat', 'condition': 'T1_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-13_ses-02_T1w.nii.gz' ),},
                        # T2:
                        {'ID' : 6, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-13_ses-02_T2w.nii.gz' ),},
                        # Localizer:
                        {'ID' : 7, 'scanType': 'epi_bold', 'condition': 'loc', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-13_ses-02_task-localizer_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-13_ses-02_task-localizer_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-13_ses-02_task-localizer_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-13_ses-02_task-localizer_run-1_eyedata.edf' ),
                            },
                        # Decision tasks:
                        {'ID' : 8, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-13_ses-02_task-yesno_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-13_ses-02_task-yesno_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-13_ses-02_task-yesno_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-13_ses-02_task-yesno_run-1_eyedata.edf' ),
                            },
                        {'ID' : 9, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-13_ses-02_task-yesno_run-2_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-13_ses-02_task-yesno_run-2_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-13_ses-02_task-yesno_run-2_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-13_ses-02_task-yesno_run-2_eyedata.edf' ),
                            },
                        {'ID' : 10, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-13_ses-02_task-yesno_run-3_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-13_ses-02_task-yesno_run-3_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-13_ses-02_task-yesno_run-3_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-13_ses-02_task-yesno_run-3_eyedata.edf' ),
                            },
                        {'ID' : 11, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-13_ses-02_task-yesno_run-4_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-13_ses-02_task-yesno_run-4_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-13_ses-02_task-yesno_run-4_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-13_ses-02_task-yesno_run-4_eyedata.edf' ),
                            },
                        {'ID' : 12, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-13_ses-02_task-yesno_run-5_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-13_ses-02_task-yesno_run-5_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-13_ses-02_task-yesno_run-5_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-13_ses-02_task-yesno_run-5_eyedata.edf' ),
                            },
                        {'ID' : 13, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-13_ses-02_task-yesno_run-6_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-13_ses-02_task-yesno_run-6_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-13_ses-02_task-yesno_run-6_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-13_ses-02_task-yesno_run-6_eyedata.edf' ),
                            },
                        ]

                # ----------------------
                # Initialise session   -
                # ----------------------

                runWholeSession( runDecisionArray, presentSession )

    ########################################################################################################################################################################################################

            if which_subject == 'sub-14':
                # subject information
                initials = 'sub-14'
                firstName = 'sub-14'
                standardFSID = 'TN_081014'
                birthdate = datetime.date( 1900, 01, 01 )
                labelFolderOfPreference = '2014_custom'
                presentSubject = Subject( initials, firstName, birthdate, standardFSID, labelFolderOfPreference )
                presentProject = Project( 'yes_no_fmri', subject = presentSubject, base_dir = os.path.join(this_project_folder, 'data') )
                sessionID = 'yes_no_fmri' + presentSubject.initials

                if s == 1:
                    sessionDate = datetime.date(2015, 2, 9)
                    sj_session1 = 'ses-01'
                if s == 2:
                    sessionDate = datetime.date(2015, 2, 10)
                    sj_session2 = 'ses-02'

                presentSession = defs_fmri_individuals(sessionID, sessionDate, presentProject, presentSubject)

                try:
                    os.mkdir(os.path.join(this_project_folder, 'data', initials))
                except OSError:
                    presentSession.logger.debug('output folders already exist')

                # ----------------------
                # Decision tasks:      -
                # ----------------------

                if s == 1:
                    runDecisionArray = [
                        # B0:
                        {'ID' : 1, 'scanType': 'inplane_anat', 'condition': 'B0_anat_mag', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'fmap', 'sub-14_ses-01_magnitude.nii.gz' ),},
                        {'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'B0_anat_phs', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'fmap', 'sub-14_ses-01_phasediff.nii.gz' ),},
                        # Whole brain TSE:
                        {'ID' : 3, 'scanType': 'inplane_anat', 'condition': 'TSE_anat_whole', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-14_ses-01_TSEwb.nii.gz' ),},
                        # Neuromalanin:
                        {'ID' : 4, 'scanType': 'inplane_anat', 'condition': 'TSE_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-14_ses-01_TSEori.nii.gz' ),},
                        # # T1:
                        # {'ID' : 5, 'scanType': 'inplane_anat', 'condition': 'T1_anat', 'session' : 1,
                        #     'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-14_ses-01_T1w.nii.gz' ),},
                        # T2:
                        {'ID' : 6, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'anat', 'sub-14_ses-01_T2w.nii.gz' ),},
                        # Localizer:
                        {'ID' : 7, 'scanType': 'epi_bold', 'condition': 'loc', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-14_ses-01_task-localizer_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-14_ses-01_task-localizer_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-14_ses-01_task-localizer_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-14_ses-01_task-localizer_run-1_eyedata.edf' ),
                            },
                        # Decision tasks:
                        {'ID' : 8, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-14_ses-01_task-yesno_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-14_ses-01_task-yesno_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-14_ses-01_task-yesno_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-14_ses-01_task-yesno_run-1_eyedata.edf' ),
                            },
                        {'ID' : 9, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-14_ses-01_task-yesno_run-2_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-14_ses-01_task-yesno_run-2_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-14_ses-01_task-yesno_run-2_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-14_ses-01_task-yesno_run-2_eyedata.edf' ),
                            },
                        {'ID' : 10, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-14_ses-01_task-yesno_run-3_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-14_ses-01_task-yesno_run-3_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-14_ses-01_task-yesno_run-3_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-14_ses-01_task-yesno_run-3_eyedata.edf' ),
                            },
                        {'ID' : 11, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-14_ses-01_task-yesno_run-4_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-14_ses-01_task-yesno_run-4_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-14_ses-01_task-yesno_run-4_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-14_ses-01_task-yesno_run-4_eyedata.edf' ),
                            },
                        {'ID' : 12, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-14_ses-01_task-yesno_run-5_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-14_ses-01_task-yesno_run-5_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-14_ses-01_task-yesno_run-5_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-14_ses-01_task-yesno_run-5_eyedata.edf' ),
                            },
                        {'ID' : 13, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 1,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-14_ses-01_task-yesno_run-6_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-14_ses-01_task-yesno_run-6_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-14_ses-01_task-yesno_run-6_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session1, 'func', 'sub-14_ses-01_task-yesno_run-6_eyedata.edf' ),
                            },
                        ]
                if s == 2:
                    runDecisionArray = [
                        # B0:
                        {'ID' : 1, 'scanType': 'inplane_anat', 'condition': 'B0_anat_mag', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'fmap', 'sub-14_ses-02_magnitude.nii.gz' ),},
                        {'ID' : 2, 'scanType': 'inplane_anat', 'condition': 'B0_anat_phs', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'fmap', 'sub-14_ses-02_phasediff.nii.gz' ),},
                        # Whole brain TSE:
                        {'ID' : 3, 'scanType': 'inplane_anat', 'condition': 'TSE_anat_whole', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-14_ses-02_TSEwb.nii.gz' ),},
                        # Neuromalanin:
                        {'ID' : 4, 'scanType': 'inplane_anat', 'condition': 'TSE_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-14_ses-02_TSEori.nii.gz' ),},
                        # T1:
                        {'ID' : 5, 'scanType': 'inplane_anat', 'condition': 'T1_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-14_ses-02_T1w.nii.gz' ),},
                        # T2:
                        {'ID' : 6, 'scanType': 'inplane_anat', 'condition': 'T2_anat', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'anat', 'sub-14_ses-02_T2w.nii.gz' ),},
                        # Localizer:
                        {'ID' : 7, 'scanType': 'epi_bold', 'condition': 'loc', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-14_ses-02_task-localizer_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-14_ses-02_task-localizer_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-14_ses-02_task-localizer_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-14_ses-02_task-localizer_run-1_eyedata.edf' ),
                            },
                        # Decision tasks:
                        {'ID' : 8, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-14_ses-02_task-yesno_run-1_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-14_ses-02_task-yesno_run-1_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-14_ses-02_task-yesno_run-1_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-14_ses-02_task-yesno_run-1_eyedata.edf' ),
                            },
                        {'ID' : 9, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-14_ses-02_task-yesno_run-2_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-14_ses-02_task-yesno_run-2_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-14_ses-02_task-yesno_run-2_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-14_ses-02_task-yesno_run-2_eyedata.edf' ),
                            },
                        {'ID' : 10, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-14_ses-02_task-yesno_run-3_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-14_ses-02_task-yesno_run-3_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-14_ses-02_task-yesno_run-3_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-14_ses-02_task-yesno_run-3_eyedata.edf' ),
                            },
                        {'ID' : 11, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-14_ses-02_task-yesno_run-4_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-14_ses-02_task-yesno_run-4_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-14_ses-02_task-yesno_run-4_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-14_ses-02_task-yesno_run-4_eyedata.edf' ),
                            },
                        {'ID' : 12, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-14_ses-02_task-yesno_run-5_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-14_ses-02_task-yesno_run-5_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-14_ses-02_task-yesno_run-5_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-14_ses-02_task-yesno_run-5_eyedata.edf' ),
                            },
                        {'ID' : 13, 'scanType': 'epi_bold', 'condition': 'task', 'session' : 2,
                            'rawDataFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-14_ses-02_task-yesno_run-6_bold.nii.gz' ),
                            'rawBehaviorFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-14_ses-02_task-yesno_run-6_events.pickle' ),
                            'physiologyFile': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-14_ses-02_task-yesno_run-6_physio.log' ),
                            'eyeLinkFilePath': os.path.join(this_raw_folder, initials, sj_session2, 'func', 'sub-14_ses-02_task-yesno_run-6_eyedata.edf' ),
                            },
                        ]

                # ----------------------
                # Initialise session   -
                # ----------------------

                runWholeSession( runDecisionArray, presentSession )
