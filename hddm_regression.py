#!/usr/bin/env python
# encoding: utf-8
"""
Created by Jan Willem de Gee on 2011-02-16.
Copyright (c) 2011 __MyCompanyName__. All rights reserved.
"""
import os, sys, pickle, time
import datetime
import math
import numpy as np
import scipy as sp
import matplotlib
matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import itertools
import pp
from IPython import embed as shell
import hddm
import kabuki
import mne
import statsmodels.formula.api as sm

sys.path.append(os.environ['ANALYSIS_HOME'])
from Tools.other_scripts import functions_jw as myfuncs

matplotlib.rcParams['pdf.fonttype'] = 42
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

# params:
version = 1
run = False

# standard params:
model_base_name = '2014_fMRI_data_combined_'
model_names = ['r1', 'r1b', 'r2', 'r3', 'r4', 'r5']
nr_samples = 100000
nr_models = 3
parallel = True
accuracy_coding = False

# -----------------
# drift diffusion -
# -----------------
def run_model(trace_id, data, model_dir, model_name, samples=10000, accuracy_coding=False):
    
    import os
    import numpy as np
    import hddm
    from patsy import dmatrix  
    
    ## version 0 ##
    v_reg = {'model': 'v ~ 1 + stimulus + BS_nMod + BS_C', 'link_func': lambda x: x}
    reg_descr = [v_reg]
    m = hddm.HDDMRegressor(data, reg_descr, include=('z'), p_outlier=.05, group_only_regressors=False)
    m.find_starting_values()
    m.sample(samples, burn=samples/10, thin=2, dbname=os.path.join(model_dir, model_name+ '_db{}'.format(trace_id)), db='pickle')

    # ## version 1 ##
    # v_reg = {'model': 'v ~ 1 + stimulus + combined_choice_all * pupil_d', 'link_func': lambda x: x}
    # reg_descr = [v_reg]
    # m = hddm.HDDMRegressor(data, reg_descr, include=('z'), p_outlier=.05, group_only_regressors=False)
    # m.find_starting_values()
    # m.sample(samples, burn=samples/10, thin=2, dbname=os.path.join(model_dir, model_name+ '_db{}'.format(trace_id)), db='pickle')
    #
    # ## version 2 ##
    # v_reg = {'model': 'v ~ 1 + stimulus + M1 * pupil_d', 'link_func': lambda x: x}
    # reg_descr = [v_reg]
    # m = hddm.HDDMRegressor(data, reg_descr, include=('z'), p_outlier=.05, group_only_regressors=False)
    # m.find_starting_values()
    # m.sample(samples, burn=samples/10, thin=2, dbname=os.path.join(model_dir, model_name+ '_db{}'.format(trace_id)), db='pickle')
    #
    # ## version 3 ##
    # v_reg = {'model': 'v ~ 1 + stimulus + combined_choice_lat * pupil_d', 'link_func': lambda x: x}
    # reg_descr = [v_reg]
    # m = hddm.HDDMRegressor(data, reg_descr, include=('z'), p_outlier=.05, group_only_regressors=False)
    # m.find_starting_values()
    # m.sample(samples, burn=samples/10, thin=2, dbname=os.path.join(model_dir, model_name+ '_db{}'.format(trace_id)), db='pickle')
    #
    # ## version 4 ##
    # v_reg = {'model': 'v ~ 1 + stimulus + combined_choice_sl * pupil_d', 'link_func': lambda x: x}
    # reg_descr = [v_reg]
    # m = hddm.HDDMRegressor(data, reg_descr, include=('z'), p_outlier=.05, group_only_regressors=False)
    # m.find_starting_values()
    # m.sample(samples, burn=samples/10, thin=2, dbname=os.path.join(model_dir, model_name+ '_db{}'.format(trace_id)), db='pickle')
    
    return m
    
def drift_diffusion_hddm(data, samples=10000, n_jobs=6, run=True, parallel=True, model_name='model', model_dir='.', accuracy_coding=False):
    
    import hddm
    import os
    
    # run the model:
    if run:
        if parallel:
            job_server = pp.Server(ppservers=(), ncpus=n_jobs)
            start_time = time.time()
            jobs = [(trace_id, job_server.submit(run_model,(trace_id, data, model_dir, model_name, samples, accuracy_coding), (), ('hddm',))) for trace_id in range(n_jobs)]
            results = []
            for s, job in jobs:
                results.append(job())
            print "Time elapsed: ", time.time() - start_time, "s"
            job_server.print_stats()
            
            # save:
            for i in range(n_jobs):
                model = results[i]
                model.save(os.path.join(model_dir, '{}_{}'.format(model_name,i)))
        else:
            model = run_model(1, data, model_dir, model_name, samples, accuracy_coding)
            model.save(os.path.join(model_dir, model_name))
    
    # load the models:
    else:
        print 'loading existing model(s)'
        if parallel:
            model = []
            for i in range(n_jobs):
                model.append(hddm.load(os.path.join(model_dir, '{}_{}'.format(model_name,i))))
        else:
            model = hddm.load(os.path.join(model_dir, model_name))
    return model

# settings:
# ---------

# model_name:
model_name = model_names[version]

# data:
data_path1 = os.path.join('data_files', '2014_fMRI_data_response_brainstem.csv')
data = pd.read_csv(data_path1)

# model dir:
model_dir = 'model_dir/'

# figures dir:
fig_dir = os.path.join('figures', model_base_name + model_name)
try:
    os.system('mkdir {}'.format(fig_dir))
    os.system('mkdir {}'.format(os.path.join(fig_dir, 'diagnostics')))
except:
    pass

# subjects:
# remove authors:
bad_subjects = np.array([2, 5, 11, 12])
for s in bad_subjects:
    data = data[np.array(data.subj_idx!=s)]
subjects = np.unique(data.subj_idx)
nr_subjects = np.unique(data.subj_idx).shape[0]
print '# subjects = {}'.format(nr_subjects)

# fix stimulus:
data.ix[data['stimulus']==0,'stimulus'] = -1

# zscore:
for s in np.unique(data.subj_idx):
    data.ix[data['subj_idx']==s,'pupil_d'] = (data.ix[data['subj_idx']==s,'pupil_d'] - np.mean(data.ix[data['subj_idx']==s,'pupil_d'])) / np.std(data.ix[data['subj_idx']==s,'pupil_d'])
    data.ix[data['subj_idx']==s,'BS_nMod'] = (data.ix[data['subj_idx']==s,'BS_nMod'] - np.mean(data.ix[data['subj_idx']==s,'BS_nMod'])) / np.std(data.ix[data['subj_idx']==s,'BS_nMod'])
    data.ix[data['subj_idx']==s,'BS_C'] = (data.ix[data['subj_idx']==s,'BS_C'] - np.mean(data.ix[data['subj_idx']==s,'BS_C'])) / np.std(data.ix[data['subj_idx']==s,'BS_C'])
    data.ix[data['subj_idx']==s,'M1'] = (data.ix[data['subj_idx']==s,'M1'] - np.mean(data.ix[data['subj_idx']==s,'M1'])) / np.std(data.ix[data['subj_idx']==s,'M1'])
    data.ix[data['subj_idx']==s,'combined_choice_all'] = (data.ix[data['subj_idx']==s,'combined_choice_all'] - np.mean(data.ix[data['subj_idx']==s,'combined_choice_all'])) / np.std(data.ix[data['subj_idx']==s,'combined_choice_all'])
    data.ix[data['subj_idx']==s,'combined_choice_lat'] = (data.ix[data['subj_idx']==s,'combined_choice_lat'] - np.mean(data.ix[data['subj_idx']==s,'combined_choice_lat'])) / np.std(data.ix[data['subj_idx']==s,'combined_choice_lat'])
    data.ix[data['subj_idx']==s,'combined_choice_sl'] = (data.ix[data['subj_idx']==s,'combined_choice_sl'] - np.mean(data.ix[data['subj_idx']==s,'combined_choice_sl'])) / np.std(data.ix[data['subj_idx']==s,'combined_choice_sl'])
# # version:
# if version == 0:
#     data['phys'] = data['BS_nMod']
# if version == 1:
#     data['phys'] = data['BS_C']
   
if run:
    print 'running {}'.format(model_base_name+model_name)
    model = drift_diffusion_hddm(data=data, samples=nr_samples, n_jobs=nr_models, run=run, parallel=parallel, model_name=model_base_name+model_name, model_dir=model_dir, accuracy_coding=accuracy_coding)
else:
    
    model_nr = 0    
    
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
    
    model = drift_diffusion_hddm(data=data, samples=nr_samples, n_jobs=nr_models, run=run, parallel=parallel, model_name=model_base_name+model_name, model_dir=model_dir, accuracy_coding=accuracy_coding)
    
    if version == 0:
        params_of_interest = ['z', 'a', 't', 'v_Intercept', 'v_stimulus', 'v_BS_nMod', 'v_BS_C',]
        params_of_interest_s = ['z_subj', 'a_subj', 't_subj', 'v_Intercept_subj', 'v_stimulus_subj', 'v_BS_nMod', 'v_BS_C_subj']
        titles = ['Starting point', 'Boundary sep.', 'Non-dec. time', 'v_Intercept rate', 'Drift v_stimulus', 'v_BS_nMod', 'v_BS_C',]
    if version == 1:
        params_of_interest = ['z', 'a', 't', 'v_Intercept', 'v_stimulus', 'v_BS_nMod', 'v_BS_C',]
        params_of_interest_s = ['z_subj', 'a_subj', 't_subj', 'v_Intercept_subj', 'v_stimulus_subj', 'v_BS_nMod', 'v_BS_C_subj']
        titles = ['Starting point', 'Boundary sep.', 'Non-dec. time', 'v_Intercept rate', 'Drift v_stimulus', 'v_BS_nMod', 'v_BS_C',]
    if version == 2:
        params_of_interest = ['z', 'a', 't', 'v_Intercept', 'v_stimulus', 'v_combined_choice_all', 'v_pupil_d', 'v_combined_choice_all:pupil_d']
        params_of_interest_s = ['z_subj', 'a_subj', 't_subj', 'v_Intercept_subj', 'v_stimulus_subj', 'v_BS_nMod', 'v_BS_C_subj']
        titles = ['Starting point', 'Boundary sep.', 'Non-dec. time', 'v_Intercept rate', 'Drift v_stimulus', 'v_combined_choice', 'v_pupil', 'v_interaction']
    if version == 3:
        params_of_interest = ['z', 'a', 't', 'v_Intercept', 'v_stimulus', 'v_M1', 'v_pupil_d', 'v_M1:pupil_d']
        params_of_interest_s = ['z_subj', 'a_subj', 't_subj', 'v_Intercept_subj', 'v_stimulus_subj', 'v_BS_nMod', 'v_BS_C_subj']
        titles = ['Starting point', 'Boundary sep.', 'Non-dec. time', 'v_Intercept rate', 'Drift v_stimulus', 'v_M1', 'v_pupil', 'v_interaction']
    if version == 4:
        params_of_interest = ['z', 'a', 't', 'v_Intercept', 'v_stimulus', 'v_combined_choice_lat', 'v_pupil_d', 'v_combined_choice_lat:pupil_d']
        params_of_interest_s = ['z_subj', 'a_subj', 't_subj', 'v_Intercept_subj', 'v_stimulus_subj', 'v_BS_nMod', 'v_BS_C_subj']
        titles = ['Starting point', 'Boundary sep.', 'Non-dec. time', 'v_Intercept rate', 'Drift v_stimulus', 'v_combined_choice_lat', 'v_pupil', 'v_interaction']
    if version == 5:
        params_of_interest = ['z', 'a', 't', 'v_Intercept', 'v_stimulus', 'v_combined_choice_sl', 'v_pupil_d', 'v_combined_choice_sl:pupil_d']
        params_of_interest_s = ['z_subj', 'a_subj', 't_subj', 'v_Intercept_subj', 'v_stimulus_subj', 'v_BS_nMod', 'v_BS_C_subj']
        titles = ['Starting point', 'Boundary sep.', 'Non-dec. time', 'v_Intercept rate', 'Drift v_stimulus', 'v_combined_choice_lat', 'v_pupil', 'v_interaction']
    
    shell()
    
    # gelman rubic:
    gr = hddm.analyze.gelman_rubin(model)
    text_file = open(os.path.join(fig_dir, 'diagnostics', 'gelman_rubic.txt'), 'w')
    for p in gr.items():
        text_file.write("%s:%s\n" % p)
    text_file.close()

    # dic:
    text_file = open(os.path.join(fig_dir, 'diagnostics', 'DIC.txt'), 'w')
    for m in range(nr_models):
        text_file.write("Model {}: {}\n".format(m, model[m].dic))
    text_file.close()
    
    # # analytic plots:
    # size_plot = nr_subjects / 3.0 * 1.5
    # model[model_nr].plot_posterior_predictive(samples=10, bins=100, figsize=(6,size_plot), save=True, path=os.path.join(fig_dir, 'diagnostics'), format='pdf')
    # model[model_nr].plot_posteriors(save=True, path=os.path.join(fig_dir, 'diagnostics'), format='pdf')
    
    # point estimates:
    results = model[model_nr].gen_stats()
    results.to_csv(os.path.join(fig_dir, 'diagnostics', 'results.csv'))
    
    # posterios:
    # ----------
    traces = []
    for p in range(len(params_of_interest)):
        traces.append(model[model_nr].nodes_db.node[params_of_interest[p]].trace.gettrace())
    
    # fix starting point:
    traces[0] = 1/(1+np.exp(-(traces[0])))
    
    stats = []
    for p in range(len(params_of_interest)):
        stat = np.min(np.mean(traces[p] > 0), np.mean(traces[p] < 0))
        stats.append(min(stat, 1-stat))
    stats = np.array(stats)
    # stats_corrected = mne.stats.fdr_correction(stats, 0.05)[1]
    stats_corrected = stats
    fig, axes = plt.subplots(nrows=1, ncols=len(params_of_interest), figsize=(len(params_of_interest)*1.5,2.5))
    ax_nr = 0
    for p in range(len(params_of_interest)):
        data = [traces[p]]
        ax = axes[ax_nr]
        for d, label, c in zip(data, ['1',], ['black']):
            sns.kdeplot(d, vertical=True, shade=True, color=c, label=label, ax=ax)
            ax.set_xlabel('Posterior probability')
            ax.set_title('{}\np={}'.format(titles[ax_nr], round(stats[p],3)))
            ax.set_xlim(xmin=0)
            # ax.set_ylim(-1,2)
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(0.5)
                ax.tick_params(width=0.5)
            ax_nr+=1
    sns.despine(offset=10, trim=True)
    axes[0].set_ylabel('Parameter estimate (a.u.)')
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'posteriors.pdf'))
    
    
    
    # # all:
    # parameters = []
    # ind = np.ones(nr_subjects, dtype=bool)
    # for p in range(len(params_of_interest_s)):
    #     parameters.append(np.array([results.lookup(['{}.'.format(params_of_interest_s[p]) + str(s)], ['mean']) for s in subjects])[ind].ravel())
    # parameters = pd.DataFrame(np.vstack(parameters).T, columns=titles)
    # parameters['subject'] = subjects
    # parameters.to_csv(os.path.join(fig_dir, 'params.csv'))
    # params_ori = pd.read_csv(os.path.join(fig_dir, 'params_ori.csv'))
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # myfuncs.correlation_plot(parameters.intercept, params_ori.dc, line=True, ax=ax)
    # fig.savefig(os.path.join(fig_dir, 'correlation.pdf'))
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # myfuncs.correlation_plot(parameters.a, params_ori.a, line=True, ax=ax)
    # fig.savefig(os.path.join(fig_dir, 'correlation_a.pdf'))
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # myfuncs.correlation_plot(parameters.t, params_ori.t, line=True, ax=ax)
    # fig.savefig(os.path.join(fig_dir, 'correlation_t.pdf'))
    
    
    
    
    
    
    