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


# params:
version = 3
run = False

# standard params:
model_base_name = '2014_fMRI_data_combined_'
model_names = ['b1', 'b2', 'd1', 'd2', 'all', 'd2b']
nr_samples = 50000
nr_models = 3
parallel = True
accuracy_coding = False

# -----------------
# drift diffusion -
# -----------------
def run_model(trace_id, data, model_dir, model_name, samples=10000, accuracy_coding=False):
    
    import hddm
    
    # m = hddm.HDDMStimCoding(data, stim_col='stimulus', split_param='v', drift_criterion=True, bias=True, include=('sv'), group_only_nodes=['sv'], p_outlier=.05)
    m = hddm.HDDMStimCoding(data, stim_col='stimulus', split_param='v', drift_criterion=True, bias=True, include=('sv'), group_only_nodes=['sv'], depends_on={'t':'split', 'v':'split', 'a':'split', 'dc':'split', 'z':'split', }, p_outlier=.05)
    # m = hddm.HDDMStimCoding(data, stim_col='stimulus', split_param='v', drift_criterion=True, bias=True, include=('sv'), group_only_nodes=['sv'], depends_on={'t':'split', 'v':'split', 'a':'split', 'z':'split', }, p_outlier=.05)
    m.find_starting_values()    
    m.sample(samples, burn=samples/10, thin=3, dbname=os.path.join(model_dir, model_name+ '_db{}'.format(trace_id)), db='pickle')
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
data_path1 = os.path.join('data_files', '2013_PNAS_data_response.csv')
data_path2 = os.path.join('data_files', '2014_fMRI_data_response.csv')
data = pd.concat((pd.read_csv(data_path1), pd.read_csv(data_path2)))

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
subjects = np.unique(data.subj_idx)
nr_subjects = np.unique(data.subj_idx).shape[0]
print '# subjects = {}'.format(nr_subjects)

# make split:
if version == 0:
    pupil_measure = 'pupil_b_lp'
if version == 1:
    pupil_measure = 'pupil_b'
if version == 2 or version == 3 or version == 5:
    pupil_measure = 'pupil_d'

if not version == 4:
    l_ind = []
    h_ind = []
    for subj_idx in subjects:
        d = data[data.subj_idx == subj_idx]
        p_h = []
        p_l = []
        rt = np.array(d['rt'])
        pupil = np.array(d[pupil_measure])
        for s in np.array(np.unique(d['session']), dtype=int):
            if version == 3 or version == 5:
                print 'remove RT!'
                pupil[np.array(d.session == s)] = myfuncs.lin_regress_resid(pupil[np.array(d.session == s)], [rt[np.array(d.session == s)]]) + pupil[np.array(d.session == s)].mean()
            p_l.append( pupil[np.array(d.session == s)] <= np.percentile(pupil[np.array(d.session == s)], 40) )
            p_h.append( pupil[np.array(d.session == s)] >= np.percentile(pupil[np.array(d.session == s)], 60) )
        l_ind.append(np.concatenate(p_l))
        h_ind.append(np.concatenate(p_h))
    l_ind = np.concatenate(l_ind)
    h_ind = np.concatenate(h_ind)
    rest_ind = -(h_ind + l_ind)
    
    # update data:
    data['split'] = np.array(h_ind, dtype=int)
    data = data[-rest_ind]

# # remove authors:
# bad_subjects = np.array([2, 5, 11, 12]) + 22
# for s in bad_subjects:
#     data = data[np.array(data.subj_idx!=s)]
# subjects = np.unique(data.subj_idx)
# nr_subjects = np.unique(data.subj_idx).shape[0]

if run:
    print 'running {}'.format(model_base_name+model_name)
    model = drift_diffusion_hddm(data=data, samples=nr_samples, n_jobs=nr_models, run=run, parallel=parallel, model_name=model_base_name+model_name, model_dir=model_dir, accuracy_coding=accuracy_coding)
else:
    
    model_nr = 2    
    
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
    
    params_of_interest_0 = ['z(0)', 'a(0)', 'v(0)', 'dc(0)', 't(0)', 'sv']
    params_of_interest_1 = ['z(1)', 'a(1)', 'v(1)', 'dc(1)', 't(1)', 'sv']
    params_of_interest_0s = ['z_subj(0)', 'a_subj(0)', 'v_subj(0)', 'dc_subj(0)', 't_subj(0)']
    params_of_interest_1s = ['z_subj(1)', 'a_subj(1)', 'v_subj(1)', 'dc_subj(1)', 't_subj(1)']
    titles = ['Starting point', 'Boundary sep.', 'Drift rate', 'Drift criterion', 'Non-dec. time', 'Drift rate var']
    
    # params_of_interest_0 = ['z(0)', 'a(0)', 'v(0)', 'dc', 't(0)', 'sv']
    # params_of_interest_1 = ['z(1)', 'a(1)', 'v(1)', 'dc', 't(1)', 'sv']
    # params_of_interest_0s = ['z_subj(0)', 'a_subj(0)', 'v_subj(0)', 't_subj(0)']
    # params_of_interest_1s = ['z_subj(1)', 'a_subj(1)', 'v_subj(1)', 't_subj(1)']
    # titles = ['Starting point', 'Boundary sep.', 'Drift rate', 'Drift criterion', 'Non-dec. time', 'Drift rate var']
    
    # point estimates:
    results = model[model_nr].gen_stats()
    results.to_csv(os.path.join(fig_dir, 'diagnostics', 'results.csv'))
    
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
    size_plot = nr_subjects / 3.0 * 1.5
    model[model_nr].plot_posterior_predictive(samples=10, bins=100, figsize=(6,size_plot), save=True, path=os.path.join(fig_dir, 'diagnostics'), format='pdf')
    model[model_nr].plot_posteriors(save=True, path=os.path.join(fig_dir, 'diagnostics'), format='pdf')
    
    # posterios:
    # ----------
    traces_0 = []
    traces_1 = []
    for p in range(len(params_of_interest_0)):
        traces_0.append(model[model_nr].nodes_db.node[params_of_interest_0[p]].trace.gettrace())
        traces_1.append(model[model_nr].nodes_db.node[params_of_interest_1[p]].trace.gettrace())
    
    # # fix starting point:
    # traces_0[0] = traces_0[0] * traces_0[1].mean()
    # traces_1[0] = traces_1[0] * traces_1[1].mean()
    
    stats = []
    for p in range(len(params_of_interest_0)):
        data = [traces_0[p], traces_1[p]]
        stat = np.mean(data[0] > data[1])
        stats.append(min(stat, 1-stat))
    stats = np.array(stats)
    # stats_corrected = mne.stats.fdr_correction(stats, 0.05)[1]
    stats_corrected = stats
    fig, axes = plt.subplots(nrows=1, ncols=len(params_of_interest_0), figsize=(len(params_of_interest_0)*1.5,2.5))
    ax_nr = 0
    for p in range(len(params_of_interest_0)):
        data = [traces_0[p], traces_1[p]]
        ax = axes[ax_nr]
        for d, label, c in zip(data, ['low', 'high'], ['blue', 'red']):
            sns.kdeplot(d, vertical=True, shade=True, color=c, label=label, ax=ax)
            # sns.distplot(d, vertical=True, hist=False, kde_kws={"shade": True}, norm_hist=True, color=c, label=label, ax=ax)
        ax.set_xlabel('Posterior probability')
        ax.set_title(titles[p]+'\np={}'.format(round(stats_corrected[p],4)))
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
    
    
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.hist(d, bins=100)
    ax.set_xlim(0, 0.4)
    ax.set_xlabel('Drift rate variability')
    ax.set_ylabel('MCMC samples')
    ax = fig.add_subplot(212)
    ax.plot(np.linspace(0, 0.4, 1000), sp.stats.gaussian_kde(d).pdf(np.linspace(0, 0.4, 1000)))
    ax.set_xlim(0, 0.4)
    ax.set_ylabel('Posterior probability density')
    ax.set_xlabel('Drift rate variability')
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'hist.pdf'))
    
    
    # paired grid:
    import corner
    df0 = pd.DataFrame(np.array(traces_0).T[:,:5], columns=['z', 'a', 'v', 'dc', 't',])
    df1 = pd.DataFrame(np.array(traces_1).T[:,:5], columns=['z', 'a', 'v', 'dc', 't',])
    df = pd.concat((df0, df1))
    df['pupil'] = np.concatenate((np.zeros(len(df0)), np.ones(len(df1)), ))
    fig = corner.corner(df0, color='b', **{'lw':1})
    corner.corner(df1, color='r', fig=fig, **{'lw':1})
    for i, j in zip(*np.triu_indices_from(np.zeros((5,5)), 1)):
        # add titles:
        r0, p0 = sp.stats.pearsonr(df0.iloc[:,i], df0.iloc[:,j])
        r1, p1 = sp.stats.pearsonr(df1.iloc[:,i], df1.iloc[:,j])
        fig.axes[(j*5)+i].set_title('r={}; r={}'.format(round(r0, 3), round(r1, 3),))
        # add regression lines:
        x_line = np.linspace(fig.axes[(j*5)+i].axis()[0], fig.axes[(j*5)+i].axis()[1], 100)
        (m,b) = sp.polyfit(df0.iloc[:,i], df0.iloc[:,j],1)
        regression_line = sp.polyval([m,b],x_line)
        fig.axes[(j*5)+i].plot(x_line, regression_line, color='b', zorder=3)
        (m,b) = sp.polyfit(df1.iloc[:,i], df1.iloc[:,j],1)
        regression_line = sp.polyval([m,b],x_line)
        fig.axes[(j*5)+i].plot(x_line, regression_line, color='r', zorder=3)
    sns.despine(offset=0, trim=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'corner.png'))
    fig.savefig(os.path.join(fig_dir, 'corner.pdf'))
    
    # g = sns.PairGrid(df, diag_sharey=False, hue='pupil', palette=['r', 'b'], vars=['z', 'a', 'v', 'dc', 't',])
    # # g.map_diag(sns.kdeplot, lw=3)
    # g.map_upper(sns.kdeplot, palette=['r', 'b'])
    # g.map_upper(sns.regplot, scatter=False, ci=None)
    # for i, j in zip(*np.triu_indices_from(g.axes, 1)):
    #     print i,j
    #     r0, p0 = sp.stats.pearsonr(np.array(traces_0).T[:,i], np.array(traces_0).T[:,j])
    #     r1, p1 = sp.stats.pearsonr(np.array(traces_1).T[:,i], np.array(traces_1).T[:,j])
    #     g.axes[i,j].set_title('r={}; r={}'.format(round(r0, 3), round(r1, 3),))
    #     g.axes[0,1].set_title('test')
    # g.savefig(os.path.join(fig_dir, 'corner.png'))
    # # g.savefig(os.path.join(fig_dir, 'corner.pdf'))
    
    # #######
    # p = 5
    # data = [traces_0[p], t0[p]]
    # fig = plt.figure(figsize=(3,3))
    # ax = fig.add_subplot(111)
    # for d, label, c in zip(data, ['All trials', 'TPR fit'], ['black', 'red']):
    #     sns.kdeplot(d, vertical=True, shade=True, color=c, label=label, ax=ax)
    #     # sns.distplot(d, vertical=True, hist=False, kde_kws={"shade": True}, norm_hist=True, color=c, label=label, ax=ax)
    # ax.set_xlabel('Posterior probability')
    # ax.set_ylabel('Drift rate var')
    # ax.set_title(titles[p]+'\np={}'.format(round(np.mean(data[0] > data[1]),4)))
    # plt.tight_layout()
    # sns.despine(offset=10, trim=True)
    # fig.savefig(os.path.join(fig_dir, 'posteriors_sv.pdf'))
    #
    # barplot:
    # --------
    
    # all:
    parameters_h = []
    parameters_l = []
    p_value = []
    ind = np.ones(nr_subjects, dtype=bool)
    for p in range(len(params_of_interest_0s)):
        parameters_h.append(np.array([results.lookup(['{}.'.format(params_of_interest_1s[p]) + str(s)], ['mean']) for s in subjects])[ind].ravel())
        parameters_l.append(np.array([results.lookup(['{}.'.format(params_of_interest_0s[p]) + str(s)], ['mean']) for s in subjects])[ind].ravel())
    
    param_names = ['z', 'a', 'v', 'dc', 't']
    parameters = pd.concat((pd.DataFrame(np.vstack(parameters_h).T, columns=param_names), pd.DataFrame(np.vstack(parameters_l).T, columns=param_names)))
    parameters['pupil'] = np.concatenate((np.ones(len(subjects)), np.zeros(len(subjects))))
    parameters['subject'] = np.concatenate((subjects, subjects))
    k = parameters.groupby(['subject', 'pupil']).mean()
    k_s = k.stack().reset_index()
    k_s.columns = ['subject', 'pupil', 'param', 'value']
    parameters.to_csv(os.path.join(fig_dir, 'params.csv'))
    
    # save source data:
    parameters['data_set'] = np.array(np.concatenate(( np.ones(21)*2, np.ones(14), np.ones(21)*2, np.ones(14) )), dtype=int)
    # parameters['data_set'] = np.array(np.concatenate(( np.ones(21)*2, np.ones(11), np.ones(21)*2, np.ones(11) )), dtype=int)
    df = parameters.ix[:, ['dc', 'pupil', 'data_set']]
    df.columns = ['dc', 'TPR', 'data_set']
    df.to_csv(os.path.join(fig_dir, 'fig4C_source_data.csv'))
    myfuncs.permutationTest(df['dc'][(df.TPR==1)&(df.data_set==1)], df['dc'][(df.TPR==0)&(df.data_set==1)], paired=True)
    myfuncs.permutationTest(df['dc'][(df.TPR==1)&(df.data_set==2)], df['dc'][(df.TPR==0)&(df.data_set==2)], paired=True)
    
    # plot:
    locs = np.arange(0,len(param_names))
    bar_width = 0.2
    fig = plt.figure(figsize=( (1+(len(params_of_interest_1s)*0.3)),2))
    ax = fig.add_subplot(111)
    sns.barplot(x='param',  y='value', units='subject', hue='pupil', hue_order=[1,0], data=k_s, palette=['r', 'b'], ci=None, linewidth=0, alpha=0.5, ax=ax)
    sns.stripplot(x="param", y="value", hue='pupil', hue_order=[1,0], data=k_s, jitter=False, size=2, palette=['r', 'b'], edgecolor='black', linewidth=0.25, ax=ax, split=True, alpha=1)
    for r in range(len(param_names)):
        values = np.vstack((k_s[(k_s['param'] == param_names[r]) & (k_s['pupil'] == 1)].value, k_s[(k_s['param'] == param_names[r]) & (k_s['pupil'] == 0)].value))
        x = np.array([locs[r]-bar_width, locs[r]+bar_width])
        ax.plot(x, values, color='black', lw=0.5, alpha=0.5)
    # # add p-values:
    for r in range(len(param_names)):
        p1 = myfuncs.permutationTest(k_s[(k_s['pupil']==1) & (k_s['param']==param_names[r])].value, k_s[(k_s['pupil']==0) & (k_s['param']==param_names[r])].value, paired=True)[1]
        if p1 < 0.05:
            plt.text(s='{}'.format(round(p1, 3)), x=locs[r], y=plt.gca().get_ylim()[1]-((plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]) / 10.0), size=5, horizontalalignment='center',)
    ax.legend_.remove()
    plt.xticks(locs, param_names, rotation=45)
    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'bars_all.pdf'))
    
    k_s = parameters.groupby(['subject', 'pupil']).mean()
    k_s = k.stack().reset_index()
    k_s.columns = ['subject', 'pupil', 'param', 'value']
    k_s['param'][(k_s['param'] == 'dc') & (k_s['subject'] >= 22)] = 'dc0'
    k_s['param'][(k_s['param'] == 'dc') & (k_s['subject'] < 22)] = 'dc1' 
    k_s = k_s[(k_s['param']=='dc0') | (k_s['param']=='dc1')]
    param_names = ['dc1', 'dc0']
    
    # plot:
    locs = np.arange(0,len(param_names))
    bar_width = 0.2
    fig = plt.figure(figsize=( 2,2))
    ax = fig.add_subplot(111)
    sns.barplot(x='param',  y='value', units='subject', hue='pupil', hue_order=[1,0], data=k_s, palette=['r', 'b'], ci=None, linewidth=0, alpha=0.5, ax=ax)
    sns.stripplot(x="param", y="value", hue='pupil', hue_order=[1,0], data=k_s, jitter=False, size=2, palette=['r', 'b'], edgecolor='black', linewidth=0.25, ax=ax, split=True, alpha=1)
    for r in range(len(param_names)):
        values = np.vstack((k_s[(k_s['param'] == param_names[r]) & (k_s['pupil'] == 1)].value, k_s[(k_s['param'] == param_names[r]) & (k_s['pupil'] == 0)].value))
        x = np.array([locs[r]-bar_width, locs[r]+bar_width])
        ax.plot(x, values, color='black', lw=0.5, alpha=0.5)
    # # add p-values:
    for r in range(len(param_names)):
        p1 = myfuncs.permutationTest(k_s[(k_s['pupil']==1) & (k_s['param']==param_names[r])].value, k_s[(k_s['pupil']==0) & (k_s['param']==param_names[r])].value, paired=True)[1]
        if p1 < 0.05:
            plt.text(s='{}'.format(round(p1, 3)), x=locs[r], y=plt.gca().get_ylim()[1]-((plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]) / 10.0), size=5, horizontalalignment='center',)
    ax.legend_.remove()
    plt.xticks(locs, param_names, rotation=45)
    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'bars_all2.pdf'))