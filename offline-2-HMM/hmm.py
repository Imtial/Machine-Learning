import numpy as np
import pandas as pd
from scipy.stats import norm as gaussian
from scipy import linalg
import math

with open('Sample input and output for HMM/Input/data.txt') as f:
    rainfall = np.loadtxt(f)

with open('Sample input and output for HMM/Input/parameters.txt') as f:
    nstates = int(next(f))
    trans_mat = np.asarray([[float(x) for x in next(f).strip().split('\t')] for i in range(nstates)])
    mu = np.asarray([float(x) for x in next(f).strip().split('\t')])
    sigma = np.asarray([float(x) for x in next(f).strip().split('\t')])

emit_mat = np.asarray([[gaussian.pdf(x, loc=mu[i], scale=math.sqrt(sigma[i])) for x in rainfall] for i in range(nstates)]).T

def initial_dist(trans_mat):
    w, vl = linalg.eig(trans_mat, left=True, right=False)
    vec = vl[:, np.isclose(w, 1)][:, 0]
    return vec / vec.sum()

def viterbi(obs, states, init_prob, trans_mat, emit_mat):
    prev_prob = np.log(init_prob * emit_mat[0]).reshape(1, -1)
    prev_st = np.full((1, states), -1)
    
    for ep in emit_mat[1:]:
        probs = prev_prob[-1].reshape(-1, 1) + np.log(trans_mat * ep)
        prev_prob = np.concatenate((prev_prob, probs.max(axis=0).reshape(1, -1)))
        prev_st = np.concatenate((prev_st, probs.argmax(axis=0).reshape(1, -1)))
    
    most_likely_st = np.array([prev_prob[-1].argmax()])
    for st in prev_st[:0:-1]:
        most_likely_st = np.append(most_likely_st, st[most_likely_st[-1]])
    
    st, counts = np.unique(most_likely_st, return_counts=True)
    
    return most_likely_st, dict((st[i], counts[i]) for i in range(states))
    

def forward(init_prob, trans_mat, emit_mat, norm):
    T = emit_mat.shape[0]
    m = trans_mat.shape[0]
    
    alpha = np.zeros((T, m))
    alpha[0] = init_prob * emit_mat[0]
    norm[0] = 1. / np.sum(alpha[0])
    alpha[0] *= norm[0]
    
    for t in range(1, T):
        for i in range(m):
            alpha[t, i] = alpha[t - 1].dot(trans_mat[:, i] * emit_mat[t, i])
        norm[t] = 1. / np.sum(alpha[t])
        alpha[t] *= norm[t]
            
    return alpha

def backward(trans_mat, emit_mat, norm):
    T = emit_mat.shape[0]
    m = trans_mat.shape[0]
    
    beta = np.zeros((T, m))
    beta[-1] = np.ones(m)
    
    for t in range(T-2, -1, -1):
        for i in range(m):
            beta[t, i] = trans_mat[i].dot(beta[t+1] * emit_mat[t+1])
        beta[t] *= norm[t]
    
    return beta

def baum_welch(obs, init_prob, trans_mat, emit_mat, n_iter):
    T = emit_mat.shape[0]
    m = trans_mat.shape[0]
    
    updated_trans_mat = trans_mat.copy()
    updated_emit_mat = emit_mat.copy()
    
    norm = np.zeros(T)
    
    for n in range(n_iter):
        alpha = forward(init_prob, updated_trans_mat, updated_emit_mat, norm)
        beta = backward(updated_trans_mat, updated_emit_mat, norm)

        alphaXbeta = alpha * beta / norm.reshape(-1, 1)
        denoms = alphaXbeta.sum(axis=0)
        for i in range(m):
            for j in range(m):
                numer = 0
                for t in range(T-1):
                    numer += alpha[t, i] * updated_trans_mat[i, j] * beta[t+1, j] * updated_emit_mat[t+1, j]
                updated_trans_mat[i, j] = numer / denoms[i]
                
        alphaXbeta *= norm.reshape(-1, 1)
        alphaXbeta /= alphaXbeta.sum(axis=1).reshape(-1, 1)

        tmp = alphaXbeta * obs.reshape(-1, 1)
        _mu_ = tmp.sum(axis=0) / alphaXbeta.sum(axis=0)

        diff = obs.reshape(-1, 1) - _mu_
        _sigma_ = (alphaXbeta * (diff ** 2)).sum(axis=0) / alphaXbeta.sum(axis=0)

        updated_emit_mat = np.asarray([[gaussian.pdf(x, loc=_mu_[i], scale=math.sqrt(_sigma_[i])) for x in obs] for i in range(m)]).T
        
    return updated_trans_mat, _mu_, _sigma_

with open('viterbi_wo_learning.txt', 'w') as f:
    emit_mat = np.asarray([[gaussian.pdf(x, loc=mu[i], scale=math.sqrt(sigma[i])) for x in rainfall] for i in range(nstates)]).T
    init_prob = np.full(nstates, 1/nstates)
    most_likely_states, state_counts = viterbi(rainfall, nstates, init_prob, trans_mat, emit_mat)
    most_likely_states = ['El Nino' if x == 0 else 'La Nina' for x in most_likely_states]
    f.write('\n'.join(most_likely_states))
    print(state_counts)
    
with open('parameters_learned.txt', 'w') as f:
    init_prob = initial_dist(trans_mat)
    trans_mat_trained, mu_trained, sigma_trained = baum_welch(rainfall, init_prob, trans_mat, emit_mat, n_iter=20)
    f.write(str(nstates)+'\n')
    
    [f.write('\t'.join([str(v) for v in row])+'\n') for row in trans_mat_trained]
    f.write('\t'.join([str(mu_x) for mu_x in mu_trained]) + '\n')
    f.write('\t'.join([str(sigma_x) for sigma_x in sigma_trained]) + '\n')
    f.write('\t'.join([str(pi_x) for pi_x in init_prob]))
    
with open('viterbi_after_learning.txt', 'w') as f:
    emit_mat_trained = np.asarray([[gaussian.pdf(x, loc=mu_trained[i], scale=math.sqrt(sigma_trained[i])) for x in rainfall] for i in range(nstates)]).T
    most_likely_states_after_train, state_counts_after_train = viterbi(rainfall, nstates, init_prob, trans_mat_trained, emit_mat_trained)
    most_likely_states_after_train = ['El Nino' if x == 0 else 'La Nina' for x in most_likely_states_after_train]
    f.write('\n'.join(most_likely_states_after_train))
    print(state_counts_after_train)