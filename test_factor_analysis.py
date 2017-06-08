### Recreating: http://www.ats.ucla.edu/stat/spss/output/factor1.htm
# data from http://www.ats.ucla.edu/stat/spss/output/principal_components_files/M255.SAV


# %cd 'C:\Users\talal.mufti\Dropbox (Teneo Holdings)\Computer Backupi\Documents\Nissan\Factor Analysis\SPSS'
import pandas as pd
import numpy as np

from savReaderWriter import SavReader, SavHeaderReader
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from scipy.linalg import eig

import matplotlib.pyplot as plt
import math

### SPSS SAV loading tools ###
### See SavWriter to write back to .SAV file.
### Writing not covered here

def getHeaderDict(path,utf8=True):
'''Get Header from SPSS .SAV file '''
    with SavHeaderReader(path, ioUtf8 =utf8) as header:
        return header.all().varLabels

def Sav2Df(path, utf8=True):
''' Load data from SPSS .SAV file into pandas DataFrame with coded columns'''
    with SavReader(path, returnHeader=1) as readerH:
        df = pd.DataFrame(list(readerH))
    df.columns = df.iloc[0]
    df = df.reindex(df.index.drop(0))
    return df

def Sav2DfWithHeader(path, utf8=True):
''' Load data from SPSS .SAV file into pandas DataFrame with
    renamed columns using header'''
    df = Sav2Df(path, utf8)
    repl_col = getHeaderDict(path, utf8)
    df = df.rename(columns=repl_col)
    return df

def get_univariate_table(dfX):
''' Equivalent to SPSS univariate table function'''
    univariate_table = dfX.describe() #descriptive statistics
    return univariate_table

def get_initial_percent_explained_variance(dfX):
''' Normalized Eigenvalues'''
    covar_init = dfX.cov().as_matrix()
    eig_init = eig(covar_init)[0]
    eig_init_perc = eig_init/sum(eig_init) *100 #Intial % Explained variance
    eig_init_perc_df = pd.DataFrame(eig_init_perc, columns=['Initial Eigenvalues'])#, index=macro_cols) wrong naming
    return eig_init_perc_df


# KMO measure from "Comparative Approaches to Using R and Python for Statistical Data Analysis"
# Note: Highly Unnpythonic
# https://books.google.com/books?id=xpjgDQAAQBAJ&dq=python+Kaiser-Meyer-Olkin&source=gbs_navlinks_s
def global_kmo(dataset_corr):
    # Inverse of the correlation matrix
    # dataset_corr is the correlation matrix of the survey results 
    corr_inv = np.linalg.inv(dataset_corr)
    # number of rows and number of columns
    nrow_inv_corr, ncol_inv_corr = dataset_corr.shape
    # a Partial correlation matrix
    A = np.ones((nrow_inv_corr,ncol_inv_corr))
    for i in xrange(nrow_inv_corr):
        for j in xrange(ncol_inv_corr) :
            #above the diagonal
            A[i,j] = - (corr_inv[i,j])/(math.sqrt(corr_inv[i,i] * corr_inv[j,j])) 
            #below the diagonal
            A[j,i] = A[i,j]
    #transform to an array of arrays ("matrix" with Python)
    dataset_corr = np.asarray(dataset_corr)
    # KMO value
    kmo_num = np.sum(np.square(dataset_corr))-np.sum(np.square(np.diagonal(dataset_corr)))
    kmo_denom = kmo_num + np.sum(np.square(A)) - np.sum(np.square(np.diagonal(A)))
    kmo_value = kmo_num / kmo_denom
    return kmo_value

## Not tested
def KMO_per_variable(dataset_corr):
    # Inverse of the correlation matrix
    # dataset_corr is the correlation matrix of the survey results 
    corr_inv = np.linalg.inv(dataset_corr)
    # number of rows and number of columns
    nrow_inv_corr, ncol_inv_corr = dataset_corr.shape
    # a Partial correlation matrix
    A = np.ones((nrow_inv_corr,ncol_inv_corr))
    for i in xrange(nrow_inv_corr):
        for j in xrange(ncol_inv_corr) :
            #above the diagonal
            A[i,j] = - (corr_inv[i,j])/(math.sqrt(corr_inv[i,i] * corr_inv[j,j])) 
            #below the diagonal
            A[j,i] = A[i,j]
    #transform to an array of arrays ("matrix" with Python)
    dataset_corr = np.asarray(dataset_corr)
    # # KMO value
    # kmo_num = np.sum(np.square(dataset_corr))-np.sum(np.square(np.diagonal(dataset_corr)))
    # kmo_denom = kmo_num + np.sum(np.square(A)) - np.sum(np.square(np.diagonal(A)))
    # kmo_value = kmo_num / kmo_denom
    # creation of an empty vector to store the results per variable. The 
    # size of the vector is equal to the number H...of variables
    kmo_j = [None]*dataset_corr.shape[1]
    for j in range(0, dataset_corr.shape[1]):
        kmo_j_num = np.sum(dataset_corr[:,[1]] ** 2) - dataset_corr[j,j] ** 2 
        kmo_j_denom = kmo_j_num + np.sum(A[:,[j]] ** 2) - A[j,j] ** 2 
        kmo_j[j] = kmo_j_num / kmo_j_denom
    return kmo_j

#from http://stackoverflow.com/questions/17628589/perform-varimax-rotation-in-python-using-numpy
# and https://en.wikipedia.org/wiki/Talk:Varimax_rotation
def varimax(Phi, gamma = 1, q = 20, tol = 1e-6):
    from numpy import eye, asarray, dot, sum, diag
    from numpy.linalg import svd
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in xrange(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d/d_old < tol: break
    return dot(Phi, R)


def compute_fa_scores(X):
    fa = FactorAnalysis()
    fa_scores = []
    for n in n_components:
        fa.n_components = n
        fa_scores.append(np.mean(cross_val_score(fa, X)))
    return fa_scores

def find_fa_n_components_by_crossval(X, n_components, plot=True):
    fa_scores = compute_fa_scores(X)
    n_components_fa = n_components[np.argmax(fa_scores)]

    print("best n_components by FactorAnalysis CV = %d" % n_components_fa)
    if plot:
        plt.figure()
        plt.plot(n_components, fa_scores, 'r', label='FA scores')

        plt.axvline(n_components_fa, color='r',
                    label='FactorAnalysis CV: %d' % n_components_fa,
                    linestyle='--')
        plt.xlabel('nb of components')
        plt.ylabel('CV scores')
        plt.legend(loc='lower right')

        plt.show()
    return n_components_fa


def main():
    # load data from http://www.ats.ucla.edu/stat/spss/output/principal_components_files/M255.SAV
    df = Sav2Df('M255.SAV')
    hmap = getHeaderDict('M255.SAV')

    # keep columns as stipulated in original article 
    keep_cols = ['item13', 'item14', 'item15', 'item16', 'item17', 'item18', 
                 'item19', 'item20', 'item21', 'item22', 'item23', 'item24']

    dfX = df[keep_cols].rename(columns=hmap)#astype(float32)
    X = dfX.as_matrix()

    univariate_table = get_univariate_table(dfX) #descriptive statistics

    eig_init_perc_df = get_initial_percent_explained_variance(dfX)


    ### Decide optimal n_components based on cross validation
    # adapted from http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html

    n_components = np.arange(X.shape[1])#[4,6,8,10,12,14,16] #range(12)
    n_components_fa = find_fa_n_components_by_crossval(X, n_components)

    ##here drop was after 4 components. in original example 3 is deemed optimum

    # with n selected, fit a FactorAnalysis model
    n_components_fa = 3#4
    fa = FactorAnalysis(n_components_fa, random_state=101)
    factor = fa.fit(X)

    covar = fa.get_covariance()
    
    # print global_kmo(covar) #results not yet validated

    eig_extraction_perc_df = get_factor_extraction_percent_explained_variance(covar)
    eig_retain_df = eig_extraction_perc_df.iloc[:n_components_fa]

    loads = pd.DataFrame(fa.components_,columns=macro_cols)#cols)
    # loads = loads.rename(columns=header_en)
    cols = loads.columns
    # loads.to_csv('loads_matrix_10_components.csv', index=False, encoding='utf-8')
    # OR just load
    # loads = pd.read_csv('loads_matrix_10_components.csv', encoding='utf-8')

    ## Write each output table to a sheet in an excel file

    writer = pd.ExcelWriter('factor_analysis_{}_components_varimax.xlsx'.format(n_components_fa),
                         engine='xlsxwriter', options={'encoding':'utf-8'})


    cutoff = 0.3#.3
    for i in xrange(len(loads)):
        s = loads.iloc[i].sort_values( ascending=False).reset_index()
        s = s[(s[i]>cutoff) | (s[i]<-cutoff)]
        # print s
        s.rename(columns={'index':'component'}).to_excel(writer, 'Postive Sorted Loads', encoding='utf-8', startcol=3*i, index=False)

    for i in xrange(len(loads)):
        s = loads.iloc[i].sort_values( ascending=False).reset_index()
        s.rename(columns={'index':'component'}).to_excel(writer, 'Loads', encoding='utf-8', startcol=3*i, index=False)

    loads.rename(columns=header_en).to_excel(writer, 'Raw Components', encoding='utf-8' )

    # repeat for rotated loads
    loads = pd.DataFrame(varimax(fa.components_),columns=macro_cols)
    cutoff = 0.3#.3
    for i in xrange(len(loads)):
        s = loads.iloc[i].sort_values( ascending=False).reset_index()
        s = s[(s[i]>cutoff) | (s[i]<-cutoff)]
        # print s
        s.rename(columns={'index':'component'}).to_excel(writer, 'Varimax Rot. Pos. Sorted Loads', encoding='utf-8', startcol=3*i, index=False)

    for i in xrange(len(loads)):
        s = loads.iloc[i].sort_values( ascending=False).reset_index()
        s.rename(columns={'index':'component'}).to_excel(writer, 'Varimax Rot. Loads', encoding='utf-8', startcol=3*i, index=False)

    loads.rename(columns=header_en).to_excel(writer, 'Varimax Raw Components', encoding='utf-8' )

    ### explained variances and noise

    # levels = [['Initial Eigenvalues', 'Extraction'],
    #           ['Total', '\% \of Variance', 'Cumulative \%']]
    # eig_df = pd.DataFrame(columns=pd.MANUALLY_DIGESTED)

    eig_init_perc_df.to_excel(writer, 'Eigenvals (perc. explained var)',  encoding='utf-8')
    eig_retain_df.to_excel(writer, 'Eigenvals (perc. explained var)',startcol=3,  encoding='utf-8')


    noise_var = pd.DataFrame(fa.noise_variance_, index=macro_cols)#.rename(index=header_en)
    noise_var.to_excel(writer, 'Noise Variance', encoding='utf-8')

    covar_df = pd.DataFrame(covar_extraction, index=cols, columns=cols
                            ).rename(columns=header_en, index=header_en)
    covar_df.to_excel(writer, 'Covariance Matrix', encoding='utf-8')

    corr_init_df.to_excel(writer, 'Correlation Mat(Data)', encoding='utf-8')

    corr_extraction_df = pd.DataFrame(corr_extraction, index=macro_cols, columns=macro_cols
                            )#.rename(columns=header_en, index=header_en)
    corr_extraction_df.to_excel(writer, 'Corr Mat of Factor Loads ', encoding='utf-8')

    precision_mat = pd.DataFrame(fa.get_precision(), index=cols, columns=cols).rename(columns=header_en)
    precision_mat.to_excel(writer, 'Precision Matrix', encoding='utf-8')


    writer.save()