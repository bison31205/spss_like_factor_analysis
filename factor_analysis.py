# -*- coding: utf-8 -*-
# %cd 'C:\Users\talal.mufti\Dropbox (Teneo Holdings)\Computer Backupi\Documents\Nissan\Factor Analysis\SPSS'

# import pandas as pd
from savReaderWriter import  SavReader, SavHeaderReader
import pandas as pd
import numpy as np

from sklearn.decomposition import FactorAnalysis
from sklearn.covariance import ShrunkCovariance
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

from scipy.linalg import eig, eigvals

en_path = 'en_result.sav'
jp_path = 'jp_result.sav'
# with SavHeaderReader(en_path, ioUtf8 =True) as header:
#     metadata = header.all()
#     # report = str(header)
#     header_en = metadata.varLabels
# with SavHeaderReader(jp_path, ioUtf8 =True) as header:
#     metadata = header.all()
#     # report = str(header)
#     header_jp = metadata.varLabels

# # reader_np = SavReaderNp(en_path)
# # array = reader_np.to_array() #doesn't work?

# readerH = SavReader(en_path, returnHeader=1)
# df_en = pd.DataFrame(list(readerH))
# df_en.columns = df_en.iloc[0]
# df_en = df_en.reindex(df_en.index.drop(0))
# df_en = df_en.rename(columns=header_en)
# df_en['lang'] = 'en'

# readerH = SavReader(jp_path, returnHeader=1, ioUtf8 =True)
# df_jp = pd.DataFrame(list(readerH))
# df_jp.columns = df_jp.iloc[0]
# df_jp = df_jp.reindex(df_jp.index.drop(0))
# df_jp = df_jp.rename(columns=header_en)#header_jp)
# df_jp['lang'] = 'jp'

# df = pd.concat([df_en, df_jp]) 

## OR ##
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


def getHeaderDict(path,utf8=True):
    with SavHeaderReader(path, ioUtf8 =utf8) as header:
        return header.all().varLabels

# def Sav2Df(path, verbose_header=True, utf8=True):
#     with SavReader(path, returnHeader=1) as readerH:
#         df = pd.DataFrame(list(readerH))
#     df.columns = df.iloc[0]
#     df = df.reindex(df.index.drop(0))
#     if verbose_header:
#         repl_col = getHeaderDict(path, utf8)
#         df = df.rename(columns=repl_col)
#     return df

def Sav2Df(path, utf8=True):
    with SavReader(path, returnHeader=1) as readerH:
        df = pd.DataFrame(list(readerH))
    df.columns = df.iloc[0]
    df = df.reindex(df.index.drop(0))
    return df

def Sav2DfWithHeader(path, utf8=True):
    df = Sav2Df(path, utf8)
    repl_col = getHeaderDict(path, utf8)
    df = df.rename(columns=repl_col)
    return df

def compute_scores(X):
    fa = FactorAnalysis()

    fa_scores = []
    for n in n_components:
        fa.n_components = n
        fa_scores.append(np.mean(cross_val_score(fa, X)))
    return fa_scores

# def shrunk_cov_score(X):
#     shrinkages = np.logspace(-2, 0, 30)
#     cv = GridSearchCV(ShrunkCovariance(), {'shrinkage': shrinkages})
#     return np.mean(cross_val_score(cv.fit(X).best_estimator_, X))

def find_fa_n_components_by_crossval(X, plot=True):
    fa_scores = compute_scores(X)
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


df = pd.read_csv('combined_results_coded_header.csv') #'combined_results.csv'

remove = [u'CollectorNm', u'RespondentID', u'CollectorID', u'StartDate',
          u'EndDate', u'IPAddress', u'EmailAddress', u'FirstName', u'LastName',
          u'CustomData',u'lang', u'q0011_0001', u'q0011_0002', u'q0011_0003',
          u'q0011_0004',  u'q0011_0005', u'q0011_0006', u'q0011_0007',
          u'q0011_0008',  u'q0011_0009', u'q0011_0010',
          u'q0012', u'q0013',u'q0014', u'q0015', u'q0016', u'q0017', u'q0018_0001', u'q0018_0002'
          ]
header_en = getHeaderDict(en_path)

dfX = df[[x for x in df.columns if x not in remove]]
dfX = dfX.fillna(dfX.mean())#astype(float)

#macro column names
rel_codes = sorted(header_en.keys())[10:-18]

macro_codes = sorted(list(set([x.split('_')[0] for x in rel_codes])))
macro_headers = ['Understanding and managing reputation',
                 'Risk and issues management','Stakeholder management',
                 'Media relations',
                 'Internal communications / Employee engagement',
                 'Business strategy','Campaigning','Digital communication',
                 'Interpersonal communications skills' , 'Writing']
macro_mapping = dict(zip(macro_codes,macro_headers))
# macro_cols = [macro_mapping[c.split('_')[0]] +' '+c.split('_')[1][-1] for c in dfX.columns]
macro_cols = [macro_mapping[c.split('_')[0]] +' '+c.split('_')[1][-1] + ': ' +header_en[c].split(':')[-1] for c in dfX.columns]
macro_repl = dict(zip(dfX.columns,macro_cols))
dfX = dfX.rename(columns=macro_repl)

# or use regular headers 
# dfX = dfX.rename(columns=header_en)#astype(float32)
# cols = dfX.columns

X = dfX.as_matrix()

# covar_init = dfX.cov().as_matrix()
corr_init_df = dfX.corr()
corr_init = corr_init_df.as_matrix()
eig_init = eigvals(corr_init)#[0]
eig_init_perc = eig_init/sum(eig_init) *100 #Intial % Explained variance
eig_init_perc_df = pd.DataFrame(eig_init_perc, columns=['Initial Eigenvalues'])#, index=macro_cols) wrong naming

### adapted from http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html
### all this to decide optimal n_components based on covariance
n_components = np.arange(X.shape[1])#[4,6,8,10,12,14,16]
# n_components_fa = find_fa_n_components_by_crossval(X) ##Output was that n_components_fa = 9

n_components_fa = 10 #10
fa = FactorAnalysis(n_components_fa, random_state=101)
fa.fit(X)
# X_fa = fa.fit_transform(X, n_components_fa)
# print pd.DataFrame(X_fa).head(10)
covar_df = fa.get_covariance()
c = covar_df
corr_extraction = c/np.sqrt(np.multiply.outer(np.diag(c), np.diag(c)))
eig_extraction = eigvals(corr_extraction)#[0]
eig_extraction_perc = eig_extraction/sum(eig_extraction) *100 #Intial % Explained variance
eig_extraction_perc_df = pd.DataFrame(eig_extraction_perc, columns=['Extraction Eigenvalues'])#, index=macro_cols) wrong naming
eig_retain_df = eig_extraction_perc_df.iloc[:n_components_fa]
## np.dot(corr_extraction, corr_extraction.T) #wrong for communality
# communalities = (corr_extraction**2).sum(axis=1)

#components = factors
loads = pd.DataFrame(fa.components_,columns=macro_cols)#cols)
# loads = loads.rename(columns=header_en)
cols = loads.columns
# loads.to_csv('loads_matrix_10_components.csv', index=False, encoding='utf-8')
# OR just load
# loads = pd.read_csv('loads_matrix_10_components.csv', encoding='utf-8')

#not working, TODO
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
