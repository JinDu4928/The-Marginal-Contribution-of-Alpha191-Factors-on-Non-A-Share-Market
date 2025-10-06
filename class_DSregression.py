from sklearn.linear_model import LassoCV
from statsmodels.api import OLS, add_constant
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.decomposition import PCA

class DSregression:
    def __init__(self, alpha_port, usa_port, alpha_ret, usa_ret):
        self.alpha_port = alpha_port
        self.usa_port = usa_port
        self.alpha_ret = alpha_ret
        self.usa_ret = usa_ret
    
    def preprocessData(self):
        # Calculate average returns of portfolios
        self.alpha_port['period'] = pd.to_datetime(self.alpha_port['period'])
        self.usa_port['period'] = pd.to_datetime(self.usa_port['period'])
        usa_port_mean = self.usa_port.groupby('usa_name')[['low1', 'low2', 'medium1', 'medium2', 'high1', 'high2']].mean().reset_index()
        usa_port_mean = usa_port_mean.set_index('usa_name')
        alpha_port_mean = self.alpha_port.groupby('alpha_name')[['low1', 'low2', 'medium1', 'medium2', 'high1', 'high2']].mean().reset_index()
        alpha_port_mean = alpha_port_mean.set_index('alpha_name')
        port_mean = pd.concat([usa_port_mean, alpha_port_mean], axis=0)
        # Convert it into a list
        self.port_ret_mean = port_mean.stack() 
        self.port_ret_mean.index = self.port_ret_mean.index.map(lambda x: f"{x[1]}_{x[0]}")
        self.port_ret_mean = self.port_ret_mean.sort_index()
        # Process portfolio monthly returns
        usa_port_wide = self.usa_port.pivot(index='period', columns='usa_name', values=['low1', 'low2', 'medium1', 'medium2', 'high1', 'high2']).dropna(axis=1, how = "all")
        usa_port_wide.columns = [f"{col[0]}_{col[1]}" for col in usa_port_wide.columns]
        alpha_port_wide = self.alpha_port.pivot(index='period', columns='alpha_name', values=['low1', 'low2', 'medium1', 'medium2', 'high1', 'high2']).dropna(axis=1, how = "all")
        alpha_port_wide.columns = [f"{col[0]}_{col[1]}" for col in alpha_port_wide.columns]
        port_ret = pd.concat([usa_port_wide, alpha_port_wide], axis=1)
        # Process usa and alpha191 monthly returns
        usa_ret_filter = self.usa_ret[(self.usa_ret['period'].dt.year >= 2005) & (self.usa_ret['period'].dt.year <= 2022)].set_index('period')
        alpha_ret_filter = self.alpha_ret[(self.alpha_ret['period'].dt.year >= 2005) & (self.alpha_ret['period'].dt.year <= 2022)].set_index('period')
        # De-mean
        usa_ret_filter = (usa_ret_filter - usa_ret_filter.mean()) / usa_ret_filter.std()
        alpha_ret_filter = (alpha_ret_filter - alpha_ret_filter.mean()) / alpha_ret_filter.std()
        # Calculate the covariance
        self.C_h = port_ret.T@usa_ret_filter / 216
        self.C_h = self.C_h.sort_index().fillna(self.C_h.mean())
        self.C_g = port_ret.T@alpha_ret_filter / 216
        self.C_g = self.C_g.sort_index().fillna(self.C_g.mean())
    
    def DSregression(self):
        # First step LASSO
        lasso_1 = LassoCV(n_alphas=200, cv=10, eps=0.05, max_iter=10000)
        lasso_1.fit(self.C_h, self.port_ret_mean)
        I_1 = set(self.C_h.columns[lasso_1.coef_ != 0])
        # Second step LASSO
        I_2 = set()
        for j in self.C_g.columns:
            lasso_j = LassoCV(n_alphas=200, cv=10, eps=0.05, max_iter=10000)
            lasso_j.fit(self.C_h, self.C_g[j])
            I_2j = set(self.C_h.columns[lasso_j.coef_ != 0])
            I_2 = I_2 | I_2j
        # OLS regression
        I = list(I_1 | I_2)
        X = pd.concat([self.C_h[I], self.C_g], axis=1)
        y = self.port_ret_mean
        res = OLS(y, add_constant(X)).fit()
        print(res.summary())
        return I_1, I_2

    def SSregression(self):
        # First step LASSO
        lasso_1 = LassoCV(n_alphas=200, cv=10, eps=0.05, max_iter=10000)
        lasso_1.fit(self.C_h, self.port_ret_mean)
        I_1 = set(self.C_h.columns[lasso_1.coef_ != 0])
        # OLS regression
        I = list(I_1)
        X = pd.concat([self.C_h[I], self.C_g], axis=1)
        y = self.port_ret_mean
        res = OLS(y, add_constant(X)).fit()
        print(res.summary())
        return I_1
    
    def ENregression(self):
        # First step ElasticNet
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
        model = ElasticNetCV( l1_ratio=l1_ratios, n_alphas=200, cv=10, random_state= 42, max_iter=10000, n_jobs=-1)
        model.fit(self.C_h, self.port_ret_mean)
        I_1 = set(self.C_h.columns[model.coef_ != 0])
        # OLS regression
        I = list(I_1)
        X = pd.concat([self.C_h[I], self.C_g], axis=1)
        y = self.port_ret_mean
        res = OLS(y, add_constant(X)).fit()
        print(res.summary())
        return I_1

    def PCAregression(self):
        pca = PCA()
        X_pca = pca.fit_transform(self.C_h)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= 0.95) + 1
        print(f"The number of original control factors: {self.C_h.shape[1]}")
        print(f"The number of retaining principal components: {n_components}")
        X_controls_pca = X_pca[:, :n_components]
        X_combined = np.hstack([X_controls_pca, self.C_g])
        y = self.port_ret_mean
        res = OLS(y, add_constant(X_combined)).fit()
        print(res.summary())
        return X_controls_pca