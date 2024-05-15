import numpy as np
import pymc as pm
import arviz as az

from scipy.stats import weibull_min, norm

__all__ = ['Generate_Windspeed_Realizations']

def Generate_Windspeed_Realizations(data, ensemble_size=100, random_seed=100, save_summary=False):

    time = np.arange(data.size)

    # fit weibull
    weibull_params = weibull_min.fit(data)
    data_gaussian  = norm.ppf(weibull_min.cdf(data, *weibull_params))
    
    # make model
    WindSpeedModel = pm.Model()
    WindSpeedModel.add_coord('time', time, mutable=True)

    with WindSpeedModel:

        # add data
        t = pm.MutableData('t', time, dims='time') # time
        y = pm.MutableData('y', data_gaussian, dims='time') # windpseed

        # prior for Auto Reg. model
        rho = pm.Normal('ρ', mu=[0.0, 0.9], sigma=[0.1, 0.1]) # ρ0 and ρ1
        std = pm.HalfNormal('σ', sigma=2.0)
        ar_init = pm.Normal.dist(mu=data_gaussian[0], sigma=0.1, size=1)
        
        # make Auto regressive model.
        ar_model  = pm.AR('ar_model', 
                          rho=rho, 
                          sigma=std, 
                          init_dist=ar_init,
                          constant=True,
                          steps=time.size-1,
                          dims='time')

        # priors for seasonality
        amp = pm.Normal('a', mu=1.0, sigma=0.1)
        phi = pm.Normal('ϕ', mu=0.0, sigma=np.pi/2)
        lam = pm.Normal('λ', mu=1.0, sigma=0.1)*365
        seasonality = pm.Deterministic('seasonality', 
                                       amp*pm.math.cos(2*np.pi*t/lam + phi), 
                                       dims='time')

        # define a Gaussian likelihood
        likelihood = pm.Normal('likelihood',
                               mu=ar_model + seasonality,
                               sigma=std,
                               observed=y,
                               dims='time')
        
        # sample
        trace = pm.sample(draws=2000, random_seed=random_seed, chains=2, cores=2, target_accept=0.95)


    # predictions  
    with WindSpeedModel:
        
        
        WindSpeedModel.add_coord('time_pred', time)
        t_pred = pm.MutableData('t_pred', time, dims='time_pred')

        ar_model_pred = pm.AR('ar_model_pred',
                              init_dist=ar_init,
                              rho=rho, 
                              sigma=std,
                              #constant=True,
                              dims='time_pred')
        
        mu_pred = ar_model_pred + amp*pm.math.cos(2*np.pi*t_pred/lam + phi)
        y_pred  = pm.Normal('y_pred', mu=mu_pred, sigma=std, dims='time_pred')

        # sample posterior
        trace.extend(pm.sample_posterior_predictive(trace.sel(draw=slice(None, None, int(2000/ensemble_size))), 
                                                    var_names=['y_pred'], 
                                                    predictions=True, 
                                                    random_seed=random_seed))
        
        # plot trace
        az.plot_posterior(trace, 
                          var_names=['ρ', 'a', 'λ', 'ϕ'], 
                          textsize=8, 
                          grid=(1,5),
                          figsize=(10,3))
        
        if save_summary:
            az.summary(trace, 
                    var_names=['ρ', 'a', 'λ', 'ϕ']).to_csv('data/posterior')
    
    # transform back to weibull
    gaussian_samples = np.squeeze(trace.predictions.y_pred.values)
    samples = np.zeros_like(gaussian_samples)

    for n in range(gaussian_samples.shape[0]):
        samples[n] = weibull_min.ppf(norm.cdf(gaussian_samples[n]), *weibull_params) 

    return samples, trace
