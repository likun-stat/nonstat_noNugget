


###################################################################################
## Main sampler

## Must provide data input 'data_input.pkl' to initiate the sampler.
## In 'data_input.pkl', one must include 
##      Y ........................................... censored observations on GEV scale
##      cen ........................................................... indicator matrix
##      initial.values ........ a dictionary: phi, tau_sqd, prob_below, prob_above, Dist, 
##                                             theta_c, X, X_s, R, Design_mat, beta_loc0, 
##                                             beta_loc1, Time, beta_scale, beta_shape
##      n_updates .................................................... number of updates
##      thinning ......................................... number of runs in each update
##      experiment_name
##      echo_interval ......................... echo process every echo_interval updates
##      sigma_m
##      prop_Sigma
##      true_params ....................... a dictionary: phi, rho, tau_sqd, theta_gpd, 
##                                              prob_below, X_s, R
##

 
      

if __name__ == "__main__":
   import nonstat_noNugget.model_sim as utils
   import nonstat_noNugget.generic_samplers as sampler
   import nonstat_noNugget.priors as priors
   import nonstat_noNugget.ns_cov as cov
   import os
   import numpy as np
   import time
   import matplotlib.pyplot as plt
   from matplotlib.backends.backend_pdf import PdfPages
   from pickle import load
   from pickle import dump
   from scipy.linalg import lapack
   from scipy.linalg import cholesky
   
   # Check whether the 'mpi4py' is installed
   test_mpi = os.system("python -c 'from mpi4py import *' &> /dev/null")
   if test_mpi != 0:
      import sys
      sys.exit("mpi4py import is failing, aborting...")
   
   # get rank and size
   from mpi4py import MPI
  
   comm = MPI.COMM_WORLD
   rank = comm.Get_rank()
   size = comm.Get_size()
   thinning = 10; echo_interval = 20; n_updates = 50001
  
   # Load data input
   with open('data_sim1.pkl', 'rb') as f:
     Y = load(f)
     initial_values = load(f)
     sigma_m = load(f)
     prop_Sigma = load(f)
     f.close()
     
   # Filename for storing the intermediate results
   filename='./nonstat_progress_'+str(rank)+'.pkl'
   
   # Generate multiple independent random streams
   random_generator = np.random.RandomState()
  
   # Constants to control adaptation of the Metropolis sampler
   c_0 = 10
   c_1 = 0.8
   offset = 3  # the iteration offset
   r_opt_1d = .41
   r_opt_2d = .35
   eps = 1e-6 # a small number
  
   # Hyper parameters for the prior of the mixing distribution parameters and 
   hyper_params_phi = np.array([0.5,0.7])
   hyper_params_tau_sqd = np.array([0.1,0.1])
   hyper_params_theta_c = np.array([0, 20])
   hyper_params_theta_gev = 25
   # hyper_params_range = np.array([0.5,1.5]) # in case where roughness is not updated
    
   # Load initial values
   Knots =  initial_values['Knots']
   phi_range_weights = initial_values['phi_range_weights']
   R_weights =  initial_values['R_weights']
   radius =  initial_values['radius']
   Stations =  initial_values['Stations']
   phi_vec = initial_values['phi_vec']
   phi_at_knots = initial_values['phi_at_knots']
   gamma = initial_values['gamma']
   gamma_vec = initial_values['gamma_vec']
   range_vec = initial_values['range_vec']
   range_at_knots = initial_values['range_at_knots']
   nu = initial_values['nu']
   X = initial_values['X']
   R_at_knots = initial_values['R_at_knots']; Rt_at_knots = R_at_knots[:,rank]
   R_s = initial_values['R_s']; Rt_s = R_s[:,rank]
   Z = initial_values['Z']
   Design_mat = initial_values['Design_mat']
   beta_loc0 = initial_values['beta_loc0']
   beta_loc1 = initial_values['beta_loc1']
   Time = initial_values['Time']
   beta_scale = initial_values['beta_scale']
   beta_shape = initial_values['beta_shape']
   
   beta_gev_params = np.array([beta_loc0[0], beta_scale[0], beta_shape[0]])
   n_beta_gev_params = beta_gev_params.shape[0]
  
   # Bookkeeping
   n_s = Y.shape[0]
   n_t = Y.shape[1]
   n_phi_range_knots = len(phi_at_knots)
   n_Rt_knots = len(Rt_at_knots)
   if n_t != size:
      import sys
      sys.exit("Make sure the number of cpus (N) = number of time replicates (n_t), i.e.\n     srun -N python nonstat_sampler.py")
   n_covariates = len(beta_loc0)
   
   n_updates_thinned = np.int(np.ceil(n_updates/thinning))
   wh_to_plot_Xs = n_s*np.array([0.25,0.5,0.75])
   wh_to_plot_Xs = wh_to_plot_Xs.astype(int)

   # Eigendecomposition of the correlation matrix
   one_vec = np.ones(n_s)
   Cor = cov.ns_cov(range_vec, one_vec, Stations, kappa = nu, cov_model = "matern")
   eig_Cor = np.linalg.eigh(Cor) #For symmetric matrices
   V = eig_Cor[1]
   d = eig_Cor[0]
   # cholesky_inv = lapack.dposv(Cor,one_vec)

   # Marginal GEV parameters: per location x time
   loc0 = Design_mat @beta_loc0
   loc0 = loc0.astype('float64')
   loc1 = Design_mat @beta_loc1
   loc1 = loc1.astype('float64')
   Loc = np.tile(loc0, n_t) + np.tile(loc1, n_t)*np.repeat(Time,n_s)
   Loc = Loc.reshape((n_s,n_t),order='F')

   scale = Design_mat @beta_scale
   scale = scale.astype('float64')
   Scale = np.tile(scale, n_t)
   Scale = Scale.reshape((n_s,n_t),order='F')
   
   shape = Design_mat @beta_shape
   shape = shape.astype('float64')
   Shape = np.tile(shape, n_t)
   Shape = Shape.reshape((n_s,n_t),order='F')
   
   Current_Rt_prior = np.sum(utils.dlevy(Rt_at_knots, m=0, s=gamma, log=True))
   Current_lik = utils.marg_transform_data_mixture_likelihood_1t(Y[:,rank], X[:,rank], Loc[:,rank], Scale[:,rank], 
                                             Shape[:,rank], phi_vec, gamma_vec, R_s[:,rank], 
                                             V, d)
   Current_Lik_recv = comm.gather(Current_lik,root=0)
   
   
   accept = 0
   # --------- Update Rt -----------
   #Propose new values
   if rank==20: print(rank, Rt_at_knots)
   Rt_s_star = np.empty(Rt_s.shape)
   
   #Propose Rt under every worker
   if rank==20: print('Current_lik:',Current_lik, "- rank:", rank)
   start_time=time.time()
   tmp_upper = cholesky(prop_Sigma['Rt'],lower=False)
   tmp_params_star = sigma_m['Rt']*random_generator.standard_normal(n_Rt_knots)
   Rt_at_knots_star = Rt_at_knots + np.matmul(tmp_upper.T , tmp_params_star)
   Rt_s_star[:] = R_weights @ Rt_at_knots_star 
   if rank==20: print(rank, Rt_at_knots_star)    
   
   # Evaluate likelihood at new values
   # Not broadcasting but evaluating at each node 
   if np.any(Rt_at_knots_star<0):
       Star_Rt_prior = -np.inf
       Star_lik = -np.inf
   else:    
       Star_Rt_prior = np.sum(utils.dlevy(Rt_at_knots_star, m=0, s=gamma, log=True))
       Star_lik = utils.marg_transform_data_mixture_likelihood_1t(Y[:,rank], X[:,rank], Loc[:,rank], Scale[:,rank], 
                                             Shape[:,rank], phi_vec, gamma_vec, Rt_s_star, 
                                             V, d)
   
   # Determine update or not
   # Not gathering but evaluating at each node 
   if rank==20: print('Star_Lik:',Star_lik)
   r = np.exp(Star_Rt_prior + Star_lik - Current_Rt_prior - Current_lik)
   if rank==20: print('r=',r, "- rank:", rank)
   if ~np.isfinite(r):
       r = 0
   if random_generator.uniform(0,1,1)<r:
       Rt_at_knots[:] = Rt_at_knots_star
       Rt_s[:] = Rt_s_star 
       Current_lik = Star_lik
       accept = 1    
   if rank==20: print('Current_lik:',Current_lik, "- rank:", rank, 'after')
   
   # Gather anyways
   Current_Lik_recv = comm.gather(Current_lik,root=0)
   R_s_recv = comm.gather(Rt_s,root=0)
   if rank ==0: R_s[:] = np.vstack(R_s_recv).T
   
   if rank==0: print(rank,np.sum(Current_Lik_recv))  
       
       
   if rank==0: 
       time_spent = time.time()-start_time
       print(str(time_spent)+'\n')
       start_time=time.time()
   
   
   accept = 0
   # --------- Update phi_vec -----------
   #Propose new values
   phi_vec_star = np.empty(n_s)
   if rank==0:
       tmp_upper = cholesky(prop_Sigma['phi'],lower=False)
       tmp_params_star = sigma_m['phi']*random_generator.standard_normal(n_phi_range_knots)
       phi_at_knots_proposal = phi_at_knots + np.matmul(tmp_upper.T , tmp_params_star)
       phi_vec_star[:] = phi_range_weights @ phi_at_knots_proposal    
   phi_vec_star = comm.bcast(phi_vec_star,root=0)
   
   # Evaluate likelihood at new values
   if np.any(phi_vec_star>=1) or np.any(phi_vec_star<=0): #U(0,1) priors
       Star_lik = -np.inf
   else: 
       Star_lik = utils.marg_transform_data_mixture_likelihood_1t(Y[:,rank], X[:,rank], Loc[:,rank], Scale[:,rank], 
                                             Shape[:,rank], phi_vec_star, gamma_vec, R_s[:,rank], 
                                             V, d)
   Star_Lik_recv = comm.gather(Star_lik,root=0)
   
   # Determine update or not
   if rank==0:
       log_num = np.sum(Star_Lik_recv)
       log_denom = np.sum(Current_Lik_recv)
       r = np.exp(log_num - log_denom)
       if ~np.isfinite(r):
           r = 0
       if random_generator.uniform(0,1,1)<r:
           phi_at_knots[:] = phi_at_knots_proposal
           phi_vec[:] = phi_vec_star 
           Current_Lik_recv[:] = Star_Lik_recv
           accept = 1
           
   # Broadcast anyways
   phi_vec = comm.bcast(phi_vec,root=0)
    
   
   accept = 0
   # --------- Update range_vec -----------
   #Propose new values
   range_vec_star = np.empty(n_s)
   V_star = np.empty(V.shape)
   d_star = np.empty(d.shape)
   if rank==0:
       start_time=time.time()
       tmp_upper = cholesky(prop_Sigma['range'],lower=False)
       tmp_params_star = sigma_m['range']*random_generator.standard_normal(n_phi_range_knots)
       range_at_knots_proposal = range_at_knots + np.matmul(tmp_upper.T , tmp_params_star)
       range_vec_star[:] = phi_range_weights @ range_at_knots_proposal
   range_vec_star = comm.bcast(range_vec_star,root=0)
   
   # Evaluate likelihood at new values
   if np.all(range_vec_star>0):
       # Not broadcasting but generating at each node
       Cor_star = cov.ns_cov(range_vec_star, one_vec, Stations, kappa = nu, cov_model = "matern")
       eig_Cor = np.linalg.eigh(Cor_star) #For symmetric matrices
       V_star[:] = eig_Cor[1]
       d_star[:] = eig_Cor[0]    
       Star_lik = utils.marg_transform_data_mixture_likelihood_1t(Y[:,rank], X[:,rank], Loc[:,rank], Scale[:,rank], 
                                             Shape[:,rank], phi_vec, gamma_vec, R_s[:,rank], 
                                             V_star, d_star)
   else:
       Star_lik = -np.inf
   
   Star_Lik_recv = comm.gather(Star_lik,root=0)
   
   # Determine update or not
   if rank==0:
       log_num = np.sum(Star_Lik_recv)
       log_denom = np.sum(Current_Lik_recv)
       r = np.exp(log_num - log_denom)
       if ~np.isfinite(r):
           r = 0
       if random_generator.uniform(0,1,1)<r:
           range_at_knots[:] = range_at_knots_proposal
           range_vec[:] = range_vec_star 
           Current_Lik_recv[:] = Star_Lik_recv
           accept = 1
   
   # Broadcast anyways
   accept = comm.bcast(accept,root=0)
   if accept==1:
       V[:] = V_star
       d[:] = d_star
     
       
   accept = 0
   # --------- Update GEV params -----------
   #Propose new values
   print(rank, beta_gev_params, loc0[0], loc1[0], scale[0], shape[0])
   beta_gev_params_star = np.empty(beta_gev_params.shape)
   if rank==0:
       print('Current_Lik_recv sum:',np.sum(Current_Lik_recv))
       start_time=time.time()
       tmp_upper = cholesky(prop_Sigma['gev_params'],lower=False)
       tmp_params_star = sigma_m['gev_params']*random_generator.standard_normal(n_beta_gev_params)
       beta_gev_params_star[:] = beta_gev_params + np.matmul(tmp_upper.T , tmp_params_star)
       
   beta_gev_params_star = comm.bcast(beta_gev_params_star,root=0)
   
   # Evaluate likelihood at new values
   # Not broadcasting but generating at each node
   loc0_star = Design_mat @np.array([beta_gev_params_star[0],0])
   scale_star = Design_mat @np.array([beta_gev_params_star[1],0])
   shape_star = Design_mat @np.array([beta_gev_params_star[2],0])
   print(rank,beta_gev_params_star, loc0_star[0], loc1[0], scale_star[0], shape_star[0], "star")
   Loc_star = np.tile(loc0_star, n_t) + np.tile(loc1, n_t)*np.repeat(Time,n_s)
   Loc_star = Loc_star.reshape((n_s,n_t),order='F')
   Scale_star = np.tile(scale_star, n_t)
   Scale_star = Scale_star.reshape((n_s,n_t),order='F')
   Shape_star = np.tile(shape_star, n_t)
   Shape_star = Shape_star.reshape((n_s,n_t),order='F')
   
   Star_lik = utils.marg_transform_data_mixture_likelihood_1t(Y[:,rank], X[:,rank], Loc_star[:,rank], Scale_star[:,rank], 
                                             Shape_star[:,rank], phi_vec, gamma_vec, R_s[:,rank], 
                                             V, d)
   Star_Lik_recv = comm.gather(Star_lik,root=0)
   
   # Determine update or not
   if rank==0:
       print('Star_Lik_recv sum:',np.sum(Star_Lik_recv))
       log_num = np.sum(Star_Lik_recv)
       log_denom = np.sum(Current_Lik_recv)
       r = np.exp(log_num - log_denom)
       if ~np.isfinite(r):
           r = 0
       if random_generator.uniform(0,1,1)<r:
           beta_gev_params[:] = beta_gev_params_star
           range_vec[:] = range_vec_star 
           Current_Lik_recv[:] = Star_Lik_recv
           accept = 1    
   
   # Broadcast anyways
   accept = comm.bcast(accept,root=0)
   print("accept = ",accept, rank)
   if accept==1:
       loc0[:] = loc0_star
       scale[:] = scale_star
       shape[:] = shape_star
       Loc[:] = Loc_star
       Scale[:] = Scale_star
       Shape[:] = Shape_star
   print(rank, loc0[0], loc1[0], scale[0], shape[0])  
       
       
   # if rank==0: 
   #     time_spent = time.time()-start_time
   #     print(str(time_spent)+'\n')
   #     start_time=time.time()
   # Lik = utils.marg_transform_data_mixture_likelihood_1t(Y[:,rank], X[:,rank], Loc[:,rank], Scale[:,rank], 
   #                                           Shape[:,rank], phi_vec, gamma_vec, R_s[:,rank], 
   #                                           V, d)
   
   # if rank==0: 
   #     time_spent = time.time()-start_time
   #     print(str(time_spent)+'\n')
       
       
       
   # if rank==0: start_time=time.time()
   # Lik_tol = utils.marg_transform_data_mixture_likelihood(Y, X, Loc, Scale, 
   #                                           Shape, phi_vec, gamma_vec, R_s, 
   #                                           V, d)
   # if rank==0: 
   #     time_spent = time.time()-start_time
   #     print(str(time_spent)+'\n')
   #     print(str(Lik_tol)+'\n')
   
   
   
   # Z_1t_accept = np.zeros(n_s)
   # R_accept = 0
   # Z_1t_trace = np.empty((n_s,n_updates_thinned)); Z_1t_trace[:] = np.nan
   # Z_1t_trace[:,0] = Z_onetime  
   # R_1t_trace = np.empty(n_updates_thinned); R_1t_trace[:] = np.nan
   # R_1t_trace[0] = R_onetime
   # if rank == 0:
   #   print("Number of time replicates = %d"%size)
   #   phi_trace = np.empty(n_updates_thinned); phi_trace[:] = np.nan
   #   phi_trace[0] = phi
   #   tau_sqd_trace = np.empty(n_updates_thinned); tau_sqd_trace[:] = np.nan
   #   tau_sqd_trace[0] = tau_sqd
   #   theta_c_trace_within_thinning = np.empty((2,thinning)); theta_c_trace_within_thinning[:] = np.nan
   #   theta_c_trace = np.empty((2,n_updates_thinned)); theta_c_trace[:] = np.nan
   #   theta_c_trace[:,0] = theta_c
   #   beta_loc0_trace_within_thinning = np.empty((n_covariates,thinning)); beta_loc0_trace_within_thinning[:] = np.nan
   #   beta_loc0_trace = np.empty((n_covariates,n_updates_thinned)); beta_loc0_trace[:] = np.nan
   #   beta_loc0_trace[:,0] = beta_loc0
   #   beta_loc1_trace_within_thinning = np.empty((n_covariates,thinning)); beta_loc1_trace_within_thinning[:] = np.nan
   #   beta_loc1_trace = np.empty((n_covariates,n_updates_thinned)); beta_loc1_trace[:] = np.nan
   #   beta_loc1_trace[:,0] = beta_loc1
   #   beta_scale_trace_within_thinning = np.empty((n_covariates,thinning)); beta_scale_trace_within_thinning[:] = np.nan
   #   beta_scale_trace = np.empty((n_covariates,n_updates_thinned)); beta_scale_trace[:] = np.nan
   #   beta_scale_trace[:,0] = beta_scale
   #   beta_shape_trace_within_thinning = np.empty((n_covariates,thinning)); beta_shape_trace_within_thinning[:] = np.nan
   #   beta_shape_trace = np.empty((n_covariates,n_updates_thinned)); beta_shape_trace[:] = np.nan
   #   beta_shape_trace[:,0] = beta_shape
    
   #   phi_accept = 0
   #   tau_sqd_accept = 0
   #   theta_c_accept = 0
   #   beta_loc0_accept = 0
   #   beta_loc1_accept = 0
   #   beta_scale_accept = 0
   #   beta_shape_accept = 0
    
   # -----------------------------------------------------------------------------------
   # -----------------------------------------------------------------------------------
   # --------------------------- Start Metropolis Updates ------------------------------
   # -----------------------------------------------------------------------------------
   # -----------------------------------------------------------------------------------
   # for iter in np.arange(1,n_updates):
   #     # Update X
   #     # print(str(rank)+" "+str(iter)+" Gathered? "+str(np.where(~cen)))
   #     X_onetime = utils.X_update(Y_onetime, cen[:,rank], cen_above[:,rank], xp, surv_p, tau_sqd, phi, gamma, Loc[:,rank], Scale[:,rank], Shape[:,rank])
      
   #     # Update Z
   #     tmp = utils.Z_update_onetime(Y_onetime, X_onetime, R_onetime, Z_onetime, cen[:,rank], cen_above[:,rank], prob_below, prob_above,
   #                                  tau_sqd, phi, gamma, Loc[:,rank], Scale[:,rank], Shape[:,rank], xp, surv_p, den_p,
   #                                  thresh_X, thresh_X_above, Cor, cholesky_inv, sigma_m['Z_onetime'], random_generator)
   #     Z_1t_accept = Z_1t_accept + tmp
      
   #     # Update R
   #     Metr_R = sampler.static_metr(Y_onetime, R_onetime, utils.Rt_update_mixture_me_likelihood, 
   #                         priors.R_prior, gamma, 2, 
   #                         random_generator,
   #                         np.nan, sigma_m['R_1t'], False, 
   #                         X_onetime, Z_onetime, cen[:,rank], cen_above[:,rank], 
   #                         prob_below, prob_above, Loc[:,rank], Scale[:,rank], Shape[:,rank], tau_sqd, phi, gamma,
   #                         xp, surv_p, den_p, thresh_X, thresh_X_above)
   #     R_accept = R_accept + Metr_R['acc_prob']
   #     R_onetime = Metr_R['trace'][0,1]
       
   #     X_s_onetime = (R_onetime**phi)*utils.norm_to_Pareto(Z_onetime)

   #     # *** Gather items ***
   #     X_s_recv = comm.gather(X_s_onetime,root=0)
   #     X_recv = comm.gather(X_onetime, root=0)
   #     Z_recv = comm.gather(Z_onetime, root=0)
   #     R_recv = comm.gather(R_onetime, root=0)

   #     if rank==0:
   #         X_s[:] = np.vstack(X_s_recv).T
   #         X[:] = np.vstack(X_recv).T
   #         # Check whether X is negative
   #         if np.any(X[~cen & ~cen_above]<0):
   #             sys.exit("X value abnormalty "+str(phi)+" "+str(tau_sqd))
               
   #         Z[:] = np.vstack(Z_recv).T
   #         R[:] = R_recv
   #         index_within = (iter-1)%thinning
   #         # print('beta_shape_accept=',beta_shape_accept, ', iter=', iter)

   #         # Update phi
   #         Metr_phi = sampler.static_metr(Y, phi, utils.phi_update_mixture_me_likelihood, priors.interval_unif, 
   #                 hyper_params_phi, 2, 
   #                 random_generator,
   #                 np.nan, sigma_m['phi'], False, 
   #                 R, Z, cen, cen_above, 
   #                 prob_below, prob_above, Loc, Scale, Shape, tau_sqd, gamma)
   #         phi_accept = phi_accept + Metr_phi['acc_prob']
   #         phi = Metr_phi['trace'][0,1]
           
   #         # Update gamma (TBD)
   #         #
           
   #         grid = utils.density_interp_grid(phi, gamma, grid_size=800)
   #         xp = grid[0]; den_p = grid[1]; surv_p = grid[2]
   #         X_s = (R**phi)*utils.norm_to_Pareto(Z)
           
   #         # Update tau_sqd
   #         Metr_tau_sqd = sampler.static_metr(Y, tau_sqd, utils.tau_update_mixture_me_likelihood, priors.invGamma_prior, 
   #                      hyper_params_tau_sqd, 2, 
   #                      random_generator,
   #                      np.nan, sigma_m['tau_sqd'], False,
   #                      X_s, cen, cen_above, 
   #                      prob_below, prob_above, Loc, Scale, Shape, 
   #                      phi, gamma, xp, surv_p, den_p)
   #         tau_sqd_accept = tau_sqd_accept + Metr_tau_sqd['acc_prob']
   #         tau_sqd = Metr_tau_sqd['trace'][0,1]
          
   #         thresh_X =  utils.qRW_me_interp(prob_below, xp, surv_p, tau_sqd, phi, gamma)
   #         thresh_X_above =  utils.qRW_me_interp(prob_above, xp, surv_p, tau_sqd, phi, gamma)

           
   #         # Update theta_c
   #         Metr_theta_c = sampler.static_metr(Z, theta_c, utils.theta_c_update_mixture_me_likelihood, 
   #                           priors.interval_unif_multi, hyper_params_theta_c, 2,
   #                           random_generator,
   #                           prop_sigma['theta_c'], sigma_m['theta_c'], False,
   #                           Dist)
   #         theta_c_accept = theta_c_accept + Metr_theta_c['acc_prob']
   #         theta_c = Metr_theta_c['trace'][:,1]
   #         theta_c_trace_within_thinning[:,index_within] = theta_c
          
   #         if Metr_theta_c['acc_prob']>0:
   #             Cor = utils.corr_fn(Dist, theta_c)
   #             # eig_Cor = np.linalg.eigh(Cor) #For symmetric matrices
   #             # V = eig_Cor[1]
   #             # d = eig_Cor[0]
   #             cholesky_inv = lapack.dposv(Cor,one_vec)
           
   #         # Update beta_loc0
   #         Metr_beta_loc0 = sampler.static_metr(Design_mat, beta_loc0, utils.loc0_gev_update_mixture_me_likelihood, 
   #                            priors.unif_prior, hyper_params_theta_gev, 2,
   #                            random_generator,
   #                            prop_sigma['beta_loc0'], sigma_m['beta_loc0'], False, 
   #                            Y, X_s, cen, cen_above, prob_below, prob_above, 
   #                            tau_sqd, phi, gamma, loc1, Scale, Shape, Time, xp, surv_p, den_p, 
   #                            thresh_X, thresh_X_above)
   #         beta_loc0_accept = beta_loc0_accept + Metr_beta_loc0['acc_prob']
   #         beta_loc0 = Metr_beta_loc0['trace'][:,1]
   #         beta_loc0_trace_within_thinning[:,index_within] = beta_loc0
   #         loc0 = Design_mat @beta_loc0
          
   #         # Update beta_loc1
   #         Metr_beta_loc1 = sampler.static_metr(Design_mat, beta_loc1, utils.loc1_gev_update_mixture_me_likelihood, 
   #                            priors.unif_prior, hyper_params_theta_gev, 2,
   #                            random_generator,
   #                            prop_sigma['beta_loc1'], sigma_m['beta_loc1'], False, 
   #                            Y, X_s, cen, cen_above, prob_below, prob_above, 
   #                            tau_sqd, phi, gamma, loc0, Scale, Shape, Time, xp, surv_p, den_p, 
   #                            thresh_X, thresh_X_above)
   #         beta_loc1_accept = beta_loc1_accept + Metr_beta_loc1['acc_prob']
   #         beta_loc1 = Metr_beta_loc1['trace'][:,1]
   #         beta_loc1_trace_within_thinning[:,index_within] = beta_loc1
   #         loc1 = Design_mat @beta_loc1
   #         Loc = np.tile(loc0, n_t) + np.tile(loc1, n_t)*np.repeat(Time,n_s)
   #         Loc = Loc.reshape((n_s,n_t),order='F')
           
   #         # Update beta_scale
   #         Metr_beta_scale = sampler.static_metr(Design_mat, beta_scale, utils.scale_gev_update_mixture_me_likelihood, 
   #                            priors.unif_prior, hyper_params_theta_gev, 2,
   #                            random_generator,
   #                            prop_sigma['beta_scale'], sigma_m['beta_scale'], False,
   #                            Y, X_s, cen, cen_above, prob_below, prob_above, 
   #                            tau_sqd, phi, gamma, Loc, Shape, Time, xp, surv_p, den_p, 
   #                            thresh_X, thresh_X_above)
   #         beta_scale_accept = beta_scale_accept + Metr_beta_scale['acc_prob']
   #         beta_scale = Metr_beta_scale['trace'][:,1]
   #         beta_scale_trace_within_thinning[:,index_within] = beta_scale
   #         scale = Design_mat @beta_scale
   #         Scale = np.tile(scale, n_t)
   #         Scale = Scale.reshape((n_s,n_t),order='F')
          
   #         # # Update beta_shape
   #         # Metr_beta_shape = sampler.static_metr(Design_mat, beta_shape, utils.shape_gev_update_mixture_me_likelihood, 
   #         #                    priors.unif_prior, hyper_params_theta_gev, 2, 
   #         #                    random_generator,
   #         #                    prop_sigma['beta_shape'], sigma_m['beta_shape'], False,
   #         #                    Y, X_s, cen, cen_above, prob_below, prob_above,
   #         #                    tau_sqd, phi, gamma, Loc, Scale, Time, xp, surv_p, den_p, 
   #         #                    thresh_X, thresh_X_above)
   #         # beta_shape_accept = beta_shape_accept + Metr_beta_shape['acc_prob']
   #         # beta_shape = Metr_beta_shape['trace'][:,1]
   #         # beta_shape_trace_within_thinning[:,index_within] = beta_shape
   #         # shape = Design_mat1 @beta_shape
   #         # Shape = np.tile(shape, n_t)
   #         # Shape = Shape.reshape((n_s,n_t),order='F')
          
   #         # cen[:] = utils.which_censored(Y, Loc, Scale, Shape, prob_below)
   #         # cen_above[:] = ~utils.which_censored(Y, Loc, Scale, Shape, prob_above)
           
           
   #     # *** Broadcast items ***
   #     phi = comm.bcast(phi,root=0)
   #     xp = comm.bcast(xp,root=0)
   #     den_p = comm.bcast(den_p,root=0)
   #     surv_p = comm.bcast(surv_p,root=0)
   #     tau_sqd = comm.bcast(tau_sqd,root=0)
   #     thresh_X = comm.bcast(thresh_X,root=0)
   #     thresh_X_above = comm.bcast(thresh_X_above,root=0)
   #     theta_c = comm.bcast(theta_c,root=0)
   #     # V = comm.bcast(V,root=0)
   #     # d = comm.bcast(d,root=0)
   #     Cor = comm.bcast(Cor,root=0)
   #     cholesky_inv = comm.bcast(cholesky_inv,root=0)
   #     Loc = comm.bcast(Loc,root=0)
   #     Scale = comm.bcast(Scale,root=0)
   #     Shape = comm.bcast(Shape,root=0)
   #     # cen = comm.bcast(cen,root=0)
   #     # cen_above = comm.bcast(cen_above,root=0)
      
   #     # ----------------------------------------------------------------------------------------
   #     # --------------------------- Summarize every 'thinning' steps ---------------------------
   #     # ----------------------------------------------------------------------------------------
   #     if (iter % thinning) == 0:
   #         index = np.int(iter/thinning)
           
   #         # Fill in trace objects
   #         Z_1t_trace[:,index] = Z_onetime  
   #         R_1t_trace[index] = R_onetime
   #         if rank == 0:
   #             phi_trace[index] = phi
   #             tau_sqd_trace[index] = tau_sqd
   #             theta_c_trace[:,index] = theta_c
   #             beta_loc0_trace[:,index] = beta_loc0
   #             beta_loc1_trace[:,index] = beta_loc1
   #             beta_scale_trace[:,index] = beta_scale
   #             beta_shape_trace[:,index] = beta_shape
          
            
   #         # Adapt via Shaby and Wells (2010)
   #         gamma2 = 1 / (index + offset)**(c_1)
   #         gamma1 = c_0*gamma2
   #         sigma_m['Z_onetime'] = np.exp(np.log(sigma_m['Z_onetime']) + gamma1*(Z_1t_accept/thinning - r_opt_1d))
   #         Z_1t_accept[:] = 0
   #         sigma_m['R_1t'] = np.exp(np.log(sigma_m['R_1t']) + gamma1*(R_accept/thinning - r_opt_1d))
   #         R_accept = 0
          
   #         if rank == 0:
   #             sigma_m['phi'] = np.exp(np.log(sigma_m['phi']) + gamma1*(phi_accept/thinning - r_opt_1d))
   #             phi_accept = 0
   #             sigma_m['tau_sqd'] = np.exp(np.log(sigma_m['tau_sqd']) + gamma1*(tau_sqd_accept/thinning - r_opt_1d))
   #             tau_sqd_accept = 0
          
   #             sigma_m['theta_c'] = np.exp(np.log(sigma_m['theta_c']) + gamma1*(theta_c_accept/thinning - r_opt_2d))
   #             theta_c_accept = 0
   #             prop_sigma['theta_c'] = prop_sigma['theta_c'] + gamma2*(np.cov(theta_c_trace_within_thinning) - prop_sigma['theta_c'])
   #             check_chol_cont = True
   #             while check_chol_cont:
   #                 try:
   #                     # Initialize prop_C
   #                     np.linalg.cholesky(prop_sigma['theta_c'])
   #                     check_chol_cont = False
   #                 except  np.linalg.LinAlgError:
   #                     prop_sigma['theta_c'] = prop_sigma['theta_c'] + eps*np.eye(2)
   #                     print("Oops. Proposal covariance matrix is now:\n")
   #                     print(prop_sigma['theta_c'])
                   
                   
   #             sigma_m['beta_loc0'] = np.exp(np.log(sigma_m['beta_loc0']) + gamma1*(beta_loc0_accept/thinning - r_opt_2d))
   #             beta_loc0_accept = 0
   #             prop_sigma['beta_loc0'] = prop_sigma['beta_loc0'] + gamma2*(np.cov(beta_loc0_trace_within_thinning) - prop_sigma['beta_loc0'])
   #             check_chol_cont = True
   #             while check_chol_cont:
   #                 try:
   #                     # Initialize prop_C
   #                     np.linalg.cholesky(prop_sigma['beta_loc0'])
   #                     check_chol_cont = False
   #                 except  np.linalg.LinAlgError:
   #                     prop_sigma['beta_loc0'] = prop_sigma['beta_loc0'] + eps*np.eye(n_covariates)
   #                     print("Oops. Proposal covariance matrix is now:\n")
   #                     print(prop_sigma['beta_loc0'])
            
   #             sigma_m['beta_loc1'] = np.exp(np.log(sigma_m['beta_loc1']) + gamma1*(beta_loc1_accept/thinning - r_opt_2d))
   #             beta_loc1_accept = 0
   #             prop_sigma['beta_loc1'] = prop_sigma['beta_loc1'] + gamma2*(np.cov(beta_loc1_trace_within_thinning) - prop_sigma['beta_loc1'])
   #             check_chol_cont = True
   #             while check_chol_cont:
   #                 try:
   #                     # Initialize prop_C
   #                     np.linalg.cholesky(prop_sigma['beta_loc1'])
   #                     check_chol_cont = False
   #                 except  np.linalg.LinAlgError:
   #                     prop_sigma['beta_loc1'] = prop_sigma['beta_loc1'] + eps*np.eye(n_covariates)
   #                     print("Oops. Proposal covariance matrix is now:\n")
   #                     print(prop_sigma['beta_loc1'])
                  
   #             sigma_m['beta_scale'] = np.exp(np.log(sigma_m['beta_scale']) + gamma1*(beta_scale_accept/thinning - r_opt_2d))
   #             beta_scale_accept = 0
   #             prop_sigma['beta_scale'] = prop_sigma['beta_scale'] + gamma2*(np.cov(beta_scale_trace_within_thinning) - prop_sigma['beta_scale'])
   #             check_chol_cont = True
   #             while check_chol_cont:
   #                 try:
   #                     # Initialize prop_C
   #                     np.linalg.cholesky(prop_sigma['beta_scale'])
   #                     check_chol_cont = False
   #                 except  np.linalg.LinAlgError:
   #                     prop_sigma['beta_scale'] = prop_sigma['beta_scale'] + eps*np.eye(n_covariates)
   #                     print("Oops. Proposal covariance matrix is now:\n")
   #                     print(prop_sigma['beta_scale'])
                 
   #             # sigma_m['beta_shape'] = np.exp(np.log(sigma_m['beta_shape']) + gamma1*(beta_shape_accept/thinning - r_opt_2d))
   #             # beta_shape_accept = 0
   #             # prop_sigma['beta_shape'] = prop_sigma['beta_shape'] + gamma2*(np.cov(beta_shape_trace_within_thinning) - prop_sigma['beta_shape'])
   #             # check_chol_cont = True
   #             # while check_chol_cont:
   #             #     try:
   #             #         # Initialize prop_C
   #             #         np.linalg.cholesky(prop_sigma['beta_shape'])
   #             #         check_chol_cont = False
   #             #     except  np.linalg.LinAlgError:
   #             #         prop_sigma['beta_shape'] = prop_sigma['beta_shape'] + eps*np.eye(n_covariates)
   #             #         print("Oops. Proposal covariance matrix is now:\n")
   #             #         print(prop_sigma['beta_shape'])
          
   #     # ----------------------------------------------------------------------------------------                
   #     # -------------------------- Echo & save every 'thinning' steps --------------------------
   #     # ----------------------------------------------------------------------------------------
   #     if (iter / thinning) % echo_interval == 0:
   #         print(rank, iter, phi, tau_sqd)
   #         if rank == 0:
   #             print('Done with '+str(index)+" updates while thinned by "+str(thinning)+" steps,\n")
               
   #             # Save the intermediate results to filename
   #             initial_values = {'phi':phi,
   #                  'gamma':gamma,
   #                  'tau_sqd':tau_sqd,
   #                  'prob_below':prob_below,
   #                  'prob_above':prob_above,
   #                  'Dist':Dist,
   #                  'theta_c':theta_c,
   #                  'X':X,
   #                  'X_s':X_s,
   #                  'R':R,
   #                  'Z':Z,
   #                  'Design_mat':Design_mat,
   #                  'beta_loc0':beta_loc0,
   #                  'beta_loc1':beta_loc1,
   #                  'Time':Time,
   #                  'beta_scale':beta_scale,
   #                  'beta_shape':beta_shape,
   #                  }
   #             with open(filename, 'wb') as f:
   #                 dump(Y, f)
   #                 dump(cen, f)
   #                 dump(cen_above,f)
   #                 dump(initial_values, f)
   #                 dump(sigma_m, f)
   #                 dump(prop_sigma, f)
   #                 dump(iter, f)
   #                 dump(phi_trace, f)
   #                 dump(tau_sqd_trace, f)
   #                 dump(theta_c_trace, f)
   #                 dump(beta_loc0_trace, f)
   #                 dump(beta_loc1_trace, f)
   #                 dump(beta_scale_trace, f)
   #                 dump(beta_shape_trace, f)
                   
   #                 dump(Z_1t_trace, f)
   #                 dump(R_1t_trace, f)
   #                 dump(Y_onetime, f)
   #                 dump(X_onetime, f)
   #                 dump(X_s_onetime, f)
   #                 dump(R_onetime, f)
   #                 dump(Z_onetime, f)
   #                 f.close()
                   
   #             # Echo trace plots
   #             pdf_pages = PdfPages('./progress.pdf')
   #             grid_size = (4,2)
   #             #-page-1
   #             fig = plt.figure(figsize = (8.75, 11.75))
   #             plt.subplot2grid(grid_size, (0,0)) # phi
   #             plt.plot(phi_trace, color='gray', linestyle='solid')
   #             plt.ylabel(r'$\phi$')
   #             plt.subplot2grid(grid_size, (0,1)) # tau_sqd
   #             plt.plot(tau_sqd_trace, color='gray', linestyle='solid')
   #             plt.ylabel(r'$\tau^2$')
   #             plt.subplot2grid(grid_size, (1,0)) # rho
   #             plt.plot(theta_c_trace[0,:], color='gray', linestyle='solid')
   #             plt.ylabel(r'Matern $\rho$')
   #             plt.subplot2grid(grid_size, (1,1)) # nu
   #             plt.plot(theta_c_trace[1,:], color='gray', linestyle='solid')
   #             plt.ylabel(r'Matern $\nu$')
   #             plt.subplot2grid(grid_size, (2,0)) # mu0: beta_0
   #             plt.plot(beta_loc0_trace[0,:], color='gray', linestyle='solid')
   #             plt.ylabel(r'Location $\mu_0$: $\beta_0$')
   #             plt.subplot2grid(grid_size, (2,1)) # mu0: beta_1
   #             plt.plot(beta_loc0_trace[1,:], color='gray', linestyle='solid')
   #             plt.ylabel(r'Location $\mu_0$: $\beta_1$') 
   #             plt.subplot2grid(grid_size, (3,0)) # mu1: beta_0
   #             plt.plot(beta_loc1_trace[0,:], color='gray', linestyle='solid')
   #             plt.ylabel(r'Location $\mu_1$: $\beta_0$')
   #             plt.subplot2grid(grid_size, (3,1)) # mu1: beta_1
   #             plt.plot(beta_loc1_trace[1,:], color='gray', linestyle='solid')
   #             plt.ylabel(r'Location $\mu_1$: $\beta_1$')
   #             plt.tight_layout()
   #             pdf_pages.savefig(fig)
   #             plt.close()
                   
   #             #-page-2
   #             fig = plt.figure(figsize = (8.75, 11.75))
   #             plt.subplot2grid(grid_size, (0,0)) # scale: beta_0
   #             plt.plot(beta_scale_trace[0,:], color='gray', linestyle='solid')
   #             plt.ylabel(r'Scale $\sigma$: $\beta_0$')
   #             plt.subplot2grid(grid_size, (0,1)) # scale: beta_1
   #             plt.plot(beta_scale_trace[1,:], color='gray', linestyle='solid')
   #             plt.ylabel(r'Scale $\sigma$: $\beta_1$')
   #             plt.subplot2grid(grid_size, (1,0))  # shape: beta_0
   #             plt.plot(beta_shape_trace[0,:], color='gray', linestyle='solid')
   #             plt.ylabel(r'Shape $\xi$: $\beta_0$')
   #             plt.subplot2grid(grid_size, (1,1))  # shape: beta_1
   #             plt.plot(beta_shape_trace[1,:], color='gray', linestyle='solid')
   #             plt.ylabel(r'Shape $\xi$: $\beta_1$')
   #             plt.subplot2grid(grid_size, (2,0))   # X^*
   #             plt.plot(Z_1t_trace[1,:], color='gray', linestyle='solid')
   #             plt.ylabel(r'$Z$'+'['+str(1)+","+str(rank)+']')
   #             where = [(2,1),(3,0),(3,1)]
   #             for wh_sub,i in enumerate(wh_to_plot_Xs):
   #                 plt.subplot2grid(grid_size, where[wh_sub]) # X^*
   #                 plt.plot(Z_1t_trace[i,:], color='gray', linestyle='solid')
   #                 plt.ylabel(r'$Z$'+'['+str(i)+","+str(rank)+']')        
   #             plt.tight_layout()
   #             pdf_pages.savefig(fig)
   #             plt.close()
   #             pdf_pages.close()                 
   #         else:
   #             with open(filename, 'wb') as f:
   #                 dump(Y, f)
   #                 dump(cen, f)
   #                 dump(cen_above,f)
   #                 dump(initial_values, f)
   #                 dump(sigma_m, f)
   #                 dump(iter, f)
   #                 dump(Z_1t_trace, f)
   #                 dump(R_1t_trace, f)
   #                 dump(Y_onetime, f)
   #                 dump(X_onetime, f)
   #                 dump(X_s_onetime, f)
   #                 dump(R_onetime, f)
   #                 dump(Z_onetime, f)
   #                 f.close()
               
