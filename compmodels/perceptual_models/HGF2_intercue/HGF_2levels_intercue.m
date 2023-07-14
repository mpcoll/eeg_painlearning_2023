function [traj, infStates] = hgf_intercue(r, p, varargin)
% Calculates the trajectories of the agent's representations under the HGF
%
% This function can be called in two ways:
%
% (1) tapas_hgf_binary(r, p)
%
%     where r is the structure generated by tapas_fitModel and p is the parameter vector in native space;
%
% (2) tapas_hgf_binary(r, ptrans, 'trans')
%
%     where r is the structure generated by tapas_fitModel, ptrans is the parameter vector in
%     transformed space, and 'trans' is a flag indicating this.
%
% --------------------------------------------------------------------------------------------------
% Copyright (C) 2012-2017 Christoph Mathys, TNU, UZH & ETHZ
%
% This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


% Transform paramaters back to their native space if needed
if ~isempty(varargin) && strcmp(varargin{1},'trans')
    p = HGF_2levels_intercue_transp(r, p);
end

% Number of levels
try
    l = r.c_prc.n_levels;
catch
    l = (length(p)+1)/5;

    if l ~= floor(l)
        error('tapas:hgf:UndetNumLevels', 'Cannot determine number of levels');
    end
end

% Unpack parameters
mu_0 = p(1:l);
sa_0 = p(l+1:2*l);
rho  = p(2*l+1:3*l);
ka   = p(3*l+1:4*l-1);
om   = p(4*l:5*l-2);
th   = exp(p(5*l-1));

% Get shocks
u = r.u(:,1); % ADd dummy trial

% Get cues
c = r.u(:, 2);

% Number of cues (or # of independent HGF)
nc = max(c);
trials = length(c);

% Number of trials (including prior)
n = length(u);


% Initialize updated quantities

% Representations
mu_all = NaN(n,l);
pi_all = NaN(n,l);

% Other quantities
muhat_all = NaN(n,l);
pihat_all = NaN(n,l);
v_all    = NaN(n,l);
w_all     = NaN(n,l-1);
da_all    = NaN(n,l);




% Representation priors
% Note: first entries of the other quantities remain
% NaN because they are undefined and are thrown away
% at the end; their presence simply leads to consistent
% trial indices.
mu_all(1,1) = 0;
mu_all(1,2:end) = mu_0(2:end);
pi_all(1,2:end) = 1./sa_0(2:end);
k = 2;
t = ones(n,1);

% Store for every cue % with two rows, current trial and previous one
for i = 1:max(c)
    vals.cue(i).mu = NaN(2,l);
    vals.cue(i).pi = NaN(2,l);
    vals.cue(i).muhat = NaN(2,l);
    vals.cue(i).pihat = NaN(2,l);
    vals.cue(i).v = NaN(2,l);
    vals.cue(i).w = NaN(2,l);
    vals.cue(i).da = NaN(2,l);
    vals.cue(i).pi(1,1) = Inf;
    vals.cue(i).mu(1,2:end) = mu_0(2:end);
    vals.cue(i).pi(1,2:end) = 1./sa_0(2:end);
    vals.cue(i).mu(1,1) = 0;
    seencues(i) = 0; % Init array of seen cues with 0

    % Update representation prior for first trial
    mu = vals.cue(i).mu;
    pi = vals.cue(i).pi;
    muhat = vals.cue(i).muhat;
    pihat = vals.cue(i).pihat;
    v = vals.cue(i).v;
    w = vals.cue(i).w;
    da = vals.cue(i).da;

      %%%%%%%%%%%%%%%%%%%%%%
    % Effect of input u(k)
    %%%%%%%%%%%%%%%%%%%%%%

    % 2nd level prediction
    muhat(k,2) = mu(k-1,2) +t(k) *rho(2);

    % 1st level
    % ~~~~~~~~~
    % Prediction
    muhat(k,1) = tapas_sgm(ka(1) *muhat(k,2), 1);

    % Precision of prediction
    pihat(k,1) = 1/(muhat(k,1)*(1 -muhat(k,1)));

    % Updates
    pi(k,1) = Inf;
    mu(k,1) = 0;

    % Prediction error
    da(k,1) = mu(k,1) -muhat(k,1);

    % 2nd level
    % ~~~~~~~~~
    % Prediction: see above

    % Precision of prediction
    pihat(k,2) = 1/(1/pi(k-1,2) +exp(ka(2) *mu(k-1,3) +om(2)));

    % Updates
    pi(k,2) = pihat(k,2) +ka(1)^2/pihat(k,1);
    mu(k,2) = muhat(k,2) +ka(1)/pi(k,2) *da(k,1);

    % Volatility prediction error
    da(k,2) = (1/pi(k,2) +(mu(k,2) -muhat(k,2))^2) *pihat(k,2) -1;


    % Last level
    % ~~~~~~~~~~
    % Prediction
    muhat(k,l) = mu(k-1,l) +t(k) *rho(l);

    % Precision of prediction
    pihat(k,l) = 1/(1/pi(k-1,l) +t(k) *th);

    % Weighting factor
    v(k,l)   = t(k) *th;
    v(k,l-1) = t(k) *exp(ka(l-1) *mu(k-1,l) +om(l-1));
    w(k,l-1) = v(k,l-1) *pihat(k,l-1);

    % Updates
    pi(k,l) = pihat(k,l) +1/2 *ka(l-1)^2 *w(k,l-1) *(w(k,l-1) +(2 *w(k,l-1) -1) *da(k,l-1));

    if pi(k,l) <= 0
        error('tapas:hgf:NegPostPrec', 'Negative posterior precision. Parameters are in a region where model assumptions are violated.');
    end

    mu(k,l) = muhat(k,l) +1/2 *1/pi(k,l) *ka(l-1) *w(k,l-1) *da(k,l-1);

    % Volatility prediction error
    da(k,l) = (1/pi(k,l) +(mu(k,l) -muhat(k,l))^2) *pihat(k,l) -1;

    % Put in structure for prior for this cue
    vals.cue(i).mu(k-1, :) = mu(k,:);
    vals.cue(i).pi(k-1, :) = pi(k,:);
    vals.cue(i).muhat(k-1, :) = muhat(k, :);
    vals.cue(i).pihat(k-1, :) = pihat(k, :);
    vals.cue(i).v(k-1, :) = v(k, :);
    vals.cue(i).w(k-1, :) = w(k, :);
    vals.cue(i).da(k-1, :) = da(k, :);

    
end

% Pass through representation update loop

for tri = 1:n

    cue = c(tri);

    % Get cues not presented
    othercues = 1:max(c);
    othercues(cue) = [];

    % Mark current cue as seen
    seencues(cue) = 1;

    % Extract from structure
    mu = vals.cue(cue).mu;
    pi = vals.cue(cue).pi;
    muhat = vals.cue(cue).muhat;
    pihat = vals.cue(cue).pihat;
    v = vals.cue(cue).v;
    w = vals.cue(cue).w;
    da = vals.cue(cue).da;


    mu_all(tri, :) = mu(k-1,:);
    pi_all(tri, :) = pi(k-1, :);
    
    %%%%%%%%%%%%%%%%%%%%%%
    % Effect of input u(k)
    %%%%%%%%%%%%%%%%%%%%%%

    % 2nd level prediction
    muhat(k,2) = mu(k-1,2) +t(k) *rho(2);

    % 1st level
    % ~~~~~~~~~
    % Prediction
    muhat(k,1) = tapas_sgm(ka(1) *muhat(k,2), 1);

    % Precision of prediction
    pihat(k,1) = 1/(muhat(k,1)*(1 -muhat(k,1)));

    
    % Precision of prediction
    pihat(k,2) = 1/(1/pi(k-1,2) +exp(ka(2) *mu(k-1,3) +om(2)));
    
    % Last level
    % ~~~~~~~~~~
    % Prediction
    muhat(k,l) = mu(k-1,l) +t(k) *rho(l);

    % Precision of prediction
    pihat(k,l) = 1/(1/pi(k-1,l) +t(k) *th);
    
    % Put current values at next trial in global representations

    muhat_all(tri, :) = muhat(k,:);
    pihat_all(tri, :) = pihat(k,:);
    v_all(tri, :)    = v(k, :);
    w_all(tri, :)     = w(k, end-1);

    
    % Updates
    pi(k,1) = Inf;
    mu(k,1) = u(tri);

    % Prediction error
    da(k,1) = mu(k,1) -muhat(k,1);

    % 2nd level
    % ~~~~~~~~~
    % Prediction: see above

    % Updates
    pi(k,2) = pihat(k,2) +ka(1)^2/pihat(k,1);
    mu(k,2) = muhat(k,2) +ka(1)/pi(k,2) *da(k,1);

    % Volatility prediction error
    da(k,2) = (1/pi(k,2) +(mu(k,2) -muhat(k,2))^2) *pihat(k,2) -1;



    % Weighting factor
    v(k,l)   = t(k) *th;
    v(k,l-1) = t(k) *exp(ka(l-1) *mu(k-1,l) +om(l-1));
    w(k,l-1) = v(k,l-1) *pihat(k,l-1);

    % Updates
    pi(k,l) = pihat(k,l) +1/2 *ka(l-1)^2 *w(k,l-1) *(w(k,l-1) +(2 *w(k,l-1) -1) *da(k,l-1));

    if pi(k,l) <= 0
        error('tapas:hgf:NegPostPrec', 'Negative posterior precision. Parameters are in a region where model assumptions are violated.');
    end

    mu(k,l) = muhat(k,l) +1/2 *1/pi(k,l) *ka(l-1) *w(k,l-1) *da(k,l-1);

    % Volatility prediction error
    da(k,l) = (1/pi(k,l) +(mu(k,l) -muhat(k,l))^2) *pihat(k,l) -1;

    
    da_all(tri, :)    = da(k, :);
    
    % Put in structure for prior for this cue
    vals.cue(cue).mu(k-1, :) = mu(k,:);
    vals.cue(cue).pi(k-1, :) = pi(k,:);
    vals.cue(cue).muhat(k-1, :) = muhat(k, :);
    vals.cue(cue).pihat(k-1, :) = pihat(k, :);
    vals.cue(cue).v(k-1, :) = v(k, :);
    vals.cue(cue).w(k-1, :) = w(k, :);
    vals.cue(cue).da(k-1, :) = da(k, :);


    % Update other cues if shock
    if u(tri)
        for o = othercues% Loop all unpresented cue
            if seencues(o) == 1% Update only seen cues
              % Extract from structure
              mu = vals.cue(o).mu;
              pi = vals.cue(o).pi;
              muhat = vals.cue(o).muhat;
              pihat = vals.cue(o).pihat;
              v = vals.cue(o).v;
              w = vals.cue(o).w;
              da = vals.cue(o).da;

               
              %%%%%%%%%%%%%%%%%%%%%%
              % PREDICTION AND PRECISION
              %%%%%%%%%%%%%%%%%%%%%%

              % 2nd level prediction
              muhat(k,2) = mu(k-1,2) +t(k) *rho(2);

              % 1st level
              % ~~~~~~~~~
              % Prediction
              muhat(k,1) = tapas_sgm(ka(1) *muhat(k,2), 1);

              % Precision of prediction
              pihat(k,1) = 1/(muhat(k,1)*(1 -muhat(k,1)));
              
               % 2nd level
              % ~~~~~~~~~
              % Prediction: see above

              % Precision of prediction
              pihat(k,2) = 1/(1/pi(k-1,2) +exp(ka(2) *mu(k-1,3) +om(2)));
              
                            % Last level
              % ~~~~~~~~~~
              % Prediction
              muhat(k,l) = mu(k-1,l) +t(k) *rho(l);

              % Precision of prediction
              pihat(k,l) = 1/(1/pi(k-1,l) +t(k) *th);

              % Weighting factor
              v(k,l)   = t(k) *th;
              v(k,l-1) = t(k) *exp(ka(l-1) *mu(k-1,l) +om(l-1));
              w(k,l-1) = v(k,l-1) *pihat(k,l-1);


              %%%%%%%%%%%%%%%%%%%%%%
              % UPDATES
              %%%%%%%%%%%%%%%%%%%%%%
              
              % 1st LEVEL Updates
              pi(k,1) = Inf;
              mu(k,1) = not(u(tri));

              % Prediction error
              da(k,1) = mu(k,1) - muhat(k,1);

           
              % 2nd LEVEL Updates
              pi(k,2) = pihat(k,2) +ka(1)^2/pihat(k,1);
              mu(k,2) = muhat(k,2) +ka(1)/pi(k,2) *da(k,1);

              % Volatility prediction error
              da(k,2) = (1/pi(k,2) +(mu(k,2) -muhat(k,2))^2) *pihat(k,2) -1;


              % Updates
              pi(k,l) = pihat(k,l) +1/2 *ka(l-1)^2 *w(k,l-1) *(w(k,l-1) +(2 *w(k,l-1) -1) *da(k,l-1));

              if pi(k,l) <= 0
                  error('tapas:hgf:NegPostPrec', 'Negative posterior precision. Parameters are in a region where model assumptions are violated.');
              end

              mu(k,l) = muhat(k,l) +1/2 *1/pi(k,l) *ka(l-1) *w(k,l-1) *da(k,l-1);

              % Volatility prediction error
              da(k,l) = (1/pi(k,l) +(mu(k,l) -muhat(k,l))^2) *pihat(k,l) -1;

              % Put in structure at first trial
              vals.cue(o).mu(k-1, :) = mu(k,:);
              vals.cue(o).pi(k-1, :) = pi(k,:);
              vals.cue(o).muhat(k-1, :) = muhat(k, :);
              vals.cue(o).pihat(k-1, :) = pihat(k, :);
              vals.cue(o).v(k-1, :) = v(k, :);
              vals.cue(o).w(k-1, :) = w(k, :);
              vals.cue(o).da(k-1, :) = da(k, :);

            end
        end
    end

end

% Remove representation priors
% mu_all(end,:)  = [];
% pi_all(end,:)  = [];

% Implied learning rate at the first level
sgmmu2 = tapas_sgm(ka(1) *mu_all(:,2), 1);
dasgmmu2 = u -sgmmu2;
lr1    = diff(sgmmu2)./dasgmmu2(2:n,1);
lr1(da_all(2:n,1)==0) = 0;

% Check validity of trajectories
if any(isnan(mu_all(:))) || any(isnan(pi_all(:)))
    error('tapas:hgf:VarApproxInvalid', 'Variational approximation invalid. Parameters are in a region where model assumptions are violated.');
else
    % Check for implausible jumps in trajectories
    dmu = diff(mu_all(:,2:end));
    dpi = diff(pi_all(:,2:end));
    rmdmu = repmat(sqrt(mean(dmu.^2)),length(dmu),1);
    rmdpi = repmat(sqrt(mean(dpi.^2)),length(dpi),1);

    jumpTol = 16;
    if any(abs(dmu(:)) > jumpTol*rmdmu(:)) || any(abs(dpi(:)) > jumpTol*rmdpi(:))
        error('tapas:hgf:VarApproxInvalid', 'Variational approximation invalid. Parameters are in a region where model assumptions are violated.');
    end
end

% Remove other dummy initial values
% muhat_all(end,:) = [];
% pihat_all(end,:) = [];
% v_all(end,:)     = [];
% w_all(end,:)     = [];
% da_all(end,:)    = [];

% Create result data structure
traj = struct;

traj.mu     = mu_all;
traj.sa     = 1./pi_all;

traj.muhat  = muhat_all;
traj.sahat  = 1./pihat_all;

traj.v      = v_all;
traj.w      = w_all;
traj.da     = da_all;

% Updates with respect to prediction
traj.ud = mu_all -muhat_all;

% Psi (precision weights on prediction errors)
psi        = NaN(n,l);
psi(:,2)   = 1./pi_all(:,2);
psi(:,3:l) = pihat_all(:,2:l-1)./pi_all(:,3:l);
traj.psi   = psi;

% Epsilons (precision-weighted prediction errors)
epsi        = NaN(n,l);
epsi(:,2:l) = psi(:,2:l) .*da_all(:,1:l-1);
traj.epsi   = epsi;

% Full learning rate (full weights on prediction errors)

% wt        = NaN(n-1,l);
% wt(:,1)   = lr1;
% wt(:,2)   = psi(2:n,2);
% wt(:,3:l) = 1/2 *(v_all(:,2:l-1) *diag(ka(2:l-1))) .*psi(:,3:l);
% traj.wt   = wt;

% Create matrices for use by the observation model
infStates = NaN(n,l,4);
infStates(:,:,1) = traj.muhat;
infStates(:,:,2) = traj.sahat;
infStates(:,:,3) = traj.mu;
infStates(:,:,4) = traj.sa;

return;
