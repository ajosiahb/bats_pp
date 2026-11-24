%
% BATS regression models for primary production
%
% version 1.0
% Scott Doney
% November 23rd, 2026
%
% model code to support of:
% Comparing Machine Learning Regression Models to Predict Ocean 
% Primary Production: Insights from the Bermuda Atlantic Time-series Study
%
% Elaine Y. Liu, Anna J. Brown, Eleanor Lawrence, Wiley H. Wolfe, 
% Scott C. Doney

% flags for options
iopt = 0;   % hyperparameter tuning, = 0 (off) or = 1 (on)
ipls = 0;   % partial least squares for predictors = 0 (off) or = 1 (on)
idepthtest = 0; % test removing depth predictor = 0 (off) or = 1 (on)
rng("default") % reproducability

% loop through regressions nmax times
nmax = 16;    % standard loop is 16

% index values for variables (see Table 1 in manuscript)
itime=1; imon=2; id=3; idstd=4; 
ipp = 5; imld = 6; ipar = 7; ichl = 8; it=9; ino3 = 10;
ipo4 = 11; ipoc = 12;

% load observed BATS primary production
xx = readmatrix('bats_env_data.csv');
% 3808 valid primary production data points

% create a environmental predictor matrix xmod
xmod = [xx(:,id) xx(:,imld) xx(:,ipar) ...
    xx(:,it) xx(:,ino3) xx(:,ipo4) xx(:,ipoc) xx(:,ichl)];
[in jn] = size(xmod);
rho = corr(xmod);
xmod = zscore(xmod); % standardize the predictors 

if idepthtest==1
% test experiment removing depth variations
   xmod(:,1) = zeros(in,1);
end

% set up 85% train/15% test dataset partitions
% 2020.4014 Mid May, index 3268
ttrain = min(xx(:,itime)) + (max(xx(:,itime)) - min(xx(:,itime))) * .85;
% ik = 3268
ik = sum(xx(:,itime)<=ttrain);

% fit regressions to BATS training data
% compute model skill metrics & predictor importance

% multiple linear regression MLR 
Mdlmlr = fitlm(xmod(1:ik,:),asinh(xx(1:ik,ipp)));
ymodmlr = predict(Mdlmlr,xmod);
rmlrtrain = corr(asinh(xx(1:ik,ipp)),ymodmlr(1:ik));
rmlrtest = corr(asinh(xx(ik+1:in,ipp)),ymodmlr(ik+1:in));
rmsmlrtrain = rms(asinh(xx(1:ik,ipp)) - ymodmlr(1:ik));
rmsmlrtest = rms(asinh(xx(ik+1:in,ipp)) - ymodmlr(ik+1:in));

Mdl = fitlm(asinh(xx(1:ik,ipp)),ymodmlr(1:ik));
w = table2array(Mdl.Coefficients);
slopemlrtrain = w(2,1);
Mdl = fitlm(asinh(xx(ik+1:in,ipp)),ymodmlr(ik+1:in));
w = table2array(Mdl.Coefficients); 
slopemlrtest = w(2,1);
[1 1 rmlrtrain rmlrtest rmsmlrtrain rmsmlrtest ...
    slopemlrtrain slopemlrtest] 

temp = dominance(xmod(1:ik,:),asinh(xx(1:ik,ipp)));
w = table2array(temp(:,2));
impmlr = abs(w); impmlr = impmlr/max(impmlr);
[1 1 impmlr']

% partial least squares PLS
[xpls,ypls,xscorepls,yscorepls,betapls,pctvarpls,msepls,statspls] ...
    = plsregress(xmod(1:ik,:),asinh(xx(1:ik,ipp)),jn);
% std(xscorepls(:,1)) = 0.0175; each column originally orthonormal
xmodpls = xmod * xpls; xmodpls = zscore(xmodpls);
temp = std(xscorepls(:,1));
xscorepls = xscorepls/temp; xpls = xpls * temp;
ymodpls = [ones(in,1) xmod]*betapls;

rplstrain = corr(asinh(xx(1:ik,ipp)),ymodpls(1:ik));
rplstest = corr(asinh(xx(ik+1:in,ipp)),ymodpls(ik+1:in));
rmsplstrain = rms(asinh(xx(1:ik,ipp)) - ymodpls(1:ik));
rmsplstest = rms(asinh(xx(ik+1:in,ipp)) - ymodpls(ik+1:in));
Mdl = fitlm(asinh(xx(1:ik,ipp)),ymodpls(1:ik));
w = table2array(Mdl.Coefficients);
slopeplstrain = w(2,1);
Mdl = fitlm(asinh(xx(ik+1:in,ipp)),ymodpls(ik+1:in));
w = table2array(Mdl.Coefficients); 
slopeplstest = w(2,1);
[0 0 rplstrain rplstest rmsplstrain rmsplstest ...
    slopeplstrain slopeplstest] 

temp = dominance(xscorepls,asinh(xx(1:ik,ipp)));
w = table2array(temp(:,2));
imppls = abs(w); imppls = imppls/max(imppls);
[0 0 imppls']
% cumsum(pctvarpls(2,:))
% 0.5575  0.6667  0.6856  0.6961  0.6970  0.6973  0.6974  0.6974
% 
% cumsum(pctvarpls(2,:))/temp(8)
% 0.7995  0.9561  0.9831  0.9982  0.9994  1.0000  1.0000  1.0000

% Stepwise MLR
Mdlmlrstep = stepwiselm(xmod(1:ik,:),asinh(xx(1:ik,ipp)), ...
    'Upper','linear');
ymodmlrstep = predict(Mdlmlrstep,xmod);
rmlrsteptrain = corr(asinh(xx(1:ik,ipp)),ymodmlrstep(1:ik));
rmlrsteptest = corr(asinh(xx(ik+1:in,ipp)),ymodmlrstep(ik+1:in));
rmsmlrsteptrain = rms(asinh(xx(1:ik,ipp)) - ymodmlrstep(1:ik));
rmsmlrsteptest = rms(asinh(xx(ik+1:in,ipp)) - ymodmlrstep(ik+1:in));

Mdl = fitlm(asinh(xx(1:ik,ipp)),ymodmlrstep(1:ik));
w = table2array(Mdl.Coefficients);
slopemlrsteptrain = w(2,1);
Mdl = fitlm(asinh(xx(ik+1:in,ipp)),ymodmlrstep(ik+1:in));
w = table2array(Mdl.Coefficients); 
slopemlrsteptest = w(2,1);
[2 1 rmlrsteptrain rmlrsteptest ...
    rmsmlrsteptrain rmsmlrsteptest ...
    slopemlrsteptrain slopemlrsteptest] 

temp = dominance([xmod(1:ik,1:5) xmod(1:ik,7:8)],asinh(xx(1:ik,ipp)));
wstep = table2array(temp(:,2));
impmlrstep(1:5) = abs(wstep(1:5)); impmlrstep(6) = 0;
impmlrstep(7:8) = abs(wstep(6:7)); 
impmlrstep = impmlrstep/max(impmlrstep);
[2 1 impmlrstep]

% for partial least squares sensitivity experiment
% replace predictors for machine learning models with PLS components
% orthonormal xscorepls standardized above
jnpls = 4;
if ipls==1
    xmod = zeros(size(xmod));
    xmod(:,1:jnpls) = xmodpls(:,1:jnpls);
end

% initialize machine learning model cases
ymodrfr=zeros(in,1); ymodlsb=zeros(in,1); ymodnnr=zeros(in,1); 
imprfr=zeros(1,jn); implsb=zeros(1,jn); 
hpoOptions = hyperparameterOptimizationOptions(Optimizer="bayesopt", ...
    MaxObjectiveEvaluations=50, Verbose=0, ShowPlots=0); 
% "bayesopt" or "gridsearch" (gridsearch much slower)

% begin loop for creating sets of model fits for each machine learning case
for n=1:nmax

% Random Forest bagging RFR
if iopt==1
    Mdlrfr = fitrensemble(xmod(1:ik,:),asinh(xx(1:ik,ipp)), ...
        Method="Bag", OptimizeHyperparameters= ...
        ["NumLearningCycles","MaxNumSplits","MinLeafSize",...
        "NumVariablesToSample"], ...        
        HyperparameterOptimizationOptions=hpoOptions);
else
    Mdlrfr = fitrensemble(xmod(1:ik,:),asinh(xx(1:ik,ipp)), ...
        Method="Bag"); % or Method="LSBoost"
end
ymod = predict(Mdlrfr,xmod);
ymodrfr = ymodrfr + 1/nmax * ymod;
imp = predictorImportance(Mdlrfr);
imp = imp/max(imp); 
imprfr = imprfr + 1/nmax*imp; 

rtrain = corr(asinh(xx(1:ik,ipp)),ymod(1:ik));
rtest = corr(asinh(xx(ik+1:in,ipp)),ymod(ik+1:in));
rmstrain = rms(asinh(xx(1:ik,ipp)) - ymod(1:ik));
rmstest = rms(asinh(xx(ik+1:in,ipp)) - ymod(ik+1:in));

Mdl = fitlm(asinh(xx(1:ik,ipp)),ymod(1:ik));
w = table2array(Mdl.Coefficients);
slopetrain = w(2,1);
Mdl = fitlm(asinh(xx(ik+1:in,ipp)),ymod(ik+1:in));
w = table2array(Mdl.Coefficients); 
slopetest = w(2,1);
[3 n rtrain rtest rmstrain rmstest ...
    slopetrain slopetest] 
[3 n imp ]

% Random Forest Least-squares boosting (LSB)
if iopt==1
    Mdllsb = fitrensemble(xmod(1:ik,:),asinh(xx(1:ik,ipp)), ...
        Method="LSBoost", OptimizeHyperparameters= ...
        ["NumLearningCycles","MaxNumSplits","MinLeafSize",...
        "NumVariablesToSample","LearnRate"], ...
        HyperparameterOptimizationOptions=hpoOptions);
else
    Mdllsb = fitrensemble(xmod(1:ik,:),asinh(xx(1:ik,ipp)), ...
        Method="LSBoost"); % or Method="LSBoost"
end

ymod = predict(Mdllsb,xmod);
ymodlsb = ymodlsb + 1/nmax * ymod;
imp = predictorImportance(Mdllsb);
imp = imp/max(imp); 
implsb = implsb + 1/nmax*imp; 

rtrain = corr(asinh(xx(1:ik,ipp)),ymod(1:ik));
rtest = corr(asinh(xx(ik+1:in,ipp)),ymod(ik+1:in));
rmstrain = rms(asinh(xx(1:ik,ipp)) - ymod(1:ik));
rmstest = rms(asinh(xx(ik+1:in,ipp)) - ymod(ik+1:in));

Mdl = fitlm(asinh(xx(1:ik,ipp)),ymod(1:ik));
w = table2array(Mdl.Coefficients);
slopetrain = w(2,1);
Mdl = fitlm(asinh(xx(ik+1:in,ipp)),ymod(ik+1:in));
w = table2array(Mdl.Coefficients); 
slopetest = w(2,1);
[4 n rtrain rtest rmstrain rmstest ...
    slopetrain slopetest] 
[4 n imp ]

% Neural net regression NNR
if iopt==1
    Mdlnnr = fitrnet(xmod(1:ik,:),asinh(xx(1:ik,ipp)), ...
        OptimizeHyperparameters= ["Activations","Lambda", ...
        "LayerSizes","Standardize"], ...        
        HyperparameterOptimizationOptions=hpoOptions);
else
    Mdlnnr = fitrnet(xmod(1:ik,:),asinh(xx(1:ik,ipp))); 
end

ymod = predict(Mdlnnr,xmod);
ymodnnr = ymodnnr + 1/nmax * ymod;

rtrain = corr(asinh(xx(1:ik,ipp)),ymod(1:ik));
rtest = corr(asinh(xx(ik+1:in,ipp)),ymod(ik+1:in));
rmstrain = rms(asinh(xx(1:ik,ipp)) - ymod(1:ik));
rmstest = rms(asinh(xx(ik+1:in,ipp)) - ymod(ik+1:in));

Mdl = fitlm(asinh(xx(1:ik,ipp)),ymod(1:ik));
w = table2array(Mdl.Coefficients);
slopetrain = w(2,1);
Mdl = fitlm(asinh(xx(ik+1:in,ipp)),ymod(ik+1:in));
w = table2array(Mdl.Coefficients); 
slopetest = w(2,1);
[5 n rtrain rtest rmstrain rmstest ...
    slopetrain slopetest] 

save bats_model_Mdl.mat Mdlmlr Mdlmlrstep Mdlrfr Mdllsb Mdlnnr

% uncomment pause if choose to rename saved mat files for individual fits
% within model fit sets
%"pause"
%pause

end % end of large iteration loop

% model skill metrics for aggregated predictions
rrfrtrain = corr(asinh(xx(1:ik,ipp)),ymodrfr(1:ik));
rrfrtest = corr(asinh(xx(ik+1:in,ipp)),ymodrfr(ik+1:in));
rmsrfrtrain = rms(asinh(xx(1:ik,ipp)) - ymodrfr(1:ik));
rmsrfrtest = rms(asinh(xx(ik+1:in,ipp)) - ymodrfr(ik+1:in));
Mdl = fitlm(asinh(xx(1:ik,ipp)),ymodrfr(1:ik));
w = table2array(Mdl.Coefficients);
sloperfrtrain = w(2,1);
Mdl = fitlm(asinh(xx(ik+1:in,ipp)),ymodrfr(ik+1:in));
w = table2array(Mdl.Coefficients); 
sloperfrtest = w(2,1);

rlsbtrain = corr(asinh(xx(1:ik,ipp)),ymodlsb(1:ik));
rlsbtest = corr(asinh(xx(ik+1:in,ipp)),ymodlsb(ik+1:in));
rmslsbtrain = rms(asinh(xx(1:ik,ipp)) - ymodlsb(1:ik));
rmslsbtest = rms(asinh(xx(ik+1:in,ipp)) - ymodlsb(ik+1:in));
Mdl = fitlm(asinh(xx(1:ik,ipp)),ymodlsb(1:ik));
w = table2array(Mdl.Coefficients);
slopelsbtrain = w(2,1);
Mdl = fitlm(asinh(xx(ik+1:in,ipp)),ymodlsb(ik+1:in));
w = table2array(Mdl.Coefficients); 
slopelsbtest = w(2,1);

rnnrtrain = corr(asinh(xx(1:ik,ipp)),ymodnnr(1:ik));
rnnrtest = corr(asinh(xx(ik+1:in,ipp)),ymodnnr(ik+1:in));
rmsnnrtrain = rms(asinh(xx(1:ik,ipp)) - ymodnnr(1:ik));
rmsnnrtest = rms(asinh(xx(ik+1:in,ipp)) - ymodnnr(ik+1:in));
Mdl = fitlm(asinh(xx(1:ik,ipp)),ymodnnr(1:ik));
w = table2array(Mdl.Coefficients);
slopennrtrain = w(2,1);
Mdl = fitlm(asinh(xx(ik+1:in,ipp)),ymodnnr(ik+1:in));
w = table2array(Mdl.Coefficients); 
slopennrtest = w(2,1);

% display aggregated prediction metrics
[3 0 rrfrtrain rrfrtest rmsrfrtrain rmsrfrtest ...
    sloperfrtrain sloperfrtest] 
[3 0 imprfr ]

[4 0 rlsbtrain rlsbtest rmslsbtrain rmslsbtest ...
    slopelsbtrain slopelsbtest] 
[4 0 implsb ]

[5 0 rnnrtrain rnnrtest rmsnnrtrain rmsnnrtest ...
    slopennrtrain slopennrtest] 

% create output file with predicted primary production from different
% regression model cases 
% itime=1; imon=2; id=3; idstd=4; ipp = 5;
ymod = [xx(:,itime) xx(:,imon) xx(:,id) ...
    xx(:,idstd) asinh(xx(:,ipp)) ymodmlr ymodrfr ymodlsb ymodnnr];

writematrix(ymod,'bats_model_output.csv');


