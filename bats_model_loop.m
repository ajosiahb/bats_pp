iopt = 1;   % = 0 (off) or = 1 (on)
idepthtest = 1; % = 0 (off) or = 1 (on)
rng("default") % reproducability

% loop through regressions nmax times
nmax = 16;    % standard loop is 16

% 3808 valid pp data points

itime=1; imon=2; id=3; idstd=4; 
ipp = 5; imld = 6; ipar = 7; ichl = 8; it=9; ino3 = 10;
ipo4 = 11; ipoc = 12;

xx = readmatrix('bats_env_data.csv');
 
xmod = [xx(:,id) xx(:,imld) xx(:,ipar) ...
    xx(:,it) xx(:,ino3) xx(:,ipo4) xx(:,ipoc) xx(:,ichl)];
[in jn] = size(xmod);
rho = corr(xmod);
xmod = zscore(xmod); % standardize the predictors 

if idepthtest==1
% test experiment removing depth variations
   xmod(:,1) = zeros(in,1);
end

% 85% train/15% test
% 2020.4014 Mid May, index 3268
ttrain = min(xx(:,itime)) + (max(xx(:,itime)) - min(xx(:,itime))) * .85;
% ik = 3268
ik = sum(xx(:,itime)<=ttrain);

% MLR 
% https://www.mathworks.com/help/stats/linearmodel.html
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

ymodrfr=zeros(in,1); ymodlsb=zeros(in,1); ymodnnr=zeros(in,1); 
imprfr=zeros(1,jn); implsb=zeros(1,jn); 
hpoOptions = hyperparameterOptimizationOptions(Optimizer="bayesopt", ...
    MaxObjectiveEvaluations=50, Verbose=0); 
% "bayesopt" or "gridsearch" (gridsearch much slower)

for n=1:nmax

% Random Forest bagging

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

% Random Forest LS Boost
% "LSBoost"	Least-squares boosting (LSBoost)
%
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

% Neural net
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

%"pause"
%pause

end % end of large iteration loop

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

[3 0 rrfrtrain rrfrtest rmsrfrtrain rmsrfrtest ...
    sloperfrtrain sloperfrtest] 
[3 0 imprfr ]

[4 0 rlsbtrain rlsbtest rmslsbtrain rmslsbtest ...
    slopelsbtrain slopelsbtest] 
[4 0 implsb ]

[5 0 rnnrtrain rnnrtest rmsnnrtrain rmsnnrtest ...
    slopennrtrain slopennrtest] 

% itime=1; imon=2; id=3; idstd=4; ipp = 5;

ymod = [xx(:,itime) xx(:,imon) xx(:,id) ...
    xx(:,idstd) asinh(xx(:,ipp)) ymodmlr ymodrfr ymodlsb ymodnnr];

writematrix(ymod,'bats_model_output.csv');

%bestPoint(Mdlrfr.HyperparameterOptimizationResults)
%bestPoint(Mdllsb.HyperparameterOptimizationResults)

sc=8;
figure(1)
subplot(2,2,1)
scatter(asinh(xx(1:ik,ipp)),ymodmlr(1:ik),sc, ...
    -xx(1:ik,id))
ylabel('Model asinh(PP) (mg/m^3/d)','FontSize',10); ...
    xlabel('asinh(PP) (mg/m^3/d)','FontSize',10); 
title('MLR BATS Training')
axis('square'); axis([-2 5 -2 5])
Mdl = fitlm(asinh(xx(1:ik,ipp)),ymodmlr(1:ik));
rmserr = rms(asinh(xx(1:ik,ipp))-ymodmlr(1:ik));
w = table2array(Mdl.Coefficients);
xlinear = [-2:5]; 
ylinear = w(1,1) + w(2,1)*xlinear;
hold on
plot(xlinear,ylinear,'k-',LineWidth=1.5)
plot(xlinear,xlinear,'k--',LineWidth=1.25)
hold off
text(0.05,0.84,"R^2 = " + num2str(Mdl.Rsquared.Ordinary,3), ...
    FontSize=10, Units='normalized')
text(0.05,0.93,"RMSE = " + num2str(rmserr,3), ...
    FontSize=10, Units='normalized')
text(0.55,0.14,"Slope = ",FontSize=10, Units='normalized')
text(0.55,0.05,num2str(w(2,1),3) + "\pm" + ...
    num2str(w(2,2),1),FontSize=10, Units='normalized')
c = colorbar; c.Label.String = 'Depth (m)'; c.FontSize=10;

subplot(2,2,2)
scatter(asinh(xx(1:ik,ipp)),ymodrfr(1:ik),sc, ...
    -xx(1:ik,id))
ylabel('Model asinh(PP) (mg/m^3/d)','FontSize',10); ...
    xlabel('asinh(PP) (mg/m^3/d)','FontSize',10); 
title('RFR BATS Training')
axis('square'); axis([-2 5 -2 5])
Mdl = fitlm(asinh(xx(1:ik,ipp)),ymodrfr(1:ik));
rmserr = rms(asinh(xx(1:ik,ipp))-ymodrfr(1:ik));
w = table2array(Mdl.Coefficients);
xlinear = [-2:5]; 
ylinear = w(1,1) + w(2,1)*xlinear;
hold on
plot(xlinear,ylinear,'k-',LineWidth=1.5)
plot(xlinear,xlinear,'k--',LineWidth=1.25)
hold off
text(0.05,0.84,"R^2 = " + num2str(Mdl.Rsquared.Ordinary,3), ...
    FontSize=10, Units='normalized')
text(0.05,0.93,"RMSE = " + num2str(rmserr,3), ...
    FontSize=10, Units='normalized')
text(0.55,0.14,"Slope = ",FontSize=10, Units='normalized')
text(0.55,0.05,num2str(w(2,1),3) + "\pm" + ...
    num2str(w(2,2),1),FontSize=10, Units='normalized')
c = colorbar; c.Label.String = 'Depth (m)'; c.FontSize=10;

subplot(2,2,3)
scatter(asinh(xx(1:ik,ipp)),ymodlsb(1:ik),sc, ...
    -xx(1:ik,id))
ylabel('Model asinh(PP) (mg/m^3/d)','FontSize',10); ...
    xlabel('asinh(PP) (mg/m^3/d)','FontSize',10); 
title('LSB BATS Training')
axis('square'); axis([-2 5 -2 5])
Mdl = fitlm(asinh(xx(1:ik,ipp)),ymodlsb(1:ik));
rmserr = rms(asinh(xx(1:ik,ipp))-ymodlsb(1:ik));
w = table2array(Mdl.Coefficients);
xlinear = [-2:5]; 
ylinear = w(1,1) + w(2,1)*xlinear;
hold on
plot(xlinear,ylinear,'k-',LineWidth=1.5)
plot(xlinear,xlinear,'k--',LineWidth=1.25)
hold off
text(0.05,0.84,"R^2 = " + num2str(Mdl.Rsquared.Ordinary,3), ...
    FontSize=10, Units='normalized')
text(0.05,0.93,"RMSE = " + num2str(rmserr,3), ...
    FontSize=10, Units='normalized')
text(0.55,0.14,"Slope = ",FontSize=10, Units='normalized')
text(0.55,0.05,num2str(w(2,1),3) + "\pm" + ...
    num2str(w(2,2),1),FontSize=10, Units='normalized')
c = colorbar; c.Label.String = 'Depth (m)'; c.FontSize=10;

subplot(2,2,4)
scatter(asinh(xx(1:ik,ipp)),ymodnnr(1:ik),sc, ...
    -xx(1:ik,id))
ylabel('Model asinh(PP) (mg/m^3/d)','FontSize',10); ...
    xlabel('asinh(PP) (mg/m^3/d)','FontSize',10); 
title('NNR BATS Training')
axis('square'); axis([-2 5 -2 5])
Mdl = fitlm(asinh(xx(1:ik,ipp)),ymodnnr(1:ik));
rmserr = rms(asinh(xx(1:ik,ipp))-ymodnnr(1:ik));
w = table2array(Mdl.Coefficients);
xlinear = [-2:5]; 
ylinear = w(1,1) + w(2,1)*xlinear;
hold on
plot(xlinear,ylinear,'k-',LineWidth=1.5)
plot(xlinear,xlinear,'k--',LineWidth=1.25)
hold off
text(0.05,0.84,"R^2 = " + num2str(Mdl.Rsquared.Ordinary,3), ...
    FontSize=10, Units='normalized')
text(0.05,0.93,"RMSE = " + num2str(rmserr,3), ...
    FontSize=10, Units='normalized')
text(0.55,0.14,"Slope = ",FontSize=10, Units='normalized')
text(0.55,0.05,num2str(w(2,1),3) + "\pm" + ...
    num2str(w(2,2),1),FontSize=10, Units='normalized')
c = colorbar; c.Label.String = 'Depth (m)'; c.FontSize=10;


%%%%%
figure(2)
subplot(2,2,1)
scatter(asinh(xx(ik+1:in,ipp)),ymodmlr(ik+1:in),sc, ...
    -xx(ik+1:in,id))
ylabel('Model asinh(PP) (mg/m^3/d)','FontSize',10); ...
    xlabel('asinh(PP) (mg/m^3/d)','FontSize',10); 
title('MLR BATS Testing')
axis('square'); axis([-2 5 -2 5])
Mdl = fitlm(asinh(xx(ik+1:in,ipp)),ymodmlr(ik+1:in));
rmserr = rms(asinh(xx(ik+1:in,ipp))-ymodmlr(ik+1:in));
w = table2array(Mdl.Coefficients);
xlinear = [-2:5]; 
ylinear = w(1,1) + w(2,1)*xlinear;
hold on
plot(xlinear,ylinear,'k-',LineWidth=1.5)
plot(xlinear,xlinear,'k--',LineWidth=1.25)
hold off
text(0.05,0.84,"R^2 = " + num2str(Mdl.Rsquared.Ordinary,3), ...
    FontSize=10, Units='normalized')
text(0.05,0.93,"RMSE = " + num2str(rmserr,3), ...
    FontSize=10, Units='normalized')
text(0.55,0.14,"Slope = ",FontSize=10, Units='normalized')
text(0.55,0.05,num2str(w(2,1),3) + "\pm" + ...
    num2str(w(2,2),2),FontSize=10, Units='normalized')
c = colorbar; c.Label.String = 'Depth (m)'; c.FontSize=10;

subplot(2,2,2)
scatter(asinh(xx(ik+1:in,ipp)),ymodrfr(ik+1:in),sc, ...
    -xx(ik+1:in,id))
ylabel('Model asinh(PP) (mg/m^3/d)','FontSize',10); ...
    xlabel('asinh(PP) (mg/m^3/d)','FontSize',10); 
title('RFR BATS Testing')
axis('square'); axis([-2 5 -2 5])
Mdl = fitlm(asinh(xx(ik+1:in,ipp)),ymodrfr(ik+1:in));
rmserr = rms(asinh(xx(ik+1:in,ipp))-ymodrfr(ik+1:in));
w = table2array(Mdl.Coefficients);
xlinear = [-2:5]; 
ylinear = w(1,1) + w(2,1)*xlinear;
hold on
plot(xlinear,ylinear,'k-',LineWidth=1.5)
plot(xlinear,xlinear,'k--',LineWidth=1.25)
hold off
text(0.05,0.84,"R^2 = " + num2str(Mdl.Rsquared.Ordinary,3), ...
    FontSize=10, Units='normalized')
text(0.05,0.93,"RMSE = " + num2str(rmserr,3), ...
    FontSize=10, Units='normalized')
text(0.55,0.14,"Slope = ",FontSize=10, Units='normalized')
text(0.55,0.05,num2str(w(2,1),3) + "\pm" + ...
    num2str(w(2,2),2),FontSize=10, Units='normalized')
c = colorbar; c.Label.String = 'Depth (m)'; c.FontSize=10;

subplot(2,2,3)
scatter(asinh(xx(ik+1:in,ipp)),ymodlsb(ik+1:in),sc, ...
    -xx(ik+1:in,id))
ylabel('Model asinh(PP) (mg/m^3/d)','FontSize',10); ...
    xlabel('asinh(PP) (mg/m^3/d)','FontSize',10); 
title('LSB BATS Testing')
axis('square'); axis([-2 5 -2 5])
Mdl = fitlm(asinh(xx(ik+1:in,ipp)),ymodlsb(ik+1:in));
rmserr = rms(asinh(xx(ik+1:in,ipp))-ymodlsb(ik+1:in));
w = table2array(Mdl.Coefficients);
xlinear = [-2:5]; 
ylinear = w(1,1) + w(2,1)*xlinear;
hold on
plot(xlinear,ylinear,'k-',LineWidth=1.5)
plot(xlinear,xlinear,'k--',LineWidth=1.25)
hold off
text(0.05,0.84,"R^2 = " + num2str(Mdl.Rsquared.Ordinary,3), ...
    FontSize=10, Units='normalized')
text(0.05,0.93,"RMSE = " + num2str(rmserr,3), ...
    FontSize=10, Units='normalized')
text(0.55,0.14,"Slope = ",FontSize=10, Units='normalized')
text(0.55,0.05,num2str(w(2,1),3) + "\pm" + ...
    num2str(w(2,2),2),FontSize=10, Units='normalized')
c = colorbar; c.Label.String = 'Depth (m)'; c.FontSize=10;

subplot(2,2,4)
scatter(asinh(xx(ik+1:in,ipp)),ymodnnr(ik+1:in),sc, ...
    -xx(ik+1:in,id))
ylabel('Model asinh(PP) (mg/m^3/d)','FontSize',10); ...
    xlabel('asinh(PP) (mg/m^3/d)','FontSize',10); 
title('NNR BATS Testing')
axis('square'); axis([-2 5 -2 5])
Mdl = fitlm(asinh(xx(ik+1:in,ipp)),ymodnnr(ik+1:in));
rmserr = rms(asinh(xx(ik+1:in,ipp))-ymodnnr(ik+1:in));
w = table2array(Mdl.Coefficients);
xlinear = [-2:5]; 
ylinear = w(1,1) + w(2,1)*xlinear;
hold on
plot(xlinear,ylinear,'k-',LineWidth=1.5)
plot(xlinear,xlinear,'k--',LineWidth=1.25)
hold off
text(0.05,0.84,"R^2 = " + num2str(Mdl.Rsquared.Ordinary,3), ...
    FontSize=10, Units='normalized')
text(0.05,0.93,"RMSE = " + num2str(rmserr,3), ...
    FontSize=10, Units='normalized')
text(0.55,0.14,"Slope = ",FontSize=10, Units='normalized')
text(0.55,0.05,num2str(w(2,1),3) + "\pm" + ...
    num2str(w(2,2),2),FontSize=10, Units='normalized')
c = colorbar; c.Label.String = 'Depth (m)'; c.FontSize=10;

%%%%%
% predictor importance

figure(3)
subplot(2,2,1)
w = [1:8];
plot(w-.24,impmlr,".",w-.08,impmlrstep,".",w+.08,imprfr,".",...
    w+.24,implsb,".",MarkerSize=20)
axis([0.5 8.5 0 1.05]);
ylabel('Normalized Importance','FontSize',12);
title('BATS Training Predictor Importance')
%axis('square')
legend('MLR','Step MLR','RFR','LSB', 'Location', 'best'); 
legend('boxoff')
xticks([1:1:8]);
xticklabels({'Depth','MLD','PAR','Temp','NO_3','PO_4','POC','Chl a'})
xtickangle(45)




