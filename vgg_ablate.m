clear; clc; close all;

disp('getting images')
load('texform_img.mat','texform_img');
imsize = 256;
tt_img = nan(imsize,imsize,3,length(texform_img));
ttex_img = nan(imsize,imsize,3,length(texform_img));
for tt=1:length(texform_img)
    img = imresize(texform_img(tt).img,[imsize imsize]);
    tt_img(:,:,:,tt) = double(repmat(img,1,1,3));
    tt_params_st(tt) = getPixelFeatures(tt_img(:,:,:,tt),tt); %#ok<SAGROW>

    img = imresize(texform_img(tt).tex_img,[imsize imsize]);
    ttex_img(:,:,:,tt) = double(repmat(img,1,1,3));
    ttex_params_st(tt) = getPixelFeatures(ttex_img(:,:,:,tt),tt); %#ok<SAGROW>
end
 
tt_params_st = rmfield(tt_params_st,{'exempName' 'color_hist_red' 'color_hist_green' 'aspect_ratio' 'boundingBox_size'});
tt_params_mat = reshape(struct2array(tt_params_st),numel(fields(tt_params_st)),[])';
tt_params = [[texform_img.big]' [texform_img.animal]' tt_params_mat];

ttex_params_st = rmfield(ttex_params_st,{'exempName' 'color_hist_red' 'color_hist_green' 'aspect_ratio' 'boundingBox_size'});
ttex_params_mat = reshape(struct2array(ttex_params_st),numel(fields(tt_params_st)),[])';
ttex_params = [[texform_img.big]' [texform_img.animal]' ttex_params_mat];

idx = isnan(ttex_params) |isnan(tt_params);
tt_params(idx) = 0;
ttex_params(idx) = 0;

pmin = repmat(min([tt_params; ttex_params]),120,1);
pmax = repmat(max([tt_params; ttex_params]),120,1);
tt_params = (tt_params-pmin)./(pmax-pmin);
ttex_params = (ttex_params-pmin)./(pmax-pmin);

clearvars -except tt_img tt_params ttex_img ttex_params

%%
disp('loading network')
net = vgg16;
layers = [1 3 5];

%%
disp('analyzing natural images')
tt_randabl = cell(1,length(layers));
tt_signeg = cell(1,length(layers));
for ll=1:length(layers)
    disp(['analyzing layer' num2str(layers(ll))])
    [tt_randabl{ll},tt_signeg{ll}] = getabl(net,layers(ll),tt_img,tt_params);
end

save('cached_analysis.mat','tt_randabl','tt_signeg')

%%
disp('analyzing texform images')
ttex_randabl = cell(1,length(layers));
ttex_signeg = cell(1,length(layers));
for ll=1:length(layers)
    disp(['analyzing layer' num2str(layers(ll))])
    [ttex_randabl{ll},ttex_signeg{ll}] = getabl(net,layers(ll),ttex_img,ttex_params);
end

save('cached_analysis.mat','ttex_randabl','ttex_signeg','-append')

%%
figure('color','w','position',[86,477,1126,801],'Name','Natural')
ha = tight_subplot(2,length(layers),0.05,0.05,0.05);
ha = reshape(ha,length(layers),2)';
col = {'b' 'r' 'k'};
for ll=1:length(layers)
    for tt=1:size(tt_params,2)
        if tt>2; lw = 1; lc = col{3}; else; lw = 2; lc = col{tt}; end
        if tt_signeg{ll}(tt)
            errorbar(ha(1,ll),tt_randabl{ll}(:,1,tt),tt_randabl{ll}(:,2,tt),'linewidth',lw,'color',lc); hold(ha(1,ll),'on')
        else
            errorbar(ha(2,ll),tt_randabl{ll}(:,1,tt),tt_randabl{ll}(:,2,tt),'linewidth',lw,'color',lc); hold(ha(2,ll),'on')
        end
    end

    % legendStr = cellfun(@mat2str,num2cell(find(tt_signeg{ll})),'UniformOutput',false);
    fixPlot(ha(1,ll),[0 11],[-0.3 0.6],'dimensions ablated','decoding accuracy',[1 5 10 15],-1:0.25:1,['layer ' num2str(layers(ll))])
    % legend('Location','eastoutside')
    
    fixPlot(ha(2,ll),[0 11],[-0.3 0.6],'dimensions ablated','decoding accuracy',[1 5 10 15],-1:0.25:1,['layer ' num2str(layers(ll))])
end

%%
figure('color','w','position',[1226,477,1126,801],'Name','Texforms')
ha = tight_subplot(2,length(layers),0.05,0.05,0.05);
ha = reshape(ha,length(layers),2)';

for ll=1:length(layers)
    for tt=1:size(tt_params,2)
        if tt>2; lw = 1; lc = col{3}; else; lw = 2; lc = col{tt}; end
        if ttex_signeg{ll}(tt)
            errorbar(ha(1,ll),ttex_randabl{ll}(:,1,tt),ttex_randabl{ll}(:,2,tt),'linewidth',lw,'color',lc); hold(ha(1,ll),'on')
        else
            errorbar(ha(2,ll),ttex_randabl{ll}(:,1,tt),ttex_randabl{ll}(:,2,tt),'linewidth',lw,'color',lc); hold(ha(2,ll),'on')
        end
    end

    % legendStr = cellfun(@mat2str,num2cell(find(ttex_signeg{ll})),'UniformOutput',false);
    fixPlot(ha(1,ll),[0 11],[-0.3 0.6],'dimensions ablated','decoding accuracy',[1 5 10 15],-1:0.25:1,['layer ' num2str(layers(ll))])
    % legend('Location','eastoutside')
    
    fixPlot(ha(2,ll),[0 11],[-0.3 0.6],'dimensions ablated','decoding accuracy',[1 5 10 15],-1:0.25:1,['layer ' num2str(layers(ll))])
end

%%
function [tt_randabl,signegabl] = getabl(net,layerNum,tt_img,tt_params)
    disp('... getting activations')
    tt_act = activations(net,tt_img,['pool' num2str(layerNum)]);
    filtSize = size(tt_act(:,:,:,1));
    centralUnits = 8;
    unitIds = (1+filtSize(1))/2 - (centralUnits-1)/2 : (1+filtSize(1))/2 + (centralUnits-1)/2;

    tt_act = tt_act(unitIds,unitIds,:,:);
    tt_act = double(reshape(tt_act,numel(tt_act)/size(tt_params,1),size(tt_params,1)));
    tt_red = pca(tt_act,'NumComponents',20);
    
    tt_randabl = randablate_and_decode(tt_params,tt_red);

    signegabl = false(1,size(tt_params,2));
    for tt=1:size(tt_params,2)
        mdl = fitlm(1:10,tt_randabl(:,1,tt)');
        stat = table2array(mdl.Coefficients);
        if stat(2,1) < 0 && stat(2,4) < 0.01 % significant negative slope
            signegabl(tt) = true;
        end
    end
end

function dec_traj = randablate_and_decode(y,X)
    nFold = 100;
    nUnits = min(size(X,2),12);
    dec_traj = nan(nUnits-2,2,size(y,2));
    for nn=1:nUnits-2
        disp(['...... ablating ' num2str(nn) ' units'])
        corr_y = nan(nFold,size(y,2));
        for ff=1:nFold
            fprintf('|')
            ablidx = randperm(nUnits,nn);
            idx = 1:nUnits;
            idx(ablidx) = [];
            Xabl = X(:,idx);
            corr_y(ff,:) = decodeFn(y,Xabl);
        end
        fprintf('\n')
        dec_traj(nn,1,:) = mean(corr_y);
        dec_traj(nn,2,:) = std(corr_y)./sqrt(nFold);
    end

    for tt=1:size(dec_traj,3)
        if all(dec_traj(:,1,tt)<0)
            dec_traj(:,1,tt) = abs(dec_traj(:,1,tt));
        end
    end
end

function corr_y = decodeFn(y,X)    
    nFold = 10;
    cvp = cvpartition(size(y,1),'KFold',nFold);

    corr_y = nan(1,size(y,2));
    for jj=1:size(y,2) % for each feature
        Ypred = nan(1,size(y,1));
        for ff=1:nFold
            Xtrain = X(cvp.training(ff),:);
            Ytrain = y(cvp.training(ff),jj);
            beta = regress(Ytrain,Xtrain);
            Ypred(cvp.test(ff)) = X(cvp.test(ff),:) * beta;
        end
        corr_y(jj) = corr(Ypred',y(:,jj));
    end
end

