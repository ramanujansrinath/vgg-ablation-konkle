clear; clc; close all;

if exist('cached_images.mat','file')
    load('cached_images.mat')
    disp('loaded cached images and features')
else
    % --- Image loading and pixel-feature extraction ---
    % Load natural/texform image pairs from the .mat file, resize both sets to a
    % common spatial resolution, force 3-channel format, and compute a struct of
    % pixel-level features for every image.
    disp('getting images')
    load('texform_img.mat','texform_img');
    imsize = 256;
    tt_img = nan(imsize,imsize,3,length(texform_img));
    ttex_img = nan(imsize,imsize,3,length(texform_img));
    parfor tt=1:length(texform_img)
        img = imresize(texform_img(tt).img,[imsize imsize]);
        tt_img(:,:,:,tt) = double(repmat(img,1,1,3));
        tt_params_st(tt) = getPixelFeatures(tt_img(:,:,:,tt),tt);
    
        img = imresize(texform_img(tt).tex_img,[imsize imsize]);
        ttex_img(:,:,:,tt) = double(repmat(img,1,1,3));
        ttex_params_st(tt) = getPixelFeatures(ttex_img(:,:,:,tt),tt);
    end
    
    % --- Feature matrix assembly ---
    % Drop uninformative or redundant struct fields, flatten each struct array
    % into a numeric matrix, and prepend the high-level semantic labels
    % (size/animacy) stored in the image metadata as the first two feature columns.
    tt_params_st = rmfield(tt_params_st,{'exempName' 'color_hist_red' 'color_hist_green' 'aspect_ratio' 'boundingBox_size'});
    tt_params_mat = reshape(struct2array(tt_params_st),numel(fields(tt_params_st)),[])';
    tt_params = [[texform_img.big]' [texform_img.animal]' tt_params_mat];
    
    ttex_params_st = rmfield(ttex_params_st,{'exempName' 'color_hist_red' 'color_hist_green' 'aspect_ratio' 'boundingBox_size'});
    ttex_params_mat = reshape(struct2array(ttex_params_st),numel(fields(tt_params_st)),[])';
    ttex_params = [[texform_img.big]' [texform_img.animal]' ttex_params_mat];
    
    % --- NaN handling and joint min-max normalization ---
    % Zero-fill missing feature values, then scale every feature column to [0,1]
    % using the min/max computed jointly across both image sets so natural and
    % texform images share the same numeric reference frame.
    idx = isnan(ttex_params) | isnan(tt_params);
    tt_params(idx) = 0;
    ttex_params(idx) = 0;
    
    pmin = repmat(min([tt_params; ttex_params]),size(tt_params,1),1);
    pmax = repmat(max([tt_params; ttex_params]),size(tt_params,1),1);
    tt_params = (tt_params-pmin)./(pmax-pmin);
    ttex_params = (ttex_params-pmin)./(pmax-pmin);
    
    save('cached_images.mat','tt_img','tt_params','ttex_img','ttex_params')
end

%%
if input('Redo feature ablation analysis? (1/0): ')
    %%
    % --- Network initialization ---
    % Load the pretrained VGG16 network and specify which max-pooling layers
    % (by index, e.g. pool1/pool3/pool5) will be used as activation sources.
    disp('loading network')
    net = vgg16;
    layers = [1 3 5];
    
    %%
    % --- Ablation analysis: natural images ---
    % For each target layer, extract central-unit activations, reduce dimensionality,
    % and run the random-ablation decoding sweep; intermediate results are saved
    % immediately so the expensive computation is not lost if the script is interrupted.
    disp('analyzing natural images')
    tt_randabl = cell(1,length(layers));
    tt_signeg = cell(1,length(layers));
    for ll=1:length(layers)
        disp(['analyzing layer' num2str(layers(ll))])
        [tt_randabl{ll},tt_signeg{ll}] = getabl(net,layers(ll),tt_img,tt_params);
    end
    
    save('cached_analysis_dims.mat','tt_randabl','tt_signeg','-append')
    
    %%
    % --- Ablation analysis: texform images ---
    % Repeat the identical pipeline for texform images and append results to the
    % existing cache file so both image types end up in the same .mat.
    disp('analyzing texform images')
    ttex_randabl = cell(1,length(layers));
    ttex_signeg = cell(1,length(layers));
    for ll=1:length(layers)
        disp(['analyzing layer' num2str(layers(ll))])
        [ttex_randabl{ll},ttex_signeg{ll}] = getabl(net,layers(ll),ttex_img,ttex_params);
    end
    
    save('cached_analysis_dims.mat','ttex_randabl','ttex_signeg','-append')
else
    load('cached_analysis_dims.mat','tt_randabl','tt_signeg','ttex_randabl','ttex_signeg')
end

%%
% --- Plotting: natural images ---
% Lay out a 2-row × N-layer grid of error-bar plots; features whose decoding
% accuracy shows a significant negative slope as more dimensions are ablated
% go in the top row (informative dimensions), all others in the bottom row.
figure('color','w','position',[86,477,1126,801],'Name','Natural')
layers = [1 3 5];
ha = tight_subplot(2,length(layers),0.05,0.05,0.05);
ha = reshape(ha,length(layers),2)';
col = {'b' 'r' 'k'};
for ll=1:length(layers)
    for tt=1:size(tt_randabl{1},3)
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
% --- Plotting: texform images ---
% Identical layout and row-assignment logic as the natural-image figure above,
% applied to the texform decoding trajectories.
figure('color','w','position',[1226,477,1126,801],'Name','Texforms')
ha = tight_subplot(2,length(layers),0.05,0.05,0.05);
ha = reshape(ha,length(layers),2)';

for ll=1:length(layers)
    for tt=1:size(tt_randabl{1},3)
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
% --- getabl: per-layer activation extraction + ablation sweep ---
% Extract activations from the named VGG16 pooling layer, crop to the 8×8
% central spatial units across all channels, reduce to 20 PCA dimensions,
% run randablate_and_decode, and flag each feature as having a significantly
% negative decoding trajectory (i.e., ablation hurts performance).
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

    % Classify each feature's trajectory by fitting a linear trend and testing
    % whether the slope is significantly negative (p < 0.01), indicating that
    % the ablated dimensions carry decodable information.
    signegabl = false(1,size(tt_params,2));
    for tt=1:size(tt_params,2)
        mdl = fitlm(1:10,tt_randabl(:,1,tt)');
        stat = table2array(mdl.Coefficients);
        if stat(2,1) < 0 && stat(2,4) < 0.01 % significant negative slope
            signegabl(tt) = true;
        end
    end
end

% --- randablate_and_decode: random ablation decoding sweep ---
% For each ablation size N (1 to nUnits-2), draw 100 random subsets of N
% dimensions to remove, decode all image features from the remaining
% dimensions, and record the mean ± SEM correlation across folds.
% Trajectories that are entirely negative are sign-flipped to positive.
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

    % Sign-flip trajectories that are entirely below zero so they display
    % as positive curves (assumes negative correlation reflects inverted but
    % valid decoding rather than chance).
    for tt=1:size(dec_traj,3)
        if all(dec_traj(:,1,tt)<0)
            dec_traj(:,1,tt) = abs(dec_traj(:,1,tt));
        end
    end
end

% --- decodeFn: cross-validated linear decoding ---
% Decode each target feature from X using 10-fold cross-validated OLS
% regression (no intercept), returning the Pearson correlation between
% held-out predictions and ground-truth values across all folds.
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
