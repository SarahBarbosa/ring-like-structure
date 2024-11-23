clear all;
load 'AllStarsLimitedAge2020.dat'
s = AllStarsLimitedAge2020;
le=length(s);
d=size(s);
%X1=s(:,end); 
% calculate Matrix

%pex=inputdlg('What is the parameter as a function of an another parameter (see next question)? 1-X, 2-Y, 3-Z, 4-vsini, 5-longitude, 6-latitude, 7-absolute magnitude 8-visual magnitude, 9-age, 10-FeH, 11-distance, 12-mass, 13-XY galaticplane, 14-Stroemgren(b-y)','Parameters'); %do not click enter, press Ok
%pe=str2num(pex{:});

%sig=sortrows(s,1);
%x=sig(:,1);
%y=sig(:,3);

sig=sortrows(s,3); % rotation plane
y=sig(:,2);
x=sig(:,3);

velocity=sig(:,4);
vsini=sig(:,8);%columns: 1-X, 2-Y, 3-Z, 4-vsini, 5-longitude, 6-latitude, 7-absolute magnitude (same distance)
%8-visual magnitude (aparent magnitude = Johnson system), 9-age, 10-FeH, 11-distance, 12-mass, 13-XY galatic plane
% XZ meridian plane
% YZ rotation plane

tcx=inputdlg('What is the increment value?','Increment'); %do not click enter, press Ok
tc=str2num(tcx{:});

Bx=inputdlg('What is the resampling number?','Resampling');
B=str2num(Bx{:});

ulx=inputdlg('Upper Limit of the interval of parameter?','Upper Limit (Square Matrix)');
ul=str2num(ulx{:});

interval=1;
ttx=(-ul:interval*tc:ul)';
tty=(-ul:interval*tc:ul)';
%ttz=(floor(z(1)):interval*tc:ceil(z(end)))';

Nx=length(ttx)-1;
Ny=length(tty)-1;
%Nvsini=length(s)-1;
h=zeros(Nx,Ny);
p=zeros(Nx,Ny);

meanbootstrap=zeros(Nx,Ny);
percentile25boot=zeros(Nx,Ny);
percentile50boot=zeros(Nx,Ny);
percentile75boot=zeros(Nx,Ny);

meanoriginal=zeros(Nx,Ny);
percentile25=zeros(Nx,Ny);
percentile50=zeros(Nx,Ny);
percentile75=zeros(Nx,Ny);

vmedian=zeros(Nx,Ny);
vmedianc=zeros(Nx,Ny);

v25=zeros(Nx,Ny);
cc25=zeros(Nx,Ny);

v50=zeros(Nx,Ny);
cc50=zeros(Nx,Ny);

v75=zeros(Nx,Ny);
cc75=zeros(Nx,Ny);

v25b=zeros(Nx,Ny);
cc25b=zeros(Nx,Ny);

v50b=zeros(Nx,Ny);
cc50b=zeros(Nx,Ny);

v75b=zeros(Nx,Ny);
cc75b=zeros(Nx,Ny);

proporc=zeros(B,1);
% parte onde faremos a estatistica da amostra original
% Galactic plane

for j=1:Nx
   
    for i=1:Ny
    condition=(ttx(j)<=x & x<ttx(j+1) & tty(i)<=y & y<tty(i+1)); %X vs. Y
    count=length(vsini(condition));
    countfXY(i,j)=count;
    
    if(count>=20)

    meanoriginal(isnan(meanoriginal))=0;
    vmedian(i,j)=median(velocity(condition));
    meanoriginal(i,j)=mean(vsini(condition)); % here vsini=Vmag
    percentile25(isnan(percentile25))=0;
    percentile50(isnan(percentile50))=0;
    percentile75(isnan(percentile75))=0;
    
    vcond25=sortrows(vsini(condition));
    % it can be used this condition or not because it already is sorted
    percentile25(i,j)=prctile(vcond25,25);
    cc25=round(count/4); % if n=1.9, round(1.9)=2, ceil(1.9)=2 and floor(1.9)=1
    if cc25==0
        v25(i,j)=0;
    else
        v25(i,j)=velocity(cc25);
    end
    vcond50=sortrows(vsini(condition));
    percentile50(i,j)=prctile(vcond50,50);
    cc50=round(count/2);
    if cc50==0
        v50(i,j)=0;
    else
        v50(i,j)=velocity(cc50);
    end
    
    vcond75=sortrows(vsini(condition));
    percentile75(i,j)=prctile(vcond75,75);
    cc75=round(3*count/4);
    if cc75==0
        v75(i,j)=0;
    else
        v75(i,j)=velocity(cc75);
    end

    end;
    end
end
% parte onde faremos a estatistica da reamostra bootstrap

vsini_boot=zeros(le,B);

vsini_boot=bootrsp(vsini,B);
for k=1:B
vsini=vsini_boot(:,k);

for j=1:Nx
   
    for i=1:Ny
    condition=(ttx(j)<=x & x<ttx(j+1) & tty(i)<=y & y<tty(i+1)); %X vs. Y
    count=length(vsini(condition));
    countfXY(i,j)=count;
    
    if(count>=20)
    meanbootstrap(isnan(meanbootstrap))=0;
    vmedianc(i,j)=median(velocity(condition));
    meanbootstrap(i,j)=mean(vsini(condition));
    percentile25boot(isnan(percentile25boot))=0;
    percentile50boot(isnan(percentile50boot))=0;
    percentile75boot(isnan(percentile75boot))=0;
    percentile25boot(i,j)=prctile(vsini(condition),25);
    cc25b=round(count/4); % if n=1.9, round(1.9)=2, ceil(1.9)=2 and floor(1.9)=1
    if cc25b==0
        v25b(i,j)=0;
    else
        v25b(i,j)=velocity(cc25b);
    end
    
    percentile50boot(i,j)=prctile(vsini(condition),50);
    cc50b=round(count/2); % if n=1.9, round(1.9)=2, ceil(1.9)=2 and floor(1.9)=1
    if cc50b==0
        v50b(i,j)=0;
    else
        v50b(i,j)=velocity(cc50b);
    end
    
    percentile75boot(i,j)=prctile(vsini(condition),75);
    cc75b=round(3*count/4); % if n=1.9, round(1.9)=2, ceil(1.9)=2 and floor(1.9)=1
    if cc75b==0
        v75b(i,j)=0;
    else
        v75b(i,j)=velocity(cc75b);
    end
   
    end;
    end
end

end

% new figure 2016
%figure;

fig1=subplot(2,2,1);
niter=1;
method='spline';
ima1=interp2(meanoriginal,niter,method);
imagesc(ima1);colorbar;colormap(jet);
xlabel('Z (pc)');
ylabel('Y (pc)');
c = colorbar;
ylabel(c,{'\langle V_{mag} \rangle (km/s)'; 'of original sample'})
[cmin,cmax] = caxis;
caxis([0,cmax]);
xticklabels = -ul:50:ul;
xticks = linspace(1, size(ima1, 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
yticklabels = -ul:50:ul;
yticks = linspace(1, size(ima1, 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))

subplot(2,2,2);
niter=1;
method='spline';
ima5=interp2(percentile25,niter,method);
imagesc(ima5);colorbar;colormap(jet);
xlabel('Z (pc)');
ylabel('Y (pc)');
c = colorbar;
ylabel(c,{'V_{mag}'; 'of original sample (q=1/4)'})
[cmin,cmax] = caxis;
caxis([0,cmax]);
xticklabels = -ul:50:ul;
xticks = linspace(1, size(ima5, 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
yticklabels = -ul:50:ul;
yticks = linspace(1, size(ima5, 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))

subplot(2,2,3);
niter=1;
method='spline';
ima9=interp2(percentile50,niter,method);
imagesc(ima9);colorbar;colormap(jet);
xlabel('Z (pc)');
ylabel('Y (pc)');
c = colorbar;
ylabel(c,{'V_{mag}'; 'of original sample (q=1/2)'})
[cmin,cmax] = caxis;
caxis([0,cmax]);
xticklabels = -ul:50:ul;
xticks = linspace(1, size(ima9, 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
yticklabels = -ul:50:ul;
yticks = linspace(1, size(ima9, 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))

subplot(2,2,4);
niter=1;
method='spline';
ima13=interp2(percentile75,niter,method);
imagesc(ima13);colorbar;colormap(jet);
xlabel('Z (pc)');
ylabel('Y (pc)');
c = colorbar;
ylabel(c,{'V_{mag}'; 'of original sample (q=3/4)'})
[cmin,cmax] = caxis;
caxis([0,cmax]);
xticklabels = -ul:50:ul;
xticks = linspace(1, size(ima13, 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
yticklabels = -ul:50:ul;
yticks = linspace(1, size(ima13, 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))

set(gcf, 'PaperPosition', [0 0 16 10]); %x_width=15cm y_width=10cm
set(gcf, 'PaperSize', [15 10]);
saveas(gcf,'C:\Paper2016-2017-2018-2020\plasmaMoreRecently\Ring-like structureMNRAS\figXY.pdf');

% final figures

% figure 2016 bootstrap
figure;

fig2=subplot(2,2,1);
niter=1;
method='spline';
ima2=interp2(meanbootstrap,niter,method);
imagesc(ima2);colorbar;colormap(jet);
xlabel('Z (pc)');
ylabel('Y (pc)');
xticklabels = -ul:50:ul;
xticks = linspace(1, size(ima2, 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
yticklabels = -ul:50:ul;
yticks = linspace(1, size(ima2, 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))
c = colorbar;
ylabel(c,{'\langle V_{mag} \rangle(km/s)'; 'of bootstrap resampling'})
[cmin,cmax] = caxis;
caxis([0,cmax]);

subplot(2,2,2);
niter=1;
method='spline';
ima6=interp2(percentile25boot,niter,method);
imagesc(ima6);colorbar;colormap(jet);
xlabel('Z (pc)');
ylabel('Y (pc)');
xticklabels = -ul:50:ul;
xticks = linspace(1, size(ima6, 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
yticklabels = -ul:50:ul;
yticks = linspace(1, size(ima6, 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))
c = colorbar;
ylabel(c,{'V_{mag}'; 'of bootstrap resampling (q=1/4)'})
[cmin,cmax] = caxis;
caxis([0,cmax]);

subplot(2,2,3);
niter=1;
method='spline';
ima10=interp2(percentile50boot,niter,method);
imagesc(ima10);colorbar;colormap(jet);
xlabel('Z (pc)');
ylabel('Y (pc)');
xticklabels = -ul:50:ul;
xticks = linspace(1, size(ima10, 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
yticklabels = -ul:50:ul;
yticks = linspace(1, size(ima10, 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))
c = colorbar;
ylabel(c,{'V_{mag}';'of bootstrap resampling (q=1/2)'})
[cmin,cmax] = caxis;
caxis([0,cmax]);

subplot(2,2,4);
niter=1;
method='spline';
ima14=interp2(percentile75boot,niter,method);
imagesc(ima14);colorbar;colormap(jet);
xlabel('Z (pc)');
ylabel('Y (pc)');
xticklabels = -ul:50:ul;
xticks = linspace(1, size(ima14, 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
yticklabels = -ul:50:ul;
yticks = linspace(1, size(ima14, 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))
c = colorbar;
ylabel(c,{'V_{mag}'; 'of bootstrap resampling (q=3/4)'})
[cmin,cmax] = caxis;
caxis([0,cmax]);

set(gcf, 'PaperPosition', [0 0 16 10]); %x_width=15cm y_width=10cm
set(gcf, 'PaperSize', [15 10]);

saveas(gcf,'C:\Paper2016-2017-2018-2020\plasmaMoreRecently\Ring-like structureMNRAS\figXYbs.pdf');

%%%% new condition Vmag e Vsini
figure;


fig1=subplot(2,2,1);
niter=1;
method='spline';
ima1=interp2(vmedian,niter,method);
imagesc(ima1);colorbar;colormap(jet);
xlabel('Z (pc)');
ylabel('Y (pc)');
c = colorbar;
ylabel(c,{'median (v sini) (km/s)'; 'of original sample'})
[cmin,cmax] = caxis;
caxis([0,cmax]);
xticklabels = -ul:50:ul;
xticks = linspace(1, size(ima1, 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
yticklabels = -ul:50:ul;
yticks = linspace(1, size(ima1, 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))

subplot(2,2,2);
niter=1;
method='spline';
ima5=interp2(v25,niter,method);
imagesc(ima5);colorbar;colormap(jet);
xlabel('Z (pc)');
ylabel('Y (pc)');
c = colorbar;
ylabel(c,{'v sini (km/s)'; 'for V_{mag}(q=1/4)'})
[cmin,cmax] = caxis;
caxis([0,cmax]);
xticklabels = -ul:50:ul;
xticks = linspace(1, size(ima5, 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
yticklabels = -ul:50:ul;
yticks = linspace(1, size(ima5, 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))

subplot(2,2,3);
niter=1;
method='spline';
ima9=interp2(v50,niter,method);
imagesc(ima9);colorbar;colormap(jet);
xlabel('Z (pc)');
ylabel('Y (pc)');
c = colorbar;
ylabel(c,{'v sini (km/s)'; 'for V_{mag}(q=1/2)'})
[cmin,cmax] = caxis;
caxis([0,cmax]);
xticklabels = -ul:50:ul;
xticks = linspace(1, size(ima9, 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
yticklabels = -ul:50:ul;
yticks = linspace(1, size(ima9, 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))

subplot(2,2,4);
niter=1;
method='spline';
ima13=interp2(v75,niter,method);
imagesc(ima13);colorbar;colormap(jet);
xlabel('Z (pc)');
ylabel('Y (pc)');
c = colorbar;
ylabel(c,{'v sini (km/s)'; 'for V_{mag}(q=3/4)'})
[cmin,cmax] = caxis;
caxis([0,cmax]);
xticklabels = -ul:50:ul;
xticks = linspace(1, size(ima13, 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
yticklabels = -ul:50:ul;
yticks = linspace(1, size(ima13, 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))

%%%% new condition Vmag e Vsini bootstrap
%%%% Figures are the same because is not make the bootstrap of velocity
figure;


fig1=subplot(2,2,1);
niter=1;
method='spline';
ima1=interp2(vmedianc,niter,method);
imagesc(ima1);colorbar;colormap(jet);
xlabel('Z (pc)');
ylabel('Y (pc)');
c = colorbar;
ylabel(c,{'median (v sini) (km/s)'; 'of bootstrap resampling'})
[cmin,cmax] = caxis;
caxis([0,cmax]);
xticklabels = -ul:50:ul;
xticks = linspace(1, size(ima1, 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
yticklabels = -ul:50:ul;
yticks = linspace(1, size(ima1, 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))

subplot(2,2,2);
niter=1;
method='spline';
ima5=interp2(v25b,niter,method);
imagesc(ima5);colorbar;colormap(jet);
xlabel('Z (pc)');
ylabel('Y (pc)');
c = colorbar;
ylabel(c,{'v sini (km/s)'; 'for V_{mag}(q=1/4)'})
[cmin,cmax] = caxis;
caxis([0,cmax]);
xticklabels = -ul:50:ul;
xticks = linspace(1, size(ima5, 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
yticklabels = -ul:50:ul;
yticks = linspace(1, size(ima5, 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))

subplot(2,2,3);
niter=1;
method='spline';
ima9=interp2(v50b,niter,method);
imagesc(ima9);colorbar;colormap(jet);
xlabel('Z (pc)');
ylabel('Y (pc)');
c = colorbar;
ylabel(c,{'v sini (km/s)'; 'for V_{mag}(q=1/2)'})
[cmin,cmax] = caxis;
caxis([0,cmax]);
xticklabels = -ul:50:ul;
xticks = linspace(1, size(ima9, 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
yticklabels = -ul:50:ul;
yticks = linspace(1, size(ima9, 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))

subplot(2,2,4);
niter=1;
method='spline';
ima13=interp2(v75b,niter,method);
imagesc(ima13);colorbar;colormap(jet);
xlabel('Z (pc)');
ylabel('Y (pc)');
c = colorbar;
ylabel(c,{'v sini (km/s)'; 'for V_{mag}(q=3/4)'})
[cmin,cmax] = caxis;
caxis([0,cmax]);
xticklabels = -ul:50:ul;
xticks = linspace(1, size(ima13, 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
yticklabels = -ul:50:ul;
yticks = linspace(1, size(ima13, 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))

