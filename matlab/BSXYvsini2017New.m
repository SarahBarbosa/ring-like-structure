clear all;
load 'NewTableFwithoutLimitAgeAndMass.dat'
s = NewTableFwithoutLimitAgeAndMass;
le=length(s);
d=size(s);
%X1=s(:,end); 
% calculate Matrix

%pex=inputdlg('What is the parameter as a function of an another parameter (see next question)? 1-X, 2-Y, 3-Z, 4-vsini, 5-longitude, 6-latitude, 7-absolute magnitude 8-visual magnitude, 9-age, 10-FeH, 11-distance, 12-mass, 13-XY galaticplane, 14-Stroemgren(b-y)','Parameters'); %do not click enter, press Ok
%pe=str2num(pex{:});

sig=sortrows(s,1);
x=sig(:,1);
y=sig(:,2);

%sig=sortrows(s,3); % rotation plane
%y=sig(:,2);
%x=sig(:,3);
vsini=sig(:,4);%columns: 1-X, 2-Y, 3-Z, 4-vsini, 5-longitude, 6-latitude, 7-absolute magnitude
%8-visual magnitude, 9-age, 10-FeH, 11-distance, 12-mass, 13-XY galatic plane

DPtotal=std(vsini);

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
meanoriginal=zeros(Nx,Ny);
meanbootstrap=zeros(Nx,Ny);
percentile25boot=zeros(Nx,Ny);
percentile50boot=zeros(Nx,Ny);
percentile75boot=zeros(Nx,Ny);
stdevs=zeros(Nx,Ny);
X=zeros(Nx,Ny);
boot.mean=zeros(Nx,Ny);
boot.se=zeros(Nx,Ny);
ci1=zeros(Nx,Ny);
ci2=zeros(Nx,Ny);
shape=zeros(Nx,Ny);

percentile25=zeros(Nx,Ny);
percentile50=zeros(Nx,Ny);
percentile75=zeros(Nx,Ny);

cl = 95; % Nível de confiança em porcentagem;
t = 2.0930; % valor utilizado no bootstrap-t;
z = 1.960; % valor utilizado no bootstrap padrão ;

proporc=zeros(B,1);
% parte onde faremos a estatistica da amostra original
% Galactic plane

for j=1:Nx
   
    for i=1:Ny
    condition=(ttx(j)<=x & x<ttx(j+1) & tty(i)<=y & y<tty(i+1)); %X vs. Y
    count=length(vsini(condition));
    countfXY(i,j)=count;
    
    if(count>=20)

    ZZ = randsample(vsini(condition),10,true);           % Sample of 10 w/replacement
    meanoriginal(isnan(meanoriginal))=0;
    meanoriginal(i,j)=median(vsini(condition));

trimpct = 20;                                 % Total trim percentage
bootout = bootstrp(B,@(ZZ) trimmean2(ZZ, ... % Takes 5000 bootstrap samples
          trimpct),vsini(condition));               %   and computes trimmed mean
boot.true = trimmean2(vsini(condition),trimpct);    % True trimmed mean
boot.mean(i,j) = mean(bootout);                    % Mean of bootstrap trimmed means
boot.bias = boot.mean-boot.true;              % Bootstrap estimate of bias
boot.se(i,j) = std(bootout);                       % Bootstrap estimate of SE

ci = bootci(B,{@(ZZ) trimmean2(ZZ, ...  % Computes BCa bootstrap 95%
    trimpct),vsini(condition)},'type','cper','alpha',.01);        %   CI for trimmed mean
ci1(i,j)=ci(1);
ci2(i,j)=ci(2);
    end;
    end
end

largura=ci2-ci1;
shape=(ci2-boot.mean)./(boot.mean-ci1);
shape(isnan(shape))=0;

%%%%%%%%%%%%%%%%%%%%%%%
figure;
subplot(2,2,1);
niter=1;
method='spline';
ima1=interp2(meanoriginal,niter,method);
%surf(ima1);
%hold on
imagesc(ima1);colorbar;colormap(jet);

xlabel('Z (pc)');
ylabel('Y (pc)');
c = colorbar();
ylabel(c,{'median (v sini) (km/s) of';'original sample'})
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
ima2=interp2(boot.mean,niter,method);
%surf(ima2);
%hold on
imagesc(ima2);colorbar;colormap(jet);

xlabel('Z (pc)');
ylabel('Y (pc)');
c = colorbar;
ylabel(c,{'median (v sini) (km/s) of'; 'bootstrap resampling'})
[cmin,cmax] = caxis;
caxis([0,cmax]);
xticklabels = -ul:50:ul;
xticks = linspace(1, size(ima1, 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
yticklabels = -ul:50:ul;
yticks = linspace(1, size(ima1, 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))


subplot(2,2,3);
niter=1;
method='spline';
ima3=interp2(largura,niter,method);
%surf(ima3);
%hold on
imagesc(ima3);colorbar;colormap(jet);

xlabel('Z (pc)');
ylabel('Y (pc)');
c = colorbar;
ylabel(c,{'Length Index'})
[cmin,cmax] = caxis;
caxis([0,cmax]);
xticklabels = -ul:50:ul;
xticks = linspace(1, size(ima3, 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
yticklabels = -ul:50:ul;
yticks = linspace(1, size(ima3, 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))


subplot(2,2,4);
niter=1;
method='spline';
ima4=interp2(shape,niter,method);
%surf(ima4);
%hold on
imagesc(ima4);colorbar;colormap(jet);

xlabel('Z (pc)');
ylabel('Y (pc)');
xticklabels = -ul:50:ul;
xticks = linspace(1, size(ima4, 2), numel(xticklabels));
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels)
yticklabels = -ul:50:ul;
yticks = linspace(1, size(ima4, 1), numel(yticklabels));
set(gca, 'YTick', yticks, 'YTickLabel', flipud(yticklabels(:)))
c = colorbar;
ylabel(c,{'Shape Index'})
[cmin,cmax] = caxis;
caxis([0,cmax]);


set(gcf, 'PaperPosition', [0 0 16 10]); %x_width=15cm y_width=10cm
set(gcf, 'PaperSize', [15 10]);

%saveas(gcf,'C:\Paper2016-2017\plasmaMoreRecently\NewFigure2017\figXYbsF.pdf');

%------------------------
% Shuffled image
%------------------------

X=boot.mean;
img=X;
blockSize = 1;

nRows = size(img, 1) / blockSize;
nCols = size(img, 2) / blockSize;
scramble = mat2cell(img, ones(1, nRows) * blockSize, ones(1, nCols) * blockSize, size(img, 3));
scramble = cell2mat(reshape(scramble(randperm(nRows * nCols)), nRows, nCols));

X=scramble;
figure;
%2D auto-correlation 
[n m]=size(boot.mean);
% Divide by the size for normalization

B=abs(fftshift(ifft2(fft2(boot.mean).*conj(fft2(boot.mean)))))./(n*m);

subplot(1,2,1);
fig1=surf(B);
shading interp;
xlabel('X (pc)','FontSize',14);
ylabel('Y (pc)','FontSize',14);

% Create ylabel
zlabel('Amplitude','FontSize',14);

title('Spatial Autocorrelation Function','FontSize',14);
months1=[-150; -75; 0; 75; 150];
set(gca,'YTickLabel',months1)
months2=[-150; -75; 0; 75; 150];
set(gca,'XTickLabel',months2)

%2D auto-correlation 
[n m]=size(X);
% Divide by the size for normalization

B=abs(fftshift(ifft2(fft2(X).*conj(fft2(X)))))./(n*m);

subplot(1,2,2);
fig1=surf(B);
shading interp;
xlabel('X (pc)','FontSize',14);
ylabel('Y (pc)','FontSize',14);

% Create ylabel
zlabel('Amplitude','FontSize',14);

title('Spatial Autocorrelation Function','FontSize',14);

set(gcf, 'PaperPosition', [0 0 16 10]); %x_width=15cm y_width=10cm
set(gcf, 'PaperSize', [15 10]);
months1=[-150; -75; 0; 75; 150];
set(gca,'YTickLabel',months1)
months2=[-150; -75; 0; 75; 150];
set(gca,'XTickLabel',months2)

%saveas(fig1,'C:\Paper2016-2017\plasmaMoreRecently\NewFigure2017\figAUTOXYbsF.pdf');
