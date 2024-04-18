##### #####




###### Ch 3 Coefficient vectors, matrix and array ##############
library(fda)

# Create basis system 
basisobj = create.constant.basis(rangeval)
basisobj = create.monomial.basis(rangeval,nbasis)
basisobj = create.fourier.basis(rangeval,nbasis,period)
basisobj = create.bspline.basis(rangeval,nbasis,norder,breaks)
plot(basisobj)

# example for fourier basis
daybasis65 = create.fourier.basis(c(0,365),65) # K = 65
# daybasis65 = create.fourier.basis(c(0,365),65)
zerobasis = create.fourier.basis(rangeval,basis,dropind = 1)
zerobasis = daybasis65[2:65]

## example for bspline 
splinebasis = create.bspline.basis(c(0,10),13) # interior knots: 9; order = 4
plot(splinebasis)

basis2 = create.bspline.basis(c(0,2*pi), 5, 2) # (range, K, order)
basis3 = create.bspline.basis(c(0,2*pi), 6, 3)
basis4 = create.bspline.basis(c(0,2*pi), 7, 4)
norder(basis2)

conbasis = create.constant.basis(c(0,1))
monbasis = create.monomial.basis(c(0,1),4)

# Generic functions
methods(class = "basisfd")
# set values of basis function
basismatrix = eval.basis(tvec, mybasis)
# derivatives
Dbasismatrix = eval.basis(tvec, mybasis,1)

##### Ch4 Build functional data objects #######
library(fda)

daybasis65 = create.fourier.basis(c(0,365),65)
tempfd = fd(coefmat, daybasis65)

## Labels for the data 
# label in order: domain; replication; range.
fdnames = list("Age (years)", "Child", "Height (cm)")

fdnames[[1]] = "Age (years"
fdnames[[2]] = "Child"
fdnames[[3]] = "Height (cm)"

# for specific names in replication or range:
station = vector("list", 35) # create a new empty list
station[[ 1]] = "St. Johns"
#...
station[[35]] = "Resolute"
fdnames = list("Day",
               "Weather Station" = station,
               "Mean temperature (deg C)")

## Methods for functional data obj
tstFn0 <- fd(c(-1, 2), create.bspline.basis(rangeval = c(0,10),norder=2))
# ts <- create.bspline.basis(c(-1,2),norder = 2)
# fd(coef, basisobj,fdname)
plot(tstFn0)


# sum,difference,power and pointwise product on fdobj but not basisobj
plot(tstFn0^2)
plot(tstFn0*tstFn0)
plot(tstFn0^(-1))
# mean(fdobj)

# evaluate values of func at specified values of argument t
thatvec = eval.fd(tvec, thawfd)
# and derivatives
D2thatvec = eval.fd(tvec, thawfd, 2)

# example of temp
daytime = (1:365) - 0.5
JJindex = c(182:365,1:181)
tempdat = daily$tempav[JJindex,] # 365*35

tempbasis = create.fourier.basis(c(0,365),65)
tempfd = smooth.basis(daytime,tempdat,tempbasis)$fd
tempfd$fdnames = list("Day (July 2 to June 30) ",
                      "Weather Station",
                      "Mean temperature (deg. C) ")
plot(tempfd,lty = 1)

# example 
basis13 = create.bspline.basis(c(0,10), 13)
tvec = seq(0,1,len=13)
sinecoef = sin(2*pi*tvec)
sinefd = fd(sinecoef, basis13, list("t","","f(t)"))
op = par(cex=1.2)
plot(sinefd, lwd=2)
points(tvec*10, sinecoef, lwd=2)
par(op)

# exercise 
splinebasis <- create.bspline.basis(c(0,1),23)
plot(splinebasis)

fdcoef1 <- rnorm(23)
fdcoef2 <- sin(2*pi*seq(0,1,len = 23))

fdobj1 <- fd(fdcoef1,splinebasis)
fdobj2 <- fd(fdcoef2,splinebasis)

tvec <- seq(0,1,len = 51)
fdval1 <- eval.fd(tvec,fdobj1)
fdval2 <- eval.fd(tvec,fdobj2)

plot(fdobj1)
points(seq(0,1,len =23), fdcoef1)
points(seq(0,1,len =51), fdval1,pch =19)

plot(fdobj2)
points(seq(0,1,len =23), sin(2*pi*seq(0,1,len = 23)))
points(seq(0,1,len =51), fdval2,pch =19)

######### Ch5 Smoothing: Computing curves from noisy data #####

## Regression-based smoothing
attach(growth)
heightbasis12 = create.bspline.basis(c(1,18), 12, 6)
basismat = eval.basis(age, heightbasis12)
heightcoef = lsfit(basismat, heightmat,
                   intercept=FALSE)$coef

heightList = smooth.basis(age, heightmat,heightbasis12)

age = growth$age
heightbasismat = eval.basis(age, heightbasis12)
y2cMap = solve(crossprod(heightbasismat),
               t(heightbasismat))
                
## Smoothing with roughness penalty
Rmat = eval.penalty(tempbasis, harmaccelLfd)
# harmaccelLfd = Lfd(nderiv = 3, bwtlist= list(0,omega^2,0))
norder = 6
nbasis = length(age) + norder - 2 # minus 2 because of interior knots
heightbasis = create.bspline.basis(c(1,18),
                                   nbasis, norder, age)
heightfdPar = fdPar(heightbasis, 4, 0.01)
#fdPar(basisobj,m or Lfd,lambda): used as a basis obj 
heightfd = smooth.basis(age, hgtm,
                        heightfdPar)$fd

# choose lambda by gcv
loglam <- seq(-6,0,0.25)
gcvsave <- vector("numeric",length(loglam))
dfsave <- gcvsave
for (i in 1:length(loglam)){
  lambdai <- 10^loglam[i]
  hgtfdPari <- fdPar(heightbasis,4,lambdai)
  fdsmooth <- smooth.basis(age,hgtm,hgtfdPari)
  gcvi = fdsmooth$gcv
  dfi = fdsmooth$df
  gcvsave[i] = sum(gcvi)
  dfsave[i] = dfi
}

## Case study: The log precipitation data
logprecav = CanadianWeather$dailyAv[dayOfYearShifted, ,'log10precip']

dayrange <- c(0,365)
daybasis <- create.fourier.basis(dayrange,365)
Lcoef = c(0,(2*pi / diff(dayrange))^2,0)
harmaccelLfd = vec2Lfd(Lcoef, dayrange)

loglam = seq(4,9,0.25)
nlam = length(loglam)
dfsave = rep(NA,nlam)
gcvsave = rep(NA,nlam)
for (ilam in 1:nlam){
  cat(paste("log10 lambda =",loglam[ilam],'\n'))
  lambda = 10^loglam[ilam]
  fdParobj = fdPar(daybasis, harmaccelLfd, lambda)
  smoothlist = smooth.basis(day.5, logprecav,
                            fdParobj)
  dfsave[ilam] = smoothlist$df
  gcvsave[ilam] = sum(smoothlist$gcv)
}

lambda = 1e6
fdParobj = fdPar(daybasis, harmaccelLfd, lambda)
logprec.fit = smooth.basis(day.5,logprecav,fdParobj)
logprec.fd = logprec.fit$fd
fdnames = list("Day (July 1 to June 30)",
               "Weather Station" = CanadianWeather$place,
               "Log 10 Precipitation (mm)")
logprec.fd$fdnames = fdnames
plot(logprec.fd)
plotfit.fd(logprecav, day.5, logprec.fd)

## Constrains on curve estimated: positive, monotone, density
# positive constrain
lambda = 1e3
WfdParobj = fdPar(daybasis, harmaccelLfd, lambda)
VanPrec = CanadianWeather$dailyAv[
  dayOfYearShifted, 'Vancouver', 'Precipitation.mm']
VanPrecPos = smooth.pos(day.5, VanPrec, WfdParobj)
Wfd = VanPrecPos$Wfdobj

Wfd$fdnames = list("Day (July 1 to June 30)",
                   "Weather Station" = CanadianWeather$place,
                   "Log 10 Precipitation (mm)")
plot(Wfd)

precfit = exp(eval.fd(day.5, Wfd))
plot(day.5, VanPrec, type="p", cex=1.2,
     xlab="Day (July 1 to June 30)",
     ylab="Millimeters",
     main="Vancouver’s Precipitation")
lines(day.5, precfit,lwd=2)

# monotone constrain

# Newborn Baby's Tibia: no data available
Wbasis = create.bspline.basis(c(1,n), nbasis)
Wfd0 = fd(matrix(0,nbasis,1), Wbasis)
WfdPar = fdPar(Wfd0, 2, 1e-4)
result = smooth.monotone(day, tib, WfdPar)
Wfd = result$Wfd
beta = result$beta
dayfine = seq(1,n,len=151)
tibhat = beta[1] + beta[2]*eval.monfd(dayfine ,Wfd)
Dtibhat = beta[2]*eval.monfd(dayfine, Wfd, 1)
D2tibhat = beta[2]*eval.monfd(dayfine, Wfd, 2)

# Berkeley female growth data
wbasis = create.bspline.basis(c(1,18), 35, 6, age)
growfdPar = fdPar(wbasis, 3, 10^(-0.5))

growthMon = smooth.monotone(age, hgtf, growfdPar)
Wfd = growthMon$Wfd
betaf = growthMon$beta
hgtfhatfd = growthMon$yhatfd

# Probability density constrain
RegPrec = ReginaPrecip[ReginaPrecip >= 2 & ReginaPrecip <= 45]
RegPrec = sort(RegPrec)
N=212
Wknots = RegPrec[round(N*seq(1/N,1,len=11),0)]
Wnbasis = length(Wknots) + 2
Wbasis = create.bspline.basis(range(RegPrec),13,4,Wknots)

Wlambda = 1e-1
WfdPar = fdPar(Wbasis, 2, Wlambda)
densityList = density.fd(RegPrec, WfdPar)
Wfd = densityList$Wfdobj
C = densityList$C  

Zfine = seq(RegPrec[1],RegPrec[N],len=201)
Wfine = eval.fd(Zfine, Wfd)
Pfine = exp(Wfine)/C

# 
logprecmat = eval.fd(day.5, logprec.fd)
logprecres = logprecav - logprecmat
# across stations
logprecvar1 = apply(logprecres^2, 1, sum)/35
# across time
logprecvar2 = apply(logprecres^2, 2, sum)/(365-12)

logstddev.fd = smooth.basis(day.5,
                            log(logprecvar1)/2, fdParobj)$fd
logprecvar1fit = exp(eval.fd(day.5, logstddev.fd))
plot(logprecvar1fit)

########## Ch 6 Description of functional data ##########

# mean and variance between observations (replication)
meanlogprec = mean.fd(logprec.fd)
stddevlogprec = std.fd(logprec.fd)

# bivariate covariance function according to time
# variance-covariance plot
logprecvar.bifd = var.fd(logprec.fd)

weektime = seq(0,365,length=53)
logprecvar_mat = eval.bifd(weektime, weektime,
                           logprecvar.bifd)
persp(weektime, weektime, logprecvar_mat,
      theta=-45, phi=25, r=3, expand = 0.5,
      ticktype='detailed',
      xlab="Day (July 1 to June 30)",
      ylab="Day (July 1 to June 30)",
      zlab="variance(log10 precip)")
contour(weektime, weektime, logprecvar_mat)

tempprecbifd = var.fd(tempfd, logprec.fd)

day5time = seq(0,365,5)
logprec.varmat = eval.bifd(day5time, day5time,
                           logprecvar.bifd)
contour(day5time, day5time, logprec.varmat,
        xlab="Day (July 1 to June 30)",
        ylab="Day (July 1 to June 30)", lwd=2,
        labcex=1)

# Functional probes
inprod()

# Phase-plane plot

#
dayvec = seq(0,365,len=101)
xivec = exp(20*cos(2*pi*(dayvec-197)/365))
xibasis = create.bspline.basis(c(0,365),13)
xifd = smooth.basis(dayvec, xivec, xibasis)$fd
tempLmat = inprod(tempbasis, xifd)
precLmat = inprod(precbasis, xifd)

# example of CI of Prince Ruper's Log precipitation
lambda = 1e6;
fdParobj = fdPar(daybasis, harmaccelLfd, lambda)
logprecList= smooth.basis(day.5, logprecav, fdParobj)
logprec.fd = logprecList$fd
fdnames = list("Day (July 1 to June 30)",
               "Weather Station" = CanadianWeather$place,
               "Log 10 Precipitation (mm)")
logprec.fd$fdnames = fdnames

logprecmat = eval.fd(day.5, logprec.fd)
logprecres = logprecav - logprecmat
logprecvar = apply(logprecres^2, 1, sum)/(35-1)
lambda = 1e8
resfdParobj = fdPar(daybasis, harmaccelLfd, lambda)
logvar.fit = smooth.basis(day.5, log(logprecvar),
                          resfdParobj)
logvar.fd = logvar.fit$fd
varvec = exp(eval.fd(daytime, logvar.fd))
SigmaE = diag(as.vector(varvec))

y2cMap = logprecList$y2cMap
c2rMap = eval.basis(day.5, daybasis)
Sigmayhat = c2rMap %*% y2cMap %*% SigmaE %*%
  t(y2cMap) %*% t(c2rMap)
logprec.stderr = sqrt(diag(Sigmayhat))
logprec29 = eval.fd(day.5, logprec.fd[29])
plot(logprec.fd[29], lwd=2, ylim=c(0.2, 1.3))
lines(day.5, logprec29 + 2*logprec.stderr,
      lty=2, lwd=2)
lines(day.5, logprec29 - 2*logprec.stderr,
      lty=2, lwd=2)
points(day.5, logprecav[,29])

######### Ch 7 Functional PCA and Canonical components analysis ######

# example
logprec.pcalist = pca.fd(logprec.fd, 2)
print(logprec.pcalist$values)
plot.pca.fd(logprec.pcalist)

# varimax
logprec.rotpcalist = varmx.pca.fd(logprec.pcalist)
plot.pca.fd(logprec.rotpcalist)

# PCA of log precipitation resuduals
logprecres.fd = smooth.basis(day.5, logprecres,
                             fdParobj)$fd
plot(logprecres.fd, lwd=2, col=1, lty=1, cex=1.2,
     xlim=c(0,365), ylim=c(-0.07, 0.07),
     xlab="Day", ylab="Residual (log 10 mm)")

# Multicariate PCA: example handwriting
fdarange = c(0, 2300)
fdabasis = create.bspline.basis(fdarange, 105, 6)
fdatime = seq(0, 2300, len=1401)
fdafd =
  smooth.basis(fdatime, handwrit, fdabasis)$fd
fdafd$fdnames[[1]] = "Milliseconds"
fdafd$fdnames[[2]] = "Replications"
fdafd$fdnames[[3]] = list("X", "Y")

nharm = 3
fdapcaList = pca.fd(fdafd, nharm)
plot.pca.fd(fdapcaList)
fdarotpcaList = varmx.pca.fd(fdapcaList)
plot.pca.fd(fdarotpcaList)

fdaeig = fdapcaList$values
neig = 12
x = matrix(1,neig-nharm,2)
x[,2] = (nharm+1):neig
y = log10(fdaeig[(nharm+1):neig])
c = lsfit(x,y,int=FALSE)$coef
par(mfrow=c(1,1),cex=1.2)
plot(1:neig, log10(fdaeig[1:neig]), "b",
     xlab="Eigenvalue Number",
     ylab="Log10 Eigenvalue")
lines(1:neig, c[1]+ c[2]*(1:neig), lty=2)


fdameanfd  = mean(fdafd)
fdameanmat = eval.fd(fdatime, fdameanfd)

#  evaluate the harmonics

harmfd  = fdarotpcaList$harm
harmmat = eval.fd(fdatime, harmfd)

fdapointtime = seq(0,2300,len=201)
fdameanpoint = eval.fd(fdapointtime, fdameanfd)
harmpointmat = eval.fd(fdapointtime, harmfd)

fac = 0.1
harmplusmat = array(0,c(201,3,2))
harmminsmat = array(0,c(201,3,2))
for (j in 1:3) {
  harmplusmat[,j,] = fdameanpoint[,1,] + fac*harmpointmat[,j,]
  harmminsmat[,j,] = fdameanpoint[,1,] - fac*harmpointmat[,j,]
}

j=3
plot(fdameanmat[,1,1]-0.035,  fdameanmat[,1,2], "l", lwd=2,
     xlim=c(-0.075,0.075), ylim=c(-0.04, 0.04),
     xlab="", ylab="")
lines(harmplusmat[,j,1]-0.035, harmplusmat[,j,2], lty=2)
lines(harmminsmat[,j,1]-0.035, harmminsmat[,j,2], lty=2)
j=2
lines(fdameanmat[,1,1]+0.035,  fdameanmat[,1,2],  lty=1, lwd=2)
lines(harmplusmat[,j,1]+0.035, harmplusmat[,j,2], lty=2)
lines(harmminsmat[,j,1]+0.035, harmminsmat[,j,2], lty=2)


# CCA
ccafdPar = fdPar(daybasis, 2, 5e6)
ncan = 3
ccalist = cca.fd(tempfd, logprec.fd, ncan,
                 ccafdPar, ccafdPar)

ccawt.temp = ccalist$ccawtfd1
ccawt.logprec = ccalist$ccawtfd2
corrs = ccalist$ccacorr

ccascr.temp = ccalist$ccavar1
ccascr.logprec = ccalist$ccavar2

#### ch 8 Registration: alighing features for samples of curves ######
attach(growth)
wbasis = create.bspline.basis(c(1,18), 35, 6, age)
growfdPar = fdPar(wbasis, 3, 10^(-0.5))

growthMon = smooth.monotone(age, hgtf, growfdPar)
Wfd = growthMon$Wfd
betaf = growthMon$beta
hgtfhatfd = growthMon$yhatfd

# 8.3 landmark registration 
accelfdUN = deriv.fd(hgtfhatfd,2)
accelmeanfdUN = mean.fd(accelfdUN)

PGSctr = rep(0,10)
agefine = seq(1,18,len=101)
par(mfrow=c(1,1), ask=TRUE)
for (icase in 1:10){ # take 10 from 54
  accveci = predict(accelfdUN[icase], agefine)
  plot(agefine,accveci,"l", ylim=c(-6,4),
       xlab="Year", ylab="Height Accel.",
       main=paste("Case",icase))
  lines(c(1,18),c(0,0),lty=2)
  PGSctr[icase] = locator(1)$x
}
PGSctrmean = mean(PGSctr)

wbasisLM = create.bspline.basis(c(1,18), 4, 3,
                                c(1,PGSctrmean,18))
WfdLM = fd(matrix(0,4,1),wbasisLM)
WfdParLM = fdPar(WfdLM,1,1e-12)

regListLM = landmarkreg(accelfdUN, ximarks =  PGSctr, 
                        x0marks =  rep(PGSctrmean,length(PGSctr)),
                        x0lim = NULL,WfdPar = WfdParLM)
accelfdLM = regListLM$regfd
accelmeanfdLM = mean.fd(accelfdLM)
warpfdLM = regListLM$warpfd
WfdLM = regListLM$Wfd

# 8.4 Continuous Registration 
wbasisCR = create.bspline.basis(c(1,18), 15, 5)
Wfd0CR = fd(matrix(0,15,54),wbasisCR) # 10 from 54
WfdParCR = fdPar(Wfd0CR, 1, 1)
regList = register.fd(mean.fd(accelfdLM),
                      accelfdLM, WfdParCR)
accelfdCR = regList$regfd
warpfdCR = regList$warpfd
WfdCR = regList$Wfd
plot(accelfdCR)
lines(accelfdLM)
plot(accelmeanfdLM,lty = 2)
lines(mean.fd(accelfdCR))

# 8.5 Decomposition into amplitude and phase sums of squares
AmpPhasList = AmpPhaseDecomp(accelfdUN, accelfdCR,
                             warpfdCR, c(3,18))
MS.amp      = AmpPhasList$MS.amp
MS.pha      = AmpPhasList$MS.pha
RSQRCR      = AmpPhasList$RSQR
CCR         = AmpPhasList$C

# 

####### Ch9 Functional linear model for scalar response ##########
library(fda)

# a scalar response for log annual precipitation 
# daily: 365 * 35; annualprec: 1 * 35 (total sum of a station in a year)
annualprec = log10(apply(daily$precav,2,sum))

tempbasis =create.fourier.basis(c(0,365),65)
tempSmooth=smooth.basis(day.5,daily$tempav,tempbasis)
tempfd =tempSmooth$fd




