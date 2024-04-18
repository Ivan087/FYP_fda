#### load data and package ####
### used for AIC BIC ######
setwd("C:/Users/wujia/school/Year 4/semester 2/FYP")
data = read.csv("./data/weather.csv")
library(FuncNN)
library(fda.usc)

# # Clearing backend
# K <- backend()
# K$clear_session()
# options(warn = -1)
# 
# # Setting seeds
# set.seed(2020)
# tensorflow::set_random_seed(1,disable_gpu = FALSE)

#### data transformation(to matrix form) ####
#timepts;pressure; temp; d_temp;humidity;sun;wind;prec
timepts = 1:365 - 0.5
dayrange = c(0,365)
pressure = matrix(rep(NA,365*24),365,24)
for(j in 1:24){
for (i in 1:365) {
  pressure[i,j] = data[(j - 1) * 365 + i,2]
}
}

temp = matrix(rep(NA,365*24),365,24)
for(j in 1:24){
  for (i in 1:365) {
    temp[i,j] = data[(j - 1) * 365 + i,3]
  }
}

d_temp = matrix(rep(NA,365*24),365,24)
for(j in 1:24){
  for (i in 1:365) {
    d_temp[i,j] = data[(j - 1) * 365 + i,4]
  }
}

humidity = matrix(rep(NA,365*24),365,24)
for(j in 1:24){
  for (i in 1:365) {
    humidity[i,j] = data[(j - 1) * 365 + i,5]
  }
}

sun = matrix(rep(NA,365*24),365,24)
for(j in 1:24){
  for (i in 1:365) {
    sun[i,j] = data[(j - 1) * 365 + i,6]
  }
}

wind = matrix(rep(NA,365*24),365,24)
for(j in 1:24){
  for (i in 1:365) {
    wind[i,j] = data[(j - 1) * 365 + i,7]
  }
}

prec = matrix(rep(NA,365*24),365,24)
for(j in 1:24){
  for (i in 1:365) {
    prec[i,j] = data[(j - 1) * 365 + i,8]
  }
}


#### basis number choosing function ####
basis_num_choice = function(nbasis,inputdata){

daybasis = create.fourier.basis(c(0,365), nbasis)

Lcoef = c(0,(2 * pi/diff(dayrange))^2,0)
harmaccelLfd = vec2Lfd(Lcoef, dayrange)

loglam = seq(-3,3,0.25)
nlam = length(loglam)
dfsave = rep(NA,nlam)
gcvsave = rep(NA,nlam)

for (ilam in 1:nlam) {
  lambda = 10^loglam[ilam]
  fdParobj = fdPar(daybasis, harmaccelLfd, lambda)
  smoothlist = smooth.basis(timepts, inputdata,fdParobj)
  dfsave[ilam] = smoothlist$df
  gcvsave[ilam] = sum(smoothlist$gcv)
}

lambda = 10^(loglam[which.min(gcvsave)])
fdParobj = fdPar(daybasis, harmaccelLfd, lambda)
temp.fit = smooth.basis(timepts,inputdata,fdParobj)
temp.fd = temp.fit$fd

fd = fd(temp.fit$fd$coefs,daybasis)
pre = eval.fd(timepts,fd)

AIC = mean(2*nbasis + 365 * log(colSums((pre - inputdata)^2)/365,base = exp(1)))
BIC = mean(log(365,exp(1))*nbasis + 365 * log(colSums((pre - inputdata)^2)/365,base = exp(1)))
return(rbind(AIC,BIC))
}
#### data smoothing####
ABIC_value = matrix(nrow = 2,ncol = 36)
for (i in 1:36) {
  ABIC_value[,i] = basis_num_choice(i*10 + 5,inputdata = temp)
  cat(paste("basis numer:",i*10+5,"\n"))
}
plot(seq(15,365,10),colSums(ABIC_value),main = "Temp",ylab = "AIC + BIC value")
which.min(colSums(ABIC_value))

ABIC_value_2 = matrix(nrow = 2,ncol = 50)
for (i in 1:50) {
  ABIC_value_2[,i] = basis_num_choice(i*2 + 150,inputdata = temp)
  cat(paste("basis numer:",i*2+150,"\n"))
}
plot(seq(152,250,2),colSums(ABIC_value_2),main = "Temp",ylab = "AIC + BIC value")
which.min(colSums(ABIC_value_2))

ABIC_value_H = matrix(nrow = 2,ncol = 36)
for (i in 1:36) {
  ABIC_value_H[,i] = basis_num_choice(i*10 + 5,inputdata = humidity)
  cat(paste("basis numer:",i*10+5,"\n"))
}
plot(seq(15,365,10),colSums(ABIC_value_H),main = "Humidity",ylab = "AIC + BIC value")

ABIC_value_H2 = matrix(nrow = 2,ncol = 50)
for (i in 1:50) {
  ABIC_value_H2[,i] = basis_num_choice(i*2 + 120,inputdata = humidity)
  cat(paste("basis numer:",i*2+120,"\n"))
}
plot(seq(122,220,2),colSums(ABIC_value_H2),main = "Humidity",ylab = "AIC + BIC value")
which.min(colSums(ABIC_value_H2))

plot(seq(15,365,10),ABIC_value[1,],main = "Wind",ylab = "AIC value")
plot(seq(15,365,10),ABIC_value[2,],main = "Wind",ylab = "BIC value")



which.min(BIC_value[1,])
which.min(BIC_value[2,])

AIC_value = c(rep(NA,50))
BIC_value = c(rep(NA,50))
for (i in 1:50) {
  AIC_value[i] = basis_num_choice(i*2 + 253,inputdata = wind)[1,]
  # BIC_value[i] = basis_num_choice(i*2 + 1,inputdata = wind)[2,]
  cat(paste("basis numer:",i*2-1,"\n"))
}
plot(seq(3,101,2),AIC_value,main = "Wind")
min(AIC_value)
which.min(AIC_value)

plot(seq(3,101,2),BIC_value)
min(BIC_value)
which.min(BIC_value)
#test 
nbasis = 365
dayrange = c(0,365)
daybasis = create.fourier.basis(dayrange, nbasis)

Lcoef = c(0,(2 * pi/diff(dayrange))^2,0)
harmaccelLfd = vec2Lfd(Lcoef, dayrange)

loglam = seq(-5,5,0.25)
nlam = length(loglam)
dfsave = rep(NA,nlam)
gcvsave = rep(NA,nlam)

for (ilam in 1:nlam) {
  cat(paste("log10 lambda =",loglam[ilam],"\n"))
  lambda = 10^loglam[ilam]
  fdParobj = fdPar(daybasis, harmaccelLfd, lambda)
  smoothlist = smooth.basis(timepts, temp,fdParobj)
  dfsave[ilam] = smoothlist$df
  gcvsave[ilam] = sum(smoothlist$gcv)
}

plot(loglam,gcvsave)
lambda = 10^(loglam[which.min(gcvsave)])
fdParobj = fdPar(daybasis, harmaccelLfd, lambda)
temp.fit = smooth.basis(timepts,temp,fdParobj)
temp.fd = temp.fit$fd

fd = fd(temp.fit$fd$coefs,daybasis)
pre = eval.fd(timepts,fd)

RSE = sqrt(colSums((pre - temp)^2)/363)
RSE
#AIC = 2K + n*ln(SSR/n)
AIC = 2*nbasis + 365 * log(colSums((pre - temp)^2)/365,base = exp(1))
mean(AIC)
BIC = log(365,base = exp(1))*nbasis + 
  365 * log(colSums((pre - temp)^2)/365,base = exp(1))
mean(BIC)

plot(temp.fd)
plotfit.fd(temp, timepts, temp.fd)

######## Model building ##########
#x
spline_basis <- create.fourier.basis(c(0,365),65)
press_fd =  Data2fd(timepts, pressure, spline_basis)
temp_fd =  Data2fd(timepts, temp, spline_basis)
dtemp_fd =  Data2fd(timepts, d_temp, spline_basis)
hum_fd =  Data2fd(timepts, humidity, spline_basis)
sun_fd =  Data2fd(timepts, sun, spline_basis)
wind_fd =  Data2fd(timepts, wind, spline_basis)
#y
prec_fd =  Data2fd(timepts, prec, spline_basis)

#coeffictient
func_cov_1 = press_fd$coefs
func_cov_2 = temp_fd$coefs
func_cov_3 = hum_fd$coefs
func_cov_4 = sun_fd$coefs
func_cov_5 = wind_fd$coefs
func_cov_6 = dtemp_fd$coefs

# coe array
cov_data = array(dim = c(nbasis, dim(pressure)[2], 5))
cov_data[,,1] = func_cov_1
cov_data[,,2] = func_cov_2
cov_data[,,3] = func_cov_3
cov_data[,,4] = func_cov_4
cov_data[,,5] = func_cov_5



#### FLM(scalar resp) ####
#y
y = colMeans(prec)

# train test split(only for testing)
train_index = sample(1:24,size = 15)
y_train <- y[train_index]
y_test <- y[-train_index]
cov_data_train <- cov_data[,train_index,]
cov_data_test <- cov_data[,-train_index,]
#
spline_basis <- create.fourier.basis(c(0,365),3)
xfdlist <- list(press_fd[train_index,],temp_fd[train_index,],
                hum_fd[train_index,],sun_fd[train_index,],
                wind_fd[train_index,])

betalist <- list(spline_basis,spline_basis,spline_basis,
                 spline_basis,spline_basis)

xfdlist <- list(press_fd[train_index,],wind_fd[train_index,])

betalist <- list(spline_basis,spline_basis)

fit.flm <- fRegress(y_train,xfdlist,betalist)
fit.flm.cv <- fRegress.CV(y_train,xfdlist,betalist)

xfdnew <- list(press_fd[-train_index,],temp_fd[-train_index,],
               hum_fd[-train_index,],sun_fd[-train_index,],
               wind_fd[-train_index,])

pred <- predict.fRegress(fit.flm,newdata = xfdnew)
(mse <- mean((y_test - pred)^2))

### FNN(scalar response) ####
nbasis = 5
fit <- fnn.fit(resp = y_train, func_cov = cov_data_train,
               basis_choice = c('fourier','fourier','fourier',
                                'fourier','fourier'),
               num_basis = c(nbasis,nbasis,nbasis,nbasis,nbasis),
               hidden_layers = 2,neurons_per_layer = c(32,32),
               activations_in_layers = c("sigmoid",'linear'),
               domain_range = list(c(0,365),c(0,365)),
               epochs = 100)
#### FNN(functional response)####
y_train = t(prec_fd$coefs)[train_index,]
y_test <- eval.fd(timepts,precip_fd)[,-train_index]

fit <- fnn.fit(resp = y_train, func_cov = cov_data_train,
               basis_choice = c('fourier','fourier'),
               num_basis = c(65,65),
               hidden_layers = 2,neurons_per_layer = c(64,64),
               activations_in_layers = c("sigmoid",'linear'),
               domain_range = list(c(0,365),c(0,365)),
               epochs = 100)
pred_coef <- fnn.predict(model = fit, func_cov = cov_data_test,
                         basis_choice = c('fourier','fourier'),
                         num_basis = c(nbasis,nbasis),
                         domain_range = list(c(0,365),c(0,365)))
pred_fd <- fd(t(pred_coef),spline_basis)

pred <- eval.fd(timepts,pred_fd)
(mse <- sum((pred-y_test)^2)/(365*23))


fit.cv <- fnn.cv(nfolds = 10,resp = y, func_cov = cov_data,
                 basis_choice = c('fourier','fourier'),
                 num_basis = c(65,65),
                 hidden_layers = 2,neurons_per_layer = c(64,64),
                 activations_in_layers = c("sigmoid",'linear'),
                 domain_range = list(c(0,365),c(0,365)),
                 epochs = 100)

pred <- fnn.predict(model = fit, func_cov = cov_data_test,
                    basis_choice = c('fourier','fourier'),
                    num_basis = c(nbasis,nbasis),
                    domain_range = list(c(0,365),c(0,365)))
(mse <- mean((y_test - pred)^2))