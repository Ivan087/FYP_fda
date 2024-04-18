#### load data and package ####
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


#### scale for data matrix ####
pressure <- t(scale(t(pressure)))
temp <- t(scale(t(temp)))
humidity <- t(scale(t(humidity)))
sun <- t(scale((t(sun))))
wind <- t(scale(t(wind)))
# prec <- t(scale(t(prec)))
#### tune function ####
library(keras)
library(pbapply)
tune_fnn = function (tune_list, resp, func_cov, scalar_cov = NULL, basis_choice, 
          domain_range, batch_size = 32, decay_rate = 0, nfolds = 5, 
          cores = 4, raw_data = FALSE) 
{
  if (is.vector(resp) == TRUE) {
    output_size = 1
  }
  else {
    output_size = ncol(resp)
  }
  if (raw_data == TRUE) {
    dim_check = length(func_cov)
  }
  else {
    dim_check = dim(func_cov)[3]
  }
  if (raw_data == TRUE) {
    dat = func_cov
    temp_tensor = array(dim = c(31, nrow(dat[[1]]), length(dat)))
    for (t in 1:length(dat)) {
      curr_func = dat[[t]]
      curr_domain = domain_range[[1]]
      basis_setup = create.bspline.basis(rangeval = c(curr_domain[1], 
                                                      curr_domain[2]), nbasis = 31, norder = 4)
      time_points = seq(curr_domain[1], curr_domain[2], 
                        length.out = ncol(curr_func))
      temp_fd = Data2fd(time_points, t(curr_func), basis_setup)
      temp_tensor[, , t] = temp_fd$coefs
    }
    func_cov = temp_tensor
  }
  if (output_size == 1) {
    tune_func = function(x, nfolds, resp, func_cov, scalar_cov, 
                         basis_choice, domain_range, batch_size, decay_rate, 
                         raw_data) {
      tensorflow::set_random_seed(1, disable_gpu = FALSE)
      colnames(x) <- NULL
      rownames(x) <- NULL
      model_results = fnn.cv(nfolds, resp, func_cov = func_cov, 
                             scalar_cov = scalar_cov, basis_choice = basis_choice, 
                             num_basis = as.numeric(as.character((x[(current_layer + 
                                                                       1):(length(basis_choice) + current_layer)]))), 
                             hidden_layers = current_layer, neurons_per_layer = as.numeric(as.character(x[(length(basis_choice) + 
                                                                                                             current_layer + 1):((length(basis_choice) + 
                                                                                                                                    current_layer) + current_layer)])), activations_in_layers = as.character(x[1:current_layer]), 
                             domain_range = domain_range, epochs = as.numeric(as.character(x[((length(basis_choice) + 
                                                                                                 current_layer) + current_layer) + 1])), loss_choice = "mse", 
                             metric_choice = list("mean_squared_error"), 
                             val_split = as.numeric(as.character(x[((length(basis_choice) + 
                                                                       current_layer) + current_layer) + 2])), learn_rate = as.numeric(as.character(x[((length(basis_choice) + 
                                                                                                                                                          current_layer) + current_layer) + 4])), patience_param = as.numeric(as.character(x[((length(basis_choice) + 
                                                                                                                                                                                                                                                 current_layer) + current_layer) + 3])), early_stopping = TRUE, 
                             print_info = FALSE, batch_size = batch_size, 
                             decay_rate = decay_rate, raw_data = FALSE)
      list_returned <- list(MSPE = model_results$MSPE$Overall_MSPE, 
                            num_basis = as.numeric(as.character((x[(current_layer + 
                                                                      1):(length(basis_choice) + current_layer)]))), 
                            hidden_layers = current_layer, neurons_per_layer = as.numeric(as.character(x[(length(basis_choice) + 
                                                                                                            current_layer + 1):((length(basis_choice) + 
                                                                                                                                   current_layer) + current_layer)])), activations_in_layers = as.character(x[1:current_layer]), 
                            epochs = as.numeric(as.character(x[((length(basis_choice) + 
                                                                   current_layer) + current_layer) + 1])), val_split = as.numeric(as.character(x[((length(basis_choice) + 
                                                                                                                                                     current_layer) + current_layer) + 2])), patience_param = as.numeric(as.character(x[((length(basis_choice) + 
                                                                                                                                                                                                                                            current_layer) + current_layer) + 3])), learn_rate = as.numeric(as.character(x[((length(basis_choice) + 
                                                                                                                                                                                                                                                                                                                               current_layer) + current_layer) + 4])))
      K <- backend()
      K$clear_session()
      
      return(list_returned)
    }
    Errors = list()
    All_Errors = list()
    Grid_List = list()
    for (i in 1:length(tune_list$num_hidden_layers)) {
      current_layer = tune_list$num_hidden_layers[i]
      df = expand.grid(rep(list(tune_list$neurons), tune_list$num_hidden_layers[i]), 
                       stringsAsFactors = FALSE)
      df2 = expand.grid(rep(list(tune_list$num_basis), 
                            length(basis_choice)), stringsAsFactors = FALSE)
      df3 = expand.grid(rep(list(tune_list$activation_choice), 
                            tune_list$num_hidden_layers[i]), stringsAsFactors = FALSE)
      colnames(df2)[length(basis_choice)] <- "Var2.y"
      colnames(df3)[i] <- "Var2.z"
      pre_grid = expand.grid(df$Var1, Var2.y = df2$Var2.y, 
                             Var2.z = df3$Var2.z, tune_list$epochs, tune_list$val_split, 
                             tune_list$patience, tune_list$learn_rate)
      combined <- unique(merge(df, pre_grid, by = "Var1"))
      combined2 <- unique(merge(df2, combined, by = "Var2.y"))
      final_grid <- suppressWarnings(unique(merge(df3, 
                                                  combined2, by = "Var2.z")))
      Grid_List[[i]] = final_grid
      results = pbapply(final_grid, 1, tune_func, nfolds = nfolds, 
                        resp = resp, func_cov = func_cov, scalar_cov = scalar_cov, 
                        basis_choice = basis_choice, domain_range = domain_range, 
                        batch_size = batch_size, decay_rate = decay_rate, 
                        raw_data = FALSE)
      MSPE_vals = c()
      for (u in 1:length(results)) {
        MSPE_vals[u] <- as.vector(results[[u]][1])
      }
      All_Errors[[i]] = results
      Errors[[i]] = results[[which.min(do.call(c, MSPE_vals))]]
      cat("\n")
      message(paste0("Done tuning for: ", current_layer, 
                     " hidden layers."))
    }
    MSPE_after = c()
    for (i in 1:length(tune_list$num_hidden_layers)) {
      MSPE_after[i] = Errors[[i]]$MSPE
    }
    best = which.min(MSPE_after)
    return(list(Parameters = Errors[[best]], All_Information = All_Errors, 
                Best_Per_Layer = Errors, Grid_List = Grid_List))
  }
  else {
    stop("Tuning isn't available yet for functional responses")
  }
}
#### basis number choosing function ####
basis_num_choice = function(nbasis,inputdata){

dayrange = c(0,365)
daybasis = create.fourier.basis(dayrange, nbasis)
#daybasis = create.bspline.basis(dayrange, nbasis = nbasis, norder = 5)

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

AIC = mean(2*nbasis + 
             365 * log(colSums((pre - inputdata)^2)/365,base = exp(1)))
BIC = mean(log(365,exp(1))*nbasis + 
             365 * log(colSums((pre - inputdata)^2)/365,base = exp(1)))

out = list(AIC,BIC)
names(out) = c("AIC","BIC")
return(out)
}
#### basis number choosing function <<5 as 1 >>####
basis_num_choice_5as1 = function(nbasis){
  
  dayrange = c(0,365)
  daybasis = create.fourier.basis(dayrange, nbasis)

  Lcoef = c(0,(2 * pi/diff(dayrange))^2,0)
  harmaccelLfd = vec2Lfd(Lcoef, dayrange)
  
  loglam = seq(-5,5,0.25)
  nlam = length(loglam)
  

  gcvsave_press = rep(NA,nlam)
  gcvsave_temp = rep(NA,nlam)
  gcvsave_hum = rep(NA,nlam)
  gcvsave_sun = rep(NA,nlam)
  gcvsave_wind = rep(NA,nlam)
  #pressure  
  for (ilam in 1:nlam) {
    lambda = 10^loglam[ilam]
    fdParobj = fdPar(daybasis, harmaccelLfd, lambda)
    smoothlist = smooth.basis(timepts, pressure,fdParobj)#

    gcvsave_press[ilam] = sum(smoothlist$gcv)#
  }
  
  lambda_press = 10^(loglam[which.min(gcvsave_press)])#
  fdParobj = fdPar(daybasis, harmaccelLfd, lambda_press)#
  
  fit = smooth.basis(timepts,pressure,fdParobj)#
  fd = fd(fit$fd$coefs,daybasis)
  pre = eval.fd(timepts,fd)
  
  AIC_press = mean(2*nbasis + 
            365 * log(colSums((pre - pressure)^2)/365,base = exp(1)))
  BIC_press = mean(log(365,exp(1))*nbasis + 
            365 * log(colSums((pre - pressure)^2)/365,base = exp(1)))
  
  #temp
  for (ilam in 1:nlam) {
    lambda = 10^loglam[ilam]
    fdParobj = fdPar(daybasis, harmaccelLfd, lambda)
    smoothlist = smooth.basis(timepts, temp,fdParobj)#
    
    gcvsave_temp[ilam] = sum(smoothlist$gcv)#
  }
  
  lambda_temp = 10^(loglam[which.min(gcvsave_temp)])#
  fdParobj = fdPar(daybasis, harmaccelLfd, lambda_temp)#
  
  fit = smooth.basis(timepts,temp,fdParobj)#
  fd = fd(fit$fd$coefs,daybasis)
  pre = eval.fd(timepts,fd)
  
  AIC_temp = mean(2*nbasis + 
                     365 * log(colSums((pre - temp)^2)/365,base = exp(1)))
  BIC_temp = mean(log(365,exp(1))*nbasis + 
                     365 * log(colSums((pre - temp)^2)/365,base = exp(1)))
  #humidity
  for (ilam in 1:nlam) {
    lambda = 10^loglam[ilam]
    fdParobj = fdPar(daybasis, harmaccelLfd, lambda)
    smoothlist = smooth.basis(timepts, humidity,fdParobj)#
    
    gcvsave_hum[ilam] = sum(smoothlist$gcv)#
  }
  
  lambda_hum = 10^(loglam[which.min(gcvsave_hum)])#
  fdParobj = fdPar(daybasis, harmaccelLfd, lambda_hum)#
  
  fit = smooth.basis(timepts,humidity,fdParobj)#
  fd = fd(fit$fd$coefs,daybasis)
  pre = eval.fd(timepts,fd)
  
  AIC_hum = mean(2*nbasis + 
                     365 * log(colSums((pre - humidity)^2)/365,base = exp(1)))
  BIC_hum = mean(log(365,exp(1))*nbasis + 
                     365 * log(colSums((pre - humidity)^2)/365,base = exp(1)))
  #wind
  for (ilam in 1:nlam) {
    lambda = 10^loglam[ilam]
    fdParobj = fdPar(daybasis, harmaccelLfd, lambda)
    smoothlist = smooth.basis(timepts, wind,fdParobj)#
    
    gcvsave_wind[ilam] = sum(smoothlist$gcv)#
  }
  
  lambda_wind = 10^(loglam[which.min(gcvsave_wind)])#
  fdParobj = fdPar(daybasis, harmaccelLfd, lambda_wind)#
  
  fit = smooth.basis(timepts,wind,fdParobj)#
  fd = fd(fit$fd$coefs,daybasis)
  pre = eval.fd(timepts,fd)
  
  AIC_wind = mean(2*nbasis + 
                     365 * log(colSums((pre - wind)^2)/365,base = exp(1)))
  BIC_wind = mean(log(365,exp(1))*nbasis + 
                     365 * log(colSums((pre - wind)^2)/365,base = exp(1)))
  #sun
  for (ilam in 1:nlam) {
    lambda = 10^loglam[ilam]
    fdParobj = fdPar(daybasis, harmaccelLfd, lambda)
    smoothlist = smooth.basis(timepts, sun,fdParobj)#
    
    gcvsave_sun[ilam] = sum(smoothlist$gcv)#
  }
  
  lambda_sun = 10^(loglam[which.min(gcvsave_sun)])#
  fdParobj = fdPar(daybasis, harmaccelLfd, lambda_sun)#
  
  fit = smooth.basis(timepts,sun,fdParobj)#
  fd = fd(fit$fd$coefs,daybasis)
  pre = eval.fd(timepts,fd)
  
  AIC_sun = mean(2*nbasis + 
                     365 * log(colSums((pre - sun)^2)/365,base = exp(1)))
  BIC_sun = mean(log(365,exp(1))*nbasis + 
                     365 * log(colSums((pre - sun)^2)/365,base = exp(1)))
  
  AIC = (AIC_press + AIC_temp + AIC_hum + AIC_wind + AIC_sun) / 5
  BIC = (BIC_press + BIC_temp + BIC_hum + BIC_wind + BIC_sun) / 5
  
  
  out = list(AIC,BIC,
        lambda_press,lambda_temp,lambda_hum,lambda_wind,lambda_sun,
        gcvsave_press,gcvsave_temp,gcvsave_hum,gcvsave_wind,gcvsave_sun)
  names(out) = c("AIC","BIC",
                 "lambda_press","lambda_temp","lambda_hum",
                 "lambda_wind","lambda_sun",
                 "gcvsave_press","gcvsave_temp","gcvsave_hum",
                 "gcvsave_wind","gcvsave_sun")
  return(out)
}
#### select number of basis ####
# 5 as 1
AIC_value_5as1 = c(rep(NA,182))
BIC_value_5as1 = c(rep(NA,182))
start = Sys.time()
for (i in 1:182) {
  fit = basis_num_choice_5as1(2 * i + 1)
  AIC_value_5as1[i] = fit$AIC
  BIC_value_5as1[i] = fit$BIC
  cat(paste("basis numer:",2 * i - 1,"\n"))
}
end = Sys.time()
end - start

plot(seq(3,365,2),AIC_value_5as1,
     xlab = "number of basis", ylab = "AIC")
plot(seq(3,365,2),AIC_value_5as1,type = "l",
     xlab = "number of basis", ylab = "AIC")
which.min(AIC_value_5as1)#273

plot(seq(3,365,2),BIC_value_5as1,
     xlab = "number of basis", ylab = "BIC")
plot(seq(3,365,2),BIC_value_5as1,type = "l",
     xlab = "number of basis", ylab = "BIC")

write.csv(cbind(AIC_value_5as1,BIC_value_5as1),file = "5as1 365.csv")

####

inputdata = wind
# 15-365, 10
AIC_value = c(rep(NA,36))
BIC_value = c(rep(NA,36))
for (i in 1:36) {
  AIC_value[i] = basis_num_choice(i*10 + 5,inputdata = inputdata)$AIC
  BIC_value[i] = basis_num_choice(i*10 + 5,inputdata = inputdata)$BIC
  cat(paste("basis numer:",i*10+5,"\n"))
}
#AIC
plot(seq(15,365,10),AIC_value,main = "wind",
     xlab = "number of basis", ylab = "AIC")
min(AIC_value)
which.min(AIC_value)
#BIC
plot(seq(15,365,10),BIC_value,main = "wind",
     xlab = "number of basis",ylab = "BIC")
min(BIC_value)
which.min(BIC_value)

# 50 ge
AIC_value = c(rep(NA,50))
BIC_value = c(rep(NA,50))
for (i in 1:7) {
  AIC_value[i] = basis_num_choice(i*2 + 1,inputdata = inputdata)$AIC
  BIC_value[i] = basis_num_choice(i*2 + 1,inputdata = inputdata)$BIC
  cat(paste("basis numer:",i*2+199,"\n"))
}
#AIC
plot(seq(3,101,2),AIC_value)
min(AIC_num)
which.min(AIC_num)
#BIC
plot(seq(201,299,2),BIC_value,main = "pressure",
     xlab = "number of basis",ylab = "BIC")
min(BIC_value)
which.min(BIC_value)

#### data smoothing ####
#x
dayrange = c(0,365)
Lcoef = c(0,(2 * pi/diff(dayrange))^2,0)
harmaccelLfd = vec2Lfd(Lcoef, dayrange)

#pressure
press_basis = create.fourier.basis(dayrange, 273)
lambda = 10^(-2.25)
fdParobj = fdPar(press_basis, harmaccelLfd, lambda)
press_fd =  smooth.basis(timepts,pressure,fdParobj)$fd
# plot(seq(1,365,1),pressure[,2])
# lines(press_fd[2,])

#temperature
temp_basis = create.fourier.basis(dayrange, 205)
lambda = 10^(-1.5)
fdParobj = fdPar(temp_basis, harmaccelLfd, lambda)
temp_fd =  smooth.basis(timepts,temp,fdParobj)$fd

#humidity
hum_basis = create.fourier.basis(dayrange, 173)
lambda = 10^(-1)
fdParobj = fdPar(hum_basis, harmaccelLfd, lambda)
hum_fd =  smooth.basis(timepts,humidity,fdParobj)$fd

#sun
sun_basis = create.fourier.basis(dayrange, 37)
lambda = 10^(2.75)
fdParobj = fdPar(sun_basis, harmaccelLfd, lambda)
sun_fd =  smooth.basis(timepts,sun,fdParobj)$fd

#wind
wind_basis = create.fourier.basis(dayrange, 65)
lambda = 10^(1.75)
fdParobj = fdPar(wind_basis, harmaccelLfd, lambda)
wind_fd =  smooth.basis(timepts,wind,fdParobj)$fd

# plot(seq(1,365,1),wind[,2])
# lines(wind_fd[2,])

#### y
#prec_fd =  Data2fd(timepts, prec, spline_basis)

#### coeffictient
func_cov_1 = press_fd$coefs
func_cov_2 = rbind(temp_fd$coefs,matrix(rep(0,68*24),68,24))
func_cov_3 = rbind(hum_fd$coefs,matrix(rep(0,100*24),100,24))
func_cov_4 = rbind(sun_fd$coefs,matrix(rep(0,236*24),236,24))
func_cov_5 = rbind(wind_fd$coefs,matrix(rep(0,208*24),208,24))
#func_cov_6 = dtemp_fd$coefs

# coe array
cov_data = array(dim = c(273,dim(pressure)[2], 5))

cov_data[,,1] = func_cov_1
cov_data[,,2] = func_cov_2
cov_data[,,3] = func_cov_3
cov_data[,,4] = func_cov_4
cov_data[,,5] = func_cov_5
#### FLM(scalar resp) ####
#y
y = colMeans(prec)

# train test split(only for testing)
set.seed(1)
train_index = sample(1:24,size = 20)
y_train <- y[train_index]
y_test <- y[-train_index]

#
tempfdata <- fdata(t(temp))
sunfdata <- fdata(t(sun))
pressfdta <- fdata(t(pressure))
windfdata <- fdata(t(wind))
humfdata <- fdata(t(humidity))

y <- colMeans(prec)
dataf_train <- data.frame("y" = y_train)
dataf_test = data.frame("y" = y_test)

f = y ~ X1 + X2 + X3 +X4 +X5

basis2 <- create.fourier.basis(c(0,365),3)
basis.b = list(X1 = basis2, X2 = basis2,
               X3 = basis2,X4 = basis2,X5 = basis2)

basis.x = list(X1 = wind_basis, X2 = temp_basis,
                X3 = press_basis,X4 = sun_basis,X5 = hum_basis)
# spline_basis <- create.fourier.basis(c(0,365),365)
# basis.x = list(X1 = spline_basis,X2 = spline_basis, 
#                X3 = spline_basis, X4 = spline_basis, X5 = spline_basis)

ldata_train = list(df = dataf_train, X1 = windfdata[train_index,],
                   X2= tempfdata[train_index,],X3 = pressfdta[train_index,],
                   X4 = sunfdata[train_index,],X5 = humfdata[train_index,])
fit = fregre.lm(f, ldata_train, basis.x = basis.x,basis.b = basis.b)

fit$residuals
mean(fit$residuals^2) #0.2034248


ldata_test = list(df = dataf_test, X1 = windfdata[-train_index],
            X2= tempfdata[-train_index,],X3 = pressfdta[-train_index,],
            X4 = sunfdata[-train_index,],X5 = humfdata[-train_index,])
pre = predict(fit,newx = ldata_test)
mean((pre - y_test)^2) #4.068367



pred_basis = predict(func_basis[[1]], test_x)
flm_weights[[i]] = func_basis$fregre.basis$coefficients

# 2 covariates
# xfdlist <- list(temp_fd[train_index,],hum_fd[train_index,])
# betalist <- list(basis2,basis2)
# fit.flm <- fRegress(y_train,xfdlist,betalist)
# xfdnew <- list(temp_fd[-train_index,],hum_fd[-train_index,])
# pred <- predict.fRegress(fit.flm,newdata = xfdnew)
# (mse <- mean((y_test - pred)^2))

### FNN(scalar response) ####
y <- colMeans(prec)
set.seed(1)
train_index = sample(1:24,size = 20)
y_train <- y[train_index]
y_test <- y[-train_index]

cov_data_train <- cov_data[,train_index,]
cov_data_test <- cov_data[,-train_index,]


## cv for parameter ##
nbasis = 65
num_basis <- c(nbasis,nbasis,nbasis,nbasis,nbasis)
hidden_layers <- 2
neurons_per_layer = c(240,120)
activations_in_layers = c("sigmoid",'linear')
tensorflow::set_random_seed(1,disable_gpu = FALSE)
fit.cv = fnn.cv(nfolds = 5,resp = y_train, func_cov = cov_data_train,
              basis_choice = c("fourier","fourier","fourier",
                               "fourier","fourier"),
              num_basis = num_basis,
              hidden_layers = hidden_layers,neurons_per_layer = neurons_per_layer,
              activations_in_layers = activations_in_layers,
              domain_range = list(c(0,365),c(0,365),c(0,365)
                                  ,c(0,365),c(0,365)),
              epochs = 50)
print("Overall CV MSPE")
fit.cv$MSPE$Overall_MSPE

tensorflow::set_random_seed(1,disable_gpu = FALSE)
fit = fnn.fit(resp = y_train, func_cov = cov_data_train,
               basis_choice = c("fourier","fourier","fourier",
                                "fourier","fourier"),
              num_basis = num_basis,
              hidden_layers = hidden_layers,neurons_per_layer = neurons_per_layer,
              activations_in_layers = activations_in_layers,
               domain_range = list(c(0,365),c(0,365),c(0,365)
                                   ,c(0,365),c(0,365)),
              # dropout = c(0.1,0.2),
               epochs = 50) #0.7216,0.855 ;0.88

pred_tr = fnn.predict(model = fit, func_cov = cov_data_train,
                   basis_choice = c('fourier','fourier','fourier'
                                    ,'fourier','fourier'),
                   num_basis = num_basis,
                   domain_range = list(c(0,365),c(0,365),c(0,365)
                                       ,c(0,365),c(0,365)))
print("Train MSE")
mean((pred_tr - y_train)^2) 

pred = fnn.predict(model = fit, func_cov = cov_data_test,
                         basis_choice = c('fourier','fourier','fourier'
                                          ,'fourier','fourier'),
                         num_basis = num_basis,
                         domain_range = list(c(0,365),c(0,365),c(0,365)
                                             ,c(0,365),c(0,365)))

print("Test MSE")
mean((pred - y_test)^2) 

fnn.fnc(model = fit, domain_range = list(c(0,365),c(0,365),c(0,365)
        ,c(0,365),c(0,365)), covariate_scaling = T)

## tune fit ##
tune_list =  list(num_hidden_layers = c(2),
                  neurons = c(8, 16),
                  epochs = c(50),
                  val_split = c(0.2),
                  patience = c(15),
                  learn_rate = c(0.1),
                  num_basis = c(3,5),
                  activation_choice = c("relu", "sigmoid"))

tensorflow::set_random_seed(1,disable_gpu = FALSE)
start = Sys.time()
tune_scalar = tune_fnn(tune_list = tune_list,resp = y_train,
                       func_cov = cov_data_train,
                       basis_choice = c("fourier","fourier","fourier",
                                        "fourier","fourier"),
                       domain_range = list(c(0,365),c(0,365),c(0,365)
                                        ,c(0,365),c(0,365)))

end = Sys.time()

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