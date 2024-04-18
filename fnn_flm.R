# require(devtools)
# install_version("tensorflow", version = "2.2.0", repos = "http://cran.us.r-project.org")
# install_version("keras", version = "2.2.5.0", repos = "http://cran.us.r-project.org")
# 
# library(reticulate)
# 
# use_condaenv(condaenv = 'python37', conda = "C:/Users/wujia/anaconda3/envs/python37/python.exe")
# use_condaenv(condaenv = 'r-reticulate',required = TRUE)
reticulate::use_condaenv(condaenv = 'tf2',required = TRUE)
# use_condaenv(condaenv = 'python37',required = TRUE)
# use_python("C:/Users/wujia/anaconda3/envs/python37/python.exe")

setwd("C:/Users/wujia/school/Year 4/semester 2/FYP/FNN-main")
source('FNN.R')
library(FuncNN)
library(fda.usc)


# Clearing backend
K <- backend()
K$clear_session()
options(warn = -1)

# Setting seeds
set.seed(2020)
tensorflow::set_random_seed(1,disable_gpu = FALSE)
### aemet data ####
data(aemet)

timepts <- aemet$temp$argvals
y <- rowMeans(aemet$logprec$data)
temp <- aemet$temp$data
windsp <- aemet$wind.speed$data

nbasis = 65
spline_basis = create.fourier.basis(c(0,365), nbasis)

temp_fd =  Data2fd(timepts, t(temp), spline_basis)
wind_fd = Data2fd(timepts,t(windsp), spline_basis)

func_cov_1 = temp_fd$coefs   #65 *73
func_cov_2 = wind_fd$coefs
cov_data = array(dim = c(65, 73, 2))
cov_data[,,1] = func_cov_1
cov_data[,,2] = func_cov_2

train_index = sample(1:73,size = 50)
y_train <- y[train_index]
y_test <- y[-train_index]
cov_data_train <- cov_data[,train_index,]
cov_data_test <- cov_data[,-train_index,]

##### FLM ############
nbasis = 15
spline_basis = create.fourier.basis(c(0,365),nbasis)
xfdlist <- list(temp_fd[train_index,],wind_fd[train_index,])
# xfdlist <- list(rep(1,73),temp_fd)
betalist <- list(spline_basis,spline_basis)

xfdnew <- list(temp_fd[-train_index,],wind_fd[-train_index,])
fit.flm <- fRegress(y_train,xfdlist,betalist)
pred <- predict.fRegress(fit.flm,newdata = xfdnew)
(mse <- mean((y_test - pred)^2))

## fda.usc
library(fda.usc)
data(tecator)
absorp<-tecator$absorp
ind<-sample(215,129) #ind = 1:129 
tt = absorp[["argvals"]]
y = tecator[["y"]]$Fat[ind]
X = absorp[ind, ]
X.d1 = fdata.deriv(X, nbasis = 19, nderiv = 1)
X.d2 = fdata.deriv(X, nbasis = 19, nderiv = 2)
par(mfrow=c(2,2))
plot(X)
plot(X.d1)
plot(X.d2)
boxplot(y)

rangett <- X$rangeval
basis1 = create.bspline.basis(rangeval = rangett, nbasis = 17)
basis2 = create.bspline.basis(rangeval = rangett, nbasis = 7)
res.basis0 = fregre.basis(X, y, basis.x = basis1, basis.b = basis2)
res.basis1 = fregre.basis(X.d1, y, basis.x = basis1, basis.b = basis2)
res.basis2 = fregre.basis(X.d2, y, basis.x = basis1, basis.b = basis2)
res.basis0$r2;res.basis1$r2;res.basis2$r2

x<-X
basis.pc0 = create.pc.basis(X,1:3)
res.pc1 = fregre.pc(X, y, basis.x = basis.pc)
summary(res.pc1)

dataf = as.data.frame(tecator[["y"]][ind,]) # Fat, Protein, Water
basis.pc2 = create.pc.basis(X.d2,1:4)
basis.pc1 = create.pc.basis(X.d1,1:4)
basis.x = list(X = basis.pc0, X.d2 =basis.pc2)
f = Fat ~ X+X.d2 
ldata = list(df = dataf, X=X,X.d2=X.d2,X.d1 = X.d1)
res.lm1 = fregre.lm(f, ldata, basis.x = basis.x)
basis.x = list(X = basis.pc0, X.d2 =basis.pc2,X.d1 = basis.pc1)
f = Fat ~ Water+X.d2 + X.d1
res.lm2 = fregre.lm(f, ldata, basis.x = basis.x)

spline_basis <- create.fourier.basis(c(0,365),65)
basis2 <- create.fourier.basis(c(0,365),2)
tempfdata <- fdata(t(temp))
sunfdata <- fdata(t(sun))
pressfdta <- fdata(t(pressure))
windfdata <- fdata(t(wind))
humfdata <- fdata(t(humidity))

y <- colMeans(prec)
dataf <- data.frame("y" = y)

basis.x = list(X1 = spline_basis, X2 = spline_basis,X3 = spline_basis,X4 = spline_basis,X5 = spline_basis)
basis.b = list(X1 = basis2, X2 = basis2,X3 = basis2,X4 = basis2,X5 = basis2)
f = y ~ X1 + X2 + X3 +X4 +X5
ldata = list(df = dataf, X1 = windfdata,X2= tempfdata, X3 = pressfdta,X4 = sunfdata,X5 = humfdata)
res.lm2 = fregre.lm(f, ldata, basis.x = basis.x,basis.b = basis.b)

fdataobj <- list(X1 = windfdata,X2= tempfdata, X3 = pressfdta,X4 = sunfdata,X5 = humfdata)
fit <- fregre.basis(fdataobj = fdataobj,y, basis.x = basis.x,basis.b = basis.b)

region.contrasts <- model.matrix(~factor(CanadianWeather$region))
rgnContr3 <- region.contrasts
dim(rgnContr3) <- c(1, 35, 4)
dimnames(rgnContr3) <- list('', CanadianWeather$place, c('const',
                                                         paste('region', c('Atlantic', 'Continental', 'Pacific'), sep='.')) )

const365 <- create.constant.basis(c(0, 365))
region.fd.Atlantic <- fd(matrix(rgnContr3[,,2], 1), const365)
# str(region.fd.Atlantic)
region.fd.Continental <- fd(matrix(rgnContr3[,,3], 1), const365)
region.fd.Pacific <- fd(matrix(rgnContr3[,,4], 1), const365)
region.fdlist <- list(const=rep(1, 35),
                      region.Atlantic=region.fd.Atlantic,
                      region.Continental=region.fd.Continental,
                      region.Pacific=region.fd.Pacific)
# str(TempRgn.mdl$betalist)

beta1 <- with(tempfd, fd(basisobj=basis, fdnames=fdnames))
beta0 <- fdPar(beta1)
betalist <- list(const=beta0, region.Atlantic=beta0,
                 region.Continental=beta0, region.Pacific=beta0)
betalist <- list(const=const365, region.Atlantic=const365,
                 region.Continental=const365, region.Pacific=const365)

TempRgn <- fRegress(tempfd$fd, region.fdlist, betalist)

### FNN ####
## raw data = FALSE, scalar response
nbasis = 5
fit <- fnn.fit(resp = y_train, func_cov = cov_data_train,
               basis_choice = c('fourier','fourier'),
               num_basis = c(nbasis,nbasis),
               hidden_layers = 2,neurons_per_layer = c(32,32),
               activations_in_layers = c("sigmoid",'linear'),
               domain_range = list(c(0,365),c(0,365)),
               epochs = 100)

pred <- fnn.predict(model = fit, func_cov = cov_data_test,
                    basis_choice = c('fourier','fourier'),
                    num_basis = c(nbasis,nbasis),
                    domain_range = list(c(0,365),c(0,365)))
(mse <- mean((y_test - pred)^2))
## raw data with scalar response 
fit <- fnn.fit(resp = y, func_cov = list(temp,windsp),
               basis_choice = c('fourier','fourier'),
               num_basis = c(65,65),
               hidden_layers = 2,neurons_per_layer = c(64,64),
               activations_in_layers = c("sigmoid",'linear'),
               domain_range = list(c(0,365),c(0,365)),
               epochs = 100,
               raw_data = TRUE)
## raw data with functional response 
fit <- fnn.fit(resp = aemet$logprec$data, func_cov = list(temp,windsp),
               basis_choice = c('fourier','fourier'),
               num_basis = c(65,65),
               hidden_layers = 2,neurons_per_layer = c(64,64),
               activations_in_layers = c("sigmoid",'linear'),
               domain_range = list(c(0,365),c(0,365)),
               epochs = 100,
               raw_data = TRUE)

## raw data = FALSE with functional response
nbasis = 65
precip_fd = smooth.basis(timepts,t(aemet$logprec$data),spline_basis)$fd
y_train = t(precip_fd$coefs)[train_index,]
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
                 hidden_layers = 3,neurons_per_layer = c(64,64,20),
                 activations_in_layers = c("sigmoid",'sigmoid','linear'),
                 domain_range = list(c(0,365),c(0,365)),
                 epochs = 100)



############### scrip from FNN_bike ########
# Choosing fold number
num_folds = 10

# Creating folds
fold_ind = createFolds(y, k = num_folds)

# Initializing matrices for results
error_mat_lm = matrix(nrow = num_folds, ncol = 2)
# error_mat_pc1 = matrix(nrow = num_folds, ncol = 2)
# error_mat_pc2 = matrix(nrow = num_folds, ncol = 2)
# error_mat_pc3 = matrix(nrow = num_folds, ncol = 2)
# error_mat_pls1 = matrix(nrow = num_folds, ncol = 2)
# error_mat_pls2 = matrix(nrow = num_folds, ncol = 2)
# error_mat_np = matrix(nrow = num_folds, ncol = 2)
# error_mat_cnn = matrix(nrow = num_folds, ncol = 2)
# error_mat_nn = matrix(nrow = num_folds, ncol = 2)
error_mat_fnn = matrix(nrow = num_folds, ncol = 2)

# Functional weights & initializations
func_weights = list()
flm_weights = list()
# nn_training_plot <- list()
# cnn_training_plot <- list()
fnn_training_plot <- list()

# Looping to get results
for (i in 1:num_folds) {
  
  ################## 
  # Splitting data #
  ##################
  
  # Test and train
  train_x = list(aemet$temp[-fold_ind[[i]],],aemet$wind.speed[-fold_ind[[i]],])
  test_x = list(aemet$temp[fold_ind[[i]],],aemet$wind.speed[fold_ind[[i]],])
  train_y = y[-fold_ind[[i]]]
  test_y = y[fold_ind[[i]]]
  
  # Setting up for FNN
  bike_data_train = array(dim = c(65, nrow(train_x[[1]]$data), 2)) # coef (nbasis, sample, 1)
  bike_data_test = array(dim = c(65, nrow(test_x[[1]]$data), 2))
  bike_data_train[,,] = cov_data[, -fold_ind[[i]],]
  bike_data_test[,,] = cov_data[, fold_ind[[i]], ]
  
  ###################################
  # Running usual functional models #
  ###################################
  
  
  # Functional Linear Model (Basis)
  # l=2^(-4:10)
  # func_basis = fregre.basis.cv(train_x, train_y, type.basis = "fourier",
  #                              lambda=l, type.CV = GCV.S, par.CV = list(trim=0.15))
  # pred_basis = predict(func_basis[[1]], test_x)
  # flm_weights[[i]] = func_basis$fregre.basis$coefficients
  # 
  # # Pulling out the coefficients
  # flm_weights[[i]] = func_basis$fregre.basis$coefficients
  # 
  # # Functional Principal Component Regression (No Penalty)
  # func_pc = fregre.pc.cv(train_x, train_y, 8)
  # pred_pc = predict(func_pc$fregre.pc, test_x)
  # 
  # # Functional Principal Component Regression (2nd Deriv Penalization)
  # func_pc2 = fregre.pc.cv(train_x, train_y, 8, lambda=TRUE, P=c(0,0,1))
  # pred_pc2 = predict(func_pc2$fregre.pc, test_x)
  # 
  # # Functional Principal Component Regression (Ridge Regression)
  # func_pc3 = fregre.pc.cv(train_x, train_y, 1:8, lambda=TRUE, P=1)
  # pred_pc3 = predict(func_pc3$fregre.pc, test_x)
  # 
  # # Functional Partial Least Squares Regression (No Penalty)
  # func_pls = fregre.pls(train_x, train_y, 1:6)
  # pred_pls = predict(func_pls, test_x)
  # 
  # # Functional Partial Least Squares Regression (2nd Deriv Penalization)
  # func_pls2 = fregre.pls.cv(train_x, train_y, 8, lambda = 1:3, P=c(0,0,1))
  # pred_pls2 = predict(func_pls2$fregre.pls, test_x)
  # 
  # # Functional Non-Parametric Regression
  # func_np = fregre.np(train_x, train_y, Ker = AKer.tri, metric = semimetric.deriv)
  # pred_np = predict(func_np, test_x)
  # 

  #####################################
  # Running Functional Neural Network #
  #####################################
  
  # Setting seeds
  set.seed(i)
  tensorflow::set_random_seed(i,disable_gpu = FALSE)
  
  # Running FNN for bike
  bike_example <- FNN(resp = train_y,
                      func_cov = bike_data_train,
                      scalar_cov = NULL,
                      basis_choice = c("fourier","fourier"),
                      num_basis = c(9,9),
                      hidden_layers = 4,
                      neurons_per_layer = c(32, 32, 32, 32),
                      activations_in_layers = c("sigmoid", "sigmoid", "relu", "linear"),
                      domain_range = list(c(0, 365),c(0,365)),
                      epochs = 500,
                      output_size = 1,
                      loss_choice = "mse",
                      metric_choice = list("mean_squared_error"),
                      val_split = 0.15,
                      learn_rate = 0.002,
                      patience_param = 15,
                      early_stop = T,
                      print_info = F)
  
  # Predicting
  pred_fnn = FNN_Predict(bike_example,
                         bike_data_test,
                         scalar_cov = NULL,
                         basis_choice = c("fourier","fourier"),
                         num_basis = c(9,9),
                         domain_range = list(c(0, 365),c(0,365)))
  
  # Weights
  func_weights[[i]] = get_weights(bike_example$model)[[1]]
  
  # Training plots
  fnn_training_plot[[i]] = data.frame(epoch = 1:500, value = c(bike_example$per_iter_info$val_loss, rep(NA, 500 - length(bike_example$per_iter_info$val_loss))))
  
  
  ###################
  # Storing Results #
  ###################
  
  # MSPE Results
  # error_mat_lm[i, 1] = mean((c(pred_basis) - test_y)^2, na.rm = T)
  # error_mat_pc1[i, 1] = mean((pred_pc - test_y)^2, na.rm = T)
  # error_mat_pc2[i, 1] = mean((pred_pc2 - test_y)^2, na.rm = T)
  # error_mat_pc3[i, 1] = mean((pred_pc3 - test_y)^2, na.rm = T)
  # error_mat_pls1[i, 1] = mean((pred_pls - test_y)^2, na.rm = T)
  # error_mat_pls2[i, 1] = mean((pred_pls2 - test_y)^2, na.rm = T)
  # error_mat_np[i, 1] = mean((pred_np - test_y)^2, na.rm = T)
  # error_mat_cnn[i, 1] = mean((c(pred_cnn) - test_y)^2, na.rm = T)
  # error_mat_nn[i, 1] = mean((c(pred_nn) - test_y)^2, na.rm = T)
  error_mat_fnn[i, 1] = mean((pred_fnn - test_y)^2, na.rm = T)
  
  # R^2 Results
  # error_mat_lm[i, 2] = 1 - sum((c(pred_basis) - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_pc1[i, 2] = 1 - sum((pred_pc - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_pc2[i, 2] = 1 - sum((pred_pc2 - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_pc3[i, 2] = 1 - sum((pred_pc3 - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_pls1[i, 2] = 1 - sum((pred_pls - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_pls2[i, 2] = 1 - sum((pred_pls2 - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_np[i, 2] = 1 - sum((pred_np - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_cnn[i, 2] = 1 - sum((pred_cnn - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  # error_mat_nn[i, 2] = 1 - sum((pred_nn - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  error_mat_fnn[i, 2] = 1 - sum((pred_fnn - test_y)^2, na.rm=TRUE)/sum((test_y - mean(test_y))^2, na.rm=TRUE)
  
  # Printing iteration number
  print(paste0("Done Iteration: ", i))
  
  # Clearning sessions
  K$clear_session()
  
}
