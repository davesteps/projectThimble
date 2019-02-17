require(dplyr)
require(keras)
require(lubridate)
require(ggplot2)
library(projectThimble)

# start <- as.Date("2000-01-01")
# end <- as.Date("2018-12-31")
sym <- c('AAPL',"GOOGL",'MSFT','^DJI','EUR=X','^FTSE','GBPUSD=X')
df <- lapply(sym, get_data) %>%
  bind_rows()

# saveRDS(df,'outputs/symbol-data.Rdata')
df <- readRDS('outputs/symbol-data.Rdata')

# binary classifier >x% increase after t period
# batch generator
# scaling?
# 1D convolution
# test set last 6 months?
# validation set?
# fill in NAs?
# volume and decimal year

str(df)
df %>%
  group_by(Sym) %>%
  mutate(Close = Close-mean(Close,na.rm = T),
         Close = Close/sd(Close,na.rm = T)
         ) %>%
ggplot()+geom_line(aes(x=Date,y=Close,color=Sym))

df_ <- df %>%
  filter(Sym == 'GOOGL') %>%
  dfDelta(testDays = 125,validDays = 50,leadDays = 8,thresh = 0.05)

df_ %>%
  mutate(year = year(Date)) %>%
  group_by(year) %>%
  summarise(n())

# train/test spllit ----------

df_train <- filter(df_, !test & !valid)

test <- df2Batch(df_,which(df_$test),64,3)
valid <- df2Batch(df_,which(df_$valid),64,3)
str(test)
str(valid)
batch_generator <- function(df_train,size,dim,sampleType){
  df2Batch(df_train,sample(dim:nrow(df_train),size),dim,sampleType)
}

train <- batch_generator(df_train,16,64,3)

data.frame(train$x[,,2]) %>%
  t() %>%
  data.frame %>%
  tidyr::gather() %>%
  mutate(i = rep(1:64,16),
         class = unlist(lapply(train$y, function(i) rep(i,64)))) %>%
  ggplot()+geom_line(aes(x=i,y=value,group=key,colour=class))+
  facet_wrap(~class,ncol = 1)

str(df_)

# model training --------
library(keras)
is_keras_available()
reticulate::py_discover_config("keras")
use_python('/anaconda3/envs/r-tensorflow/bin/python')

# install_keras()
keras::k_clear_session()
keras::k_reset_uids()

input_shape <- c(dim(train$x)[-1])

# fixed params ----------
epochs <- 100
batch_size = 16
aug <- F
# -------------------
n1 = 16
n2 = n1
ks <- 5

model <- keras_model_sequential() 

model %>%
  layer_conv_1d(filters = 32,kernel_size=5,input_shape = c(300,1),activation = "relu") %>%
  layer_conv_1d(filters = 32,kernel_size=5,activation = "relu") %>%
  layer_max_pooling_1d() %>%
  layer_dropout(0.2,) %>%
  layer_conv_1d(filters = 32,kernel_size=5,activation = "relu") %>%
  layer_conv_1d(filters = 32,kernel_size=5,activation = "relu") %>%
  layer_max_pooling_1d() %>%
  layer_dropout(0.2) %>%
  layer_flatten() %>%
  layer_dense(128,activation = 'relu') %>%
  layer_dropout(0.2) %>%
  layer_dense(2,activation = 'sigmoid')


opt <- optimizer_rmsprop()
# opt <- optimizer_adam()

model %>% 
  compile(optimizer=opt,
          loss='binary_crossentropy',
          metrics = "accuracy")

es <- callback_early_stopping(patience = 12)
alr <- callback_reduce_lr_on_plateau(patience = 4)

training <- model %>%
  fit_generator(
    batch_generator(df_train,batch_size),
    steps_per_epoch = floor(dim(train$x)[1]/batch_size),
    epochs = epochs,
    validation_data =  list(valid$x, valid$y),
    callbacks = list(alr,es)
  )

# Training ----------------------------------------------------------------

model %>%
  fit(
    x_train, imdb$train$y,
    batch_size = batch_size,
    epochs = epochs,
    validation_data = list(x_test, imdb$test$y)
  )




