require(dplyr)
require(keras)
require(lubridate)
require(ggplot2)
require(abind)

# binary classifier >x% increase after t period
# batch generator
# scaling?
# 1D convolution
# test set last 6 months?
# validation set?
# fill in NAs?
# volume and decimal year

df <- readRDS('outputs/symbol-data.Rdata')

str(df)
df %>%
  group_by(Sym) %>%
  mutate(Close = Close-mean(Close,na.rm = T),
         Close = Close/sd(Close,na.rm = T)
         ) %>%
ggplot()+geom_line(aes(x=Date,y=Close,color=Sym))

df_ %>%
  mutate(year = year(Date)) %>%
  group_by(year) %>%
  summarise(n())

# train/test spllit ----------

str(df)
unique(df$Sym)



proc_df <- function(df_,testDays=60,validDays=60,leadDays = 10, thresh = 0.06){
  
  # df_ <- filter(df,Sym == 'GOOGL')
  # testDays <- 60
  # validDays <- 60
  # leadDays <- 10
  # thresh <- 0.06
  

  # View(df_)
  # nrow(df_)
  df_ <- df_ %>%
    arrange((Date)) %>%
    mutate(change = lead(Close,leadDays),
           delta = (change-Close)/Close,
           thresh = delta > thresh) %>%
    filter(!is.na(change))
  # head(df_)
  # df_ %>% group_by(thresh) %>% summarise(n())
  # ggplot(df_)+
  #   geom_line(aes(x=Date,y=Close))+
  #   geom_point(data=filter(df_,thresh),aes(x=Date,y=Close),col=2)
  df_ <- arrange(df_,rev(Date))
  df_$test <- F
  df_$valid <- F
  df_$test[1:testDays] <- T
  df_$valid[(testDays+1):(testDays+validDays)] <- T
  df_ <- arrange(df_,(Date))
  
  df_$day <- lubridate::yday(df_$Date)/365
  # str(df_)
  # df_$Volume <- df_$Volume/quantile(df_$Volume,0.95)
  # df_$Volume[df_$Volume>1] <- 1
  # hist(df_$Volume)
  # ggplot(df_)+geom_line(aes(x=Date,y=Volume))
  df_
  
}


df2Batch <- function(df_,idx,dim=64){
  # generates  XY arrays from df
  # dim <- 64
  # idx <- which(df_$test)
  
  Y <- df_$thresh[idx]
  X <- plyr::laply(idx, function(i) {
    # i <- idx[1]
    df_1 <- df_ %>%
      slice((i-(dim-1)):i) %>%
      select(Open,High,Low,Close,day,Volume)
  
    vol <- scale(df_1$Volume)
    
    price <- df_1 %>%
      as.matrix() %>% 
      apply(., 1, function(r) sample(r,1)) %>%
      scale()
      # {.-mean(.)} %>% {./sd(.)}
    abind(vol,price)
  })
  
  list(x=X,y=Y)
  
}

df_ <- df %>%
  filter(Sym == 'GOOGL') %>%
  proc_df(thresh = 0.05)

df_train <- filter(df_, !test & !valid)

test <- df2Batch(df_,which(df_$test))
valid <- df2Batch(df_,which(df_$valid))

batch_generator <- function(df_train,size = 16,dim=64){
  df2Batch(df_train,sample(dim:nrow(df_train),size),dim)
}

train <- batch_generator(df_train)
str(test$x)
data.frame(train$x[,,1]) %>%
  t() %>%
  data.frame %>%
  tidyr::gather() %>%
  mutate(i = rep(1:64,16),
         class = unlist(lapply(train$y, function(i) rep(i,64)))) %>%
  ggplot()+geom_line(aes(x=i,y=value,group=key,colour=class))

str(df_)

# model training --------
library(keras)
is_keras_available()

keras::
keras::install_keras()
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