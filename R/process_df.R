

#' Title
#'
#' @param df_ 
#' @param testDays 
#' @param validDays 
#' @param leadDays 
#' @param thresh 
#'
#' @return
#' @export
#'
#' @examples
dfDelta <- function(df_,testDays=60,validDays=60,leadDays = 10, thresh = 0.06){
  
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




df2Batch <- function(df_,idx,dim=64,sampleType=3){
  # generates  XY arrays from df
  # dim <- 64
  # idx <- which(df_$test)
  
  Y <- df_$thresh[idx]
  X <- plyr::laply(idx, createSample,df_,dim,sampleType)
  list(x=X,y=Y)
  
}


createSample <- function(i,df_,dim,sampleType = 1) {
  # i <- idx[1]
  df_1 <- df_ %>%
    slice((i-(dim-1)):i) %>%
    select(Open,High,Low,Close,day,Volume)
  
  vol <- scale(df_1$Volume)
  
  price <- df_1 %>%
    select(Open,High,Low,Close) %>%
    as.matrix() %>% 
    {switch( sampleType,
      '1' = .[,4],
      '2' = apply(., 1, function(r) sample(r,1)) ,
      '3' = apply(., 1, function(r) runif(1,r[3],r[2])) 
    )} %>%
    scale()
  # {.-mean(.)} %>% {./sd(.)}
  abind::abind(vol,price)
}