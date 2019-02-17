library(projectThimble)
context('df2Batch')

test_that("Test that createSample returns reasonable results", {
  df <- dfDelta(data_GOOGL_2018,testDays = 60,validDays = 60,leadDays = 10,thresh = 0.05)
  
  # given an index a df and dim
  # returns scaled price and volume, 
  
  # test each sampleType
  dim <- 32
  i <- 64
  # closing price
  samp1 <- createSample(i,df,dim,1)
  df1 <- df[(i-(dim-1)):64,]
  expect_true(all(samp1[,2] == scale(df1$Close)))
  
  # random sample
  samp2 <- createSample(i,df,dim,2)
  samp3 <- createSample(i,df,dim,3)

  expect_equal(samp1[,1] , samp2[,1])
  expect_equal(samp1[,1] , samp3[,1])
  expect_true(all(samp1[,1] == scale(df1$Volume)))

  # test scaling
  
  
  
  
})

test_that("Test that df2batch returns reasonable results", {
  # skip(message = '')
  df <- dfDelta(data_GOOGL_2018,testDays = 60,validDays = 60,leadDays = 10,thresh = 0.05)
  
  set.seed(1)
  
  b <- df2Batch(df,which(df$valid),32)
    
  expect_equal(dim(b$x),c(60,32,2))
  
  
})