library(projectThimble)
context('deltaDf')


test_that("Test that dfDelta returns reasonable results", {

  procdf <- dfDelta(data_GOOGL_2018,testDays = 60,validDays = 40,leadDays = 10,thresh = 0.05)
  
  expect_equal(sum(procdf$test),60)
  expect_equal(sum(procdf$valid),40)
  expect_equal(nrow(data_GOOGL_2018)-nrow(procdf),10)
  
})