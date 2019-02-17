library(projectThimble)
context('get_data')


test_that("Test that get_data returns data frame", {

  start <- as.Date("2018-12-01")
  end <- as.Date("2018-12-31")
  sym <- "GOOGL"
  
  df <- get_data(sym,start,end)
  
  expect_equal(nrow(df),18)
  expect_true(all(c('Open','High','Low','Close','Volume','Date','Sym') %in% names(df)))
  expect_equal(df$Sym[1],sym)
  expect_identical(df,data_GOOGL_2018_01)
  
})