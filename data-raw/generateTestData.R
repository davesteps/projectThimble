

# data for test_getdata ------
start <- as.Date("2018-12-01")
end <- as.Date("2018-12-31")
sym <- "GOOGL"
data_GOOGL_2018_01 <- get_data(sym,start,end)
devtools::use_data(data_GOOGL_2018_01,overwrite = T)

# process_df data =======

start <- as.Date("2018-01-01")
end <- as.Date("2018-12-31")
sym <- "GOOGL"
data_GOOGL_2018 <- get_data(sym,start,end)
devtools::use_data(data_GOOGL_2018,overwrite = T)
