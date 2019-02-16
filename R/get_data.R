
library(quantmod)
require(dplyr)

start <- as.Date("2000-01-01")
end <- as.Date("2018-12-31")

get_data <- function(sym){
  # sym <- 'AAPL'
  # str(sym)
  getSymbols(sym, src = "yahoo", from = start, to = end,auto.assign = F) %>%
    as.data.frame() %>%
    setNames(c('Open','High','Low','Close','Volume','Adjusted')) %>%
    mutate(Date = as.Date(row.names(.))) %>%
    mutate(Sym=sym)
  
}

sym <- c('AAPL',"GOOGL",'MSFT','^DJI','EUR=X','^FTSE','GBPUSD=X')

df <- lapply(sym, get_data) %>%
  bind_rows()

saveRDS(df,'outputs/symbol-data.Rdata')
