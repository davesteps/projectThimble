
get_data <- function(sym,start,end){
  # sym <- 'AAPL'
  # str(sym)
  quantmod::getSymbols(sym, src = "yahoo", from = start, to = end,auto.assign = F) %>%
    as.data.frame() %>%
    setNames(c('Open','High','Low','Close','Volume','Adjusted')) %>%
    mutate(Date = as.Date(row.names(.))) %>%
    mutate(Sym=sym)
  
}

