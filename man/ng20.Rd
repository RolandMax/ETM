% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/data.R
\docType{data}
\name{ng20}
\alias{ng20}
\title{Bag of words sample of the 20 newsgroups dataset}
\description{
Data available at \url{https://github.com/adjidieng/ETM/tree/master/data/20ng}
}
\examples{
data(ng20)
str(ng20$vocab)
str(ng20$bow_tr$tokens)
str(ng20$bow_tr$counts)
}
