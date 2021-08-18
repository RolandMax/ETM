install.packages("torch")
install.packages("word2vec")
install.packages("doc2vec")
install.packages("udpipe")
install.packages("remotes")
library(torch)
remotes::install_github('bnosac/ETM', INSTALL_opts = '--no-multiarch')


library(tidyverse)
library(torch) # torch seems to be a deep learning library   
library(ETM)
library(doc2vec)
library(word2vec)


# load some data already provided by the loaded packages
data(be_parliament_2020, package = "doc2vec")

data("brussels_reviews", package = "udpipe")

# load some SPAR data 
load(file = "data/Artikel_Corpus.RData")

# first convert the quanteda corpus object into a data frame 
library(quanteda)
y <- Artikel_Corpus %>% convert(to = "data.frame")
x <- y %>% select(doc_id, text)

# text data of git example
be_parliament_2020 %>% glimpse()


x      <- data.frame(doc_id           = be_parliament_2020$doc_id, 
                     text             = be_parliament_2020$text_nl, 
                     stringsAsFactors = FALSE)


x <- brussels_reviews %>% filter(language == "es") %>% 
    mutate(doc_id = as.character(id),
           text = feedback) %>% 
    select(doc_id, text) %>% data.frame() %>% 
    glimpse()
    

# txt_clean_word2vec() does some pre processing 
# is part of the word2vec package
x$text <- txt_clean_word2vec(x$text)

# compute the word embeddings
set.seed(1234)
w2v        <- word2vec(x = x$text, dim = 100, type = "skip-gram", iter = 10, min_count = 5, threads = 2)

# keep the word embeddings for later
embeddings <- as.matrix(w2v)

# inspect the usual word embeddings
predict(w2v, newdata = c("merkel"), type = "nearest", top_n = 10)

# udpipe seems to be a very recent lib for pre processing
# including POS tagging
library(udpipe)
dtm   <- strsplit.data.frame(x, group = "doc_id", term = "text", split = " ")
dtm   <- document_term_frequencies(dtm)
dtm   <- document_term_matrix(dtm)
dtm   <- dtm_remove_tfidf(dtm, prob = 0.50)

# Make sure document/term/matrix and the embedding matrix have the same vocabulary
vocab        <- intersect(rownames(embeddings), colnames(dtm))
embeddings   <- dtm_conform(embeddings, rows = vocab)
dtm          <- dtm_conform(dtm,     columns = vocab)
dim(dtm)
dim(embeddings)

# Learn k topics with a 100-dimensional hyperparameter for the variational inference
torch_manual_seed(4321)
model          <- ETM(k = 10, dim = 100, embeddings = embeddings, dropout = 0.5)
optimizer      <- optim_adam(params = model$parameters, lr = 0.005, weight_decay = 0.0000012)
loss_evolution <- model$fit(data = dtm, optimizer = optimizer, epoch = 20, batch_size = 1000)
plot(loss_evolution$loss_test, xlab = "Epoch", ylab = "Loss", main = "Loss evolution on test set")


terminology  <- predict(model, type = "terms", top_n = 10)
terminology %>% glimpse
    


