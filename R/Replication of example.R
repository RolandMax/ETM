install.packages("torch")
install.packages("word2vec")
install.packages("doc2vec")
install.packages("udpipe")
install.packages("remotes")
library(torch)
remotes::install_github('bnosac/ETM', INSTALL_opts = '--no-multiarch')


library(tidyverse)
library(torch)
library(ETM)
library(doc2vec)
library(word2vec)
data(be_parliament_2020, package = "doc2vec")

be_parliament_2020 %>% glimpse()

x      <- data.frame(doc_id           = be_parliament_2020$doc_id, 
                     text             = be_parliament_2020$text_nl, 
                     stringsAsFactors = FALSE)

x$text <- txt_clean_word2vec(x$text)


set.seed(1234)
w2v        <- word2vec(x = x$text, dim = 25, type = "skip-gram", iter = 10, min_count = 5, threads = 2)
embeddings <- as.matrix(w2v)

predict(w2v, newdata = c("migranten", "belastingen"), type = "nearest", top_n = 4)


library(udpipe)
dtm   <- strsplit.data.frame(x, group = "doc_id", term = "text", split = " ")
dtm   <- document_term_frequencies(dtm)
dtm   <- document_term_matrix(dtm)
dtm   <- dtm_remove_tfidf(dtm, prob = 0.50)

vocab        <- intersect(rownames(embeddings), colnames(dtm))
embeddings   <- dtm_conform(embeddings, rows = vocab)
dtm          <- dtm_conform(dtm,     columns = vocab)
dim(dtm)
dim(embeddings)


torch_manual_seed(4321)
model          <- ETM(k = 25, dim = 100, embeddings = embeddings, dropout = 0.5)
optimizer      <- optim_adam(params = model$parameters, lr = 0.005, weight_decay = 0.0000012)
loss_evolution <- model$fit(data = dtm, optimizer = optimizer, epoch = 20, batch_size = 1000)
plot(loss_evolution$loss_test, xlab = "Epoch", ylab = "Loss", main = "Loss evolution on test set")


terminology  <- predict(model, type = "terms", top_n = 5)
terminology



