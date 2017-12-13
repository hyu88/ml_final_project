load documents.mat

emb = trainWordEmbedding(documents);
writeWordEmbedding(emb,"wordEmbedding.vec");