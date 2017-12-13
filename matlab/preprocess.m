data = importdata('origin.csv');

data = replace(data,{',','_','-'},' ');
data = eraseURLs(data);
data = eraseTags(data);
data = erasePunctuation(data);
data = lower(data);

documents = tokenizedDocument(data);
documents = removeWords(documents,stopWords);
documents = removeLongWords(documents,15);
documents = normalizeWords(documents);

save documents.mat documents