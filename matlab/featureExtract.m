load documents3.mat
emb = readWordEmbedding("wordEmbedding.vec");

sampleNum = size(documents,1);
X = zeros(sampleNum,emb.Dimension);
Y = zeros(sampleNum,1);

for i = 1:sampleNum
    a = string(documents(i));
%     resp = str2num(a{1});
    resp = a{1};
    
%     if resp == 1
%         Y(i) = 1;
%     else
%         Y(i) = -1;
%     end

    if strcmp(resp,'posit')
        Y(i) = 1;
    elseif strcmp(resp,'neg')
        Y(i) = -1;
    else
        Y(i) = 0;
    end

%     if resp == 0
%         Y(i) = 1;
%     elseif resp == 1
%         Y(i) = -1;
%     else
%         Y(i) = 0;
%     end
    
    len = size(a,2);
    for j = 2:len
        vec = word2vec(emb,a{j});
        if ~isnan(vec(1))
            X(i,:) = X(i,:) + vec;
        end
    end
end

mx = mean(X);
sx = std(X,0,1);

MX = mx(ones(sampleNum,1),:);
SX = sx(ones(sampleNum,1),:);

X = (X - MX) ./ SX;

testX = X;
testY = Y;
save train3.mat X Y