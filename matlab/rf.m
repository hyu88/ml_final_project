%% load training samples
load train3.mat

%% feature tuning (criterion and min_leaf_size
trainX = X;
trainY = Y;
c = cvpartition(Y,'Holdout',0.3);
trainX = X(training(c),:);
trainY = Y(training(c),:);
testX = X(test(c),:);
testY = Y(test(c),:);

errGini = zeros(1,5);
for i = 1:5
    Mdl = TreeBagger(150,trainX,trainY,'OOBPrediction','On','Method',...
        'classification','MinLeafSize',2*i,'SplitCriterion','gdi');
    en = oobError(Mdl);
    errGini(i) = en(150);
end

errEntropy = zeros(1,5);
for i = 1:5
    Mdl = TreeBagger(150,trainX,trainY,'OOBPrediction','On','Method',...
        'classification','MinLeafSize',2*i,'SplitCriterion','deviance');
    en = oobError(Mdl);
    errEntropy(i) = en(150);
end

figure
hold on
plot(x,errGini);
plot(x,errEntropy);
xlabel('MinLeafSize');
ylabel('oobErr')
legend('Gini','Entropy');

%% train with the optimal hyperparameters, and decide the number of trees
Mdl = TreeBagger(250,trainX,trainY,'OOBPrediction','On','Method',...
    'classification','MinLeafSize',10,'SplitCriterion','deviance');
en = oobError(Mdl);

plot(en)
xlabel('number of trees');
ylabel('oob error')

%% test 
load test.mat
Yhat = predict(Mdl, testX);
accuracy = sum(str2num(char(Yhat)) == testY) / size(testY,1);