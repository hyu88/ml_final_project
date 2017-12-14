%% load training data
load train3.mat

%% feature tuning
c = cvpartition(Y,'Holdout',0.3);
trainX = X(training(c),:);
trainY = Y(training(c),:);
testX = X(test(c),:);
testY = Y(test(c),:);

C = logspace(-3,3,7);
linearErr = zeros(1,7);
for i = 1:7
    t = templateSVM('KernelFunction','linear','BoxConstrain',C(i));
    Mdl = fitcecoc(trainX,trainY,'Learner',t,'CrossVal','on',...
        'KFold',3);
    linearErr(i) = kfoldLoss(Mdl);
end

gamma = [0.1, 1, 10];
gaussianErr = zeros(3,7);
for i = 1:7
    for j = 1:3
        t = templateSVM('KernelFunction','guassion','BoxConstrain',C(i),...
            'KernelScale',gamma(j));
        Mdl = fitcecoc(trainX,trainY,'Learner',t,'CrossVal','on',...
            'KFold',3);
        gaussianErr(j,i) = kfoldLoss(Mdl);
    end
end

order = [2, 3];
polyErr = zeros(2,7);
for i = 1:7
    for j = 1:2
        t = templateSVM('KernelFunction','polynomial','BoxConstrain',C(i),...
            'PolynomialOrder',order(j));
        Mdl = fitcecoc(trainX,trainY,'Learner',t,'CrossVal','on',...
            'KFold',3);
        polyErr(j,i) = kfoldLoss(Mdl);
    end
end

%% test

t = templateSVM('KernelFunction','linear','BoxConstrain',10);
Mdl = fitcecoc(X,Y,'Learner',t);


load test.mat
Yhat = predict(Mdl, testX);
accuracy = sum(str2num(char(Yhat)) == testY) / size(testY,1);