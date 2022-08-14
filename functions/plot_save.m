function [] = plot_save(xtrain,ytrain,xtest,ytest,nTest,pMean,pVar,figName,file_name)
figure(); hold on;
f = [pMean(1:nTest) + 2*sqrt(pVar(1:nTest)); flip(pMean(1:nTest) - 2*sqrt(pVar(1:nTest)),1)];
plot(xtrain,ytrain,'b','LineWidth',2);
fill([xtest(1:nTest); flip(xtest(1:nTest),1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
plot(xtest(1:nTest), ytest(1:nTest), 'g','LineWidth',2); 
plot(xtest(1:nTest), pMean(1:nTest), 'k','LineWidth',2);
title(file_name, 'FontSize', 20);
set(gca,'FontSize', 15);
legend('Training Data', 'Uncertainty Region', 'Test Data', 'Prediction', 'Location', 'best');
savefig(figName);
end