clear
clc
dataOrigin = load('data_LDA.txt');
%dataOrigin=zscore(dataOrigin);
%dataOrigin=[line1,line2];
%line1=zscore(dataOrigin(:,1));
%line2=zscore(dataOrigin(:,2));


line1=dataOrigin(:,1);
line2=dataOrigin(:,2);


%dataOrigin=[line1,line2];
%figure;
%hold on;
%plot(line1(1:1:201),line2(1:1:201),'ro');
%plot(line1(202:1:402),line2(202:1:402),'b+')
%legend('class1','class2');
%xlabel('x');
%ylabel('y');
%axis([2 18 0 20]);
%title('orginal data');
%drawnow;

class1=dataOrigin(1:201,:);
class2=dataOrigin(202:402,:);


Dist=pdist(dataOrigin);
Dist(Dist==0)=inf;
Dist=min(Dist);
para=5*mean(Dist);

N=size(dataOrigin,1);
K0=ones(N,N);
K01=ones(N,N);
for i=1:402
    for j=1:402
        I=dataOrigin(i,:);
        J=dataOrigin(j,:);
         K01(i,j)=norm(dataOrigin(i,:)-dataOrigin(j,:));
        K0(i,j)=exp(-K01(i,j)^2/(2*para^2));
    end
end
oneN=ones(N,N)/N;
K=K0-oneN*K0-K0*oneN+oneN*K0*oneN;
%K=zscore(K);
B=(1/201)*ones(201,201);
B=blkdiag(B,B);
KBK=K*B*K;
KK=K*K;
invKK=inv(KK);
A=(inv(K*K))*(K*B*K);
[V,L]=eig(A);
[~,IX]=sort(L,'descend');
V=V(:,IX);
newspace=V(:,1:2);
%[a,b]=max(max(L));
%newspace=V(:,b);
newdata=K0*newspace;


figure;hold on;
%plot(newspace(1:1:402),'c*');
%newdata=ones(402,2);

plot(newdata(1:1:201,1),newdata(1:1:201,2),'ro');
plot(newdata(202:1:402,1),newdata(202:1:402,2),'b+')

legend('class1','class2');
xlabel('x');
ylabel('y');
title('new data');
drawnow;

















