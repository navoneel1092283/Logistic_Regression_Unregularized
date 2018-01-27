function [J, grad] = costFunction(theta, X, y)
J = 0;
grad = zeros(size(theta));
m=length(y);
x=zeros(m,1);
%pl=zeros(m,1);
cal=zeros(m,1);
error=zeros(m,1);
p=X*theta;
for i=1:m
	cal(i)=(-y(i)*log(sigmoid(p(i))))-((1-y(i))*log(1-sigmoid(p(i))));
end;
J=(1/m)*sum(cal);

for i=1:m
	error(i)=sigmoid(p(i))-y(i);
end;

for j=1:length(theta)
	x=X(:,j);
	grad(j)=(1/m)*(sum(error.*x));	
	
end;
