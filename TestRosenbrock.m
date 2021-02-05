valleyCoeff=25;
x0 = [-100,100];
options=optimoptions('lsqnonlin','Algorithm','levenberg-marquardt','display', 'iter-detailed','MaxIterations', 1000, 'SpecifyObjectiveGradient',true,'ScaleProblem', 'Jacobian','FiniteDifferenceType', 'central','CheckGradients', true); %
objfun = @(x)rosenbrock(x, valleyCoeff);

[x ,~,~,exitflag,~]= lsqnonlin(objfun,x0,[],[],options);

x


function [f,g]=rosenbrock(x,valleycoeff)
 f=[valleycoeff*(x(2)-x(1)*x(1));
     1.0-x(1)];
 
 g= [-2.0*valleycoeff*x(1), valleycoeff
    -1.0, 0.0];

end


        