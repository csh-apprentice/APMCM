function [ellipse] = ellipseFit(x,y)
xlength = length(x);
xmax = max(abs(x));
ymax = max(abs(y));
x = x./xmax;
y = y./ymax;%%归一化处理
if(xlength ~= length(y) | xlength  < 5)
    warning('the length of x and y must be same and >= 5');
else
    xigema.one = xlength;  xigema.x = sum(x);     xigema.y = sum(y);
    xigema.xx =sum(x.*x);  xigema.xy = sum(x.*y); xigema.yy = sum(y.*y);
    xigema.xxx = sum(x.^3); xigema.xxy = sum(x.*x.*y);
    xigema.xyy = sum(x.*y.*y); xigema.yyy = sum(y.^3);
    xigema.xxxy = sum(x.^3.*y);xigema.xxyy = sum(x.*x.*y.*y);
    xigema.xyyy = sum(x.*y.^3);xigema.yyyy = sum(y.^4);
    B = [xigema.xxyy xigema.xyyy xigema.xxy xigema.xyy xigema.xy;
        xigema.xyyy xigema.yyyy xigema.xyy xigema.yyy xigema.yy;
        xigema.xxy xigema.xyy xigema.xx xigema.xy xigema.x;
        xigema.xyy xigema.yyy xigema.xy xigema.yy xigema.y;
        xigema.xy xigema.yy xigema.x xigema.y xigema.one];
    if(rank(B)<5)
        warning("Please check input set again")
    else
        C = [xigema.xxxy;xigema.xxyy;xigema.xxx;xigema.xxy;xigema.xx];
        D = -inv(B)*C;
        %标准方程参数(注意反归一化)
        norm.a = 1/(xmax*xmax*D(5));
        norm.b = D(1)/(xmax*ymax*D(5));
        norm.c = D(2)/(ymax*ymax*D(5));
        norm.d = D(3)/(xmax*D(5));
        norm.e = D(4)/(ymax*D(5));
        %参数方程参数
        ellipse.x0  =(norm.b*norm.e-2*norm.c*norm.d)/(4*norm.a*norm.c-norm.b^2);
        ellipse.y0 = (norm.b*norm.d-2*norm.a*norm.e)/(4*norm.a*norm.c-norm.b^2);
        ellipse.a = sqrt((2*(norm.a*ellipse.x0^2+norm.c*ellipse.y0^2+norm.b*ellipse.x0*ellipse.y0-1))...
            /(norm.a+norm.c+sqrt((norm.a-norm.c)^2+norm.b^2)));
        ellipse.b = sqrt((2*(norm.a*ellipse.x0^2+norm.c*ellipse.y0^2+norm.b*ellipse.x0*ellipse.y0-1))...
            /(norm.a+norm.c-sqrt((norm.a-norm.c)^2+norm.b^2)));
        ellipse.phi = 0.5*atan2(norm.b,(norm.a-norm.c));
        %绘制图像
        theta =  linspace(0.1*pi,1.9*pi,100)
        plot(x*xmax,y*ymax,'o')
        hold on
        x = ellipse.a * cos(theta);
        y = ellipse.b * sin(theta);
        Ex = (x*cos(ellipse.phi)-y*sin(ellipse.phi) + ellipse.x0);
        Ey = (x*sin(ellipse.phi)+y*cos(ellipse.phi) + ellipse.y0);
        plot(Ex,Ey,'r','linewidth',1.6)
        axis equal
    end
end
end