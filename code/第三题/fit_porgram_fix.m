close all;
A = readmatrix("dfcontour10.xlsx"); 
A(1,:) = [];
B = readmatrix("breaklist10.xlsx");
B (1,:) = [];
n=33;   % n 参数  断点个数
MAX= size(A,1); % MAX  parameter number of all the points 
C = zeros(n+1,3);  %需要考虑边界的问题  (n+1)*2
C(1,1)=1;C(1,2)=B(1,2);C(1,3)=A(1,12);
C(n+1,1)=B(n,2);C(n+1,2)=MAX;C(n+1,3)=A(n+1,12);
for count = 2:n
    C(count,1) =B(count-1,2)+1;%start seq
    C(count,2) =B(count,2)+1;%end seq    load start with 1!!!!!!!
    C(count,3) =A(B(count-1,2)+1,12);%0 line 1 elli
end

fit_para = zeros(n+1,12);
%1>> -1 for line 1 for ellipse
%2    a: 6.1538
%3    b: 20.8861
%4    phi: 0.0033
%5    X0: 14.9238
%6    Y0: 14.0687
%7    Y0_in: 14.0189
%8    long_axis: 41.7722
%9    short_axis: 12.3075
%10   sweep angle
%11   start_seq
%12   end_seq
%13   start_x
%14   start_y
%15   end_x
%16   end_y


for count = 1:n+1
    X = A(C(count,1):C(count,2),2);
    Y = A(C(count,1):C(count,2),3);
    S = A(C(count,1):C(count,2),1)+1; % start with 1 !!!
    if C(count,3)==1
        ellipse_para = fit_ellipse(X,Y);
         %ellipse_para = ellipseFit(X,Y);
        fit_para(count,1) = 1; %ellipse
        fit_para(count,2) = ellipse_para.a;
        fit_para(count,3) = ellipse_para.b;
        fit_para(count,4) = ellipse_para.phi;
        fit_para(count,5) = ellipse_para.X0;
        fit_para(count,6) = ellipse_para.Y0;
        fit_para(count,7) = ellipse_para.Y0_in;
        fit_para(count,8) = ellipse_para.long_axis;
        fit_para(count,9) = ellipse_para.short_axis;
        %fit_para(count,10) = ellipse_para.status;
       
    elseif C(count,3)==0
%1>> -1 for line 1 for ellipse
%2    x1
%3    x0
%4    Y1
%5    Y0
%6    length 


        fit_para(count,1) = -1;%line
        line_para_x = polyfit(S,X,1);
        line_para_y = polyfit(S,Y,1);       
        fit_para(count,2) = line_para_x(1);
        fit_para(count,3) = line_para_x(2);
        fit_para(count,4) = line_para_y(1);
        fit_para(count,5) = line_para_y(2);
        %fit_para(count,6) = 
        
    else
        print("erorr");
    end
end
 %放置点的位置 11start 12 end

for count  = 1:n+1
   if (fit_para(count,1)==-1)
      length  = sqrt(A(C(count,1),2)-A(C(count,2),2))*(A(C(count,1),2)-A(C(count,2),2))+(A(C(count,1),3)-A(C(count,2),3))*(A(C(count,1),3)-A(C(count,2),3));
      fit_para(count,6) = length;
      
   elseif (fit_para(count,1)==1)
       s1 = C(count,1);
       s2 = C(count,2);
       a=[A(s1,2)-fit_para(count,5),A(s1,3)-fit_para(count,6)];
       b=[A(s2,2)-fit_para(count,5),A(s2,3)-fit_para(count,6)];
        fit_para(count,10) = dot(a,b)/sqrt(norm(a)*norm(b));
   %     fit_para(count,10) = acosd(dot(a,b)/sqrt(norm(a)*norm(b)));
        fit_para(count,10) = count;
   end
end
%1>> -1 for line 1 for ellipse
%2    a: 6.1538
%3    b: 20.8861
%4    phi: 0.0033
%5    X0: 14.9238
%6    Y0: 14.0687
%7    Y0_in: 14.0189
%8    long_axis: 41.7722
%9    short_axis: 12.3075
%10   sweep angle
%11   start_seq
%12   end_seq
%13   start_x
%14   start_y
%15   end_x
%16   end_y


for count = 1:n+1
    s=C(count,1);
    e=C(count,2);
    fit_para(count,11) = s;
    fit_para(count,12) = e;
    fit_para(count,13) = A(s,2);fit_para(count,14) = A(s,3);
    fit_para(count,15) = A(e,2);fit_para(count,16) = A(e,3);
    
end

data = fit_para;
pathout = 'fit_parameter10.xlsx';
Title = {'note','a/x1','b/x0','phi/y1','xc/y0','yc/length',"Y0_in",'long_axis','short_axis','angle','start_seq','end_seq','start_x','start_y','end_x','end_y',};
xlswrite(pathout,Title,1,'A1');
xlswrite(pathout,data,1,'A2');


