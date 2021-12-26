import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


#image 库为自定义图像处理库，封装如下主要功能
#1.亚像素边缘提取       TODO
#2.滤波                Bingo
#3....                   TODO
class image:
    tolerance=0
    grey=10
    def __init__(self, im_path,channel,low_bound=254):
        self.im=cv.imread(im_path)
        self.imgray = cv.cvtColor(self.im, cv.COLOR_BGR2GRAY)
        self.channel=channel  #channel 为需要处理图像轮廓线的个数(即图像的轮廓线由几个部分构成)
        self.row=self.im.shape[0]
        self.low_bound=low_bound
        self.approx_polygon=[]
        self.exact_polygon=[]
        self.exact_polygon_list=[]
        #print(self.row)
        self.col=self.im.shape[1]
        #print(self.col)
        self.Transform_Matrix_X=np.mat(np.zeros([self.row,self.col]))
        self.Transform_Matrix_Y=np.mat(np.zeros([self.row,self.col]))
#对离散点列的周长计算,ktype表示计算周长的方法，其中ktype=0表示封闭多边形，ktype=1表示开多边形的线
    def calculate_Perimeter(self,polyline,ktype=False):
        perimeter=0
        short_line=0
        if ktype==False:
            for i in range (len(polyline)):
                short_line=np.sqrt(pow(float(polyline[i][0])-float(polyline[i-1][0]),2)+pow(float(polyline[i][1])-float(polyline[i-1][1]),2))
                perimeter=perimeter+short_line
        #short_line=np.sqrt(pow(float(polyline[0][0]),2)+pow(float(polyline[-1][1]),2))
        #perimeter=perimeter+short_line
        else:
            for i in range (1,len(polyline)):
                short_line=np.sqrt(pow(float(polyline[i][0])-float(polyline[i-1][0]),2)+pow(float(polyline[i][1])-float(polyline[i-1][1]),2))
                perimeter=perimeter+short_line
        print("周长为",perimeter)
        return perimeter

#把多边形点列提取成为list
    def my_tolist(self,sublist):
        A=[]
        for i in range (len(sublist)):
            A.append(sublist[i][0])
        return A
    
#利用opencv的轮廓线检测线输出一个粗略的像素级别的轮廓线,ktype=0表示画出整条轮廓线，ktype=1表示画出多边形轮廓线
    def get_origin_contour_v2(self,boundary_path,index=[[-1,False]]):
        ret, thresh = cv.threshold(self.imgray, self.low_bound, 255, 0)
        self.contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        #对得到的每一条轮廓线中包含像素个数进行存储并且排序
        #具体定义Index中每一元素(i,num),其中i为轮廓线编号，num为轮廓线像素个数
        self.Index=[]
        for i in range (len(self.contours)):
            self.Index.append([i,len(self.contours[i])])
        self.Index.sort(key=lambda x:x[1])
        
        #周长
        perimiter=0
        #控制点列
        pts_list=[]
        #多边形轮廓
        approx_list=[]
        img = self.im.copy()
        for i in range (len(index)):               #[self.Index[index[i][0]][0]]
            approx3 = cv.approxPolyDP(self.contours[self.Index[index[i][0]][0]],1,True)#拟合精确度
            #approx3 = cv.approxPolyDP(self.contours[self.Index[index[i]][0]],1,True)#拟合精确度
            #print("the length of approx_List is",len(approx3))
            sub_List=self.delete_polygon(approx3)
            #print("the length of approx_List is",len(approx3))
            #print("the length of sub_list is",len(sub_List))
            #print(sub_List)
            for j in range (len(sub_List)):
                img=cv.polylines(img,[np.asarray(sub_List[j])],index[i][1],(255,0,0),2)
                print("第",i+1,"段轮廓线的周长为：")
                A=self.my_tolist(sub_List[j])
                pts_list.append(A)
                perimiter+=self.calculate_Perimeter(A,index[i][1])
                approx_list.append(np.asarray(sub_List[j]))    
                
        print("像素级别轮廓线的总周长为",perimiter)
        #print(len(approx_list))
        #print(approx_list[0])
        #img  =cv.polylines(img,approx_list,False,(255,0,0),2)
        
        #亚像素级别的处理  TODO
        #exact_polygon_list=[]
        sub_p=0
        
        for i in range (len(pts_list)):
            self.approx_polygon=pts_list[i]
            sub_p+=self.sub_pixel_correct()
            self.exact_polygon_list.append(self.exact_polygon)
            #复位
            self.exact_polygon=[]
        #计算多边形的周长
        print("亚像素级别的总周长为 ：",sub_p)
        #self.approx_polygon=approx3
        #for i in range (len(approx3)):
            #self.approx_polygon.append(approx3[i][0])
        #self.calculate_Perimeter(self.approx_polygon)
        
        
        
        #plt.subplot(121),plt.imshow(self.im)
        #plt.title('whole counterline'), plt.xticks([]), plt.yticks([])
        plt.imshow(img)
        plt.title('polygon counterline')
        plt.savefig(boundary_path)
        plt.show()
        
        
#利用opencv的轮廓线检测线输出一个粗略的像素级别的轮廓线,多边界切割
    def get_origin_contour(self,boundary_path,index=-1):
        ret, thresh = cv.threshold(self.imgray, self.low_bound, 255, 0)
        self.contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        #对得到的每一条轮廓线中包含像素个数进行存储并且排序
        #具体定义Index中每一元素(i,num),其中i为轮廓线编号，num为轮廓线像素个数
        self.Index=[]
        for i in range (len(self.contours)):
            self.Index.append([i,len(self.contours[i])])
        self.Index.sort(key=lambda x:x[1])
        #整体轮廓
        #cv.drawContours(self.im,self.contours,self.Index[index][0],(255,0,0),3)  
        
        #多边形轮廓
        img = self.im.copy()
        approx3 = cv.approxPolyDP(self.contours[self.Index[index][0]],1,True)#拟合精确度
        img  =cv.polylines(img,[approx3],True,(255,0,0),2)
        #亚像素级别的处理  TODO
        
        
        #计算多边形的周长
        #self.approx_polygon=approx3
        for i in range (len(approx3)):
            self.approx_polygon.append(approx3[i][0])
        self.calculate_Perimeter(self.approx_polygon)
        
        
        
        #plt.subplot(121),plt.imshow(self.im)
        #plt.title('whole counterline'), plt.xticks([]), plt.yticks([])
        plt.imshow(img)
        plt.title('polygon counterline')
        plt.savefig(boundary_path)
        plt.show()


#设置图像的sobel算子  
    def set_sobel(self):
        self.sobelx=cv.Sobel(self.imgray,cv.CV_64F,1,0)    #水平梯度
        self.sobely=cv.Sobel(self.imgray,cv.CV_64F,0,1)    #竖直梯度
        self.fabs_sobelx=cv.convertScaleAbs(self.sobelx)    #负数取绝对值
        self.fabs_sobely=cv.convertScaleAbs(self.sobely)    #负数取绝对值
        
#偏移向量的计算
    def offset_compute(self,g_up,g_down,g):
        dividend=np.linalg.norm(g_up)+np.linalg.norm(g_down)-2*np.linalg.norm(g)
        divisor=np.linalg.norm(g_down)-np.linalg.norm(g_up)
        if dividend!=0 and np.fabs(0.5*divisor/dividend)<=1:
            return 0.5*divisor/dividend
        else: 
            return 0

#判断是否为图像的边界
    def judge_image_boundary(self,i,j):
        if i<=self.tolerance or j<=self.tolerance or i>=self.row-self.tolerance-1 or j>=self.col-self.tolerance-1:
            return True
        else:
            return False

#非光滑Devernay算法
    def Devernary_smooth(self,heatmap_path):
        for i in range (self.row):
            for j in range (self.col):
                #若搜索到的点为非可能边缘点
                if (self.fabs_sobelx[i,j]==0 and self.fabs_sobely[i,j]==0) or self.judge_image_boundary(i,j):
                    continue
                #存储坐标
                coord_up=np.array([0,0])
                coord_down=np.array([0,0])
                coord_mid=np.array([i,j])
                #梯度g_xy
                g_up=np.array([0,0])
                g_down=np.array([0,0])
                g=np.array([self.sobelx[i,j],self.sobely[i,j]])
                
                #第一类大情况，此时梯度向量落入到第一或者是第四空间
                if self.fabs_sobelx[i,j]>=self.fabs_sobely[i,j]:
                    t=self.fabs_sobely[i,j]/self.fabs_sobelx[i,j]
                    #print("the value of t is ",t)
                    #print("self.fabs_sobelx[i,j] ",self.fabs_sobelx[i,j])
                    #print("self.fabs_sobely[i,j] ",self.fabs_sobely[i,j])
                    #若x和y的梯度方向同号，此时落入第一空间
                    if self.sobelx[i,j]*self.sobely[i,j]>=0:   
                        #存贮上方点的梯度向量和坐标向量
                        g_up[0]=(1-t)*self.sobelx[i,j+1]+t*self.sobelx[i-1,j+1]
                        g_up[1]=(1-t)*self.sobely[i,j+1]+t*self.sobely[i-1,j+1]
                        coord_up[0]=(1-t)*i+t*(i-1)
                        coord_up[1]=(1-t)*(j+1)+t*(j+1)
                        #存贮下方点的梯度向量和坐标向量
                        g_down[0]=(1-t)*self.sobelx[i,j-1]+t*self.sobelx[i+1,j-1]
                        g_down[1]=(1-t)*self.sobely[i,j-1]+t*self.sobely[i+1,j-1]
                        coord_down[0]=(1-t)*i+t*(i+1)
                        coord_down[1]=(1-t)*(j-1)+t*(j-1)
                        #计算偏移值
                        offset=self.offset_compute(g_up,g_down,g)
                        self.Transform_Matrix_X[i,j]=(coord_up[0]-coord_mid[0])*offset
                        self.Transform_Matrix_Y[i,j]=(coord_up[1]-coord_mid[1])*offset
                    #若x和y的梯度方向异号，此时落入第四空间
                    else:
                        #存贮上方点的梯度向量和坐标向量
                        g_up[0]=(1-t)*self.sobelx[i,j-1]+t*self.sobelx[i-1,j-1]
                        g_up[1]=(1-t)*self.sobely[i,j-1]+t*self.sobely[i-1,j-1]
                        coord_up[0]=(1-t)*i+t*(i-1)
                        coord_up[1]=(1-t)*(j-1)+t*(j-1)
                        #存贮下方点的梯度向量和坐标向量
                        g_down[0]=(1-t)*self.sobelx[i,j+1]+t*self.sobelx[i+1,j+1]
                        g_down[1]=(1-t)*self.sobely[i,j+1]+t*self.sobely[i+1,j+1]
                        coord_down[0]=(1-t)*i+t*(i+1)
                        coord_down[1]=(1-t)*(j+1)+t*(j+1)
                        #计算偏移值
                        offset=self.offset_compute(g_up,g_down,g)
                        self.Transform_Matrix_X[i,j]=(coord_up[0]-coord_mid[0])*offset
                        self.Transform_Matrix_Y[i,j]=(coord_up[1]-coord_mid[1])*offset
                #第二类大情况，此时梯度向量落入到第二或者是第三空间
                else:
                    t=self.fabs_sobelx[i,j]/self.fabs_sobely[i,j]
                    #若x和y的梯度方向同号，此时落入第二空间
                    if self.sobelx[i,j]*self.sobely[i,j]>=0:   
                        #存贮上方点的梯度向量和坐标向量
                        g_up[0]=(1-t)*self.sobelx[i-1,j]+t*self.sobelx[i-1,j+1]
                        g_up[1]=(1-t)*self.sobely[i-1,j]+t*self.sobely[i-1,j+1]
                        coord_up[0]=(1-t)*(i-1)+t*(i-1)
                        coord_up[1]=(1-t)*j+t*(j+1)
                        #存贮下方点的梯度向量和坐标向量
                        g_down[0]=(1-t)*self.sobelx[i+1,j]+t*self.sobelx[i+1,j-1]
                        g_down[1]=(1-t)*self.sobely[i+1,j]+t*self.sobely[i+1,j-1]
                        coord_down[0]=(1-t)*(i+1)+t*(i+1)
                        coord_down[1]=(1-t)*j+t*(j-1)
                        #计算偏移值
                        offset=self.offset_compute(g_up,g_down,g)
                        self.Transform_Matrix_X[i,j]=(coord_up[0]-coord_mid[0])*offset
                        self.Transform_Matrix_Y[i,j]=(coord_up[1]-coord_mid[1])*offset
                    #若x和y的梯度方向同号，此时落入第二空间
                    else:
                        #存贮上方点的梯度向量和坐标向量
                        g_up[0]=(1-t)*self.sobelx[i-1,j]+t*self.sobelx[i-1,j-1]
                        g_up[1]=(1-t)*self.sobely[i-1,j]+t*self.sobely[i-1,j-1]
                        coord_up[0]=(1-t)*(i-1)+t*(i-1)
                        coord_up[1]=(1-t)*j+t*(j-1)
                        #存贮下方点的梯度向量和坐标向量
                        g_down[0]=(1-t)*self.sobelx[i+1,j]+t*self.sobelx[i+1,j+1]
                        g_down[1]=(1-t)*self.sobely[i+1,j]+t*self.sobely[i+1,j+1]
                        coord_down[0]=(1-t)*(i+1)+t*(i+1)
                        coord_down[1]=(1-t)*j+t*(j+1)
                        #计算偏移值
                        offset=self.offset_compute(g_up,g_down,g)
                        self.Transform_Matrix_X[i,j]=(coord_up[0]-coord_mid[0])*offset
                        self.Transform_Matrix_Y[i,j]=(coord_up[1]-coord_mid[1])*offset
        fig1=plt.figure()
        plt.title("X-Trans-Image", fontsize=24)
        sns_plot = sns.heatmap(self.Transform_Matrix_X,cmap='coolwarm')
        plt.savefig(heatmap_path[0])
        fig2=plt.figure()
        plt.title("Y-Trans-Image", fontsize=24)
        sns_plot = sns.heatmap(self.Transform_Matrix_Y,cmap='coolwarm')
        plt.savefig(heatmap_path[1])
#保留部分边缘的黑框，以免丢失信息,side=0时代表上边界，side=1时代表左边界，side=2时代表下边界，side=3时代表右边界
    def save_the_black(self):
        count=0
        for i in range (self.row):
            for j in range (self.col):
                if self.judge_image_boundary(i,j)==True and i==0:
                    #上边界的下边一个像素块为在图形内部
                    if int(self.imgray[i+1,j])<self.grey:
                        self.im[i,j]=[100,100,100]
                        count=count+1
                if self.judge_image_boundary(i,j)==True and j==0:
                    #左边界的右边一个像素块为在图形内部
                    if int(self.imgray[i,j+1])<self.grey:
                        self.im[i,j]=[0,0,0]
                        count=count+1
                if self.judge_image_boundary(i,j)==True and i==self.row-1:
                    #下边界的上边一个像素块为在图形内部
                    if int(self.imgray[i-1,j])<self.grey:
                        self.im[i,j]=[0,0,0]
                        count=count+1
                if self.judge_image_boundary(i,j)==True and j==self.col-1:
                    #右边界的左边一个像素块为在图形内部
                    if int(self.imgray[i-1,j])<self.grey:
                        self.im[i,j]=[0,0,0]
                        count=count+1
        print("count =",count)

#对多边形的控制点进行修改，并返回多个数组
    def delete_polygon(self,darray):
        sum_list=[]   #存储分段polyline
        A=darray.tolist()
        sub_list=[]
        for i in range (len(A)):
            if self.judge_image_boundary(A[i][0][1],A[i][0][0])==False:
                sub_list.append(A[i])
            else:
                if(len(sub_list)==0):
                    continue
                else:
                    sum_list.append(sub_list)
                    sub_list=[]
        if len(sub_list)!=0:
            sum_list.append(sub_list)
        return sum_list

#对图像边框进行滤波,num表示为第几个图像        
    def filtering_bd(self,path):
        for i in range (self.row):
            for j in range (self.col):
                if self.judge_image_boundary(i,j)==True:
                    self.im[i,j]=[255,255,255]      #设置为白色       
        cv.imshow('img',self.im)
        #print("dddddddddddddddd")
        self.save_the_black()
        print(cv.imwrite(path,self.im))        

#对openncv得到的轮廓线引入亚像素级别的修正
    def sub_pixel_correct(self):
        for i in range (len(self.approx_polygon)):
            #print("self.approx_polygon[i][0] ",self.approx_polygon[i][0])
            index_row=self.approx_polygon[i][1]
            #print(index_row)
            index_col=self.approx_polygon[i][0]
            #print(index_col)
            #trans=[self.Transform_Matrix_X[index_row][index_col],self.Transform_Matrix_Y[index_row][index_col]]+self.approx_polygon[i]
            trans=[self.Transform_Matrix_Y[index_row,index_col]+self.approx_polygon[i][0],self.Transform_Matrix_X[index_row,index_col]+self.approx_polygon[i][1]]
            self.exact_polygon.append(trans)
        print("亚像素级别的修正周长为",self.calculate_Perimeter(self.exact_polygon))
        print("总控制点数目为",len(self.exact_polygon))
        return self.calculate_Perimeter(self.exact_polygon)
            
#向excel文件中写入数据
    def write_in_excel(self,path,page):
        A = np.asarray(self.exact_polygon)
        data = pd.DataFrame(A)
        writer = pd.ExcelWriter(path)		# 写入Excel文件
        data.to_excel(writer, page, float_format='%.5f')		# ‘page_1’是写入excel的sheet名
        writer.save()
        writer.close()
        
        
#向excel文件中写入数据
    def write_in_excel_v2(self,path,page):
        for i in range (len(self.exact_polygon_list)):
            A = np.asarray(self.exact_polygon_list[i])
            data = pd.DataFrame(A)
            writer = pd.ExcelWriter(path[i])		# 写入Excel文件
            data.to_excel(writer, page[i], float_format='%.5f')		# ‘page_1’是写入excel的sheet名
            writer.save()
            writer.close()        
    
