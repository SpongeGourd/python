# -*- coding: utf-8 -*-
"""
Created on Tue Jan 07 13:17:17 2016

@author: n.li
"""

import paramiko,base64,time
import sys 
   
hostname='10.8.92.190'
username='hotel'
password='ctrip,bigdata,OTA,2016'
port = 1022

tablename = sys.argv[1]
cur_date = sys.argv[2] 
cur_date_hyphen = sys.argv[3]

#tablename = '151021_hod_MJSY'
#cur_date = 20160120 
#cur_date_hyphen = '2016-01-20'

paramiko.util.log_to_file('app_abtest_email.log')    
s=paramiko.SSHClient() 
s.set_missing_host_key_policy(paramiko.AutoAddPolicy())    
s.connect(hostname = hostname,username=username, password=password,port = port)    
stdin,stdout,stderr=s.exec_command('ifconfig;free;df -h')
stdin,stdout,stderr=s.exec_command('''hive -e "set hive.cli.print.header=true;select treatmentid, \'exp\' as experiment, \'all\' as anlyzclass, \'all\' as classlevel, avg_totalordersord, std_totalordersord, avg_uncancelordersord, std_uncancelordersord, avg_totalgpcii, std_totalgpcii, avg_uncancelgpcii, std_uncancelgpcii, avg_returngpcii, std_returngpcii, avg_uncancelreturngpcii, std_uncancelreturngpcii, \'d\' as d from dw_htlbizdb.tmp_ssh_abtest_email_%s_%s" > /home/hotel/lina/app_abtest_email_%s_%s.txt'''%(tablename,cur_date,tablename,cur_date))
stdin,stdout,stderr=s.exec_command('''hive -e "set hive.cli.print.header=true;select * from dw_htlbizdb.tmp_lina_app_abtest_3a where d=\'%s\';" > /home/hotel/lina/app_abtest_email_detail_%s_%s.txt'''%(cur_date_hyphen,tablename,cur_date))

time.sleep(180)

# now, connect and use paramiko Transport to negotiate SSH2 across the connection
t = paramiko.Transport((hostname, port))
t.connect( username=username, password=password)
sftp = paramiko.SFTPClient.from_transport(t)
# dirlist on remote host
dirlist = sftp.listdir('.')
sftp.get('/home/hotel/lina/app_abtest_email_%s_%s.txt'%(tablename,cur_date), '/home/op1/lina/app_abtest_email_%s_%s.txt'%(tablename,cur_date))
sftp.get('/home/hotel/lina/app_abtest_email_detail_%s_%s.txt'%(tablename,cur_date), '/home/op1/lina/app_abtest_email_detail_%s_%s.txt'%(tablename,cur_date))

time.sleep(30) 

# Graph drawing on 77
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from pylab import *
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import pandas as pd 


result_all = pd.read_csv("/home/op1/lina/app_abtest_email_%s_%s.txt"%(tablename,cur_date),sep='\t')
result_detail = pd.read_csv("/home/op1/lina/app_abtest_email_detail_%s_%s.txt"%(tablename,cur_date),sep='\t')

#result_all = pd.read_csv("d:/Users/n.li/Desktop/app_abtest_email_151021_hod_MJSY_20160120.txt",sep='\t')
#result_detail = pd.read_csv("d:/Users/n.li/Desktop/app_abtest_email_detail_151021_hod_MJSY_20160120.txt",sep='\t')



def delete_tabname (table):
    default_colnames = table.columns
    colnames =  [line.split('.')[1] for line in default_colnames]
    table.columns = colnames

delete_tabname(result_detail)
result = result_all.append(result_detail)
result = result.drop(['experiment','d'], 1)

column_names = result.columns[3:]
plot_names = [line.split('_')[1] for line in column_names]
plot_names = sorted(set(plot_names),key=plot_names.index)
plot_names = [line for line in plot_names]

#pre-data-analyze
control_line = result[result["treatmentid"]=='Control']
test_lines = result[result["treatmentid"]!='Control']
test_versionname = list(test_lines['treatmentid'].drop_duplicates())
test_versionname.sort()

#trans plot_names to Chinese
myfont = matplotlib.font_manager.FontProperties(fname='/home/op1/ssh/msyh.ttf')  
#myfont = matplotlib.font_manager.FontProperties(fname="C:\Windows\Fonts\msyh.ttf", size=14)
name_dicts={'totalordersord':'CR','uncancelordersord':u'CR\n非取消',\
'totalgpcii':'GP','uncancelgpcii':u'GP\n非取消',\
'returngpcii':u'GP\n减返现','uncancelreturngpcii':u'GP\n非取消减返现'}
plot_names_CN = []
for line in plot_names:
    if(line in name_dicts.keys()):
        plot_names_CN.append(name_dicts[line])
    else:
        plot_names_CN.append(line)

#rank title_names and trans to Chinese
#anlyzclasslevel = control_line[['anlyzclass','classlevel']].sort(['anlyzclass','classlevel'])
anlyzclasslevel = control_line[['anlyzclass','classlevel']]
anlyzclasslevel['cbind'] = anlyzclasslevel['anlyzclass']+':'+anlyzclasslevel['classlevel']

#anlyzclass 
#anlyz_dicts={'all':' ','star':u'酒店星级','goldstar':u'酒店级别','citylevel':u'酒店城市',\
#'advancedgroup':u'提前天数','staydays':u'入住天数','hourgroup':u'预订时段','week':u'入住日',\
#'consumetag':u'用户消费能力','price_sensitive_tag':u'用户慷慨程度'}

anlyz_dicts = pd.DataFrame({'anlyzclass':['all',\
'star','goldstar','citylevel',\
'advancedgroup','staydays','hourgroup','week',\
'consumetag','price_sensitive_tag'\
],\
'name':[' ',\
u'酒店星级',u'酒店级别',u'酒店城市',\
u'提前天数',u'入住天数',u'预订时段',u'入住首日',\
u'用户消费能力',u'用户慷慨程度'\
]})

#classlevel
#delete 'mgrgroup'
ranknum = pd.DataFrame({'cbind':['all:all',\
'star:2','star:3','star:4','star:5',\
'goldstar:Premium','goldstar:GoldSilver','goldstar:NoGoldstar',\
'citylevel:LevelOne','citylevel:LevelTwoA','citylevel:LevelTwoB','citylevel:LevelThree','citylevel:GangAoTai',\
'advancedgroup:0','advancedgroup:1','advancedgroup:2','advancedgroup:3','advancedgroup:over3',\
'staydays:1','staydays:2','staydays:3','staydays:over3',\
'hourgroup:0AM8AM','hourgroup:8AM13AM','hourgroup:1PM6PM','hourgroup:6PM12PM',\
'week:Weekday','week:Weekend',\
'consumetag:0','consumetag:1','consumetag:2','consumetag:3','consumetag:noconsumetag',\
'price_sensitive_tag:0','price_sensitive_tag:1','price_sensitive_tag:2','price_sensitive_tag:3','price_sensitive_tag:nosensitivetag'\
],'rank':range(38),\
'name':[u'总体情况',\
u'二星酒店及以下',u'三星酒店',u'四星酒店',u'五星酒店',\
u'特牌酒店',u'金银牌酒店',u'非挂牌酒店',\
u'一线城市酒店',u'二线A城市酒店',u'二线B城市酒店',u'三线城市酒店',u'港澳台地区酒店',\
u'入住当天预订',u'提前一天预订',u'提前两天预订',u'提前三天预订',u'提前三天以上预订',\
u'入住一晚',u'入住两晚',u'入住三晚',u'入住三晚以上',\
u'0点至8点预订',u'8点至13点预订',u'13点至18点预订',u'18点至24点预订',\
u'工作日入住',u'周末入住',\
u'消费能力低用户',u'消费能力中等用户',u'消费能力较高用户',u'消费能力很高用户',u'无订单消费能力未知',\
u'非常小气用户',u'有点小气用户',u'有点大方用户',u'非常大方用户',u'无订单大方小气未知'\
]})
anlyz_level = pd.merge(anlyzclasslevel, ranknum, how='inner')
anlyz_level_sorted = anlyz_level.sort(['rank'])


#plot nums
plot_num={'all':1,'star':4,'goldstar':3,'citylevel':5,\
'advancedgroup':5,'staydays':4,'hourgroup':4,'week':2,\
'consumetag':5,'price_sensitive_tag':5}

def setaxes (columns_num, step, plot_names_CN, myfont, ax, size, linewidth):
    axes.cla()
    ax.set_xlim(0,step*columns_num+1) 
    ax.set_xticks(range(0,step*columns_num+1,1))
    labels_xtick = ['']
    labels_xtick.extend(plot_names_CN)
    ax.set_xticklabels(labels_xtick,size=size,fontproperties=myfont)   #axsize=20
    plt.yticks(size=size)
    plot([0,step*columns_num+1],[0,0],color='#FE0A0A', linewidth=linewidth)    #red   # linewidth=3

def drawmain (columns_num, step, test, control, ax, fontsize, linewidth, markersize):
    for m in xrange(columns_num):
        Test= np.random.normal(test[3+2*m], test[4+2*m], 10000)
        Control = np.random.normal(control[3+2*m], control[4+2*m], 10000)
        
        lift = (Test-Control)/Control
        high = np.percentile(lift,97.5)
        middle = np.mean(lift)
        low = np.percentile(lift,2.5)
                    
        x11=100*middle
        x1=[1+step*m,1+step*m]
        y1=[100*low,100*high]
        max_y = max(12,y1[1]*1.1)
        min_y = min(-12,y1[0]*1.1)
        ax.set_ylim(min_y,max_y) 
        
        #plot the lines
        if (y1[0]>0):
            plot(x1,y1,color='#2577E3', linewidth=linewidth)    #blue    #linewidth=3      
        elif (y1[1]<0):
            plot(x1,y1,color='#FF9913', linewidth=linewidth)    #yellow
        else:
            plot(x1,y1,color='#828282', linewidth=linewidth)    #grey
    
        #plot the dots
        if (y1[0]>0):
            plot(x1[0],x11,color='#2577E3', markeredgecolor='#2577E3', marker='o', markersize=markersize)    #markersize=8       
        elif (y1[1]<0):
            plot(x1[0],x11,color='#FF9913', markeredgecolor='#FF9913', marker='o', markersize=markersize)              
        else:
            plot(x1[0],x11,color='#828282', markeredgecolor='#828282', marker='o', markersize=markersize)              
        
        #plot number
        plt.text(x1[0]+0.1,x11,'%s%%'%round(x11,2), fontsize=fontsize)  #fontsize=20

def drawtitle (anlyzclass, classlevel, anlyz_level_sorted, n, fontsize):
    #anlyzclass_CH = anlyz_dicts.iloc[n,1]
    inner_cbind = anlyzclass + ':'+ classlevel
    anlyzclasslevel_line = anlyz_level_sorted[anlyz_level_sorted["cbind"]==inner_cbind]
    classlevel_CH = anlyzclasslevel_line.iloc[0,3]
    #plt.title(treat+' - '+anlyzclass_CH+' - '+classlevel_CH, fontsize=30,fontproperties=myfont)   
    #treat_v=treat.split('_')[1]    
    #plt.title(u'版本'+treat_v+' - '+classlevel_CH, fontsize=fontsize,fontproperties=myfont)   #fontsize=30
    plt.title(classlevel_CH, fontsize=fontsize,fontproperties=myfont)
    #pylab.grid(color='#E5E5E5',linestyle='-')  #light grey


pp = PdfPages('/home/op1/lina/app_abtest_email_sum_%s_%s.pdf'%(tablename,cur_date))
#pp = PdfPages('d:/Users/n.li/Desktop/app_abtest_email_sum_%s_%s.pdf'%(tablename,cur_date))

step = 1

for treat in (test_versionname):
    inner_test_lines = test_lines[test_lines["treatmentid"]==treat]
    print treat
    
    #draw teatment face
    fig = plt.figure(figsize=(16,11), facecolor="white",frameon=False)  
    treat_CH = u'版本'+treat.split('_')[1] 
    plt.text(0.5,0.5,treat_CH,fontsize=30,horizontalalignment='center',verticalalignment='center',fontproperties=myfont)
    #axes.cla()
    pp.savefig(fig)
    #pp.close()
    
    for n in xrange(anlyz_dicts.shape[0]):
        anlyzclass = anlyz_dicts.iloc[n,0]
        levelnums = plot_num[anlyzclass]
        
        #open one page
        fig = plt.figure(figsize=(16,11), facecolor="white")  
        #decide plot location and size
      
        for j in xrange(levelnums):
            anlyz_df = anlyz_level_sorted[anlyz_level_sorted["anlyzclass"]==anlyzclass].sort(['rank'])       
            classlevel = anlyz_df.iloc[j,1]
            #print anlyzclass,classlevel
            in_inner_test = inner_test_lines[(inner_test_lines["anlyzclass"]==anlyzclass) & (inner_test_lines["classlevel"]==classlevel)]
            in_inner_control = control_line[(control_line["anlyzclass"]==anlyzclass) & (control_line["classlevel"]==classlevel)]
            test = in_inner_test.iloc[0,:].values
            control = in_inner_control.iloc[0,:].values
            columns_num = len(plot_names)
                          
            if (levelnums <= 1):
                axes = plt.subplot(111+j) 
                ax=plt.gca()  #get current axes
                setaxes (columns_num, step, plot_names_CN, myfont, ax, 20, 3) 
                drawmain(columns_num, step, test, control, ax, 20, 3, 8)
                drawtitle (anlyzclass, classlevel, anlyz_level_sorted, n, 25)
            elif (levelnums <= 2):
                axes = plt.subplot(121+j) 
                ax=plt.gca()
                setaxes (columns_num, step, plot_names_CN, myfont, ax, 12, 2)
                drawmain(columns_num, step, test, control, ax, 12, 2, 6) 
                drawtitle (anlyzclass, classlevel, anlyz_level_sorted, n, 20)
            elif (levelnums <= 4):
                axes = plt.subplot(221+j)
                subplots_adjust(hspace = 0.3)
                ax=plt.gca()
                setaxes (columns_num, step, plot_names_CN, myfont, ax, 10, 2) 
                drawmain(columns_num, step, test, control, ax, 12, 2, 6)
                drawtitle (anlyzclass, classlevel, anlyz_level_sorted, n, 20)
            else:
                axes = plt.subplot(231+j)
                subplots_adjust(hspace = 0.4)
                ax=plt.gca()
                setaxes (columns_num, step, plot_names_CN, myfont, ax, 8, 2) 
                drawmain(columns_num, step, test, control, ax, 8, 2, 6)
                drawtitle (anlyzclass, classlevel, anlyz_level_sorted, n, 20)
                        
            #one page one title            
            anlyzclass_CH = anlyz_dicts.iloc[n,1]
            plt.suptitle(treat_CH+' - '+anlyzclass_CH, fontproperties=myfont, fontsize=100)
                                                                                                                                                              
        pp.savefig(fig) 
    #break #only on treat                
pp.close()

sftp.put('/home/op1/lina/app_abtest_email_sum_%s_%s.pdf'%(tablename,cur_date), '/home/hotel/lina/app_abtest_email_sum_%s_%s.pdf'%(tablename,cur_date))
time.sleep(120)


stdin,stdout,stderr=s.exec_command("echo 'AB测试细分各维度的分析结果详见附件，谢谢！' | sudo mutt w.jiang@Ctrip.com yihongchen@Ctrip.com n.li@Ctrip.com -s'个性化AB测试维度报表' -a /home/hotel/lina/app_abtest_email_sum_%s_%s.pdf"%(tablename,cur_date))
s.close()
t.close()















      
