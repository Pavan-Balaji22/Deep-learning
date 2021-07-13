#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import pandas as pd

# reading data obtained from the experiments 
final_data = pd.read_csv("Final Data.csv",index_col=0) # without regularization
final_data_reg= pd.read_csv("Final Data_reg.csv",index_col=0) # with regularization


# In[10]:


# Preparing Data to Plot No of examples(N) vs Ein and eout

example1 = final_data[(final_data['Variance'] == final_data['Variance'].unique()[1] ) 
                     & (final_data['Degree of polynomial'] == 2)]

example2 = final_data[(final_data['Variance'] == final_data['Variance'].unique()[1] )
                     & (final_data['Degree of polynomial'] == 19)]

# Creating figure to plot

fig,axes = plt.subplots(nrows=1,ncols=2,figsize = (15,7))
fig.suptitle("No of examples(N) vs Errors (Ein,Eout)\n\nfor different degree of polynomial\n",size = 'x-large')

# Plotting in axes[0] for degree = 2 

axes[0].plot(example1['No of examples'],example1['Eout'],label ="Eout")
axes[0].plot(example1['No of examples'],example1['Ein'],label ="Ein")
axes[0].set_xlabel("No of Examples",size = 'x-large')
axes[0].set_ylabel("Mean Square Error",size = 'x-large')
axes[0].set_title("Degree of polynomial = {}\n".format(example1['Degree of polynomial'].iloc[0]),size = 'x-large')
axes[0].set_ylim(0,max(example1["Eout"].max(),example2["Eout"].max()))
axes[0].legend()

# Plotting in axes[1] for degree = 19

axes[1].plot(example2['No of examples'],example2['Eout'],label ="Eout")
axes[1].plot(example2['No of examples'],example2['Ein'],label ="Ein")
axes[1].set_xlabel("No of Examples",size = 'x-large')
axes[1].set_ylabel("Mean Square Error",size = 'x-large')
axes[1].set_title("Degree of polynomial = {}\n".format(example2['Degree of polynomial'].iloc[0]),size = 'x-large')
axes[1].set_ylim(0,max(example1["Eout"].max(),example2["Eout"].max()))
axes[1].legend()

fig.tight_layout()

# Saving figure

plt.savefig(fname="No of examples(N) vs Errors (Ein,Eout)1.png",dpi = 600)


# In[9]:


# Preparing Data to Plot No of examples(N) vs E_Bias

example1 = final_data[(final_data['Variance'] == final_data['Variance'].unique()[1] ) 
                     & (final_data['Degree of polynomial'] == 2)]

example2 = final_data[(final_data['Variance'] == final_data['Variance'].unique()[1] )
                     & (final_data['Degree of polynomial'] == 19)]

# Creating figure to plot

fig,axes = plt.subplots(nrows=1,ncols=2,figsize = (15,7))
fig.suptitle("No of examples(N) vs E_Bias\n\nfor different degree of polynomial\n",size = 'x-large')

# Plotting in axes[0] for degree = 2

axes[0].plot(example1['No of examples'],example1['E_bias'],label ="E_bias")
axes[0].set_xlabel("No of Examples",size = 'x-large')
axes[0].set_ylabel("Mean Square Error",size = 'x-large')
axes[0].set_title("Degree of polynomial = {}\n".format(example1['Degree of polynomial'].iloc[0]),size = 'x-large')
axes[0].set_ylim(0,max(example1["E_bias"].max(),example2["E_bias"].max()))
axes[0].legend()

# Plotting in axes[1] for degree = 19

axes[1].plot(example2['No of examples'],example2['E_bias'],label ="E_bias")
axes[1].set_xlabel("No of Examples",size = 'x-large')
axes[1].set_ylabel("Mean Square Error",size = 'x-large')
axes[1].set_title("Degree of polynomial = {}\n".format(example2['Degree of polynomial'].iloc[0]),size = 'x-large')
axes[1].set_ylim(0,max(example1["E_bias"].max(),example2["E_bias"].max()))
axes[1].legend()
fig.tight_layout()

# Saving figure

plt.savefig(fname="No of examples(N) vs E_Bias-1..png",dpi = 600)


# In[70]:


# Preparing Data to Plot Impact of noise on Error values

vexample1 = final_data[(final_data['No of examples'] == 100 ) 
                     & (final_data['Degree of polynomial'] == 20)]

# Creating figure to plot

fig,axes=plt.subplots(1,3,figsize = (12,6))

# Seperating data for the three variances

pl1_data=vexample1[vexample1['Variance'].apply(str) == vexample1['Variance'].apply(str).unique()[0]]
pl2_data=vexample1[vexample1['Variance'].apply(str) == vexample1['Variance'].apply(str).unique()[1]]
pl3_data=vexample1[vexample1['Variance'].apply(str) == vexample1['Variance'].apply(str).unique()[2]]

# Plotting in axes[0] for noise = 0.01

axes[0].bar(pl1_data['Variance'].apply(str),pl1_data['E_bias'],width =.5)
axes[0].bar((pl1_data['Variance']+1).apply(str),pl1_data['Ein'],width =.5)
axes[0].bar((pl1_data['Variance']+2).apply(str),pl1_data['Eout'],width =.5)
axes[0].set_xticks(axes[0].get_xticks())
axes[0].set_xticklabels(['E_bias','Ein','Eout'],size = 'x-large')
axes[0].set_title('Sigma = 0.01\n',size = 'x-large')
axes[0].set_ylim(0,max(pl1_data["Eout"].max(),pl2_data["Eout"].max(),pl3_data["Eout"].max()))

# Plotting in axes[1] for noise = 0.1

axes[1].bar(pl2_data['Variance'].apply(str),pl2_data['E_bias'],width =.5)
axes[1].bar((pl2_data['Variance']+1).apply(str),pl2_data['Ein'],width =.5)
axes[1].bar((pl2_data['Variance']+2).apply(str),pl2_data['Eout'],width =.5)
axes[1].set_xticks(axes[0].get_xticks())
axes[1].set_xticklabels(['E_bias','Ein','Eout'],size = 'x-large')
axes[1].set_title('Sigma = 0.1\n',size = 'x-large')
axes[1].set_ylim(0,max(pl1_data["Eout"].max(),pl2_data["Eout"].max(),pl3_data["Eout"].max()))

# Plotting in axes[2] for noise = 1

axes[2].bar(pl3_data['Variance'].apply(str),pl3_data['E_bias'],width =.5)
axes[2].bar((pl3_data['Variance']+1).apply(str),pl3_data['Ein'],width =.5)
axes[2].bar((pl3_data['Variance']+2).apply(str),pl3_data['Eout'],width =.5)
axes[2].set_xticks(axes[0].get_xticks())
axes[2].set_xticklabels(['E_bias','Ein','Eout'],size = 'x-large')
axes[2].set_title('Sigma = 1\n',size = 'x-large')
axes[2].set_ylim(0,max(pl1_data["Eout"].max(),pl2_data["Eout"].max(),pl3_data["Eout"].max()))
fig.legend(["E_bias","Ein","Eout"])
fig.suptitle("Impact of noise level on Error values\n",size = 'x-large')
fig.tight_layout()

# Saving figure

plt.savefig(fname="Impact of noise on Error values1.png",dpi = 600)


# In[6]:


# Preparing Data to Plot Degree of polynomial(d) vs Ein and eout

dexample1 = final_data[(final_data['Variance'] == final_data['Variance'].unique()[1] ) 
                     & (final_data['No of examples'] == 5)  & (final_data['Degree of polynomial'] > 0)]

dexample2 = final_data[(final_data['Variance'] == final_data['Variance'].unique()[1] ) 
                     & (final_data['No of examples'] ==50) & (final_data['Degree of polynomial'] > 0)]

# Creating figure to plot

fig,axes = plt.subplots(nrows=1,ncols=2,figsize = (15,7))
fig.suptitle("Degree of polynomial(d) vs Errors (Ein,Eout)\n\nfor different No of examples\n",size = 'x-large')

# Plotting in axes[0] for Number of examples = 5

axes[0].plot(dexample1['Degree of polynomial'],dexample1['Eout'],label ="Eout")
axes[0].plot(dexample1['Degree of polynomial'],dexample1['Ein'],label ="Ein")
axes[0].set_xlabel("Degree of polynomial",size = 'x-large')
axes[0].set_ylabel("Mean Square Error",size = 'x-large')
axes[0].set_xticks(range(0,21))
axes[0].set_title("No of examples = {}\n".format(dexample1['No of examples'].iloc[0]),size = 'x-large')
axes[0].set_ylim(0,max(dexample1["Eout"].max(),dexample2["Eout"].max()))
axes[0].legend()

# Plotting in axes[1] for Number of examples = 50

axes[1].plot(dexample2['Degree of polynomial'],dexample2['Eout'],label ="Eout")
axes[1].plot(dexample2['Degree of polynomial'],dexample2['Ein'],label ="Ein")
axes[1].set_xlabel("Degree of polynomial",size = 'x-large')
axes[1].set_ylabel("Mean Square Error",size = 'x-large')
axes[1].set_title("No of examples = {}\n".format(dexample2['No of examples'].iloc[0]),size = 'x-large')
axes[1].set_xticks(range(0,21))
axes[1].set_ylim(0,max(dexample1["Eout"].max(),dexample2["Eout"].max()))
axes[1].legend()
fig.tight_layout()

# Saving Figure

plt.savefig(fname="Degree of polynomial(d) vs Errors (Ein,Eout)1.png",dpi = 600)


# In[4]:


# Preparing Data to Plot Degree of polynomial(d) vs Ein and eout with regularization

rdexample1 = final_data_reg[(final_data_reg['Variance'] == final_data_reg['Variance'].unique()[1] ) 
                     & (final_data_reg['No of examples'] == 10)  & (final_data_reg['Degree of polynomial'] > 0)]

rdexample2 = final_data_reg[(final_data_reg['Variance'] == final_data_reg['Variance'].unique()[1] ) 
                     & (final_data_reg['No of examples'] ==200) & (final_data_reg['Degree of polynomial'] > 0)]

# Creating figure to plot

fig,axes = plt.subplots(nrows=1,ncols=2,figsize = (15,7))
fig.suptitle("Degree of polynomial(d) vs Errors (Ein,Eout)\n\nfor different No of examples\n\n with regularisation\n",size = 'x-large')

# Plotting in axes[0] for Number of examples = 10

axes[0].plot(rdexample1['Degree of polynomial'],rdexample1['Eout'],label ="Eout")
axes[0].plot(rdexample1['Degree of polynomial'],rdexample1['Ein'],label ="Ein")
axes[0].set_xlabel("Degree of polynomial",size = 'x-large')
axes[0].set_ylabel("Mean Square Error",size = 'x-large')
axes[0].set_xticks(range(0,21))
axes[0].set_title("No of examples = {}\n".format(rdexample1['No of examples'].iloc[0]),size = 'x-large')
axes[0].set_ylim(0,max(rdexample1["Eout"].max(),rdexample2["Eout"].max()))
axes[0].legend()

# Plotting in axes[1] for Number of examples = 200

axes[1].plot(rdexample2['Degree of polynomial'],rdexample2['Eout'],label ="Eout")
axes[1].plot(rdexample2['Degree of polynomial'],rdexample2['Ein'],label ="Ein")
axes[1].set_xlabel("Degree of polynomial",size = 'x-large')
axes[1].set_ylabel("Mean Square Error",size = 'x-large')
axes[1].set_title("No of examples = {}\n".format(rdexample2['No of examples'].iloc[0]),size = 'x-large')
axes[1].set_xticks(range(0,21))
axes[1].set_ylim(0,max(rdexample1["Eout"].max(),rdexample2["Eout"].max()))
axes[1].legend()
fig.tight_layout()

# Saving Figure

plt.savefig(fname="Degree of polynomial(d) vs Errors (Ein,Eout)2.png",dpi = 600)


# In[69]:


# Preparing Data to Plot Impact of Regularisation on Error values

rexample1 = final_data[(final_data['No of examples'] > 50)  
                    & (final_data['Variance'].apply(str) == final_data['Variance'].apply(str).unique()[1])
                     & (final_data['Degree of polynomial'] == 20)]

rexample2 = final_data_reg[(final_data_reg['No of examples'] > 50 )
                           & (final_data['Variance'].apply(str) == final_data['Variance'].apply(str).unique()[1])
                             & (final_data_reg['Degree of polynomial'] == 20)]

# Creating figure to plot

fig,axes=plt.subplots(1,3,figsize = (8,6))
fig.suptitle("Impact of Regularisation on Error values\n",size = 'x-large')

# Plotting values of E_bais in axes[0] 

axes[0].bar('not\nregularised',rexample1['E_bias'],width = .5)
axes[0].bar('regularised',rexample2['E_bias'],width = .5)
axes[0].set_title('E_bias\n',size = 'x-large')
axes[0].set_ylim(0,max(rexample1[['Ein','Eout','E_bias']].max().max(),rexample2[['Ein','Eout','E_bias']].max().max())+.1)

# Plotting values of Ein in axes[1] 

axes[1].bar('not\nregularised',rexample1['Ein'],width = .5)
axes[1].bar('regularised',rexample2['Ein'],width = .5)
axes[1].set_title('Ein\n',size = 'x-large')
axes[1].set_ylim(0,max(rexample1[['Ein','Eout','E_bias']].max().max(),rexample2[['Ein','Eout','E_bias']].max().max())+.1)

# Plotting values of Eout  in axes[2] 

axes[2].bar('not\nregularised',rexample1['Eout'],width = .5)
axes[2].bar('regularised',rexample2['Eout'],width = .5)
axes[2].set_title('Eout\n',size = 'x-large')
axes[2].set_ylim(0,max(rexample1[['Ein','Eout','E_bias']].max().max(),rexample2[['Ein','Eout','E_bias']].max().max())+.1)
fig.legend(["Not regularised","Regularised"])

fig.tight_layout()

# Saving Figure

plt.savefig(fname="Impact of Regularisation on Error values.png",dpi = 600)


# In[ ]:




