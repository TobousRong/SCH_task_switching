import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

# # 加载数据
data = np.load('D:/taskswitch100/SCHZ/RT/RT_roc_data.npz')
RT_mean_fpr = data['mean_fpr1']
RT_mean_tpr = data['mean_tpr1']
RT_roc_auc=data['roc_auc1']
# data = np.load('D:/taskswitch100/SCHZ/RT/noRT_roc_data.npz')
# noRT_mean_fpr = data['mean_fpr2']
# noRT_mean_tpr = data['mean_tpr2']
# noRT_roc_auc=data['roc_auc2']
# 加载数据
data = np.load('D:/taskswitch100/SCHZ/RT/single_RT_roc_data.npz')
single_RT_mean_fpr = data['mean_fpr3']
single_RT_mean_tpr = data['mean_tpr3']
single_RT_roc_auc=data['roc_auc3']
# data = np.load('D:/taskswitch100/SCHZ/RT/single_noRT_roc_data.npz')
# single_noRT_mean_fpr = data['mean_fpr4']
# single_noRT_mean_tpr = data['mean_tpr4']
# single_noRT_roc_auc=data['roc_auc4']
# # DEG
# data = np.load('D:/taskswitch100/SCHZ/DEGRT_results/DEGRT_roc_data.npz')
# DEGRT_mean_fpr = data['mean_fpr6']
# DEGRT_mean_tpr = data['mean_tpr6']
# DEGRT_roc_auc=data['roc_auc6']
# # PC
# data = np.load('D:/taskswitch100/SCHZ/PCRT_results/PCRT_roc_data.npz')
# PCRT_mean_fpr = data['mean_fpr5']
# PCRT_mean_tpr = data['mean_tpr5']
# PCRT_roc_auc=data['roc_auc5']
# 绘制ROC曲线
plt.figure(figsize=(8, 6))

# 绘制第一个模型的ROC曲线
plt.plot(RT_mean_fpr, RT_mean_tpr, color='black', lw=2, label='Multimodal(AUC = %0.2f)' % RT_roc_auc)
# plt.plot(noRT_mean_fpr,noRT_mean_tpr, color='grey', lw=2, label='RT not regressed (AUC = %0.2f)' % noRT_roc_auc)
# 绘制第二个模型的ROC曲线
plt.plot(single_RT_mean_fpr, single_RT_mean_tpr, color='dimgrey', lw=2, label='Unimodal  (AUC = %0.2f)' % single_RT_roc_auc)
# plt.plot(single_noRT_mean_fpr, single_noRT_mean_tpr, color='grey', lw=2, label='Unimodal (AUC = %0.2f)' % single_noRT_roc_auc)
# plt.plot(DEGRT_mean_fpr, DEGRT_mean_tpr, color='steelblue', lw=2, label='Deg(AUC = %0.2f)' % DEGRT_roc_auc)
# plt.plot(PCRT_mean_fpr, PCRT_mean_tpr, color='lightsteelblue', lw=2, label='PC(AUC = %0.2f)' % PCRT_roc_auc)
# 绘制对角线
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# 设置图表属性
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.tick_params(labelsize=20)
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
# plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right",fontsize=20)
# 保存为TIFF格式
plt.savefig('D:/taskswitch100/SCHZ/MU.tiff', format='tiff', dpi=300)  # 指定文件名、格式和DPI
# 显示图表
plt.show()
