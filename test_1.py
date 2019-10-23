#anthor:williams_wang


from copula import pyCopula
import numpy as np
a1=np.array([np.random.rand(10)])
a2=np.array([np.random.rand(10)])
a3=np.array([np.random.rand(10)])
aa=np.concatenate((a1,a2),axis=0)
#aa=np.concatenate((aa,a3),axis=0)

bb1=([1,2,3,2,1,4],[2,2,5,3,1,4],[1,7,5,8,88,7])
#bb1=np.array(bb1)
print(bb1)

c1=pyCopula.Copula(bb1)
s1=c1.gendata(5)
print(s1)

print(aa)
#cc=pyCopula.Copula(aa)
#samp=cc.gendata(5)
#print(samp)
