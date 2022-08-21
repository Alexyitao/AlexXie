import matplotlib.pyplot as plt
import numpy as np

n_jobs= []
max_depth = ['1','2','4','6','9','12','16','18','22']
accuracy = [16.92,28.42,55.42,74.25,94,97.67,97.67,98.17,98.67]

plt.plot(max_depth,accuracy)
#plt.ylim(95,100)
plt.title("The relationship between accuracy and max_depth")
plt.xlabel("The value of max_depth",fontsize = 14)
plt.ylabel("Accuracy(%)",fontsize = 14)
plt.show()