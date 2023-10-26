import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv(
    '90.csv', 
    sep=',',)

# median = data.median(numeric_only=True, axis=1)
# median.plot(kind='bar')
# plt.xticks(np.arange(0, 90, 5.0))
# plt.xlabel('Generation')
# plt.ylabel('Median Score')
# plt.show()

# plt.clf()
# sum = data.sum(numeric_only=True, axis=1)
# sum.plot(kind='bar')
# plt.xticks(np.arange(0, 90, 5.0))
# plt.xlabel('Generation')
# plt.ylabel('Sum')
# plt.show()

# plt.clf()
# top = data.max(numeric_only=True, axis=1)
# top.plot(kind='bar')
# plt.xticks(np.arange(0, 90, 5.0))
# plt.xlabel('Generation')
# plt.ylabel('Top Score')
# plt.show()

fig, axs = plt.subplots(3, figsize=(10, 10))
fig.suptitle('Score Over Generations')

median = data.median(numeric_only=True, axis=1)
median.plot(kind='bar', ax=axs[0], color='#F9B572')
axs[0].set_xticks(np.arange(0, 90, 5.0))
axs[0].set(ylabel='Median Score')
plt.show()

top = data.max(numeric_only=True, axis=1)
top.plot(kind='bar', ax=axs[1], color='#A1CCD1')
axs[1].set_xticks(np.arange(0, 90, 5.0))
axs[1].set(xlabel='Generation', ylabel='Top')
plt.show()

sum = data.sum(numeric_only=True, axis=1)
sum.plot(kind='bar',ax=axs[2], color='#99B080')
axs[2].set_xticks(np.arange(0, 90, 5.0))
axs[2].set(ylabel='Sum')
plt.show()

fig.tight_layout()
plt.savefig("score_graph.png")
