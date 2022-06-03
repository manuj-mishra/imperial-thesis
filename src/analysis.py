import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats

df = pd.read_csv('./taxonomy.csv')
df.density = df.rstring.apply(lambda x: bin(x).count("1"))

# # Percent plot
# sns.displot(df.density, discrete=True, stat='percent')

g = sns.jointplot(x = df.density, y=df.conv_perc, marker="+")
g.plot_joint(sns.kdeplot, fill=True)
g.plot_marginals(sns.displot, color="r")

plt.show()


dist = getattr(stats, 'norm')
parameters = dist.fit(df.density)
print(parameters)