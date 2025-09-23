import pandas as pd
import matplotlib.pyplot as plt
f= "./harmonic_chain_2particles.csv"
df = pd.read_csv(f, header=None)

df["diff"]= df[3]-df[4]

print(df)
df.plot( y = 'diff')

plt.show()
