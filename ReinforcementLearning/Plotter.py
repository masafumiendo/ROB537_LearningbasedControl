import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Plotter:

    def reward_history(self, result):

        fig = plt.figure(figsize=(12, 7))
        plt.plot(result[0, :], alpha=.3, c='g')
        plt.plot(result[1, :], alpha=.3, c='r')
        plt.plot(pd.Series(result[0, :]).rolling(window=50).mean(), label='Q-learning with fixed door', c='g')
        plt.plot(pd.Series(result[1, :]).rolling(window=50).mean(), label='Q-learning with moving door', c='r')
        plt.legend(bbox_to_anchor=(1, 0), loc='lower right', borderaxespad=1, fontsize=12)
        plt.savefig('fig/reward_history.png')
        plt.close(fig)

    def Q_value_history(self, Q_value, condition):

        fig, ax = plt.subplots()
        fig.set_size_inches(11.7, 8.27)
        sns.heatmap(data=Q_value, cmap="viridis", annot=True, linecolor="white", linewidths=.5, square=True, cbar=False)
        plt.tight_layout()
        plt.savefig('fig/Q_' + condition + '.png')
        