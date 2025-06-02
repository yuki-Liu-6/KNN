import seaborn as sns
sns.countplot(x=y).set_title("Class Distribution")
plt.savefig("results/class_dist.png")