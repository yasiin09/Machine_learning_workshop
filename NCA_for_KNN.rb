
nca = NeighborhoodComponentsAnalysis(n_components = 2, random_state = 42)
nca.fit(x_scaled, y)
X_nca = nca.transform(x_scaled)

# Uyguladığımız veriyi görselleştiriyoruz
nca_data = pd.DataFrame(X_nca, columns = ["p1","p2"])
nca_data["target"] = y
sns.scatterplot(x = "p1",  y = "p2", hue = "target", data = X_nca)
plt.title("NCA: p1 vs p2")

X_train_nca, X_test_nca, Y_train_nca, Y_test_nca = train_test_split(X_nca, y, test_size = 0.3, random_state = 42)


