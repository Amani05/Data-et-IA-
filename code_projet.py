import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('archive/googleplaystore.csv')
df_reviews = pd.read_csv('archive/googleplaystore_user_reviews.csv')

fichier_excel = 'C:/Users/User/Documents/L3_sdn/Aclab/archive/googleplaystore.xlsx'
fichier_review_excel = 'C:/Users/User/Documents/L3_sdn/Aclab/archive/googleplaystore_user_reviews.xlsx'

# Pour exporter le fichier en excel pour une meilleur lisibilité
# df.to_excel(fichier_excel, index=False)
# df_reviews.to_excel(fichier_review_excel, index=False)


print(df.shape)

print(df_reviews.shape)
print(df.dtypes)
print(df_reviews.dtypes)
print(df.isna().sum())
print(df_reviews.isna().sum())
df_reviews.dropna(subset=['Translated_Review'], inplace=True)
print(df_reviews.isna().sum())
df.dropna(subset=['Rating'], inplace=True)
print(df.isna().sum())
# l'id 10472 a une valuer Rating > 5.0
df_rating_fuzzy = df[df['Rating'] > 5] 
df_rating_fuzzy
df.drop(index=10472, inplace=True)
df['Type'].value_counts()

df['Genres'].value_counts()
# Liste des genres à conserver
category_to_keep = ["FAMILY", "BUSINESS", "TOOLS", "PRODUCTIVITY"]

# Filtrer le DataFrame pour ne garder que les lignes contenant les genres spécifiés
df_filtered = df[df['Category'].isin(category_to_keep)]


plt.figure(figsize=(10, 6))
sns.boxplot(data=df_filtered, x='Category', y='Rating')
plt.title('Distribution des Ratings par Catégorie')
plt.xlabel('Catégorie')
plt.ylabel('Rating')
plt.xticks(rotation=45)  # Ajuster la rotation des étiquettes x si nécessaire
plt.show()


# Calculer le premier quartile (Q1) et le troisième quartile (Q3)
Q1 = df_filtered.groupby('Category')['Rating'].quantile(0.25)
Q3 = df_filtered.groupby('Category')['Rating'].quantile(0.75)

# Calcul de l'IQR pour chaque catégorie
IQR = Q3 - Q1

# Définir le seuil pour les outliers bas
threshold_low = Q1 - 1.1 * IQR

# Appliquer le filtre pour supprimer les outliers bas
for category in df_filtered['Category'].unique():
    filter_condition = (df_filtered['Category'] == category) & (df_filtered['Rating'] < threshold_low[category])
    df_filtered = df_filtered[~filter_condition]





    # Convertir 'Reviews' en entier
df_filtered['Reviews'] = pd.to_numeric(df_filtered['Reviews'], errors='coerce')

# Nettoyer et convertir 'Size' en megabytes
def convert_size(size):
    if 'M' in size:
        return float(size.replace('M', ''))
    elif 'k' in size:
        return float(size.replace('k', '')) / 1024  # Convertir k en M
    return 0
    # return 'Varies with device' 

df_filtered['Size'] = df_filtered['Size'].map(convert_size)

# Convertir la colonne 'Installs' en un format numérique propre pour le traitement
df_filtered['Installs'] = df_filtered['Installs'].str.replace('+', '').str.replace(',', '').astype(int)

# Nettoyer et convertir 'Price' en float
df_filtered['Price'] = df_filtered['Price'].str.replace('$', '').astype(float)

# Convertir 'Last Updated' en format de date
df_filtered['Last Updated'] = pd.to_datetime(df_filtered['Last Updated'])

# Afficher les types de données modifiés et les premières lignes pour vérification
df_filtered.head(), df_filtered.dtypes




# Distribution des catégories d'applications
plt.figure(figsize=(12, 6))
sns.countplot(y=df_filtered['Category'], order=df_filtered['Category'].value_counts().index)
plt.title("Distribution des Catégories d'Applications")
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(data=df_filtered, x='Category', y='Rating')
plt.title('Distribution des Ratings par Catégorie')
plt.xlabel('Catégorie')
plt.ylabel('Rating')
plt.xticks(rotation=45)  # Ajuster la rotation des étiquettes x si nécessaire
plt.show()

# Distribution des prix
plt.figure(figsize=(10, 5))
sns.histplot(df_filtered['Price'], bins=30, kde=True)
plt.title('Distribution des Prix des Applications')
plt.show()





# Trouver l'application la plus populaire par catégorie
most_popular_apps = df_filtered.loc[df_filtered.groupby('Category')['Installs'].idxmax()]

# # Afficher les applications les plus populaires par catégorie
print(most_popular_apps[['Category', 'App', 'Installs', 'Rating']])





# Convertir 'Last Updated' en datetime
df_filtered['Last Updated'] = pd.to_datetime(df_filtered['Last Updated'])

# Calculer la moyenne des notes par année
ratings_over_time = df_filtered.groupby(df_filtered['Last Updated'].dt.year)['Rating'].mean()

# Tracer la tendance des notes au fil du temps
plt.figure(figsize=(10, 5))
ratings_over_time.plot(kind='line', marker='o')
plt.title('Tendance des Notes Moyennes par Année')
plt.xlabel('Année')
plt.ylabel('Note Moyenne')
plt.grid(True)
plt.show()





# Tracer un scatter plot pour examiner la relation entre les notes et le nombre de téléchargements
plt.figure(figsize=(10, 6))
plt.scatter(df_filtered['Rating'], df_filtered['Installs'], alpha=0.5)
plt.title('Relation entre les Notes et le Nombre de Téléchargements')
plt.xlabel('Rating')
plt.ylabel('Installs ')
plt.yscale('log')  
plt.grid(True)
plt.show()




# Créer un scatter plot pour chaque catégorie
g = sns.FacetGrid(df_filtered, col="Category", col_wrap=4, height=4)
g.map(sns.scatterplot, "Rating", "Installs")

# Ajuster les échelles pour voir les tendances dans chaque catégorie
g.set(yscale="log")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Relation entre les Notes et les Téléchargements par Catégorie')

# Afficher les graphiques
plt.show()





# Appliquer la transformation logarithmique directement aux données
df_filtered['Log Installs'] = np.log10(df_filtered['Installs'] + 1)  # Ajouter 1 pour éviter le log de zéro

# Créer un jointplot avec la transformation appliquée
sns.jointplot(x='Rating', y='Log Installs', data=df_filtered, kind='hex', color='blue')
plt.show()






from sklearn.preprocessing import OneHotEncoder


# Initialiser OneHotEncoder
encoder = OneHotEncoder()  # Retourne une matrice sparse par défaut

# Sélectionner les colonnes catégorielles
categorical_columns = ['Type', 'Content Rating']
categorical_data = df_filtered[categorical_columns]

# Appliquer OneHotEncoder sur les données catégorielles
encoded_data = encoder.fit_transform(categorical_data)

# Créer un DataFrame avec les données encodées à partir de la matrice sparse
encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(categorical_columns))

# Concaténer le DataFrame encodé avec les autres données
# Assurez-vous de réinitialiser l'index si nécessaire pour éviter des problèmes de concaténation
df_filtered.reset_index(drop=True, inplace=True)
encoded_df.reset_index(drop=True, inplace=True)
full_data = pd.concat([df_filtered, encoded_df], axis=1)

# Supprimer les colonnes catégorielles originales car elles sont maintenant remplacées par leur version encodée
full_data.drop(columns=categorical_columns, inplace=True)

# Afficher les premières lignes du nouveau DataFrame pour vérification
full_data.head()




# Définir la cible en fonction des catégories d'intérêt
full_data['Target'] = full_data['Category'].apply(
    lambda x: 'Demoli' if x in ['TOOLS', 'PRODUCTIVITY'] else ('Ajmi' if x in ['FAMILY', 'BUSINESS'] else 'None')
)



from sklearn.model_selection import train_test_split

# Sélectionner les features et la cible
X = full_data.drop(columns=['Target', 'App', 'Category', 'Genres', 'Last Updated', 'Current Ver', 'Android Ver'])  # Exclure les colonnes non numériques et la cible
y = full_data['Target']

# Diviser les données en ensembles d'apprentissage et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




from sklearn.linear_model import LogisticRegression

# Initialiser le modèle de régression logistique
model = LogisticRegression(max_iter=1000)

# Entraîner le modèle
model.fit(X_train, y_train)



from sklearn.metrics import accuracy_score

# Prédire sur l'ensemble de test
y_pred = model.predict(X_test)

# Calculer la précision
accuracy = accuracy_score(y_test, y_pred)
print("Précision du modèle : {:.2f}%".format(accuracy * 100))



# Filtrer les applications pour Mme Ajmi et M. Demoli
predictions = model.predict(full_data.drop(columns=['Target', 'App', 'Category', 'Genres', 'Last Updated', 'Current Ver', 'Android Ver']))
full_data['Prediction'] = predictions

# Afficher les résultats filtrés
filtered_apps_demoli = full_data[full_data['Prediction'] == 'Demoli']
filtered_apps_ajmi = full_data[full_data['Prediction'] == 'Ajmi']



full_data[full_data['Target'] == 'Ajmi']



full_data[full_data['Target'] == 'Demoli']



from sklearn.neighbors import KNeighborsClassifier

# Initialiser le modèle KNN
knn = KNeighborsClassifier(n_neighbors=5)



# Entraîner le modèle
knn.fit(X_train, y_train)



from sklearn.metrics import accuracy_score

# Prédire sur l'ensemble de test
y_pred = knn.predict(X_test)

# Calculer la précision
accuracy = accuracy_score(y_test, y_pred)
print("Précision du modèle KNN : {:.2f}%".format(accuracy * 100))


from sklearn.preprocessing import StandardScaler

# Initialiser le StandardScaler
scaler = StandardScaler()

# Standardiser les données d'entraînement et de test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entraîner à nouveau le modèle KNN avec les données standardisées
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
print("Précision après standardisation : {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))



from sklearn.model_selection import GridSearchCV

# Grille de paramètres à tester
param_grid = {'n_neighbors': range(1, 20)}

# Recherche par grille avec validation croisée
knn_cv = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
knn_cv.fit(X_train_scaled, y_train)

# Meilleur nombre de voisins
print("Meilleur nombre de voisins:", knn_cv.best_params_)
print("Meilleure précision obtenue : {:.2f}%".format(knn_cv.best_score_ * 100))


# Tester avec la distance de Manhattan
knn_manhattan = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
knn_manhattan.fit(X_train_scaled, y_train)
y_pred_manhattan = knn_manhattan.predict(X_test_scaled)
print("Précision avec Manhattan : {:.2f}%".format(accuracy_score(y_test, y_pred_manhattan) * 100))


from sklearn.feature_selection import SelectKBest

# Sélectionner les K meilleures caractéristiques
selector = SelectKBest(k=10)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Réentraîner le modèle
knn.fit(X_train_selected, y_train)
y_pred_selected = knn.predict(X_test_selected)
print("Précision avec les caractéristiques sélectionnées : {:.2f}%".format(accuracy_score(y_test, y_pred_selected) * 100))


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Créer le modèle d'arbre de décision
tree = DecisionTreeClassifier(max_depth=5, random_state=42)  # Limiter la profondeur de l'arbre pour éviter le surajustement

# Entraîner le modèle
tree.fit(X_train, y_train)

# Prédire sur l'ensemble de test
y_pred = tree.predict(X_test)

# Calculer la précision
accuracy = accuracy_score(y_test, y_pred)
print("Précision de l'arbre de décision : {:.2f}%".format(accuracy * 100))

# Utiliser le modèle pour filtrer les applications selon les intérêts
full_data['Prediction'] = tree.predict(X)

# Filtrer et afficher les résultats
filtered_apps_demoli = full_data[full_data['Prediction'] == 'Demoli']
filtered_apps_ajmi = full_data[full_data['Prediction'] == 'Ajmi']
print(filtered_apps_demoli.head())
print(filtered_apps_ajmi.head())


from sklearn.tree import plot_tree

# Supposons que 'tree' est votre modèle d'arbre de décision entraîné
plt.figure(figsize=(20,10))
plot_tree(tree, filled=True, feature_names=X.columns, class_names=['Ajmi', 'Demoli', 'None'], rounded=True)
plt.show()



from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Créer le modèle d'arbre de décision
tree = DecisionTreeClassifier(max_depth=3, random_state=42)  # Limiter la profondeur de l'arbre pour éviter le surajustement

# Entraîner le modèle
tree.fit(X_train, y_train)

# Prédire sur l'ensemble de test
y_pred = tree.predict(X_test)

# Calculer la précision
accuracy = accuracy_score(y_test, y_pred)
print("Précision de l'arbre de décision : {:.2f}%".format(accuracy * 100))

# Utiliser le modèle pour filtrer les applications selon les intérêts
full_data['Prediction'] = tree.predict(X)

# Filtrer et afficher les résultats
filtered_apps_demoli = full_data[full_data['Prediction'] == 'Demoli']
filtered_apps_ajmi = full_data[full_data['Prediction'] == 'Ajmi']
print(filtered_apps_demoli.head())
print(filtered_apps_ajmi.head())



from sklearn.tree import plot_tree

# Supposons que 'tree' est votre modèle d'arbre de décision entraîné
plt.figure(figsize=(20,10))
plot_tree(tree, filled=True, feature_names=X.columns, class_names=['Ajmi', 'Demoli', 'None'], rounded=True)
plt.show()



