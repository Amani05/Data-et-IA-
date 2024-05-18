### Introduction

Dans le cadre de ce projet, nous avons analysé des données provenant du Play Store afin d'aider deux utilisateurs spécifiques, Mme Ajmi et M. Demoli, à sélectionner les applications les plus pertinentes en fonction de leurs intérêts et préférences. Les utilisateurs sont souvent confrontés à un grand nombre d'applications similaires, rendant difficile la sélection de l'application la plus adaptée à leurs besoins spécifiques.

**Données récoltées :**
1. **`googleplaystore.csv`** :
   - **App** : Nom de l'application.
   - **Category** : Catégorie de l'application.
   - **Rating** : Note de l'application.
   - **Reviews** : Nombre d'évaluations.
   - **Size** : Taille de l'application.
   - **Installs** : Nombre de téléchargements/installations.
   - **Type** : Type d'application (gratuite ou payante).
   - **Price** : Prix de l'application.
   - **Content Rating** : Groupe d'âge cible de l'application.
   - **Genres** : Genre(s) de l'application.
   - **Last Updated** : Date de la dernière mise à jour.
   - **Current Ver** : Version actuelle de l'application.
   - **Android Ver** : Version d'Android requise pour l'application.

2. **`googleplaystore_user_reviews.csv`** :
   - **App** : Nom de l'application.
   - **Translated_Review** : Revue traduite en anglais.
   - **Sentiment** : Sentiment exprimé dans la revue.
   - **Sentiment_Polarity** : Polarité du sentiment (de -1 à 1).
   - **Sentiment_Subjectivity** : Subjectivité du sentiment (de 0 à 1).

**Questions d'intérêt :**
1. **Quels sont les facteurs qui influencent le classement et la popularité des applications ?**
   - Analyse des attributs tels que le nombre de téléchargements, les évaluations, la taille, et la catégorie pour déterminer leur impact sur les notes des applications.
2. **Comment pouvons-nous prédire les catégories d'applications les plus pertinentes pour Mme Ajmi et M. Demoli ?**
   - Développement de modèles de machine learning pour classer les applications en fonction des préférences des utilisateurs :
     - Mme Ajmi est intéressée par les applications des catégories `FAMILY` et `BUSINESS`.
     - M. Demoli est intéressé par les applications des catégories `TOOLS` et `PRODUCTIVITY`.
3. **Quelle est la polarité et la subjectivité des avis des utilisateurs et comment influencent-ils la perception des applications ?**
   - Utilisation des données des avis pour prédire la polarité et la subjectivité en fonction des caractéristiques des applications.

Ce projet vise à développer un modèle de machine learning capable de filtrer les applications pour ne voir que celles qui sont les plus pertinentes pour Mme Ajmi et M. Demoli, en fonction de leurs intérêts, de leurs préférences et de leurs besoins.

---



### Choix d'Analyse pour Répondre aux Questions d'Intérêt

Pour répondre aux questions d'intérêt posées dans notre projet, nous avons adopté une approche méthodique en utilisant des techniques d'analyse exploratoire des données (EDA) et de machine learning.

**Exploration Initiale des Données**

Nous avons commencé par examiner la forme, les types de données et les valeurs manquantes dans nos jeux de données pour comprendre leur structure et identifier les anomalies. Cela nous a permis de garantir que chaque colonne était du type approprié et de déterminer les étapes nécessaires pour nettoyer les données.

**Traitement des Valeurs Manquantes et Anormales**

Les lignes avec des valeurs manquantes dans des colonnes critiques ont été supprimées pour garantir l'intégrité des données. Nous avons également corrigé les valeurs anormales, telles que les notes dépassant l'échelle valide, en les supprimant.

**Filtrage des Données et Visualisation**

Pour cibler les intérêts spécifiques de Mme Ajmi et M. Demoli, nous avons filtré les données pour ne conserver que les catégories pertinentes (`FAMILY`, `BUSINESS`, `TOOLS`, et `PRODUCTIVITY`). Des visualisations, telles que des boxplots, ont été utilisées pour explorer la distribution des notes par catégorie, permettant ainsi d'identifier les tendances et les anomalies.

**Nettoyage et Transformation des Données**

Nous avons converti les colonnes telles que la taille, le nombre d'installations, et le prix en formats numériques appropriés. Les outliers ont été identifiés et supprimés pour éviter que les valeurs extrêmes n'affectent nos analyses et modèles.

**Définition de la Target**

La variable cible (`target`) pour nos modèles de machine learning est la colonne `Category`. Cela nous permet de prédire la catégorie des applications en fonction des caractéristiques disponibles.

**Justification**

Ces étapes étaient nécessaires pour garantir que les données étaient propres et prêtes pour l'entraînement des modèles de machine learning. La visualisation des données a aidé à identifier les tendances et les relations entre les variables, essentielles pour répondre aux questions d'intérêt de Mme Ajmi et M. Demoli. En nettoyant les données et en traitant les valeurs manquantes et anormales, nous avons amélioré la qualité des résultats et la performance des modèles.

---

### Conclusion

Les choix d'analyse ont été guidés par la nécessité de préparer les données pour une modélisation efficace. Les étapes de nettoyage, de transformation et de visualisation des données ont assuré que les données étaient prêtes pour l'entraînement des modèles de machine learning et pour l'extraction d'informations pertinentes pour les utilisateurs cibles.

---

### Qualité des Visualisations

Les visualisations jouent un rôle crucial dans la compréhension et l'interprétation des données. Voici une description et une analyse de la qualité des visualisations utilisées dans notre projet :

**Distribution des Catégories d'Applications**

- **Description** : Un graphique à barres horizontales montre la distribution des applications dans les catégories d'intérêt (`FAMILY`, `BUSINESS`, `TOOLS`, `PRODUCTIVITY`).
- **Qualité** : Les catégories sont clairement étiquetées, et les fréquences sont faciles à comparer grâce aux barres colorées.

**Distribution des Ratings par Catégorie**

- **Description** : Un boxplot visualise la distribution des notes pour chaque catégorie, montrant la médiane, les quartiles et les outliers.
- **Qualité** : Les labels des axes sont clairs, et les boîtes et moustaches permettent de visualiser la dispersion et les outliers des notes.

**Distribution des Prix**

- **Description** : Un histogramme montre la distribution des prix des applications, accompagné d'une courbe de densité pour une meilleure visualisation.
- **Qualité** : Les bins sont clairement définis, et la courbe KDE aide à visualiser la densité, rendant la distribution facile à interpréter.

**Applications les Plus Populaires par Catégorie**

- **Description** : Une table montre les applications les plus populaires dans chaque catégorie, basée sur le nombre d'installations.
- **Qualité** : Les données sont présentées dans un format tabulaire facile à lire, avec des colonnes montrant les informations clés.

**Tendance des Notes Moyennes par Année**

- **Description** : Un graphique linéaire montre l'évolution des notes moyennes des applications au fil des ans.
- **Qualité** : Les années et les notes moyennes sont clairement marquées, et la tendance est facile à suivre grâce aux marqueurs et à la ligne.

**Relation entre les Notes et le Nombre de Téléchargements**

- **Description** : Un scatter plot examine la relation entre les notes des applications et le nombre de téléchargements.
- **Qualité** : Les axes sont bien étiquetés, et l'échelle logarithmique pour les téléchargements aide à gérer la large plage de valeurs.

**Scatter Plot par Catégorie**

- **Description** : Des scatter plots pour chaque catégorie montrent la relation entre les notes et les téléchargements.
- **Qualité** : Chaque catégorie est séparée dans son propre graphique, facilitant la comparaison, et l'échelle logarithmique aide à visualiser les différences dans les téléchargements.

**Joint Plot avec Transformation Logarithmique**

- **Description** : Un joint plot hexagonal montre la relation entre les notes et les téléchargements après une transformation logarithmique.
- **Qualité** : Les hexagones colorés indiquent la densité des points, et la transformation logarithmique rend les données plus maniables et met en évidence les tendances.

---

### Conclusion

Les visualisations utilisées dans ce projet sont essentielles pour comprendre et interpréter les données du Play Store. Elles permettent d'identifier des tendances, des relations et des anomalies cruciales pour répondre aux questions d'intérêt de Mme Ajmi et M. Demoli. La lisibilité et la clarté de ces visualisations assurent qu'elles sont auto-suffisantes et faciles à interpréter, aidant ainsi à tirer des conclusions valides et pertinentes.

---

