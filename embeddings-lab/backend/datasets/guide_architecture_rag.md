---
title: "Guide d'architecture RAG et de recherche vectorielle"
subtitle: "Embeddings 384 dimensions, chunking, overlap, similarité cosinus et évaluation"
reference: "DT-GUI-2026-008"
version: "3.2"
author: "Cellule Innovation — Julien Marchetti, architecte logiciel"
service: "Direction Technique — Cellule Innovation"
classification: "Interne — Équipes techniques"
date: "2026-06-30"
domain: "rag"
keywords: ["RAG", "embeddings", "384 dimensions", "chunking", "overlap", "cosinus", "base vectorielle", "HNSW", "reranking", "Precision@K"]
---

# Guide d'architecture RAG et de recherche vectorielle

> Embeddings 384 dimensions, chunking, overlap, similarité cosinus et évaluation

## Résumé

Ce guide décrit les principes de conception d'une architecture de génération augmentée par la recherche, dite RAG. Il couvre la représentation vectorielle des textes par embeddings de dimension 384, la mesure de similarité par cosinus, les stratégies de découpage des documents en fragments et le dimensionnement du recouvrement, l'indexation par graphe navigable hiérarchique, la recherche hybride combinant correspondance lexicale et sémantique, le réordonnancement par modèle de reclassement, la construction du contexte de génération et l'ancrage des réponses, ainsi que le protocole d'évaluation fondé sur Precision@K, Recall@K, MRR et NDCG. Il se conclut par un inventaire des pièges les plus fréquents et une liste de contrôle de mise en production.

## Sommaire

1. [Vue d'ensemble d'une architecture RAG](#1-vue-d-ensemble-d-une-architecture-rag)
2. [Embeddings : représenter le sens par un vecteur de dimension 384](#2-embeddings-representer-le-sens-par-un-vecteur-de-dimension-384)
3. [Similarité cosinus et géométrie de l'espace vectoriel](#3-similarite-cosinus-et-geometrie-de-l-espace-vectoriel)
4. [Ingestion documentaire : extraction et normalisation](#4-ingestion-documentaire-extraction-et-normalisation)
5. [Chunking : stratégies de découpage en fragments](#5-chunking-strategies-de-decoupage-en-fragments)
6. [Overlap : rôle et dimensionnement du recouvrement](#6-overlap-role-et-dimensionnement-du-recouvrement)
7. [Indexation vectorielle : structures et paramètres](#7-indexation-vectorielle-structures-et-parametres)
8. [Recherche hybride et réordonnancement](#8-recherche-hybride-et-reordonnancement)
9. [Construction du contexte et génération ancrée](#9-construction-du-contexte-et-generation-ancree)
10. [Évaluation : métriques de recherche et de génération](#10-evaluation-metriques-de-recherche-et-de-generation)
11. [Optimisation de la latence et maîtrise des coûts](#11-optimisation-de-la-latence-et-maitrise-des-couts)
12. [Pièges classiques et liste de contrôle de mise en production](#12-pieges-classiques-et-liste-de-controle-de-mise-en-production)

---

## 1. Vue d'ensemble d'une architecture RAG

_La génération augmentée par la recherche répond à un problème précis : faire répondre un modèle de langage à partir de connaissances qu'il n'a pas mémorisées._

Un modèle de langage ne connaît que ce qui figurait dans ses données d'entraînement. Il ignore les documents internes d'une organisation, les données postérieures à sa date de coupure et tout contenu privé. Trois approches permettent de dépasser cette limite. Le réentraînement complet est prohibitif en coût et en délai. L'affinage spécialise le modèle sur un domaine mais convient mal à des connaissances factuelles évolutives, et n'offre aucune traçabilité de la source. La génération augmentée par la recherche, elle, conserve le modèle inchangé et lui fournit au moment de la requête les extraits documentaires pertinents.

Le fonctionnement se décompose en deux temps distincts. Hors ligne, une phase d'indexation transforme le corpus documentaire en une structure interrogeable : les documents sont extraits, découpés en fragments, convertis en vecteurs numériques et stockés dans un index. En ligne, à chaque requête, la question de l'utilisateur est convertie en vecteur par le même modèle, les fragments les plus proches sont récupérés, puis assemblés dans un contexte transmis au modèle de génération avec la consigne de répondre en s'appuyant exclusivement sur ces éléments.

Cette architecture présente quatre avantages déterminants. Les connaissances sont actualisables en réindexant, sans toucher au modèle. Les réponses sont traçables, chaque affirmation pouvant être rattachée à un passage source vérifiable. Le contrôle d'accès est applicable au niveau du document, ce qu'un affinage ne permet pas. Enfin le coût marginal d'ajout de connaissances est celui d'une indexation, sans commune mesure avec un entraînement. Son principal inconvénient est que la qualité de la réponse est bornée par celle de la recherche : un système qui ne retrouve pas le bon passage ne peut produire la bonne réponse, quel que soit le modèle de génération employé.

### Composants d'une chaîne RAG

- **Extraction** : conversion des documents sources en texte structuré exploitable.
- **Découpage** : segmentation en fragments de taille compatible avec le modèle d'embeddings.
- **Vectorisation** : calcul d'une représentation numérique dense de chaque fragment.
- **Indexation** : stockage des vecteurs dans une structure permettant la recherche approchée.
- **Recherche** : récupération des fragments les plus proches du vecteur de la requête.
- **Reclassement** : réordonnancement fin des candidats par un modèle plus précis.
- **Génération** : composition d'une réponse ancrée sur les fragments retenus.
- **Évaluation** : mesure continue de la qualité de chaque étape de la chaîne.

> **Point d'attention.** La qualité finale d'un système RAG est plafonnée par celle de son étage de recherche. Optimiser le modèle de génération avant d'avoir mesuré la recherche est une erreur de séquence classique.

---

## 2. Embeddings : représenter le sens par un vecteur de dimension 384

_Un embedding projette un texte dans un espace vectoriel où la proximité géométrique traduit la proximité de sens._

Un modèle d'embeddings transforme une séquence de texte en un vecteur de nombres réels de dimension fixe. Dans notre architecture de référence, cette dimension est de 384, ce qui signifie que chaque fragment de texte, qu'il fasse dix mots ou cinq cents, est représenté par exactement 384 nombres. La propriété fondamentale, obtenue par l'entraînement du modèle, est que deux textes de sens proche produisent des vecteurs proches dans cet espace, indépendamment des mots employés. « Comment réinitialiser mon mot de passe » et « procédure de récupération d'identifiants oubliés » n'ont presque aucun mot en commun mais produisent des vecteurs très proches.

Le choix de la dimension résulte d'un arbitrage explicite. Une dimension plus élevée offre une capacité de représentation supérieure et capture des nuances plus fines, mais elle augmente proportionnellement l'empreinte mémoire de l'index, le temps de calcul de chaque comparaison et le coût de stockage. Sur des corpus de quelques centaines de milliers à quelques millions de fragments, l'écart de qualité mesuré entre 384 et 768 dimensions se situe généralement entre deux et quatre points de Precision@5, alors que l'empreinte double. La dimension 384 constitue le point d'équilibre retenu pour la majorité des usages internes.

Le calcul de l'empreinte mémoire est direct et mérite d'être maîtrisé pour dimensionner une plateforme. Un vecteur de 384 dimensions en virgule flottante simple précision occupe 384 multiplié par 4 octets, soit 1 536 octets, environ 1,5 kilooctet. Un million de fragments représente donc 1,5 gigaoctet de vecteurs bruts, auxquels s'ajoute la surcharge de la structure d'index, typiquement de 50 à 80 pour cent pour un graphe navigable hiérarchique. La quantification scalaire sur un octet par dimension divise cette empreinte par quatre au prix d'une perte de rappel généralement inférieure à un point.

Trois précautions doivent être respectées. Le même modèle doit impérativement être utilisé pour l'indexation et pour la requête : deux modèles différents produisent des espaces vectoriels sans relation, et la recherche retourne du bruit. Le changement de modèle impose une réindexation complète du corpus, opération dont la durée doit être anticipée. Enfin, certains modèles requièrent un préfixe distinct pour les requêtes et pour les documents ; l'omission de ce préfixe dégrade silencieusement la qualité sans produire d'erreur visible.

### Empreinte mémoire selon la dimension et la quantification

| Dimension | Précision | Octets par vecteur | 1 M de fragments | Avec index HNSW |
|---|---|---|---|---|
| 384 | float32 | 1 536 | 1,54 Go | ≈ 2,6 Go |
| 384 | int8 quantifié | 384 | 0,38 Go | ≈ 0,7 Go |
| 768 | float32 | 3 072 | 3,07 Go | ≈ 5,2 Go |
| 768 | int8 quantifié | 768 | 0,77 Go | ≈ 1,3 Go |
| 1 024 | float32 | 4 096 | 4,10 Go | ≈ 6,9 Go |
| 1 536 | float32 | 6 144 | 6,14 Go | ≈ 10,4 Go |

> **Point d'attention.** Changer de modèle d'embeddings impose une réindexation complète. Ce coût doit être intégré dès la conception : prévoyez une procédure de réindexation sans interruption de service.

---

## 3. Similarité cosinus et géométrie de l'espace vectoriel

_Le cosinus mesure l'angle entre deux vecteurs, indépendamment de leur longueur : c'est exactement la propriété recherchée._

La similarité cosinus entre deux vecteurs se définit comme le produit scalaire des deux vecteurs divisé par le produit de leurs normes euclidiennes. Elle s'échelonne de moins un à un : une valeur de un signifie que les vecteurs pointent exactement dans la même direction, zéro qu'ils sont orthogonaux et donc sans relation, et moins un qu'ils sont opposés. En pratique, sur des embeddings de texte, les valeurs observées se concentrent entre zéro et un, les valeurs négatives étant rares.

L'intérêt du cosinus par rapport à la distance euclidienne tient à son insensibilité à la norme des vecteurs. Or la norme d'un embedding est partiellement corrélée à la longueur du texte représenté et à sa densité informationnelle, grandeurs qui ne doivent pas influencer la mesure de pertinence : un fragment court et un fragment long traitant du même sujet doivent être considérés comme également pertinents. En ne conservant que la direction, le cosinus isole l'information sémantique de ces facteurs parasites.

Une optimisation importante découle de cette définition. Si tous les vecteurs sont normalisés à la norme unitaire au moment de l'indexation, le dénominateur de la formule vaut un et la similarité cosinus se réduit au simple produit scalaire. L'économie porte sur deux calculs de norme et une division par comparaison, ce qui devient significatif lorsque des millions de comparaisons sont effectuées. La normalisation systématique à l'indexation est donc la pratique de référence, et la plupart des bases vectorielles l'appliquent automatiquement lorsque la mesure cosinus est déclarée.

L'interprétation des scores appelle une mise en garde. Les valeurs absolues de similarité ne sont pas comparables entre modèles : un score de 0,72 peut correspondre à une correspondance excellente avec un modèle et médiocre avec un autre, selon la manière dont l'espace a été entraîné. Certains modèles produisent des distributions très resserrées où tous les scores se situent entre 0,60 et 0,90. Un seuil de pertinence doit donc toujours être calibré empiriquement sur le modèle effectivement utilisé et sur un corpus représentatif, jamais repris d'une documentation externe.

### Ordres de grandeur observés (indicatifs, à recalibrer par modèle)

- Score supérieur à 0,85 : quasi-paraphrase ou reprise littérale.
- Score entre 0,70 et 0,85 : forte pertinence sémantique, même sujet traité.
- Score entre 0,50 et 0,70 : pertinence partielle, sujet connexe.
- Score entre 0,35 et 0,50 : lien thématique faible, souvent du bruit.
- Score inférieur à 0,35 : sans rapport, à filtrer.

### Mesures de similarité et cas d'usage

| Mesure | Formule abrégée | Plage | Sensible à la norme | Usage |
|---|---|---|---|---|
| Cosinus | produit scalaire / (norme × norme) | [-1, 1] | Non | Référence pour le texte |
| Produit scalaire | somme des produits terme à terme | Non bornée | Oui | Équivalent au cosinus si normalisé |
| Distance euclidienne | racine de la somme des carrés des écarts | [0, ∞[ | Oui | Espaces normalisés uniquement |
| Distance de Manhattan | somme des valeurs absolues des écarts | [0, ∞[ | Oui | Rare sur embeddings denses |

> **Point d'attention.** Ne transposez jamais un seuil de similarité d'un modèle à un autre. Calibrez-le sur votre corpus, en observant la distribution des scores des paires annotées pertinentes et non pertinentes.

---

## 4. Ingestion documentaire : extraction et normalisation

_La qualité de l'extraction conditionne tout ce qui suit : un texte mal extrait produit des fragments incohérents et des vecteurs bruités._

L'extraction diffère fortement selon le format source. Un fichier Markdown ou HTML porte une structure explicite qu'il suffit de préserver. Un document bureautique expose sa structure logique via ses styles de titre, information précieuse à condition que l'auteur les ait employés plutôt que d'avoir mis en forme manuellement du texte en gras. Un PDF constitue le cas le plus difficile : il décrit une mise en page et non une structure logique, et l'ordre de lecture doit être reconstitué à partir des positions des blocs de texte, exercice qui échoue régulièrement sur les mises en page multicolonnes.

Quatre catégories d'artefacts doivent être traitées lors de la normalisation. Les en-têtes et pieds de page se répètent à chaque page et polluent les fragments sans apporter d'information ; ils se détectent par leur récurrence à position constante. Les numéros de page isolés produisent des fragments dégénérés. Les césures en fin de ligne coupent les mots et doivent être recollées. Les tableaux, enfin, perdent leur structure à l'extraction linéaire et méritent un traitement dédié, par exemple une conversion en Markdown préservant les relations ligne-colonne.

La préservation de la hiérarchie des titres constitue l'élément le plus précieux de l'extraction, souvent négligé. Connaître qu'un fragment appartient à la section « Sécurité » puis à la sous-section « Authentification multifacteur » permet deux usages majeurs : découper aux frontières naturelles de sections plutôt qu'arbitrairement, et préfixer le texte du fragment par son chemin hiérarchique avant vectorisation, ce qui désambiguïse considérablement les fragments courts ou anaphoriques.

Un contrôle qualité systématique doit être appliqué après extraction. Trois indicateurs simples détectent la majorité des problèmes : le ratio de caractères alphabétiques sur le total, qui chute en cas d'extraction dégradée ; la longueur moyenne des phrases, anormalement courte lorsque l'ordre de lecture est cassé ; et le taux de mots reconnus par un dictionnaire, révélateur des problèmes d'encodage et de césure. Tout document sous seuil est mis en quarantaine et signalé plutôt qu'indexé silencieusement.

### Difficultés d'extraction par format

| Format | Structure disponible | Difficulté principale | Traitement recommandé |
|---|---|---|---|
| Markdown | Explicite et fiable | Aucune | Analyse syntaxique directe |
| HTML | Explicite | Bruit de navigation et publicité | Extraction du contenu principal |
| DOCX | Styles de titre | Styles non utilisés par l'auteur | Repli sur heuristique de mise en forme |
| PDF natif | Positions absolues | Ordre de lecture, multicolonne | Analyse de mise en page |
| PDF scanné | Aucune | Nécessite une reconnaissance optique | OCR puis contrôle qualité renforcé |
| Présentations | Par diapositive | Texte fragmenté et elliptique | Fusion des notes et du contenu |
| Tableurs | Cellules | Sémantique portée par la position | Sérialisation ligne par ligne |

> **Point d'attention.** Un document mal extrait pollue durablement l'index et n'est jamais détecté par les métriques globales. Le contrôle qualité à l'ingestion est le meilleur investissement de la chaîne.

---

## 5. Chunking : stratégies de découpage en fragments

_Le découpage est le paramètre le plus déterminant de la qualité d'un système RAG, avant même le choix du modèle d'embeddings._

Le découpage répond à une contrainte technique et à une exigence sémantique. La contrainte technique tient à la fenêtre du modèle d'embeddings, qui accepte un nombre borné de jetons, typiquement 512, et tronque silencieusement au-delà : un document entier soumis tel quel verrait tout son contenu au-delà de la limite purement ignoré. L'exigence sémantique est plus subtile : un vecteur unique représentant un document de quatre-vingts pages moyenne tous ses sujets et ne correspond précisément à aucun, ce qui le rend inutilisable pour retrouver un passage particulier.

Quatre stratégies de découpage sont couramment employées. Le découpage à taille fixe, le plus simple, coupe tous les N caractères ou jetons sans considération pour le contenu ; il est rapide et prévisible mais coupe fréquemment au milieu d'une phrase. Le découpage récursif par séparateurs, recommandé par défaut, tente successivement de couper aux titres, puis aux paragraphes, puis aux phrases, puis aux mots, ce qui respecte les frontières naturelles tant que la taille cible le permet. Le découpage structurel suit la hiérarchie du document et produit des fragments de taille très variable. Le découpage sémantique, enfin, détecte les ruptures thématiques en comparant les embeddings de phrases successives.

Le dimensionnement de la taille cible arbitre entre précision et contexte. Des fragments courts, de 128 à 256 jetons, produisent des vecteurs très spécifiques et une bonne précision, mais risquent de séparer une information de son contexte indispensable : un fragment énonçant « ce seuil ne doit pas être dépassé » sans mentionner de quel seuil il s'agit est inexploitable. Des fragments longs, de 1 024 jetons et plus, préservent le contexte mais diluent le signal : un fragment traitant de cinq sujets produit un vecteur moyen qui ne correspond bien à aucune requête portant sur l'un d'eux.

La valeur de 512 jetons constitue le point de départ recommandé pour de la documentation technique en prose, avec des ajustements selon la nature du corpus. Une base de questions-réponses gagne à des fragments courts, chaque paire constituant une unité autonome. Un corpus juridique ou contractuel, où le sens d'une clause dépend étroitement de son environnement, justifie des fragments plus longs. La bonne pratique est de tester au moins trois tailles sur un jeu d'évaluation représentatif avant de figer le paramètre.

### Recommandations de taille par type de corpus

- Documentation technique en prose : 512 jetons, découpage récursif.
- Base de questions-réponses : une paire par fragment, sans découpage supplémentaire.
- Textes juridiques et contractuels : 768 à 1 024 jetons pour préserver le contexte.
- Comptes rendus et transcriptions : 256 à 384 jetons, découpage par tour de parole.
- Articles de presse et publications : 512 jetons avec préfixe de titre.
- Code source : découpage par fonction ou par classe, jamais par taille fixe.
- Tableaux de données : une ligne par fragment avec les en-têtes répétés.

### Stratégies de découpage comparées

| Stratégie | Principe | Avantage | Inconvénient | Recommandation |
|---|---|---|---|---|
| Taille fixe | Coupe tous les N jetons | Simple, prévisible | Coupe au milieu des phrases | Éviter |
| Récursif par séparateurs | Titres, paragraphes, phrases, mots | Respecte les frontières naturelles | Taille variable | Par défaut |
| Structurel | Suit la hiérarchie du document | Fragments cohérents | Taille très hétérogène | Documents bien structurés |
| Sémantique | Détecte les ruptures thématiques | Cohérence maximale | Coûteux à l'indexation | Corpus à forte valeur |
| Par fenêtre glissante | Fragments largement recouvrants | Rappel élevé | Index volumineux, redondance | Corpus critique et petit |

> **Point d'attention.** Testez systématiquement au moins trois tailles de fragment sur votre jeu d'évaluation. Les gains obtenus par ce simple réglage dépassent fréquemment ceux d'un changement de modèle d'embeddings.

---

## 6. Overlap : rôle et dimensionnement du recouvrement

_Le recouvrement assure qu'aucune information ne soit tronquée à la frontière de deux fragments consécutifs._

Le problème que résout le recouvrement est simple à énoncer. Considérons une information tenant en trois phrases, dont la coupure de découpage tombe après la première. Le premier fragment se termine par une amorce sans conclusion, le second commence par une suite sans prémisse. Aucun des deux ne porte l'information complète, et une requête portant sur cette information ne correspondra bien ni à l'un ni à l'autre. Le recouvrement consiste à faire commencer chaque fragment quelques dizaines de jetons avant la fin du précédent, de sorte qu'une portion de texte de la taille du recouvrement apparaisse intégralement dans au moins un fragment.

Le dimensionnement s'exprime généralement en pourcentage de la taille du fragment. Une valeur de 10 à 20 pour cent constitue la plage recommandée : 64 jetons de recouvrement pour des fragments de 512 jetons correspond à 12,5 pour cent. En deçà de 10 pour cent, le recouvrement ne couvre plus une unité sémantique complète et perd son utilité. Au-delà de 25 pour cent, le coût devient significatif sans gain mesurable : le volume de l'index croît approximativement selon l'inverse de un moins le taux de recouvrement.

Le coût du recouvrement se manifeste sous trois formes. Le volume indexé augmente mécaniquement, avec les conséquences correspondantes sur la mémoire, le temps d'indexation et le coût de calcul des vecteurs. La redondance dans les résultats s'accroît : deux fragments recouvrants portant tous deux l'information recherchée occupent deux places dans les K premiers résultats, réduisant la diversité de ce qui est présenté. Cette seconde conséquence est la plus gênante en pratique et impose une déduplication au moment de la restitution.

La déduplication peut s'opérer selon plusieurs critères. Le plus simple consiste à ne retenir qu'un fragment par document parmi les K premiers, stratégie efficace pour la diversité mais qui pénalise les documents traitant longuement du sujet. Une approche plus fine fusionne les fragments adjacents du même document en un passage étendu, ce qui restitue à l'utilisateur un contexte plus lisible tout en occupant une seule place dans la liste. C'est l'approche recommandée, dite fusion de fragments contigus.

### Effet du taux de recouvrement sur le volume et la qualité

| Recouvrement | En jetons (base 512) | Facteur de volume | Rappel aux frontières | Recommandation |
|---|---|---|---|---|
| 0 % | 0 | 1,00 | Faible | Déconseillé |
| 10 % | 51 | 1,11 | Correct | Minimum utile |
| 12,5 % | 64 | 1,14 | Bon | Valeur de référence |
| 20 % | 102 | 1,25 | Très bon | Corpus critique |
| 30 % | 154 | 1,43 | Très bon | Rendement décroissant |
| 50 % | 256 | 2,00 | Excellent | Coût disproportionné |

> **Point d'attention.** Le recouvrement augmente la redondance des résultats. Prévoyez systématiquement une fusion des fragments contigus d'un même document au moment de la restitution.

---

## 7. Indexation vectorielle : structures et paramètres

_Comparer une requête à tous les vecteurs est exact mais impraticable au-delà de quelques dizaines de milliers de fragments._

La recherche exhaustive, dite force brute, calcule la similarité entre le vecteur de requête et chacun des vecteurs indexés. Elle garantit l'exactitude du résultat mais son coût croît linéairement avec la taille du corpus. À un million de fragments en dimension 384, une recherche exhaustive représente environ 384 millions de multiplications par requête, soit plusieurs centaines de millisecondes sur un processeur généraliste, ce qui interdit tout usage interactif à volume réel. Les index approchés renoncent à la garantie d'exactitude pour gagner plusieurs ordres de grandeur.

Le graphe navigable hiérarchique, désigné par l'acronyme HNSW, constitue la structure la plus employée. Il organise les vecteurs en plusieurs couches de graphes de voisinage, la couche supérieure étant très clairsemée et les couches inférieures de plus en plus denses. La recherche commence au sommet, progresse de proche en proche vers le voisin le plus proche de la requête, puis descend d'une couche et recommence, affinant progressivement. Cette navigation atteint une complexité logarithmique en pratique, avec un rappel typiquement supérieur à 0,98 pour des paramètres correctement réglés.

Trois paramètres gouvernent le compromis entre rappel, latence et mémoire. Le nombre de connexions par nœud détermine la densité du graphe : une valeur élevée améliore le rappel et augmente la mémoire, avec des valeurs usuelles entre 16 et 64. La taille de la liste dynamique à la construction contrôle la qualité du graphe bâti, avec des valeurs de 100 à 500 ; elle n'affecte que le temps d'indexation. La taille de la liste dynamique à la recherche arbitre dynamiquement entre rappel et latence, et constitue le seul paramètre ajustable sans reconstruire l'index, ce qui en fait le levier opérationnel privilégié.

Deux familles alternatives méritent d'être connues. Les index à quantification réduisent la précision des vecteurs, par quantification scalaire sur un octet par dimension ou par quantification par produit, divisant l'empreinte mémoire par quatre à trente-deux. Les index à partitionnement, de type IVF, regroupent les vecteurs en cellules autour de centroïdes et ne parcourent que les cellules les plus proches de la requête ; ils consomment moins de mémoire que HNSW mais présentent un rappel plus sensible au réglage et supportent mal les insertions incrémentales.

### Réglages de départ recommandés pour HNSW

- Connexions par nœud : 32 pour un corpus général, 16 si la mémoire est contrainte.
- Liste dynamique à la construction : 200, à augmenter si le rappel mesuré est insuffisant.
- Liste dynamique à la recherche : 100 en production, ajustable selon la latence observée.
- Mesurer le rappel réel contre une recherche exhaustive sur un échantillon de 1 000 requêtes.
- Récupérer trois à quatre fois plus de candidats que nécessaire si un reclassement suit.
- Surveiller la dégradation du graphe après un fort volume de suppressions.

### Familles d'index vectoriels

| Type | Rappel | Latence | Mémoire | Insertion incrémentale | Usage |
|---|---|---|---|---|---|
| Force brute | 1,00 | Élevée | Minimale | Immédiate | Moins de 50 000 vecteurs |
| HNSW | 0,95 à 0,99 | Très faible | Élevée | Bonne | Référence générale |
| HNSW quantifié int8 | 0,94 à 0,98 | Très faible | Modérée | Bonne | Grands corpus |
| IVF-Flat | 0,90 à 0,98 | Faible | Modérée | Reconstruction périodique | Corpus statiques |
| IVF-PQ | 0,80 à 0,92 | Très faible | Très faible | Reconstruction périodique | Très grands corpus |
| Index inversé lexical | Sans objet | Très faible | Faible | Immédiate | Complément hybride |

---

## 8. Recherche hybride et réordonnancement

_La recherche vectorielle échoue précisément là où la recherche lexicale excelle, et réciproquement : les combiner est presque toujours gagnant._

La recherche vectorielle présente une faiblesse structurelle sur les termes rares et exacts : références de produit, codes d'erreur, noms propres, numéros de version, acronymes internes. Ces chaînes portent peu de sens distribué et leur représentation vectorielle est mal apprise, si bien qu'une requête portant sur « l'erreur E-4417 » retourne des fragments traitant génériquement d'erreurs. La recherche lexicale, fondée sur la correspondance de termes pondérée par leur rareté, excelle au contraire sur ces cas mais échoue dès que le vocabulaire diverge.

La recherche hybride exécute les deux recherches en parallèle et fusionne leurs résultats. La méthode de fusion la plus robuste est la fusion par rang réciproque, qui attribue à chaque document un score fondé sur son rang dans chaque liste plutôt que sur son score brut. Cette approche évite d'avoir à normaliser des scores d'échelles incomparables, problème notoirement délicat, et s'avère remarquablement stable en pratique. Une constante de lissage, typiquement fixée à 60, atténue l'influence des premiers rangs.

Le réordonnancement constitue la seconde amélioration majeure. Il repose sur un modèle d'encodage croisé qui, contrairement au modèle d'embeddings, traite simultanément la requête et le document candidat et produit un score de pertinence directement. Cette architecture est bien plus précise car elle autorise une interaction complète entre les deux textes, mais elle est aussi bien plus coûteuse : elle impose un calcul par paire, ce qui interdit son usage sur l'ensemble du corpus. Elle s'applique donc en second étage, sur les vingt à cinquante candidats retournés par la recherche vectorielle.

L'architecture en deux étages, dite récupération puis reclassement, combine les avantages des deux approches : le premier étage est rapide et assure le rappel, le second est précis et assure la qualité de l'ordonnancement final. Le gain mesuré est typiquement de cinq à quinze points de NDCG selon les corpus, pour un coût de latence de cinquante à deux cents millisecondes. Il s'agit du meilleur rapport gain sur effort disponible après l'optimisation du découpage.

### Mise en œuvre d'une recherche en deux étages

- Étage 1 : récupérer 30 à 50 candidats par recherche hybride pour assurer le rappel.
- Étage 2 : reclasser ces candidats par encodage croisé et ne conserver que les 5 meilleurs.
- Fusionner les fragments contigus d'un même document avant reclassement.
- Appliquer les filtres de métadonnées et d'habilitation à l'étage 1, jamais après.
- Mesurer séparément le rappel de l'étage 1 et la précision de l'étage 2.
- Prévoir un mode dégradé sans reclassement en cas d'indisponibilité du modèle.

### Comparaison des approches de recherche

| Approche | Termes rares | Reformulation | Latence | Gain NDCG typique |
|---|---|---|---|---|
| Lexicale seule | Excellente | Nulle | 10 ms | Référence |
| Vectorielle seule | Faible | Excellente | 30 ms | +8 à +20 points |
| Hybride par fusion de rangs | Excellente | Excellente | 45 ms | +12 à +28 points |
| Hybride avec reclassement | Excellente | Excellente | 150 ms | +20 à +40 points |

> **Point d'attention.** Si le bon passage n'est pas dans les candidats de l'étage 1, aucun reclassement ne le fera apparaître. Optimisez d'abord le rappel du premier étage, la précision du second ensuite.

---

## 9. Construction du contexte et génération ancrée

_L'ancrage est la propriété qui distingue une réponse fiable d'une réponse plausible._

Le contexte transmis au modèle de génération assemble les fragments retenus, précédés d'instructions et suivis de la question. Trois principes gouvernent sa construction. Chaque fragment doit porter un identifiant de source explicite, afin que le modèle puisse citer précisément et que les citations soient vérifiables automatiquement. L'ordre des fragments importe : les modèles accordent une attention supérieure au début et à la fin du contexte, phénomène documenté sous le nom de perte au milieu, ce qui conduit à placer les fragments les plus pertinents aux extrémités. Enfin le nombre de fragments doit rester mesuré : au-delà de huit à dix, la dilution dégrade la qualité davantage que l'apport d'information ne la améliore.

L'instruction système doit être explicite sur trois points. Elle impose de répondre exclusivement à partir des passages fournis, sans mobiliser de connaissance externe. Elle exige la citation de la source de chaque affirmation. Elle prescrit enfin le comportement en l'absence d'élément suffisant : indiquer explicitement que l'information n'est pas disponible dans les documents consultés. Cette troisième instruction est la plus importante et la plus fréquemment omise ; sans elle, le modèle comblera les lacunes par ses connaissances générales, produisant des réponses plausibles mais non fondées.

La vérification de l'ancrage doit être automatisée et non laissée à la seule instruction. Deux mécanismes complémentaires sont recommandés. Le premier consiste à vérifier syntaxiquement que chaque référence citée correspond bien à un fragment effectivement présent dans le contexte, contrôle trivial qui détecte les citations inventées. Le second, plus élaboré, soumet chaque affirmation de la réponse à un contrôle d'implication par rapport aux fragments cités, à l'aide d'un second appel au modèle ou d'un modèle spécialisé.

Le comportement de refus mérite une attention particulière lors de l'évaluation. Un système qui ne refuse jamais est un système qui invente ; un système qui refuse trop est inutilisable. Le jeu d'évaluation doit donc comporter une part significative de questions dont la réponse ne figure pas dans le corpus, typiquement quinze à vingt pour cent, et le taux de refus correct sur ces questions constitue une métrique de premier plan, au même titre que la précision de la recherche.

### Éléments d'une instruction système efficace

- Répondre exclusivement à partir des passages fournis, sans connaissance externe.
- Citer la référence de chaque affirmation par son identifiant de fragment.
- Indiquer explicitement l'absence d'information plutôt que de combler la lacune.
- Ne pas reformuler une incertitude du texte source en affirmation catégorique.
- Signaler les contradictions entre passages plutôt que d'en choisir un arbitrairement.
- Conserver les unités, ordres de grandeur et conditions énoncés dans les sources.
- Répondre dans la langue de la question, quelle que soit celle des sources.

### Paramètres de construction du contexte

| Paramètre | Valeur recommandée | Effet d'une valeur trop élevée | Effet d'une valeur trop faible |
|---|---|---|---|
| Nombre de fragments | 5 à 8 | Dilution, perte au milieu, coût | Information manquante |
| Budget de jetons du contexte | 3 000 à 6 000 | Coût et latence | Troncature des passages |
| Seuil de score minimal | Calibré empiriquement | Refus excessifs | Réponses non fondées |
| Ordre des fragments | Meilleurs aux extrémités | — | Perte au milieu |
| Fusion des fragments contigus | Activée | — | Redondance, contexte morcelé |

> **Point d'attention.** Un système RAG qui ne refuse jamais de répondre n'est pas un bon système : c'est un système qui invente. Mesurez le taux de refus correct comme une métrique de premier plan.

---

## 10. Évaluation : métriques de recherche et de génération

_Sans évaluation quantitative, toute optimisation relève de l'intuition et toute régression passe inaperçue._

L'évaluation d'un système RAG se conduit à deux niveaux distincts qu'il ne faut jamais confondre. L'évaluation de la recherche mesure la capacité à retrouver les bons passages, indépendamment de toute génération. L'évaluation de la génération mesure la qualité de la réponse produite à partir de passages donnés. Séparer les deux est indispensable au diagnostic : une réponse insatisfaisante peut provenir d'une recherche défaillante ou d'une génération défaillante, et la correction diffère radicalement selon le cas.

Les métriques de recherche s'appuient sur un jeu de requêtes annotées. La Precision@K mesure la proportion de résultats pertinents parmi les K premiers retournés : elle répond à la question de l'utilité de ce qui est montré. La Recall@K mesure la proportion des passages pertinents existants qui apparaissent dans les K premiers : elle répond à la question de l'exhaustivité. Ces deux métriques varient en sens inverse avec K et ne doivent jamais être optimisées isolément.

Le rang réciproque moyen, ou MRR, se calcule comme la moyenne sur toutes les requêtes de l'inverse du rang du premier résultat pertinent. Une bonne réponse en première position contribue pour 1, en deuxième position pour 0,5, en cinquième position pour 0,2. Cette métrique reflète bien l'expérience réelle d'un utilisateur qui consulte les résultats dans l'ordre et s'arrête au premier satisfaisant. Le gain cumulé actualisé normalisé, ou NDCG, va plus loin en intégrant des niveaux de pertinence gradués et une décote logarithmique du rang ; c'est la métrique la plus complète et celle à privilégier comme indicateur unique de synthèse.

Les métriques de génération portent sur trois dimensions. La fidélité mesure la proportion d'affirmations de la réponse effectivement soutenues par les passages fournis, et constitue la métrique anti-hallucination. La pertinence de la réponse mesure l'adéquation à la question posée, indépendamment de son exactitude. La complétude mesure la couverture des éléments attendus. Ces métriques s'évaluent soit par annotation humaine sur un échantillon, soit par un modèle juge, dont les jugements doivent eux-mêmes être calibrés contre l'annotation humaine avant d'être considérés comme fiables.

### Constitution d'un jeu d'évaluation exploitable

- Au minimum 200 requêtes, idéalement 300 à 500, issues d'usages réels.
- Quatre familles : factuelles, procédurales, exploratoires et hors corpus.
- Quinze à vingt pour cent de questions dont la réponse ne figure pas dans le corpus.
- Annotation en double aveugle avec résolution des désaccords par un tiers.
- Accord inter-annotateurs supérieur à 0,70, sinon clarifier la définition de la pertinence.
- Pertinence graduée sur trois niveaux pour permettre le calcul du NDCG.
- Scission en deux tiers de développement et un tiers réservé à la validation finale.
- Réévaluation automatique à chaque modification de la chaîne, avec seuil de régression.

### Métriques d'évaluation d'un système RAG

| Métrique | Niveau | Ce qu'elle mesure | Plage | Cible usuelle |
|---|---|---|---|---|
| Precision@1 | Recherche | Le premier résultat est-il pertinent | [0, 1] | > 0,75 |
| Precision@5 | Recherche | Part de pertinents dans les 5 premiers | [0, 1] | > 0,80 |
| Recall@10 | Recherche | Part des pertinents trouvés dans les 10 premiers | [0, 1] | > 0,85 |
| MRR | Recherche | Inverse moyen du rang du premier pertinent | [0, 1] | > 0,80 |
| NDCG@10 | Recherche | Gain cumulé actualisé, pertinence graduée | [0, 1] | > 0,80 |
| Rappel du premier étage | Recherche | Part des pertinents dans les candidats | [0, 1] | > 0,95 |
| Fidélité | Génération | Affirmations soutenues par les sources | [0, 1] | > 0,95 |
| Pertinence de la réponse | Génération | Adéquation à la question posée | [0, 1] | > 0,90 |
| Taux de refus correct | Génération | Refus sur questions hors corpus | [0, 1] | > 0,90 |
| Latence p95 | Système | 95e centile du temps de réponse | ms | < 1 000 ms |

> **Point d'attention.** N'optimisez jamais les paramètres sur le jeu qui servira à la validation finale. Le sur-ajustement produit des métriques flatteuses et une performance réelle décevante.

---

## 11. Optimisation de la latence et maîtrise des coûts

_Un système précis mais lent n'est pas utilisé : la latence perçue est une exigence fonctionnelle._

La latence d'une requête RAG se décompose en quatre postes mesurables séparément. Le calcul du vecteur de la requête, de l'ordre de dix à trente millisecondes sur un modèle de dimension 384. La recherche vectorielle, de dix à cinquante millisecondes selon la taille de l'index et le réglage. Le reclassement, de cinquante à deux cents millisecondes selon le nombre de candidats. La génération enfin, de un à cinq secondes, largement dominante. Cette décomposition doit être instrumentée par une trace distribuée : optimiser sans mesurer la répartition conduit systématiquement à optimiser le mauvais poste.

Plusieurs leviers réduisent la latence perçue sans dégrader la qualité. La diffusion progressive de la réponse générée, jeton par jeton, ramène le délai avant premier affichage à quelques centaines de millisecondes, ce qui transforme l'expérience même si la durée totale est inchangée. La mise en cache des vecteurs de requêtes fréquentes élimine un poste entier. L'affichage des passages sources avant la génération occupe l'utilisateur pendant l'attente et lui permet souvent de trouver sa réponse sans attendre la synthèse.

La maîtrise des coûts suit une logique parallèle. Le coût d'indexation est ponctuel mais peut être significatif sur de gros corpus ; il se maîtrise par le traitement par lots, la détection des documents inchangés par empreinte, et la réindexation incrémentale plutôt que complète. Le coût de recherche est essentiellement de l'infrastructure et suit le dimensionnement mémoire de l'index, d'où l'intérêt de la quantification. Le coût de génération, dominant en exploitation, est proportionnel au nombre de jetons traités : réduire le nombre de fragments du contexte de huit à cinq réduit le coût de près de quarante pour cent, souvent sans perte mesurable de qualité.

Une architecture de repli doit être prévue pour chaque composant. En cas d'indisponibilité du modèle de reclassement, la chaîne doit continuer avec le seul ordonnancement vectoriel. En cas d'indisponibilité du modèle de génération, elle doit basculer en mode recherche de passages, qui reste utile. En cas de dépassement du budget de latence, elle doit retourner les résultats disponibles plutôt qu'une erreur. Ces modes dégradés doivent être testés régulièrement et non seulement implémentés.

### Leviers de réduction des coûts

- Détecter les documents inchangés par empreinte et éviter leur retraitement.
- Vectoriser par lots plutôt que fragment par fragment.
- Quantifier l'index en entier sur un octet pour diviser la mémoire par quatre.
- Réduire le nombre de fragments du contexte de génération après mesure de l'impact.
- Mettre en cache les réponses aux questions fréquentes avec invalidation à la réindexation.
- Router les questions simples vers un modèle plus léger sur critère de complexité.
- Purger les fragments des documents supprimés plutôt que de les filtrer à la lecture.

### Budget de latence et leviers d'optimisation

| Étape | Latence typique | Part | Levier principal |
|---|---|---|---|
| Vectorisation de la requête | 10 à 30 ms | 1 % | Cache des requêtes fréquentes |
| Recherche vectorielle | 10 à 50 ms | 2 % | Réglage de la liste dynamique |
| Filtrage par métadonnées | 5 à 20 ms | 1 % | Index sur les champs filtrés |
| Reclassement | 50 à 200 ms | 6 % | Réduction du nombre de candidats |
| Construction du contexte | 5 à 15 ms | 1 % | Négligeable |
| Génération | 1 000 à 5 000 ms | 89 % | Diffusion progressive, contexte réduit |

---

## 12. Pièges classiques et liste de contrôle de mise en production

_La plupart des échecs de projets RAG proviennent d'un petit nombre de causes récurrentes et évitables._

Le piège le plus fréquent est l'absence de jeu d'évaluation. Sans mesure, l'équipe optimise sur des impressions, ne détecte pas les régressions et ne peut arbitrer entre deux configurations. La constitution du jeu d'évaluation doit précéder l'optimisation et non la suivre : elle est le premier livrable technique du projet, avant même que le moteur ne fonctionne correctement.

Le deuxième piège est la sous-estimation de la qualité d'extraction. Une chaîne parfaitement conçue sur des documents mal extraits produit des résultats médiocres sans que la cause en soit apparente : les métriques globales se dégradent sans désigner de coupable. L'inspection manuelle d'un échantillon de fragments extraits, opération d'une heure, révèle immédiatement les problèmes d'ordre de lecture, de césure et de pollution par les en-têtes.

Le troisième piège est l'absence de gestion des accès. Un système qui retourne un extrait d'un document confidentiel à un utilisateur non habilité constitue une fuite de données, y compris lorsque le document lui-même n'est pas accessible. Le filtrage doit s'appliquer avant la recherche vectorielle et non après, sous peine de laisser fuiter l'information de l'existence même du document par le biais du nombre de résultats.

Le quatrième piège est le sur-ajustement aux quelques requêtes de démonstration. Une équipe qui affine ses paramètres jusqu'à ce que les cinq questions présentées en réunion fonctionnent parfaitement produit un système qui échoue sur toutes les autres. La discipline du jeu d'évaluation réservé, utilisé une seule fois, est la seule protection contre ce biais qui se manifeste avec une régularité remarquable.

### Liste de contrôle avant mise en production

- Un jeu d'évaluation d'au moins 200 requêtes annotées existe et est versionné.
- Les métriques de recherche et de génération sont mesurées séparément.
- Le rappel du premier étage est mesuré et dépasse 0,95.
- Un échantillon de fragments extraits a été inspecté manuellement.
- Les documents en échec d'extraction sont détectés, mis en quarantaine et signalés.
- Le filtrage par habilitation s'applique avant la recherche vectorielle.
- Un test d'habilitation dédié vérifie l'absence de fuite inter-utilisateurs.
- Le comportement de refus est mesuré sur des questions hors corpus.
- Chaque réponse générée cite des sources vérifiables et cliquables.
- La vérification syntaxique des citations est automatisée.
- Une trace distribuée décompose la latence par étape.
- Des modes dégradés sont implémentés et testés pour chaque composant externe.
- La procédure de réindexation complète est documentée et chronométrée.
- Les retours utilisateurs de pertinence sont collectés et exploités.
- Un seuil de régression bloque le déploiement en cas de dégradation des métriques.

### Symptômes, causes probables et actions correctives

| Symptôme | Cause probable | Vérification | Action |
|---|---|---|---|
| Bonnes réponses absentes des résultats | Rappel du premier étage insuffisant | Mesurer Recall@50 | Augmenter les candidats, activer l'hybride |
| Résultats pertinents mais mal classés | Absence de reclassement | Comparer Precision@1 et Recall@10 | Ajouter un encodage croisé |
| Échec sur les codes et références exactes | Recherche purement vectorielle | Tester des requêtes à termes rares | Activer la recherche hybride |
| Fragments incohérents ou tronqués | Découpage ou extraction défaillante | Inspecter un échantillon de fragments | Revoir le découpage, contrôler l'extraction |
| Réponses plausibles mais fausses | Ancrage insuffisant | Mesurer la fidélité | Renforcer l'instruction, vérifier les citations |
| Refus trop fréquents | Seuil de score trop élevé | Distribution des scores des pertinents | Recalibrer le seuil empiriquement |
| Résultats redondants | Recouvrement sans déduplication | Inspecter les 10 premiers résultats | Fusionner les fragments contigus |
| Latence excessive | Reclassement sur trop de candidats | Trace distribuée par étape | Réduire les candidats, diffuser progressivement |

> **Point d'attention.** Si vous ne deviez retenir qu'une règle : construisez le jeu d'évaluation avant le moteur. Tout le reste en découle.

---
