# TCC
TCC que estou começando a fazer junto a meu professor Ernande Melo. É sobre reconhecimento de imagens, por enquanto estou desenvolvendo o algoritmo para reconhecer pessoas, e a database é compostas de 40 pessoas com 10 fotos cada. A missão é exatamente fazer com que o algoritmo a reconheça corretamente.

## Abordagem
Primeiro, organizar o dataset foi fundamental. No início, as 5 primeiras imagens foram para treino e as outras 5 para teste. Mas quando mandei as pares para treino e as ímpares para teste, a acurácia subiu muito (93,5% para 98%).

Utilizei 3 modelos misturados: SVC, KNN e NC. Basicamente, utilizo o SVC, e os que deram errado mando pro KNN, e os que erraram vão pro NC.
