# TCC
TCC que estou começando a fazer junto a meu professor Ernande Melo. É sobre reconhecimento de imagens, por enquanto estou desenvolvendo o algoritmo para reconhecer pessoas, e a database é compostas de 40 pessoas com 10 fotos cada. A missão é exatamente fazer com que o algoritmo a reconheça corretamente.

## Faces

Foi o primeiro a ser completado. Primeiro, organizar o dataset foi fundamental. Dividi uma metade para treino e outra para teste.

### Problemas particulares enfrentados

Uma das faces era difícil para o modelo reconhecer: a pessoa tinha 8 fotos sem óculos e 2 usando óculos. O que acontecia: As 2 fotos de óculos iam para a base de treino. E como os testes só tinham fotos sem óculos, o modelo não reconhecia a pessoa. Quando manipulei esse caso em especial, colocando 4 s/ óculos e 1 com óculos para treino e teste, a acurácia subiu para 98% (quando os 3 modelos estão juntos).

Somente o SVC fazia quase todo o trabalho, 96,5% de acurácia.

Utilizei 3 modelos misturados: SVC, KNN e NC. Basicamente, utilizo o SVC, e os que deram errado mando pro KNN, e os que erraram vão pro NC.
