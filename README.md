# Investigação de Algoritmos de Representação de Mapas de Características para Classificação de Lesões da Cavidade Oral.

# Image Lime

O algoritmo imageLIME é uma técnica utilizada para explicar as decisões de classificação tomadas por modelos de aprendizagem profunda, especificamente aproximando o comportamento do modelo usando um modelo mais simples e interpretável. Isso é conseguido por meio da estrutura LIME (Locally Interpretable Model-Agnostic Explanations). Aqui está uma explicação passo a passo de como funciona o algoritmo imageLIME, com base nas fontes fornecidas:

**Segmentação**: a imagem de entrada é segmentada em recursos. Isso pode ser feito usando superpixels, uma grade regular ou uma matriz numérica personalizada. A escolha do método de segmentação depende dos requisitos específicos da análise. Para imagens fotográficas, os superpixels geralmente produzem melhores resultados, pois segmentam a imagem em regiões com valores de pixel semelhantes. Contudo, para outros tipos de imagens, como espectrogramas, uma grade mais regular ou um mapa de segmentação personalizado pode ser mais apropriado 1.

**Geração de imagens sintéticas**: imagens sintéticas são geradas incluindo ou excluindo aleatoriamente recursos da imagem original. Cada pixel em um recurso excluído é substituído pelo valor médio de pixel da imagem. Este processo é repetido para um grande número de imagens sintéticas para capturar a variabilidade no comportamento do modelo 15.

**Classificação de Imagens Sintéticas**: As imagens sintéticas são então classificadas usando o modelo de aprendizagem profunda. Esta etapa ajuda a entender como o comportamento do modelo muda com diferentes combinações de recursos 1.
Ajuste do modelo de regressão: Um modelo de regressão é ajustado usando as pontuações de classificação das imagens sintéticas como alvo e a presença ou ausência de cada recurso como preditores binários. Este modelo aproxima a relação entre as características e a pontuação de classificação, permitindo-nos compreender a importância de cada característica 1.
Cálculo da importância do recurso: A importância de cada recurso é calculada usando o modelo de regressão. Isto fornece um mapa de importância da característica, onde as áreas com valores positivos mais elevados correspondem a regiões da imagem de entrada que contribuem positivamente para o resultado de classificação 1.

**Visualização**: o mapa de importância do recurso resultante pode ser visualizado sobre a imagem original para identificar quais partes da imagem são mais influentes na decisão de classificação. Essa visualização pode ajudar a entender quais características o modelo está focando e se está tomando as decisões de classificação esperadas 15.

O algoritmo imageLIME é particularmente útil para interpretar as decisões de modelos de aprendizagem profunda no contexto de classificação de imagens. Ao destacar os recursos mais importantes de uma imagem, permite uma compreensão mais transparente de como o modelo está tomando suas decisões, o que pode ser crucial para depurar, melhorar modelos e garantir que eles estejam focando nos aspectos corretos dos dados de entrada.

# Support Vector Machine

O algoritmo Support Vector Machine (SVM) é um poderoso método de aprendizado de máquina usado para tarefas de classificação e regressão. Ele é excelente no tratamento de dados de alta dimensão e pode lidar com relacionamentos não lineares por meio do uso de funções de kernel. Aqui está uma explicação simplificada de como funciona o algoritmo SVM:

**Representação de dados**: no SVM, os dados são representados em um espaço de recursos de alta dimensão. Para dados linearmente separáveis, este espaço pode ser bidimensional, mas para dados não lineares, a dimensionalidade é aumentada usando uma função kernel para transformar os dados em um espaço de dimensão superior onde se tornam linearmente separáveis.

**Maximizando Margem**: O algoritmo SVM visa encontrar o hiperplano ideal (em duas dimensões, uma linha) que separa os pontos de dados de diferentes classes com a margem máxima. A margem é a distância entre o hiperplano e os pontos de dados mais próximos de cada classe, conhecidos como vetores de suporte. O objetivo é maximizar essa distância, tornando o classificador mais robusto a ruídos e outliers.

**Funções do kernel**: para dados separáveis não linearmente, o SVM usa funções do kernel para mapear os dados em um espaço de dimensão superior onde eles se tornam separáveis linearmente. As funções comuns do kernel incluem o kernel linear, o kernel polinomial, a função de base radial (RBF) e o kernel sigmóide. A escolha da função do kernel depende da natureza dos dados e do problema em questão.

**Margem suave**: na prática, devido a ruídos e valores discrepantes, muitas vezes não é possível encontrar um hiperplano perfeito que separe todos os pontos de dados corretamente. Para acomodar isso, o SVM permite uma “margem suave”, onde alguns pontos de dados podem ser classificados incorretamente ou ficar fora da margem. Isto é controlado por um parâmetro C, que penaliza erros de classificação.

**Treinamento e previsão**: o SVM é treinado resolvendo um problema de otimização para encontrar o melhor hiperplano. Uma vez treinados, os novos pontos de dados são classificados determinando de que lado do hiperplano eles caem. Para SVMs não lineares, a mesma função de kernel é usada para mapear os novos dados no espaço de dimensão superior e depois classificá-los com base no hiperplano.

**Aplicações**: O SVM é amplamente utilizado em vários campos, incluindo classificação de texto, classificação de imagens, detecção de spam, reconhecimento de escrita e análise de expressão genética. Sua capacidade de lidar com dados de alta dimensão e relacionamentos não lineares o torna uma ferramenta versátil para muitas tarefas de aprendizado de máquina.

Em resumo, o algoritmo SVM é um método de aprendizado de máquina poderoso e versátil que se destaca em tarefas de classificação e regressão, especialmente ao lidar com dados de alta dimensão e relacionamentos não lineares. Ao maximizar a margem entre as classes, o SVM cria um classificador robusto que é menos sensível a ruídos e valores discrepantes.

# Minimum Redundancy Maximum Relevance (MRMR)

O algoritmo Mínima Redundância Máxima Relevância (MRMR) é um método de seleção de recursos projetado para identificar os recursos mais relevantes de um conjunto de dados para uso na modelagem, minimizando ao mesmo tempo a redundância entre esses recursos. O objetivo é selecionar um subconjunto de recursos que sejam altamente relevantes para a variável de resposta e que não sejam excessivamente redundantes entre si. Este algoritmo é particularmente útil em problemas de classificação onde o objetivo é encontrar as características mais informativas para prever o resultado. Veja como funciona o algoritmo MRMR:

**Inicialização**: O algoritmo começa com um conjunto vazio de recursos selecionados (S). Também mantém um conjunto complementar (S_c) de recursos ainda não selecionados.

**Seleção de recursos**: o algoritmo seleciona o recurso com a maior relação relevância/redundância (MIQ) do conjunto de recursos ainda não selecionados (S_c). A relevância de uma característica é medida pela sua informação mútua com a variável resposta, e a redundância é medida pela informação mútua entre esta característica e todas as características previamente selecionadas. A relação relevância/redundância (MIQ) é calculada como a relevância dividida pela redundância.

**Adicionando recursos ao S**: O recurso selecionado é adicionado ao conjunto de recursos selecionados (S). Este processo é repetido até que o conjunto de recursos ainda não selecionados (S_c) esteja vazio ou até que nenhum recurso tenha uma relação relevância-redundância diferente de zero.

**Tratamento de recursos de relevância zero**: se houver recursos com relevância zero em S_c, eles serão adicionados a S em ordem aleatória.

**Processo Iterativo**: O algoritmo continua este processo de seleção do recurso mais relevante e adicioná-lo a S até que todos os recursos tenham sido considerados ou até que a redundância seja zero para todos os recursos em S_c.

**Terminação do algoritmo**: O algoritmo termina quando todos os recursos foram adicionados a S ou quando nenhum recurso tem uma relação relevância-redundância diferente de zero em S_c.

O algoritmo MRMR é particularmente eficaz porque equilibra a necessidade de relevância dos recursos com o desejo de evitar redundância entre os recursos. Este equilíbrio é alcançado através da utilização de um quociente de informação mútua (MIQ) que combina informações de relevância e redundância. O algoritmo é eficiente na prática, pois classifica recursos por meio de um esquema de adição direta, exigindo cálculos O(|Ω|·|S|), onde |Ω| é o tamanho de todo o conjunto de recursos e |S| é o tamanho do conjunto de recursos selecionado 1.

O MRMR é aplicável a características categóricas e numéricas, tornando-o versátil em diferentes domínios. No entanto, é importante notar que o MRMR é um método de aprendizagem supervisionado, exigindo um conjunto de dados rotulado para treinamento. É particularmente útil em cenários onde o objetivo é identificar um pequeno conjunto de recursos que sejam relevantes e não redundantes, melhorando assim a eficiência e a precisão das tarefas de modelagem subsequentes. 

# Preparo

### Separar o dataset em Treino, Teste e Validação

### Treinamento no Matlab

Primeiro passo é acessar o deepnetwork designer no menu de ferrametas e selecionar a rede SqueezeNet
<p align="center">
  <img src="images\appDes.png" width="1000" title="hover text">
</p>

Segundo passo é trocar a camada convolucional padrão para a convolucional e densa para o problema de 4 e 6 classes

<p align="center">
  <img src="images\convMatlab.png" width="250" title="hover text">
</p>

As configurações necessárias são: tamanho do filtro de 1,1 e a quantidade de filtros é o numero de classes do problema, as displasias KO e WT são 6 filtros e a displasia são 4 filtros

<p align="center">
  <img src="images\convLayerSet.png" width="250" title="hover text">
</p>

Com as configurações  de design corretas clicamos em "DATA" no menu

<p align="center">
  <img src="images\navBar.png" width="250" title="hover text">
</p>

Nas configurações escolhemos 20% para validação e selecionamos a pasta onde estão as imagens de treino separadas em cada pasta representadas pela sua classe

<p align="center">
  <img src="images\importData.png" width="1000" title="hover text">
</p>

Depois de importado vamos para o menu de treino e colocamos essas configurações
Solver: ADAM
Initial Rate: 0,0001
Max Epochs: 16
Mini batchsize: 32

e assim inicia-se o treinameto

<p align="center">
  <img src="images\trainingSet.png" width="250" title="hover text">
</p>

Finalizado o treinamento, salvar o modelo treinado no matlab para utilização em python, para uso rode o script: saveModel.mat



### Extração pelo ImageLime

Para extração rodamos o script "imageLime.mat" para cada classe do nosso problema, o script salvará as features extraidas pelo imageLime em um arquivo CSV

Importante notar que as features salvas em uma matriz (5,X) onde X é o número de imagens, portanto para ajustar a matrix devemos transpor e obter uma matrix (X, 5).

Para isso usamos o script.

Terminado as operações de transposição para cada classe devemos juntar as features de todas as imagens, para isso concatenamos as features no axis = 1, como mostrado na figura a seguir:

<p align="center">
  <img src="images\navBar.png" width="250" title="hover text">
</p>

Adicionado as features extraidas pelo image lime devemos fazer o mesmo processo pa


# Resultados

Resultados das combinações entre SqueezeNet e os tipos de displasia

### SqueezeNet com Camada Densa - Displasia

<p align="center">
  <img src="images\dispDense.png" width="1000" title="hover text">
</p>

### SqueezeNet com Camada Convolucional - Displasia

<p align="center">
  <img src="images\dispConv.png" width="1000" title="hover text">
</p>

### SqueezeNet com Camada Densa - Displasia KO

<p align="center">
  <img src="images\dispKODense.png" width="1000" title="hover text">
</p>

### SqueezeNet com Camada Convolucional - Displasia KO

<p align="center">
  <img src="images\dispKOConv.png" width="1000" title="hover text">
</p>

### SqueezeNet com Camada Densa - Displasia WT

<p align="center">
  <img src="images\dispDenWT.png" width="1000" title="hover text">
</p>

### SqueezeNet com Camada Convolucional - Displasia WT

<p align="center">
  <img src="images\dispWTConv.png" width="1000" title="hover text">
</p>

| Displasia Camada Densa | número de features | Kernel Type   | layer   | Acurácia |
| :-----: | :---: | :---: |:---: | :---: |
| SqueezeNet |                           1000   | **-**  | Global Average Pooling | 99,84%   |
| SqueezeNet + SVM |                     1000   | Poly   | Global Average Pooling | 94,22 %   |
| SqueezeNet + SVM + 5 features |        1005   | Poly   | Global Average Pooling | 70,55%   |
| SqueezeNet + SVM + MRMR |              400   | Poly   | Global Average Pooling | 94,32 %   |
| SqueezeNet + SVM + MRMR + 5 features | 405   | Poly   | Global Average Pooling | 70,35%   |

| Displasia Camada Convolucional | número de features | Kernel Type | layer | Acurácia |
| :-----: | :---: | :---: |:---: | :---: |
| SqueezeNet |                          1000   | **-**  | Global Average Pooling | 99,92%   |
| SqueezeNet + SVM|                     1000   | Poly   | Global Average Pooling | 95,78%   |
| SqueezeNet + SVM + 5 features|        1005   | Poly   | Global Average Pooling | 71,77%   |
| SqueezeNet + SVM + MRMR|              400   | Poly   | Global Average Pooling | 96,03%   |
| SqueezeNet + SVM + MRMR + 5 features| 405   | Poly   | Global Average Pooling | 71,14%   |

| Displasia KO Camada Densa | número de features | Kernel Type   | layer   | Acurácia |
| :-----: | :---: | :---: |:---: | :---: |
| SqueezeNet |                          1000   | **-**  | Global Average Pooling | 96,97%   |
| SqueezeNet + SVM|                     1000   | Poly   | Global Average Pooling | 94,22%   |
| SqueezeNet + SVM + 5 features|        1005   | Poly   | Global Average Pooling | 77,10%   |
| SqueezeNet + SVM + MRMR|              400   | Poly   | Global Average Pooling | 93,40%   |
| SqueezeNet + SVM + MRMR + 5 features| 405   | Poly   | Global Average Pooling | 80,88%   |

| Displasia KO Camada Convolucional | número de features | Kernel Type   | layer   | Acurácia |
| :-----: | :---: | :---: |:---: | :---: |
| SqueezeNet |                          1000   | **-**  | Global Average Pooling | 95,98%    |
| SqueezeNet + SVM|                     1000   | Poly   | Global Average Pooling | 94,66%   |
| SqueezeNet + SVM + 5 features|        1005   | Poly   | Global Average Pooling | 74,06%   |
| SqueezeNet + SVM + MRMR|              400   | Poly   | Global Average Pooling | 96,32%    |
| SqueezeNet + SVM + MRMR + 5 features| 405   | Poly   | Global Average Pooling | 75,89%   |

| Displasia WT Camada Densa | número de features | Kernel Type   | layer   | Acurácia |
| :-----: | :---: | :---: |:---: | :---: |
| SqueezeNet |                          1000   | **-**  | Global Average Pooling | 97,11%   |
| SqueezeNet + SVM|                     1000   | Poly   | Global Average Pooling | 94,36%    |
| SqueezeNet + SVM + 5 features|        1005   | Poly   | Global Average Pooling | 83,82%   |
| SqueezeNet + SVM + MRMR|              400   | Poly   | Global Average Pooling | 92,32%   |
| SqueezeNet + SVM + MRMR + 5 features| 405   | Poly   | Global Average Pooling | 83,01%   |

| Displasia WT Camada Convolucional | número de features | Kernel Type   | layer   | Acurácia |
| :-----: | :---: | :---: |:---: | :---: |
| SqueezeNet |                          1000   | **-**  | Global Average Pooling | 95,83%   |
| SqueezeNet + SVM|                     1000   | Poly   | Global Average Pooling | 93,82%    |
| SqueezeNet + SVM + 5 features|        1005   | Poly   | Global Average Pooling | 83,19%   |
| SqueezeNet + SVM + MRMR|              400   | Poly   | Global Average Pooling | 92,11%   |
| SqueezeNet + SVM + MRMR + 5 features| 405   | Poly   | Global Average Pooling | 83,19%   |
