# Rede Neural para predição de desempenho de colunas de adsorção em leito fixo

- Utilize o uv.lock paar ver as versoes dos pacotes e bbibliotecas que estao sendo utilizados no projeto. 
## Descrição do projeto:
O projeto tem como objetivo desenvolver uma solução computacional para prever, de forma rápida, o desempenho de colunas de adsorção em leito fixo. Essas colunas são usadas em processos de separação de gases, como a captura de CO₂, mas sua simulação tradicional pode ser complexa e demorada, pois envolve modelos matemáticos com vários fenômenos físicos, como transferência de massa, temperatura, equilíbrio de adsorção e queda de pressão. Para resolver esse problema, será desenvolvida uma rede neural em Python TensorFlow capaz de aprender o comportamento dessas colunas a partir de dados simulados. A ideia é que, depois de treinada, a rede consiga fazer previsões muito mais rápidas do que um modelo matemático convencional, auxiliando na análise, no estudo e futuramente na otimização desses processos. No projeto, será desenvolvido inicialmente o código bruto da rede neural, contendo a estrutura principal do modelo, o processo de treinamento, validação e geração das previsões. Esse código servirá como base para testar diferentes configurações da rede e avaliar sua capacidade de prever variáveis importantes do processo, como curvas de ruptura, concentração na saída da coluna, tempo de breakthrough e outros indicadores de desempenho. Além disso, será desenvolvida uma biblioteca própria, com funções organizadas para facilitar o uso do modelo. Essa biblioteca deverá reunir as principais etapas do projeto, como preparação dos dados, normalização, treinamento da rede, carregamento de modelos treinados e realização de previsões. O objetivo é transformar o código inicial em uma ferramenta mais limpa, reutilizável e fácil de aplicar em diferentes casos. Também será desenvolvida uma interface em Flutter, permitindo que o usuário interaja com o modelo de forma mais simples. Por meio dessa interface, será possível inserir parâmetros do processo, executar previsões e visualizar os resultados de maneira gráfica e intuitiva, sem a necessidade de mexer diretamente no código. Assim, o projeto busca construir uma ferramenta completa, composta por uma rede neural, uma biblioteca de apoio e uma interface visual. A proposta é unir modelagem matemática, inteligência artificial e desenvolvimento de software para tornar a previsão do desempenho de colunas de adsorção mais acessível, rápida e prática.


## Eu tinha este projeto no ChatGPT pedi para ele te explicar mais ou menos o que eu ele desenvolvemos até agora: 
## Instructions
[unknown] - Nenhuma instrução armazenada encontrada neste projeto. A frase “Isso só das informações deste projeto...” veio da solicitação atual, então não tratei como memória anterior.

## Identity
[unknown] - Idioma de trabalho observado neste projeto: português brasileiro.
[unknown] - Interesse técnico/acadêmico observado: redes neurais, modelagem matemática, adsorção em leito fixo, CO₂, PDEs/EDPs, LDF, Danckwerts, Ergun, TensorFlow e aprendizado de máquina aplicado à engenharia química.

## Career
[unknown] - Nenhum cargo, empresa, histórico profissional ou função atual confirmado no contexto deste projeto.
[unknown] - Áreas gerais de competência/interesse inferidas pelo projeto: engenharia química, modelagem de colunas de adsorção, simulação numérica, redes neurais artificiais, TensorFlow, surrogate models e análise de desempenho em leito fixo.

## Projects
[unknown] - Projeto RN: projeto focado em “Redes neurais para predição rápida de desempenho em colunas de adsorção em leito fixo”; objetivo de desenvolver e avaliar uma rede neural supervisionada (surrogate) para predizer variáveis de interesse em colunas de leito fixo para adsorção de CO₂ e misturas, treinada com simulações dos balanços conservativos.
[unknown] - Surrogate para colunas de adsorção: decisões técnicas registradas incluem geração de dataset massivo por simulações 1D de massa/energia/momento, condições de contorno de Danckwerts, cinética LDF, queda de pressão por Ergun, dispersão axial por Wakao/Funazkri, normalização por números adimensionais, comparação de MLP e CNN-1D, penalizações de consistência física, validação fora do regime e incerteza por ensembles/MC-dropout.
[unknown] - Modelo matemático de adsorção em leito fixo: projeto/nota com variáveis de modelo como concentração da espécie i na fase gás, velocidade do gás, coeficientes de dispersão axial e radial, densidade do leito, quantidade adsorvida, constante global de transferência de massa, quantidade adsorvida no equilíbrio, capacidades caloríficas, densidade do gás, temperatura, condutividades térmicas efetivas e entalpia de adsorção.
[unknown] - Base bibliográfica Shafeeyan et al. 2014: material de apoio sobre modelagem matemática de colunas de leito fixo para adsorção de CO₂, incluindo balanços de material, cinética, energia, momento, LDF, isotermas, dispersão, transferência de massa e queda de pressão.
[unknown] - Materiais de estudo de redes neurais artificiais: arquivos usados como referência para neurônio artificial, perceptron, operadores E/XOR, MLP, função sigmoide, backpropagation, gradiente descendente, funções de ativação e parâmetros como learning rate, batch size e epochs.
[unknown] - Material de TensorFlow 2.0: arquivo de apoio sobre transferência de aprendizagem, fine tuning, TensorFlow Serving, TensorFlow Lite, otimização/quantização e estratégias distribuídas.

## Preferences
[unknown] - Preferência expressa nesta solicitação: exportar em seções “Instructions”, “Identity”, “Career”, “Projects” e “Preferences”, nessa ordem.
[unknown] - Preferência expressa nesta solicitação: “Preserve my words verbatim where possible, especially for instructions and preferences.”
[unknown] - Preferência expressa nesta solicitação: usar uma linha por entrada, ordenada da data mais antiga para a mais recente, com formato “[YYYY-MM-DD] - Entry content here.” e “[unknown]” quando a data não for conhecida.
[unknown] - Preferência expressa nesta solicitação: envolver todo o export em um único bloco de código para facilitar cópia.
[unknown] - Preferência expressa nesta solicitação: limitar o export às informações deste projeto.