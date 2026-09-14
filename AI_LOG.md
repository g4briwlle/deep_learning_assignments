# Arquivo com uso de IA no PA1

## Gabrielle

Criei um "notebook" no Gemini Pro com o enunciado do trabalho e com os slides do conteúdo. 

Comecei pedindo como consertar o meu código da rede ResNet que tinha esboçado (que posteriormente a escolha da rede foi alterada) e pedi códigos de funções completas (eu revisando eles) em momentos de ansiedade de não terminar o trabalho a tempo.

Pedi revisão de códigos feitos em vários momentos e revisão de se os conceitos batiam com os resultados (ex: a mAP deveria ser essa nesse momento?).

Pedi muita ajuda para integrar o .ipynb rodado no colab pelo meu computador, ruim e sem gpu, com o github e fazer commits pelo próprio colab. 

Pedi ajuda em alguns conflitos de merge e versionamento do repositório. Em algum momento tive alguma dúvida sobre modularização de código também.

Pedi ajuda pra entender o que alguns enunciados queriam dizer e o que era esperado de ser feito (senti que as instruções do trabalho foram muito vagas).


## Daniel

Criei um "gem" no Gemini Pro com o enunciado do trabalho e com os slides do conteúdo de Segmentação Semântica e, conforme o decorrer do trabalho, adicionei o código implementado ao contexto desse gem.

Pedi ajuda para entender como estava a arquitetura do modelo e sua matemática por trás implementar o eixo 1 da parte 3 e percebi que a arquitetura inicial de ResUnet34 tinha algumas falhas estruturais que não permitiam a implementação de outro decoder, requisito para o eixo 1. Pedi ajuda para a implementação da nova arquitetura, de UNet com Encoder desacoplado.

Pedi ajuda para criar um cache local dos dados, diminuindo a parte I/O bound de carga dos dados para diminuir o tempo gasto treinando o modelo colocando a carga apenas na GPU.

Pedi ajuda com como fazer as transformações necessárias nas imagens para melhorar o modelo de acordo com o diagnóstico da parte 5.