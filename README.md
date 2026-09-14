# Checkpoints

Os pesos dos modelos treinados estão disponíveis na raíz do repositório, com nomes descritivos de qual parte do assingment o arquivo `.pth` se refere.

# Comando para avaliar o modelo

Tendo o repositório clonado, crie e ative um ambiente virtual e baixe todos os pacotes presentes no `pyproject.toml` como preferir.

Rode o arquivo eval_network da seguinte maneira para testar o modelo final (o implementando na parte 5):

```bash
python eval_network.py
```

Se quiser testar algum outro modelo de outra parte, adicione mais um argumento ao comando:

```bash
python eval_network.py unet_part1
```

Há uma lista pré-estabelecida de modelos disponíveis para a avaliação. Caso adicione um nome inválido, o programa de avisará e te dará a lista de nomes válidos.