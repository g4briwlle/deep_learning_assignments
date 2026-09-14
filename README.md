# Ambiente

Crie o ambiente virtual e baixe os pacotes como preferir. Recomendamos o uso do pacote uv:

```bash
pip install uv
uv sync
```

ele baixará as bibliotecas de acordo com o arquivo `pyproject.toml`, mas baixar as dependências com

```bash
pip install -r requirements.txt
```

também deve funcionar

# Checkpoints

Os pesos dos modelos treinados estão disponíveis na raíz do repositório, com nomes descritivos de qual parte do assingment o arquivo `.pth` se refere.

# Comando para avaliar o modelo

Rode o arquivo eval_network da seguinte maneira para testar o modelo final (o implementando na parte 5):

```bash
python eval_network.py
```

Se quiser testar algum outro modelo de outra parte, adicione mais um argumento ao comando:

```bash
python eval_network.py unet_part1
```

Há uma lista pré-estabelecida de modelos disponíveis para a avaliação. Caso adicione um nome inválido, o programa de avisará e te dará a lista de nomes válidos.