# sem_mascara
## Projeto de Detecção de Máscara Facial
Este projeto utiliza um modelo pré-treinado para identificar se pessoas em quadros de vídeo estão usando máscara facial ou não. O modelo processa arquivos de vídeo, extrai frames e classifica cada quadro para determinar a presença da máscara.

## Estrutura do Projeto
bash
Copy code
facemask-detection
├── src
│   ├── frame_processor.py      # Funções para carregar o modelo e classificar imagens
│   ├── video_loader.py         # Funcionalidade para extrair frames de vídeos
│   ├── serial.py               # Processa frames de forma sequencial e salva os resultados
│   └── utils
│       └── __init__.py         # Placeholder para funções utilitárias
├── models
│   └── facemask_detection.h5   # Modelo pré-treinado para detecção de máscara
├── data
│   ├── raw
│   │   └── tokyo.mp4           # Vídeo de exemplo para processamento
│   └── processed
│       └── frames              # Diretório para frames extraídos
├── requirements.txt            # Dependências do projeto
└── README.md                   # Documentação do projeto
Instruções de Configuração
Clone o Repositório

bash
Copy code
git clone <repository-url>
cd facemask-detection
Instale as Dependências
Certifique-se de ter o Python instalado. Depois, instale os pacotes necessários:

bash
Copy code
pip install -r requirements.txt
Utilização
Extrair Frames do Vídeo
Execute o script video_loader.py para extrair frames do vídeo de exemplo:

bash
Copy code
python src/video_loader.py
Classificar os Frames (Processamento Sequencial)
Após extrair os frames, execute o script serial.py para classificar cada frame:

bash
Copy code
python src/serial.py
Visualizar Resultados
Os resultados serão salvos no arquivo results_serial.csv na raiz do projeto. Este arquivo contém o nome dos frames e a probabilidade de não estar usando máscara.

Utilização de Programação Paralela
Para acelerar o processamento dos frames, é possível utilizar programação paralela (multiprocessamento). Um exemplo de script paralelo seria parallel.py, que processa múltiplos frames simultaneamente, aproveitando todos os núcleos do processador:

bash
Copy code
python src/parallel.py
O arquivo de resultados será salvo como results_parallel.csv.

Isso reduz significativamente o tempo de classificação em grandes volumes de dados.

Exemplo de implementação simples em Python (multiprocessing):

python
Copy code
from multiprocessing import Pool

def classify_frame(frame_path):
    # Função para classificar um único frame
    ...

if __name__ == "__main__":
    frame_list = [...]  # Lista de caminhos dos frames
    with Pool() as pool:
        results = pool.map(classify_frame, frame_list)
Observações
Certifique-se de que o arquivo facemask_detection.h5 está presente no diretório models.

Você pode modificar o parâmetro every_n no video_loader.py para controlar quantos frames são extraídos do vídeo.
