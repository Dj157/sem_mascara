# Detecção de Máscara Facial em Vídeos Públicos com Programação Paralela

## Introdução

Este projeto tem como objetivo detectar automaticamente pessoas **sem máscara facial** em vídeos de multidões, utilizando técnicas de **Inteligência Artificial (IA)** e **programação paralela** para otimizar o tempo de processamento.  
A aplicação visa auxiliar políticas de saúde pública, permitindo análise rápida de grandes volumes de vídeo em ambientes como eventos, ruas ou transporte.

link do video utilizado 7.9GB: https://drive.google.com/file/d/1E6oUgTasz6nmu-yKv9NtYcUFMQW7M3MI/view

link do video partido para fazer testes:https://drive.google.com/drive/folders/1gCbbZeGvEEQuvc0zyyB_4gheStexio2F


---

## Descrição do Problema

Processar vídeos extensos de multidões para identificar o uso (ou não) de máscaras faciais é um desafio computacional relevante, especialmente em contextos de pandemia.  
O processamento sequencial é lento e pouco eficiente em grandes datasets, tornando essencial o uso de paralelismo para viabilizar análises em tempo hábil, mesmo em hardware modesto como o MacBook Air M1.

---

## Descrição da Solução

O pipeline foi estruturado de forma modular, com otimização para o MacBook Air M1 (4 núcleos), facilitando tanto o processamento paralelo quanto a manutenção do código.

### Estrutura dos scripts principais

- **`video_loader.py`**: Extrai frames do vídeo de entrada (ajustável por taxa de frames).
- **`frame_processor.py`**: Detecta rostos em cada frame e classifica cada rosto como "com máscara" ou "sem máscara" usando um modelo CNN leve.
- **`serial.py`**: Executa o pipeline **sequencialmente** (para servir de comparação de performance).
- **`macpipe.py`**: Executa o pipeline em **paralelo**, otimizando para múltiplos núcleos e threads, de acordo com o hardware.
- **`pipeline.py`**: Script principal, responsável por orquestrar a execução, receber argumentos de entrada (quantidade de threads, vídeo, etc.), iniciar os pipelines e salvar resultados e métricas.

### Passo a passo do processamento

1. **Extração dos frames:**  
   O vídeo é dividido em frames (`video_loader.py`), por exemplo, 1 frame a cada X segundos.

2. **Processamento de cada frame:**  
   Cada frame passa por:
   - **Detecção de rostos** (`frame_processor.py`)
   - **Classificação de máscara**: cada rosto é classificado como "com máscara" ou "sem máscara" por uma CNN (ex: MobileNetV2, YOLOv5 adaptado).

3. **Paralelismo:**  
   O pipeline paraleliza a análise dividindo os frames entre múltiplos processos, utilizando até 4 threads (otimizado para o M1, que possui 4 núcleos físicos).

4. **Coleta e visualização dos resultados:**  
   Os resultados de cada thread são unificados.  
   O sistema contabiliza e exporta o total de pessoas com/sem máscara, por frame e no total, gerando tabelas de eficiência e permitindo futura geração de vídeos anotados com bounding boxes.

### Exemplo de frame analisado

Cada frame do vídeo é processado para identificar rostos e classificar o uso de máscara, como neste exemplo abaixo (rua movimentada, multidão sob chuva):

![Exemplo de frame analisado](frame_00662.jpg)

---

## Resultados

Os testes foram realizados em um **MacBook Air M1 (4 núcleos físicos)**.  
O tempo de execução, speedup e eficiência foram calculados para diferentes números de threads, utilizando um vídeo grande.

| Threads | Tempo (s) | Speedup | Eficiência (%) |
|---------|-----------|---------|---------------|
|   1     | 3231,57   | 1,00    | 100,0         |
|   2     | 2233,94   | 1,45    | 72,4          |
|   4     | 1288,01   | 2,51    | 62,7          |
|   8     | 1269,37   | 2,55    | 31,9          |
|   16    | 1396,12   | 2,32    | 14,5          |

- **Speedup:** Tempo sequencial (1 thread) dividido pelo tempo paralelo.
- **Eficiência:** Speedup / nº de threads × 100%

**Observações:**
- O melhor resultado foi atingido com 4 threads, respeitando o limite físico do M1.
- Usar mais threads não traz ganhos reais (por limitação de hardware e overhead de gerenciamento).

---

## Conclusão

- A utilização de **programação paralela** permitiu acelerar o processamento em mais de 2,5x no MacBook Air M1, tornando viável a análise de vídeos longos em tempo razoável.
- O projeto foi cuidadosamente otimizado para o hardware disponível, demonstrando ganhos reais de eficiência com paralelismo bem aplicado.
- O pipeline é modular e facilmente adaptável a outros contextos, vídeos ou modelos de detecção.
- O uso de mais threads que núcleos reais não aumenta a performance devido ao overhead de gerenciamento.
- O sistema está pronto para aplicações reais em análise de uso de máscaras em grandes multidões, podendo ser expandido para análise em tempo real ou integração com outros sistemas de monitoramento.
