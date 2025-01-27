# Detector de Hotspots em Painéis Solares usando YOLOv8

Este é um sistema de detecção de hotspots em painéis solares usando o modelo YOLOv8 e PyTorch. O programa realiza treinamento e detecção de anomalias térmicas (hotspots) em imagens termográficas de painéis solares.

## Requisitos

### Recomendações Principais

- Python 3.x
- PyTorch
- TorchVision
- TorchAudio
- Ultralytics (YOLOv8)
- OpenCV (cv2)
- NumPy
- Pandas
- Matplotlib
- PIL (Python Imaging Library)
- PyYAML
- Jupyter Notebook (melhor visualização)
- tqdm (barras de progresso)
- shutil (manipulação de arquivos)
- random (geração números aleatórios)
- math (funções matemáticas)
- gc (gerenciamento de memória)

O comando de instalação completo se precisar:

pip install torch torchvision torchaudio ultralytics opencv-python numpy pandas matplotlib pillow pyyaml tqdm jupyter

### Hardware
- GPU compatível com CUDA 
- Memória RAM suficiente para processamento de imagens

## Estrutura do Projeto

O projeto dsegue a seguinte estrutura de diretórios:
```
projeto/
├── datasets/
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
├── runs/
│   └── detect/
└── Teste TCC painel solar10.ipynb
```

## Configuração Inicial

1. O dataset deve seguir a seguinte estrutura:
   - Imagens no formato .jpg
   - Anotações correspondentes no formato .txt (formato YOLO)
   - Divida suas imagens entre as pastas train/ e test/ (padrão do processo)

2. Configure o caminho do dataset:
   
   dataset_directory = r'caminho/para/seu/dataset'
   

## Funcionalidades

### 1. Preparação de Dados
- Organização de arquivos entre diretórios train/, valid/ e test/
- Geração automática do arquivo data.yaml

### 2. Treinamento
- Utiliza YOLOv8n (modelo nano), mas pode usar superiores
- Configurações otimizadas para detecção de hotspots
- Parâmetros ajustáveis para maior precisão:
  - 40 épocas
  - Tamanho de imagem: 640x640
  - Otimizador: AdamW
  - Learning rate: 0.001
  - Patience: 100

### 3. Predição e Visualização
- Detecção em imagens de teste
- Visualização de resultados com bounding boxes
- Animação em das detecções ou em grid

## Como Usar

1. Preparação do programa para o VScode
  
   pip install torch torchvision torchaudio ultralytics opencv-python numpy pandas matplotlib pillow pyyaml
 
2. **Resultados**
   - Os resultados do treinamento serão salvos em `runs/detect/`
   - O modelo treinado será salvo como `best.pt`
   - Visualizações e métricas serão geradas automaticamente

## Parâmetros Personalizáveis

- `train_ratio`: Proporção de dados para treino (padrão: 0.8)
- `valid_ratio`: Proporção de dados para validação (padrão: 0.2)
- `epochs`: Número de épocas de treinamento (padrão: 40)
- `conf`: Limiar de confiança para detecção (padrão: 0.3)

## Observações Importantes

1. O programa requer uma GPU compatível com CUDA para melhor performance
2. As configurações CUDA são otimizadas para GPUs específicas e podem precisar de ajustes
3. O modelo está configurado para detecção de classe única por enquanto (hotspot)

## Resolução de Problemas (mini histórico que eu fiz)

Erro de Memória CUDA
   - Reduzi o tamanho do batch
   - Diminui a resolução das imagens
   - Fiz diversos ajustes voltados pra placa GTX 1650, mas podem ser desnecessários.

## Customização

Para adaptar o modelo para outros tipos de detecção:
1. Modifique a classe em `data_yaml`
2. Ajuste os parâmetros de treinamento em `model.train()` (muitos foram pra GTX 1650 mas não são necessários sempre)
3. Adapte as funções de visualização conforme necessário, a etapa depois do predict é puramente cosmética.
```

