# sub2k-face

**Reconhecimento facial com múltiplas identidades, onde um Arduino Uno decide "quem é essa pessoa" sem nunca processar um único pixel de imagem.**

O problema que isso resolve: reconhecimento facial de verdade não roda num Arduino Uno - ele não tem interface de câmera, e mesmo se tivesse, uma imagem mínima já estoura de longe os 2KB de RAM do chip. A rota óbvia seria um ESP32-CAM, mas esse território já está bem explorado (existem vários projetos prontos usando TensorFlow Lite Micro ou o framework esp-who da própria Espressif). O `sub2k-face` separa o problema: a câmera e a parte pesada (captura + compressão) ficam no PC; o Arduino recebe só uma **representação já comprimida** - um vetor de 32 inteiros - e faz apenas a comparação final. É o mesmo princípio do `sub2k-intent` (comprimir algo grande numa representação mínima que cabe em hardware raquítico), aplicado a imagem em vez de texto.

---

## A ideia central

- **O Arduino nunca vê um pixel.** Tudo que chega nele por Serial é um vetor de 32 bytes (`int8`) - do mesmo tamanho e natureza de um vetor de embedding de texto, não de uma imagem.
- **A compressão é PCA/Eigenfaces**, a técnica clássica de reconhecimento facial pré-deep-learning - uma imagem 32x32 (1024 pixels) vira um vetor de 32 dimensões através de uma matriz de projeção fixa, treinada uma única vez, offline.
- **A decisão é sempre do Arduino.** Ele guarda a tabela de identidades cadastradas em flash (`PROGMEM`) e escolhe a mais próxima por distância Euclidiana - o PC só transporta o vetor, não decide quem é quem.
- **Cadastro de identidades é dado, não código.** Hoje isso ainda exige regravar o sketch (a tabela é fixa em `PROGMEM`) - ver "Direção futura" pra como isso vira dado sem recompilar.

## Pipeline

| Etapa | Onde roda | O que faz |
|---|---|---|
| Captura | PC (webcam) | `cv2.VideoCapture`, frame colorido bruto |
| Detecção de rosto | PC (OpenCV Haar cascade) | localiza e recorta o rosto, com margem |
| Normalização | PC | escala de cinza, resize 32x32, equalização de histograma |
| Projeção PCA | PC | multiplica o vetor de 1024 pixels pela matriz de projeção → vetor de 32 dimensões |
| Quantização | PC | vetor float → `int8`, escala calibrada no treino |
| Transporte | Serial (USB, 115200 baud) | 33 bytes: `0xAA` + 32 bytes do vetor |
| Decisão | Arduino Uno (`firmware/sub2k_face.ino`) | distância Euclidiana contra cada identidade cadastrada; menor distância abaixo do threshold → match |
| Resposta | Serial | 13 bytes: status + nome (ou "desconhecido") |

## Validação antes do hardware

Antes de depender de fotos reais, a técnica foi validada com um gerador procedural de rostos sintéticos (múltiplas identidades, cada uma com parâmetros fixos + ruído de captura simulado) e com um "Arduino falso" em Python fazendo a mesma comparação que o firmware faz. Dois bugs reais apareceram nesse processo e valem registrar, porque são o tipo de erro fácil de repetir:

- **Métrica errada**: cosseno (a métrica usada no `sub2k-intent` pra texto) não separa identidades no espaço PCA - a técnica clássica usa distância Euclidiana, porque magnitude carrega informação real nesse espaço.
- **Vazamento de seed**: uma primeira versão do gerador sintético recriava os parâmetros de cada identidade a cada chamada, fazendo "identidade 0" do enrollment ser uma pessoa diferente de "identidade 0" da query - dava 0% de acerto por um bug de dados, não por limitação da técnica.

Depois de corrigidos: separação limpa entre 5 identidades cadastradas + 4 impostoras, com uma janela de threshold (~55-80) em que a suite de testes com dados sintéticos atinge 100% de acerto, 0% de falso aceite e 0% de falso rejeite - estável em múltiplas seeds. `tests/test_multi_identity.py` roda essa suite sem precisar de câmera nem Arduino.

## Estrutura do projeto

```
sub2k-face/
├── pc/                             # tudo que roda no computador
│   ├── face_capture.py             # detecção + normalização de rosto (compartilhado)
│   ├── pca_eigenfaces.py           # treino/projeção/quantização PCA, com save/load
│   ├── train_pca.py                # CLI: treina a base PCA a partir de uma pasta de fotos
│   ├── enroll.py                   # CLI: cadastra uma identidade via webcam
│   ├── recognize_live.py           # loop de reconhecimento ao vivo
│   ├── serial_protocol.py          # encode/decode das mensagens trocadas com o Arduino
│   └── arduino_link.py             # wrapper de conexão serial
│
├── firmware/                       # o que vai pro Arduino
│   ├── sub2k_face.ino              # lógica de decisão (distância Euclidiana + threshold)
│   └── identities.h                # tabela de identidades em PROGMEM (gerado, não editar à mão)
│
├── tools/
│   └── generate_identities_header.py  # converte identities.json -> firmware/identities.h
│
├── tests/                          # validação sem hardware
│   ├── synthetic_faces.py          # gerador procedural de rostos sintéticos
│   └── test_multi_identity.py      # suite de testes de separação entre N identidades
│
├── requirements.txt
└── README.md
```

## Como usar

**1. Treine a base PCA (uma vez só, offline):**
```bash
pip install -r requirements.txt
python3 pc/train_pca.py --images_dir ./fotos_treino --out model.npz --components 32
```
Junte algumas dezenas de fotos de rosto variadas (não precisam ser das pessoas que serão cadastradas - essa base é genérica). Se trocar a base depois, precisa recadastrar todo mundo.

**2. Cadastre as identidades:**
```bash
python3 pc/enroll.py --name OTTO --model model.npz --samples 8
python3 pc/enroll.py --name PESSOA_B --model model.npz --samples 8
```
Salva a referência de cada pessoa em `identities.json`.

**3. Gere a tabela de identidades e grave o firmware no Arduino:**
```bash
python3 tools/generate_identities_header.py --identities identities.json --out firmware/identities.h
```
Depois compile e grave `firmware/sub2k_face.ino` (junto com o `identities.h` gerado) no Arduino Uno, via Arduino IDE ou `arduino-cli`. Precisa regravar sempre que a tabela de identidades mudar (ver "Direção futura").

**4. Reconhecimento ao vivo:**
```bash
python3 pc/recognize_live.py --model model.npz --port /dev/ttyUSB0
```

## Fluxo completo no Windows

Os scripts em `pc/` resolvem os imports entre si (`face_capture`, `pca_eigenfaces` etc.) sozinhos, mesmo chamados com o caminho `pc\...` a partir da raiz do projeto - não precisa entrar na pasta `pc\` antes.

```powershell
# 0. (uma vez) criar e ativar um ambiente virtual, e instalar as dependências
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 1. treinar a base PCA com fotos reais
python pc\train_pca.py --images_dir .\fotos_treino --out model.npz --components 32

# 2. cadastrar identidades via webcam
python pc\enroll.py --name OTTO --model model.npz --samples 8

# 3. gerar a tabela de identidades pro firmware
python tools\generate_identities_header.py --identities identities.json --out firmware\identities.h

# 4. compilar e gravar o firmware no Arduino (exemplo com arduino-cli;
#    ajuste a porta COM conforme o Gerenciador de Dispositivos > Portas COM e LPT)
arduino-cli compile --fqbn arduino:avr:uno firmware
arduino-cli upload -p COM3 --fqbn arduino:avr:uno firmware

# 5. reconhecimento ao vivo (troque COM3 pela porta correta)
python pc\recognize_live.py --model model.npz --port COM3
```

Se preferir a Arduino IDE em vez do `arduino-cli`: abra `firmware\sub2k_face.ino` (o `identities.h` gerado precisa estar na mesma pasta), selecione a placa "Arduino Uno" e a porta COM certa, e clique em "Carregar" - dispensa o passo 4 acima.

## Protocolo serial (115200 baud)

```
PC -> Arduino:  [0xAA] [vetor: 32 bytes int8]
Arduino -> PC:  [status: 1 byte] [nome: 12 bytes ASCII, padded com \0]
status = 0x01 (match) ou 0x00 (desconhecido)
```

Tamanho fixo, sem checksum - o volume de dados é minúsculo e a conexão é um cabo USB direto, checksum viraria complexidade sem benefício real aqui (diferente do `uno-nvscript`, onde o bytecode enviado é gravado permanentemente na EEPROM e um upload corrompido seria mais caro de corrigir).

## Segurança e limites conhecidos

- **Nenhum rosto detectado no frame** → o Python simplesmente não envia nada nesse ciclo; não é um erro, é o caso comum (ninguém na frente da câmera).
- **Timeout de resposta do Arduino** → `pc/arduino_link.py` estoura `TimeoutError` com mensagem explícita em vez de travar esperando indefinidamente.
- **Mismatch de dimensão** → se o modelo PCA carregado não tiver exatamente 32 componentes (o tamanho fixo do protocolo), `pc/enroll.py`/`pc/recognize_live.py` recusam rodar com uma mensagem clara, em vez de falhar de forma obscura lá na frente.
- **Capacidade da tabela de identidades**: cada identidade custa ~44 bytes em flash (12 do nome + 32 do vetor) - o Uno tem 32KB de flash, então escala pra centenas de identidades sem tocar na RAM.
- **Threshold calibrado com dados sintéticos** (constante `THRESHOLD_EUCLIDEAN` em `firmware/sub2k_face.ino`, hoje em 70) - precisa ser recalibrado com fotos reais, já que variação real de iluminação/pose tende a ser maior que a simulada.

## Requisitos

- Python 3.10+ no PC, com webcam
- `opencv-python`, `numpy`, `pyserial` (`pip install -r requirements.txt`)
- Arduino Uno (ATmega328P) ou qualquer AVR com flash suficiente pra tabela de identidades - **não precisa de ESP32-CAM**, já que a câmera fica no PC
- Arduino IDE ou `arduino-cli` pra compilar/gravar `firmware/sub2k_face.ino`

## Status

- ✅ Validação da técnica (PCA + Euclidiana + multi-identidade) com dados sintéticos
- ✅ Pipeline Python completo (captura → detecção → PCA → quantização → protocolo serial)
- ✅ Teste ponta-a-ponta com "Arduino falso" em Python (30/30 casos corretos)
- ✅ Firmware `.ino` do Arduino (tabela de identidades + lógica de decisão)
- ⬜ Validação com fotos reais e recalibração de threshold (ainda usa o valor calibrado com dados sintéticos)

## Direção futura: combinar com o uno-nvscript

O firmware deste projeto hoje é fixo (tabela de identidades embutida em `PROGMEM`, gravada junto com o sketch a cada cadastro/remoção). Uma direção natural é reescrever esse firmware sobre a VM do [`uno-nvscript`](../uno-nvscript): a tabela de identidades viraria dado enviado por Serial e gravado em EEPROM, em vez de exigir recompilar o sketch pra cadastrar ou remover uma pessoa - o mesmo ganho que o `uno-nvscript` já dá pra regras de automação. **Essa integração ainda não está implementada** - é um próximo passo natural, não algo que já roda.

## Sobre este projeto

Desenvolvido como parte de uma série de experimentos de engenharia de restrição extrema em hardware ultra-limitado, no contexto do ecossistema de automação/IoT do **DevSoft JARVIS AI**.

**Autor:** Bruno Nunes da Silva (criador do DevSoft JARVIS AI)<br>
**Conheça o DevSoft JARVIS AI:** https://devsoft-ai.webnode.page/<br>
**Canal no YouTube:** https://www.youtube.com/@devsoftai5538

## Licença

MIT - use, modifique e distribua livremente, mantendo os créditos de autoria.
