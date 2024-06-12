# furniture_background_generation

### Сборка пайплайна:

#### Установка зависимостей:
```bash
conda create -n "ins_ever" python=3.10
conda activate ins_ever
pip install -r base_requirements.txt
pip install git+https://github.com/sberbank-ai/Real-ESRGAN.git
```

### Загрузка весов:
```bash
gdown 1LmXAzEuhVfM-DVr-1nQ7wqy2hSqeZlLL
unzip weights.zip -d weights
```

### Тестирование 

```bash
python test.py
```