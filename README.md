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


#### Через телеграм:

Зайдите в файл tg_bot.py и вставьте вот [сюда](https://github.com/vaskers5/furniture_background_generation/blob/470cb1d607713400e619e628b78b6e37deb19473/tg_bot.py#L24C1-L24C24) телеграм токен для вашего бота. Далее вы можете тестировать бота с помощью кнопок в интерфейсе

```
python tg_bot.py
```


#### Через пример python:

```bash
python test.py
```
