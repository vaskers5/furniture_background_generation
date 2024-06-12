# Имя Docker образа
IMAGE_NAME = furniture_background_generation

# Путь к Dockerfile
DOCKERFILE_PATH = dockerfiles/Dockerfile.pipeline

# Цель по умолчанию
all: build run

# Сборка Docker образа
build:
	docker build -f $(DOCKERFILE_PATH) -t $(IMAGE_NAME) .

# Запуск Docker контейнера
run:
	docker run -it --rm $(IMAGE_NAME)

# Очистка Docker образов
clean:
	docker rmi $(IMAGE_NAME)