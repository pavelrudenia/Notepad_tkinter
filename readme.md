

# Распознавание именованных сущностей

<img src="https://github.com/dialogue-evaluation/RuREBus/raw/master/rurebus_logo.png" alt="RuREBus logo" width="200"/>

**RuREBus** – это соревнование по распознаванию именованных сущностей (NER) в русскоязычных текстах. Здесь представлен проект, в котором мы исследуем различные алгоритмы глубокого обучения для решения этой задачи.

## Цель работы

Наша цель состоит в том, чтобы научиться применять нейросетевые подходы в задачах распознавания именованных сущностей и показать, что контекстно-зависимые представления, полученные из внутренних состояний языковой модели, в комбинации с дистрибутивными представлениями, полезны в задачах NER и позволяют достичь высоких результатов.

## Задание

1. **Скачайте датасет RuREBus** – вы можете найти его в официальном репозитории [RuREBus](https://github.com/dialogue-evaluation/RuREBus).
2. **Реализуйте нейронную сеть** – используйте указанный в варианте алгоритм нейронной сети для задачи NER.
3. **Оцените качество модели на тестовых данных** – протестируйте вашу модель на тестовых данных и получите метрики качества.
4. **Создайте отчёт в виде README на GitHub** – создайте описание вашего проекта, включая результаты, инструкции по запуску и другую полезную информацию.




## Исходный код

| Описание | Код |
| --- | --- |
| Импорт необходимых библиотек | ```python import pandas as pd import torch import torch.nn as nn import torch.optim as optim from torch.utils.data import Dataset, DataLoader from sklearn.model_selection import train_test_split ``` |
| Загрузка датасета из CSV файла | ```python data = pd.read_csv("rurebus_dataset.csv") ``` |
| Разделение данных на тренировочный и тестовый наборы | ```python train_data, test_data = train_test_split(data, test_size=0.2, random_state=42) train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42) ``` |
| Определение класса `RurebusDataset` | ```python class RurebusDataset(Dataset): def __init__(self, data): self.data = data def __len__(self): return len(self.data) def __getitem__(self, index): input_sequence = self.data.iloc[index]['sequence'] label = self.data.iloc[index]['label'] return input_sequence, label ``` |
## Помощь и контрибуция

Если у вас возникли вопросы или вы хотите внести свой вклад в проект, пожалуйста, создайте issue или pull request в этом репозитории.

## Лицензия

Этот проект лицензирован под лицензией MIT. Подробности могут быть найдены в файле [LICENSE](https://github.com/username/repo/blob/main/LICENSE).
