

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
| В начале импортируются все необходимые библиотеки. pandas используется для работы с данными, torch - для работы с нейросетями, scikit-learn - для разделения данных на тренировочный, валидационный и тестовый наборы данных, а также для оценки качества модели на тестовом наборе данных. | ```python import pandas as pd import torch import torch.nn as nn import torch.optim as optim from torch.utils.data import Dataset, DataLoader from sklearn.model_selection import train_test_split ``` |
| Загружается датасет из CSV файла с помощью библиотеки pandas. Предполагается, что файл с данными называется "rurebus_dataset.csv". | ```python data = pd.read_csv("rurebus_dataset.csv") ``` |
| Данные разделяются на тренировочный, валидационный и тестовый наборы с помощью функции train_test_split из scikit-learn. Сначала данные разделяются на тренировочный и тестовый наборы, а затем тренировочный набор разделяется на тренировочный и валидационный наборы. | ```python train_data, test_data = train_test_split(data, test_size=0.2, random_state=42) train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42) ``` |
| Определяется класс RurebusDataset, который является подклассом torch.utils.data.Dataset. В этом классе определены методы __init__, __len__ и __getitem__, которые необходимы для создания кастомного датасета в PyTorch. __init__ инициализирует датасет, __len__ возвращает длину датасета, а __getitem__ возвращает элемент датасета по индексу. | ```python class RurebusDataset(Dataset): def __init__(self, data): self.data = data def __len__(self): return len(self.data) def __getitem__(self, index): input_sequence = self.data.iloc[index]['sequence'] label = self.data.iloc[index]['label'] return input_sequence, label ``` |
| Определяется класс модели CharCNNBiLSTMCRF, который является подклассом torch.nn.Module. В этом классе определены методы __init__ - для инициализации модели и forward - для применения модели к входным данным. Модель состоит из эмбеддингов символов, сверточного слоя, пулинга, двунаправленного LSTM слоя и полносвязного слоя. | ```class CharCNNBiLSTMCRF(nn.Module): def __init__(self, char_vocab_size, char_embedding_dim, lstm_hidden_dim, num_classes): super(CharCNNBiLSTMCRF, self).__init__() self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim) self.conv1d = nn.Conv1d(char_embedding_dim, 100, kernel_size=3, padding=1) self.max_pool = nn.MaxPool1d(kernel_size=2) self.bilstm = nn.LSTM(100, lstm_hidden_dim, bidirectional=True) self.fc = nn.Linear(2 * lstm_hidden_dim, num_classes) def forward(self, char_seqs): char_embeddings = self.char_embedding(char_seqs) char_embeddings = char_embeddings.permute(0, 2, 1) conv_output = self.conv1d(char_embeddings) max_pool_output = self.max_pool(conv_output) lstm_output, _ = self.bilstm(max_pool_output) lstm_output = lstm_output.permute(0, 2, 1) max_pool_output = torch.max(lstm_output, dim=2)[0] output = self.fc(max_pool_output) return output ``` |
| Определяется функция prepare_data, которая создает экземпляры RurebusDataset для каждого набора данных (тренировочного, валидационного и тестового) и возвращает эти экземпляры. | ```def prepare_data(): dataset = RurebusDataset(train_data) val_dataset = RurebusDataset(val_data) test_dataset = RurebusDataset(test_data) return dataset, val_dataset, test_dataset``` |
| Определяется функция collate_fn, которая объединяет последовательности символов и метки в батчи. Последовательности символов преобразуются в тензоры значений Unicode с помощью функции ord. Затем последовательности символов заполняются до максимальной длины пакета с помощью функции pad_sequence из torch.nn.utils.rnn. | ```def collate_fn(data): char_seqs, labels = zip(*data) char_seqs = [torch.tensor([ord(c) for c in seq]) for seq in char_seqs] char_seqs = nn.utils.rnn.pad_sequence(char_seqs, batch_first=True) labels = torch.tensor(labels) return char_seqs, labels``` |
| Определяются функции train и evaluate для обучения и оценки модели соответственно. Функция train выполняет пакетное обучение модели на тренировочных данных, вычисляет функцию потерь CrossEntropyLoss, выполняет обратное распространение ошибки и оптимизацию модели с помощью оптимизатора Adam. Функция evaluate оценивает модель на данных из data_loader и вычисляет среднюю функцию потерь и точность модели.. | ```def train(model, train_data, val_data, num_epochs, lr, device): train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn) val_loader = DataLoader(val_data, batch_size=32, collate_fn=collate_fn) loss_fn = nn.CrossEntropyLoss() optimizer = optim.Adam(model.parameters(), lr=lr) model.to(device) for epoch in range(num_epochs): model.train() train_loss = 0.0 for char_seqs, labels in train_loader: char_seqs = char_seqs.to(device) labels = labels.to(device) optimizer.zero_grad() output = model(char_seqs) loss = loss_fn(output, labels) loss.backward() optimizer.step() train_loss += loss.item() * char_seqs.size(0) train_loss /= len(train_data) val_loss, val_accuracy = evaluate(model, val_loader, device) print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}") def evaluate(model, data_loader, device): model.eval() loss_fn = nn.CrossEntropyLoss() total_loss = 0.0 total_correct = 0 with torch.no_grad(): for char_seqs, labels in data_loader: char_seqs = char_seqs.to(device) labels = labels.to(device) output = model(char_seqs) loss = loss_fn(output, labels) total_loss += loss.item() * char_seqs.size(0) preds = torch.argmax(output, dim=1) total_correct += torch.sum(preds == labels).item() avg_loss = total_loss / len(data_loader.dataset) accuracy = total_correct / len(data_loader.dataset) * 100 return avg_loss, accuracy ``` |
| Наконец, выполняется подготовка данных, определение модели, гиперпараметров обучения (количество эпох, скорость обучения) и устройства (GPU или CPU). Затем модель обучается с помощью функции train, а затем оценивается на тестовом наборе данных с помощью функции evaluate.| ```train_data, val_data, test_data = prepare_data() char_vocab_size = 256 char_embedding_dim = 100 lstm_hidden_dim = 128 num_classes = 3 model = CharCNNBiLSTMCRF(char_vocab_size, char_embedding_dim, lstm_hidden_dim, num_classes) num_epochs = 10 lr = 0.001 device = "cuda" if torch.cuda.is_available() else "cpu" train(model, train_data, val_data, num_epochs, lr, device) test_loader = DataLoader(test_data, batch_size=32, collate_fn=collate_fn) test_loss, test_accuracy = evaluate(model, test_loader, device) print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")``` |
## Помощь и контрибуция

Если у вас возникли вопросы или вы хотите внести свой вклад в проект, пожалуйста, создайте issue или pull request в этом репозитории.

## Лицензия

Этот проект лицензирован под лицензией MIT. Подробности могут быть найдены в файле [LICENSE](https://github.com/username/repo/blob/main/LICENSE).
