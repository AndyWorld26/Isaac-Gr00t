# Глава 2: Пайплайн инференса VLA-модели на чистом Python

## 1. Архитектура чистого Python-контура (Без симуляторов)

Тестирование, валидация и непосредственное развертывание Isaac GR00T на бортовом вычислителе робота (например, NVIDIA Jetson Thor) требуют полной изоляции нейросетевого контура от тяжелых графических сред симуляции вроде Isaac Sim/Omniverse. В физической реальности или при офлайн-анализе датасетов робот общается с управляющим софтом через потоки сырых данных: байты из видеозахвата камер (V4L2/OpenCV), пакеты проприоцепции из шины EtherCAT/CAN и текстовые строки из высокоуровневого планировщика задач.

Контур инференса на «чистом Python» решает критически важные задачи:

- **Профайлинг и оптимизация задержек (Latency Tracking)**: Измерение чистого времени выполнения (forward pass) Системы 2 и Системы 1 без накладных расходов на расчет контактов твердых тел и рендеринг физики.
- **Офлайн-валидация (Policy Evaluation)**: Прогон предобученной модели по заранее записанным логам (HDF5/Zarr) успешных демонстраций человека для оценки метрик сходимости траекторий.
- **Разделение вычислений**: Возможность запустить тяжелый VLM-бэкбон на удаленном сервере с картами NVIDIA H100 (Система 2), а легкий трансформер действий (Система 1) выполнять локально на роботе, передавая между ними только компактные эмбеддинги по протоколу gRPC.

```
+-----------------------------------------------------------------------+
|                         ПРОГРАММНЫЙ СТЭК PYTHON                       |
|                                                                       |
|  +--------------------+  +--------------------+  +-----------------+  |
|  | Камеры (Head/Wrist)|  |Текстовая инструкция|  |  Проприоцепция  |  |
|  |   [NumPy / RGB]    |  |      [String]      |  | [Joints / IMU]  |  |
|  +--------------------+  +--------------------+  +-----------------+  |
|            |                       |                      |           |
|            v                       v                      v           |
|  +-----------------------------------------------------------------+  |
|  |           Модуль предварительной подготовки (Preprocessing)     |  |
|  |       Токенизация, аугментация, приведение к тензорам PyTorch   |  |
|  +-----------------------------------------------------------------+  |
|                                    |                                  |
|                                    v                                  |
|  +-----------------------------------------------------------------+  |
|  |                Сквозная VLA-модель (Isaac GR00T)                |  |
|  |         Включает: Vision Backbone + Flow Matching Transformer   |  |
|  +-----------------------------------------------------------------+  |
|                                    |                                  |
|                                    v                                  |
|  +-----------------------------------------------------------------+  |
|  |                 Постпроцессинг (Action Postprocessing)          |  |
|  |        Де-нормализация, извлечение чанков, валидация дельт      |  |
|  +-----------------------------------------------------------------+  |
|                                    |                                  |
+------------------------------------|----------------------------------+
                                     v
                        [Вектор траекторий Delta EEF]
                                     |
                                     v
                   Локальный контроллер (WBC / IK Solver)
```


## 2. Подготовка и тензоризация входных данных (Data Preprocessing)

Прежде чем передать данные в граф вычислений PyTorch, разнородные входные модальности необходимо привести к строго определенным размерностям тензоров (Tensor Shapes), нормализовать числовые диапазоны и выровнять их по временной оси.

### A. Текстовые инструкции (Language Tokenization)

Высокоуровневая команда токенизируется стандартными средствами NLP. Длина последовательности (Sequence Length) фиксируется с помощью паддинга (padding) или усечения (truncation), чтобы гарантировать статическую форму графа во время инференса, что критично для последующей компиляции модели через TensorRT.

### B. Визуальные данные (Vision Processing)

Входные RGB-кадры с обзорной камеры (Head) и запястных камер (Wrist) масштабируются под размерность рецептивного поля визуального энкодера (обычно $224 \times 224$ или $336 \times 336$ пикселей). Значения пикселей переводятся в диапазон $[0.0, 1.0]$ и нормализуются средним значением и стандартным отклонением распределения ImageNet.

### C. Состояние робота (Proprioception Vector)

Вектор проприоцепции включает в себя кинематические параметры. Для стабильности градиентов углы суставов переводятся из абсолютных радианов в диапазон $[-1.0, 1.0]$ на основе жестких конструктивных лимитов сочленений конкретного робота.

### Реализация класса препроцессинга данных

```python
import torch
import torchvision.transforms as T
from transformers import AutoTokenizer
import numpy as np
from typing import Dict, Any, Tuple

class GR00TDataPreprocessor:
    """
    Класс для подготовки разнородных данных (текст, изображения, проприоцепция)
    к формату, требуемому для инференса Isaac GR00T.
    """
    def __init__(self, model_checkpoint: str, max_text_length: int = 64, img_size: int = 224):
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.max_text_length = max_text_length
        self.img_size = img_size
        
        # Стандартный пайплайн нормализации для Vision-Backbone (ImageNet приоритеты)
        self.vision_transform = T.Compose([
            T.ToDevice(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')),
            T.Resize((self.img_size, self.img_size)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_text(self, text_instruction: str) -> Dict[str, torch.Tensor]:
        """Токенизация текстовой команды пользователя."""
        tokens = self.tokenizer(
            text_instruction,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length
        )
        return {k: v.cuda() if torch.cuda.is_available() else v for k, v in tokens.items()}

    def preprocess_images(self, head_img: np.ndarray, wrist_img: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Преобразование сырых NumPy массивов изображений (H, W, C) в тензоры PyTorch (1, C, H, W).
        """
        # Перевод в тензоры формата (C, H, W) и масштабирование в [0.0, 1.0]
        head_tensor = torch.from_numpy(head_img).permute(2, 0, 1).float() / 255.0
        wrist_tensor = torch.from_numpy(wrist_img).permute(2, 0, 1).float() / 255.0
        
        # Применение нормализации и ресайза, добавление Batch-размерности (B=1)
        head_processed = self.vision_transform(head_tensor).unsqueeze(0)
        wrist_processed = self.vision_transform(wrist_tensor).unsqueeze(0)
        
        return head_processed, wrist_processed

    def preprocess_proprioception(self, joint_positions: np.ndarray, 
                                  joint_velocities: np.ndarray, 
                                  ee_pose: np.ndarray,
                                  joint_limits: Tuple[np.ndarray, np.ndarray]) -> torch.Tensor:
        """
        Сборка единого вектора проприоцепции и его мин-макс нормализация.
        """
        min_lim, max_lim = joint_limits
        # Нормализация позиций суставов в диапазон [-1, 1]
        norm_joints = 2.0 * (joint_positions - min_lim) / (max_lim - min_lim + 1e-5) - 1.0
        
        # Конкатенация всех признаков в один плоский вектор
        proprio_vector = np.concatenate([
            norm_joints,          # Состояние суставов
            joint_velocities,     # Динамика движения
            ee_pose               # Текущая позиция кисти (X, Y, Z, Qx, Qy, Qz, Qw)
        ], axis=0)
        
        # Превращение в тензор PyTorch с батч-размерностью
        proprio_tensor = torch.from_numpy(proprio_vector).float().unsqueeze(0)
        return proprio_tensor.cuda() if torch.cuda.is_available() else proprio_tensor
```

## 3. Инициализация и структура VLA-модели

Для демонстрации работы пайплайна на чистом Python опишем архитектурные компоненты Isaac GR00T в виде PyTorch-модулей. Архитектура состоит из статической Системы 2 (замороженный когнитивный эмбеддер) и Системы 1, вычисляющей векторные поля скоростей траекторий.

```python
import torch
import torch.nn as nn

class VisionLanguageBackbone(nn.Module):
    """
    Система 2: Когнитивный слой. Принимает текст и изображения с камер,
    генерирует общее мультимодальное латентное пространство (Tokens).
    """
    def __init__(self, embed_dim: int = 1024):
        super().__init__()
        self.embed_dim = embed_dim
        # Эмуляция глубоких слоев проекции (MLP Connectors)
        self.text_projection = nn.Linear(768, embed_dim)
        self.visual_projection = nn.Linear(512, embed_dim)
        
    def forward(self, text_tokens: Dict[str, torch.Tensor], 
                head_img: torch.Tensor, wrist_img: torch.Tensor) -> torch.Tensor:
        # В реальной модели здесь происходит проход через ViT и текстовый энкодер LLM.
        batch_size = head_img.shape[0]
        
        # Генерируем контекстный эмбеддинг сцены (имитация выхода Системы 2)
        device = head_img.device
        mock_latent_context = torch.randn(batch_size, 32, self.embed_dim, device=device)
        return mock_latent_context


class ProprioceptionEncoder(nn.Module):
    """Внутренний энкодер физического состояния робота."""
    def __init__(self, input_dim: int = 21, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.Linear(512, output_dim)
        )
        
    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        return self.net(proprio)


class FlowMatchingActionTransformer(nn.Module):
    """
    Система 1: Реактивный слой. Архитектура DiT (Diffusion Transformer),
    предсказывающая векторное поле скоростей траектории робота.
    """
    def __init__(self, action_dim: int = 7, horizon: int = 16, latent_dim: int = 1024):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        
        # Сеть прогнозирования вектора скорости v_theta(x_t, t)
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 256)
        )
        
        self.input_projection = nn.Linear(action_dim, 512)
        self.context_bridge = nn.Linear(latent_dim, 512)
        
        # Финальный слой генерации дельт поля скоростей
        self.transformer_blocks = nn.Sequential(
            nn.Linear(512 + 256 + 512, 512),
            nn.SiLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, 
                latent_context: torch.Tensor, proprio_embed: torch.Tensor) -> torch.Tensor:
        """
        Аргументы:
            x_t: Текущее зашумленное состояние траектории [B, Horizon, Action_Dim]
            t: Шаг времени интеграции (Flow Matching time) [B]
            latent_context: Токены Системы 2 [B, Seq_Len, Latent_Dim]
            proprio_embed: Геометрический эмбеддинг робота [B, Proprio_Out]
        Выход:
            v_t: Вектор скорости изменения траектории [B, Horizon, Action_Dim]
        """
        B, T, D = x_t.shape
        
        # Проекция входного шума траектории
        x_feat = self.input_projection(x_t) # [B, T, 512]
        
        # Проекция времени
        t_feat = self.time_embed(t.unsqueeze(-1)).unsqueeze(1).expand(-1, T, -1) # [B, T, 256]
        
        # Агрегация контекста Системы 2 (усреднение по токенам для упрощения)
        ctx_feat = self.context_bridge(latent_context.mean(dim=1)).unsqueeze(1).expand(-1, T, -1) # [B, T, 512]
        
        # Конкатенируем признаки вдоль размерности каналов
        feat_cat = torch.cat([x_feat, t_feat, ctx_feat], dim=-1)
        
        # Предсказание направления вектора скорости
        v_t = self.transformer_blocks(feat_cat)
        return v_t
```

## 4. Реализация цикла инференса (Inference Loop & Flow Matching Sampling)

Центральным элементом работы с GR00T на чистом Python является построение сэмплера траекторий. Так как модель обучается по методологии Flow Matching, генерация физических действий происходит путем интегрирования обыкновенного дифференциального уравнения (ODE) от состояния чистого Гауссова шума (t=0) до чистой упорядоченной траектории движений (t=1).

Ниже представлена математически выверенная реализация численного интегратора методом Эйлера для генерации пачки действий (Action Chunking):

```python
class GR00TPolicyInference:
    """
    Управляющий класс инференса, инкапсулирующий математику Flow Matching интеграции
    и генерацию временных окон действий (Action Chunking).
    """
    def __init__(self, backbone: VisionLanguageBackbone, 
                 proprio_encoder: ProprioceptionEncoder, 
                 action_transformer: FlowMatchingActionTransformer):
        super().__init__()
        self.backbone = backbone
        self.proprio_encoder = proprio_encoder
        self.action_transformer = action_transformer
        
        self.action_dim = action_transformer.action_dim
        self.horizon = action_transformer.horizon

    @torch.no_grad()
    def generate_action_chunk(self, text_tokens: Dict[str, torch.Tensor], 
                              head_img: torch.Tensor, wrist_img: torch.Tensor, 
                              proprio_tensor: torch.Tensor, 
                              num_ode_steps: int = 4) -> torch.Tensor:
        """
        Основной цикл генерации траекторий на базе численного сэмплирования Flow Matching.
        
        Параметры:
            num_ode_steps: Количество шагов численной интеграции (для FM достаточно 2-4 шага).
        """
        B = head_img.shape[0]
        device = head_img.device
        
        # Шаг 1: Извлечение тяжелого контекста Системы 2 (Выполняется 1 раз за такт)
        latent_context = self.backbone(text_tokens, head_img, wrist_img)
        
        # Шаг 2: Извлечение высокочастотного признака проприоцепции
        proprio_embed = self.proprio_encoder(proprio_tensor)
        
        # Шаг 3: Инициализация начальной точки интеграции из стандартного Гауссова распределения
        # Форма: [Batch, Horizon (16 шагов времени вперед), Action_Dim (7 параметров EEF)]
        x_t = torch.randn(B, self.horizon, self.action_dim, device=device)
        
        # Расчет шага интеграции
        dt = 1.0 / num_ode_steps
        
        # Шаг 4: Численное интегрирование по сетке t в диапазоне [0, 1]
        for step in range(num_ode_steps):
            # Текущая координата времени на траектории денойзинга
            t_val = step * dt
            t_tensor = torch.full((B,), t_val, device=device, dtype=torch.float)
            
            # Опрос модели для получения вектора скорости изменения траектории
            v_t = self.action_transformer(x_t, t_tensor, latent_context, proprio_embed)
            
            # Шаг Эйлера вперед по направлению прямолинейного потока
            x_t = x_t + v_t * dt
            
        # На выходе x_1 — полностью очищенная и сформированная траектория действий робота
        action_chunk = x_t
        return action_chunk
```

## 5. Десериализация и постпроцессинг команд (Action Chunking & EEF Deltas)

После завершения интеграции Flow Matching мы получаем тензор формы `[1, 16, 7]`. Модель предсказала скользящее окно (chunk) из 16 последовательных временных шагов на будущее. Чтобы отправить эти команды на исполнительные механизмы физического робота без симулятора, необходимо выполнить операцию постпроцессинга.
- Денормализация (De-normalization): Перевод абстрактных значений из диапазона модели обратно в физические величины: метры для смещений и радианы для углов вращения.
- Экстракция шага (Receding Horizon Selection): Из 16 предсказанных шагов берется подмножество M (в жестких контурах управления реального времени берется самый первый шаг 
M=0, а остальные отбрасываются, так как на следующем такте через 20 миллисекунд вся цепочка будет пересчитана заново на основе свежих кадров с камер).

```python
class ActionPostprocessor:
    """Десериализация выходов модели в физические единицы управления."""
    def __init__(self, max_translation_delta: float = 0.05, max_rotation_delta: float = 0.15):
        # Максимальные физические лимиты смещения за один командный такт
        self.max_trans = max_translation_delta # 5 сантиметров
        self.max_rot = max_rotation_delta     # ~8.5 градусов

    def process_predicted_chunk(self, action_chunk: torch.Tensor) -> Dict[str, Any]:
        """
        Преобразование выходного тензора в словарь управляющих дельт для робота.
        """
        # Извлекаем первый шаг из предсказанного окна (Receding Horizon Control)
        # Форма входного тензора: [1, Horizon, 7] -> Извлекаем [7]
        current_action = action_chunk[0, 0, :].cpu().numpy()
        
        # Разделение вектора на физические компоненты
        raw_translation = current_action[0:3]   # dx, dy, dz
        raw_rotation = current_action[3:6]      # d_roll, d_pitch, d_yaw
        raw_gripper = current_action[6]         # Gripper command
        
        # Денормализация (Линейное масштабирование признаков)
        # Предполагается, что выходы сети ограничены Tanh активацией в [-1, 1]
        phys_dx = raw_translation[0] * self.max_trans
        phys_dy = raw_translation[1] * self.max_trans
        phys_dz = raw_translation[2] * self.max_trans
        
        phys_droll = raw_rotation[0] * self.max_rot
        phys_dpitch = raw_rotation[1] * self.max_rot
        phys_dyaw = raw_rotation[2] * self.max_rot
        
        # Бинаризация или ограничение команды схвата
        phys_gripper = 1.0 if raw_gripper > 0.0 else 0.0
        
        return {
            "translation_deltas_m": np.array([phys_dx, phys_dy, phys_dz]),
            "rotation_deltas_rad": np.array([phys_droll, phys_dpitch, phys_dyaw]),
            "gripper_state": phys_gripper
        }
```

## 6. Полный самодостаточный скрипт сквозного инференса (End-to-End Mock Script)

Ниже представлен готовый к запуску исполняемый скрипт, объединяющий все вышеописанные компоненты в единый сквозной цикл инференса (Inference Pipeline). Скрипт генерирует синтетические данные (имитируя захват кадров с веб-камер ноутбука или робота и чтение шины данных) и рассчитывает итоговый вектор физических перемещений.

```python
import torch
import numpy as np
import time

def run_pure_python_groot_inference():
    print("=== Инициализация контура инференса Isaac GR00T на чистом Python ===")
    
    # 1. Задаем параметры конфигурации аппаратной платформы
    JOINT_COUNT = 7 # Пример: манипулятор Franka Emika Panda (7 степеней свободы)
    joint_min_limits = np.array([-2.89, -1.76, -2.89, -3.07, -2.89, -0.01, -2.89])
    joint_max_limits = np.array([ 2.89,  1.76,  2.89, -0.06,  2.89,  3.75,  2.89])
    joint_limits = (joint_min_limits, joint_max_limits)
    
    # Расчет размерности проприоцептивного вектора: 
    # 7 позиций суставов + 7 скоростей суставов + 7 координат позы EEF (X,Y,Z, Qw,Qx,Qy,Qz) = 21 признак
    PROPRIO_DIM = JOINT_COUNT * 2 + 7 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое вычислительное устройство: {device}")
    
    # 2. Инстанцирование компонентов архитектуры GR00T
    # Используем заглушку чекпоинта для токенайзера bert/distilbert
    preprocessor = GR00TDataPreprocessor(model_checkpoint="distilbert-base-uncased")
    
    backbone = VisionLanguageBackbone(embed_dim=1024).to(device).eval()
    proprio_encoder = ProprioceptionEncoder(input_dim=PROPRIO_DIM, output_dim=256).to(device).eval()
    action_transformer = FlowMatchingActionTransformer(action_dim=7, horizon=16, latent_dim=1024).to(device).eval()
    
    policy = GR00TPolicyInference(backbone, proprio_encoder, action_transformer)
    postprocessor = ActionPostprocessor()
    
    # 3. Имитация получения данных с датчиков робота в реальном времени (Mock Sensors)
    mock_head_camera_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    mock_wrist_camera_frame = np.random.randint(0, 255, (240, 240, 3), dtype=np.uint8)
    
    mock_current_joints = np.array([0.0, -0.4, 0.0, -1.5, 0.0, 1.5, 0.7])
    mock_current_velocities = np.array([0.05, -0.01, 0.02, 0.0, -0.01, 0.01, 0.0])
    mock_current_ee_pose = np.array([0.45, 0.12, 0.33, 1.0, 0.0, 0.0, 0.0]) # X, Y, Z, Q
    
    user_instruction = "Pick up the metallic part from the table and place it inside the sorting box"
    
    print(f"\nВходная текстовая команда: '{user_instruction}'")
    print("Запуск препроцессинга данных...")
    
    # 4. Прогон через препроцессинг
    text_features = preprocessor.preprocess_text(user_instruction)
    head_tensor, wrist_tensor = preprocessor.preprocess_images(mock_head_camera_frame, mock_wrist_camera_frame)
    proprio_tensor = preprocessor.preprocess_proprioception(
        mock_current_joints, mock_current_velocities, mock_current_ee_pose, joint_limits
    )
    
    # Вывод размерностей для проверки корректности тензоризации
    print(f" -> Размерность тензора обзорной камеры: {head_tensor.shape}")
    print(f" -> Размерность тензора запястной камеры: {wrist_tensor.shape}")
    print(f" -> Размерность тензора проприоцепции:  {proprio_tensor.shape}")
    
    # 5. Инференс VLA-модели (Вычисление ODE траекторий Flow Matching)
    print("\nЗапуск итеративного Flow Matching сэмплирования траекторий...")
    start_time = time.perf_counter()
    
    predicted_chunk = policy.generate_action_chunk(
        text_tokens=text_features,
        head_img=head_tensor,
        wrist_img=wrist_tensor,
        proprio_tensor=proprio_tensor,
        num_ode_steps=4 # Оптимальное число шагов для Flow Matching в GR00T
    )
    
    inference_time = (time.perf_counter() - start_time) * 1000
    print(f"Инференс завершен успешно! Время выполнения: {inference_time:.2f} мс")
    print(f"Размерность выходного чанка действий (Actions Chunk Shape): {predicted_chunk.shape}")
    
    # 6. Постпроцессинг и десериализация команд в физический интерфейс робота
    control_commands = postprocessor.process_predicted_chunk(predicted_chunk)
    
    print("\n=== РЕЗУЛЬТАТ ДЕСЕРИАЛИЗАЦИИ ДЛЯ НИЗКОУРОВНЕВОГО КОНТРОЛЛЕРА ===")
    print(f"Линейное смещение инструмента (dx, dy, dz) метры:\n   {control_commands['translation_deltas_m']}")
    print(f"Угловое смещение инструмента (d_roll, d_pitch, d_yaw) радианы:\n   {control_commands['rotation_deltas_rad']}")
    print(f"Команда для актуатора схвата (0.0 - открыть, 1.0 - закрыть): {control_commands['gripper_state']}")
    print("================================================================")
    print("Пайплайн готов к интеграции с локальным робототехническим софтом.")

if __name__ == "__main__":
    run_pure_python_groot_inference()
```