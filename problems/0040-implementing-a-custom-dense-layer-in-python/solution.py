
import numpy as np
import copy
import math

# DO NOT CHANGE SEED
np.random.seed(42)

# DO NOT CHANGE LAYER CLASS
class Layer(object):

	def set_input_shape(self, shape):
		self.input_shape = shape

	def layer_name(self):
		return self.__class__.__name__

	def parameters(self):
		return 0

	def forward_pass(self, X, training):
		raise NotImplementedError()

	def backward_pass(self, accum_grad):
		raise NotImplementedError()

	def output_shape(self):
		raise NotImplementedError()

# Your task is to implement the Dense class based on the above structure
class Dense(Layer):
    def __init__(self, n_units, input_shape=None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.W = None
        self.w0 = None
        self.W_opt = None  # оптимизатор для весов
        self.w0_opt = None  # оптимизатор для смещений

    def initialize(self, optimizer):
        # Проверяем, что input_shape задан
        if self.input_shape is None:
            raise ValueError("input_shape must be set before initialization")
        
        # Количество входных признаков
        n_features = self.input_shape[0]
        
        # Инициализация весов из равномерного распределения
        limit = 1 / np.sqrt(n_features)
        self.W = np.random.uniform(-limit, limit, (n_features, self.n_units))
        
        # Инициализация смещений нулями
        self.w0 = np.zeros((1, self.n_units))
        
        # Создаем копии оптимизаторов для весов и смещений
        self.W_opt = copy.deepcopy(optimizer)
        self.w0_opt = copy.deepcopy(optimizer)

    def parameters(self):
        # Общее количество параметров = веса + смещения
        if self.W is None:
            return 0
        return self.W.size + self.w0.size

    def forward_pass(self, X, training=True):
        # Сохраняем вход для обратного прохода
        self.layer_input = X
        
        # Вычисляем выход: X @ W + w0
        # X имеет форму (batch_size, n_features)
        # W имеет форму (n_features, n_units)
        # w0 имеет форму (1, n_units)
        return np.dot(X, self.W) + self.w0

    def backward_pass(self, accum_grad):
        # accum_grad - градиент потерь по выходу слоя
        # Форма: (batch_size, n_units)
        
        # Градиент по входу: dL/dX = accum_grad @ W.T
        # Форма: (batch_size, n_features)
        grad_input = np.dot(accum_grad, self.W.T)
        
        # Если слой обучаемый, обновляем параметры
        if self.trainable:
            # Градиент по весам: dL/dW = X.T @ accum_grad
            # Форма: (n_features, n_units)
            grad_W = np.dot(self.layer_input.T, accum_grad)
            
            # Градиент по смещениям: сумма по батчу
            # Форма: (1, n_units)
            grad_w0 = np.sum(accum_grad, axis=0, keepdims=True)
            
            # Обновляем параметры с помощью оптимизаторов
            self.W = self.W_opt.update(self.W, grad_W)
            self.w0 = self.w0_opt.update(self.w0, grad_w0)
        
        return grad_input

    def output_shape(self):
        # Возвращаем форму выхода: (n_units,)
        return (self.n_units,)
