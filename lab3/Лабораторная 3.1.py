import torch
import random


x = torch.rand(5, 3, requires_grad=True, dtype=torch.float32)
print("Исходный тензор:\n", x)

cubed_tensor = x ** 3
print("\nТензор после возведения в куб:\n", cubed_tensor)


random_multiplier = random.randint(1, 10)
print(f"\nСлучайное число: {random_multiplier}")


scaled_tensor = cubed_tensor * random_multiplier
print("\nТензор после умножения на случайное число:\n", scaled_tensor)


exp_tensor = torch.exp(scaled_tensor)
print("\nЭкспонента от тензора:\n", exp_tensor)


exp_tensor.backward(gradient=torch.ones_like(exp_tensor))
gradient_tensor = x.grad
print("\nПроизводная:\n", gradient_tensor)



