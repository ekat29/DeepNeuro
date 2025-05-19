import random

numbers = [random.randint(1, 100) for _ in range(20)]


sum_even = 0
for num in numbers:
    if num % 2 == 0:
        sum_even += num

# 3. Выводим результат
print("Список чисел:", numbers)
print("Сумма чётных чисел:", sum_even)

