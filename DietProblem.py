import numpy as np
import scipy.optimize as opt
import pandas as pd

""" Лабораторная работа №1 по Исследованию операций """
""" Тема: Задача о рационе. М-метод """

"""Больше об условиях задачи, а также о начальных вычислениях, где показано, как я решила 
реализовать ограничения с отклонениями и как должны выглядеть симплекс-таблицы, можно 
посмотреть на приложенной фотографии (симплекс-таблицы в процессе написания кода немного 
изменили свой вид, но суть осталась та же) """

""" Для запуска программы нет каких-то специфических инструкций, просто прогоняем код 
и следуем инструкции в консоли. Можно при желании ввести необходимую норму БЖУ самому, 
но можно воспользоваться и данными по умолчанию. Нужно ввести допустимое отклонение в процентах.
Можно поменять размер m в main(), попробовать запустить с одним и тем же np.random.seed(), 
но разными ограмичениями и дельта, чтобы получить те же блюда, но с другим результатом  """


def generate_meals(n=50):
    # np.random.seed(13)
    meals = []
    for i in range(n):
        cost = np.random.uniform(10, 100)  # стоимость 100 г
        protein = np.random.uniform(5, 30)  # белки на 100 г
        fat = np.random.uniform(5, 40)  # жиры на 100 г
        carbs = np.random.uniform(5, 60)  # углеводы на 100 г
        meals.append((cost, protein, fat, carbs))
    return meals


def get_goal():
    user = input("Желаете ввести суточную потребность в питательных веществах вручную? (y/any): ")
    if user.lower() == 'y':
        A = float(input("Введите количество белков (г): "))
        B = float(input("Введите количество жиров (г): "))
        G = float(input("Введите количество углеводов (г): "))
    else:
        A, B, G = 90, 60, 250  # дефолт

    deviation = float(input("Введите допустимое отклонение в % (положительное): ")) / 100
    return A, B, G, deviation


def matrix_for_m_method(meals, A, B, G, delta):
    n = len(meals)
    c = np.array([-meal[0] for meal in meals])  # по условию задачи стоимость минимизируется
    A_eq = np.array([[meal[1] for meal in meals],
                     [meal[1] for meal in meals],
                     [meal[2] for meal in meals],
                     [meal[2] for meal in meals],
                     [meal[3] for meal in meals],
                     [meal[3] for meal in meals]])
    b_eq = np.array([A + A * delta, A - A * delta,
                     B + B * delta, B - B * delta,
                     G + G * delta, G - G * delta])
    return c, A_eq, b_eq

def matrix_for_linprog(meals, A, B, G):
    n = len(meals)
    c = np.array([-meal[0] for meal in meals])  # по условию задачи стоимость минимизируется
    A_eq = np.array([[meal[1] for meal in meals],
                     [meal[2] for meal in meals],
                     [meal[3] for meal in meals]])
    b_eq = np.array([A, B, G])
    return c, A_eq, b_eq

def m_method(c_eq, A_eq, b, M):
    m, n = A_eq.shape   # n - кол-во столбцов = 50
    #  m - кол-во строк = 6

    # # R + s1 + s3 + s5 для достаточного базиса
    # [1, 0, 0, 0, 0, 0],                # s1 = R4
    # [0, 1, 0, 0, 0, 0],                # R1
    # [0, 0, 1, 0, 0, 0],                # s3 = R5
    # [0, 0, 0, 1, 0, 0],                # R2
    # [0, 0, 0, 0, 1, 0],                # s5 = R6
    # [0, 0, 0, 0, 0, 1]                 # R3

    # избыточные s
    A_s = np.array([[0, 0, 0],
                   [-1, 0, 0],                # s2
                   [0, 0, 0],
                   [0, -1, 0],                # s4
                   [0, 0, 0],
                   [0, 0, -1]], dtype=float)  # s6
    A = np.hstack((A_eq, np.eye(m), A_s))
    c_s = np.hstack((c_eq, np.full(m, -M)))
    c = np.hstack((c_s, np.full(3, 0)))

    simplex_table = np.hstack((A, b.reshape(-1, 1)))    # horizontally, b_T
    simplex_table = np.vstack((np.hstack((c, [0])), simplex_table))
    # сверху — коэффициенты целевой функции
    # gоследний столбец - b

    # искусственные переменные R входят в начальный базис
    basis = list(range(n, n + m))

    # print(f"\nЦелевая:\n", simplex_table[0])

    # к канонической форме
    # rows_R = [2, 4, 6]  # строки с R
    for i in range(1, m+1):
        simplex_table[0] += M * simplex_table[i]

    df = pd.DataFrame(simplex_table)
    print("\nСимплекс-таблица:\n", df)
    # print(f"\nЦелевая:\n", simplex_table[0])

    it = 0

    while True:
        # на оптимальность
        z_raw = simplex_table[0, :-1]
        input_idx = np.argmax(z_raw) if np.any(z_raw > 0) else -1

        # выход
        if input_idx == -1:
            print("\nЧисло итераций: ", it)
            break

        # столбец
        leader_col = simplex_table[1:, input_idx]
        solve_col = simplex_table[1:, -1]

        # если ведущий элемент будет отрицательным,
        # это приведет к неразрешимому значению
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.where(leader_col > 0, solve_col / leader_col, np.inf)

        if np.all(ratios == np.inf):
            raise ValueError("\nНет допустимого решения\n")

        leader_row = np.argmin(ratios) + 1  # +1 - целевая строка

        # новая ведущая строка:
        simplex_table[leader_row] /= simplex_table[leader_row, input_idx]

        # другие:
        for i in range(len(simplex_table)):
            if i != leader_row:
                simplex_table[i] -= simplex_table[i, input_idx] * simplex_table[leader_row]

        # базис
        basis[leader_row - 1] = input_idx

        # вывод информации
        # print(f'Включаем переменную: x{input_idx + 1}\t исключаем переменную: x{raw_names[leader_row]}')
        # print(pd.DataFrame(simplex_table))

        it += 1

    # solution = np.zeros(n + m)
    solution = np.zeros(len(simplex_table[0]) - 1)
    solution[basis] = simplex_table[1:, -1]
    return solution[:n]


def print_results(meals, result):
    result /= 100
    total_cost = sum(result[i] * meals[i][0] for i in range(len(meals)))
    total_protein = sum(result[i] * meals[i][1] for i in range(len(meals)))
    total_fat = sum(result[i] * meals[i][2] for i in range(len(meals)))
    total_carbs = sum(result[i] * meals[i][3] for i in range(len(meals)))
    print(f"\nСтоимость: {total_cost:.2f} руб.")
    print(f"Белки: {total_protein:.2f} г")
    print(f"Жиры: {total_fat:.2f} г")
    print(f"Углеводы: {total_carbs:.2f} г")


def main():
    meals = generate_meals()
    A, B, G, delta = get_goal()

    c, A_eq, b_eq = matrix_for_m_method(meals, A, B, G, delta)
    m = 10 ** 5

    if delta == 0:
        # result_m = m_no_delta(A, B, G, m) * 100
        pass
    else:
        result_m = m_method(c, A_eq, b_eq, m) * 100

    c_prog, A_prog, b_prog = matrix_for_linprog(meals, A, B, G)
    result_linprog = opt.linprog(c_prog, A_eq=A_prog, b_eq=b_prog, method='highs')
    # result_linprog = opt.linprog(c, A_eq=A_eq, b_eq=b_eq, method='highs')
    if result_linprog.success:
        result_prog = result_linprog.x * 100
    else:
        print("\nlinprog не нашел решения. Причина:", result_linprog.message)
        result_prog = None

    print("\nРезультат M-метода:\n", result_m)
    print_results(meals, result_m)
    if result_prog is not None:
        print("\nРезультат linprog:\n", result_prog)
        print_results(meals, result_prog)
    else:
        print("linprog не нашел решения")

    # сравнение
    differences = result_m - result_prog
    print("\nРазличия между методами:\n", differences)


if __name__ == "__main__":
    main()
