import numpy as np
import pulp as pl

""" Лабораторная работа №2 по Исследованию операций """
""" Тема: Задача о рационе. Метод ветвей и границ """

""" Берем предыдущую работу, немного ее докручиваем, чтобы она была больше похожа на реальную жизненную программу: 

    1. Ограничение delta задается в %, одинаково для всех и задается пользователем также, как задается БЖУ. 
Например, "максимальное отклонение - 10% в обе стороны". 
    2. Добавляем ограничения, что каждое из предлагаемых блюд должно встречаться в результате не более x раз. 
Тестировать при x=1 или x=2. Смысл простой - никто не захочет в день есть 5 цезарей или 7 томатных супов. 
    3. Задачу нужно решить целочисленно с помощью метода ветвей и границ (он проще в реализации, больше 
используется на практике и легче интерпретируется). 
"""

""" Для запуска программы нет каких-то специфических инструкций, просто прогоняем код 
и следуем инструкции в консоли """


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
    max_repeats = int(input("Максимальное количество порций одного блюда: "))
    return A, B, G, deviation, max_repeats


def solve_diet_problem(meals, A, B, G, delta, max_repeats):
    n = len(meals)
    problem = pl.LpProblem("DietProblem", pl.LpMinimize)
    x = pl.LpVariable.dicts("x", range(n), lowBound=0, upBound=max_repeats, cat='Integer')

    # целевая функция: минимизация стоимости
    problem += pl.lpSum(meals[i][0] * x[i] for i in range(n))

    # ограничения по БЖУ с допуском delta
    problem += pl.lpSum(meals[i][1] * x[i] for i in range(n)) >= A * (1 - delta)
    problem += pl.lpSum(meals[i][1] * x[i] for i in range(n)) <= A * (1 + delta)

    problem += pl.lpSum(meals[i][2] * x[i] for i in range(n)) >= B * (1 - delta)
    problem += pl.lpSum(meals[i][2] * x[i] for i in range(n)) <= B * (1 + delta)

    problem += pl.lpSum(meals[i][3] * x[i] for i in range(n)) >= G * (1 - delta)
    problem += pl.lpSum(meals[i][3] * x[i] for i in range(n)) <= G * (1 + delta)

    # problem.solve(pl.PULP_CBC_CMD(msg=0))
    problem.solve(pl.COIN_CMD(msg=0))

    if pl.LpStatus[problem.status] == 'Optimal':
        result = {i: x[i].value() for i in range(n) if x[i].value() > 0}
        return result
    else:
        raise ValueError("Оптимальное решение не найдено")


def branch_and_bound(meals, A, B, G, delta, max_repeats):
    # решение ослабленной задачи (релаксация)
    solution = solve_diet_problem(meals, A, B, G, delta, max_repeats)

    # рекурсивное ветвление
    for i, value in solution.items():
        if not value.is_integer():
            # ветвь вниз (x[i] <= floor(value))
            meals_copy = meals.copy()
            meals_copy[i] = (meals[i][0], 0, 0, 0)  # зануляем это блюдо, чтобы исключить его
            result1 = branch_and_bound(meals_copy, A, B, G, delta, max_repeats)

            # ветвь вверх (x[i] >= ceil(value))
            meals_copy2 = meals.copy()
            meals_copy2[i] = (meals[i][0], 0, 0, 0)  # зануляем это блюдо
            result2 = branch_and_bound(meals_copy2, A, B, G, delta, max_repeats)

            return min(result1, result2, key=lambda res: sum(meals[j][0] * res.get(j, 0) for j in range(len(meals))))

    return solution


def main():
    meals = generate_meals()
    A, B, G, delta, max_repeats = get_goal()

    best_solution = branch_and_bound(meals, A, B, G, delta, max_repeats)
    total_cost = sum(meals[i][0] * count for i, count in best_solution.items())

    total_protein = sum(meals[i][1] * count for i, count in best_solution.items())
    total_fat = sum(meals[i][2] * count for i, count in best_solution.items())
    total_carbs = sum(meals[i][3] * count for i, count in best_solution.items())

    print("\nРешение с помощью метода ветвей и границ:\n")

    # print("Оптимальный рацион:", best_solution)
    formatted_solution = {i: f"{int(count * 100)} г" for i, count in best_solution.items()}

    print("Оптимальный рацион:", formatted_solution)
    print(f"Общая стоимость: {round(total_cost, 2)} р.")
    print(f"Белки: {round(total_protein, 2)} г, Жиры: {round(total_fat, 2)} г, Углеводы: {round(total_carbs, 2)} г")


if __name__ == "__main__":
    main()
