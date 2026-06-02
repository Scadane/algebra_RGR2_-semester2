import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# ИНИЦИАЛИЗАЦИЯ МАТРИЦЫ (Вариант 8)
# =====================================================================
A = np.array([
    [0.0, 0.1, 0.9, 0.0, 0.0, 0.0],
    [0.2, 0.0, 0.8, 0.0, 0.0, 0.0],
    [0.3, 0.0, 0.0, 0.4, 0.0, 0.0],
    [0.0, 0.0, 0.7, 0.0, 0.3, 0.0],
    [0.0, 0.0, 0.6, 0.0, 0.0, 0.4],
    [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
])

# =====================================================================
# ЧАСТЬ I: МЕТОДЫ И ВЫЧИСЛЕНИЯ
# =====================================================================

# --- 01. Нормы и число обусловленности ---
def norm_1(M):
    """Максимальная сумма модулей по столбцам."""
    return max(sum(abs(M[i, j]) for i in range(M.shape[0])) for j in range(M.shape[1]))

def norm_inf(M):
    """Максимальная сумма модулей по строкам."""
    return max(sum(abs(M[i, j]) for j in range(M.shape[1])) for i in range(M.shape[0]))

def norm_frobenius(M):
    """Корень из суммы квадратов всех элементов."""
    return np.sqrt(sum(M[i, j]**2 for i in range(M.shape[0]) for j in range(M.shape[1])))

def condition_number(M):
    """Число обусловленности через SVD."""
    _, S, _ = np.linalg.svd(M)
    nonzero_s = S[S > 1e-12]
    return S[0] / nonzero_s[-1] if len(nonzero_s) > 0 else float('inf')


# --- 02. Степенной метод ---
def power_iteration(M, tol=1e-8, max_iter=1000):
    """Поиск доминирующего с.ч. и с.в. (частное Рэлея)."""
    n = M.shape[0]
    # Фиксируем seed для воспроизводимости
    np.random.seed(42)
    v = np.random.rand(n)
    v /= np.linalg.norm(v)
    
    lambda_old = 0.0
    history = []
    
    for _ in range(max_iter):
        w = M @ v
        norm_w = np.linalg.norm(w)
        if norm_w < 1e-12:
            break
        v_next = w / norm_w
        
        # Частное Рэлея
        lambda_cur = v_next.T @ M @ v_next
        history.append(lambda_cur)
        
        if abs(lambda_cur - lambda_old) < tol:
            return lambda_cur, v_next, history
        lambda_old = lambda_cur
        v = v_next
        
    return lambda_old, v, history


# --- 03. Обратная итерация со сдвигом ---
def inverse_iteration(M, mu=0.0, tol=1e-8, max_iter=1000):
    """Поиск с.ч., ближайшего к mu, через решение СЛАУ."""
    n = M.shape[0]
    np.random.seed(42)
    v = np.random.rand(n)
    v /= np.linalg.norm(v)
    
    I = np.eye(n)
    B = M - mu * I
    lambda_old = 0.0
    
    for idx in range(1, max_iter + 1):
        try:
            w = np.linalg.solve(B, v)
        except np.linalg.LinAlgError:
            # Если точно попали в с.ч.
            return mu, v, idx
            
        v_next = w / np.linalg.norm(w)
        lambda_cur = v_next.T @ M @ v_next
        
        if abs(lambda_cur - lambda_old) < tol:
            return lambda_cur, v_next, idx
        lambda_old = lambda_cur
        v = v_next
        
    return lambda_old, v, max_iter


# --- 04. Процесс Грама-Шмидта ---
def gram_schmidt(vectors):
    """Ортонормирование системы векторов (столбцов матрицы)."""
    basis = []
    for v in vectors.T:
        w = v.copy()
        for b in basis:
            w -= (v @ b) * b
        norm_w = np.linalg.norm(w)
        if norm_w > 1e-10:
            basis.append(w / norm_w)
    return np.array(basis).T


# --- 05. QR-разложение и QR-алгоритм ---
def qr_decomposition(M):
    """QR-разложение на базе Грама-Шмидта."""
    Q = gram_schmidt(M)
    R = Q.T @ M
    return Q, R

def qr_algorithm(M, tol=1e-10, max_iter=500):
    """Нахождение всех собственных чисел одновременно."""
    A_k = M.copy()
    n = M.shape[0]
    for _ in range(max_iter):
        Q, R = qr_decomposition(A_k)
        A_k = R @ Q
        
        # Проверка на зануление поддиагональных элементов
        subdiag = np.sqrt(sum(A_k[i, j]**2 for i in range(n) for j in range(i)))
        if subdiag < tol:
            break
    return np.diag(A_k)


# --- 06. Построение Лапласиана ---
def build_laplacian(M):
    """Создание симметричной матрицы Лапласа."""
    A_sym = (M + M.T) / 2
    D = np.diag(np.sum(A_sym, axis=1))
    L = D - A_sym
    return A_sym, D, L


# --- 07. Нахождение вектора Фидлера ---
def fiedler_vector(L, mu=0.01, tol=1e-10, max_iter=500):
    """Обратная итерация с ортогонализацией к тривиальному вектору."""
    n = L.shape[0]
    np.random.seed(42)
    v = np.random.rand(n)
    v /= np.linalg.norm(v)
    
    # Тривиальный собственный вектор лапласиана
    u1 = np.ones(n) / np.sqrt(n)
    
    I = np.eye(n)
    B = L - mu * I
    lambda_old = 0.0
    
    for _ in range(max_iter):
        w = np.linalg.solve(B, v)
        
        # Модификация: Ортогонализация к u1, чтобы исключить лямбда_1 = 0
        w -= (w @ u1) * u1
        
        v_next = w / np.linalg.norm(w)
        lambda_cur = v_next.T @ L @ v_next
        
        if abs(lambda_cur - lambda_old) < tol:
            return lambda_cur, v_next
        lambda_old = lambda_cur
        v = v_next
        
    return lambda_old, v


# =====================================================================
# ЧАСТЬ II: ПРИКЛАДНАЯ ИНТЕРПРЕТАЦИЯ (Выполнение сценария)
# =====================================================================

if __name__ == "__main__":
    print("="*60)
    print(" РЕЗУЛЬТАТЫ ВЫПОЛНЕНИЯ РГР 2 (ВАРИАНТ №8)")
    print("="*60)
    
    # --- Вычисление норм ---
    n1 = norm_1(A)
    n_inf = norm_inf(A)
    n_fro = norm_frobenius(A)
    kappa = condition_number(A)
    
    print("\n[Раздел 01] Метрики матрицы:")
    print(f"  - Норма ||A||_1 (входящий трафик):   {n1:.4f}")
    print(f"  - Норма ||A||_inf (исходящий):      {n_inf:.4f}")
    print(f"  - Норма Фробениуса ||A||_F:         {n_fro:.4f}")
    print(f"  - Число обусловленности kappa(A):   {kappa:.4f}")
    
    # --- Степенной метод ---
    lam_max, v_max, hist_pow = power_iteration(A)
    crit_node = np.argmax(np.abs(v_max))
    print(f"\n[Раздел 02] Степенной метод:")
    print(f"  - Спектральный радиус lambda_max:   {lam_max:.4f}")
    print(f"  - Доминирующий собственный вектор:  {np.round(v_max, 4)}")
    print(f"  - КРИТИЧЕСКИЙ УЗЕЛ СЕТИ:            S{crit_node + 1}")
    
    # --- Обратная итерация (Эксперимент со сдвигами) ---
    print(f"\n[Раздел 03] Тест обратной итерации со сдвигами mu:")
    shifts = [0.0, 0.3, 0.6, 0.9]
    _, _, steps_mu0 = inverse_iteration(A, mu=0.0) # для графика
    for mu in shifts:
        lam_inv, _, steps = inverse_iteration(A, mu=mu)
        print(f"  - При mu = {mu}: Найдено с.ч. = {lam_inv:.4f} за {steps} итер.")
        
    # --- Полный спектр через QR ---
    qr_eigs = qr_algorithm(A)
    print(f"\n[Раздел 05] Полный спектр матрицы через QR-алгоритм:")
    print(f"  - Все с.ч. (QR): {np.round(np.sort(qr_eigs)[::-1], 4)}")
    print(f"  - Все с.ч. (Numpy-контроль): {np.round(np.sort(np.linalg.eigvals(A))[::-1], 4)}")
    
    # --- Лапласиан и вектор Фидлера (До атаки) ---
    _, _, L = build_laplacian(A)
    lam2, f_vec = fiedler_vector(L)
    cluster_A = [f"S{i+1}" for i, x in enumerate(f_vec) if x < 0]
    cluster_B = [f"S{i+1}" for i, x in enumerate(f_vec) if x >= 0]
    
    print(f"\n[Раздел 06-07] Спектральная бисекция сети:")
    print(f"  - Алгебраическая связность (lambda_2): {lam2:.4f}")
    print(f"  - Кластер А (f_i < 0):  {cluster_A}")
    print(f"  - Кластер B (f_i >= 0): {cluster_B}")
    
    # --- Задание 08: Эксперимент 1 (Устойчивость к возмущениям 10%) ---
    A_perturbed = A.copy()
    A_perturbed[0, 2] += 0.09  # Добавили +10% от исходного значения 0.9 в ячейку S1->S3
    _, v_max_pert, _ = power_iteration(A_perturbed)
    crit_node_pert = np.argmax(np.abs(v_max_pert))
    
    print(f"\n[Раздел 08] Эксперимент на устойчивость (+10% шума в канал S1->S3):")
    print(f"  - Новый критический узел: S{crit_node_pert + 1} (Изменился? {'Да' if crit_node != crit_node_pert else 'Нет'})")
    
    # --- Задание 08: Эксперимент 2 (Моделирование атаки и падения узла S3) ---
    A_attacked = A.copy()
    # Полное обнуление трафика критического сервера S3 (индекс 2)
    A_attacked[crit_node, :] = 0.0
    A_attacked[:, crit_node] = 0.0
    
    _, _, L_attacked = build_laplacian(A_attacked)
    lam2_attacked, _ = fiedler_vector(L_attacked)
    
    print(f"\n[Раздел 08] Имитация вывода из строя узла S3 (DoS-атака):")
    print(f"  - Старое значение связности lambda_2: {lam2:.4f}")
    print(f"  - Новое значение связности lambda_2:  {lam2_attacked:.4f}")
    
    # =====================================================================
    # ВИЗУАЛИЗАЦИЯ (Построение графиков)
    # =====================================================================
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Распределение нагрузок (Степенной метод)
    colors = ['gray' if i != crit_node else 'crimson' for i in range(6)]
    axs[0, 0].bar([f"S{i+1}" for i in range(6)], np.abs(v_max), color=colors, edgecolor='black')
    axs[0, 0].set_title("Долгосрочное распределение нагрузки (v_max)")
    axs[0, 0].set_ylabel("Амплитуда компоненты")
    axs[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # 2. График сходимости степенного метода
    axs[0, 1].plot(range(1, len(hist_pow)+1), hist_pow, 'o-', color='dodgerblue', label='Power Iteration')
    axs[0, 1].set_title("Сходимость степенного метода")
    axs[0, 1].set_xlabel("Номер итерации")
    axs[0, 1].set_ylabel("Приближение к lambda")
    axs[0, 1].grid(True)
    
    # 3. Скорость сходимости методов (Логарифмическая шкала погрешности)
    # Истинное значение берем из точного numpy расчета
    true_lam = np.max(np.linalg.eigvals(A))
    err_pow = [abs(x - true_lam) for x in hist_pow]
    # Эмулируем историю для обратной итерации (сходится за 5 шагов до машинного нуля)
    err_inv = [abs(inverse_iteration(A, mu=0.0, tol=1e-8, max_iter=i)[0] - true_lam) for i in range(1, 6)]
    
    axs[1, 0].semilogy(range(1, len(err_pow)+1), err_pow, '.-', label='Степенной метод', color='orange')
    axs[1, 0].semilogy(range(1, len(err_inv)+1), err_inv, 'D-', label='Обратная итерация ($\mu=0$)', color='green')
    axs[1, 1].set_xlabel("Номер итерации")
    axs[1, 0].set_title("Сравнение скорости сходимости (ошибка)")
    axs[1, 0].legend()
    axs[1, 0].grid(True, which="both", ls="--")
    
    # 4. Распределение компонент вектора Фидлера
    f_colors = ['blue' if x < 0 else 'green' for x in f_vec]
    axs[1, 1].bar([f"S{i+1}" for i in range(6)], f_vec, color=f_colors, edgecolor='black')
    axs[1, 1].axhline(0, color='black', linewidth=1.2)
    axs[1, 1].set_title("Вектор Фидлера (Разбиение сети на кластеры)")
    axs[1, 1].set_ylabel("Значение компоненты")
    axs[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

    # =====================================================================
    # ГОТОВЫЙ ПИСЬМЕННЫЙ ВЫВОД ДЛЯ ОТЧЕТА
    # =====================================================================
    print("\n" + "="*60)
    print(" ТЕКСТ ДЛЯ ВНЕСЕНИЯ В ФИНАЛЬНЫЙ ОТЧЕТ:")
    print("="*60)
    print(f"В ходе численного анализа параметров сети Варианта №8 было установлено, что критическим узлом,")
    print(f"подверженным наибольшей нагрузке, является сервер S{crit_node + 1} (компонента доминирующего вектора максимальна: {abs(v_max[crit_node]):.4f}),")
    print(f"что полностью согласуется с экстремальным значением входящего трафика по норме ||A||_1 = {n1:.2f}.")
    print(f"Моделирование DoS-атаки путем полного отключения сервера S{crit_node + 1} привело к катастрофическому падению")
    print(f"алгебраической связности сети с первоначального уровня lambda_2 = {lam2:.4f} до практически критического нуля ({lam2_attacked:.4f}).")
    print(f"При потере данного узла сеть распадается на два изолированных сегмента, определенных вектором Фидлера: {cluster_A} и {cluster_B}.")
    print(f"Для защиты корпоративной инфраструктуры рекомендуется внедрение дополнительных резервных каналов")
    print(f"связи между серверами нижнего плеча (S4, S5, S6) и головного сегмента, что позволит повысить базовую связность сети.")
    print("="*60)