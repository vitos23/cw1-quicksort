# Параллельная быстрая сортировка

## Команды

Запуск тестов: `cargo test`.

Запуск бенчмарка: `cargo run --release`.

## Результаты

В тестировании принимали участие 4 реализации быстрой сортировки:

1. **Последовательная**
2. **Параллельная (свои примитивы)** - имеет полилогарифмический span, используются собственные параллельные примитивы,
   написанные целиком в fork-join модели
3. **Параллельная (примитивы rayon)** - параллельная реализация, использующая `filter` из библиотеки rayon
4. **Параллельная (простая)** - простая параллельная реализация быстрой сортировки в fork-join модели:
   отличается от последовательной тем, что массивы с меньшими и большими элементами сортируются через fork/join.

| Процессор              | Последовательная | Параллельная (свои примитивы) | Параллельная (примитивы rayon) | Параллельная (простая) |
|------------------------|------------------|-------------------------------|--------------------------------|------------------------|
| AMD Ryzen 7 2700X 4GHz | 9418 мс / **1x** | 11264 мс / **0.84x**          | 6313 мс / **1.49x**            | 2703 мс / **3.48x**    |
| AMD Ryzen 5 5500U      | 9330 мс / **1x** | 13702 мс / **0.68x**          | 8332 мс / **1.12x**            | 2945 мс / **3.17x**    |
| Apple M3 Max           | 6921 мс / **1x** | 4620 мс / **1.5x**            | 3536 мс / **1.96x**            | 2103 мс / **3.29x**    |