# MapReduce
MapReduce processes large datasets in parallel across a cluster. The map stage creates key-value pairs. The shuffle groups values by key, and reduce aggregates each group.

## Advantages
MapReduce offers scalability and fault tolerance. Failed tasks can be restarted on another worker. Data locality reduces network transfers by processing data near its storage location.

## Limitations
MapReduce writes intermediate results to disk. This makes iterative algorithms and interactive queries slower than memory-based engines.

## بالعربية
يعالج MapReduce مجموعات البيانات الكبيرة بالتوازي. تشمل مزاياه قابلية التوسع وتحمل الأعطال. تجمع مرحلة الخريطة أزواج المفاتيح والقيم، ثم تجمع مرحلة الاختزال النتائج حسب المفتاح.
