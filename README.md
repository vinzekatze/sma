# sma

> ⚠️ **Дисклеймер.** Это исследовательский проект, а не финансовый продукт. Прогнозы не
> являются индивидуальной инвестиционной рекомендацией и не гарантируют результата.
> Данные MOEX ISS API приходят с задержкой ~15 минут. Используйте на свой риск.


## Установка

### Docker (рекомендуется)

```bash
docker build -t sma .
docker compose up
```

### Вручную
Установка:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Запуск:
```bash
./run.sh
# или вручную:
uvicorn sma.api.app:app --reload --host 0.0.0.0 --port 8000
```

Открыть `http://localhost:8000`.
