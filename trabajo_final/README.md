# Bruno — Mozo Virtual (Proyecto CACIC 2025)

**Bruno** un **Mozo Virtual** desarrollado como parte del **Proyecto CACIC 2025**. Sigue estos sencillos pasos para tener la aplicación funcionando en tu equipo local.

***

## 1. Prepara el Entorno Virtual

Es fundamental crear y utilizar un **entorno virtual** para aislar las dependencias del proyecto y evitar conflictos con otras instalaciones de Python en tu sistema.

1. **Crea** el entorno virtual:
    ```bash
    python3 -m venv .venv
    ```
2. **Activa** el entorno virtual para comenzar a trabajar dentro de él:
    ```bash
    source .venv/bin/activate
    ```
    *(Cuando el entorno esté activo, verás `(.venv)` al inicio de tu línea de comandos.)*

***

## 2. Instalar Dependencias

Con el entorno virtual activado, procede a instalar todas las bibliotecas y herramientas necesarias para el correcto funcionamiento de **Bruno**.

```bash
python -m pip install -r requirements.txt
```

## 3. Ejecutar Bruno

Una vez instaladas las dependencias, navega a la carpeta principal del código fuente y ejecuta la aplicación.

1. **Accede** a la carpeta principal del código:
    ```bash
    cd trabajo_final/src
    ```

2. **Ejecuta** el programa:
    ```bash
    python3 bruno_mozo_virtual.py
    ```