## Interpretación del experimento
- Con un umbral de **0.30** aparecen más detecciones, pero aumenta la posibilidad de aceptar objetos incorrectos.
- Con **0.60** se obtiene un equilibrio entre cantidad y confianza.
- Con **0.85** solo permanecen las detecciones más seguras, pero pueden perderse objetos pequeños o lejanos.
- La ausencia de detecciones es un resultado válido y no debe provocar una excepción.




## Explicación sencilla para el maestro
La aplicación original utilizaba YOLO, BLIP y Stable Diffusion para detectar, describir y transformar una imagen.
En mi aplicación cambié completamente el objetivo y los modelos:

1. **DETR** encuentra personas, vehículos, semáforos y señales.
2. **DPT-Hybrid** genera un mapa de profundidad relativa.
3. Para cada objeto, el programa consulta la cercanía dentro de su caja.
4. Se combinan cercanía, tamaño y tipo de objeto.
5. Se asigna un riesgo bajo, medio o alto.
6. Gradio presenta la imagen anotada, el mapa de cercanía y una tabla.

La principal limitación es que DPT trabaja con una sola fotografía, por lo que no entrega una distancia confiable en metros. La puntuación es demostrativa y no está diseñada para controlar un automóvil.






# CONCLUSION
Desarrollé una aplicación para analizar escenas viales. DETR detecta y localiza los objetos, mientras que DPT-Hybrid estima un mapa de cercanía relativa. Después cruzo la caja de cada detección con el mapa de DPT y calculo una puntuación usando 65 % de cercanía, 25 % de tamaño y 10 % de prioridad del objeto. 


En las cinco imágenes se mostraron 59 objetos: 8 en nivel alto, 25 en medio y 26 en bajo. La escena carretera2 tuvo la mayor concentración de riesgo. Después del calentamiento inicial, el procesamiento tardó alrededor de 0.20 segundos por imagen y utilizó solamente 0.51 GB de la GPU. La principal limitación es que la profundidad no está expresada en metros y el riesgo es una regla académica, no un sistema certificado de conducción.