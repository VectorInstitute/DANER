---

# Get/Put Annotation

<div grid="~ cols-2 gap-x-10 gap-y-2">

<div>

- Get Annotation

  - http://127.0.0.1:5000/annotation

```python{all}
{
    "index": 13606,
    "ents": [
        {'token': 'Bowling', 'label': 'null', 'iob': 2, "confidence": 0.06744539737701416},
        {'token': '(', 'label': 'null', 'iob': 2, "confidence": 0.11347659677267075},
        {'token': 'to', 'label': 'null', 'iob': 2, "confidence": 0.152160182595253},
        {'token': 'date', 'label': 'null', 'iob': 2, "confidence": 0.1574113368988037},
        {'token': ')', 'label': 'null', 'iob': 2, "confidence": 0.11822989583015442},
        {'token': ':', 'label': 'null', 'iob': 2, "confidence": 0.13801075518131256},
        {'token': 'Wasim', 'label': 'null', 'iob': 2, "confidence": 0.13231147825717926},
        {'token': 'Akram', 'label': 'null', 'iob': 2, "confidence": 0.16222646832466125},
        {'token': '25-8-61-1', 'label': 'null', 'iob': 2, "confidence": 0.1321057677268982},
        ...
    ],
}
```

</div>

<div>

- Put Annotation
- http://127.0.0.1:5000/annotation?index={index}&annotator={annotator}&label={label}

```python{all}
{
    "index": 13606
    "annotator": "admin"
    "ents": [
        {'token': 'Bowling', 'label': 'MISC', 'iob': 3},
        {'token': '(', 'label': 'null', 'iob': 2},
        {'token': 'to', 'label': 'null', 'iob': 2},
        {'token': 'date', 'label': 'null', 'iob': 2},
        {'token': ')', 'label': 'null', 'iob': 2},
        {'token': ':', 'label': 'null', 'iob': 2},
        {'token': 'Wasim', 'label': 'PER', 'iob': 3},
        {'token': 'Akram', 'label': 'PER', 'iob': 1},
        {'token': '25-8-61-1', 'label': 'null', 'iob': 2},
        ...
}
```

</div>
</div>

<style>
.wrapper2 {
  display: grid;
  grid-template-columns: 1.5fr 2fr;
}
</style>

<!--
Basically, when the frontend send get annotation request, the backend service will give back a json file with the data index and model's current predcition, which we can use as auto suggestion to facillitate labeling process.

In contrast, when the frontend send put annotation request, it needs to send the data index, the annotator name and label information.

-->
