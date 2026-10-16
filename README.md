# MultiLinear
# modelica2imti

Convertit un modèle Modelica **plat** (format `getFlatModel` d'OpenModelica, ce fichier peut être obtenu en faisant "instancier le modèle dans OMEdit) en **modèle multilinéaire implicite (iMTI)** au format CPN1, puis calcule l'équilibre, la linéarisation et les valeurs propres généralisées.

Forme cible (Kaufmann et al., [arXiv:2510.16534](https://arxiv.org/abs/2510.16534)) :

```
0 = Φ · s(v),    s_r(v) = ∏_i (1 − |S_ir| + S_ir · v_i),    v = (ż, z, u, α)
```

## Installation

Python ≥ 3.9 avec `numpy`, `scipy`, `sympy` (testé sous 3.12).

## Utilisation

```bash
python modelica2imti.py modele_plat.mo [--out imti.npz] [--csv equilibre.csv] [--post-step] [--all]
```

| Option | Effet |
|---|---|
| `--out` | fichier `.npz` : `Phi`, `S`, `names`, `vbar` (équilibre), `E`, `A`, `B`, `n`, `m`, `q` (défaut `imti.npz`) |
| `--csv` | équilibre exporté par nom Modelica d'origine (hors variables éliminées comme sorties mortes) |
| `--post-step` | linéarise avec consigne = `offset + height` des sources Step (sinon : valeur avant l'échelon) |
| `--all` | affiche toutes les valeurs propres finies |

En Python :

```python
import modelica2imti as M
r = M.analyze("modele_plat.mo", overrides={"GFL1.Kfd": 1.001}, verbose=False)
r["lam"]   # valeurs propres finies du modèle d'origine
```

`analyze` renvoie aussi `lam_imti`, `Aref`, `Er`, `states`, `algs`, `val`, `P`.

## Pipeline

1. **Lecture** : paramètres évalués numériquement, formes `Complex` converties en équations réelles.
2. **Réduction** : alias `x = y`, constantes, sorties mortes éliminés ; vérification que le système est carré.
3. **Lifting trigonométrique** : `cos/sin` d'un état θ deviennent deux états (sans contrainte `C²+S²=1`, d'où une valeur propre nulle supplémentaire dans l'iMTI).
4. **Multilinéarisation exacte** : copies `α = x` pour les puissances > 1 ; construction de (Φ, S).
5. **Équilibre** (moindres carrés, départ = valeurs `start`), Jacobien selon les éq. (15)–(16) du papier, valeurs propres de (A, E).
6. **Validation** : comparaison avec le Jacobien symbolique du modèle non multilinéarisé.

## Hypothèses et limites

- **Booléens, `when`, `pre`** ignorés (état logique fixé). `arg(V)` ignoré (sortie pure).
- **Saturations** : branche non saturée retenue ; le script vérifie qu'elles sont inactives à l'équilibre.
- **Sources dépendant de `time`** : traitées comme entrées (valeur = `offset`).
- **Formes `Complex` reconnues** : `multiply`, `negate`, `conj`, `exp`, `abs` (style Dynawo). Un autre style peut demander d'ajouter des motifs dans `parse_equations`.
- **Équations polynomiales uniquement** (hors `cos/sin` d'un état) ; sinon erreur explicite.
- **DAE d'indice > 1** : signalé ; le faisceau (A, E) est testé régulier, seules les valeurs propres finies sont rapportées.
- **Mode quasi non amorti** (amortissement relatif < 10⁻³) : avertissement, la conclusion stable/instable dépend alors de la précision numérique.
- La validation compare l'iMTI au modèle *tel que lu par le script* : une erreur de lecture (paramètres, équation ignorée) n'est pas détectée. Pour cela, recoupez avec une linéarisation indépendante (par ex. celle d'OpenModelica).
