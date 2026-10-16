#!/usr/bin/env python3
"""
modelica2imti.py  --  Modelica (modèle PLAT, format OpenModelica getFlatModel)
                      -> modèle multilinéaire implicite (iMTI, format CPN1)
                      -> point d'équilibre, linéarisation, valeurs propres généralisées.

Forme cible (Kaufmann et al., arXiv:2510.16534) :
    0 = Phi @ s(v),   s_r(v) = prod_i (1 - |S_ir| + S_ir * v_i),   v = (zdot, z, u, alpha)

Usage :  python modelica2imti.py modele_plat.mo [--out imti.npz]
"""
import re, sys, math, time, collections, argparse
import numpy as np
import sympy as sp
from scipy.optimize import least_squares
from scipy.linalg import eig

ID = r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"
NAME = re.compile(r"(?<![\w.])(" + ID + r")(?![\w.])(?!\s*\()")
MATH = dict(cos=math.cos, sin=math.sin, atan2=math.atan2, exp=math.exp, sqrt=math.sqrt, abs=abs)
mg = lambda n: n.replace(".", "__")
NOTES = []
ALIAS = {}


# ----------------------------------------------------------------------------------------
# 1. Lecture du modèle plat
# ----------------------------------------------------------------------------------------
def read_model(path):
    txt = open(path, encoding="utf-8").read()
    m = re.search(r"^\s*class\s+(\S+)\s(.*?)^\s*end\s+\1\s*;", txt, re.S | re.M)
    if not m:
        sys.exit("pas de bloc 'class ... end ...;' trouvé : fournir un modèle plat (getFlatModel)")
    parts = re.split(r"^\s*(initial equation|equation)\s*$", m.group(2), flags=re.M)
    eq = "".join(c for k, c in zip(parts[1::2], parts[2::2]) if k == "equation")
    return parts[0], eq


DECL = re.compile(r"^((?:(?:final|protected|parameter|constant)\s+)*)"
                  r"(Real|Boolean|Integer|enumeration\([^)]*\))\s+(" + ID + r")\s*(\([^)]*\))?\s*(?:=\s*(.+))?$")


def parse_decls(decl):
    decl = re.sub(r"/\*.*?\*/", "", decl, flags=re.S)
    praw, vstart = {}, {}
    for stmt in decl.split(";"):
        s = " ".join(stmt.split())
        s = re.sub(r'\s*"[^"]*"\s*$', "", s)          # chaîne de description
        s = re.sub(r'\w+\s*=\s*"[^"]*"', "", s)       # attributs chaînes (unit="1"...)
        m = DECL.match(s)
        if not m or m.group(2) != "Real":
            continue
        pre, _, name, attrs, rhs = m.groups()
        if "parameter" in pre or "constant" in pre:
            praw[name] = rhs
        else:
            sm = re.search(r"start\s*=\s*([^,)]+)", attrs or "")
            vstart[name] = sm.group(1).strip() if sm else None
    return praw, vstart


def num(expr, P):
    e = NAME.sub(lambda m: "(" + repr(P[m.group(1)]) + ")", expr.replace("^", "**"))
    return float(eval(e, {"__builtins__": {}}, dict(MATH)))


def eval_params(praw):
    env, vals, pend = dict(MATH), {}, dict(praw)
    while pend:
        progress = False
        for k, e in list(pend.items()):
            try:
                v = float(eval(NAME.sub(lambda m: mg(m.group(1)), e.replace("^", "**")), {"__builtins__": {}}, env))
            except NameError:
                continue
            env[mg(k)] = vals[k] = v
            del pend[k]
            progress = True
        if not progress:
            break
    if pend:
        print("WARN paramètres non résolus :", list(pend)[:5])
    return vals


# ----------------------------------------------------------------------------------------
# 2. Équations : formes spéciales (Complex, if, when) -> équations réelles
# ----------------------------------------------------------------------------------------
def parse_equations(eq_text):
    eq_text = re.sub(r"\bwhen\b.*?\bend when\s*;", "", eq_text, flags=re.S)
    out, inputs = [], []
    n_when = 0
    for stmt in eq_text.split(";"):
        s = " ".join(stmt.split())
        if not s:
            continue
        if re.search(r"\b(true|false)\b|\bpre\(", s):         # logique booléenne : hors périmètre
            n_when += 1
            continue
        m = re.match(rf"^({ID})\.V = Complex\.'\*'\.multiply\(Complex\.'constructor'\.fromReal\(({ID}), 0\.0\), "
                     rf"Modelica\.ComplexMath\.exp\(Complex\.'\*'\.multiply\(Complex\(0\.0, 1\.0\), "
                     rf"Complex\.'constructor'\.fromReal\(({ID}), 0\.0\)\)\)\)$", s)
        if m:                                                  # V = A * exp(j*phi)
            t, a, p = m.groups()
            out += [(f"{t}.V.re", f"{a} * cos({p})"), (f"{t}.V.im", f"{a} * sin({p})")]
            continue
        m = re.match(rf"^({ID}) = Dynawo\.Types\.ComplexVoltagePu\.'-'\.negate\(({ID})\)$", s)
        if m:
            a, b = m.groups()
            out += [(f"{a}.re", f"-{b}.re"), (f"{a}.im", f"-{b}.im")]
            continue
        m = re.match(rf"^({ID}) = Modelica\.ComplexMath\.(real|imag)\(Dynawo\.Types\.ComplexVoltagePu\.'\*'\.multiply\("
                     rf"({ID}), Modelica\.ComplexMath\.conj\(({ID})\)\)\)$", s)
        if m:                                                  # P = Re(V conj(I)), Q = Im(V conj(I))
            y, fn, v, i = m.groups()
            out.append((y, f"{v}.re * {i}.re + {v}.im * {i}.im" if fn == "real" else f"{v}.im * {i}.re - {v}.re * {i}.im"))
            continue
        m = re.match(rf"^({ID}) = Complex\(0\.0, 0\.0\)$", s)
        if m:
            out += [(f"{m.group(1)}.re", "0.0"), (f"{m.group(1)}.im", "0.0")]
            continue
        m = re.match(rf"^({ID}) = Modelica\.ComplexMath\.'abs'\(({ID})\)$", s)
        if m:                                                  # |V| : y^2 = re^2 + im^2 (polynomial)
            y, v = m.groups()
            out.append((f"{y} * {y}", f"{v}.re * {v}.re + {v}.im * {v}.im"))
            continue
        if re.match(rf"^{ID} = Modelica\.ComplexMath\.arg\(", s):
            NOTES.append("arg(V) (atan2) ignoré : sortie pure, non multilinéarisable exactement")
            continue
        if re.search(r"\btime\b", s):                           # source dépendant du temps -> entrée
            inputs.append(s.split("=", 1)[0].strip())
            continue
        m = re.match(r"^(.*?)=\s*if\b.*\bthen\s+0\.0\s+else\s+(.+)$", s)
        if m:                                                  # intégrateur saturé : branche non saturée
            s = f"{m.group(1)}= {m.group(2)}"
            NOTES.append("saturations (if/then/else) : branche non saturée retenue (à vérifier à l'équilibre)")
        lhs, rhs = s.split("=", 1)
        out.append((lhs.strip(), rhs.strip()))
    if n_when:
        NOTES.append(f"{n_when} équations booléennes / blocs when ignorés (état logique fixé)")
    return out, inputs


def to_sym(e, P):
    e = re.sub(r"\bder\(\s*(" + ID + r")\s*\)", lambda m: "D__" + m.group(1), e).replace("^", "**")
    return sp.sympify(NAME.sub(lambda m: "(" + repr(P[m.group(1)]) + ")" if m.group(1) in P else mg(m.group(1)), e))


# ----------------------------------------------------------------------------------------
# 3. Réduction : alias, constantes, élimination des sorties mortes
# ----------------------------------------------------------------------------------------
def alias_pair(r):
    if r.is_Add and len(r.args) == 2:
        a, b = r.args
        for p, q in ((a, b), (b, a)):
            if p.is_Symbol and (-q).is_Symbol:
                return p, -q
    return None


def reduce_system(pairs, inputs_raw, P, vstart):
    eqs = [to_sym(l, P) - to_sym(r, P) for l, r in pairs]
    states = {mg(m.group(1)) for l, r in pairs for m in re.finditer(r"\bder\(\s*(" + ID + r")\s*\)", l + " " + r)}
    inputs = {mg(i) for i in inputs_raw}
    start = {}
    for k, e in vstart.items():
        if e is not None:
            try:
                start[mg(k)] = num(e, P)
            except KeyError:
                pass
    n0 = len(eqs)

    # --- alias x = y (union-find) ---
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    rest = []
    for e in eqs:
        pr = alias_pair(e)
        if pr and not pr[0].name.startswith("D__") and not pr[1].name.startswith("D__"):
            parent[find(pr[0])] = find(pr[1])
        else:
            rest.append(e)
    classes = collections.defaultdict(list)
    for x in list(parent):
        classes[find(x)].append(x)
    mapping, cstart = {}, {}
    for mem in classes.values():
        sts = [m for m in mem if m.name in states]
        ins = [m for m in mem if m.name in inputs]
        if len(sts) > 1 or (sts and ins):
            raise ValueError(f"alias entre plusieurs états/entrées : {[m.name for m in mem]}")
        rep = (ins or sts or sorted(mem, key=lambda m: (len(m.name), m.name)))[0]
        for m in [rep] + sorted(mem, key=lambda m: m.name):
            if m.name in start:
                cstart[rep.name] = start[m.name]
                break
        mapping.update({m: rep for m in mem if m != rep})
    ALIAS.clear(); ALIAS.update({a.name: b.name for a, b in mapping.items()})
    rest = [e.xreplace(mapping) for e in rest]
    for k, v in cstart.items():
        start[k] = v
    n_alias = n0 - len(rest)

    # --- constantes x = c  (et der(x) = 0  =>  x constant) ---
    const = {}
    changed = True
    while changed:
        changed = False
        for i, e in enumerate(rest):
            fs = e.free_symbols
            if not fs:
                rest.pop(i); changed = True; break
            if len(fs) == 1:
                x = next(iter(fs)); d = sp.diff(e, x)
                if not d.free_symbols and d != 0:
                    val = float(-e.subs(x, 0) / d)
                    rep = {x: sp.Float(val)}
                    if x.name.startswith("D__"):
                        st = sp.Symbol(x.name[3:])
                        assert abs(val) < 1e-12, "dérivée constante non nulle"
                        if st.name not in start:
                            raise ValueError(f"état constant {st.name} sans valeur 'start'")
                        rep = {x: sp.Float(0), st: sp.Float(start[st.name])}
                        const[st.name] = start[st.name]
                        states.discard(st.name)
                    else:
                        const[x.name] = val
                    rest.pop(i)
                    rest = [r.xreplace(rep) for r in rest]
                    changed = True
                    break

    # --- sorties mortes (variable présente dans une seule équation) ---
    n_before = len(rest)
    while True:
        occ = collections.defaultdict(list)
        for i, e in enumerate(rest):
            for s in e.free_symbols:
                occ[s].append(i)
        victim = None
        for s, idx in occ.items():
            n = s.name
            if n in inputs or len(idx) != 1:
                continue
            if n.startswith("D__"):
                x = sp.Symbol(n[3:])
                if all(j == idx[0] for j in occ.get(x, [])):
                    victim = idx[0]; break
            elif n not in states:
                victim = idx[0]; break
        if victim is None:
            break
        rest.pop(victim)
    stats = dict(raw=n0, alias=n_alias, const=len(const), dead=n_before - len(rest), final=len(rest))
    return rest, states, inputs, start, const, stats


def classify(eqs, states_hint, inputs):
    syms = set().union(*[e.free_symbols for e in eqs])
    D = {s for s in syms if s.name.startswith("D__")}
    states = sorted({s.name[3:] for s in D})
    ins = sorted(i for i in inputs if sp.Symbol(i) in syms)
    algs = sorted((s for s in syms if not s.name.startswith("D__") and s.name not in states and s.name not in ins),
                  key=lambda s: s.name)
    return states, ins, algs


# ----------------------------------------------------------------------------------------
# 4. Lifting trigonométrique (cos/sin d'un état -> 2 états, sans contrainte, comme dans le papier)
# ----------------------------------------------------------------------------------------
def lift_trig(eqs, start):
    eqs = list(eqs)
    thetas = set()
    for e in eqs:
        for f in e.atoms(sp.sin, sp.cos):
            if not f.args[0].is_Symbol:
                raise ValueError(f"argument trigonométrique non symbolique : {f}")
            thetas.add(f.args[0])
    pairs = []
    for th in sorted(thetas, key=lambda s: s.name):
        Dth = sp.Symbol("D__" + th.name)
        idx = [i for i, e in enumerate(eqs) if Dth in e.free_symbols]
        if len(idx) != 1:
            raise ValueError(f"{th} n'est pas un état avec une EDO unique : lifting impossible")
        e = eqs.pop(idx[0])
        a = sp.diff(e, Dth)
        if a.free_symbols:
            raise ValueError("EDO de l'angle non résolue en der(theta)")
        f = sp.expand(-(e - a * Dth) / a)                      # der(theta) = f
        Cn, Sn = sp.Symbol("cos__" + th.name), sp.Symbol("sin__" + th.name)
        eqs = [q.xreplace({sp.cos(th): Cn, sp.sin(th): Sn}) for q in eqs]
        if any(th in q.free_symbols for q in eqs):
            raise ValueError(f"{th} apparaît hors d'un cos/sin : lifting impossible")
        eqs += [sp.Symbol("D__" + Cn.name) + Sn * f, sp.Symbol("D__" + Sn.name) - Cn * f]
        th0 = start[th.name]
        start[Cn.name], start[Sn.name] = math.cos(th0), math.sin(th0)
        pairs.append((Cn.name, Sn.name, th.name))
    return eqs, pairs


# ----------------------------------------------------------------------------------------
# 5. Multilinéarisation exacte + format CPN1
# ----------------------------------------------------------------------------------------
def multilinearize(eqs, states, inputs, algs):
    D = [sp.Symbol("D__" + s) for s in states]
    X = [sp.Symbol(s) for s in states]
    U = [sp.Symbol(u) for u in inputs]
    A = list(algs)
    base = D + X + U + A
    polys, maxdeg = [], collections.Counter()
    for e in eqs:
        try:
            P = sp.Poly(sp.expand(e), *base)
        except sp.PolynomialError:
            raise ValueError(f"équation non polynomiale (hors périmètre exact) : {e}")
        polys.append(P)
        for mon, _ in P.terms():
            for s, k in zip(base, mon):
                maxdeg[s] = max(maxdeg[s], k)
    copies = {s: [sp.Symbol(f"{s.name}__c{j}") for j in range(1, k)] for s, k in maxdeg.items() if k > 1}
    A_all = A + [c for s in copies for c in copies[s]]
    allv = D + X + U + A_all
    idx = {s: i for i, s in enumerate(allv)}
    rows = []
    for P in polys:
        row = collections.defaultdict(float)
        for mon, c in P.terms():
            ids = []
            for s, k in zip(base, mon):
                if k:
                    ids.append(idx[s]); ids += [idx[c_] for c_ in copies.get(s, [])[: k - 1]]
            row[frozenset(ids)] += float(c)
        rows.append(row)
    for s, cs in copies.items():                                  # alpha_j - x = 0
        for c in cs:
            rows.append({frozenset([idx[c]]): 1.0, frozenset([idx[s]]): -1.0})
    cols = sorted({k for r in rows for k in r}, key=lambda k: (len(k), sorted(k)))
    cidx = {k: j for j, k in enumerate(cols)}
    Phi = np.zeros((len(rows), len(cols)))
    S = np.zeros((len(allv), len(cols)))
    for i, r in enumerate(rows):
        for k, c in r.items():
            Phi[i, cidx[k]] = c
    for j, k in enumerate(cols):
        for v in k:
            S[v, j] = 1.0
    layout = dict(n=len(states), m=len(U), q=len(A_all))
    src = {c.name: s.name for s, cs in copies.items() for c in cs}
    return Phi, S, [s.name for s in allv], layout, src


def s_eval(S, v):
    return np.prod(1 - np.abs(S) + S * v[:, None], axis=0)


def jacobian(Phi, S, v):
    """J = Phi * dS/dv  (éq. (15)-(16) du papier ; produit « exclusif » par préfixes/suffixes,
    robuste aux zéros)."""
    F = 1 - np.abs(S) + S * v[:, None]
    Nv, R = F.shape
    pre = np.ones((Nv + 1, R)); suf = np.ones((Nv + 1, R))
    for i in range(Nv):
        pre[i + 1] = pre[i] * F[i]
    for i in range(Nv - 1, -1, -1):
        suf[i] = suf[i + 1] * F[i]
    return Phi @ (S * pre[:-1] * suf[1:]).T


# ----------------------------------------------------------------------------------------
# 6. Équilibre, linéarisation, valeurs propres généralisées
# ----------------------------------------------------------------------------------------
def gen_eig(A, E, tol=1e-10):
    al, be = eig(A, E, right=False, homogeneous_eigvals=True)
    fin = np.abs(be) > tol * np.maximum(np.abs(al), 1e-300) * 0 + tol * (np.abs(al) + np.abs(be))
    return al[fin] / be[fin], int((~fin).sum())


def analyze(path, overrides=None, post_step=False, show_all=False, out=None, csv=None, verbose=True):
    """Pipeline complet ; renvoie un dict (spectre, équilibre, matrices). overrides = {nom_param: valeur}."""
    pr = print if verbose else (lambda *a, **k: None)
    NOTES.clear()
    decl, eq_text = read_model(path)
    praw, vstart = parse_decls(decl)
    for k_, v_ in (overrides or {}).items():
        if k_ not in praw:
            raise KeyError(f"paramètre inconnu : {k_}")
        praw[k_] = repr(float(v_))
    P = eval_params(praw)
    pairs, inputs_raw = parse_equations(eq_text)
    eqs, states_set, inputs, start, const, st = reduce_system(pairs, inputs_raw, P, vstart)
    pr(f"[parse] {len(P)} paramètres, {len(vstart)} variables réelles déclarées, {st['raw']} équations scalaires")
    pr(f"[réduction] alias éliminés: {st['alias']}, constantes: {st['const']}, sorties mortes: {st['dead']}"
          f" -> {st['final']} équations")

    states, ins, algs = classify(eqs, states_set, inputs)
    n_unk = len(states) + len(algs)
    pr(f"[structure] {len(states)} états, {len(algs)} algébriques, {len(ins)} entrées ({', '.join(ins)}) ;"
          f" inconnues (zdot+alpha) = {n_unk}, équations = {len(eqs)}")
    if n_unk != len(eqs):
        sys.exit("système non carré : vérifier le modèle (équations redondantes ou manquantes)")
    for i in ins:                                                # valeur d'entrée = offset de la source (avant l'échelon)
        b = i.replace("__", ".")[:-2]
        start[i] = P[b + ".offset"] + (P[b + ".height"] if post_step else 0.0)

    eqs_ref, states_ref, algs_ref = list(eqs), list(states), list(algs)
    start_ref = dict(start)
    eqs_l, trig = lift_trig(eqs, start)
    states_l, _, algs_l = classify(eqs_l, None, inputs)
    pr(f"[lifting] {len(trig)} angle(s) -> (cos, sin) : " + ", ".join(t[2].split('__')[-3:-1][0] + '.' + t[2].split('__')[-1] for t in trig))

    Phi, S, names, lay, src = multilinearize(eqs_l, states_l, ins, algs_l)
    n, m, q = lay["n"], lay["m"], lay["q"]
    Nv, R = S.shape
    pr(f"[iMTI] n={n} états, m={m} entrées, q={q} algébriques | Nv={Nv} variables, Nphi={Phi.shape[0]} équations,"
          f" R={R} facteurs CPN1 (tenseur plein : {Phi.shape[0]}*2^{Nv} éléments, CPN1 : {Phi.shape[0]}*{R} = {Phi.size})")
    assert Phi.shape[0] == n + q
    iD, iZ = np.arange(0, n), np.arange(n, 2 * n)
    iU, iA = np.arange(2 * n, 2 * n + m), np.arange(2 * n + m, Nv)

    # ---------- équilibre ----------
    v0 = np.zeros(Nv)
    missing = 0
    for k, nm in enumerate(names):
        base = src.get(nm, nm[2:] if nm.startswith("D__") else nm)
        if nm.startswith("D__"):
            v0[k] = 0.0
        elif base in start:
            v0[k] = start[base]
        else:
            missing += 1
    idxW = np.concatenate([iZ, iA])
    cs_pairs = [(names.index(c), names.index(s)) for c, s, _ in trig]

    def fun(w):
        v = v0.copy(); v[idxW] = w
        return np.concatenate([Phi @ s_eval(S, v), [v[c] ** 2 + v[s] ** 2 - 1 for c, s in cs_pairs]])

    def jac(w):
        v = v0.copy(); v[idxW] = w
        J = jacobian(Phi, S, v)[:, idxW]
        ex = np.zeros((len(cs_pairs), len(idxW)))
        for r, (c, s) in enumerate(cs_pairs):
            ex[r, list(idxW).index(c)] = 2 * v[c]; ex[r, list(idxW).index(s)] = 2 * v[s]
        return np.vstack([J, ex])

    w0 = v0[idxW].copy()
    r0 = np.linalg.norm(fun(w0), np.inf)
    sol = least_squares(fun, w0, jac=jac, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15, x_scale="jac", max_nfev=500)
    vbar = v0.copy(); vbar[idxW] = sol.x
    r1 = np.linalg.norm(fun(sol.x), np.inf)
    pr(f"[équilibre] {missing} variables sans 'start' (init. 0) ; ||h||inf : départ {r0:.2e} -> {r1:.2e}")

    # ---------- linéarisation CPN1 ----------
    t0 = time.perf_counter()
    for _ in range(200):
        J = jacobian(Phi, S, vbar)
    t_cpn = (time.perf_counter() - t0) / 200
    E = -np.hstack([J[:, iD], np.zeros((n + q, q))])
    A = np.hstack([J[:, iZ], J[:, iA]])
    B = J[:, iU]
    lam, n_inf = gen_eig(A, E)
    lam = lam[np.argsort(-lam.real)]
    rng = np.random.default_rng(0)
    regular = all(np.linalg.matrix_rank(A - (rng.normal() + 1j * rng.normal()) * E) == n + q for _ in range(3))
    rE = np.linalg.matrix_rank(E)
    pr(f"[linéarisation CPN1] {t_cpn*1e3:.3f} ms ; rang(E)={rE} ; {len(lam)} valeurs propres finies, "
          f"{n_inf} infinies ; faisceau (A,E) {'régulier' if regular else 'SINGULIER !'}")
    if len(lam) < rE:
        pr(f"  -> {rE - len(lam)} v.p. finies de moins que rang(E) : DAE d'indice > 1 "
              "(contraintes entre états via les variables algébriques)")

    # ---------- référence : Jacobien symbolique du modèle NON multilinéarisé ----------
    Dr = [sp.Symbol("D__" + s) for s in states_ref]
    Xr = [sp.Symbol(s) for s in states_ref]
    Ur = [sp.Symbol(u) for u in ins]
    Ar = list(algs_ref)
    F = sp.Matrix(eqs_ref)
    allr = Dr + Xr + Ur + Ar
    jac_or_empty = lambda V: F.jacobian(V) if len(V) else sp.zeros(len(eqs_ref), 0)
    fl = sp.lambdify(allr, [F, F.jacobian(Dr), F.jacobian(Xr), F.jacobian(Ar), jac_or_empty(Ur)], "numpy")
    val = {nm: vbar[k] for k, nm in enumerate(names)}
    for c, s, th in trig:
        val[th] = math.atan2(val[s], val[c])
    args_ref = [0.0] * len(Dr) + [val[s] for s in states_ref] + [val[u] for u in ins] + [val[a.name] for a in algs_ref]
    t0 = time.perf_counter()
    for _ in range(20):
        Fr, JDr, JZr, JAr, JUr = fl(*args_ref)
    t_ref = (time.perf_counter() - t0) / 20
    Er = -np.hstack([np.array(JDr, float), np.zeros((len(eqs_ref), len(Ar)))])
    Aref = np.hstack([np.array(JZr, float), np.array(JAr, float)])
    lam_ref, _ = gen_eig(Aref, Er)
    lam_ref = lam_ref[np.argsort(-lam_ref.real)]
    pr(f"[référence non lifté] ||F||inf={np.abs(np.array(Fr, float)).max():.2e} ; "
          f"{len(lam_ref)} v.p. finies ; Jacobien sympy-lambdify {t_ref*1e3:.3f} ms")

    # comparaison des spectres
    used, err = set(), 0.0
    for l in lam_ref:
        d = np.abs(lam - l); d[list(used)] = np.inf
        k = int(np.argmin(d)); used.add(k); err = max(err, d[k] / max(1.0, abs(l)))
    extra = [lam[k] for k in range(len(lam)) if k not in used]
    pr(f"[validation] chaque v.p. du modèle d'origine retrouvée dans l'iMTI : écart relatif max = {err:.2e}")
    pr(f"             v.p. supplémentaires de l'iMTI (modes du lifting) : {[complex(round(x.real,6), round(x.imag,6)) for x in extra]}")
    stable = np.all(lam_ref.real < 1e-6)
    pr(f"[stabilité petits signaux] max Re(lambda) = {lam_ref.real.max():.4f}  ->  {'STABLE' if stable else 'INSTABLE'}")
    if show_all:
        for l in lam_ref:
            pr(f"    {l.real:12.4f} {l.imag:+12.4f} j  f={abs(l.imag)/2/math.pi:8.3f} Hz")
    marg = [l for l in lam_ref if l.imag > 0 and abs(l.real) < 1e-3 * abs(l)]
    for l in marg:
        pr(f"  ATTENTION mode quasi non amorti : {l.real:+.4f} {l.imag:+.2f}j ({abs(l.imag)/2/math.pi:.2f} Hz, "
           f"amort. = {-l.real/abs(l):.1e}) -> la conclusion STABLE/INSTABLE dépend de la précision "
           "(paramètres, tolérance du solveur) ; faire une analyse de sensibilité")
    pr("  modes dominants (les plus proches de l'axe imaginaire) :")
    for l in lam_ref[:8]:
        zeta = -l.real / abs(l) if abs(l) > 0 else float('nan')
        pr(f"    {l.real:12.4f} {l.imag:+12.4f} j   (f = {abs(l.imag)/2/math.pi:8.3f} Hz, amort. = {zeta:6.3f})")

    # vérifs : saturations inactives
    for k, nm in enumerate(names):
        if nm.endswith("limIntegrator__y") or nm.endswith("limIntegrator1__y"):
            base = nm.replace("__y", "")
            lo, hi = P.get(base.replace("__", ".") + ".outMin"), P.get(base.replace("__", ".") + ".outMax")
            if lo is not None:
                ok = "inactive, OK" if lo < vbar[k] < hi else "ACTIVE : résultat invalide !"
                pr(f"  saturation {base.replace('__', '.')}: y={vbar[k]:+.4f} dans [{lo:g}, {hi:g}] -> {ok}")
    for s in dict.fromkeys(NOTES):
        pr("  note:", s)
    if out:
        np.savez(out, Phi=Phi, S=S, names=np.array(names), vbar=vbar, E=E, A=A, B=B, n=n, m=m, q=q)
        pr(f"[sauvegarde] {out}  (Phi {Phi.shape}, S {S.shape})")
    if csv:
        with open(csv, "w") as fh:
            fh.write("nom;valeur_equilibre\n")
            for nm in vstart:
                rep = ALIAS.get(mg(nm), mg(nm))
                v_ = val.get(rep, const.get(rep))
                if v_ is not None:
                    fh.write(f"{nm};{v_:.12g}\n")
        pr(f"[export] équilibre par nom Modelica d'origine -> {csv}")
    return dict(lam=lam_ref, lam_imti=lam, Aref=Aref, Er=Er, states=list(states_ref),
                algs=[a.name for a in algs_ref], val=val, P=P)




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--out", default="imti.npz")
    ap.add_argument("--post-step", action="store_true", help="linéariser avec consigne = offset + height (après l'échelon)")
    ap.add_argument("--all", action="store_true", help="afficher toutes les valeurs propres finies")
    ap.add_argument("--csv", default=None, help="exporter l'équilibre (valeurs par nom Modelica) dans ce fichier")
    a = ap.parse_args()
    analyze(a.model, post_step=a.post_step, show_all=a.all, out=a.out, csv=a.csv)


if __name__ == "__main__":
    main()
