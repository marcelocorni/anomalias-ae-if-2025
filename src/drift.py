import math

from .comum import ident, parquet


def cdf(con, relation, feature, boundaries):
    expressions = ','.join(f'avg(({ident(feature)} <= {float(v):.17g})::DOUBLE)' for v in boundaries)
    return list(con.execute(f'SELECT {expressions} FROM {relation}').fetchone())


def shares(con, relation, feature, boundaries):
    if not boundaries:
        return [1.]
    clauses = ' '.join(f'WHEN {ident(feature)} <= {float(v):.17g} THEN {i}' for i,v in enumerate(boundaries))
    groups = dict(con.execute(f'SELECT CASE {clauses} ELSE {len(boundaries)} END AS bucket,count(*) FROM {relation} GROUP BY bucket').fetchall())
    total = sum(groups.values())
    return [groups.get(k,0)/total for k in range(len(boundaries)+1)]


def psi(expected, actual, epsilon=1e-6):
    return sum((max(a,epsilon)-max(e,epsilon))*math.log(max(a,epsilon)/max(e,epsilon)) for e,a in zip(expected,actual))


def perfil(con, relation, feature):
    result = con.execute(f'''SELECT count(*),min({ident(feature)}),max({ident(feature)}),
        avg({ident(feature)}),stddev_samp({ident(feature)}),
        quantile_cont({ident(feature)},[0.25,0.5,0.75]),avg(({ident(feature)}=0)::DOUBLE)
        FROM {relation}''').fetchone()
    n, low, high, mean, std, qs, zeros = result
    return dict(rows=n, min=low, max=high, mean=mean, std=std, q1=qs[0], median=qs[1], q3=qs[2], zero_share=zeros)


def analisar(con, by_window, train_key, features, settings):
    reference = parquet(by_window[train_key])
    output = []
    for feature in features:
        p = perfil(con, reference, feature)
        psi_probs = [k/settings['psi_bins'] for k in range(1,settings['psi_bins'])]
        ks_probs = [k/(settings['ks_grid_points']-1) for k in range(settings['ks_grid_points'])]
        boundaries = sorted(set(con.execute(f'SELECT quantile_cont({ident(feature)}, {psi_probs}) FROM {reference}').fetchone()[0]))
        grid = sorted(set(con.execute(f'SELECT quantile_cont({ident(feature)}, {ks_probs}) FROM {reference}').fetchone()[0]))
        expected, reference_cdf = shares(con,reference,feature,boundaries), cdf(con,reference,feature,grid)
        for key, files in by_window.items():
            current = parquet(files)
            actual, observed_cdf = shares(con,current,feature,boundaries), cdf(con,current,feature,grid)
            profile = perfil(con,current,feature)
            output.append(dict(window=key, feature=feature, reference_window=train_key,
                               psi=psi(expected,actual), ks_approx=max(abs(a-b) for a,b in zip(reference_cdf,observed_cdf)),
                               reference=p, current=profile, median_shift=profile['median']-p['median'],
                               ks_grid_points=len(grid), psi_bins=len(expected)))
    return dict(reference_window=train_key, method='PSI epsilon=1e-6; KS aproximado na grade de quantis exatos do treino; sem p-valor', rows=output)
