"""
Tech Challenge B — Dashboard de Resultados
Execute: python dashboard_b.py
Abre automaticamente no navegador em http://localhost:8000
"""

import http.server, webbrowser, threading, os
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64, io

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, confusion_matrix
)
import shap

sns.set_theme(style='whitegrid')
print("Treinando modelos...")

# ── Dados ─────────────────────────────────────────────────────────────────────
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

X, y = df[list(data.feature_names)], df['target']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)
y_pred_lr_val = lr.predict(X_val_s)

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
y_pred_dt_val = dt.predict(X_val)

def m(y_true, y_pred):
    return {
        'acc': accuracy_score(y_true, y_pred),
        'rec': recall_score(y_true, y_pred, pos_label=0),
        'f1':  f1_score(y_true, y_pred, pos_label=0),
    }

val  = {'lr': m(y_val, y_pred_lr_val),  'dt': m(y_val, y_pred_dt_val)}
test = {'lr': m(y_test, y_pred_lr),     'dt': m(y_test, y_pred_dt)}

# ── Gráficos ──────────────────────────────────────────────────────────────────
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight', transparent=True)
    buf.seek(0); plt.close(fig)
    return base64.b64encode(buf.read()).decode()

# Distribuição
fig, ax = plt.subplots(figsize=(5, 3.5))
counts = df['target'].map({0:'Maligno',1:'Benigno'}).value_counts()
bars = ax.bar(counts.index, counts.values, color=['#ef4444','#22c55e'], edgecolor='white', width=0.5)
for b, v in zip(bars, counts.values):
    ax.text(b.get_x()+b.get_width()/2, v+4, str(v), ha='center', fontweight='bold', color='white')
ax.set_facecolor('none'); fig.patch.set_alpha(0)
ax.tick_params(colors='white'); ax.spines[:].set_color('#ffffff22')
ax.set_title('Distribuição dos Diagnósticos', color='white', fontweight='bold')
img_dist = fig_to_b64(fig)

# Matrizes de confusão
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, y_pred, title, cmap in zip(axes, [y_pred_lr, y_pred_dt],
    ['Regressão Logística', 'Árvore de Decisão'], ['Blues','Greens']):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=['Maligno','Benigno'], yticklabels=['Maligno','Benigno'],
                linewidths=2, linecolor='#0f172a', annot_kws={'size':14,'weight':'bold','color':'white'})
    ax.set_title(title, color='white', fontweight='bold')
    ax.set_xlabel('Previsto', color='#94a3b8'); ax.set_ylabel('Real', color='#94a3b8')
    ax.tick_params(colors='#94a3b8'); ax.set_facecolor('none')
fig.patch.set_alpha(0)
img_cm = fig_to_b64(fig)

# Feature importance
imp = pd.Series(dt.feature_importances_, index=data.feature_names).sort_values(ascending=True).tail(10)
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(imp.index, imp.values, color='#38bdf8', edgecolor='none')
ax.set_facecolor('none'); fig.patch.set_alpha(0)
ax.tick_params(colors='white'); ax.spines[:].set_color('#ffffff22')
ax.set_title('Feature Importance — Árvore de Decisão', color='white', fontweight='bold')
img_fi = fig_to_b64(fig)

# SHAP
explainer = shap.LinearExplainer(lr, X_train_s)
shap_vals = explainer.shap_values(X_test_s)
fig, ax = plt.subplots(figsize=(7, 4))
shap.summary_plot(shap_vals, X_test, feature_names=list(data.feature_names),
                  plot_type='bar', show=False, color='#f472b6')
ax = plt.gca(); ax.set_facecolor('none'); fig.patch.set_alpha(0)
ax.tick_params(colors='white'); ax.spines[:].set_color('#ffffff22')
ax.set_title('SHAP — Importância Global', color='white', fontweight='bold')
img_shap = fig_to_b64(fig)

# Comparação validação vs teste
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
metricas = ['acc', 'rec', 'f1']
labels_m = ['Acurácia', 'Recall (Maligno)', 'F1-Score (Maligno)']
x = np.arange(2); modelos = ['Reg. Logística', 'Árvore']
cores = ['#38bdf8', '#f472b6']
for ax, met, label in zip(axes, metricas, labels_m):
    vals_val  = [val['lr'][met],  val['dt'][met]]
    vals_test = [test['lr'][met], test['dt'][met]]
    ax.bar(x - 0.2, vals_val,  0.35, label='Validação', color=cores[0], alpha=0.7)
    ax.bar(x + 0.2, vals_test, 0.35, label='Teste',     color=cores[1], alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(modelos, color='white')
    ax.set_ylim(0.7, 1.05); ax.set_title(label, color='white', fontweight='bold')
    ax.tick_params(colors='white'); ax.set_facecolor('none')
    ax.spines[:].set_color('#ffffff22'); ax.legend(fontsize=8)
fig.patch.set_alpha(0)
img_comp = fig_to_b64(fig)

print("Gráficos prontos!")

# ── HTML ──────────────────────────────────────────────────────────────────────
def pct(v): return f"{v*100:.1f}%"

HTML = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tech Challenge B — Diagnóstico Hospitalar</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:#f0f7f4;
    --surface:#ffffff;
    --card:#ffffff;
    --border:#c8e6d4;
    --border-strong:#a0d4b5;
    --accent1:#1a7a4a;
    --accent2:#25a266;
    --accent3:#0f5c36;
    --danger:#d63a3a;
    --warning:#e07c1a;
    --text:#0d2b1e;
    --muted:#4a7060;
    --light:#e8f5ee;
    --mono:'IBM Plex Mono',monospace;
    --sans:'Plus Jakarta Sans',sans-serif;
  }}
  *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
  html{{scroll-behavior:smooth}}
  body{{background:var(--bg);color:var(--text);font-family:var(--sans);min-height:100vh;overflow-x:hidden}}

  /* cruz hospitalar decorativa no fundo */
  body::before{{
    content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
    background-image:
      linear-gradient(rgba(26,122,74,.06) 1px,transparent 1px),
      linear-gradient(90deg,rgba(26,122,74,.06) 1px,transparent 1px);
    background-size:48px 48px;
  }}

  .wrap{{position:relative;z-index:1;max-width:1100px;margin:0 auto;padding:0 24px 80px}}

  /* ── header ── */
  header{{
    padding:56px 0 44px;
    border-bottom:2px solid var(--border-strong);
    margin-bottom:52px;
    display:flex;flex-direction:column;gap:14px;
    animation:fadeUp .6s ease both;
  }}
  .header-top{{display:flex;align-items:center;gap:16px}}
  .cross{{
    width:44px;height:44px;flex-shrink:0;
    background:var(--accent1);border-radius:10px;
    display:flex;align-items:center;justify-content:center;
    font-size:1.4rem;color:white;
  }}
  .tag{{
    font-family:var(--mono);font-size:10px;letter-spacing:.18em;
    color:var(--accent1);text-transform:uppercase;
    background:var(--light);padding:4px 10px;border-radius:4px;
    border:1px solid var(--border);display:inline-block;
  }}
  h1{{font-size:clamp(1.8rem,4vw,2.8rem);font-weight:700;line-height:1.15;color:var(--text)}}
  h1 span{{color:var(--accent1)}}
  .subtitle{{color:var(--muted);font-size:1rem;font-weight:400;max-width:600px;line-height:1.7}}

  /* ── sections ── */
  section{{margin-bottom:56px;animation:fadeUp .5s ease both}}
  section:nth-child(2){{animation-delay:.1s}} section:nth-child(3){{animation-delay:.2s}}
  section:nth-child(4){{animation-delay:.3s}} section:nth-child(5){{animation-delay:.4s}}
  section:nth-child(6){{animation-delay:.5s}}
  .section-label{{
    font-family:var(--mono);font-size:10px;letter-spacing:.2em;
    color:var(--accent2);text-transform:uppercase;margin-bottom:6px;
  }}
  h2{{font-size:1.2rem;font-weight:700;margin-bottom:24px;color:var(--text);
    padding-bottom:10px;border-bottom:1px solid var(--border)}}

  /* ── metric cards ── */
  .metrics-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:16px}}
  .model-block{{display:flex;flex-direction:column;gap:12px}}
  .model-title{{
    font-family:var(--mono);font-size:11px;color:var(--accent1);
    letter-spacing:.12em;padding:8px 12px;
    background:var(--light);border-radius:6px;border:1px solid var(--border);
  }}
  .metric-card{{
    background:var(--card);border:1px solid var(--border);border-radius:14px;
    padding:20px 24px;position:relative;overflow:hidden;
    box-shadow:0 2px 8px rgba(26,122,74,.08);
    transition:transform .2s,box-shadow .2s,border-color .2s;
  }}
  .metric-card:hover{{transform:translateY(-3px);box-shadow:0 8px 24px rgba(26,122,74,.15);border-color:var(--accent2)}}
  .metric-card::before{{
    content:'';position:absolute;top:0;left:0;right:0;height:3px;
    background:var(--accent-line,var(--accent1));border-radius:14px 14px 0 0;
  }}
  .metric-card.teal {{--accent-line:var(--accent2)}}
  .metric-card.dark {{--accent-line:var(--accent3)}}
  .metric-card.red  {{--accent-line:var(--danger)}}
  .metric-label{{font-size:.72rem;color:var(--muted);margin-bottom:6px;text-transform:uppercase;letter-spacing:.1em}}
  .metric-value{{font-family:var(--mono);font-size:2rem;font-weight:700;color:var(--accent1)}}
  .badge{{display:inline-block;font-family:var(--mono);font-size:10px;padding:3px 8px;
    border-radius:4px;margin-top:6px;margin-right:4px}}
  .badge.val{{background:var(--light);color:var(--accent1);border:1px solid var(--border)}}
  .badge.test{{background:var(--accent1);color:white}}

  /* ── chart cards ── */
  .chart-card{{
    background:var(--card);border:1px solid var(--border);border-radius:16px;
    padding:28px;margin-bottom:20px;
    box-shadow:0 2px 12px rgba(26,122,74,.07);
  }}
  .chart-card img{{width:100%;border-radius:8px}}
  .chart-title{{
    font-size:.8rem;font-weight:600;color:var(--muted);margin-bottom:16px;
    font-family:var(--mono);letter-spacing:.06em;
    display:flex;align-items:center;gap:8px;
  }}
  .chart-title::before{{content:'▸';color:var(--accent2)}}

  .two-col{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
  @media(max-width:680px){{.two-col{{grid-template-columns:1fr}}}}

  /* ── discussion ── */
  .discussion-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}}
  .disc-card{{
    background:var(--card);border:1px solid var(--border);border-radius:14px;
    padding:24px;transition:border-color .2s,box-shadow .2s;
    box-shadow:0 2px 8px rgba(26,122,74,.06);
  }}
  .disc-card:hover{{border-color:var(--accent2);box-shadow:0 6px 20px rgba(26,122,74,.12)}}
  .disc-icon{{font-size:1.6rem;margin-bottom:10px}}
  .disc-title{{font-weight:700;margin-bottom:8px;font-size:.95rem;color:var(--accent3)}}
  .disc-text{{color:var(--muted);font-size:.85rem;line-height:1.7}}

  /* ── split badges ── */
  .split-badge{{display:flex;gap:12px;margin-bottom:28px;flex-wrap:wrap}}
  .split-item{{
    background:var(--card);border:1px solid var(--border);border-radius:12px;
    padding:16px 20px;text-align:center;flex:1;min-width:120px;
    box-shadow:0 2px 8px rgba(26,122,74,.06);
  }}
  .split-pct{{font-family:var(--mono);font-size:1.7rem;font-weight:700;color:var(--accent1)}}
  .split-label{{font-size:.72rem;color:var(--muted);text-transform:uppercase;margin-top:6px;line-height:1.5}}

  /* ── footer ── */
  footer{{
    border-top:2px solid var(--border);padding-top:28px;color:var(--muted);
    font-size:.8rem;font-family:var(--mono);
    display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;
    align-items:center;
  }}
  .footer-logo{{
    background:var(--accent1);color:white;padding:4px 10px;
    border-radius:4px;font-size:.75rem;
  }}

  @keyframes fadeUp{{from{{opacity:0;transform:translateY(20px)}}to{{opacity:1;transform:translateY(0)}}}}
</style>
</head>
<body>
<div class="wrap">

  <header>
    <div class="header-top">
      <div class="cross">🏥</div>
      <span class="tag">PosTech · Tech Challenge · Fase 1 — Projeto B</span>
    </div>
    <h1>Sistema de Suporte ao<br><span>Diagnóstico Hospitalar</span></h1>
    <p class="subtitle">Classificação de exames médicos com Machine Learning para apoio a médicos e equipes clínicas. Dataset: Breast Cancer Wisconsin — 569 pacientes.</p>
  </header>

  <!-- Divisão dos dados -->
  <section>
    <div class="section-label">01 — Metodologia</div>
    <h2>Divisão Treino / Validação / Teste</h2>
    <div class="split-badge">
      <div class="split-item"><div class="split-pct">70%</div><div class="split-label">Treino<br>{X_train.shape[0]} amostras</div></div>
      <div class="split-item"><div class="split-pct">15%</div><div class="split-label">Validação<br>{X_val.shape[0]} amostras</div></div>
      <div class="split-item"><div class="split-pct">15%</div><div class="split-label">Teste<br>{X_test.shape[0]} amostras</div></div>
      <div class="split-item"><div class="split-pct">569</div><div class="split-label">Total<br>amostras</div></div>
    </div>
    <div class="chart-card">
      <div class="chart-title">distribuicao_diagnosticos.png</div>
      <img src="data:image/png;base64,{img_dist}" alt="Distribuição">
    </div>
  </section>

  <!-- Métricas -->
  <section>
    <div class="section-label">02 — Avaliação</div>
    <h2>Métricas — Validação vs Teste</h2>
    <div class="metrics-grid">
      <div class="model-block">
        <div class="model-title">// Regressão Logística</div>
        <div class="metric-card">
          <div class="metric-label">Acurácia</div>
          <div class="metric-value">{pct(test['lr']['acc'])}</div>
          <span class="badge val">Val: {pct(val['lr']['acc'])}</span>
          <span class="badge test">Teste: {pct(test['lr']['acc'])}</span>
        </div>
        <div class="metric-card pink">
          <div class="metric-label">Recall (Maligno)</div>
          <div class="metric-value">{pct(test['lr']['rec'])}</div>
          <span class="badge val">Val: {pct(val['lr']['rec'])}</span>
          <span class="badge test">Teste: {pct(test['lr']['rec'])}</span>
        </div>
        <div class="metric-card green">
          <div class="metric-label">F1-Score (Maligno)</div>
          <div class="metric-value">{pct(test['lr']['f1'])}</div>
        </div>
      </div>
      <div class="model-block">
        <div class="model-title">// Árvore de Decisão</div>
        <div class="metric-card">
          <div class="metric-label">Acurácia</div>
          <div class="metric-value">{pct(test['dt']['acc'])}</div>
          <span class="badge val">Val: {pct(val['dt']['acc'])}</span>
          <span class="badge test">Teste: {pct(test['dt']['acc'])}</span>
        </div>
        <div class="metric-card pink">
          <div class="metric-label">Recall (Maligno)</div>
          <div class="metric-value">{pct(test['dt']['rec'])}</div>
          <span class="badge val">Val: {pct(val['dt']['rec'])}</span>
          <span class="badge test">Teste: {pct(test['dt']['rec'])}</span>
        </div>
        <div class="metric-card green">
          <div class="metric-label">F1-Score (Maligno)</div>
          <div class="metric-value">{pct(test['dt']['f1'])}</div>
        </div>
      </div>
    </div>
  </section>

  <!-- Comparação val vs teste -->
  <section>
    <div class="section-label">03 — Comparação</div>
    <h2>Validação vs Teste por Métrica</h2>
    <div class="chart-card">
      <div class="chart-title">comparacao_validacao_teste.png</div>
      <img src="data:image/png;base64,{img_comp}" alt="Comparação">
    </div>
  </section>

  <!-- Matrizes -->
  <section>
    <div class="section-label">04 — Avaliação Visual</div>
    <h2>Matrizes de Confusão — Conjunto de Teste</h2>
    <div class="chart-card">
      <div class="chart-title">matrizes_confusao.png</div>
      <img src="data:image/png;base64,{img_cm}" alt="Matrizes de Confusão">
    </div>
  </section>

  <!-- Explicabilidade -->
  <section>
    <div class="section-label">05 — Explicabilidade</div>
    <h2>Feature Importance & SHAP</h2>
    <div class="two-col">
      <div class="chart-card">
        <div class="chart-title">feature_importance.png — Árvore de Decisão</div>
        <img src="data:image/png;base64,{img_fi}" alt="Feature Importance">
      </div>
      <div class="chart-card">
        <div class="chart-title">shap_summary.png — Regressão Logística</div>
        <img src="data:image/png;base64,{img_shap}" alt="SHAP">
      </div>
    </div>
  </section>

  <!-- Discussão -->
  <section>
    <div class="section-label">06 — Discussão Crítica</div>
    <h2>Interpretação dos Resultados</h2>
    <div class="discussion-grid">
      <div class="disc-card">
        <div class="disc-icon">🔀</div>
        <div class="disc-title">Por que 3 conjuntos?</div>
        <div class="disc-text">A validação permite comparar e ajustar modelos sem contaminar a avaliação final. O teste é usado uma única vez, simulando dados reais que o modelo nunca viu.</div>
      </div>
      <div class="disc-card">
        <div class="disc-icon">🎯</div>
        <div class="disc-title">Métrica Prioritária</div>
        <div class="disc-text">O Recall da classe Maligno foi priorizado. Um falso negativo — maligno classificado como benigno — pode atrasar tratamentos e colocar vidas em risco.</div>
      </div>
      <div class="disc-card">
        <div class="disc-icon">🧠</div>
        <div class="disc-title">Explicabilidade</div>
        <div class="disc-text">SHAP e Feature Importance revelam que bordas irregulares e perímetro são os fatores mais determinantes — alinhado com o conhecimento clínico oncológico.</div>
      </div>
      <div class="disc-card">
        <div class="disc-icon">🏥</div>
        <div class="disc-title">Uso na Prática</div>
        <div class="disc-text">O sistema pode apoiar a triagem e priorização de casos urgentes. Nunca substitui o julgamento médico — o profissional sempre tem a palavra final.</div>
      </div>
    </div>
  </section>

  <footer>
    <span>Tech Challenge · Fase 1 · Projeto B · PosTech</span>
    <span class="footer-logo">🏥 Sistema de Diagnóstico Hospitalar</span>
    <span>Breast Cancer Wisconsin · scikit-learn · SHAP</span>
  </footer>

</div>
</body>
</html>"""

with open('dashboard_b.html', 'w', encoding='utf-8') as f:
    f.write(HTML)

PORT = 8000
os.chdir(os.path.dirname(os.path.abspath(__file__)) or '.')
handler = http.server.SimpleHTTPRequestHandler
httpd   = http.server.HTTPServer(('', PORT), handler)
url = f'http://localhost:{PORT}/dashboard_b.html'
print(f'\n✅ Dashboard pronto! Abrindo em {url}')
print('   Pressione Ctrl+C para encerrar.\n')
threading.Timer(1.0, lambda: webbrowser.open(url)).start()
httpd.serve_forever()