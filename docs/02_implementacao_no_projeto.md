# Implementacao PB96 no Projeto — Como esta no nosso repositorio

Paper: https://bengio.abracadoudou.com/cv/publications/pdf/potvin_1996_informs.pdf

## Visao geral do pipeline (diagramatico)
CLI (`experiments.run`) -> carrega instancia -> inicializacao da populacao -> selecao -> crossover -> reparo -> mutacao -> avaliacao -> substituicao/elitismo -> outputs (CSV/JSON)

## Mapeamento dos componentes do paper para o codigo

### Objetivo e ranking (K, route time)
STATUS: [MATCH]
Codigo: `src/vrptw_ga/metrics.py` (`rank_key`, `dominates`), `src/vrptw_ga/selection.py` (`rank_population`), `src/vrptw_ga/model.py` (`Solution.total_route_time`), `src/vrptw_ga/evaluate.py` (`evaluate_solution`)
Explicacao: A comparacao privilegia solucao viavel e, entre viaveis, minimiza K e route time. Distancia entra como desempate final. Viabilidade considera TW e capacidade.

### Representacao por rotas
STATUS: [MATCH]
Codigo: `src/vrptw_ga/model.py` (`Route`, `Solution`), `src/vrptw_ga/crossover/route_based_pb96.py`
Explicacao: Operadores de crossover PB96 trabalham sobre listas de rotas, alinhado ao paper.

### Selecao (ranking linear + SUS)
STATUS: [MATCH]
Codigo: `src/vrptw_ga/selection.py` (`linear_ranking_fitness`, `stochastic_universal_sampling`)
Explicacao: Implementacao direta do ranking linear com MAX=1.6 e MIN=0.4 e SUS.

### Crossover SBX/RBX + reparo e descarte
STATUS: [MATCH]
Codigo: `src/vrptw_ga/crossover/route_based_pb96.py` (`pb96_crossover`, `sbx`, `rbx`), `src/vrptw_ga/constructive.py` (`repair_and_insert_unrouted`), `src/vrptw_ga/ga.py` (`_crossover` e loop de retries)
Explicacao: SBX/RBX sao sorteados. O reparo remove duplicados, insere clientes nao roteados em posicao viavel com menor desvio e descarta o filho se falhar. Ha retry com fallback para melhor pai.

### Mutacao
STATUS: [DIFFERENT]
Codigo: `src/vrptw_ga/operators.py` (swap/inversion), `src/vrptw_ga/ga.py` (`_mutate`)
Explicacao: O paper descreve 1M/2M e LSM (Or-opt). Aqui usamos swap/inversion em permutacao e re-decode. Mantemos GA-only sem busca local.

### Inicializacao
STATUS: [PARTIAL]
Codigo: `src/vrptw_ga/constructive.py` (`greedy_feasible_construction`), `src/vrptw_ga/ga.py` (`_init_population`)
Explicacao: O paper usa I1 de Solomon com parametros aleatorios. Aqui usamos greedy viavel e mix com permutacao aleatoria.

### Substituicao e elitismo
STATUS: [MATCH]
Codigo: `src/vrptw_ga/ga.py` (elitismo + rank na populacao combinada)
Explicacao: Mantemos o melhor individuo e aplicamos selecao por ranking para sobreviventes.

### Avaliacao e metricas
STATUS: [PARTIAL]
Codigo: `src/vrptw_ga/evaluate.py`, `src/vrptw_ga/metrics.py`, `src/experiments/run.py`
Explicacao: Reportamos K, distancia, timewarp, waiting, service e route time. O paper reporta waiting e route time; usamos distance para penalizacao em infeasiveis.

### Parametros padrao
STATUS: [MATCH]
Codigo: `src/vrptw_ga/ga.py` (`GAConfig`), `src/experiments/run.py`
Explicacao: pop=150, gens=50, pc=0.6, pm=0.6 sao os defaults.

## Flags de CLI relacionadas ao PB96
- `--crossover pb96` aplica SBX/RBX + reparo
- `--objective lexicographic` aplica ranking por viabilidade, K, route time
- `--init mixed` usa greedy viavel + permutacoes aleatorias
- `--pop`, `--gens`, `--crossover_rate`, `--mutation_rate` ajustam parametros do GA
- `--penalty_tw` define penalidade de timewarp para solucoes inviaveis
- `--decoder` escolhe decodificador quando cromossomo e usado
- `--repair_tw` aplica reparo por split em violacoes de TW no decode

## Componentes explicitamente excluidos (GA-only)
- LSM (Or-opt) do paper
- Qualquer busca local ou hibridizacao (2-opt, relocate, tabu, SA, VNS)

