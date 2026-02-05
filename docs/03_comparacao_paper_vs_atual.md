# Comparacao: PB96 (Paper) vs Implementacao Atual (GA-only)

Paper: https://bengio.abracadoudou.com/cv/publications/pdf/potvin_1996_informs.pdf

## Tabela comparativa

| Componente | No paper (resumo) | No nosso codigo (resumo) | Status | Impacto esperado | Ajuste minimo recomendado (GA-only) |
|---|---|---|---|---|---|
| Objective/ranking tie-break | Minimiza K e, em empate, route time (travel + waiting + service) | Ranking por viabilidade, K, route time, distancia | MATCH | Alinha com criterio do paper para solucoes viaveis | Nenhum |
| Representacao | Operadores atuam em solucoes (rotas) | PB96 crossover opera sobre rotas | MATCH | Coerencia estrutural com PB96 | Nenhum |
| Selecao | Ranking linear + SUS | `linear_ranking_fitness` + `stochastic_universal_sampling` | MATCH | Mesma pressao seletiva | Nenhum |
| Crossover (SBX/RBX + descarte) | SBX e RBX com reparo e descarte se insercao falhar | `pb96_crossover` + reparo com descarte e retry | MATCH | Offspring mais coerente e viavel | Nenhum |
| Repair strategy | Remove duplicados e insere clientes nao roteados na melhor posicao viavel | `repair_and_insert_unrouted` faz remocao e insercao viavel; descarta se falhar | MATCH | Mantem viabilidade e evita duplicatas | Nenhum |
| Mutation (1M/2M/LSM) | 1M, 2M e LSM (Or-opt) | Swap/inversion em permutacao + decode | DIFFERENT | Pode afetar exploracao local; mantem GA-only | Implementar 1M/2M sem busca local, se permitido |
| Initialization (I1) | Solomon I1 com parametros aleatorios | Greedy viavel + permutacao aleatoria | PARTIAL | Pop inicial diferente do paper | Implementar I1 ou aproximacao do I1 |
| Metrics/reporting | Reporta K, distancia, waiting, route time | Reporta K, distancia, timewarp, waiting, service, route time | PARTIAL | Adiciona timewarp; ok para GA-only | Nenhum; manter consistencia |
| Defaults de parametros | pop=150, gens=50, pc=0.6, pm=0.6 | Defaults iguais no GAConfig | MATCH | Replicabilidade | Nenhum |

## Plano de alinhamento
1. Implementar 1M e 2M como operadores de mutacao sem busca local. Impacto alto, GA-only.
2. Implementar inicializacao I1 de Solomon com parametros aleatorios. Impacto medio.
3. Documentar explicitamente a ausencia de LSM (Or-opt) por restricao do curso. Bloqueado por GA-only.
4. Manter criterio de ranking por route time como criterio principal para viaveis. Ja alinhado.
5. Expandir testes de consistencia de reparo e descarte em instancias pequenas. Impacto baixo.

Bloqueios por restricao do curso:
- LSM (Or-opt) e qualquer busca local (2-opt, relocate, tabu, SA, VNS) nao podem ser implementadas.

