# Requisitos do Paper PB96 (Potvin & Bengio 1996) — GA para VRPTW

Paper: https://bengio.abracadoudou.com/cv/publications/pdf/potvin_1996_informs.pdf

## Definicao do problema e criterio de comparacao
VRPTW com distancias euclidianas, tempo de viagem igual a distancia, janelas de tempo e capacidade. A comparacao de solucoes usa ordem lexicografica: primeiro minimizar K (numero de rotas). Em empate de K, minimizar route time (soma de tempo de viagem, waiting e servico). A viabilidade e tratada antes da comparacao. Ver Secao 2.1, p.4; Secao 3.1, p.12.

## Representacao
Os operadores geneticos atuam diretamente em solucoes (rotas), nao em um cromossomo unico. Ver Secao 2, p.3-4.

## Selecao
Selecao por ranking linear com MAX=1.6 e MIN=0.4, seguida de SUS (stochastic universal sampling) para reduzir variancia. Ver Secao 2.1, p.4-5.

## Crossover e reparo
Dois operadores: SBX (sequence-based) e RBX (route-based). Depois do crossover, aplica reparo:
- Remove duplicados.
- Insere clientes nao roteados na posicao viavel que minimiza o desvio.
Se algum cliente nao puder ser inserido de forma viavel, o filho e descartado e novos pais sao selecionados. Ver Secao 2.2.1 e 2.2.2, p.5-6.

## Mutacao
Tres operadores descritos:
- 1M (one-level exchange): tenta esvaziar rotas pequenas movendo clientes.
- 2M (two-level exchange): troca em dois niveis para viabilizar insercao.
- LSM (local search mutation) baseado em Or-opt ate otimo local.
Ver Secao 2.3, p.9-10.

## Parametros
Configuracao usada no paper:
- Populacao: 150
- Geracoes: 50
- Taxa de crossover: 0.6
- Taxa de mutacao: 0.6
Ver Secao 3.1, p.12.

## Inicializacao
Populacao inicial gerada pelo heuristico I1 de Solomon com parametros aleatorios. Ver Secao 3.1, p.12.

## Benchmark e protocolo
Instancias de Solomon (R1/R2, C1/C2, RC1/RC2). Metricas reportadas: K, distancia, waiting time, route time e tempo computacional. Route time = travel time + waiting time + unload/service time. Ver Secao 3 e 3.1, p.11-12.

## Checklist de requisitos do paper
- [ ] Definicao VRPTW com tempo de viagem = distancia e janelas de tempo
- [ ] Comparacao lexicografica: minimizar K, depois route time
- [ ] Representacao direta por rotas
- [ ] Selecao por ranking linear (MAX=1.6, MIN=0.4)
- [ ] SUS para selecao de pais
- [ ] Crossover SBX
- [ ] Crossover RBX
- [ ] Reparo por remocao de duplicados e insercao viavel com menor desvio
- [ ] Descarte do filho se insercao viavel falhar
- [ ] Mutacao 1M
- [ ] Mutacao 2M
- [ ] LSM (Or-opt) como mutacao
- [ ] Inicializacao com I1 de Solomon e parametros aleatorios
- [ ] Parametros padrao: pop=150, gens=50, pc=0.6, pm=0.6
- [ ] Protocolo Solomon com metricas K, distancia, waiting, route time
