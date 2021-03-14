import json
import numpy as np
from lib import utils

json_path = utils.get_relative_file('runtime', 'cache', 'top_1_dealer_pairs_directed.json')
exp_result_path = utils.get_relative_file('runtime', 'exp_result_of_pprz_transformer_bond_prediction.log')

print('\nloading json data ... ')
top_1_dealer_pairs = utils.load_json(json_path)

print(f'\nloading exp results ...')
with open(exp_result_path, 'rb') as f:
    content = f.readlines()

content = list(map(utils.decode_2_utf8, content))
content = list(map(lambda x: [x.split('.pkl: ')[0], json.loads(x.split('.pkl: ')[1].replace("'", '"'))], content))

d_dealer_performance = {}
for dealer, performance in content:
    d_dealer_performance[dealer] = performance

print('\nformatting dealers from dealer pairs')

d_node = {}
nodes = []
edges = []

nodes2 = []

d_dealer_2_dealer_2_count = {}
d_dealer_2_dealer_2_count_directed = {}
for dealer_pair, count in top_1_dealer_pairs:
    dealer_1, dealer_2 = dealer_pair.split('_')

    if dealer_1 not in d_dealer_performance or dealer_2 not in d_dealer_performance:
        continue

    f1_1 = d_dealer_performance[dealer_1]['f1']
    f1_2 = d_dealer_performance[dealer_2]['f1']

    if dealer_1 not in d_dealer_2_dealer_2_count:
        d_dealer_2_dealer_2_count[dealer_1] = {}
        d_node[f'{dealer_1}'] = len(nodes)
        nodes.append({'name': f'{dealer_1}', 'f1': f1_1})
    d_dealer_2_dealer_2_count[dealer_1][dealer_2] = count

    if dealer_2 not in d_dealer_2_dealer_2_count:
        d_dealer_2_dealer_2_count[dealer_2] = {}
        d_node[f'{dealer_2}'] = len(nodes)
        nodes.append({'name': f'{dealer_2}', 'f1': f1_2})
    d_dealer_2_dealer_2_count[dealer_2][dealer_1] = count

    if dealer_1 not in d_dealer_2_dealer_2_count_directed:
        d_dealer_2_dealer_2_count_directed[dealer_1] = {'buy': {}, 'sell': {}}
    d_dealer_2_dealer_2_count_directed[dealer_1]['sell'][dealer_2] = count

    if dealer_2 not in d_dealer_2_dealer_2_count_directed:
        d_dealer_2_dealer_2_count_directed[dealer_2] = {'buy': {}, 'sell': {}}
    d_dealer_2_dealer_2_count_directed[dealer_2]['buy'][dealer_1] = count

    _diff = np.abs(f1_1 - f1_2)
    _sim = 1 - _diff

    if _diff > 0.2:
        continue

    edges.append({
        'source': d_node[f'{dealer_1}'],
        'target': d_node[f'{dealer_2}'],
        'weight': round(_diff, 3),
        'distance': _diff * 100,
        'color': f'rgb({int(255 - _sim * 250)}, {int(255 - _sim * 205)}, {int(255 - _sim * 150)})',
    })


print('\n------------------------')
print(f'len of d_dealer_2_dealer_2_count: {len(d_dealer_2_dealer_2_count)}')
print(f'avg len of each dealer: {np.mean(list(map(lambda x: len(x[1]), d_dealer_2_dealer_2_count.items())))}\n\n')

total_mean_diff = []

for dealer_1, dealer_2_dict in d_dealer_2_dealer_2_count_directed.items():
    f1_1 = d_dealer_performance[dealer_1]['f1']

    buy_from_dealers = list(dealer_2_dict['buy'].keys())
    sell_to_dealers = list(dealer_2_dict['sell'].keys())

    buy_from_dealers = list(filter(lambda x: np.abs(d_dealer_performance[x]['f1'] - f1_1) < 0.2, buy_from_dealers))
    sell_to_dealers = list(filter(lambda x: np.abs(d_dealer_performance[x]['f1'] - f1_1) < 0.2, sell_to_dealers))

    buy_from_dealers = list(map(lambda x: f'{x} ({int(round(d_dealer_performance[x]["f1"] * 100))})', buy_from_dealers))
    sell_to_dealers = list(map(lambda x: f'{x} ({int(round(d_dealer_performance[x]["f1"] * 100))})', sell_to_dealers))

    nodes2.append({
        'name': f'{dealer_1} ({int(round(d_dealer_performance[dealer_1]["f1"] * 100))})',
        # 'name': dealer_1,
        'f1': f1_1,
        'counterparties': len(list(set(buy_from_dealers + sell_to_dealers))),
        'imports': sell_to_dealers,
        'outgoing': sell_to_dealers,
        'incoming': buy_from_dealers,
        'dealer_2_dict': d_dealer_2_dealer_2_count[dealer_1],
    })


d_dealer_2_dealer_2_f1_diff = {}
for dealer_1, dealer_2_dict in d_dealer_2_dealer_2_count.items():
    f1_1 = d_dealer_performance[dealer_1]['f1']

    nodes[d_node[dealer_1]]['counterparties'] = len(dealer_2_dict)
    nodes[d_node[dealer_1]]['size'] = len(dealer_2_dict) + 5
    nodes[d_node[dealer_1]]['color'] = f'rgb({int(255 - f1_1 * 100)}, {int(max(255 - f1_1 * 265, 10))}, {int(max(255 - f1_1 * 275, 5))})'
    continue

    if dealer_1 not in d_dealer_2_dealer_2_f1_diff:
        d_dealer_2_dealer_2_f1_diff[dealer_1] = {'f1': f1_1, 'counterparties': {}}

    mean_diffs = []
    for dealer_2, count in dealer_2_dict.items():
        f1_2 = d_dealer_performance[dealer_2]['f1']

        _diff = np.abs(f1_1 - f1_2)
        mean_diffs.append(_diff)

        d_dealer_2_dealer_2_f1_diff[dealer_1]['counterparties'][dealer_2] = {
            'count': count,
            'f1': f1_2,
            'diff': _diff,
        }

    mean_diff = np.mean(mean_diffs)
    d_dealer_2_dealer_2_f1_diff[dealer_1]['mean_diff'] = mean_diff

    print(f'dealer_{dealer_1} mean_diff: {mean_diff}, f1: {f1_1}, len counterparties: {len(dealer_2_dict)}')

    total_mean_diff.append(mean_diff)


sims = np.array(list(map(lambda x: 1 - x['weight'], edges)))
sims = (sims - np.min(sims)) / (np.max(sims) - np.min(sims)) * 20

for i, v in enumerate(sims):
    edges[i]['width'] = v
    _sim = (v / 20) ** 2
    edges[i]['color'] = f'rgb({int(255 - _sim * 250)}, {int(255 - _sim * 205)}, {int(255 - _sim * 150)})',

counters = np.array(list(map(lambda x: x['counterparties'], nodes)))
counters = (counters - np.min(counters)) / (np.max(counters) - np.min(counters))
counters = counters * counters * 20 + 5

f1s = np.array(list(map(lambda x: x['f1'], nodes)))
f1s = (f1s - np.min(f1s)) / (np.max(f1s) - np.min(f1s))

for i, v in enumerate(counters):
    nodes[i]['size'] = v
    _f1 = f1s[i]
    nodes[i]['color'] = f'rgb({int(255 - _f1 * 100)}, {int(255 - _f1 * 235)}, {int(255 - _f1 * 245)})'

nodes.sort(key=lambda x: -x['counterparties'])

visualize_data = {
    'nodes': nodes,
    'links': edges,
}

counters = np.array(list(map(lambda x: x['counterparties'], nodes2)))
counters = (counters - np.min(counters)) / (np.max(counters) - np.min(counters))
counters = counters * counters * 7 + 10

f1s = np.array(list(map(lambda x: x['f1'], nodes2)))
f1s = (f1s - np.min(f1s)) / (np.max(f1s) - np.min(f1s))

for i, v in enumerate(counters):
    nodes2[i]['size'] = int(v)
    _f1 = f1s[i] ** 2
    nodes2[i]['color'] = f'rgb({int(210 - _f1 * 90)}, {int(200 - _f1 * 195)}, {int(200 - _f1 * 200)})'

nodes2.sort(key=lambda x: -x['f1'])


print('\nsaving visualization data ...')
utils.write_json(utils.get_relative_file('runtime', 'json', 'visualize_top_1_dealer_pairs.json'), visualize_data)
utils.write_json(utils.get_relative_file('runtime', 'json', 'visualize_top_1_dealer_pairs_bundling_directed.json'), nodes2)
exit()

# ------------------------
# len of d_dealer_2_dealer_2_count: 69
# avg len of each dealer: 6.115942028985507

print(f'\ntotal average mean diff: {np.mean(total_mean_diff)}')
print(f'total std mean diff: {np.std(total_mean_diff)}')
print(f'total max mean diff: {np.max(total_mean_diff)}')
print(f'total min mean diff: {np.min(total_mean_diff)}')

print('\ndone')
